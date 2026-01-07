import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..networks import BaseICDClassifier

class Sparsemax(nn.Module):
    """Sparsemax activation function.

    Attributes:
        dim (int): Dimension over which to apply sparsemax.
    """
    def __init__(self, dim=-1):
        """Initialize the sparsemax activation.

        Args:
            dim (int): Dimension to apply sparsemax over.
        """
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        """Compute sparsemax activations.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Sparsemax-transformed output.
        """
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)
        
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)
        
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range_values = torch.arange(start=1, end=number_of_logits + 1, device=input.device, dtype=input.dtype).view(1, -1)
        bound = 1 + range_values * zs
        cumsum_zs = torch.cumsum(zs, dim=dim)
        is_gt = bound > cumsum_zs
        k = torch.max(is_gt * range_values, dim=dim, keepdim=True)[0]
        zs_sparse = is_gt * zs
        taus = (torch.sum(zs_sparse, dim=dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)
        
        self.output = torch.max(torch.zeros_like(input), input - taus)
        
        output = self.output.transpose(0, 1)
        output = output.reshape(original_size)
        return output.transpose(0, self.dim)

class GhostBatchNorm(nn.Module):
    """Batch normalization with virtual batch splitting.

    Attributes:
        input_dim (int): Feature dimension.
        virtual_batch_size (int): Virtual batch size for normalization.
        bn (nn.BatchNorm1d): Batch norm module.
    """
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        """Initialize ghost batch normalization.

        Args:
            input_dim (int): Feature dimension.
            virtual_batch_size (int): Virtual batch size.
            momentum (float): Batch norm momentum.
        """
        super(GhostBatchNorm, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        """Apply ghost batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.training and x.shape[0] > self.virtual_batch_size:
            chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        return self.bn(x)

class GLU(nn.Module):
    """Gated Linear Unit used in TabNet blocks.

    Attributes:
        output_dim (int): Output feature dimension.
        bn (GhostBatchNorm): Normalization layer.
        fc (nn.Linear): Linear projection for gating.
    """
    def __init__(self, input_dim, output_dim, fc=None, virtual_batch_size=128, momentum=0.02):
        """Initialize the GLU block.

        Args:
            input_dim (int): Input feature dimension.
            output_dim (int): Output feature dimension.
            fc (Optional[nn.Linear]): Optional shared linear layer.
            virtual_batch_size (int): Virtual batch size.
            momentum (float): Batch norm momentum.
        """
        super(GLU, self).__init__()
        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)
        self.bn = GhostBatchNorm(2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

    def forward(self, x):
        """Apply gated linear transformation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, :self.output_dim], torch.sigmoid(x[:, self.output_dim:]))
        return out

class AttentiveTransformer(nn.Module):
    """Attentive transformer for feature selection in TabNet.

    Attributes:
        fc (nn.Linear): Projection layer.
        bn (GhostBatchNorm): Normalization layer.
        sparsemax (Sparsemax): Sparse attention activation.
    """
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):
        """Initialize the attentive transformer.

        Args:
            input_dim (int): Input feature dimension.
            output_dim (int): Output feature dimension.
            virtual_batch_size (int): Virtual batch size.
            momentum (float): Batch norm momentum.
        """
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.bn = GhostBatchNorm(output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, priors, processed_feat):
        """Compute attentive feature masks.

        Args:
            priors (torch.Tensor): Prior mask tensor.
            processed_feat (torch.Tensor): Feature representation.

        Returns:
            torch.Tensor: Sparse attention mask.
        """
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.sparsemax(x)
        return x

class TabNetICD(BaseICDClassifier):
    """TabNet-based classifier for ICD prediction.

    Attributes:
        flatten_dim (int): Flattened input dimension.
        input_bn (nn.BatchNorm1d): Batch norm for inputs.
        n_d (int): Decision layer width.
        n_a (int): Attention layer width.
        n_steps (int): Number of decision steps.
        gamma (float): Relaxation parameter.
        epsilon (float): Numerical stability constant.
        virtual_batch_size (int): Virtual batch size for normalization.
        shared_layers (nn.ModuleList): Shared feature transformer layers.
        feat_transformers (nn.ModuleList): Step-specific feature transformers.
        att_transformers (nn.ModuleList): Step-specific attention modules.
        final_mapping (nn.Linear): Output projection layer.
        embedding_dim (int): Embedding dimension for outputs.
    """
    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        seq_len: int,
        dropout: float = 0.3,
        n_d=64,
        n_a=64,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        **kwargs
    ):
        """Initialize the TabNet classifier.

        Args:
            num_classes (int): Number of output classes.
            input_dim (int): Input feature dimension.
            seq_len (int): Sequence length.
            dropout (float): Dropout probability.
            n_d (int): Decision layer width.
            n_a (int): Attention layer width.
            n_steps (int): Number of decision steps.
            gamma (float): Relaxation parameter.
            n_independent (int): Independent GLU layers per step.
            n_shared (int): Shared GLU layers across steps.
            epsilon (float): Numerical stability constant.
            virtual_batch_size (int): Virtual batch size.
            momentum (float): Batch norm momentum.
            **kwargs (Any): Additional model arguments.
        """
        super().__init__(num_classes, input_dim, seq_len, dropout)
        
                               
        self.flatten_dim = input_dim * seq_len
        self.input_bn = nn.BatchNorm1d(self.flatten_dim, momentum=0.01)
        
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.virtual_batch_size = virtual_batch_size
        
        self.shared_layers = nn.ModuleList()
        current_input_dim = self.flatten_dim
        for i in range(n_shared):
            if i == 0:
                self.shared_layers.append(GLU(current_input_dim, n_d + n_a, virtual_batch_size=virtual_batch_size, momentum=momentum))
            else:
                self.shared_layers.append(GLU(n_d + n_a, n_d + n_a, virtual_batch_size=virtual_batch_size, momentum=momentum))
        
        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()
        
        for step in range(n_steps):
            transformer = nn.ModuleList()
            for i in range(n_independent):
                if i == 0:
                    transformer.append(GLU(n_d + n_a, n_d + n_a, virtual_batch_size=virtual_batch_size, momentum=momentum))
                else:
                    transformer.append(GLU(n_d + n_a, n_d + n_a, virtual_batch_size=virtual_batch_size, momentum=momentum))
            self.feat_transformers.append(transformer)
            self.att_transformers.append(AttentiveTransformer(n_a, self.flatten_dim, virtual_batch_size=virtual_batch_size, momentum=momentum))
            
        self.final_mapping = nn.Linear(n_d, num_classes)
        self.embedding_dim = n_d 

    def forward(self, x_num: torch.Tensor, return_embeddings: bool = False, **kwargs) -> torch.Tensor:
        """Run the forward pass on numeric features.

        Args:
            x_num (torch.Tensor): Numeric input tensor.
            return_embeddings (bool): Whether to return embeddings.
            **kwargs (Any): Additional model inputs.

        Returns:
            torch.Tensor: Class logits or embeddings.
        """
                                                      
        x = x_num.reshape(x_num.size(0), -1)
        x = self.input_bn(x)
        
        batch_size = x.shape[0]
        priors = torch.ones(x.shape, device=x.device)
        out_accum = torch.zeros(batch_size, self.n_d, device=x.device)
        
                              
        masked_x = x
        x_processed = torch.zeros(batch_size, self.n_d + self.n_a, device=x.device)
        
                                            
        for layer in self.shared_layers:
            x_processed = torch.add(x_processed, layer(masked_x)) * 0.7071            
            masked_x = x_processed                                        

        for step in range(self.n_steps):
            mask = self.att_transformers[step](priors, x_processed[:, self.n_d:])
            
                           
            priors = torch.mul(priors, (self.gamma - mask))
            
                           
            masked_x = torch.mul(mask, x)
            
                                                   
            x_processed = torch.zeros(batch_size, self.n_d + self.n_a, device=x.device)
            for layer in self.shared_layers:
                x_processed = torch.add(x_processed, layer(masked_x)) * 0.7071
                masked_x = x_processed                                                                        
            
                                      
            for layer in self.feat_transformers[step]:
                x_processed = torch.add(x_processed, layer(x_processed)) * 0.7071
            
            out = F.relu(x_processed[:, :self.n_d])
            out_accum = torch.add(out_accum, out)
        
        if return_embeddings:
            return out_accum
            
        return self.final_mapping(out_accum)
