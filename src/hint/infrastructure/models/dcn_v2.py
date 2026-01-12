import torch
import torch.nn as nn
from ..networks import BaseICDClassifier

class CrossNetV2(nn.Module):
    """Cross network module for DCNv2.

    Attributes:
        num_layers (int): Number of cross layers.
        kernels (nn.ParameterList): Learnable cross kernels.
        biases (nn.ParameterList): Learnable cross biases.
    """
    def __init__(self, input_dim, num_layers):
        """Initialize the cross network.

        Args:
            input_dim (int): Input feature dimension.
            num_layers (int): Number of cross layers.
        """
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
                                                                       
        self.kernels = nn.ParameterList([nn.Parameter(torch.randn(input_dim, 1)) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])
        
                         
        for p in self.kernels:
            nn.init.xavier_normal_(p)

    def forward(self, x):
        """Apply cross layers to the input.

        Args:
            x (torch.Tensor): Input tensor of shape [B, D].

        Returns:
            torch.Tensor: Cross network output.
        """
                               
        x_0 = x.unsqueeze(2)            
        x_l = x_0
        
        for i in range(self.num_layers):
                                                          
                                     
                                 
            xl_w = torch.matmul(x_l.transpose(1, 2), self.kernels[i])            
            
                                 
            x_l = x_0 * xl_w + self.biases[i].unsqueeze(0).unsqueeze(2) + x_l
            
        return x_l.squeeze(2)

class DCNv2ICD(BaseICDClassifier):
    """Deep & Cross Network v2 classifier for ICD prediction.

    Attributes:
        flat_dim (int): Flattened input dimension.
        input_bn (nn.BatchNorm1d): Input batch normalization.
        cross_net (CrossNetV2): Cross network module.
        deep_net (nn.Sequential): Deep MLP stack.
        final_mapping (nn.Linear): Output projection layer.
        embedding_dim (int): Dimension of stacked features.
    """
    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        seq_len: int,
        dropout: float = 0.2,
        cross_layers=3,
        deep_layers=[256, 256],
        cross_scale: float = 1.0,
        **kwargs
    ):
        """Initialize the DCNv2 classifier.

        Args:
            num_classes (int): Number of output classes.
            input_dim (int): Input feature dimension.
            seq_len (int): Sequence length.
            dropout (float): Dropout probability.
            cross_layers (int): Number of cross layers.
            deep_layers (list): Hidden sizes for deep layers.
            **kwargs (Any): Additional model arguments.
        """
        super().__init__(num_classes, input_dim, seq_len, dropout)
        
                                 
        self.flat_dim = input_dim * seq_len
        self.input_bn = nn.BatchNorm1d(self.flat_dim)
        self.cross_scale = cross_scale
        
                          
        self.cross_net = CrossNetV2(self.flat_dim, num_layers=cross_layers)
        
                         
        layers = []
        in_dim = self.flat_dim
        for hid_dim in deep_layers:
            layers.append(nn.Linear(in_dim, hid_dim))
            layers.append(nn.BatchNorm1d(hid_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hid_dim
        self.deep_net = nn.Sequential(*layers)
        
                           
        stack_dim = self.flat_dim + deep_layers[-1]
        self.final_mapping = nn.Linear(stack_dim, num_classes)
        self.embedding_dim = stack_dim                       

    def forward(self, x_num: torch.Tensor, return_embeddings: bool = False, **kwargs) -> torch.Tensor:
        """Run the forward pass on numeric features.

        Args:
            x_num (torch.Tensor): Numeric input tensor.
            return_embeddings (bool): Whether to return stacked features.
            **kwargs (Any): Additional model inputs.

        Returns:
            torch.Tensor: Class logits or stacked features.
        """
                                      
        x = x_num.reshape(x_num.size(0), -1)
        x = self.input_bn(x)
        
                            
        x_cross = self.cross_net(x)
        if self.cross_scale != 1.0:
            x_cross = x_cross * self.cross_scale
        x_deep = self.deep_net(x)
        
                        
        x_stack = torch.cat([x_cross, x_deep], dim=1)
        
        if return_embeddings:
            return x_stack
            
        return self.final_mapping(x_stack)
