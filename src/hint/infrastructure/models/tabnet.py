"""Summary of the tabnet module.

Longer description of the module purpose and usage.
"""

import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

from ..networks import BaseICDClassifier



class Sparsemax(nn.Module):

    """Summary of Sparsemax purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    dim (Any): Description of dim.
    output (Any): Description of output.
    """

    def __init__(self, dim=-1):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        dim (Any): Description of dim.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super(Sparsemax, self).__init__()

        self.dim = dim



    def forward(self, input):

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
        input (Any): Description of input.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
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

    """Summary of GhostBatchNorm purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    bn (Any): Description of bn.
    input_dim (Any): Description of input_dim.
    virtual_batch_size (Any): Description of virtual_batch_size.
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        input_dim (Any): Description of input_dim.
        virtual_batch_size (Any): Description of virtual_batch_size.
        momentum (Any): Description of momentum.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super(GhostBatchNorm, self).__init__()

        self.input_dim = input_dim

        self.virtual_batch_size = virtual_batch_size

        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)



    def forward(self, x):

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
        x (Any): Description of x.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        if self.training and x.shape[0] > self.virtual_batch_size:

            chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)

            res = [self.bn(x_) for x_ in chunks]

            return torch.cat(res, dim=0)

        return self.bn(x)



class GLU(nn.Module):

    """Summary of GLU purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    bn (Any): Description of bn.
    fc (Any): Description of fc.
    output_dim (Any): Description of output_dim.
    """

    def __init__(self, input_dim, output_dim, fc=None, virtual_batch_size=128, momentum=0.02):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        input_dim (Any): Description of input_dim.
        output_dim (Any): Description of output_dim.
        fc (Any): Description of fc.
        virtual_batch_size (Any): Description of virtual_batch_size.
        momentum (Any): Description of momentum.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super(GLU, self).__init__()

        self.output_dim = output_dim

        if fc:

            self.fc = fc

        else:

            self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)

        self.bn = GhostBatchNorm(2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)



    def forward(self, x):

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
        x (Any): Description of x.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        x = self.fc(x)

        x = self.bn(x)

        out = torch.mul(x[:, :self.output_dim], torch.sigmoid(x[:, self.output_dim:]))

        return out



class AttentiveTransformer(nn.Module):

    """Summary of AttentiveTransformer purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    bn (Any): Description of bn.
    fc (Any): Description of fc.
    sparsemax (Any): Description of sparsemax.
    """

    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        input_dim (Any): Description of input_dim.
        output_dim (Any): Description of output_dim.
        virtual_batch_size (Any): Description of virtual_batch_size.
        momentum (Any): Description of momentum.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super(AttentiveTransformer, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim, bias=False)

        self.bn = GhostBatchNorm(output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)

        self.sparsemax = Sparsemax(dim=-1)



    def forward(self, priors, processed_feat):

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
        priors (Any): Description of priors.
        processed_feat (Any): Description of processed_feat.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        x = self.fc(processed_feat)

        x = self.bn(x)

        x = torch.mul(x, priors)

        x = self.sparsemax(x)

        return x



class TabNetICD(BaseICDClassifier):

    """Summary of TabNetICD purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    att_transformers (Any): Description of att_transformers.
    embedding_dim (Any): Description of embedding_dim.
    epsilon (Any): Description of epsilon.
    feat_transformers (Any): Description of feat_transformers.
    final_mapping (Any): Description of final_mapping.
    flatten_dim (Any): Description of flatten_dim.
    gamma (Any): Description of gamma.
    input_bn (Any): Description of input_bn.
    n_a (Any): Description of n_a.
    n_d (Any): Description of n_d.
    n_steps (Any): Description of n_steps.
    shared_layers (Any): Description of shared_layers.
    virtual_batch_size (Any): Description of virtual_batch_size.
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

        proj_dim: int = 0,

        **kwargs

    ):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        num_classes (Any): Description of num_classes.
        input_dim (Any): Description of input_dim.
        seq_len (Any): Description of seq_len.
        dropout (Any): Description of dropout.
        n_d (Any): Description of n_d.
        n_a (Any): Description of n_a.
        n_steps (Any): Description of n_steps.
        gamma (Any): Description of gamma.
        n_independent (Any): Description of n_independent.
        n_shared (Any): Description of n_shared.
        epsilon (Any): Description of epsilon.
        virtual_batch_size (Any): Description of virtual_batch_size.
        momentum (Any): Description of momentum.
        kwargs (Any): Description of kwargs.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
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



        proj_dim = int(proj_dim) if proj_dim else 0
        if proj_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(n_d, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.final_mapping = nn.Linear(proj_dim, num_classes)
            self.embedding_dim = proj_dim
        else:
            self.proj = None
            self.final_mapping = nn.Linear(n_d, num_classes)
            self.embedding_dim = n_d



    def forward(self, x_num: torch.Tensor, return_embeddings: bool = False, **kwargs) -> torch.Tensor:

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
        x_num (Any): Description of x_num.
        return_embeddings (Any): Description of return_embeddings.
        kwargs (Any): Description of kwargs.
        
        Returns:
        torch.Tensor: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
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



        if self.proj is not None:
            out_accum = self.proj(out_accum)

        if return_embeddings:
            return out_accum

        return self.final_mapping(out_accum)
