"""Summary of the dcn_v2 module.

Longer description of the module purpose and usage.
"""

import torch

import torch.nn as nn

from ..networks import BaseICDClassifier



class CrossNetV2(nn.Module):

    """Summary of CrossNetV2 purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    biases (Any): Description of biases.
    kernels (Any): Description of kernels.
    num_layers (Any): Description of num_layers.
    """

    def __init__(self, input_dim, num_layers):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        input_dim (Any): Description of input_dim.
        num_layers (Any): Description of num_layers.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super(CrossNetV2, self).__init__()

        self.num_layers = num_layers



        self.kernels = nn.ParameterList([nn.Parameter(torch.randn(input_dim, 1)) for _ in range(num_layers)])

        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])





        for p in self.kernels:

            nn.init.xavier_normal_(p)



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



        x_0 = x.unsqueeze(2)

        x_l = x_0



        for i in range(self.num_layers):







            xl_w = torch.matmul(x_l.transpose(1, 2), self.kernels[i])





            x_l = x_0 * xl_w + self.biases[i].unsqueeze(0).unsqueeze(2) + x_l



        return x_l.squeeze(2)



class DCNv2ICD(BaseICDClassifier):

    """Summary of DCNv2ICD purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    cross_net (Any): Description of cross_net.
    cross_scale (Any): Description of cross_scale.
    deep_net (Any): Description of deep_net.
    embedding_dim (Any): Description of embedding_dim.
    final_mapping (Any): Description of final_mapping.
    flat_dim (Any): Description of flat_dim.
    input_bn (Any): Description of input_bn.
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

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        num_classes (Any): Description of num_classes.
        input_dim (Any): Description of input_dim.
        seq_len (Any): Description of seq_len.
        dropout (Any): Description of dropout.
        cross_layers (Any): Description of cross_layers.
        deep_layers (Any): Description of deep_layers.
        cross_scale (Any): Description of cross_scale.
        kwargs (Any): Description of kwargs.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
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





        x_cross = self.cross_net(x)

        if self.cross_scale != 1.0:

            x_cross = x_cross * self.cross_scale

        x_deep = self.deep_net(x)





        x_stack = torch.cat([x_cross, x_deep], dim=1)



        if return_embeddings:

            return x_stack



        return self.final_mapping(x_stack)
