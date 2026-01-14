"""Summary of the tcn module.

Longer description of the module purpose and usage.
"""

import torch

import torch.nn as nn

from torch.nn.utils.parametrizations import weight_norm



class Chomp1d(nn.Module):

    """Summary of Chomp1d purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    chomp_size (Any): Description of chomp_size.
    """

    def __init__(self, chomp_size):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        chomp_size (Any): Description of chomp_size.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super(Chomp1d, self).__init__()

        self.chomp_size = chomp_size



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

        return x[:, :, :-self.chomp_size].contiguous()



class TemporalBlock(nn.Module):

    """Summary of TemporalBlock purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    chomp1 (Any): Description of chomp1.
    chomp2 (Any): Description of chomp2.
    conv1 (Any): Description of conv1.
    conv2 (Any): Description of conv2.
    downsample (Any): Description of downsample.
    dropout1 (Any): Description of dropout1.
    dropout2 (Any): Description of dropout2.
    net (Any): Description of net.
    relu (Any): Description of relu.
    relu1 (Any): Description of relu1.
    relu2 (Any): Description of relu2.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        n_inputs (Any): Description of n_inputs.
        n_outputs (Any): Description of n_outputs.
        kernel_size (Any): Description of kernel_size.
        stride (Any): Description of stride.
        dilation (Any): Description of dilation.
        padding (Any): Description of padding.
        dropout (Any): Description of dropout.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,

                                           stride=stride, padding=padding, dilation=dilation))

        self.chomp1 = Chomp1d(padding)

        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)



        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,

                                           stride=stride, padding=padding, dilation=dilation))

        self.chomp2 = Chomp1d(padding)

        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)



        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,

                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.relu = nn.ReLU()

        self.init_weights()



    def init_weights(self):

        """Summary of init_weights.
        
        Longer description of the init_weights behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.conv1.weight.data.normal_(0, 0.01)

        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:

            self.downsample.weight.data.normal_(0, 0.01)



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

        out = self.net(x)

        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)



class TemporalConvNet(nn.Module):

    """Summary of TemporalConvNet purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    network (Any): Description of network.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        num_inputs (Any): Description of num_inputs.
        num_channels (Any): Description of num_channels.
        kernel_size (Any): Description of kernel_size.
        dropout (Any): Description of dropout.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super(TemporalConvNet, self).__init__()

        layers = []

        num_levels = len(num_channels)

        for i in range(num_levels):

            dilation_size = 2 ** i

            in_channels = num_inputs if i == 0 else num_channels[i-1]

            out_channels = num_channels[i]

            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,

                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]



        self.network = nn.Sequential(*layers)



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

        return self.network(x)
