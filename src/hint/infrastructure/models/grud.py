"""Summary of the grud module.

Longer description of the module purpose and usage.
"""

import torch

import torch.nn as nn

import torch.nn.functional as F

from typing import Optional

from ..networks import BaseICDClassifier



class GRUDClassifier(BaseICDClassifier):

    """Summary of GRUDClassifier purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        b_gamma_x (Any): Description of b_gamma_x.
        dropout (Any): Description of dropout.
        fc (Any): Description of fc.
        h_layer (Any): Description of h_layer.
        hidden_dim (Any): Description of hidden_dim.
        input_dim (Any): Description of input_dim.
        lin_gamma_h (Any): Description of lin_gamma_h.
        r_layer (Any): Description of r_layer.
        w_gamma_x (Any): Description of w_gamma_x.
        z_layer (Any): Description of z_layer.
    """

    def __init__(self, num_classes: int, input_dim: int, seq_len: int, hidden_dim: int = 64, dropout: float = 0.3, **kwargs):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            num_classes (Any): Description of num_classes.
            input_dim (Any): Description of input_dim.
            seq_len (Any): Description of seq_len.
            hidden_dim (Any): Description of hidden_dim.
            dropout (Any): Description of dropout.
            kwargs (Any): Description of kwargs.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        super().__init__(num_classes, input_dim, seq_len, dropout=dropout, **kwargs)

        self.hidden_dim = hidden_dim

        self.input_dim = input_dim

        self.w_gamma_x = nn.Parameter(torch.Tensor(input_dim))

        self.b_gamma_x = nn.Parameter(torch.Tensor(input_dim))



        self.lin_gamma_h = nn.Linear(input_dim, hidden_dim)



        self.register_buffer('empirical_mean', torch.zeros(input_dim))



        input_size = input_dim * 2 + hidden_dim



        self.z_layer = nn.Linear(input_size, hidden_dim)

        self.r_layer = nn.Linear(input_size, hidden_dim)

        self.h_layer = nn.Linear(input_size, hidden_dim)



        self.fc = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout)



        self._init_parameters()



    def _init_parameters(self):

        """Summary of _init_parameters.
        
        Longer description of the _init_parameters behavior and usage.
        
        Args:
            None (None): This function does not accept arguments.
        
        Returns:
            Any: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        nn.init.xavier_uniform_(self.z_layer.weight)

        nn.init.xavier_uniform_(self.r_layer.weight)

        nn.init.xavier_uniform_(self.h_layer.weight)

        nn.init.uniform_(self.w_gamma_x, -0.01, 0.01)

        nn.init.constant_(self.b_gamma_x, 0)



    def forward(self, x_num: Optional[torch.Tensor] = None,

                mask: Optional[torch.Tensor] = None,

                delta: Optional[torch.Tensor] = None,

                **kwargs) -> torch.Tensor:

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
            x_num (Any): Description of x_num.
            mask (Any): Description of mask.
            delta (Any): Description of delta.
            kwargs (Any): Description of kwargs.
        
        Returns:
            torch.Tensor: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        if x_num is None:

            raise ValueError("GRU-D requires x_num")



        x = x_num.permute(0, 2, 1)

        b, t, c = x.size()



        if mask is None:

            mask = torch.ones_like(x).to(x.device)

        else:

            mask = mask.permute(0, 2, 1)



        if delta is None:

            delta = torch.ones_like(x).to(x.device)

        else:

            delta = delta.permute(0, 2, 1)



        h = torch.zeros(b, self.hidden_dim).to(x.device)



        x_last_obsv = torch.zeros(b, c).to(x.device)



        for i in range(t):

            x_t = x[:, i, :]

            m_t = mask[:, i, :]

            d_t = delta[:, i, :]



            gamma_x = torch.exp(-torch.relu(d_t * self.w_gamma_x + self.b_gamma_x))



            gamma_h = torch.exp(-torch.relu(self.lin_gamma_h(d_t)))



            x_impute = m_t * x_t + (1 - m_t) * (

                gamma_x * x_last_obsv + (1 - gamma_x) * self.empirical_mean

            )



            h_decay = gamma_h * h



            combined = torch.cat([x_impute, m_t, h_decay], dim=1)



            z = torch.sigmoid(self.z_layer(combined))

            r = torch.sigmoid(self.r_layer(combined))



            combined_new = torch.cat([x_impute, m_t, r * h_decay], dim=1)

            h_tilde = torch.tanh(self.h_layer(combined_new))



            h = (1 - z) * h_decay + z * h_tilde



            x_last_obsv = m_t * x_t + (1 - m_t) * x_last_obsv



        return self.fc(self.dropout(h))
