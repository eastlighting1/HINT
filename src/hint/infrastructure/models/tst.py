"""Summary of the tst module.

Longer description of the module purpose and usage.
"""

import torch

import torch.nn as nn

from typing import Optional

from ..networks import BaseICDClassifier



class TSTClassifier(BaseICDClassifier):

    """Summary of TSTClassifier purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    dropout (Any): Description of dropout.
    fc (Any): Description of fc.
    pos_enc (Any): Description of pos_enc.
    projector (Any): Description of projector.
    transformer_encoder (Any): Description of transformer_encoder.
    """

    def __init__(self, num_classes: int, input_dim: int, seq_len: int, d_model: int = 128, n_heads: int = 4, num_layers: int = 3, d_ff: int = 256, dropout: float = 0.2, **kwargs):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        num_classes (Any): Description of num_classes.
        input_dim (Any): Description of input_dim.
        seq_len (Any): Description of seq_len.
        d_model (Any): Description of d_model.
        n_heads (Any): Description of n_heads.
        num_layers (Any): Description of num_layers.
        d_ff (Any): Description of d_ff.
        dropout (Any): Description of dropout.
        kwargs (Any): Description of kwargs.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super().__init__(num_classes, input_dim, seq_len)



        self.projector = nn.Linear(input_dim, d_model)

        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model))



        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)



        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(d_model, num_classes)



    def forward(self, x_num: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
        x_num (Any): Description of x_num.
        kwargs (Any): Description of kwargs.
        
        Returns:
        torch.Tensor: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """



        if x_num is None: raise ValueError("TST requires x_num")

        x = x_num.permute(0, 2, 1)





        x = self.projector(x) + self.pos_enc[:, :x.size(1), :]





        x = self.transformer_encoder(x)





        x = x.mean(dim=1)



        return self.fc(self.dropout(x))
