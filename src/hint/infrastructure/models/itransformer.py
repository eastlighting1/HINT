"""Summary of the itransformer module.

Longer description of the module purpose and usage.
"""

import torch

import torch.nn as nn

from typing import Optional

from ..networks import BaseICDClassifier



class iTransformerClassifier(BaseICDClassifier):

    """Summary of iTransformerClassifier purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    enc_embedding (Any): Description of enc_embedding.
    encoder (Any): Description of encoder.
    fc (Any): Description of fc.
    """

    def __init__(self, num_classes: int, input_dim: int, seq_len: int, d_model: int = 128, n_heads: int = 4, num_layers: int = 3, **kwargs):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        num_classes (Any): Description of num_classes.
        input_dim (Any): Description of input_dim.
        seq_len (Any): Description of seq_len.
        d_model (Any): Description of d_model.
        n_heads (Any): Description of n_heads.
        num_layers (Any): Description of num_layers.
        kwargs (Any): Description of kwargs.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super().__init__(num_classes, input_dim, seq_len)





        self.enc_embedding = nn.Linear(seq_len, d_model)



        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)





        self.fc = nn.Linear(input_dim * d_model, num_classes)



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



        if x_num is None: raise ValueError("iTransformer requires x_num")







        x = self.enc_embedding(x_num)





        x = self.encoder(x)





        x = x.flatten(start_dim=1)



        return self.fc(x)
