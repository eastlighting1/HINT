import torch
import torch.nn as nn
from typing import Optional
from ..networks import BaseICDClassifier

class iTransformerClassifier(BaseICDClassifier):
    def __init__(self, num_classes: int, input_dim: int, seq_len: int, d_model: int = 128, n_heads: int = 4, num_layers: int = 3, **kwargs):
        super().__init__(num_classes, input_dim, seq_len)
        
        # Inverted: Project entire time series of a variate into a vector
        self.enc_embedding = nn.Linear(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output: Flatten variates or pool
        self.fc = nn.Linear(input_dim * d_model, num_classes)

    def forward(self, x_num: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # x_num: (B, C, T)
        if x_num is None: raise ValueError("iTransformer requires x_num")
        
        # Embedding: Each variate (channel) is a token
        # Input (B, C, T) -> Project T to d_model -> (B, C, D)
        x = self.enc_embedding(x_num)
        
        # Transformer: Attention across variates
        x = self.encoder(x)
        
        # Flatten: (B, C * D)
        x = x.flatten(start_dim=1)
        
        return self.fc(x)