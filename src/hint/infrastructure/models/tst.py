import torch
import torch.nn as nn
from typing import Optional
from ..networks import BaseICDClassifier

class TSTClassifier(BaseICDClassifier):
    def __init__(self, num_classes: int, input_dim: int, seq_len: int, d_model: int = 128, n_heads: int = 4, num_layers: int = 3, d_ff: int = 256, dropout: float = 0.2, **kwargs):
        super().__init__(num_classes, input_dim, seq_len)
        
        self.projector = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x_num: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # x_num: (B, C, T) -> (B, T, C)
        if x_num is None: raise ValueError("TST requires x_num")
        x = x_num.permute(0, 2, 1)
        
        # Projection & Positional Encoding
        x = self.projector(x) + self.pos_enc[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Global Average Pooling
        x = x.mean(dim=1)
        
        return self.fc(self.dropout(x))