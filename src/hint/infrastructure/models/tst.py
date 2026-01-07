import torch
import torch.nn as nn
from typing import Optional
from ..networks import BaseICDClassifier

class TSTClassifier(BaseICDClassifier):
    """Transformer-based classifier for time-series features.

    Attributes:
        projector (nn.Linear): Projects input features to model dimension.
        pos_enc (nn.Parameter): Learnable positional encoding.
        transformer_encoder (nn.TransformerEncoder): Encoder stack.
        dropout (nn.Dropout): Dropout layer before classification.
        fc (nn.Linear): Output projection to class logits.
    """
    def __init__(self, num_classes: int, input_dim: int, seq_len: int, d_model: int = 128, n_heads: int = 4, num_layers: int = 3, d_ff: int = 256, dropout: float = 0.2, **kwargs):
        """Initialize the transformer-based classifier.

        Args:
            num_classes (int): Number of output classes.
            input_dim (int): Input feature dimension.
            seq_len (int): Sequence length.
            d_model (int): Model dimension.
            n_heads (int): Number of attention heads.
            num_layers (int): Number of encoder layers.
            d_ff (int): Feed-forward dimension.
            dropout (float): Dropout probability.
            **kwargs (Any): Additional model arguments.
        """
        super().__init__(num_classes, input_dim, seq_len)
        
        self.projector = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x_num: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Run the forward pass on numeric time-series data.

        Args:
            x_num (Optional[torch.Tensor]): Numeric input features.
            **kwargs (Any): Additional model inputs.

        Returns:
            torch.Tensor: Class logits.

        Raises:
            ValueError: If x_num is not provided.
        """
                                       
        if x_num is None: raise ValueError("TST requires x_num")
        x = x_num.permute(0, 2, 1)
        
                                          
        x = self.projector(x) + self.pos_enc[:, :x.size(1), :]
        
                     
        x = self.transformer_encoder(x)
        
                                
        x = x.mean(dim=1)
        
        return self.fc(self.dropout(x))
