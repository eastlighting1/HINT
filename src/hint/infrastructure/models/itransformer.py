import torch
import torch.nn as nn
from typing import Optional
from ..networks import BaseICDClassifier

class iTransformerClassifier(BaseICDClassifier):
    """Classifier based on the iTransformer architecture.

    Attributes:
        enc_embedding (nn.Linear): Encoder embedding layer.
        encoder (nn.TransformerEncoder): Transformer encoder stack.
        fc (nn.Linear): Output projection to class logits.
    """
    def __init__(self, num_classes: int, input_dim: int, seq_len: int, d_model: int = 128, n_heads: int = 4, num_layers: int = 3, **kwargs):
        """Initialize the iTransformer classifier.

        Args:
            num_classes (int): Number of output classes.
            input_dim (int): Input feature dimension.
            seq_len (int): Sequence length.
            d_model (int): Model dimension.
            n_heads (int): Number of attention heads.
            num_layers (int): Number of encoder layers.
            **kwargs (Any): Additional model arguments.
        """
        super().__init__(num_classes, input_dim, seq_len)
        
                                                                         
        self.enc_embedding = nn.Linear(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
                                          
        self.fc = nn.Linear(input_dim * d_model, num_classes)

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
                          
        if x_num is None: raise ValueError("iTransformer requires x_num")
        
                                                      
                                                              
        x = self.enc_embedding(x_num)
        
                                                
        x = self.encoder(x)
        
                             
        x = x.flatten(start_dim=1)
        
        return self.fc(x)
