import torch
import torch.nn as nn
from ..networks import BaseICDClassifier

class FeatureTokenizer(nn.Module):
    """Tokenize numeric features into embeddings.

    Attributes:
        weight (nn.Parameter): Learnable feature weights.
        bias (nn.Parameter): Learnable feature biases.
    """
    def __init__(self, num_features, d_token):
        """Initialize the tokenizer parameters.

        Args:
            num_features (int): Number of input features.
            d_token (int): Token embedding dimension.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_token))
        self.bias = nn.Parameter(torch.randn(num_features, d_token))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        """Tokenize inputs into per-feature embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape [B, F].

        Returns:
            torch.Tensor: Token embeddings of shape [B, F, D].
        """
                                  
                                         
                                                                  
        return x.unsqueeze(-1) * self.weight + self.bias

class FTTransformerICD(BaseICDClassifier):
    """FT-Transformer classifier for ICD prediction.

    Attributes:
        num_features (int): Flattened feature count.
        d_token (int): Token embedding dimension.
        tokenizer (FeatureTokenizer): Tokenizer module.
        cls_token (nn.Parameter): Classification token embedding.
        transformer (nn.TransformerEncoder): Transformer encoder stack.
        ln (nn.LayerNorm): Layer normalization for CLS output.
        final_mapping (nn.Linear): Output projection to class logits.
        embedding_dim (int): Embedding dimension for downstream use.
    """
    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        seq_len: int,
        dropout: float = 0.1,
        d_token=192,
        n_blocks=3,
        n_heads=8,
        d_ffn=None,
        **kwargs
    ):
        """Initialize the FT-Transformer classifier.

        Args:
            num_classes (int): Number of output classes.
            input_dim (int): Input feature dimension.
            seq_len (int): Sequence length.
            dropout (float): Dropout probability.
            d_token (int): Token embedding dimension.
            n_blocks (int): Number of transformer blocks.
            n_heads (int): Number of attention heads.
            d_ffn (Optional[int]): Feed-forward dimension override.
            **kwargs (Any): Additional model arguments.
        """
        super().__init__(num_classes, input_dim, seq_len, dropout)
        
                                                                                                          
        self.num_features = input_dim * seq_len
        self.d_token = d_token
        
                              
        self.tokenizer = FeatureTokenizer(self.num_features, d_token)
        
                      
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        
                                
                                                                                                         
                                                                             
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, 
            nhead=n_heads, 
            dim_feedforward=d_ffn if d_ffn else int(4/3 * d_token),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        
                 
        self.ln = nn.LayerNorm(d_token)
        self.final_mapping = nn.Linear(d_token, num_classes)
        self.embedding_dim = d_token

    def forward(self, x_num: torch.Tensor, return_embeddings: bool = False, **kwargs) -> torch.Tensor:
        """Run the forward pass on numeric features.

        Args:
            x_num (torch.Tensor): Numeric input tensor.
            return_embeddings (bool): Whether to return CLS embeddings.
            **kwargs (Any): Additional model inputs.

        Returns:
            torch.Tensor: Class logits or CLS embeddings.
        """
                                      
        x = x_num.reshape(x_num.size(0), -1)
        
                                       
        x_emb = self.tokenizer(x)
        
                       
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat([cls_tokens, x_emb], dim=1)              
        
                              
        x_out = self.transformer(x_emb)
        
                             
        cls_output = self.ln(x_out[:, 0, :])
        
        if return_embeddings:
            return cls_output
            
        return self.final_mapping(cls_output)
