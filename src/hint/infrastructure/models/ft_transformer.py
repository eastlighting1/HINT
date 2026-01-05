import torch
import torch.nn as nn
from ..networks import BaseICDClassifier

class FeatureTokenizer(nn.Module):
    """
    Converts numerical features into embeddings.
    For each feature scalar x_i, computes embedding e_i = x_i * W_i + b_i
    """
    def __init__(self, num_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_token))
        self.bias = nn.Parameter(torch.randn(num_features, d_token))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        # x: (Batch, Num_Features)
        # weight: (Num_Features, d_token)
        # x * weight (broadcast) -> (Batch, Num_Features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias

class FTTransformerICD(BaseICDClassifier):
    """
    Revisiting Deep Learning Models for Tabular Data (NeurIPS 2021)
    """
    def __init__(self, num_classes: int, input_dim: int, seq_len: int, dropout: float = 0.1,
                 d_token=192, n_blocks=3, n_heads=8, d_ffn=None, **kwargs):
        super().__init__(num_classes, input_dim, seq_len, dropout)
        
        # Strategy: Flatten Time Dimension -> Treat each point in time-series as a distinct feature column
        self.num_features = input_dim * seq_len
        self.d_token = d_token
        
        # 1. Feature Tokenizer
        self.tokenizer = FeatureTokenizer(self.num_features, d_token)
        
        # 2. CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # 3. Transformer Encoder
        # Use RegLU (ReLU^2) approx with GELU if strictly needed, but standard Transformer uses GELU/ReLU
        # Paper suggests ReGLU, here using GELU for PyTorch native efficiency
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
        
        # 4. Head
        self.ln = nn.LayerNorm(d_token)
        self.final_mapping = nn.Linear(d_token, num_classes)
        self.embedding_dim = d_token

    def forward(self, x_num: torch.Tensor, return_embeddings: bool = False, **kwargs) -> torch.Tensor:
        # x_num: (B, C, T) -> (B, C*T)
        x = x_num.reshape(x_num.size(0), -1)
        
        # Tokenize: (B, F) -> (B, F, D)
        x_emb = self.tokenizer(x)
        
        # Add CLS Token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat([cls_tokens, x_emb], dim=1) # (B, F+1, D)
        
        # Transformer encoding
        x_out = self.transformer(x_emb)
        
        # Take CLS token only
        cls_output = self.ln(x_out[:, 0, :])
        
        if return_embeddings:
            return cls_output
            
        return self.final_mapping(cls_output)