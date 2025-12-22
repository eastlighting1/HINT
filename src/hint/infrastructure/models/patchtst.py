import torch
import torch.nn as nn
from typing import Optional
from ..networks import BaseICDClassifier

class PatchTSTClassifier(BaseICDClassifier):
    def __init__(self, num_classes: int, input_dim: int, seq_len: int, patch_len: int = 8, stride: int = 4, d_model: int = 128, n_heads: int = 4, num_layers: int = 3, **kwargs):
        super().__init__(num_classes, input_dim, seq_len)
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim
        
        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1
        if self.num_patches <= 0: self.num_patches = 1 # Fallback
        
        # Patch Embedding: Project each patch to d_model
        # Channel Independence: weights shared across channels
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, self.input_dim, self.num_patches, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.flatten = nn.Flatten(start_dim=2) # Flatten patches and d_model
        self.head = nn.Linear(self.input_dim * self.num_patches * d_model, num_classes)

    def forward(self, x_num: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # x_num: (B, C, T)
        if x_num is None: raise ValueError("PatchTST requires x_num")
        
        # Padding if needed
        B, C, T = x_num.shape
        if T < self.patch_len:
             x_num = torch.nn.functional.pad(x_num, (0, self.patch_len - T))
             T = self.patch_len
             
        # Patching
        # Unfold to (B, C, N, P)
        patches = x_num.unfold(dimension=2, size=self.patch_len, step=self.stride)
        
        # Embed: (B, C, N, P) -> (B, C, N, D)
        x = self.patch_embed(patches)
        
        # Positional Encoding
        if x.size(2) <= self.pos_enc.size(2):
            x = x + self.pos_enc[:, :, :x.size(2), :]
            
        # Reshape for Transformer: (B * C, N, D) - Treating channels independently
        x = x.view(B * C, -1, x.size(-1))
        
        # Transformer
        x = self.transformer(x)
        
        # Reshape back: (B, C, N, D)
        x = x.view(B, C, -1, x.size(-1))
        
        # Flatten and Classify
        x = x.flatten(start_dim=1)
        return self.head(x)