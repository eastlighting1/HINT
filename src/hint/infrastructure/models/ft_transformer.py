"""Summary of the ft_transformer module.

Longer description of the module purpose and usage.
"""

import torch

import torch.nn as nn

from ..networks import BaseICDClassifier



class FeatureTokenizer(nn.Module):

    """Summary of FeatureTokenizer purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        bias (Any): Description of bias.
        weight (Any): Description of weight.
    """

    def __init__(self, num_features, d_token):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            num_features (Any): Description of num_features.
            d_token (Any): Description of d_token.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        super().__init__()

        self.weight = nn.Parameter(torch.randn(num_features, d_token))

        self.bias = nn.Parameter(torch.randn(num_features, d_token))

        nn.init.xavier_uniform_(self.weight)

        nn.init.zeros_(self.bias)



    def forward(self, x):

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
            x (Any): Description of x.
        
        Returns:
            Any: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """







        return x.unsqueeze(-1) * self.weight + self.bias



class FTTransformerICD(BaseICDClassifier):

    """Summary of FTTransformerICD purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        cls_token (Any): Description of cls_token.
        d_token (Any): Description of d_token.
        embedding_dim (Any): Description of embedding_dim.
        final_mapping (Any): Description of final_mapping.
        ln (Any): Description of ln.
        num_features (Any): Description of num_features.
        tokenizer (Any): Description of tokenizer.
        transformer (Any): Description of transformer.
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

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            num_classes (Any): Description of num_classes.
            input_dim (Any): Description of input_dim.
            seq_len (Any): Description of seq_len.
            dropout (Any): Description of dropout.
            d_token (Any): Description of d_token.
            n_blocks (Any): Description of n_blocks.
            n_heads (Any): Description of n_heads.
            d_ffn (Any): Description of d_ffn.
            kwargs (Any): Description of kwargs.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
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

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
            x_num (Any): Description of x_num.
            return_embeddings (Any): Description of return_embeddings.
            kwargs (Any): Description of kwargs.
        
        Returns:
            torch.Tensor: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
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
