import torch
import torch.nn as nn
from typing import Sequence, Dict, List

class DilatedResidualBlock(nn.Module):
    """
    Temporal Convolutional Network (TCN) Block.

    Applies dilated convolutions with residual connections.

    Args:
        in_c: Number of input channels.
        out_c: Number of output channels.
        kernel: Convolution kernel size.
        stride: Stride for the convolution.
        dilation: Dilation factor.
        dropout: Dropout probability.
    """
    def __init__(
        self, 
        in_c: int, 
        out_c: int, 
        kernel: int, 
        stride: int, 
        dilation: int, 
        dropout: float
    ) -> None:
        super().__init__()
        padding = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_c, out_c, kernel, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_c, out_c, kernel, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        
        self.shortcut = nn.Conv1d(in_c, out_c, 1, bias=False) if in_c != out_c else None
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.conv1(x)[..., : x.size(2)]
        out = self.relu1(self.bn1(out))
        out = self.drop1(out)
        out = self.conv2(out)[..., : x.size(2)]
        out = self.relu2(self.bn2(out))
        out = self.drop2(out)
        out = out + identity
        return self.relu_out(out)

class HINT_CNN(nn.Module):
    """
    GFINet-based CNN implementation.

    Constructs a multi-branch TCN with categorical feature gating.

    Args:
        in_chs: List of input channel counts for [group1, group2, rest].
        n_cls: Number of output classes.
        g1: Feature indices for group 1.
        g2: Feature indices for group 2.
        rest: Feature indices for the rest.
        cat_vocab_sizes: Dictionary mapping categorical feature names to vocabulary sizes.
        embed_dim: Embedding dimension for numeric and categorical features.
        cat_embed_dim: Embedding dimension specifically for categorical features.
        head_drop: Dropout rate for the classification head.
        tcn_drop: Dropout rate for TCN blocks.
        kernel: Kernel size for TCN blocks.
        layers: Number of layers in TCN stacks.
    """
    def __init__(
        self,
        in_chs: Sequence[int],
        n_cls: int,
        g1: Sequence[int],
        g2: Sequence[int],
        rest: Sequence[int],
        cat_vocab_sizes: Dict[str, int],
        embed_dim: int = 128,
        cat_embed_dim: int = 32,
        head_drop: float = 0.3,
        tcn_drop: float = 0.2,
        kernel: int = 5,
        layers: int = 5,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        dilations = [2**i for i in range(layers)]

        self.register_buffer("g1", torch.tensor(g1))
        self.register_buffer("g2", torch.tensor(g2))
        self.register_buffer("rest", torch.tensor(rest))

        def build_tcn_stack(in_dim: int, out_dim: int) -> nn.Sequential:
            base = [nn.Sequential(nn.Conv1d(in_dim, out_dim, 1), nn.BatchNorm1d(out_dim), nn.ReLU())]
            for d in dilations:
                base.append(DilatedResidualBlock(out_dim, out_dim, kernel, 1, d, tcn_drop))
            return nn.Sequential(*base)

        self.numeric_branches = nn.ModuleList()
        for channels in in_chs:
            if channels > 0:
                self.numeric_branches.append(build_tcn_stack(channels, embed_dim))
            else:
                self.numeric_branches.append(None)

        self.cat_embeddings = nn.ModuleList()
        self.cat_branches = nn.ModuleList()
        total_cat_dim = 0
        for feat_name, vocab_size in cat_vocab_sizes.items():
            self.cat_embeddings.append(nn.Embedding(vocab_size, cat_embed_dim, padding_idx=0))
            self.cat_branches.append(build_tcn_stack(cat_embed_dim, cat_embed_dim))
            total_cat_dim += cat_embed_dim

        num_in_numeric = 3 * embed_dim
        self.gate_linear = nn.Sequential(nn.Linear(total_cat_dim, num_in_numeric), nn.Sigmoid())
        head_in = num_in_numeric + total_cat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.ReLU(),
            nn.Dropout(head_drop),
            nn.Linear(head_in // 2, n_cls),
        )

    def forward(self, x_full: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x_full: Full numeric input tensor.
            x_cat: Categorical input tensor.

        Returns:
            Class logits.
        """
        numeric_inputs = [
            x_full.index_select(1, self.g1), 
            x_full.index_select(1, self.g2), 
            x_full.index_select(1, self.rest)
        ]
        
        batch_size = x_full.size(0)
        device = x_full.device

        numeric_pooled = []
        for branch, tensor in zip(self.numeric_branches, numeric_inputs):
            if branch is None or tensor.size(1) == 0:
                numeric_pooled.append(torch.zeros(batch_size, self.embed_dim, device=device))
            else:
                numeric_pooled.append(branch(tensor)[:, :, -1])

        cat_pooled = []
        for idx, (emb, branch) in enumerate(zip(self.cat_embeddings, self.cat_branches)):
            feature = x_cat[:, idx, :]
            embedded = emb(feature).permute(0, 2, 1)
            cat_pooled.append(branch(embedded)[:, :, -1])

        p_num = torch.cat(numeric_pooled, dim=1)
        p_cat = torch.cat(cat_pooled, dim=1) if cat_pooled else torch.zeros(batch_size, 0, device=device)

        gate = self.gate_linear(p_cat)
        features = torch.cat([p_num * gate, p_cat], dim=1)
        return self.head(features)