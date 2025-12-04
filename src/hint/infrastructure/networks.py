import torch
import torch.nn as nn
from typing import List, Sequence, Optional, Dict
from transformers import AutoModel

# -------------------------------------------------------------------
# CNN Components (Ported from CNN.py)
# -------------------------------------------------------------------

class DilatedResidualBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int, stride: int, dilation: int, dropout: float) -> None:
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
        identity = x if self.shortcut is None else self.shortcut(x)

        out = self.conv1(x)[..., : x.size(2)]
        out = self.relu1(self.bn1(out))
        out = self.drop1(out)

        out = self.conv2(out)[..., : x.size(2)]
        out = self.relu2(self.bn2(out))
        out = self.drop2(out)

        out = out + identity
        return self.relu_out(out)


class GFINet_CNN(nn.Module):
    """
    Multi-branch temporal CNN with ICD-conditioned gating.
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
            base: List[nn.Module] = [
                nn.Sequential(
                    nn.Conv1d(in_dim, out_dim, 1),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                )
            ]
            for d in dilations:
                base.append(DilatedResidualBlock(out_dim, out_dim, kernel, 1, d, tcn_drop))
            return nn.Sequential(*base)

        self.numeric_branches = nn.ModuleList()
        for channels in in_chs:
            if channels > 0:
                self.numeric_branches.append(build_tcn_stack(channels, embed_dim))
            else:
                self.numeric_branches.append(None)

        num_in_numeric = 3 * embed_dim

        self.cat_feat_names = list(cat_vocab_sizes.keys())
        self.cat_embeddings = nn.ModuleList()
        self.cat_branches = nn.ModuleList()
        total_cat_dim = 0

        for feat_name in self.cat_feat_names:
            vocab_size = cat_vocab_sizes[feat_name]
            self.cat_embeddings.append(nn.Embedding(vocab_size, cat_embed_dim, padding_idx=0))
            self.cat_branches.append(build_tcn_stack(cat_embed_dim, cat_embed_dim))
            total_cat_dim += cat_embed_dim

        self.gate_linear = nn.Sequential(nn.Linear(total_cat_dim, num_in_numeric), nn.Sigmoid())

        head_in = num_in_numeric + total_cat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.ReLU(),
            nn.Dropout(head_drop),
            nn.Linear(head_in // 2, n_cls),
        )

    def forward(self, x_full: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        xg1 = x_full.index_select(1, self.g1)
        xg2 = x_full.index_select(1, self.g2)
        xr = x_full.index_select(1, self.rest)

        numeric_inputs = [xg1, xg2, xr]
        batch_size = x_full.size(0)
        device = x_full.device

        numeric_pooled: List[torch.Tensor] = []
        for branch, tensor in zip(self.numeric_branches, numeric_inputs):
            if branch is None or tensor.size(1) == 0:
                numeric_pooled.append(torch.zeros(batch_size, self.embed_dim, device=device))
                continue
            hidden = branch(tensor)
            numeric_pooled.append(hidden[:, :, -1])

        p1, p2, p3 = numeric_pooled

        cat_pooled: List[torch.Tensor] = []
        for idx, (embedding, branch) in enumerate(zip(self.cat_embeddings, self.cat_branches)):
            feature = x_cat[:, idx, :]
            embedded = embedding(feature).permute(0, 2, 1)
            cat_hidden = branch(embedded)
            cat_pooled.append(cat_hidden[:, :, -1])

        p_num = torch.cat([p1, p2, p3], dim=1)
        p_cat = torch.cat(cat_pooled, dim=1) if cat_pooled else torch.zeros(batch_size, 0, device=device)

        gate = self.gate_linear(p_cat)
        gated_num = p_num * gate

        features = torch.cat([gated_num, p_cat], dim=1)
        return self.head(features)

# -------------------------------------------------------------------
# ICD Components (Ported from ICD.py)
# -------------------------------------------------------------------

class MedBERTClassifier(nn.Module):
    """
    BERT-based classifier head for ICD coding.
    Originally build_classifier_head in ICD.py
    """
    def __init__(self, model_name: str, num_num: int, num_cls: int, drop: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dp = nn.Dropout(drop)
        self.fc = nn.Linear(hidden + num_num, num_cls)

    def forward(self, input_ids, mask, numerical):
        out = self.bert(input_ids=input_ids, attention_mask=mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state.mean(dim=1)
        
        # Concatenate BERT embedding with numerical features
        x = torch.cat((pooled, numerical), dim=1)
        return self.fc(self.dp(x))

    def set_backbone_grad(self, requires_grad: bool):
        for p in self.bert.parameters():
            p.requires_grad = requires_grad