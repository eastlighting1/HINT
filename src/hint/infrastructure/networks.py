import torch
import torch.nn as nn
from typing import Optional, Union, Sequence, List

class BaseICDClassifier(nn.Module):
    """Base class for ICD classification networks.

    This class stores core configuration fields used by ICD classifiers.

    Attributes:
        num_classes (int): Number of output classes.
        input_dim (int): Number of input features.
        seq_len (int): Sequence length for time-series input.
        dropout_val (float): Dropout probability.
    """
    def __init__(self, num_classes: int, input_dim: int, seq_len: int, dropout: float = 0.3, **kwargs):
        """Initialize common classifier metadata.

        Args:
            num_classes (int): Number of output classes.
            input_dim (int): Number of input features.
            seq_len (int): Sequence length for time-series input.
            dropout (float): Dropout probability.
            **kwargs (Any): Additional model arguments.
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.dropout_val = dropout
        
    def forward(self, input_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None, 
                x_num: Optional[torch.Tensor] = None,
                return_embeddings: bool = False,
                **kwargs) -> torch.Tensor:
        """Run the classifier forward pass.

        Args:
            input_ids (Optional[torch.Tensor]): Optional token IDs.
            attention_mask (Optional[torch.Tensor]): Optional attention mask.
            x_num (Optional[torch.Tensor]): Optional numeric features.
            return_embeddings (bool): Whether to return embeddings.
            **kwargs (Any): Additional model inputs.

        Returns:
            torch.Tensor: Model outputs or logits.
        """
        raise NotImplementedError

def get_network_class(model_name: str) -> type:
    """Resolve a model name to its network class.

    Args:
        model_name (str): Name of the model architecture.

    Returns:
        type: Network class for the requested model.

    Raises:
        ValueError: If the model name is unknown.
    """
    from .models.medbert import MedBERTClassifier
    from .models.grud import GRUDClassifier
    from .models.tst import TSTClassifier
    from .models.latent_ode import LatentODEClassifier     
    from .models.itransformer import iTransformerClassifier
    
    from .models.tabnet import TabNetICD
    from .models.dcn_v2 import DCNv2ICD
    from .models.ft_transformer import FTTransformerICD
    
    mapping = {
        "MedBERT": MedBERTClassifier,
        "GRU-D": GRUDClassifier,
        "TST": TSTClassifier,
        "LatentODE": LatentODEClassifier,
        "iTransformer": iTransformerClassifier,
        "TabNet": TabNetICD,
        "DCNv2": DCNv2ICD,
        "FTTransformer": FTTransformerICD
    }
    
    if model_name not in mapping:
        raise ValueError(f"Model {model_name} not found. Options: {list(mapping.keys())}")
    return mapping[model_name]


class DilatedResidualBlock(nn.Module):
    """Residual block with dilated temporal convolutions.

    Attributes:
        conv1 (nn.Conv1d): First dilated convolution.
        bn1 (nn.BatchNorm1d): Batch norm after conv1.
        relu1 (nn.ReLU): Activation for conv1 output.
        drop1 (nn.Dropout): Dropout after conv1.
        conv2 (nn.Conv1d): Second dilated convolution.
        bn2 (nn.BatchNorm1d): Batch norm after conv2.
        relu2 (nn.ReLU): Activation for conv2 output.
        drop2 (nn.Dropout): Dropout after conv2.
        shortcut (Optional[nn.Conv1d]): Projection for residual path.
        relu_out (nn.ReLU): Final activation.
    """
    def __init__(self, in_c: int, out_c: int, kernel: int, stride: int, dilation: int, dropout: float) -> None:
        """Initialize the dilated residual block.

        Args:
            in_c (int): Input channel count.
            out_c (int): Output channel count.
            kernel (int): Convolution kernel size.
            stride (int): Convolution stride.
            dilation (int): Dilation factor.
            dropout (float): Dropout probability.
        """
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
        """Apply residual block transformation.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T].

        Returns:
            torch.Tensor: Output tensor with matching temporal length.
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

class GFINet_CNN(nn.Module):
    """Temporal CNN model for intervention prediction.

    This network combines numeric time-series, categorical embeddings,
    and optional ICD features into a shared prediction head.

    Attributes:
        embed_dim (int): Embedding dimension for numeric branch.
        numeric_branch (nn.Module): TCN stack for numeric features.
        cat_embeddings (nn.ModuleList): Embeddings for categorical inputs.
        cat_branches (nn.ModuleList): TCN stacks for categorical inputs.
        icd_dim (int): ICD feature dimensionality.
        icd_projector (Optional[nn.Module]): Projection for ICD features.
        head (nn.Module): Prediction head.
    """
    def __init__(self, in_chs: Union[int, Sequence[int]], n_cls: int, vocab_sizes: List[int], icd_dim: int = 0, embed_dim: int = 128, cat_embed_dim: int = 32, head_drop: float = 0.3, tcn_drop: float = 0.2, kernel: int = 5, layers: int = 5) -> None:
        """Initialize the intervention CNN model.

        Args:
            in_chs (Union[int, Sequence[int]]): Numeric input channel counts.
            n_cls (int): Number of output classes.
            vocab_sizes (List[int]): Vocabulary sizes for categorical inputs.
            icd_dim (int): ICD feature dimensionality.
            embed_dim (int): Numeric embedding dimension.
            cat_embed_dim (int): Categorical embedding dimension.
            head_drop (float): Dropout for the head layer.
            tcn_drop (float): Dropout for TCN blocks.
            kernel (int): TCN kernel size.
            layers (int): Number of TCN layers.
        """
        super().__init__()
        self.embed_dim = embed_dim
        dilations = [2**i for i in range(layers)]
        def build_tcn_stack(in_dim: int, out_dim: int) -> nn.Sequential:
            """Build a temporal convolution stack.

            Args:
                in_dim (int): Input channel count.
                out_dim (int): Output channel count.

            Returns:
                nn.Sequential: TCN stack module.
            """
            base: List[nn.Module] = [nn.Sequential(nn.Conv1d(in_dim, out_dim, 1), nn.BatchNorm1d(out_dim), nn.ReLU())]
            for d in dilations: base.append(DilatedResidualBlock(out_dim, out_dim, kernel, 1, d, tcn_drop))
            return nn.Sequential(*base)
        if isinstance(in_chs, (list, tuple)): numeric_in_dim = sum(in_chs)
        else: numeric_in_dim = in_chs
        self.numeric_branch = build_tcn_stack(numeric_in_dim, embed_dim)
        self.cat_embeddings = nn.ModuleList()
        self.cat_branches = nn.ModuleList()
        for vs in vocab_sizes:
            self.cat_embeddings.append(nn.Embedding(vs, cat_embed_dim, padding_idx=0))
            self.cat_branches.append(build_tcn_stack(cat_embed_dim, cat_embed_dim))
        self.icd_dim = icd_dim
        self.icd_projector = None
        if icd_dim > 0: self.icd_projector = nn.Sequential(nn.Linear(icd_dim, embed_dim), nn.ReLU(), nn.Dropout(0.2))
        total_feature_dim = embed_dim
        if vocab_sizes: total_feature_dim += (len(vocab_sizes) * cat_embed_dim)
        if icd_dim > 0: total_feature_dim += embed_dim
        self.head = nn.Sequential(nn.Linear(total_feature_dim, total_feature_dim // 2), nn.ReLU(), nn.Dropout(head_drop), nn.Linear(total_feature_dim // 2, n_cls))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_icd: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the forward pass for intervention prediction.

        Args:
            x_num (torch.Tensor): Numeric time-series inputs.
            x_cat (torch.Tensor): Categorical time-series inputs.
            x_icd (Optional[torch.Tensor]): Optional ICD feature vector.

        Returns:
            torch.Tensor: Class logits for intervention prediction.
        """
        num_out = self.numeric_branch(x_num)
        num_pool = num_out[:, :, -1] 
        cat_pool_list = []
        
        if x_cat is not None and len(self.cat_embeddings) > 0:
            for i, (emb, branch) in enumerate(zip(self.cat_embeddings, self.cat_branches)):
                if x_cat.dim() == 2:
                    feat = x_cat[:, i]
                    x_emb = emb(feat)
                    cat_pool_list.append(x_emb)
                else:
                    feat = x_cat[:, i, :]
                    x_emb = emb(feat).permute(0, 2, 1)
                    cat_out = branch(x_emb)
                    cat_pool_list.append(cat_out[:, :, -1])

        icd_feat = None
        if self.icd_projector is not None and x_icd is not None: icd_feat = self.icd_projector(x_icd)
        feats = [num_pool] + cat_pool_list
        if icd_feat is not None: feats.append(icd_feat)
        final_feat = torch.cat(feats, dim=1)
        return self.head(final_feat)
