import torch
import torch.nn as nn
from typing import Optional, Union, Sequence, List

from .models.tcn import TemporalConvNet

class BaseICDClassifier(nn.Module):
    """Base class for ICD classification networks.

    This class stores core configuration fields used by ICD classifiers
    and standardizes the forward interface for downstream services.

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

        Raises:
            NotImplementedError: Always raised for the abstract base class.
        """
        raise NotImplementedError

class TCNClassifier(nn.Module):
    """Unified Temporal CNN model for intervention prediction.

    Combines the numeric and categorical processing capabilities into a shared
    prediction head using TemporalConvNet blocks. Supports optional ICD feature
    fusion via projection.

    Attributes:
        embed_dim (int): Embedding dimension for numeric branch output.
        numeric_branch (TemporalConvNet): TCN stack for numeric inputs.
        cat_embeddings (nn.ModuleList): Embeddings for categorical inputs.
        cat_branches (nn.ModuleList): TCN stacks for categorical embeddings.
        icd_dim (int): ICD context dimension.
        icd_projector (Optional[nn.Sequential]): Projector for ICD features.
        head (nn.Sequential): Output head for classification.
    """
    def __init__(self, in_chs: Union[int, Sequence[int]], n_cls: int, vocab_sizes: List[int], icd_dim: int = 0, embed_dim: int = 128, cat_embed_dim: int = 32, head_drop: float = 0.3, tcn_drop: float = 0.2, kernel: int = 5, layers: int = 5) -> None:
        """Initialize the TCNClassifier.

        Args:
            in_chs (Union[int, Sequence[int]]): Numeric feature channel size(s).
            n_cls (int): Number of output classes.
            vocab_sizes (List[int]): Cardinalities for categorical features.
            icd_dim (int): ICD context dimension.
            embed_dim (int): Output embedding size for numeric branch.
            cat_embed_dim (int): Embedding size for categorical features.
            head_drop (float): Dropout for the output head.
            tcn_drop (float): Dropout for the TCN stacks.
            kernel (int): TCN kernel size.
            layers (int): Number of dilated residual layers.
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        if isinstance(in_chs, (list, tuple)):
            numeric_in_dim = sum(in_chs)
        else:
            numeric_in_dim = in_chs
            
        num_channels = [embed_dim] * layers
        
        self.numeric_branch = TemporalConvNet(
            num_inputs=numeric_in_dim,
            num_channels=num_channels,
            kernel_size=kernel,
            dropout=tcn_drop
        )
        
        self.cat_embeddings = nn.ModuleList()
        self.cat_branches = nn.ModuleList()
        
        cat_channels = [cat_embed_dim] * layers
        
        for vs in vocab_sizes:
            self.cat_embeddings.append(nn.Embedding(vs, cat_embed_dim, padding_idx=0))
            self.cat_branches.append(TemporalConvNet(
                num_inputs=cat_embed_dim,
                num_channels=cat_channels,
                kernel_size=kernel,
                dropout=tcn_drop
            ))
            
        self.icd_dim = icd_dim
        self.icd_projector = None
        if icd_dim > 0:
            self.icd_projector = nn.Sequential(nn.Linear(icd_dim, embed_dim), nn.ReLU(), nn.Dropout(0.2))
        
        total_feature_dim = embed_dim
        if vocab_sizes:
            total_feature_dim += (len(vocab_sizes) * cat_embed_dim)
        if icd_dim > 0:
            total_feature_dim += embed_dim
        
        self.head = nn.Sequential(nn.Linear(total_feature_dim, total_feature_dim // 2), nn.ReLU(), nn.Dropout(head_drop), nn.Linear(total_feature_dim // 2, n_cls))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_icd: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the TCNClassifier forward pass.

        Args:
            x_num (torch.Tensor): Numeric time-series input.
            x_cat (torch.Tensor): Categorical features or sequences.
            x_icd (Optional[torch.Tensor]): ICD context vector.

        Returns:
            torch.Tensor: Logits shaped (batch, n_cls).
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
        if self.icd_projector is not None and x_icd is not None:
            icd_feat = self.icd_projector(x_icd)
        
        feats = [num_pool] + cat_pool_list
        if icd_feat is not None:
            feats.append(icd_feat)
        
        final_feat = torch.cat(feats, dim=1)
        return self.head(final_feat)

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
        "TCNClassifier": TCNClassifier
    }
    
    if model_name not in mapping:
        raise ValueError(f"Model {model_name} not found. Options: {list(mapping.keys())}")
    return mapping[model_name]