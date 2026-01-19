"""Summary of the networks module.

Longer description of the module purpose and usage.
"""

import torch

import torch.nn as nn

from typing import Optional, Union, Sequence, List



from .models.tcn import TemporalConvNet



class BaseICDClassifier(nn.Module):

    """Summary of BaseICDClassifier purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    dropout_val (Any): Description of dropout_val.
    input_dim (Any): Description of input_dim.
    num_classes (Any): Description of num_classes.
    seq_len (Any): Description of seq_len.
    """

    def __init__(self, num_classes: int, input_dim: int, seq_len: int, dropout: float = 0.3, **kwargs):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        num_classes (Any): Description of num_classes.
        input_dim (Any): Description of input_dim.
        seq_len (Any): Description of seq_len.
        dropout (Any): Description of dropout.
        kwargs (Any): Description of kwargs.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
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

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
        input_ids (Any): Description of input_ids.
        attention_mask (Any): Description of attention_mask.
        x_num (Any): Description of x_num.
        return_embeddings (Any): Description of return_embeddings.
        kwargs (Any): Description of kwargs.
        
        Returns:
        torch.Tensor: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        raise NotImplementedError



class TCNClassifier(nn.Module):

    """Summary of TCNClassifier purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    cat_branches (Any): Description of cat_branches.
    cat_embeddings (Any): Description of cat_embeddings.
    embed_dim (Any): Description of embed_dim.
    head (Any): Description of head.
    icd_dim (Any): Description of icd_dim.
    icd_projector (Any): Description of icd_projector.
    numeric_branch (Any): Description of numeric_branch.
    """

    def __init__(self, in_chs: Union[int, Sequence[int]], n_cls: int, vocab_sizes: List[int], icd_dim: int = 0, embed_dim: int = 128, cat_embed_dim: int = 32, head_drop: float = 0.3, tcn_drop: float = 0.2, kernel: int = 5, layers: int = 5, use_icd_gating: bool = True) -> None:

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        in_chs (Any): Description of in_chs.
        n_cls (Any): Description of n_cls.
        vocab_sizes (Any): Description of vocab_sizes.
        icd_dim (Any): Description of icd_dim.
        embed_dim (Any): Description of embed_dim.
        cat_embed_dim (Any): Description of cat_embed_dim.
        head_drop (Any): Description of head_drop.
        tcn_drop (Any): Description of tcn_drop.
        kernel (Any): Description of kernel.
        layers (Any): Description of layers.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
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
        self.use_icd_gating = use_icd_gating

        self.icd_projector = None

        if icd_dim > 0:

            self.icd_projector = nn.Sequential(nn.Linear(icd_dim, embed_dim), nn.ReLU(), nn.Dropout(0.2))



        total_feature_dim = embed_dim

        if vocab_sizes:

            total_feature_dim += (len(vocab_sizes) * cat_embed_dim)

        if icd_dim > 0:

            total_feature_dim += embed_dim


        cat_feature_dim = (len(vocab_sizes) * cat_embed_dim) + (embed_dim if icd_dim > 0 else 0)
        self.icd_gate = None
        if self.use_icd_gating and icd_dim > 0 and cat_feature_dim > 0:
            self.icd_gate = nn.Sequential(nn.Linear(cat_feature_dim, embed_dim), nn.Sigmoid())

        self.head = nn.Sequential(nn.Linear(total_feature_dim, total_feature_dim // 2), nn.ReLU(), nn.Dropout(head_drop), nn.Linear(total_feature_dim // 2, n_cls))



    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_icd: Optional[torch.Tensor] = None) -> torch.Tensor:

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
        x_num (Any): Description of x_num.
        x_cat (Any): Description of x_cat.
        x_icd (Any): Description of x_icd.
        
        Returns:
        torch.Tensor: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
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



        if self.icd_gate is not None and icd_feat is not None:
            if cat_pool_list:
                p_cat = torch.cat(cat_pool_list + [icd_feat], dim=1)
            else:
                p_cat = icd_feat
            gate = self.icd_gate(p_cat)
            num_pool = num_pool * gate

        feats = [num_pool] + cat_pool_list

        if icd_feat is not None:

            feats.append(icd_feat)



        final_feat = torch.cat(feats, dim=1)

        return self.head(final_feat)



def get_network_class(model_name: str) -> type:

    """Summary of get_network_class.
    
    Longer description of the get_network_class behavior and usage.
    
    Args:
    model_name (Any): Description of model_name.
    
    Returns:
    type: Description of the return value.
    
    Raises:
    Exception: Description of why this exception might be raised.
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

        "TCN": TCNClassifier

    }



    if model_name not in mapping:

        raise ValueError(f"Model {model_name} not found. Options: {list(mapping.keys())}")

    return mapping[model_name]
