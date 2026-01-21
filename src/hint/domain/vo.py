"""Summary of the vo module.

Longer description of the module purpose and usage.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field



class HyperparamVO(BaseModel):

    """Summary of HyperparamVO purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    model_config = ConfigDict(frozen=True)



class ArtifactsConfig(HyperparamVO):

    """Summary of ArtifactsConfig purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    patients_file: str = "patients.parquet"

    vitals_file: str = "vitals_labs.parquet"

    vitals_mean_file: str = "vitals_labs_mean.parquet"

    interventions_file: str = "interventions.parquet"

    features_file: str = "dataset_123.parquet"

    vent_targets_file: str = "targets_vent.parquet"

    icd_targets_file: str = "targets_icd.parquet"

    icd_meta_file: str = "meta_icd_classes.json"

    labels_file: str = "labels.parquet"

    output_h5_prefix: str = "train_coding"

    model_name: Optional[str] = None



class ETLKeys(str, Enum):

    """Summary of ETLKeys purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    STAY_ID = "sid"

    HOUR_IN = "hour"

    INPUT_DYN_VITALS = "X_num"

    INPUT_DYN_CATEGORICAL = "X_cat"

    STATIC_INPUT_IDS = "static_input_ids"

    STATIC_ATTN_MASK = "static_attention_mask"

    STATIC_CANDS = "static_candidates"



class ICDKeys(str, Enum):

    """Summary of ICDKeys purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    TARGET_ICD_MULTI = "y"

    FEATURE_ICD_EMBEDDING = "X_icd"



class InterventionKeys(str, Enum):

    """Summary of InterventionKeys purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    TARGET_VENT_STATE = "y_vent"



def _load_exact_features() -> List[str]:

    """Summary of _load_exact_features.
    
    Longer description of the _load_exact_features behavior and usage.
    
    Args:
    None (None): This function does not accept arguments.
    
    Returns:
    List[str]: Description of the return value.
    
    Raises:
    Exception: Description of why this exception might be raised.
    """

    candidates = [

        Path("resources/exact_level2_104.txt"),

        Path("./resources/exact_level2_104.txt"),

        Path(__file__).parent.parent.parent.parent.parent / "resources" / "exact_level2_104.txt"

    ]



    for path in candidates:

        if path.exists():

            try:

                with path.open("r", encoding="utf-8") as f:

                    return [line.strip() for line in f if line.strip()]

            except Exception:

                continue



    return []



class ETLConfig(HyperparamVO):

    """Summary of ETLConfig purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    raw_dir: str = "./data/raw"

    proc_dir: str = "./data/processed"

    resources_dir: str = "./resources"

    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)

    keys: ETLKeys = Field(default_factory=lambda: ETLKeys)



    min_los_icu_days: float = 0.0

    min_duration_hours: int = 0

    max_duration_hours: int = 999999

    min_age: int = 0

    input_window_h: int = 6

    gap_h: int = 6

    pred_window_h: int = 4

    age_bin_edges: Tuple[int, int, int, int] = (15, 40, 65, 90)



    exact_level2_104: List[str] = Field(default_factory=_load_exact_features)



class ICDDataConfig(HyperparamVO):

    """Summary of ICDDataConfig purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    input_h5_prefix: str = "train_coding"

    output_h5_prefix: str = "train_intervention"

    inferred_col_name: str = "icd_inferred_code"

    data_cache_dir: str = "data/cache"



class ICDArtifactsConfig(HyperparamVO):

    """Summary of ICDArtifactsConfig purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    model_name: str = "icd_model"

    stacker_name: str = "icd_stacker"



class ExecutionConfig(HyperparamVO):

    """Summary of ExecutionConfig purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    subset_ratio: float = 1.0



class ICDConfig(HyperparamVO):

    """Summary of ICDConfig purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    data: ICDDataConfig = Field(default_factory=ICDDataConfig)

    artifacts: ICDArtifactsConfig = Field(default_factory=ICDArtifactsConfig)

    keys: ICDKeys = Field(default_factory=lambda: ICDKeys)



    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


    models_to_run: List[str] = Field(default_factory=lambda: ["MedBERT"])

    model_configs: Dict[str, Any] = Field(default_factory=dict)



    loss_type: str = "clpl"
    adaptive_clpl_head_size: int = 800
    adaptive_clpl_tail_sample_size: int = 800
    adaptive_clpl_logit_clip: float = 30.0
    lambda_sparse: float = 1e-3



    model_name: str = "Charangan/MedBERT"

    batch_size: int = 2048

    lr: float = 1e-5
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 2
    lr_plateau_min_lr: float = 1.0e-6

    epochs: int = 100

    patience: int = 5

    dropout: float = 0.3

    num_workers: int = 4

    pin_memory: bool = True

    max_length: int = 32

    test_split_size: float = 0.2

    val_split_size: float = 0.15625

    top_k_labels: int = 500

    topk_eval: int = 5

    sampler_alpha: float = 0.5

    cb_beta: float = 0.999

    focal_gamma: float = 1.5

    logit_adjust_tau: float = 1.0

    entropy_reg_lambda: float = 1e-3

    pseudo_label_ce_weight: float = 0.0
    pseudo_label_margin: float = 0.0
    class_weight_power: float = 0.5
    class_weight_clip: float = 10.0

    freeze_bert_epochs: int = 1

    use_amp: bool = True

    grad_clip_norm: float = 1.0

    pca_components: float = 0.95

    xgb_params: dict = Field(default_factory=dict)

    xai_bg_size: int = 128

    xai_sample_size: int = 5

    xai_nsamples: Union[str, int] = 200



class CNNDataConfig(HyperparamVO):

    """Summary of CNNDataConfig purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    input_h5_prefix: str = "train_intervention"

    data_cache_dir: str = "data/cache"

    exclude_cols: List[str] = Field(default_factory=lambda: ["ICD9_CODES"])



class CNNArtifactsConfig(HyperparamVO):

    """Summary of CNNArtifactsConfig purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    model_name: str = "intervention_model"



class CNNConfig(HyperparamVO):

    """Summary of CNNConfig purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    None (None): No documented attributes.
    """

    data: CNNDataConfig = Field(default_factory=CNNDataConfig)

    artifacts: CNNArtifactsConfig = Field(default_factory=CNNArtifactsConfig)

    keys: InterventionKeys = Field(default_factory=lambda: InterventionKeys)



    seq_len: int = 120

    batch_size: int = 512

    epochs: int = 100

    lr: float = 3e-4

    patience: int = 10

    grad_clip_norm: float = 1.0

    use_cosine_scheduler: bool = True

    T_0: int = 10

    lr_patience: int = 5

    focal_gamma: float = 2.0
    use_weighted_sampler: bool = True

    early_stop_metric: str = "f1"

    label_smoothing: float = 0.1

    ema_decay: float = 0.999

    embed_dim: int = 128

    cat_embed_dim: int = 32

    use_icd_gating: bool = True

    dropout: float = 0.5

    tcn_kernel_size: int = 5

    tcn_layers: int = 5

    tcn_dropout: float = 0.4
