from pydantic import BaseModel, Field, ConfigDict
from typing import List, Tuple, Union, Optional, Dict, Any
from enum import Enum
from pathlib import Path

class HyperparamVO(BaseModel):
    """Base configuration value object.

    This class provides shared Pydantic configuration for immutable
    hyperparameter containers.
    """
    model_config = ConfigDict(frozen=True)

class ArtifactsConfig(HyperparamVO):
    """Artifact file name settings for ETL outputs.

    Attributes:
        patients_file (str): Patient cohort file name.
        vitals_file (str): Vitals and lab file name.
        vitals_mean_file (str): Aggregated vitals file name.
        interventions_file (str): Interventions file name.
        features_file (str): Feature dataset file name.
        vent_targets_file (str): Ventilation targets file name.
        icd_targets_file (str): ICD targets file name.
        icd_meta_file (str): ICD metadata file name.
        labels_file (str): Labels file name.
        output_h5_prefix (str): Prefix for output HDF5 files.
        model_name (Optional[str]): Optional model name override.
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
    """Standardized column keys used in the ETL pipeline.

    Attributes:
        STAY_ID (str): ICU stay identifier.
        HOUR_IN (str): Hour index within stay.
        INPUT_DYN_VITALS (str): Numeric dynamic feature key.
        INPUT_DYN_CATEGORICAL (str): Categorical dynamic feature key.
        STATIC_INPUT_IDS (str): Static tokenizer input IDs key.
        STATIC_ATTN_MASK (str): Static attention mask key.
        STATIC_CANDS (str): Static candidate list key.
    """
    STAY_ID = "sid"
    HOUR_IN = "hour"
    INPUT_DYN_VITALS = "X_num"
    INPUT_DYN_CATEGORICAL = "X_cat"
    STATIC_INPUT_IDS = "static_input_ids"
    STATIC_ATTN_MASK = "static_attention_mask"
    STATIC_CANDS = "static_candidates"

class ICDKeys(str, Enum):
    """Standardized keys for ICD targets and features.

    Attributes:
        TARGET_ICD_MULTI (str): Multi-label ICD target key.
        FEATURE_ICD_EMBEDDING (str): ICD embedding feature key.
    """
    TARGET_ICD_MULTI = "y"
    FEATURE_ICD_EMBEDDING = "X_icd"

class InterventionKeys(str, Enum):
    """Standardized keys for intervention targets.

    Attributes:
        TARGET_VENT_STATE (str): Ventilation state target key.
    """
    TARGET_VENT_STATE = "y_vent"

def _load_exact_features() -> List[str]:
    """Load the exact feature whitelist for vitals mapping.

    This function searches known locations for the feature list and
    returns the first successfully loaded result.

    Returns:
        List[str]: List of feature names to retain.
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
    """Configuration values for the ETL pipeline.

    Attributes:
        raw_dir (str): Directory containing raw data.
        proc_dir (str): Directory for processed outputs.
        resources_dir (str): Directory for auxiliary resources.
        artifacts (ArtifactsConfig): Artifact file naming settings.
        keys (ETLKeys): Standardized key names.
        min_los_icu_days (float): Minimum ICU length of stay in days.
        min_duration_hours (int): Minimum stay duration in hours.
        max_duration_hours (int): Maximum stay duration in hours.
        min_age (int): Minimum patient age.
        input_window_h (int): Input window size in hours.
        gap_h (int): Gap between input and prediction windows.
        pred_window_h (int): Prediction window size in hours.
        age_bin_edges (Tuple[int, int, int, int]): Age bin edges.
        exact_level2_104 (List[str]): Exact feature list for filtering.
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
    """Data-related configuration for ICD training.

    Attributes:
        input_h5_prefix (str): Input HDF5 prefix for ICD training data.
        output_h5_prefix (str): Output HDF5 prefix for generated data.
        inferred_col_name (str): Column name for inferred ICD codes.
        data_cache_dir (str): Cache directory for intermediate artifacts.
    """
    input_h5_prefix: str = "train_coding"
    output_h5_prefix: str = "train_intervention"
    inferred_col_name: str = "icd_inferred_code"
    data_cache_dir: str = "data/cache"

class ICDArtifactsConfig(HyperparamVO):
    """Artifact naming configuration for ICD models.

    Attributes:
        model_name (str): Base name for ICD model artifacts.
        stacker_name (str): Base name for stacking model artifacts.
    """
    model_name: str = "icd_model"
    stacker_name: str = "icd_stacker"

class ExecutionConfig(HyperparamVO):
    """Execution-time settings.

    Attributes:
        subset_ratio (float): Fraction of data to use during training.
    """
    subset_ratio: float = 1.0

class ICDConfig(HyperparamVO):
    """Configuration for ICD model training and inference.

    Attributes:
        data (ICDDataConfig): Data-related settings.
        artifacts (ICDArtifactsConfig): Artifact naming settings.
        keys (ICDKeys): Standardized ICD keys.
        execution (ExecutionConfig): Execution settings.
        models_to_run (List[str]): Model names to execute.
        model_configs (Dict[str, Any]): Per-model overrides.
        loss_type (str): Loss function identifier.
        model_name (str): Pretrained model identifier.
        batch_size (int): Training batch size.
        lr (float): Learning rate.
        epochs (int): Number of training epochs.
        patience (int): Early stopping patience.
        dropout (float): Dropout probability.
        num_workers (int): Data loader worker count.
        pin_memory (bool): Whether to pin memory in data loaders.
        max_length (int): Maximum sequence length for token inputs.
        test_split_size (float): Test split proportion.
        val_split_size (float): Validation split proportion.
        top_k_labels (int): Number of labels to keep.
        topk_eval (int): Top-k evaluation size.
        sampler_alpha (float): Sampling alpha for class balancing.
        cb_beta (float): Class-balanced loss beta.
        focal_gamma (float): Focal loss gamma.
        logit_adjust_tau (float): Logit adjustment temperature.
        entropy_reg_lambda (float): Entropy regularization weight.
        freeze_bert_epochs (int): Epochs to freeze the backbone.
        use_amp (bool): Whether to use mixed precision training.
        grad_clip_norm (float): Global norm for gradient clipping.
        pca_components (float): PCA component threshold.
        xgb_params (dict): XGBoost parameter overrides.
        xai_bg_size (int): Background size for explainers.
        xai_sample_size (int): Sample size for explanations.
        xai_nsamples (Union[str, int]): Number of explainer samples.
    """
    data: ICDDataConfig = Field(default_factory=ICDDataConfig)
    artifacts: ICDArtifactsConfig = Field(default_factory=ICDArtifactsConfig)
    keys: ICDKeys = Field(default_factory=lambda: ICDKeys)
    
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    models_to_run: List[str] = Field(default_factory=lambda: ["MedBERT"])
    model_configs: Dict[str, Any] = Field(default_factory=dict)
    
    loss_type: str = "clpl"

    model_name: str = "Charangan/MedBERT"
    batch_size: int = 2048
    lr: float = 1e-5
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
    freeze_bert_epochs: int = 1
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    pca_components: float = 0.95
    xgb_params: dict = Field(default_factory=dict)
    xai_bg_size: int = 128
    xai_sample_size: int = 5
    xai_nsamples: Union[str, int] = 200

class CNNDataConfig(HyperparamVO):
    """Data-related configuration for intervention training.

    Attributes:
        input_h5_prefix (str): Input HDF5 prefix for intervention data.
        data_cache_dir (str): Cache directory for intermediate artifacts.
        exclude_cols (List[str]): Columns to exclude from features.
    """
    input_h5_prefix: str = "train_intervention"
    data_cache_dir: str = "data/cache"
    exclude_cols: List[str] = Field(default_factory=lambda: ["ICD9_CODES"])

class CNNArtifactsConfig(HyperparamVO):
    """Artifact naming configuration for intervention models.

    Attributes:
        model_name (str): Base name for intervention model artifacts.
    """
    model_name: str = "intervention_model"

class CNNConfig(HyperparamVO):
    """Configuration for intervention model training.

    Attributes:
        data (CNNDataConfig): Data-related settings.
        artifacts (CNNArtifactsConfig): Artifact naming settings.
        keys (InterventionKeys): Standardized intervention keys.
        seq_len (int): Sequence length for time-series input.
        batch_size (int): Training batch size.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        patience (int): Early stopping patience.
        use_cosine_scheduler (bool): Whether to use cosine scheduling.
        T_0 (int): Cosine scheduler initial period.
        lr_patience (int): Plateau scheduler patience.
        focal_gamma (float): Focal loss gamma.
        early_stop_metric (str): Metric key to early stop on.
    """
    data: CNNDataConfig = Field(default_factory=CNNDataConfig)
    artifacts: CNNArtifactsConfig = Field(default_factory=CNNArtifactsConfig)
    keys: InterventionKeys = Field(default_factory=lambda: InterventionKeys)
    
    seq_len: int = 120
    batch_size: int = 512
    epochs: int = 100
    lr: float = 0.001
    patience: int = 10
    use_cosine_scheduler: bool = False
    T_0: int = 10
    lr_patience: int = 5
    focal_gamma: float = 2.0
    early_stop_metric: str = "f1"
    label_smoothing: float = 0.1
    ema_decay: float = 0.999
    embed_dim: int = 128
    cat_embed_dim: int = 32
    dropout: float = 0.5
    tcn_kernel_size: int = 5
    tcn_layers: int = 5
    tcn_dropout: float = 0.4
