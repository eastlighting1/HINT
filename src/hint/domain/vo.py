from pydantic import BaseModel, Field, ConfigDict
from typing import List, Tuple, Union, Optional, Dict, Any

class HyperparamVO(BaseModel):
    model_config = ConfigDict(frozen=True)

class ArtifactsConfig(HyperparamVO):
    patients_file: str = "patients.parquet"
    vitals_file: str = "vitals_labs.parquet"
    vitals_mean_file: str = "vitals_labs_mean.parquet"
    interventions_file: str = "interventions.parquet"
    features_file: str = "dataset_123.parquet"
    labels_file: str = "labels.parquet"
    output_h5_prefix: str = "train_coding"
    model_name: Optional[str] = None

class ETLConfig(HyperparamVO):
    raw_dir: str = "./data/raw"
    proc_dir: str = "./data/processed"
    resources_dir: str = "./resources"
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    min_los_icu_days: float = 0.0
    min_duration_hours: int = 0
    max_duration_hours: int = 999999
    min_age: int = 0
    input_window_h: int = 6
    gap_h: int = 6
    pred_window_h: int = 4
    age_bin_edges: Tuple[int, int, int, int] = (15, 40, 65, 90)
    
    exact_level2_104: List[str] = Field(default_factory=lambda: [
        "alanine aminotransferase", "albumin", "albumin ascites", "albumin pleural", 
        "albumin urine", "alkaline phosphate", "anion gap", "aspartate aminotransferase",
        "basophils", "bicarbonate", "bilirubin", "blood urea nitrogen", "calcium",
        "calcium ionized", "calcium urine", "cardiac index", "cardiac output fick",
        "cardiac output thermodilution", "central venous pressure", "chloride",
        "chloride urine", "cholesterol", "cholesterol hdl", "cholesterol ldl", "co2",
        "co2 (etco2, pco2, etc.)", "creatinine", "creatinine ascites", 
        "creatinine body fluid", "creatinine pleural", "creatinine urine", 
        "diastolic blood pressure", "eosinophils", "fibrinogen", 
        "fraction inspired oxygen", "fraction inspired oxygen set", 
        "glascow coma scale total", "glucose", "heart rate", "height", "hematocrit", 
        "hemoglobin", "lactate", "lactate dehydrogenase", 
        "lactate dehydrogenase pleural", "lactic acid", "lymphocytes", 
        "lymphocytes ascites", "lymphocytes atypical", "lymphocytes atypical csf", 
        "lymphocytes body fluid", "lymphocytes percent", "lymphocytes pleural", 
        "magnesium", "mean blood pressure", "mean corpuscular hemoglobin", 
        "mean corpuscular hemoglobin concentration", "mean corpuscular volume", 
        "monocytes", "monocytes csf", "neutrophils", "oxygen saturation", 
        "partial pressure of carbon dioxide", "partial pressure of oxygen", 
        "partial thromboplastin time", "peak inspiratory pressure", "ph", "ph urine", 
        "phosphate", "phosphorous", "plateau pressure", "platelets", 
        "positive end-expiratory pressure", "positive end-expiratory pressure set", 
        "post void residual", "potassium", "potassium serum", "prothrombin time inr", 
        "prothrombin time pt", "pulmonary artery pressure mean", 
        "pulmonary artery pressure systolic", "pulmonary capillary wedge pressure", 
        "red blood cell count", "red blood cell count ascites", 
        "red blood cell count csf", "red blood cell count pleural", 
        "red blood cell count urine", "respiratory rate", "respiratory rate set", 
        "sodium", "systemic vascular resistance", "systolic blood pressure", 
        "temperature", "tidal volume observed", "tidal volume set", 
        "tidal volume spontaneous", "total protein", "total protein urine", 
        "troponin-i", "troponin-t", "venous pvo2", "weight", "white blood cell count", 
        "white blood cell count urine"
    ])

class ICDDataConfig(HyperparamVO):
    input_h5_prefix: str = "train_coding"
    output_h5_prefix: str = "train_intervention"
    inferred_col_name: str = "icd_inferred_code"
    data_cache_dir: str = "data/cache"

class ICDArtifactsConfig(HyperparamVO):
    model_name: str = "icd_model"
    stacker_name: str = "icd_stacker"

# [NEW] Execution Config for subsampling strategy
class ExecutionConfig(HyperparamVO):
    subset_ratio: float = 1.0

class ICDConfig(HyperparamVO):
    data: ICDDataConfig = Field(default_factory=ICDDataConfig)
    artifacts: ICDArtifactsConfig = Field(default_factory=ICDArtifactsConfig)
    
    # [NEW] Fields added to match icd_config.yaml
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    models_to_run: List[str] = Field(default_factory=lambda: ["MedBERT"])
    model_configs: Dict[str, Any] = Field(default_factory=dict)

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
    pca_components: float = 0.95
    xgb_params: dict = Field(default_factory=dict)
    xai_bg_size: int = 128
    xai_sample_size: int = 5
    xai_nsamples: Union[str, int] = 200

class CNNDataConfig(HyperparamVO):
    input_h5_prefix: str = "train_intervention"
    data_cache_dir: str = "data/cache"
    exclude_cols: List[str] = Field(default_factory=lambda: ["ICD9_CODES"])

class CNNArtifactsConfig(HyperparamVO):
    model_name: str = "intervention_model"

class CNNConfig(HyperparamVO):
    data: CNNDataConfig = Field(default_factory=CNNDataConfig)
    artifacts: CNNArtifactsConfig = Field(default_factory=CNNArtifactsConfig)
    seq_len: int = 120
    batch_size: int = 512
    epochs: int = 100
    lr: float = 0.001
    patience: int = 10
    use_cosine_scheduler: bool = False
    T_0: int = 10
    lr_patience: int = 5
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    ema_decay: float = 0.999
    embed_dim: int = 128
    cat_embed_dim: int = 32
    dropout: float = 0.5
    tcn_kernel_size: int = 5
    tcn_layers: int = 5
    tcn_dropout: float = 0.4