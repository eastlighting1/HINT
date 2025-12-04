from omegaconf import DictConfig
from hint.foundation.dtos import AppContext
from hint.domain.vo import ETLConfig, ICDConfig, CNNConfig

def load_app_context(cfg: DictConfig) -> AppContext:
    """
    Convert Hydra DictConfig to strongly typed AppContext.
    """
    # ETL Config extraction (assuming default values if not present in Hydra cfg)
    etl_cfg = ETLConfig(
        raw_dir=cfg.get("data", {}).get("raw_dir", "./data/raw"),
        proc_dir=cfg.get("data", {}).get("proc_dir", "./data/processed"),
        resources_dir=cfg.get("data", {}).get("resources_dir", "./resources")
    )
    
    # ICD Config extraction
    icd_raw = cfg.get("icd", {})
    icd_cfg = ICDConfig(
        data_path=icd_raw.get("data_path", "data/processed/dataset_123_answer.parquet"),
        model_name=icd_raw.get("model_name", "Charangan/MedBERT"),
        batch_size=icd_raw.get("batch_size", 2048),
        lr=icd_raw.get("lr", 1e-5),
        epochs=icd_raw.get("epochs", 100),
        patience=icd_raw.get("patience", 5),
        dropout=icd_raw.get("dropout", 0.3),
        # Add other fields as necessary from Hydra config
        xgb_params=icd_raw.get("xgb_params", {})
    )

    # CNN Config extraction
    cnn_raw = cfg.get("cnn", {})
    cnn_data = cnn_raw.get("data", {})
    cnn_model = cnn_raw.get("model", {})
    
    cnn_cfg = CNNConfig(
        data_path=cnn_data.get("path", "data/processed/dataset_123_inferred.parquet"),
        data_cache_dir=cnn_data.get("data_cache_dir", "data/cache"),
        exclude_cols=cnn_data.get("exclude_cols", ["ICD9_CODES"]),
        seq_len=cnn_model.get("seq_len", 120),
        batch_size=cnn_model.get("batch_size", 512),
        epochs=cnn_model.get("epochs", 100),
        lr=cnn_model.get("lr", 0.001),
        patience=cnn_model.get("patience", 10),
        focal_gamma=cnn_model.get("focal_gamma", 2.0),
        label_smoothing=cnn_model.get("label_smoothing", 0.1),
        ema_decay=cnn_model.get("ema_decay", 0.999),
        embed_dim=cnn_model.get("embed_dim", 128),
        cat_embed_dim=cnn_model.get("cat_embed_dim", 32),
        dropout=cnn_model.get("dropout", 0.5),
        tcn_kernel_size=cnn_model.get("tcn_kernel_size", 5),
        tcn_layers=cnn_model.get("tcn_layers", 5),
        tcn_dropout=cnn_model.get("tcn_dropout", 0.4)
    )

    return AppContext(
        etl=etl_cfg,
        icd=icd_cfg,
        cnn=cnn_cfg,
        mode=cfg.get("mode", "train"),
        seed=cfg.get("seed", 42)
    )
