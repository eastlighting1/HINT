from omegaconf import DictConfig, OmegaConf
from hint.foundation.dtos import AppContext
from hint.domain.vo import ETLConfig, ICDConfig, CNNConfig

def load_app_context(cfg: DictConfig) -> AppContext:
    """
    Build the strongly typed application context from the Hydra configuration.

    Args:
        cfg: Hydra DictConfig assembled from the YAML configuration files.

    Returns:
        Application context populated with ETL, ICD, and CNN settings.
    """
    etl_cfg = ETLConfig(
        raw_dir=cfg.get("data", {}).get("raw_dir", "./data/raw"),
        proc_dir=cfg.get("data", {}).get("proc_dir", "./data/processed"),
        resources_dir=cfg.get("data", {}).get("resources_dir", "./resources"),
    )

    icd_raw = OmegaConf.to_container(cfg.get("icd", {}), resolve=True)
    icd_cfg = ICDConfig(**icd_raw)

    cnn_raw = cfg.get("cnn", {})
    cnn_data = cnn_raw.get("data", {})
    cnn_model = cnn_raw.get("model", {})

    cnn_cfg = CNNConfig(
        data_path=cnn_data.get("path", "data/processed/dataset_123_inferred.parquet"),
        data_cache_dir=cnn_data.get("data_cache_dir", "data/cache"),
        exclude_cols=cnn_data.get("exclude_cols", ["ICD9_CODES"]),
        **cnn_model,
    )

    return AppContext(
        etl=etl_cfg,
        icd=icd_cfg,
        cnn=cnn_cfg,
        mode=cfg.get("mode", "train"),
        seed=cfg.get("seed", 42),
    )
