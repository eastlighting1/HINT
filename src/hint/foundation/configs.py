from pathlib import Path
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from hint.foundation.dtos import AppContext
from hint.domain.vo import ETLConfig, ICDConfig, CNNConfig

class HydraConfigLoader:
    """
    Wrapper around Hydra to load configurations programmatically.
    """
    def __init__(self, config_name: str = "config", config_path: str = "configs"):
        self.config_name = config_name
        self.config_path = config_path

    def load(self) -> DictConfig:
        GlobalHydra.instance().clear()
        
        cwd = Path.cwd()
        candidate_path = cwd / self.config_path
        
        if not candidate_path.exists():
            current_file = Path(__file__).resolve()
            candidate_path = (current_file.parents[3] / self.config_path).resolve()
            
        if not candidate_path.exists():
             raise FileNotFoundError(f"Configuration directory not found at {candidate_path}")

        with hydra.initialize_config_dir(version_base=None, config_dir=str(candidate_path)):
            cfg = hydra.compose(config_name=self.config_name)
            OmegaConf.resolve(cfg)
            
        return cfg

def load_app_context(cfg: DictConfig) -> AppContext:
    """
    Build the strongly typed application context from the Hydra configuration.
    """
    etl_data = cfg.get("etl", {}).get("data", {})
    etl_cohort = cfg.get("etl", {}).get("cohort", {})
    etl_proc = cfg.get("etl", {}).get("processing", {})
    
    etl_cfg = ETLConfig(
        raw_dir=str(etl_data.get("raw_dir", "./data/raw")),
        proc_dir=str(etl_data.get("proc_dir", "./data/processed")),
        resources_dir=str(etl_data.get("resources_dir", "./resources")),
        **etl_cohort,
        **etl_proc
    )

    icd_raw = OmegaConf.to_container(cfg.get("icd", {}), resolve=True)
    icd_cfg = ICDConfig(**icd_raw)

    cnn_raw = cfg.get("cnn", {})
    cnn_data = cnn_raw.get("data", {})
    cnn_model = cnn_raw.get("model", {})

    cnn_cfg = CNNConfig(
        feature_path=str(cnn_data.get("feature_path", "data/cache/train.h5")),
        label_path=str(cnn_data.get("label_path", "data/processed/labels.parquet")),
        data_cache_dir=str(cnn_data.get("data_cache_dir", "data/cache")),
        exclude_cols=list(cnn_data.get("exclude_cols", ["ICD9_CODES"])),
        **cnn_model,
    )

    return AppContext(
        etl=etl_cfg,
        icd=icd_cfg,
        cnn=cnn_cfg,
        mode=cfg.get("mode", "train"),
        seed=cfg.get("seed", 42),
    )
