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
    etl_raw = OmegaConf.to_container(cfg.get("etl", {}), resolve=True) or {}
    etl_cfg = ETLConfig(
        **etl_raw.get("data", {}),
        **etl_raw.get("cohort", {}),
        **etl_raw.get("processing", {}),
        artifacts=etl_raw.get("artifacts", {}),
    )

    icd_raw = OmegaConf.to_container(cfg.get("icd", {}), resolve=True) or {}
    icd_cfg = ICDConfig(**icd_raw)

    cnn_raw = OmegaConf.to_container(cfg.get("cnn", {}), resolve=True) or {}
    cnn_cfg = CNNConfig(**cnn_raw)

    return AppContext(
        etl=etl_cfg,
        icd=icd_cfg,
        cnn=cnn_cfg,
        mode=cfg.get("mode", "train"),
        seed=cfg.get("seed", 42),
    )
