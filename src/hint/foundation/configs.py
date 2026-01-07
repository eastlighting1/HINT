from pathlib import Path
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from hint.foundation.dtos import AppContext
from hint.domain.vo import ETLConfig, ICDConfig, CNNConfig

class HydraConfigLoader:
    """Load Hydra configuration files for the application.

    This helper resolves the configuration directory and composes the
    requested Hydra configuration.

    Attributes:
        config_name (str): Base name of the Hydra configuration.
        config_path (str): Path to the configuration directory.
    """
    def __init__(self, config_name: str = "config", config_path: str = "configs"):
        """Initialize the config loader with name and path settings.

        Args:
            config_name (str): Hydra config name to compose.
            config_path (str): Directory containing Hydra configs.
        """
        self.config_name = config_name
        self.config_path = config_path

    def load(self) -> DictConfig:
        """Compose and resolve the Hydra configuration.

        Returns:
            DictConfig: Fully resolved Hydra configuration object.

        Raises:
            FileNotFoundError: If the configuration directory is missing.
        """
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
    """Build the AppContext from the Hydra configuration.

    This function converts nested Hydra config sections into strongly
    typed configuration objects.

    Args:
        cfg (DictConfig): Hydra configuration tree.

    Returns:
        AppContext: Structured runtime configuration context.
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
