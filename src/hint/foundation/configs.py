"""Summary of the configs module.

Longer description of the module purpose and usage.
"""

from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from hint.foundation.dtos import AppContext
from hint.domain.vo import ETLConfig, ICDConfig, InterventionConfig



class HydraConfigLoader:

    """Summary of HydraConfigLoader purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        config_name (Any): Description of config_name.
        config_path (Any): Description of config_path.
    """

    def __init__(self, config_name: str = "config", config_path: str = "configs"):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            config_name (Any): Description of config_name.
            config_path (Any): Description of config_path.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        self.config_name = config_name

        self.config_path = config_path



    def load(self) -> DictConfig:

        """Summary of load.
        
        Longer description of the load behavior and usage.
        
        Args:
            None (None): This function does not accept arguments.
        
        Returns:
            DictConfig: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
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

    """Summary of load_app_context.
    
    Longer description of the load_app_context behavior and usage.
    
    Args:
        cfg (Any): Description of cfg.
    
    Returns:
        AppContext: Description of the return value.
    
    Raises:
        Exception: Description of why this exception might be raised.
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

    intervention_raw = OmegaConf.to_container(cfg.get("intervention", {}), resolve=True) or {}

    intervention_cfg = InterventionConfig(**intervention_raw)

    return AppContext(

        etl=etl_cfg,

        icd=icd_cfg,

        intervention=intervention_cfg,

        mode=cfg.get("mode", "train"),

        seed=cfg.get("seed", 42),

    )
