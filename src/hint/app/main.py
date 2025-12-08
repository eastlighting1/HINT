import hydra
from omegaconf import DictConfig
from pathlib import Path
from hint.app.factory import AppFactory

BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = BASE_DIR / "configs"

@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for HINT pipeline.
    Orchestrates ETL, ICD, and CNN workflows based on configuration mode.
    """
    factory = AppFactory(cfg)
    mode = cfg.get("mode", "train")
    
    factory.observer.log("INFO", f"Main: Starting HINT pipeline in mode='{mode}'.")

    if mode in ["all", "etl"]:
        factory.observer.log("INFO", "Main: Launching ETL pipeline stage.")
        etl_service = factory.create_etl_service()
        etl_service.run_pipeline()
        factory.observer.log("INFO", "Main: ETL pipeline stage completed.")

    if mode in ["all", "train", "icd"]:
        factory.observer.log("INFO", "Main: Launching ICD training stage.")
        icd_service = factory.create_icd_service()
        icd_service.train()
        icd_service.run_xai()
        factory.observer.log("INFO", "Main: ICD training stage completed.")

    if mode in ["all", "train", "cnn"]:
        factory.observer.log("INFO", "Main: Launching CNN training stage.")
        trainer, evaluator = factory.create_cnn_services()
        trainer.train_model(trainer.tr_src, trainer.val_src)
        evaluator.calibrate(trainer.val_src)
        evaluator.evaluate(evaluator.te_src)
        evaluator.run_xai(evaluator.val_src, evaluator.te_src)
        factory.observer.log("INFO", "Main: CNN training stage completed.")

    factory.observer.log("INFO", "Main: Pipeline finished successfully.")

if __name__ == "__main__":
    main()
