import hydra
from omegaconf import DictConfig
from hint.app.factory import AppFactory

@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for HINT pipeline.
    Orchestrates ETL, ICD, and CNN workflows based on configuration mode.
    """
    factory = AppFactory(cfg)
    mode = cfg.get("mode", "train")
    
    factory.observer.log("INFO", f"Main: Starting HINT pipeline in mode='{mode}'")

    # 1. ETL Pipeline
    if mode in ["all", "etl"]:
        etl_service = factory.create_etl_service()
        etl_service.run_pipeline()

    # 2. ICD Pipeline
    if mode in ["all", "train", "icd"]:
        icd_service = factory.create_icd_service()
        icd_service.train()
        icd_service.run_xai()

    # 3. CNN Pipeline
    if mode in ["all", "train", "cnn"]:
        trainer, evaluator = factory.create_cnn_services()
        
        # Train
        trainer.train_model(trainer.tr_src, trainer.val_src)
        
        # Calibrate & Test
        evaluator.calibrate(trainer.val_src)
        evaluator.evaluate(evaluator.te_src)
        evaluator.run_xai(evaluator.te_src)

    factory.observer.log("INFO", "Main: Pipeline finished successfully.")

if __name__ == "__main__":
    main()
