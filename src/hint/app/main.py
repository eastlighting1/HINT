import sys
from hint.app.factory import AppFactory

def main() -> None:
    factory = AppFactory()
    mode = factory.ctx.mode
    logged_start = False

    if mode in ["all", "etl"]:
        etl_service = factory.create_etl_service()
        if not logged_start:
            etl_service.observer.log("INFO", f"App: Starting application in [{mode}] mode.")
            logged_start = True
        etl_service.observer.log("INFO", "App: Running ETL pipeline.")
        etl_service.run_pipeline()

    if mode in ["all", "icd"]:
        icd_service = factory.create_icd_service()
        if not logged_start:
            icd_service.observer.log("INFO", f"App: Starting application in [{mode}] mode.")
            logged_start = True
        icd_service.observer.log("INFO", "App: Training ICD model.")
        icd_service.train()
        icd_service.observer.log("INFO", "App: Generating intervention dataset with ICD inference.")
        icd_service.generate_intervention_dataset(factory.ctx.cnn)

    if mode in ["all", "cnn", "train"]:
        cnn_service = factory.create_cnn_service()
        if not logged_start and cnn_service.observer:
            cnn_service.observer.log("INFO", f"App: Starting application in [{mode}] mode.")
            logged_start = True
        if cnn_service.train_dataset is not None:
            cnn_service.observer.log("INFO", "App: Running CNN training service.")
            cnn_service.train_model()
        else:
            cnn_service.observer.log("ERROR", "App: CNN training skipped due to missing data sources.")

if __name__ == "__main__":
    main()
