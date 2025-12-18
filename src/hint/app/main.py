import sys
from hint.app.factory import AppFactory

def main() -> None:
    factory = AppFactory()
    mode = factory.ctx.mode
    logged_start = False

    # 1. ETL Mode
    if mode in ["all", "etl"]:
        etl_service = factory.create_etl_service()
        if not logged_start:
            etl_service.observer.log("INFO", f"App: Starting application in [{mode}] mode.")
            logged_start = True
        etl_service.observer.log("INFO", "App: Running ETL pipeline.")
        etl_service.run_pipeline()

    # 2. ICD Mode (Training + Augmentation)
    if mode in ["all", "icd"]:
        icd_service = factory.create_icd_service()
        if not logged_start:
            icd_service.observer.log("INFO", f"App: Starting application in [{mode}] mode.")
            logged_start = True
        
        # Train ICD Model
        icd_service.observer.log("INFO", "App: Training ICD model.")
        icd_service.execute() 
        
        # Generate Data for Intervention
        icd_service.observer.log("INFO", "App: Generating intervention dataset with ICD inference.")
        icd_service.generate_intervention_dataset(factory.ctx.cnn)

    # 3. Intervention Mode
    if mode in ["all", "intervention"]:
        int_service = factory.create_intervention_service()
        if not logged_start:
            if int_service.observer:
                int_service.observer.log("INFO", f"App: Starting application in [{mode}] mode.")
            logged_start = True
            
        if int_service.train_ds is not None:
            int_service.observer.log("INFO", "App: Running Intervention Prediction training.")
            int_service.execute()
        else:
            int_service.observer.log("ERROR", "App: Intervention training skipped (no data).")

if __name__ == "__main__":
    main()