import sys
from hint.app.factory import AppFactory

def main() -> None:
    """Run the application based on the configured mode."""
    factory = AppFactory()
    mode = factory.ctx.mode
    logged_start = False

    if mode in ["all", "etl"]:
        etl_service = factory.create_etl_service()
        if not logged_start:
            etl_service.observer.log("INFO", f"App: Starting application in [{mode}] mode.")
            logged_start = True
        etl_service.observer.log("INFO", "App: Entering ETL stage.")
        etl_service.run_pipeline()

    if mode in ["all", "icd"]:
        icd_service = factory.create_icd_service()
        if not logged_start:
            icd_service.observer.log("INFO", f"App: Starting application in [{mode}] mode.")
            logged_start = True
        
        icd_service.observer.log("INFO", "App: Entering ICD training stage.")
        icd_service.execute() 
        
        icd_service.observer.log("INFO", "App: Entering ICD inference stage for intervention data generation.")
        icd_service.generate_intervention_dataset(factory.ctx.cnn)

    if mode in ["all", "intervention"]:
        int_service = factory.create_intervention_service()
        if not logged_start:
            if int_service.observer:
                int_service.observer.log("INFO", f"App: Starting application in [{mode}] mode.")
            logged_start = True
            
        if int_service.train_ds is not None:
            int_service.observer.log("INFO", "App: Entering intervention training stage.")
            int_service.execute()
        else:
            int_service.observer.log("ERROR", "App: Intervention training skipped (no data).")

if __name__ == "__main__":
    main()
