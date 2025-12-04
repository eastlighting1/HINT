from typing import List
from ...foundation.configs import HINTConfig
from ...foundation.interfaces import TelemetryObserver
from .extractor import DataExtractor
from .preprocessor import DataPreprocessor

class ETLService:
    """
    Orchestrator for the entire Data Pipeline.
    Manages both Extraction (Raw -> Parquet) and Preprocessing (Parquet -> HDF5).
    
    Args:
        config: HINT configuration.
        observer: Telemetry observer.
    """
    def __init__(self, config: HINTConfig, observer: TelemetryObserver):
        self.extractor = DataExtractor(config, observer)
        self.preprocessor = DataPreprocessor(config, observer)
        self.observer = observer

    def run_pipeline(self, steps: List[str] = ["all"]) -> None:
        """
        Run selected steps of the pipeline.

        Args:
            steps: List of steps to execute ('extract', 'preprocess', 'all').
        """
        self.observer.log("INFO", f"ETL Service: Starting pipeline with steps={steps}")

        if "all" in steps or "extract" in steps:
            with self.observer.trace("ETL_Extraction"):
                self.extractor.extract_static()
                self.extractor.extract_timeseries()
                self.extractor.build_dataset()

        if "all" in steps or "preprocess" in steps:
            with self.observer.trace("ETL_Preprocessing"):
                self.preprocessor.run_preprocessing()
        
        self.observer.log("INFO", "ETL Service: Completed.")