from typing import List
from hint.foundation.interfaces import TelemetryObserver, PipelineComponent
from hint.domain.vo import ETLConfig, CNNConfig

class ETLService:
    """
    Orchestrator for the data processing pipeline.
    Executes registered PipelineComponents in order.
    """
    def __init__(
        self, 
        etl_config: ETLConfig,
        cnn_config: CNNConfig,
        components: List[PipelineComponent],
        observer: TelemetryObserver
    ):
        self.etl_config = etl_config
        self.cnn_config = cnn_config
        self.components = components
        self.observer = observer

    def run_pipeline(self) -> None:
        """Execute all configured pipeline steps sequentially."""
        self.observer.log("INFO", "ETL Service: Starting data pipeline execution.")
        
        for component in self.components:
            name = component.__class__.__name__
            with self.observer.create_progress(f"Running {name}", total=1) as progress:
                task = progress.add_task(f"Running {name}", total=None)
                try:
                    self.observer.log("INFO", f"ETL Service: Executing component {name}...")
                    component.execute()
                    progress.update(task, completed=1)
                    self.observer.log("INFO", f"ETL Service: Component {name} completed successfully.")
                except Exception as e:
                    self.observer.log("ERROR", f"ETL Service: Component {name} failed: {e}")
                    raise e
        
        self.observer.log("INFO", "ETL Service: Pipeline execution finished.")
