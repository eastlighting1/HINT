from typing import List
from ...foundation.interfaces import TelemetryObserver, PipelineComponent, Registry

class ETLService:
    """
    Orchestrator for the data processing pipeline.
    Executes registered PipelineComponents in order.
    """
    def __init__(
        self, 
        registry: Registry,
        observer: TelemetryObserver,
        components: List[PipelineComponent]
    ):
        self.registry = registry
        self.observer = observer
        self.components = components

    def run_pipeline(self) -> None:
        """Execute all configured pipeline steps sequentially."""
        self.observer.log("INFO", "ETL Service: Starting data pipeline execution.")
        
        total_steps = len(self.components)
        with self.observer.create_progress("ETL Pipeline", total=total_steps) as progress:
            task = progress.add_task("Initializing...", total=total_steps)
            
            for i, component in enumerate(self.components, 1):
                name = component.__class__.__name__
                
                progress.update(task, description=f"[{i}/{total_steps}] Running {name}")
                
                try:
                    self.observer.log("INFO", f"ETL Service: Executing component {name}...")
                    component.execute()
                    self.observer.log("INFO", f"ETL Service: Component {name} completed successfully.")

                    progress.advance(task)
                    
                except Exception as e:
                    self.observer.log("ERROR", f"ETL Service: Component {name} failed: {e}")
                    raise e
        
        self.observer.log("INFO", "ETL Service: Pipeline execution finished.")