from typing import List
from ...foundation.interfaces import TelemetryObserver, PipelineComponent, Registry

class ETLService:
    """Orchestrate ETL pipeline component execution.

    This service runs each ETL component in sequence and logs progress.

    Attributes:
        registry (Registry): Artifact registry for outputs.
        observer (TelemetryObserver): Logging and metric observer.
        components (List[PipelineComponent]): Ordered pipeline steps.
    """
    def __init__(
        self, 
        registry: Registry,
        observer: TelemetryObserver,
        components: List[PipelineComponent]
    ):
        """Initialize the ETL service.

        Args:
            registry (Registry): Artifact registry for outputs.
            observer (TelemetryObserver): Logging and metric observer.
            components (List[PipelineComponent]): Pipeline components in order.
        """
        self.registry = registry
        self.observer = observer
        self.components = components

    def run_pipeline(self) -> None:
        """Run all configured ETL components in order."""
        self.observer.log("INFO", "ETL Service: Starting pipeline execution.")
        
        total_steps = len(self.components)
        with self.observer.create_progress("ETL Pipeline", total=total_steps) as progress:
            task = progress.add_task("Initializing...", total=total_steps)
            
            for i, component in enumerate(self.components, 1):
                name = component.__class__.__name__
                
                progress.update(task, description=f"[{i}/{total_steps}] Running {name}")
                
                try:
                    self.observer.log("INFO", f"ETL Service: Step {i}/{total_steps} start component={name}.")
                    component.execute()
                    self.observer.log("INFO", f"ETL Service: Step {i}/{total_steps} complete component={name}.")

                    progress.advance(task)
                    
                except Exception as e:
                    self.observer.log("ERROR", f"ETL Service: Step {i}/{total_steps} failed component={name} error={e}.")
                    raise e
        
        self.observer.log("INFO", "ETL Service: Pipeline execution finished.")
