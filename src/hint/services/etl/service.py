"""Summary of the service module.

Longer description of the module purpose and usage.
"""

from typing import List

from ...foundation.interfaces import PipelineComponent, Registry, TelemetryObserver



class ETLService:

    """Summary of ETLService purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    components (Any): Description of components.
    observer (Any): Description of observer.
    registry (Any): Description of registry.
    """

    def __init__(

        self,

        registry: Registry,

        observer: TelemetryObserver,

        components: List[PipelineComponent]

    ):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        registry (Any): Description of registry.
        observer (Any): Description of observer.
        components (Any): Description of components.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.registry = registry

        self.observer = observer

        self.components = components



    def run_pipeline(self) -> None:

        """Summary of run_pipeline.
        
        Longer description of the run_pipeline behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.observer.log("INFO", "ETL Service: Stage 1/3 starting pipeline execution.")



        total_steps = len(self.components)
        self.observer.log("INFO", f"ETL Service: Stage 2/3 component_count={total_steps}.")

        with self.observer.create_progress("ETL Pipeline", total=total_steps) as progress:

            task = progress.add_task("[bold green]Overall Pipeline", total=total_steps)



            for i, component in enumerate(self.components, 1):

                name = component.__class__.__name__



                progress.update(task, description=f"[bold green]Overall [{i}/{total_steps}]: {name}")



                try:

                    self.observer.log("INFO", f"ETL Service: Step {i}/{total_steps} start component={name}.")

                    component.execute()

                    self.observer.log("INFO", f"ETL Service: Step {i}/{total_steps} complete component={name}.")



                    progress.advance(task)



                except Exception as e:

                    self.observer.log("ERROR", f"ETL Service: Step {i}/{total_steps} failed component={name} error={e}.")

                    raise e



        self.observer.log("INFO", "ETL Service: Stage 3/3 pipeline execution finished.")
