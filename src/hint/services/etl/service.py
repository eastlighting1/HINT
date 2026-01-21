"""Summary of the service module.

Longer description of the module purpose and usage.
"""

from typing import List
import time

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

        self.observer.log("INFO", "[STAGE 1 START] ETL pipeline execution started.")

        total_steps = len(self.components)
        self.observer.log("INFO", f"[1.0] Pipeline components ready. count={total_steps}")

        use_live_view = hasattr(self.observer, "start_progress_view")

        if use_live_view:
            self.observer.start_progress_view("ETL Pipeline", total_steps)

        try:
            progress = None
            task = None
            if not use_live_view:
                progress = self.observer.create_progress("ETL Pipeline", total=total_steps)
                progress.__enter__()
                task = progress.add_task("[bold green]Overall Pipeline", total=total_steps)

            for i, component in enumerate(self.components, 1):

                name = component.__class__.__name__

                if progress and task is not None:
                    progress.update(task, description=f"[bold green]Overall [{i}/{total_steps}]: {name}")
                if use_live_view:
                    self.observer.progress_update(f"[{i}/{total_steps}] {name}")

                try:
                    component_start = time.time()
                    if hasattr(self.observer, "trace_event"):
                        self.observer.trace_event(
                            "etl_component_start",
                            {"component": name, "index": i, "total": total_steps},
                        )

                    self.observer.log(
                        "INFO",
                        f"[1.{i}] Component start. index={i} total={total_steps} name={name}",
                    )

                    component.execute()

                    self.observer.log(
                        "INFO",
                        f"[1.{i}] Component complete. index={i} total={total_steps} name={name}",
                    )

                    duration = time.time() - component_start
                    if hasattr(self.observer, "trace_event"):
                        self.observer.trace_event(
                            "etl_component_end",
                            {"component": name, "index": i, "duration_sec": duration},
                        )
                    if hasattr(self.observer, "trace_bottleneck"):
                        self.observer.trace_bottleneck(f"etl_component:{name}", duration, threshold_sec=120.0)
                    if use_live_view:
                        self.observer.progress_footer(f"Completed {name} in {duration:.1f}s")

                    if progress and task is not None:
                        progress.advance(task)
                    if use_live_view:
                        self.observer.progress_advance()

                except Exception as e:

                    self.observer.log(
                        "ERROR",
                        f"[1.{i}] Component failed. index={i} total={total_steps} name={name} error={e}",
                    )

                    raise e
        finally:
            if not use_live_view:
                try:
                    if progress:
                        progress.__exit__(None, None, None)
                except Exception:
                    pass
            if use_live_view:
                self.observer.stop_progress_view()



        self.observer.log("INFO", "[STAGE 1 END] ETL pipeline execution finished.")
