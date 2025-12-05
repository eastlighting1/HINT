from typing import List
from ...foundation.interfaces import TelemetryObserver, PipelineComponent
from ...domain.vo import ETLConfig, CNNConfig

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
        
        # 전체 파이프라인을 아우르는 하나의 진행바 생성
        total_steps = len(self.components)
        with self.observer.create_progress("ETL Pipeline", total=total_steps) as progress:
            # 메인 태스크 생성
            task = progress.add_task("Initializing...", total=total_steps)
            
            for i, component in enumerate(self.components, 1):
                name = component.__class__.__name__
                
                # 진행바 텍스트 업데이트 (한 줄 유지)
                progress.update(task, description=f"[{i}/{total_steps}] Running {name}")
                
                try:
                    self.observer.log("INFO", f"ETL Service: Executing component {name}...")
                    component.execute()
                    self.observer.log("INFO", f"ETL Service: Component {name} completed successfully.")
                    
                    # 진행률 업데이트
                    progress.advance(task)
                    
                except Exception as e:
                    self.observer.log("ERROR", f"ETL Service: Component {name} failed: {e}")
                    raise e
        
        self.observer.log("INFO", "ETL Service: Pipeline execution finished.")