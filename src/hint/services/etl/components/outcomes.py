import polars as pl
import numpy as np
from pathlib import Path
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig

class OutcomesBuilder(PipelineComponent):
    """Build the event-driven outcomes skeleton based on actual observations.

    This component creates a time grid driven by observed vitals to align
    subsequent outcome labels with real measurement timestamps.

    Attributes:
        cfg (ETLConfig): ETL configuration.
        registry (Registry): Artifact registry.
        observer (TelemetryObserver): Logging observer.
    """

    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        """Initialize the outcomes builder.

        Args:
            config (ETLConfig): ETL configuration.
            registry (Registry): Artifact registry.
            observer (TelemetryObserver): Logging observer.
        """
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        """Generate and persist the event-driven outcomes skeleton.

        Raises:
            FileNotFoundError: If required upstream artifacts are missing.
        """
        proc_dir = Path(self.cfg.proc_dir)
        patients_path = proc_dir / self.cfg.artifacts.patients_file
        vitals_path = proc_dir / self.cfg.artifacts.vitals_mean_file

        if not patients_path.exists():
            raise FileNotFoundError(f"Dependency missing: {patients_path}. Run StaticExtractor first.")
        if not vitals_path.exists():
            raise FileNotFoundError(f"Dependency missing: {vitals_path}. Run TimeSeriesAggregator first.")

        self.observer.log("INFO", "OutcomesBuilder: Stage 1/4 loading cohort and vital timestamps")
        patients = pl.read_parquet(patients_path).select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"])
        
        vitals = pl.read_parquet(vitals_path).select(["ICUSTAY_ID", "HOURS_IN"]).unique()

        self.observer.log("INFO", "OutcomesBuilder: Stage 2/4 building event-driven grid")
        
        skeleton = (
            vitals.join(patients, on="ICUSTAY_ID", how="inner")
            .rename({"HOURS_IN": "HOUR_IN"})
            .sort(["ICUSTAY_ID", "HOUR_IN"])
        )
        
        self.observer.log("INFO", f"OutcomesBuilder: Created {skeleton.height} event-driven rows")

        self.observer.log("INFO", "OutcomesBuilder: Stage 3/4 initializing outcome flags")
        final_df = skeleton.with_columns([
            pl.lit(0).cast(pl.Int8).alias("OUTCOME_FLAG"), 
            pl.lit(0).cast(pl.Int8).alias("VENT"),
            pl.lit(0).cast(pl.Int8).alias("VASO") 
        ])

        self.observer.log("INFO", "OutcomesBuilder: Stage 4/4 saving outcome skeleton")
        out_path = proc_dir / self.cfg.artifacts.interventions_file
        final_df.write_parquet(out_path)
        self.observer.log("INFO", f"OutcomesBuilder: Saved event skeleton to {out_path}")
