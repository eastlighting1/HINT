import polars as pl
import numpy as np
from pathlib import Path
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig

class OutcomesBuilder(PipelineComponent):
    """Generate the dense intervention skeleton (time grid) for each patient.
    
    Instead of relying on sparse OUTPUTEVENTS, this builds a continuous hourly grid 
    (0 to STAY_HOURS) for every patient in the cohort, initializing outcome flags to 0.
    """

    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        proc_dir = Path(self.cfg.proc_dir)
        patients_path = proc_dir / self.cfg.artifacts.patients_file

        if not patients_path.exists():
            raise FileNotFoundError(f"Dependency missing: {patients_path}. Run StaticExtractor first.")

        self.observer.log("INFO", "OutcomesBuilder: Stage 1/4 loading cohort info")
        cohort = pl.read_parquet(patients_path).select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STAY_HOURS"])

        self.observer.log("INFO", "OutcomesBuilder: Stage 2/4 building dense hourly grid")
        ids_df = cohort.to_pandas()

        stays = ids_df["STAY_HOURS"].values.astype(int)

        repeats = stays + 1
        total_rows = repeats.sum()

        self.observer.log("INFO", f"OutcomesBuilder: Creating {total_rows} hourly rows for {len(ids_df)} patients")

        subject_ids = np.repeat(ids_df["SUBJECT_ID"].values, repeats)
        hadm_ids = np.repeat(ids_df["HADM_ID"].values, repeats)
        icustay_ids = np.repeat(ids_df["ICUSTAY_ID"].values, repeats)

        hours_in = np.concatenate([np.arange(n) for n in repeats])

        skeleton = pl.DataFrame({
            "SUBJECT_ID": subject_ids,
            "HADM_ID": hadm_ids,
            "ICUSTAY_ID": icustay_ids,
            "HOUR_IN": hours_in.astype(np.int32)
        })

        self.observer.log("INFO", "OutcomesBuilder: Stage 3/4 initializing outcome flags")
        final_df = skeleton.with_columns([
            pl.lit(0).cast(pl.Int8).alias("OUTCOME_FLAG"), 
            pl.lit(0).cast(pl.Int8).alias("VENT"),
            pl.lit(0).cast(pl.Int8).alias("VASO") 
        ])

        self.observer.log("INFO", "OutcomesBuilder: Stage 4/4 saving outcome skeleton")
        out_path = proc_dir / self.cfg.artifacts.interventions_file
        final_df.write_parquet(out_path)
        self.observer.log("INFO", f"OutcomesBuilder: Saved dense skeleton to {out_path}")
