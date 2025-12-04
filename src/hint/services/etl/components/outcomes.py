import polars as pl
from pathlib import Path
from typing import List

from hint.foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from hint.domain.vo import ETLConfig

class OutcomesBuilder(PipelineComponent):
    """
    Builds hourly outcome flags from OUTPUTEVENTS.
    Ported from make_data.py: step_build_outcomes
    """
    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        raw_dir = Path(self.cfg.raw_dir)
        self.observer.log("INFO", "OutcomesBuilder: Deriving OUTCOME_FLAG from OUTPUTEVENTS...")

        icu = (
            pl.read_csv(str(raw_dir / "ICUSTAYS.csv.gz"), infer_schema_length=0)
            .with_columns([
                pl.col("INTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("INTIME"),
                pl.col("OUTTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("OUTTIME"),
            ])
            .select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"])
        )

        all_flags: List[pl.DataFrame] = []
        # Support multiple chunks if present
        files = list(raw_dir.glob("OUTPUTEVENTS.csv.gz"))
        if not files:
             self.observer.log("WARNING", "OutcomesBuilder: No OUTPUTEVENTS found.")
             return

        for fname in files:
            ev = (
                pl.read_csv(str(fname), infer_schema_length=0)
                .with_columns([
                    pl.col("CHARTTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("CHARTTIME")
                ])
                .join(icu, on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"], how="left")
                .with_columns([
                    ((pl.col("CHARTTIME") - pl.col("INTIME")).dt.total_hours())
                    .floor().cast(pl.Int32).alias("HOUR_IN")
                ])
                .filter((pl.col("HOUR_IN") >= 0) & (pl.col("CHARTTIME") <= pl.col("OUTTIME")))
                .select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"])
                .unique()
                .with_columns(pl.lit(1).alias("OUTCOME_FLAG"))
            )
            all_flags.append(ev)

        if not all_flags:
            return

        flags_union = (
            pl.concat(all_flags)
            .with_columns([
                pl.col("SUBJECT_ID").cast(pl.Int32),
                pl.col("HADM_ID").cast(pl.Int32),
                pl.col("ICUSTAY_ID").cast(pl.Int32),
                pl.col("HOUR_IN").cast(pl.Int32),
                pl.col("OUTCOME_FLAG").cast(pl.Int8),
            ])
            .sort(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"])
        )

        self.registry.save_dataframe(flags_union, "interventions.parquet")
        self.observer.log("INFO", f"OutcomesBuilder: Saved interventions.parquet (rows={flags_union.height})")
