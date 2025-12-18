import polars as pl
from pathlib import Path
from typing import List
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig


class OutcomesBuilder(PipelineComponent):
    """Generate outcome flags from output events.

    Joins OUTPUTEVENTS with ICU stay boundaries to mark hours containing output activity.

    Attributes:
        cfg (ETLConfig): ETL configuration with directory paths.
        registry (Registry): Registry placeholder to align interfaces.
        observer (TelemetryObserver): Telemetry adapter for logging progress.
    """

    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        """Create intervention parquet with outcome indicators.

        Returns:
            None

        Raises:
            FileNotFoundError: If required raw files are missing.
        """
        raw_dir = Path(self.cfg.raw_dir)
        proc_dir = Path(self.cfg.proc_dir)
        proc_dir.mkdir(parents=True, exist_ok=True)

        self.observer.log("INFO", "OutcomesBuilder: Loading ICU boundaries for time alignment")

        icu = (
            pl.read_csv(raw_dir / "ICUSTAYS.csv.gz", infer_schema_length=0)
            .with_columns(
                [
                    pl.col("INTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("INTIME"),
                    pl.col("OUTTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("OUTTIME"),
                ]
            )
            .select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"])
        )

        all_flags: List[pl.DataFrame] = []
        files = list(raw_dir.glob("OUTPUTEVENTS.csv.gz"))
        if not files:
            files = list(raw_dir.glob("OUTPUTEVENTS.csv"))
            if not files:
                self.observer.log("WARNING", "OutcomesBuilder: No OUTPUTEVENTS found in raw directory")
                return

        self.observer.log("INFO", f"OutcomesBuilder: Found {len(files)} OUTPUTEVENTS files to process")

        for fname in files:
            ev = (
                pl.read_csv(fname, infer_schema_length=0)
                .with_columns([pl.col("CHARTTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("CHARTTIME")])
                .join(icu, on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"], how="left")
                .with_columns([((pl.col("CHARTTIME") - pl.col("INTIME")).dt.total_hours()).floor().cast(pl.Int32).alias("HOUR_IN")])
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
            .with_columns(
                [
                    pl.col("SUBJECT_ID").cast(pl.Int32),
                    pl.col("HADM_ID").cast(pl.Int32),
                    pl.col("ICUSTAY_ID").cast(pl.Int32),
                    pl.col("HOUR_IN").cast(pl.Int32),
                    pl.col("OUTCOME_FLAG").cast(pl.Int8),
                ]
            )
            .sort(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"])
        )

        out_path = proc_dir / self.cfg.artifacts.interventions_file
        flags_union.write_parquet(out_path)
        self.observer.log("INFO", f"OutcomesBuilder: Saved {out_path.name} rows={flags_union.height}")
