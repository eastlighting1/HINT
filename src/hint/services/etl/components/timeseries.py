import polars as pl
from pathlib import Path
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig


class TimeSeriesAggregator(PipelineComponent):
    """Aggregate chart and lab events into hourly statistics.

    Streams raw events, aligns them to ICU admissions, and produces per-hour aggregates for downstream processing.

    Attributes:
        cfg (ETLConfig): ETL configuration with directories and artifact names.
        registry (Registry): Registry placeholder for consistency.
        observer (TelemetryObserver): Telemetry adapter for logging progress.
    """

    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        """Build hourly aggregates from chart and lab events.

        Returns:
            None

        Raises:
            FileNotFoundError: If raw inputs cannot be found.
        """
        raw_dir = Path(self.cfg.raw_dir)
        resources_dir = Path(self.cfg.resources_dir)
        proc_dir = Path(self.cfg.proc_dir)
        proc_dir.mkdir(parents=True, exist_ok=True)

        self.observer.log("INFO", f"TimeSeriesAggregator: Scanning raw directory {raw_dir}")

        def process_events(table: str, time_col: str) -> pl.LazyFrame:
            fpath = raw_dir / f"{table.upper()}.csv.gz"
            if not fpath.exists():
                fpath = raw_dir / f"{table.upper()}.csv"
            self.observer.log("INFO", f"TimeSeriesAggregator: Loading {fpath.name} into lazy frame")

            def to_datetime_iso(col: str) -> pl.Expr:
                base = pl.col(col).str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
                return pl.when(base.is_null() & pl.col(col).is_not_null()).then(
                    pl.col(col).str.replace(r"Z$", "+00:00", literal=False).str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
                ).otherwise(base)

            def duration_hours(start: str, end: str) -> pl.Expr:
                return (pl.col(end).dt.epoch(time_unit="us") - pl.col(start).dt.epoch(time_unit="us")) / 3_600_000_000

            def floor_int(expr: pl.Expr, dtype=pl.Int32) -> pl.Expr:
                return expr.floor().cast(dtype)

            ev = pl.scan_csv(fpath, infer_schema_length=0, has_header=True).with_columns(
                [
                    pl.col("ITEMID").cast(pl.Int64),
                    pl.col("SUBJECT_ID").cast(pl.Int64),
                    pl.col("HADM_ID").cast(pl.Int64),
                    to_datetime_iso(time_col.upper()).alias("EVENT_TS"),
                ]
            )

            varmap = (
                pl.read_csv(resources_dir / "itemid_to_variable_map.csv")
                .with_columns(pl.col("ITEMID").cast(pl.Int64))
                .select(["ITEMID", pl.col("MIMIC LABEL").alias("LABEL")])
                .lazy()
            )

            icu_map = (
                pl.scan_csv(raw_dir / "ICUSTAYS.csv.gz", infer_schema_length=0, has_header=True)
                .with_columns(
                    [
                        pl.col("SUBJECT_ID").cast(pl.Int64),
                        pl.col("HADM_ID").cast(pl.Int64),
                        pl.col("ICUSTAY_ID").cast(pl.Int64),
                        to_datetime_iso("INTIME").alias("INTIME_DT"),
                    ]
                )
                .select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME_DT"])
                .unique(subset=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"])
            )

            return (
                ev.join(varmap, on="ITEMID", how="inner")
                .join(icu_map, on=["SUBJECT_ID", "HADM_ID"], how="left")
                .with_columns(
                    [
                        duration_hours("INTIME_DT", "EVENT_TS").alias("DURATION_HOURS"),
                        floor_int(duration_hours("INTIME_DT", "EVENT_TS")).alias("HOURS_IN"),
                        pl.col("ICUSTAY_ID").cast(pl.Int64),
                        pl.col("VALUENUM").cast(pl.Float64),
                    ]
                )
                .select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOURS_IN", "LABEL", "VALUENUM"])
            )

        char_lf = process_events("chartevents", "charttime")
        lab_lf = process_events("labevents", "charttime")
        full_lf = pl.concat([char_lf, lab_lf])
        self.observer.log("INFO", "TimeSeriesAggregator: Concatenated chart and lab lazy frames")

        vitals_labs = (
            full_lf.group_by(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOURS_IN", "LABEL"])
            .agg(
                [
                    pl.col("VALUENUM").mean().alias("MEAN"),
                    pl.col("VALUENUM").count().alias("COUNT"),
                    pl.col("VALUENUM").std().alias("STDDEV"),
                ]
            )
            .collect()
        )

        self.observer.log("INFO", f"TimeSeriesAggregator: Aggregation complete rows={vitals_labs.height}")

        vitals_path = proc_dir / self.cfg.artifacts.vitals_file
        vitals_labs.write_parquet(vitals_path)
        self.observer.log("INFO", f"TimeSeriesAggregator: Saved {vitals_path.name}")

        vitals_mean = vitals_labs.select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOURS_IN", "LABEL", "MEAN"])
        vitals_mean_path = proc_dir / self.cfg.artifacts.vitals_mean_file
        vitals_mean.write_parquet(vitals_mean_path)
        self.observer.log("INFO", f"TimeSeriesAggregator: Saved {vitals_mean_path.name}")
