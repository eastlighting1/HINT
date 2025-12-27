import polars as pl
from pathlib import Path
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig

class TimeSeriesAggregator(PipelineComponent):
    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        raw_dir = Path(self.cfg.raw_dir)
        resources_dir = Path(self.cfg.resources_dir)
        proc_dir = Path(self.cfg.proc_dir)
        proc_dir.mkdir(parents=True, exist_ok=True)

        self.observer.log("INFO", f"TimeSeriesAggregator: Scanning raw directory {raw_dir}")

        def process_events(table: str, time_col: str) -> pl.LazyFrame:
            fpath = raw_dir / f"{table.upper()}.csv.gz"
            if not fpath.exists(): fpath = raw_dir / f"{table.upper()}.csv"
            
            def to_datetime_iso(col: str) -> pl.Expr:
                base = pl.col(col).str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
                return pl.when(base.is_null() & pl.col(col).is_not_null()).then(
                    pl.col(col).str.replace(r"Z$", "+00:00", literal=False).str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
                ).otherwise(base)

            ev = pl.scan_csv(fpath, infer_schema_length=0, has_header=True).with_columns([
                pl.col("ITEMID").cast(pl.Int64),
                pl.col("SUBJECT_ID").cast(pl.Int64),
                pl.col("HADM_ID").cast(pl.Int64),
                to_datetime_iso(time_col.upper()).alias("EVENT_TS"),
            ])

            varmap = pl.read_csv(resources_dir / "itemid_to_variable_map.csv").with_columns(pl.col("ITEMID").cast(pl.Int64)).select(["ITEMID", pl.col("MIMIC LABEL").alias("LABEL")]).lazy()
            icu_map = pl.scan_csv(raw_dir / "ICUSTAYS.csv.gz", infer_schema_length=0, has_header=True).with_columns([
                pl.col("ICUSTAY_ID").cast(pl.Int64),
                to_datetime_iso("INTIME").alias("INTIME_DT"),
            ]).select(["ICUSTAY_ID", "INTIME_DT"]).unique(subset=["ICUSTAY_ID"])

            return ev.join(varmap, on="ITEMID", how="inner").join(icu_map, on="ICUSTAY_ID", how="left").with_columns([
                ((pl.col("EVENT_TS").dt.epoch("us") - pl.col("INTIME_DT").dt.epoch("us")) / 3.6e9).floor().cast(pl.Int32).alias("HOURS_IN"),
                pl.col("VALUENUM").cast(pl.Float64)
            ]).select(["ICUSTAY_ID", "HOURS_IN", "LABEL", "VALUENUM"])

        full_lf = pl.concat([process_events("chartevents", "charttime"), process_events("labevents", "charttime")])
        
        # [Optimization] Outlier Filtering
        clean_lf = full_lf.filter(
            pl.col("VALUENUM").is_not_null() & pl.col("VALUENUM").is_finite() & (pl.col("VALUENUM") >= 0) & (pl.col("VALUENUM") < 10000)
        )

        vitals_labs = clean_lf.group_by(["ICUSTAY_ID", "HOURS_IN", "LABEL"]).agg([
            pl.col("VALUENUM").mean().alias("MEAN"),
            pl.col("VALUENUM").count().alias("COUNT"),
            pl.col("VALUENUM").std().alias("STDDEV"),
        ]).collect()

        vitals_labs.write_parquet(proc_dir / self.cfg.artifacts.vitals_file)
        vitals_labs.select(["ICUSTAY_ID", "HOURS_IN", "LABEL", "MEAN"]).write_parquet(proc_dir / self.cfg.artifacts.vitals_mean_file)
        self.observer.log("INFO", "TimeSeriesAggregator: Finished")