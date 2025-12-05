import polars as pl
from pathlib import Path
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig

class StaticExtractor(PipelineComponent):
    """
    Extracts static patient cohort data (ICUSTAYS, ADMISSIONS, PATIENTS).
    Fixed: Polars LazyFrame execution order and output path.
    """
    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        raw_dir = Path(self.cfg.raw_dir)
        proc_dir = Path(self.cfg.proc_dir)
        
        # Ensure output directory exists
        proc_dir.mkdir(parents=True, exist_ok=True)
        
        self.observer.log("INFO", "StaticExtractor: Loading raw tables...")
        
        def find_raw_file(table: str) -> str:
            candidates = sorted(raw_dir.glob(f"{table}*.csv*"))
            if not candidates:
                for p in sorted(raw_dir.glob("*.csv*")):
                    if p.name.lower().startswith(table.lower()):
                        candidates.append(p)
            if not candidates:
                raise FileNotFoundError(f"No raw file for '{table}' in {raw_dir}")
            best = sorted(candidates, key=lambda p: (p.suffix != ".gz", str(p).lower()))[0]
            return str(best)

        def to_datetime_iso(col: str) -> pl.Expr:
            base = pl.col(col).str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
            return pl.when(base.is_null() & pl.col(col).is_not_null()).then(
                pl.col(col).str.replace(r"Z$", "+00:00", literal=False)
                .str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
            ).otherwise(base)

        def floor_int(expr: pl.Expr, dtype=pl.Int32) -> pl.Expr:
            return expr.floor().cast(dtype)

        def between_closed(expr: pl.Expr, left: pl.Expr, right: pl.Expr) -> pl.Expr:
            return expr.is_not_null() & left.is_not_null() & right.is_not_null() & (expr >= left) & (expr <= right)

        # 1. Process ICUSTAYS with explicit aliases to prevent optimization issues
        icu_fp = find_raw_file("ICUSTAYS")
        icu_raw = (
            pl.scan_csv(icu_fp, infer_schema_length=0)
            .with_columns([
                to_datetime_iso("INTIME").alias("INTIME_DT"), 
                to_datetime_iso("OUTTIME").alias("OUTTIME_DT")
            ])
            .with_columns([
                pl.col("LOS").cast(pl.Float64).alias("LOS_ICU"),
                floor_int((pl.col("OUTTIME_DT") - pl.col("INTIME_DT")).dt.total_hours()).alias("STAY_HOURS"),
                pl.col("INTIME_DT").alias("INTIME"),
                pl.col("OUTTIME_DT").alias("OUTTIME")
            ])
        )

        icu = (
            icu_raw.filter(pl.col("LOS_ICU") >= float(self.cfg.min_los_icu_days))
            .filter(pl.col("STAY_HOURS") >= int(self.cfg.min_duration_hours))
            .filter(pl.col("STAY_HOURS") <= int(self.cfg.max_duration_hours))
            .filter(~pl.col("FIRST_CAREUNIT").str.contains("NICU", literal=False))
            .select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS_ICU", "STAY_HOURS"])
        )

        # 2. Process ADMISSIONS
        adm_fp = find_raw_file("ADMISSIONS")
        adm = (
            pl.scan_csv(adm_fp, infer_schema_length=0)
            .with_columns([
                to_datetime_iso("ADMITTIME").alias("ADMITTIME"),
                to_datetime_iso("DISCHTIME").alias("DISCHTIME"),
                to_datetime_iso("DEATHTIME").alias("DEATHTIME"),
            ])
            .select(["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME", "ETHNICITY", "ADMISSION_TYPE", "INSURANCE"])
        )

        # 3. Process PATIENTS
        pat_fp = find_raw_file("PATIENTS")
        pat = (
            pl.scan_csv(pat_fp, infer_schema_length=0)
            .with_columns([
                to_datetime_iso("DOB").alias("DOB"), 
                to_datetime_iso("DOD").alias("DOD")
            ])
            .select(["SUBJECT_ID", "DOB", "DOD"])
        )

        # Join and Filter
        df = (
            icu.join(adm, on=["SUBJECT_ID", "HADM_ID"], how="inner")
            .join(pat, on="SUBJECT_ID", how="inner")
            .with_columns([
                floor_int(((pl.col("INTIME") - pl.col("DOB")).dt.total_days() / 365.2425)).alias("AGE"),
                pl.when(between_closed(pl.col("DEATHTIME"), pl.col("INTIME"), pl.col("OUTTIME"))).then(1).otherwise(0).alias("MORT_ICU"),
                pl.when(between_closed(pl.col("DEATHTIME"), pl.col("ADMITTIME"), pl.col("DISCHTIME"))).then(1).otherwise(0).alias("MORT_HOSP"),
            ])
            .filter(pl.col("AGE") >= int(self.cfg.min_age))
            .select([
                "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ETHNICITY", "ADMISSION_TYPE", "INSURANCE",
                "AGE", "MORT_ICU", "MORT_HOSP", "DOB", "DOD", "ADMITTIME", "DISCHTIME",
                "INTIME", "OUTTIME", "LOS_ICU", "STAY_HOURS"
            ])
            .collect()
        )

        out_path = proc_dir / "patients.parquet"
        df.write_parquet(out_path)
        self.observer.log("INFO", f"StaticExtractor: Saved cohort to {out_path} (rows={df.height})")