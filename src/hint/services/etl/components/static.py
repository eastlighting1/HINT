import polars as pl
from pathlib import Path
from typing import Any

from hint.foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from hint.domain.vo import ETLConfig

class StaticExtractor(PipelineComponent):
    """
    Extracts static patient cohort data (ICUSTAYS, ADMISSIONS, PATIENTS).
    Ported from make_data.py: step_extract_static
    """
    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        raw_dir = Path(self.cfg.raw_dir)
        
        self.observer.log("INFO", "StaticExtractor: Loading raw tables...")
        
        # Helper to find files (logic ported from find_raw_file)
        def find_raw_file(table: str) -> str:
            candidates = sorted(raw_dir.glob(f"{table}*.csv*"))
            if not candidates:
                for p in sorted(raw_dir.glob("*.csv*")):
                    if p.name.lower().startswith(table.lower()):
                        candidates.append(p)
            if not candidates:
                raise FileNotFoundError(f"No raw file for '{table}' in {raw_dir}")
            # Prefer compressed
            best = sorted(candidates, key=lambda p: (p.suffix != ".gz", str(p).lower()))[0]
            return str(best)

        # Helper for ISO datetime (logic ported from to_datetime_iso)
        def to_datetime_iso(col: str) -> pl.Expr:
            base = pl.col(col).str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
            return pl.when(base.is_null() & pl.col(col).is_not_null()).then(
                pl.col(col)
                .str.replace(r"Z$", "+00:00", literal=False)
                .str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
            ).otherwise(base)

        def floor_int(expr: pl.Expr, dtype=pl.Int32) -> pl.Expr:
            return expr.floor().cast(dtype)

        def between_closed(expr: pl.Expr, left: pl.Expr, right: pl.Expr) -> pl.Expr:
            return expr.is_not_null() & left.is_not_null() & right.is_not_null() & (expr >= left) & (expr <= right)

        # 1. Process ICUSTAYS
        icu_fp = find_raw_file("ICUSTAYS")
        icu_raw = (
            pl.scan_csv(icu_fp, infer_schema_length=0)
            .with_columns([to_datetime_iso("INTIME"), to_datetime_iso("OUTTIME")])
            .with_columns([
                pl.col("LOS").cast(pl.Float64).alias("LOS_ICU"),
                floor_int((pl.col("OUTTIME") - pl.col("INTIME")).dt.total_hours()).alias("STAY_HOURS"),
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
                to_datetime_iso("ADMITTIME"),
                to_datetime_iso("DISCHTIME"),
                to_datetime_iso("DEATHTIME"),
            ])
            .select(["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME", "ETHNICITY", "ADMISSION_TYPE", "INSURANCE"])
        )

        # 3. Process PATIENTS
        pat_fp = find_raw_file("PATIENTS")
        pat = pl.scan_csv(pat_fp, infer_schema_length=0).with_columns(
            [to_datetime_iso("DOB"), to_datetime_iso("DOD")]
        ).select(["SUBJECT_ID", "DOB", "DOD"])

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

        self.registry.save_dataframe(df, "patients.parquet")
        self.observer.log("INFO", f"StaticExtractor: Saved cohort to patients.parquet (rows={df.height})")
