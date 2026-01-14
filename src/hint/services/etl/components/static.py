"""Summary of the static module.

Longer description of the module purpose and usage.
"""

import polars as pl

from pathlib import Path

from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver

from ....domain.vo import ETLConfig



class StaticExtractor(PipelineComponent):

    """Summary of StaticExtractor purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    cfg (Any): Description of cfg.
    observer (Any): Description of observer.
    registry (Any): Description of registry.
    """



    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        config (Any): Description of config.
        registry (Any): Description of registry.
        observer (Any): Description of observer.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.cfg = config

        self.registry = registry

        self.observer = observer



    def execute(self) -> None:

        """Summary of execute.
        
        Longer description of the execute behavior and usage.
        
        Args:
        None (None): This function does not accept arguments.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        raw_dir = Path(self.cfg.raw_dir)

        proc_dir = Path(self.cfg.proc_dir)

        proc_dir.mkdir(parents=True, exist_ok=True)



        self.observer.log("INFO", f"StaticExtractor: Searching raw CSVs under {raw_dir}")



        def find_raw_file(table: str) -> Path:

            """Summary of find_raw_file.
            
            Longer description of the find_raw_file behavior and usage.
            
            Args:
            table (Any): Description of table.
            
            Returns:
            Path: Description of the return value.
            
            Raises:
            Exception: Description of why this exception might be raised.
            """

            cand_gz = raw_dir / f"{table}.csv.gz"

            cand_csv = raw_dir / f"{table}.csv"

            if cand_gz.exists(): return cand_gz

            if cand_csv.exists(): return cand_csv

            raise FileNotFoundError(f"No raw file for '{table}' in {raw_dir}")



        def to_datetime_iso(col: str) -> pl.Expr:

            """Summary of to_datetime_iso.
            
            Longer description of the to_datetime_iso behavior and usage.
            
            Args:
            col (Any): Description of col.
            
            Returns:
            pl.Expr: Description of the return value.
            
            Raises:
            Exception: Description of why this exception might be raised.
            """

            base = pl.col(col).str.to_datetime(time_unit="us", time_zone="UTC", strict=False)

            return pl.when(base.is_null() & pl.col(col).is_not_null()).then(

                pl.col(col).str.replace(r"Z$", "+00:00", literal=False).str.to_datetime(time_unit="us", time_zone="UTC", strict=False)

            ).otherwise(base)



        self.observer.log("INFO", "StaticExtractor: Stage 1/4 loading ICU stays")

        icu_fp = find_raw_file("ICUSTAYS")

        self.observer.log("INFO", f"StaticExtractor: Loading ICU stays from {icu_fp.name}")



        icu = (

            pl.scan_csv(icu_fp, infer_schema_length=0)

            .with_columns([

                pl.col("SUBJECT_ID").cast(pl.Int64),

                pl.col("HADM_ID").cast(pl.Int64),

                pl.col("ICUSTAY_ID").cast(pl.Int64),

                pl.col("LOS").cast(pl.Float64).alias("LOS"),

                to_datetime_iso("INTIME").alias("INTIME"),

                to_datetime_iso("OUTTIME").alias("OUTTIME"),

                pl.col("LOS").cast(pl.Float64).alias("LOS_ICU")

            ])

            .with_columns(

                ((pl.col("OUTTIME") - pl.col("INTIME")).dt.total_hours()).floor().cast(pl.Int32).alias("STAY_HOURS")

            )

            .filter(

                (pl.col("LOS_ICU") >= float(self.cfg.min_los_icu_days)) &

                (pl.col("STAY_HOURS") >= int(self.cfg.min_duration_hours)) &

                (pl.col("STAY_HOURS") < int(self.cfg.max_duration_hours))

            )

        )



        self.observer.log("INFO", "StaticExtractor: Stage 2/4 loading admissions")

        adm_fp = find_raw_file("ADMISSIONS")

        adm = (

            pl.scan_csv(adm_fp, infer_schema_length=0)

            .with_columns([

                pl.col("SUBJECT_ID").cast(pl.Int64),

                pl.col("HADM_ID").cast(pl.Int64),

                to_datetime_iso("ADMITTIME"),

                to_datetime_iso("DISCHTIME"),

                to_datetime_iso("DEATHTIME")

            ])

            .select(["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME", "ETHNICITY", "ADMISSION_TYPE", "INSURANCE"])

        )



        self.observer.log("INFO", "StaticExtractor: Stage 3/4 loading patients")

        pat_fp = find_raw_file("PATIENTS")

        pat = (

            pl.scan_csv(pat_fp, infer_schema_length=0)

            .with_columns([

                pl.col("SUBJECT_ID").cast(pl.Int64),

                to_datetime_iso("DOB"),

                to_datetime_iso("DOD")

            ])

            .select(["SUBJECT_ID", "DOB", "DOD"])

        )



        self.observer.log("INFO", "StaticExtractor: Stage 4/4 joining tables and computing outcomes")

        df = (

            icu.join(adm, on=["SUBJECT_ID", "HADM_ID"], how="inner")

            .join(pat, on="SUBJECT_ID", how="inner")

            .with_columns([

                ((pl.col("INTIME") - pl.col("DOB")).dt.total_days() / 365.2425).floor().cast(pl.Int32).alias("AGE"),

                pl.when((pl.col("DEATHTIME").is_not_null()) & (pl.col("DEATHTIME") >= pl.col("INTIME")) & (pl.col("DEATHTIME") <= pl.col("OUTTIME")))

                  .then(1).otherwise(0).alias("MORT_ICU"),

                pl.when((pl.col("DEATHTIME").is_not_null()) & (pl.col("DEATHTIME") >= pl.col("ADMITTIME")) & (pl.col("DEATHTIME") <= pl.col("DISCHTIME")))

                  .then(1).otherwise(0).alias("MORT_HOSP"),

            ])

            .filter(pl.col("AGE") >= int(self.cfg.min_age))

            .collect()

        )



        out_path = proc_dir / self.cfg.artifacts.patients_file

        df.write_parquet(out_path)

        self.observer.log("INFO", f"StaticExtractor: Saved cohort to {out_path} rows={df.height}")
