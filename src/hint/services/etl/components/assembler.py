# src/hint/services/etl/components/assembler.py

import polars as pl
import re
from pathlib import Path
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig


class FeatureAssembler(PipelineComponent):
    """Assemble feature tables with 3-channel representation (VAL, MSK, DELTA).

    Attributes:
        cfg (ETLConfig): ETL configuration.
        registry (Registry): Artifact registry.
        observer (TelemetryObserver): Logging observer.
    """

    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        """Initialize the feature assembler."""
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        """Run feature assembly generating V, M, D channels."""
        proc_dir = Path(self.cfg.proc_dir)
        raw_dir = Path(self.cfg.raw_dir)
        
        proc_dir.mkdir(parents=True, exist_ok=True)

        patients_path = proc_dir / self.cfg.artifacts.patients_file
        vitals_path = proc_dir / self.cfg.artifacts.vitals_mean_file
        interventions_path = proc_dir / self.cfg.artifacts.interventions_file

        # Check dependencies
        missing_deps = []
        if not patients_path.exists(): missing_deps.append(f"Patients ({patients_path.name})")
        if not vitals_path.exists(): missing_deps.append(f"Vitals ({vitals_path.name})")
        if not interventions_path.exists(): missing_deps.append(f"Interventions ({interventions_path.name})")

        if missing_deps:
            error_msg = f"FeatureAssembler: Missing dependencies: {', '.join(missing_deps)}."
            self.observer.log("ERROR", error_msg)
            raise FileNotFoundError(error_msg)

        # 1. Load Cohort
        self.observer.log("INFO", f"FeatureAssembler: Stage 1/6 loading cohort")
        patients = pl.read_parquet(patients_path).with_columns([
            pl.col("SUBJECT_ID").cast(pl.Int64),
            pl.col("HADM_ID").cast(pl.Int64),
            pl.col("ICUSTAY_ID").cast(pl.Int64),
            pl.col("INTIME").cast(pl.Datetime),
        ])

        # Filter Cohort (Age, Stay Duration)
        first_icustay = (
            patients.select(["SUBJECT_ID", "ICUSTAY_ID", "INTIME"])
            .sort(["SUBJECT_ID", "INTIME"])
            .group_by("SUBJECT_ID").first()
            .rename({"ICUSTAY_ID": "FIRST_ICUSTAY"})
            .select(["SUBJECT_ID", "FIRST_ICUSTAY"])
        )

        pat = (
            patients.join(first_icustay, on="SUBJECT_ID", how="inner")
            .filter(pl.col("ICUSTAY_ID") == pl.col("FIRST_ICUSTAY"))
            .filter(pl.col("AGE") >= self.cfg.min_age)
            .filter((pl.col("STAY_HOURS") >= self.cfg.min_duration_hours) & (pl.col("STAY_HOURS") < self.cfg.max_duration_hours))
            .select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ETHNICITY", "ADMISSION_TYPE", "INSURANCE", "AGE", "INTIME"])
            .unique()
        )
        self.observer.log("INFO", f"FeatureAssembler: Cohort size={pat.height}")

        # 2. Load and Pivot Vitals
        self.observer.log("INFO", "FeatureAssembler: Stage 2/6 loading vitals")
        vitals_lazy = pl.scan_parquet(vitals_path)
        
        def normalize_name_py(s: str) -> str:
            s = s.lower()
            s = re.sub(r"[^a-z0-9]+", "_", s)
            s = re.sub(r"^_+|_+$", "", s)
            return s

        target_vars = set([normalize_name_py(s) for s in self.cfg.exact_level2_104])
        
        vitals_filtered = vitals_lazy.rename({"HOURS_IN": "HOUR_IN"}).join(
            pat.select(["ICUSTAY_ID"]).lazy(), on="ICUSTAY_ID", how="inner"
        )
        
        vitals_norm = (
            vitals_filtered
            .with_columns(
                pl.col("LABEL").str.to_lowercase()
                .str.replace_all(r"[^a-z0-9]+", "_")
                .str.replace_all(r"^_+|_+$", "")
                .alias("LEVEL2_NORM")
            )
            .filter(pl.col("LEVEL2_NORM").is_in(target_vars))
            .group_by(["ICUSTAY_ID", "HOUR_IN", "LEVEL2_NORM"])
            .agg(pl.col("MEAN").mean().alias("VALUE"))
            .collect()
        )

        if vitals_norm.height > 0:
            vl_wide = vitals_norm.pivot(
                values="VALUE", 
                index=["ICUSTAY_ID", "HOUR_IN"], 
                on="LEVEL2_NORM", 
                aggregate_function="mean"
            )
        else:
            vl_wide = pat.select(["ICUSTAY_ID"]).with_columns(pl.lit(0).alias("HOUR_IN"))

        # Align columns
        v_cols_raw = [normalize_name_py(n) for n in self.cfg.exact_level2_104]
        for col_name in v_cols_raw:
            if col_name not in vl_wide.columns:
                vl_wide = vl_wide.with_columns(pl.lit(None).cast(pl.Float64).alias(col_name))

        # Drop Leakage Columns
        leakage_cols = [
            "tidal_volume_observed", "tidal_volume_set", "tidal_volume_spontaneous", 
            "peak_inspiratory_pressure", "positive_end_expiratory_pressure", 
            "fraction_inspired_oxygen_set", "respiratory_rate_set"
        ]
        norm_leakage = [normalize_name_py(c) for c in leakage_cols]
        existing_leakage = [c for c in norm_leakage if c in vl_wide.columns]
        if existing_leakage:
            self.observer.log("INFO", f"FeatureAssembler: Dropping leakage columns: {existing_leakage}")
            vl_wide = vl_wide.drop(existing_leakage)
        
        # Identify remaining feature columns
        feature_cols = [c for c in vl_wide.columns if c not in ["ICUSTAY_ID", "HOUR_IN"]]

        # 3. Create 3-Channel Representation (VAL, MSK, DELTA)
        self.observer.log("INFO", "FeatureAssembler: Stage 3/6 creating 3-channel tensors (VAL, MSK, DELTA)")
        
        # Load Interventions (Skeleton)
        interventions = pl.read_parquet(interventions_path)
        base = interventions.select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN", "VENT", "OUTCOME_FLAG"])
        
        # Join with pivoted vitals
        df = base.join(vl_wide, on=["ICUSTAY_ID", "HOUR_IN"], how="left").sort(["ICUSTAY_ID", "HOUR_IN"])

        # Generate Channels
        channel_exprs = []
        
        for col in feature_cols:
            # Note: We use V__ prefix for values to distinguish, but original col is used for mask/delta logic
            val_col = f"V__{col}"
            msk_col = f"M__{col}"
            delta_col = f"D__{col}"
            
            # Mask: 1 if present, 0 if missing
            channel_exprs.append(pl.col(col).is_not_null().cast(pl.Int8).alias(msk_col))
            
            # Value: Forward Fill -> Zero Fill
            # Renaming original col to V__ prefix
            channel_exprs.append(pl.col(col).forward_fill().over("ICUSTAY_ID").fill_null(0.0).alias(val_col))
            
            # Delta: Time since last observation
            # 1. Get HOUR_IN where observation exists
            last_obs = pl.when(pl.col(col).is_not_null()).then(pl.col("HOUR_IN")).otherwise(None)
            # 2. Forward fill last observed time
            # 3. Delta = Current Time - Last Observed Time
            channel_exprs.append(
                (pl.col("HOUR_IN") - last_obs.forward_fill().over("ICUSTAY_ID").fill_null(0)).alias(delta_col)
            )

        # Apply transformations and keep only channel columns + keys + targets
        # We drop original feature columns to avoid confusion
        df_channels = df.with_columns(channel_exprs).drop(feature_cols)

        # 4. Load ICD9
        self.observer.log("INFO", "FeatureAssembler: Stage 4/6 loading ICD9 codes")
        cand = raw_dir / "DIAGNOSES_ICD.csv"
        if not cand.exists() and (raw_dir / "DIAGNOSES_ICD.csv.gz").exists():
            cand = raw_dir / "DIAGNOSES_ICD.csv.gz"

        if cand.exists():
            icd9 = (
                pl.read_csv(cand, infer_schema_length=0)
                .select(["HADM_ID", "ICD9_CODE"])
                .with_columns([
                    pl.col("HADM_ID").cast(pl.Int64),
                    pl.col("ICD9_CODE").cast(pl.Utf8)
                ])
                .drop_nulls()
                .group_by("HADM_ID").agg(pl.col("ICD9_CODE").unique().alias("ICD9_CODES"))
            )
        else:
            icd9 = patients.select(["HADM_ID"]).unique().with_columns(pl.lit([]).cast(pl.List(pl.Utf8)).alias("ICD9_CODES"))

        # 5. Static Features
        self.observer.log("INFO", "FeatureAssembler: Stage 5/6 joining static features")
        age_bins = self.cfg.age_bin_edges
        static = pat.with_columns([
            pl.when(pl.col("AGE") < age_bins[1]).then(pl.lit("AGE_15_39"))
            .when(pl.col("AGE") < age_bins[2]).then(pl.lit("AGE_40_64"))
            .when(pl.col("AGE") < age_bins[3]).then(pl.lit("AGE_65_89"))
            .otherwise(pl.lit("AGE_90PLUS")).alias("AGE_BIN"),
            pl.col("ADMISSION_TYPE").alias("ADM4"),
            pl.col("ETHNICITY").alias("ETH4"),
            pl.col("INSURANCE").alias("INS5"),
        ])

        # One-hot encoding for static features
        def one_hot_fixed(d, col, levels, prefix):
            # Select HADM_ID along with the target column to preserve the join key
            d = d.select(["HADM_ID", pl.col(col)]).to_dummies(columns=[col])
            for lv in levels:
                if f"{col}_{lv}" not in d.columns: d = d.with_columns(pl.lit(0).alias(f"{col}_{lv}"))
            rename_map = {c: f"{prefix}__{c.split(col + '_', 1)[1]}" for c in d.columns if col + "_" in c}
            return d.rename(rename_map)

        s_age = one_hot_fixed(static, "AGE_BIN", ["AGE_15_39", "AGE_40_64", "AGE_65_89", "AGE_90PLUS"], "S__AGE")
        
        # Merge all
        feat = (
            df_channels
            .join(icd9, on="HADM_ID", how="left")
            .join(s_age.with_columns(pl.col("HADM_ID").cast(pl.Int64)), on="HADM_ID", how="left")
            # --- FIX: Join with pat to restore INTIME column for HOD calculation ---
            .join(pat.select(["ICUSTAY_ID", "INTIME"]), on="ICUSTAY_ID", how="left")
            # -----------------------------------------------------------------------
            .with_columns([
                pl.col("VENT").fill_null(0).cast(pl.Int8),
                (pl.col("INTIME") + pl.duration(hours=pl.col("HOUR_IN"))).dt.hour().alias("HOD").cast(pl.Int8),
                pl.when(pl.col("ICD9_CODES").is_null()).then(pl.lit([]).cast(pl.List(pl.Utf8))).otherwise(pl.col("ICD9_CODES")),
            ])
            .drop(["INTIME"])
            .sort(["ICUSTAY_ID", "HOUR_IN"])
        )

        out_path = proc_dir / self.cfg.artifacts.features_file
        feat.write_parquet(out_path)
        self.observer.log("INFO", f"FeatureAssembler: Wrote {out_path.name} rows={feat.height} (3-Channel format)")