"""Summary of the assembler module.

Longer description of the module purpose and usage.
"""

import polars as pl
import polars.selectors as cs

import re

from pathlib import Path

from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver

from ....domain.vo import ETLConfig, InterventionConfig





class FeatureAssembler(PipelineComponent):

    """Summary of FeatureAssembler purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        cfg (Any): Description of cfg.
        intervention_cfg (Any): Description of intervention_cfg.
        observer (Any): Description of observer.
        registry (Any): Description of registry.
    """



    def __init__(self, config: ETLConfig, intervention_config: InterventionConfig, registry: Registry, observer: TelemetryObserver):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            config (Any): Description of config.
            intervention_config (Any): Description of intervention_config.
            registry (Any): Description of registry.
            observer (Any): Description of observer.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        self.cfg = config

        self.intervention_cfg = intervention_config

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

        proc_dir = Path(self.cfg.proc_dir)

        raw_dir = Path(self.cfg.raw_dir)



        proc_dir.mkdir(parents=True, exist_ok=True)



        patients_path = proc_dir / self.cfg.artifacts.patients_file

        vitals_path = proc_dir / self.cfg.artifacts.vitals_mean_file

        interventions_path = proc_dir / self.cfg.artifacts.interventions_file





        missing_deps = []

        if not patients_path.exists(): missing_deps.append(f"Patients ({patients_path.name})")

        if not vitals_path.exists(): missing_deps.append(f"Vitals ({vitals_path.name})")

        if not interventions_path.exists(): missing_deps.append(f"Interventions ({interventions_path.name})")



        if missing_deps:

            error_msg = f"[1.5.0] Missing dependencies. items={', '.join(missing_deps)}"

            self.observer.log("ERROR", error_msg)

            raise FileNotFoundError(error_msg)





        self.observer.log("INFO", "[1.5.1] Loading cohort")

        patients = pl.read_parquet(patients_path).with_columns([

            pl.col("SUBJECT_ID").cast(pl.Int64),

            pl.col("HADM_ID").cast(pl.Int64),

            pl.col("ICUSTAY_ID").cast(pl.Int64),

            pl.col("INTIME").cast(pl.Datetime),

        ])





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

        self.observer.log("INFO", f"[1.5.1] Cohort size={pat.height}")





        self.observer.log("INFO", "[1.5.2] Loading vitals")

        vitals_lazy = pl.scan_parquet(vitals_path)



        def normalize_name_py(s: str) -> str:

            """Summary of normalize_name_py.
            
            Longer description of the normalize_name_py behavior and usage.
            
            Args:
                s (Any): Description of s.
            
            Returns:
                str: Description of the return value.
            
            Raises:
                Exception: Description of why this exception might be raised.
            """

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





        v_cols_raw = [normalize_name_py(n) for n in self.cfg.exact_level2_104]

        for col_name in v_cols_raw:

            if col_name not in vl_wide.columns:

                vl_wide = vl_wide.with_columns(pl.lit(None).cast(pl.Float64).alias(col_name))





        leakage_markers = [
            "tidal_volume",
            "respiratory_rate_set",
            "fraction_inspired_oxygen_set",
            "peak_inspiratory_pressure",
            "positive_end_expiratory_pressure",
            "peep",
        ]
        leakage_cols = [
            c for c in vl_wide.columns
            if c not in ["ICUSTAY_ID", "HOUR_IN"]
            and any(marker in c for marker in leakage_markers)
        ]
        if leakage_cols:
            self.observer.log("INFO", f"[1.5.3] Dropping leakage columns. cols={leakage_cols}")
            vl_wide = vl_wide.drop(leakage_cols)

        feature_cols = [c for c in vl_wide.columns if c not in ["ICUSTAY_ID", "HOUR_IN"]]





        self.observer.log("INFO", "[1.5.4] Creating 3-channel tensors (VAL, MSK, DELTA)")





        interventions = pl.read_parquet(interventions_path)

        base = interventions.select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN", "VENT", "OUTCOME_FLAG"])





        df = base.join(vl_wide, on=["ICUSTAY_ID", "HOUR_IN"], how="left").sort(["ICUSTAY_ID", "HOUR_IN"])





        channel_exprs = []



        for col in feature_cols:



            val_col = f"V__{col}"

            msk_col = f"M__{col}"

            delta_col = f"D__{col}"





            cleaned = pl.when(pl.col(col).is_finite()).then(pl.col(col)).otherwise(None)
            channel_exprs.append(cleaned.is_not_null().cast(pl.Int8).alias(msk_col))







            channel_exprs.append(cleaned.forward_fill().over("ICUSTAY_ID").alias(val_col))







            last_obs = pl.when(cleaned.is_not_null()).then(pl.col("HOUR_IN")).otherwise(None)





            channel_exprs.append(

                (pl.col("HOUR_IN") - last_obs.forward_fill().over("ICUSTAY_ID").fill_null(0)).alias(delta_col)

            )







        df_channels = df.with_columns(channel_exprs).drop(feature_cols)



        time_norm = float(self.intervention_cfg.seq_len)

        df_channels = df_channels.with_columns([

            pl.col("VENT").cast(pl.Float32).alias("V__vent_state"),

            pl.lit(1).cast(pl.Int8).alias("M__vent_state"),

            pl.lit(0).cast(pl.Int32).alias("D__vent_state"),

            (pl.col("HOUR_IN") / time_norm).cast(pl.Float32).alias("V__time"),

            pl.lit(1).cast(pl.Int8).alias("M__time"),

            pl.lit(0).cast(pl.Int32).alias("D__time"),

        ])





        self.observer.log("INFO", "[1.5.5] Loading ICD9 codes")

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





        self.observer.log("INFO", "[1.5.6] Joining static features")

        age_bins = self.cfg.age_bin_edges

        static = pat.with_columns([

            pl.when(pl.col("AGE") < age_bins[1]).then(pl.lit("AGE_15_39"))

            .when(pl.col("AGE") < age_bins[2]).then(pl.lit("AGE_40_64"))

            .when(pl.col("AGE") < age_bins[3]).then(pl.lit("AGE_65_89"))

            .otherwise(pl.lit("AGE_90PLUS")).alias("AGE_BIN"),

            pl.col("ADMISSION_TYPE").cast(pl.Utf8).str.to_uppercase().fill_null("OTHER").alias("ADM4"),

            pl.col("ETHNICITY").cast(pl.Utf8).str.to_uppercase().fill_null("OTHER").alias("ETH4"),

            pl.col("INSURANCE").cast(pl.Utf8).str.to_uppercase().fill_null("OTHER").alias("INS5"),

        ])



        adm_col = pl.col("ADM4")

        eth_col = pl.col("ETH4")

        ins_col = pl.col("INS5")



        static = static.with_columns([

            pl.when(adm_col == "EMERGENCY").then(pl.lit("EMERGENCY"))

            .when(adm_col == "URGENT").then(pl.lit("URGENT"))

            .when(adm_col == "ELECTIVE").then(pl.lit("ELECTIVE"))

            .otherwise(pl.lit("OTHER")).alias("ADM4"),

            pl.when(eth_col.str.contains("WHITE")).then(pl.lit("WHITE"))

            .when(eth_col.str.contains("BLACK")).then(pl.lit("BLACK"))

            .when(eth_col.str.contains("HISPANIC")).then(pl.lit("HISPANIC"))

            .otherwise(pl.lit("OTHER")).alias("ETH4"),

            pl.when(ins_col.str.contains("MEDICARE")).then(pl.lit("MEDICARE"))

            .when(ins_col.str.contains("MEDICAID")).then(pl.lit("MEDICAID"))

            .when(ins_col.str.contains("PRIVATE")).then(pl.lit("PRIVATE"))

            .when(ins_col.str.contains("SELF")).then(pl.lit("SELF_PAY"))

            .otherwise(pl.lit("OTHER")).alias("INS5"),

        ])





        def one_hot_fixed(d, col, levels, prefix):

            """Summary of one_hot_fixed.
            
            Longer description of the one_hot_fixed behavior and usage.
            
            Args:
                d (Any): Description of d.
                col (Any): Description of col.
                levels (Any): Description of levels.
                prefix (Any): Description of prefix.
            
            Returns:
                Any: Description of the return value.
            
            Raises:
                Exception: Description of why this exception might be raised.
            """



            d = d.select(["HADM_ID", pl.col(col)]).to_dummies(columns=[col])

            for lv in levels:

                if f"{col}_{lv}" not in d.columns: d = d.with_columns(pl.lit(0).alias(f"{col}_{lv}"))

            rename_map = {c: f"{prefix}__{c.split(col + '_', 1)[1]}" for c in d.columns if col + "_" in c}

            return d.rename(rename_map)



        s_age = one_hot_fixed(static, "AGE_BIN", ["AGE_15_39", "AGE_40_64", "AGE_65_89", "AGE_90PLUS"], "S__AGE")

        s_adm = one_hot_fixed(static, "ADM4", ["EMERGENCY", "URGENT", "ELECTIVE", "OTHER"], "S__ADM")

        s_eth = one_hot_fixed(static, "ETH4", ["WHITE", "BLACK", "HISPANIC", "OTHER"], "S__ETH")

        s_ins = one_hot_fixed(static, "INS5", ["MEDICARE", "MEDICAID", "PRIVATE", "SELF_PAY", "OTHER"], "S__INS")





        feat = (

            df_channels

            .join(icd9, on="HADM_ID", how="left")

            .join(s_age.with_columns(pl.col("HADM_ID").cast(pl.Int64)), on="HADM_ID", how="left")

            .join(s_adm.with_columns(pl.col("HADM_ID").cast(pl.Int64)), on="HADM_ID", how="left")

            .join(s_eth.with_columns(pl.col("HADM_ID").cast(pl.Int64)), on="HADM_ID", how="left")

            .join(s_ins.with_columns(pl.col("HADM_ID").cast(pl.Int64)), on="HADM_ID", how="left")



            .join(pat.select(["ICUSTAY_ID", "INTIME"]), on="ICUSTAY_ID", how="left")



            .with_columns([

                cs.starts_with("S__").fill_null(0).cast(pl.Int8),

                pl.col("VENT").fill_null(0).cast(pl.Int8),

                (pl.col("INTIME") + pl.duration(hours=pl.col("HOUR_IN"))).dt.hour().alias("HOD").cast(pl.Int8),

                pl.when(pl.col("ICD9_CODES").is_null()).then(pl.lit([]).cast(pl.List(pl.Utf8))).otherwise(pl.col("ICD9_CODES")),

            ])

            .drop(["INTIME"])

            .sort(["ICUSTAY_ID", "HOUR_IN"])

        )



        out_path = proc_dir / self.cfg.artifacts.features_file

        feat.write_parquet(out_path)

        self.observer.log("INFO", f"[1.5.7] Dataset written. path={out_path.name} rows={feat.height}")
