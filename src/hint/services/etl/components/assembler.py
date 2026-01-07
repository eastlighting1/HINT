import polars as pl
from pathlib import Path
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig


class FeatureAssembler(PipelineComponent):
    """Assemble hourly features from vitals, interventions, and static data.

    This component assumes that upstream components (StaticExtractor, TimeSeriesAggregator,
    OutcomesBuilder, VentilationTagger) have successfully generated the required parquet artifacts.
    """

    def __init__(self, config: ETLConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        """Create the feature parquet used by downstream labeling and modeling."""
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
            error_msg = (
                f"FeatureAssembler: Missing dependencies: {', '.join(missing_deps)}. "
                "Ensure StaticExtractor, TimeSeriesAggregator, OutcomesBuilder, and VentilationTagger "
                "have been executed successfully."
            )
            self.observer.log("ERROR", error_msg)
            raise FileNotFoundError(error_msg)

        self.observer.log("INFO", f"FeatureAssembler: Stage 1/6 loading cohort from {patients_path}")
        patients = pl.read_parquet(patients_path)

        first_icustay = (
            patients.select(["SUBJECT_ID", "ICUSTAY_ID", "INTIME"])
            .sort(["SUBJECT_ID", "INTIME"])
            .group_by("SUBJECT_ID")
            .first()
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
        self.observer.log("INFO", f"FeatureAssembler: Stage 1/6 cohort filters complete stays={pat.height}")

        vitals_lazy = pl.scan_parquet(vitals_path)

        self.observer.log("INFO", "FeatureAssembler: Stage 2/6 loading interventions and vitals sources")
        interventions = pl.read_parquet(interventions_path)

        self.observer.log("INFO", "FeatureAssembler: Stage 3/6 resolving ICD9 source")
        cand = raw_dir / "DIAGNOSES_ICD.csv"
        if not cand.exists() and (raw_dir / "DIAGNOSES_ICD.csv.gz").exists():
            cand = raw_dir / "DIAGNOSES_ICD.csv.gz"

        if cand.exists():
            self.observer.log("INFO", f"FeatureAssembler: Loading ICD9 codes from {cand}")
            icd9 = (
                pl.read_csv(cand, infer_schema_length=0)
                .select(["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
                .with_columns(
                    [
                        pl.col("SUBJECT_ID").cast(pl.Int64),
                        pl.col("HADM_ID").cast(pl.Int64),
                        pl.col("ICD9_CODE").cast(pl.Utf8),
                    ]
                )
                .drop_nulls()
                .group_by(["SUBJECT_ID", "HADM_ID"])
                .agg([pl.col("ICD9_CODE").unique().sort().alias("ICD9_CODES")])
            )
        else:
            self.observer.log("WARNING", "FeatureAssembler: DIAGNOSES_ICD source not found. Using empty ICD codes.")
            icd9 = patients.select(["SUBJECT_ID", "HADM_ID"]).unique().with_columns(pl.lit([]).cast(pl.List(pl.Utf8)).alias("ICD9_CODES"))

        self.observer.log("INFO", "FeatureAssembler: Stage 4/6 preparing vitals pivot")
        varmap_fp = Path(self.cfg.resources_dir) / "itemid_to_variable_map.csv"

        if varmap_fp.exists():
            self.observer.log("INFO", "FeatureAssembler: Optimizing vitals pivot via unique label mapping")

            varmap = (
                pl.read_csv(varmap_fp, infer_schema_length=0)
                .select(["MIMIC LABEL", "LEVEL2"])
                .drop_nulls()
                .with_columns(
                    [
                        pl.col("MIMIC LABEL").str.to_lowercase()
                        .str.replace_all(r"[^a-z0-9]+", "_")
                        .str.replace_all(r"^_+|_+$", "")
                        .alias("MIMIC_NORM"),
                        pl.col("LEVEL2").str.to_lowercase().alias("LEVEL2_NORM"),
                    ]
                )
                .unique()
            )

            vitals_filtered = vitals_lazy.rename({"HOURS_IN": "HOUR_IN"}).join(
                pat.select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]).lazy(),
                on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
                how="inner"
            )

            unique_labels = vitals_filtered.select("LABEL").unique().collect()

            label_mapping = (
                unique_labels
                .with_columns(
                    pl.col("LABEL").str.to_lowercase()
                    .str.replace_all(r"[^a-z0-9]+", "_")
                    .str.replace_all(r"^_+|_+$", "")
                    .alias("LABEL_NORM")
                )
                .join(varmap, left_on="LABEL_NORM", right_on="MIMIC_NORM", how="inner")
                .filter(pl.col("LEVEL2_NORM").is_in([s.lower() for s in self.cfg.exact_level2_104]))
                .select(["LABEL", "LEVEL2_NORM"])
            )

            vitals_norm = (
                vitals_filtered
                .join(label_mapping.lazy(), on="LABEL", how="inner")
                .group_by(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN", "LEVEL2_NORM"])
                .agg(pl.col("MEAN").mean().alias("VALUE"))
                .collect()
            )

            vl_wide = vitals_norm.pivot(
                values="VALUE", 
                index=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"], 
                on="LEVEL2_NORM", 
                aggregate_function="mean"
            )
        else:
            self.observer.log("WARNING", "FeatureAssembler: Variable map missing; pivot skipped")
            vl_wide = pat.select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]).with_columns(pl.lit(0).alias("HOUR_IN"))

        self.observer.log("INFO", "FeatureAssembler: Stage 5/6 aligning vitals columns and names")
        for name in self.cfg.exact_level2_104:
            col_name = name.lower()
            if col_name not in vl_wide.columns:
                vl_wide = vl_wide.with_columns(pl.lit(None).cast(pl.Float64).alias(col_name))

        rename_map = {c: f"V__{c.replace(' ', '_')}" for c in vl_wide.columns if c not in ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"]}
        vl_wide = vl_wide.rename(rename_map)

        v_cols = [f"V__{n.replace(' ', '_').lower()}" for n in self.cfg.exact_level2_104]
        vl_wide = vl_wide.select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"] + [c for c in v_cols if c in vl_wide.columns])

        self.observer.log("INFO", "FeatureAssembler: Stage 6/6 joining static and intervention features")

        vent_df = interventions.select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN", "VENT", "OUTCOME_FLAG"])

        base = (
            vl_wide.join(pat.select(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"]), on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"], how="inner")
            .with_columns((pl.col("INTIME") + pl.duration(hours=pl.col("HOUR_IN"))).dt.hour().alias("HOD"))
            .drop("INTIME")
        )

        age_bins = self.cfg.age_bin_edges
        static = pat.with_columns(
            [
                pl.when(pl.col("AGE") < age_bins[1]).then(pl.lit("AGE_15_39"))
                .when(pl.col("AGE") < age_bins[2]).then(pl.lit("AGE_40_64"))
                .when(pl.col("AGE") < age_bins[3]).then(pl.lit("AGE_65_89"))
                .otherwise(pl.lit("AGE_90PLUS")).alias("AGE_BIN"),
                pl.col("ADMISSION_TYPE").alias("ADM4"),
                pl.col("ETHNICITY").alias("ETH4"),
                pl.col("INSURANCE").alias("INS5"),
            ]
        )

        def one_hot_fixed(df, col, levels, prefix):
            d = df.select([pl.col(col)]).to_dummies(columns=[col])
            for lv in levels:
                cname = f"{col}_{lv}"
                if cname not in d.columns:
                    d = d.with_columns(pl.lit(0).alias(cname))
            rename_map_inner = {c: f"{prefix}__{c.split(col + '_', 1)[1]}" for c in d.columns if col + "_" in c}
            return pl.concat([df.drop(col), d.rename(rename_map_inner)], how="horizontal")

        static = one_hot_fixed(static, "AGE_BIN", ["AGE_15_39", "AGE_40_64", "AGE_65_89", "AGE_90PLUS"], "S__AGE")

        feat = (
            base.join(vent_df, on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"], how="left")
            .join(icd9, on=["SUBJECT_ID", "HADM_ID"], how="left")
            .join(static, on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"], how="left")
            .with_columns(
                [
                    pl.col("VENT").fill_null(0).cast(pl.Int8),
                    pl.col("HOD").cast(pl.Int8),
                    pl.when(pl.col("ICD9_CODES").is_null()).then(pl.lit([]).cast(pl.List(pl.Utf8))).otherwise(pl.col("ICD9_CODES")),
                ]
            )
            .sort(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"])
        )

        out_path = proc_dir / self.cfg.artifacts.features_file
        feat.write_parquet(out_path)
        self.observer.log("INFO", f"FeatureAssembler: Wrote {out_path.name} rows={feat.height}")
