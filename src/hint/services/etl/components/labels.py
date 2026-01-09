import polars as pl
import json
import numpy as np
from pathlib import Path
from collections import Counter
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig, ICDConfig, CNNConfig

class LabelGenerator(PipelineComponent):
    """Generate ICD candidate sets and ventilation transition targets.

    This component writes both ventilation transition labels and ICD
    candidate lists required for downstream modeling.
    """

    def __init__(self, etl_config: ETLConfig, icd_config: ICDConfig, cnn_config: CNNConfig, registry: Registry, observer: TelemetryObserver):
        """Initialize the label generator.

        Args:
            etl_config (ETLConfig): ETL configuration.
            icd_config (ICDConfig): ICD configuration.
            cnn_config (CNNConfig): Intervention configuration.
            registry (Registry): Artifact registry.
            observer (TelemetryObserver): Telemetry observer.
        """
        self.etl_cfg = etl_config
        self.icd_cfg = icd_config
        self.cnn_cfg = cnn_config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        """Generate both ventilation and ICD targets.

        Raises:
            FileNotFoundError: If the features dataset is missing.
        """
        proc_dir = Path(self.etl_cfg.proc_dir)
        proc_dir.mkdir(parents=True, exist_ok=True)

        ds_path = proc_dir / self.etl_cfg.artifacts.features_file
        if not ds_path.exists():
            raise FileNotFoundError(f"Input dataset not found at {ds_path}")

        self.observer.log("INFO", f"LabelGenerator: Loading features from {ds_path}")
        ds = pl.read_parquet(ds_path)

        self._generate_vent_targets(ds, proc_dir)
        self._generate_icd_targets(ds, proc_dir)

    def _generate_vent_targets(self, ds: pl.DataFrame, proc_dir: Path) -> None:
        """Generate transition-based ventilation targets.

        Args:
            ds (pl.DataFrame): Input dataset with ventilation labels.
            proc_dir (Path): Output directory for artifacts.
        """
        self.observer.log("INFO", "LabelGenerator: Stage 1/2 generating ventilation transitions.")
        sorted_ds = ds.select(["ICUSTAY_ID", "HOUR_IN", "VENT"]).sort(["ICUSTAY_ID", "HOUR_IN"])

        targets = sorted_ds.with_columns([
            pl.col("VENT").alias("curr_vent"),
            pl.col("VENT").shift(-1).alias("next_vent"),
            pl.col("ICUSTAY_ID").shift(-1).alias("next_stay_id")
        ]).filter(
            pl.col("ICUSTAY_ID") == pl.col("next_stay_id")
        ).with_columns(
            pl.when((pl.col("curr_vent") == 0) & (pl.col("next_vent") == 1)).then(pl.lit(0))
            .when((pl.col("curr_vent") == 1) & (pl.col("next_vent") == 0)).then(pl.lit(1))
            .when((pl.col("curr_vent") == 1) & (pl.col("next_vent") == 1)).then(pl.lit(2))
            .when((pl.col("curr_vent") == 0) & (pl.col("next_vent") == 0)).then(pl.lit(3))
            .otherwise(pl.lit(None))
            .cast(pl.Int64)
            .alias("target_vent")
        ).select(["ICUSTAY_ID", "HOUR_IN", "target_vent"])

        out_path = proc_dir / self.etl_cfg.artifacts.vent_targets_file
        targets.write_parquet(out_path)
        self.observer.log("INFO", f"LabelGenerator: Saved Vent transition targets to {out_path}")

    def _generate_icd_targets(self, ds: pl.DataFrame, proc_dir: Path) -> None:
        """Generate ICD candidate sets for partial label learning.

        Args:
            ds (pl.DataFrame): Input dataset with ICD code lists.
            proc_dir (Path): Output directory for artifacts.
        """
        self.observer.log("INFO", "LabelGenerator: Stage 2/2 generating ICD candidate sets.")

        all_codes = []
        raw_rows = ds.select("ICD9_CODES").to_series().to_list()
        
        for codes in raw_rows:
            if codes:
                all_codes.extend(codes)
            
        code_counts = Counter(all_codes)
        top_k = self.icd_cfg.top_k_labels
        
        if top_k:
            sorted_codes = [c for c, _ in code_counts.most_common(top_k)]
        else:
            sorted_codes = [c for c, _ in code_counts.most_common()]
            
        code_to_idx = {c: i for i, c in enumerate(sorted_codes)}

        def map_codes(codes_list):
            if not codes_list:
                return []
            return [code_to_idx[c] for c in codes_list if c in code_to_idx]

        unique_stays = ds.select(["ICUSTAY_ID", "ICD9_CODES"]).unique(subset=["ICUSTAY_ID"])
        
        targets = unique_stays.with_columns(
            pl.col("ICD9_CODES").map_elements(map_codes, return_dtype=pl.List(pl.Int64)).alias("candidates")
        ).select(["ICUSTAY_ID", "candidates"])

        out_path = proc_dir / self.etl_cfg.artifacts.icd_targets_file
        targets.write_parquet(out_path)
        
        meta_path = proc_dir / self.etl_cfg.artifacts.icd_meta_file
        with open(meta_path, "w") as f:
            json.dump({"icd_classes": sorted_codes}, f, indent=2)
            
        self.observer.log("INFO", f"LabelGenerator: Saved ICD candidate sets to {out_path}")
