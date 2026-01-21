"""Summary of the labels module.

Longer description of the module purpose and usage.
"""

import polars as pl

import json

import numpy as np

from pathlib import Path

from typing import Any, List, Dict

from collections import Counter

from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver

from ....domain.vo import ETLConfig, ICDConfig, CNNConfig



class LabelGenerator(PipelineComponent):

    """Summary of LabelGenerator purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    cnn_cfg (Any): Description of cnn_cfg.
    etl_cfg (Any): Description of etl_cfg.
    icd_cfg (Any): Description of icd_cfg.
    observer (Any): Description of observer.
    registry (Any): Description of registry.
    """



    def __init__(self, etl_config: ETLConfig, icd_config: ICDConfig, cnn_config: CNNConfig, registry: Registry, observer: TelemetryObserver):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        etl_config (Any): Description of etl_config.
        icd_config (Any): Description of icd_config.
        cnn_config (Any): Description of cnn_config.
        registry (Any): Description of registry.
        observer (Any): Description of observer.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.etl_cfg = etl_config

        self.icd_cfg = icd_config

        self.cnn_cfg = cnn_config

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

        """Summary of _generate_vent_targets.
        
        Longer description of the _generate_vent_targets behavior and usage.
        
        Args:
        ds (Any): Description of ds.
        proc_dir (Any): Description of proc_dir.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.observer.log("INFO", "LabelGenerator: Stage 1/2 generating ventilation transitions.")



        gap_h = int(self.etl_cfg.gap_h)

        pred_h = int(self.etl_cfg.pred_window_h)



        sorted_ds = ds.select(["ICUSTAY_ID", "HOUR_IN", "VENT"]).sort(["ICUSTAY_ID", "HOUR_IN"])



        future_cols = []

        for i in range(pred_h):

            future_cols.append(

                pl.col("VENT").shift(-(gap_h + i)).over("ICUSTAY_ID").alias(f"vent_f{i}")

            )



        targets = sorted_ds.with_columns(future_cols)



        window_valid = pl.all_horizontal([pl.col(f"vent_f{i}").is_not_null() for i in range(pred_h)])

        stay_on = pl.all_horizontal([pl.col(f"vent_f{i}") == 1 for i in range(pred_h)])

        stay_off = pl.all_horizontal([pl.col(f"vent_f{i}") == 0 for i in range(pred_h)])



        if pred_h > 1:

            onset = pl.any_horizontal([

                (pl.col(f"vent_f{i}") == 0) & (pl.col(f"vent_f{i+1}") == 1)

                for i in range(pred_h - 1)

            ])

            wean = pl.any_horizontal([

                (pl.col(f"vent_f{i}") == 1) & (pl.col(f"vent_f{i+1}") == 0)

                for i in range(pred_h - 1)

            ])

        else:

            onset = pl.lit(False)

            wean = pl.lit(False)



        targets = targets.with_columns(

            pl.when(window_valid & onset).then(pl.lit(0))

            .when(window_valid & ~onset & wean).then(pl.lit(1))

            .when(window_valid & stay_on).then(pl.lit(2))

            .when(window_valid & stay_off).then(pl.lit(3))

            .otherwise(pl.lit(None))

            .cast(pl.Int64)

            .alias("target_vent")

        ).select(["ICUSTAY_ID", "HOUR_IN", "target_vent"]).filter(
            pl.col("target_vent").is_not_null()
        )



        out_path = proc_dir / self.etl_cfg.artifacts.vent_targets_file

        targets.write_parquet(out_path)

        self.observer.log("INFO", f"LabelGenerator: Saved Vent transition targets to {out_path}")



    def _generate_icd_targets(self, ds: pl.DataFrame, proc_dir: Path) -> None:

        """Summary of _generate_icd_targets.
        
        Longer description of the _generate_icd_targets behavior and usage.
        
        Args:
        ds (Any): Description of ds.
        proc_dir (Any): Description of proc_dir.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.observer.log("INFO", "LabelGenerator: Stage 2/2 generating ICD candidate sets.")



        all_codes = []

        unique_stays = ds.select(["ICUSTAY_ID", "ICD9_CODES"]).unique(subset=["ICUSTAY_ID"])

        raw_rows = unique_stays.select("ICD9_CODES").to_series().to_list()

        for codes in raw_rows:

            if codes:

                all_codes.extend(codes)



        code_counts = Counter(all_codes)
        sorted_codes = [c for c, _ in code_counts.most_common()]



        code_to_idx = {c: i for i, c in enumerate(sorted_codes)}



        def map_codes(codes_list):

            """Summary of map_codes.
            
            Longer description of the map_codes behavior and usage.
            
            Args:
            codes_list (Any): Description of codes_list.
            
            Returns:
            Any: Description of the return value.
            
            Raises:
            Exception: Description of why this exception might be raised.
            """

            if codes_list is None:

                return []

            if hasattr(codes_list, "to_list"):

                codes_list = codes_list.to_list()

            if len(codes_list) == 0:

                return []

            return [code_to_idx[c] for c in codes_list if c in code_to_idx]



        targets = unique_stays.with_columns(

            pl.col("ICD9_CODES").map_elements(map_codes, return_dtype=pl.List(pl.Int64)).alias("candidates")

        ).select(["ICUSTAY_ID", "candidates"])



        out_path = proc_dir / self.etl_cfg.artifacts.icd_targets_file

        targets.write_parquet(out_path)



        meta_path = proc_dir / self.etl_cfg.artifacts.icd_meta_file

        idx_to_code = {i: c for i, c in enumerate(sorted_codes)}
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "icd_classes": sorted_codes,
                    "icd_to_idx": code_to_idx,
                    "idx_to_icd": idx_to_code,
                },
                f,
                indent=2,
            )



        self.observer.log("INFO", f"LabelGenerator: Saved ICD candidate sets to {out_path}")
