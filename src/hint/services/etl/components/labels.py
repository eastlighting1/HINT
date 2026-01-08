import polars as pl
import json
import numpy as np
from pathlib import Path
from typing import Any, List, Dict
from collections import Counter
from sklearn.preprocessing import LabelEncoder

from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig, ICDConfig, CNNConfig

def _to_python_list(val: Any) -> List[str]:
    """Normalize a value into a list of strings.

    Args:
        val (Any): Raw value to convert.

    Returns:
        List[str]: Normalized list of strings.
    """
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            from ast import literal_eval
            parsed = literal_eval(val)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except:
            if val.startswith("[") and val.endswith("]"):
                return [t.strip(" '\"") for t in val.strip("[]").split(",") if t.strip()]
            return [val]
    return []

class LabelGenerator(PipelineComponent):
    """Generate ventilation and ICD target labels.

    Attributes:
        etl_cfg (ETLConfig): ETL configuration.
        icd_cfg (ICDConfig): ICD configuration.
        cnn_cfg (CNNConfig): CNN configuration.
        registry (Registry): Artifact registry.
        observer (TelemetryObserver): Logging observer.
    """

    def __init__(self, etl_config: ETLConfig, icd_config: ICDConfig, cnn_config: CNNConfig, registry: Registry, observer: TelemetryObserver):
        """Initialize the label generator.

        Args:
            etl_config (ETLConfig): ETL configuration.
            icd_config (ICDConfig): ICD configuration.
            cnn_config (CNNConfig): CNN configuration.
            registry (Registry): Artifact registry.
            observer (TelemetryObserver): Logging observer.
        """
        self.etl_cfg = etl_config
        self.icd_cfg = icd_config
        self.cnn_cfg = cnn_config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        """Run label generation for ventilation and ICD targets."""
        proc_dir = Path(self.etl_cfg.proc_dir)
        if not proc_dir.exists():
            proc_dir.mkdir(parents=True, exist_ok=True)

        ds_path = proc_dir / self.etl_cfg.artifacts.features_file
        if not ds_path.exists():
            raise FileNotFoundError(f"Input dataset not found at {ds_path}")

        self.observer.log("INFO", f"LabelGenerator: Loading features from {ds_path}")
        ds = pl.read_parquet(ds_path)

        self._generate_vent_targets(ds, proc_dir)

        self._generate_icd_targets(ds, proc_dir)

    def _generate_vent_targets(self, ds: pl.DataFrame, proc_dir: Path) -> None:
        """Generate ventilation targets and persist them.

        Args:
            ds (pl.DataFrame): Feature dataset.
            proc_dir (Path): Output directory.
        """
        self.observer.log("INFO", "LabelGenerator: Generating Zone 3 (Ventilation) targets")
        bounds = (
            ds.group_by("ICUSTAY_ID").agg(
                [
                    pl.min("HOUR_IN").alias("min_h"),
                    pl.max("HOUR_IN").alias("max_h"),
                    pl.first("SUBJECT_ID"),
                    pl.first("HADM_ID"),
                ]
            )
        )

        grid_rows = []
        for row in bounds.iter_rows(named=True):
            h_start, h_end = row["min_h"], row["max_h"]
            sid, hid, iid = row["SUBJECT_ID"], row["HADM_ID"], row["ICUSTAY_ID"]
            for h in range(h_start, h_end + 1):
                grid_rows.append((sid, hid, iid, h))

        base = pl.DataFrame(grid_rows, schema=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"], orient="row")

        filled = (
            base.join(ds.select(["ICUSTAY_ID", "HOUR_IN", "VENT"]), on=["ICUSTAY_ID", "HOUR_IN"], how="left")
            .sort(["ICUSTAY_ID", "HOUR_IN"])
            .with_columns(pl.col("VENT").fill_null(strategy="forward").fill_null(0).cast(pl.Int8))
        )

        pred_win = self.etl_cfg.pred_window_h
        expressions = []
        for i in range(pred_win):
            expressions.append(pl.col("VENT").shift(-i).alias(f"v_{i}"))

        windowed = filled.with_columns(expressions)
        cols = [pl.col(f"v_{i}") for i in range(pred_win)]
        sum_expr = sum(cols)

        onset_cond = pl.lit(False)
        for i in range(pred_win - 1):
            onset_cond = onset_cond | ((pl.col(f"v_{i+1}") - pl.col(f"v_{i}")) == 1)

        labels = (
            windowed.with_columns([sum_expr.alias("sum_v"), onset_cond.alias("has_onset")]).with_columns(
                pl.when(pl.col("sum_v") == pred_win)
                .then(pl.lit("STAY ON"))
                .when(pl.col("sum_v") == 0)
                .then(pl.lit("STAY OFF"))
                .when(pl.col("has_onset"))
                .then(pl.lit("ONSET"))
                .otherwise(pl.lit("WEAN"))
                .alias("VENT_CLASS")
            )
        )

        valid_labels = labels.filter(pl.col(f"v_{pred_win-1}").is_not_null()).select(["ICUSTAY_ID", "HOUR_IN", "VENT_CLASS"])
        class_map = {"ONSET": 0, "WEAN": 1, "STAY ON": 2, "STAY OFF": 3}

        final_labels = (
            valid_labels.filter(pl.col("VENT_CLASS").is_in(class_map.keys()))
            .with_columns(pl.col("VENT_CLASS").replace(class_map).cast(pl.Int64).alias("target_vent"))
            .select(["ICUSTAY_ID", "HOUR_IN", "target_vent"])
        )

        out_path = proc_dir / self.etl_cfg.artifacts.vent_targets_file
        final_labels.write_parquet(out_path)
        self.observer.log("INFO", f"LabelGenerator: Saved Vent targets to {out_path} rows={final_labels.height}")

    def _generate_icd_targets(self, ds: pl.DataFrame, proc_dir: Path) -> None:
        """Generate ICD targets and persist them.

        Args:
            ds (pl.DataFrame): Feature dataset.
            proc_dir (Path): Output directory.
        """
        self.observer.log("INFO", "LabelGenerator: Generating Zone 2 (ICD) targets with frequency sorting")

        icd_col = "ICD9_CODES"
        unique_codes_df = ds.select(["ICUSTAY_ID", icd_col]).unique()
        raw_codes_map = {row["ICUSTAY_ID"]: _to_python_list(row[icd_col]) for row in unique_codes_df.iter_rows(named=True)}

        all_codes = []
        for codes in raw_codes_map.values():
            all_codes.extend(codes)

        code_counts = Counter(all_codes)
        top_k = self.icd_cfg.top_k_labels

        # 1. 빈도순 정렬 (가장 많이 등장하는 코드가 0번이 되도록)
        if top_k and len(code_counts) > top_k:
            sorted_codes = [code for code, _ in code_counts.most_common(top_k)]
        else:
            sorted_codes = [code for code, _ in code_counts.most_common()]

        special_tokens = ["__OTHER__", "__MISSING__"]

        # 2. 최종 클래스 리스트 및 매핑 생성
        final_classes = sorted_codes + [t for t in special_tokens if t not in sorted_codes]
        class_to_idx = {cls: idx for idx, cls in enumerate(final_classes)}
        class_set = set(final_classes)

        target_rows = []
        for sid, codes in raw_codes_map.items():
            raw_label = codes[0] if codes else "__MISSING__"

            target_str = raw_label if raw_label in class_set else "__OTHER__"
            
            # 3. 매핑 딕셔너리를 사용하여 인덱스 변환
            target_idx = class_to_idx[target_str]

            target_rows.append({
                "ICUSTAY_ID": sid,
                "target_icd": target_idx
            })

        df_icd = pl.DataFrame(target_rows)
        out_path = proc_dir / self.etl_cfg.artifacts.icd_targets_file
        df_icd.write_parquet(out_path)

        meta_path = proc_dir / self.etl_cfg.artifacts.icd_meta_file
        with open(meta_path, "w") as f:
            # 순서가 중요하므로 정렬된 리스트를 저장
            json.dump({"icd_classes": final_classes}, f, indent=2)

        self.observer.log("INFO", f"LabelGenerator: Saved ICD targets to {out_path} and meta to {meta_path} (sorted by frequency)")