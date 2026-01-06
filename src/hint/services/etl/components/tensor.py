import h5py
import hdf5plugin
import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Any
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer

from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import CNNConfig, ETLConfig, ICDConfig
from .labels import _to_python_list

class TensorConverter(PipelineComponent):
    """Convert processed features and labels into HDF5 tensors.

    This component merges feature and label datasets, builds ICD inputs, and
    writes split-aware HDF5 artifacts using semantic keys.

    Attributes:
        etl_cfg (ETLConfig): ETL configuration for paths and keys.
        cnn_cfg (CNNConfig): CNN configuration for sequence length and cache.
        icd_cfg (ICDConfig): ICD configuration for tokenizer and limits.
        registry (Registry): Registry for artifact resolution.
        observer (TelemetryObserver): Telemetry adapter for logging progress.
    """

    def __init__(self, etl_config: ETLConfig, cnn_config: CNNConfig, icd_config: ICDConfig, registry: Registry, observer: TelemetryObserver):
        self.etl_cfg = etl_config
        self.cnn_cfg = cnn_config
        self.icd_cfg = icd_config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        """Generate HDF5 tensors for training, validation, and test splits.

        This method loads the processed features and labels, builds ICD inputs,
        performs group-aware splits, and serializes HDF5 datasets.
        """
        self.observer.log("INFO", "TensorConverter: Stage 1/4 Loading Data")
        proc_dir = Path(self.etl_cfg.proc_dir)

        feat_path = proc_dir / self.etl_cfg.artifacts.features_file
        vent_path = proc_dir / self.etl_cfg.artifacts.vent_targets_file
        icd_path = proc_dir / self.etl_cfg.artifacts.icd_targets_file
        meta_path = proc_dir / self.etl_cfg.artifacts.icd_meta_file

        df_feat = pl.read_parquet(feat_path)
        df_vent = pl.read_parquet(vent_path)
        df_icd = pl.read_parquet(icd_path)

        df = df_feat.join(df_vent, on=["ICUSTAY_ID", "HOUR_IN"], how="inner")
        df = df.join(df_icd, on="ICUSTAY_ID", how="left")

        self.observer.log("INFO", "TensorConverter: Stage 2/4 Preparing ICD Feature Inputs")

        with open(meta_path, "r") as f:
            meta_data = json.load(f)
            icd_classes = meta_data["icd_classes"]

        class_set = set(icd_classes)
        code_to_idx = {c: i for i, c in enumerate(icd_classes)}

        icd_col = "ICD9_CODES"
        unique_codes_df = df.select(["ICUSTAY_ID", icd_col]).unique()
        raw_codes_map = {row["ICUSTAY_ID"]: _to_python_list(row[icd_col]) for row in unique_codes_df.iter_rows(named=True)}

        tokenizer = AutoTokenizer.from_pretrained(self.icd_cfg.model_name)
        stay_metadata = {}
        texts = []
        stay_ids_order = []

        for sid, codes in raw_codes_map.items():
            cands = []
            for c in codes:
                if c in class_set:
                    cands.append(c)
                else:
                    cands.append("__OTHER__")
            if not cands:
                cands = ["__MISSING__"]
            unique_cands = list(set(cands))
            cand_indices = [code_to_idx[c] for c in unique_cands]

            texts.append(" ".join(codes))
            stay_ids_order.append(sid)
            stay_metadata[sid] = {"candidates": cand_indices}

        encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=self.icd_cfg.max_length, return_tensors="np")
        for idx, sid in enumerate(stay_ids_order):
            stay_metadata[sid]["input_ids"] = encodings["input_ids"][idx]
            stay_metadata[sid]["attention_mask"] = encodings["attention_mask"][idx]

        max_cands_len = max(len(m["candidates"]) for m in stay_metadata.values())

        self.observer.log("INFO", "TensorConverter: Stage 3/4 Selecting Columns")
        id_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"]
        exclude = self.cnn_cfg.data.exclude_cols + ["VENT_CLASS", "ICD9_CODES", "target_vent", "target_icd"]

        numeric_cols = []
        categorical_cols = []
        for name, dtype in zip(df.columns, df.dtypes):
            if name in id_cols or name in exclude:
                continue
            if dtype in pl.NUMERIC_DTYPES or dtype == pl.Boolean:
                numeric_cols.append(name)
            elif dtype == pl.Utf8:
                categorical_cols.append(name)

        if categorical_cols:
            encoding_exprs = []
            for col in categorical_cols:
                df = df.with_columns(pl.col(col).fill_null("__NULL__").cast(pl.Categorical))
                encoding_exprs.append((pl.col(col).to_physical() + 1).cast(pl.Int32).alias(col))
            df = df.with_columns(encoding_exprs)

        unique_stays = df.select("ICUSTAY_ID").unique().to_numpy().ravel()
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, teva_idx = next(gss1.split(unique_stays, groups=unique_stays))
        train_ids = unique_stays[train_idx]
        teva_ids = unique_stays[teva_idx]

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        test_idx, val_idx = next(gss2.split(teva_ids, groups=teva_ids))
        val_ids = teva_ids[val_idx]
        test_ids = teva_ids[test_idx]

        self.observer.log("INFO", "TensorConverter: Stage 4/4 Writing HDF5 splits with Semantic Keys")
        cache_dir = Path(self.cnn_cfg.data.data_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / "stats.json", "w") as f:
             json.dump({"icd_classes": icd_classes}, f, indent=2)

        prefix = self.etl_cfg.artifacts.output_h5_prefix
        
        self._process_split(df.filter(pl.col("ICUSTAY_ID").is_in(train_ids)), "train", numeric_cols, categorical_cols, cache_dir, prefix, stay_metadata, max_cands_len)
        self._process_split(df.filter(pl.col("ICUSTAY_ID").is_in(val_ids)), "val", numeric_cols, categorical_cols, cache_dir, prefix, stay_metadata, max_cands_len)
        self._process_split(df.filter(pl.col("ICUSTAY_ID").is_in(test_ids)), "test", numeric_cols, categorical_cols, cache_dir, prefix, stay_metadata, max_cands_len)

    def _flush_batch(self, h5_datasets, buffers, y_key, y_vent_key):
        """Flush buffered arrays into the HDF5 datasets.

        Args:
            h5_datasets (Dict[str, Any]): Target HDF5 datasets.
            buffers (Dict[str, List[np.ndarray]]): Buffered arrays to write.
            y_key (str): Dataset key for ICD targets.
            y_vent_key (str): Dataset key for ventilation targets.
        """
        batch_len = len(buffers["y"])
        if batch_len == 0:
            return

        current_size = h5_datasets[y_key].shape[0]
        for key in h5_datasets:
            h5_datasets[key].resize(current_size + batch_len, axis=0)

        h5_datasets["X_num"][current_size:] = np.concatenate(buffers["X_num"], axis=0)
        h5_datasets["X_cat"][current_size:] = np.concatenate(buffers["X_cat"], axis=0)
        h5_datasets[y_key][current_size:] = np.concatenate(buffers["y"], axis=0)
        h5_datasets[y_vent_key][current_size:] = np.concatenate(buffers["y_vent"], axis=0)
        h5_datasets["sid"][current_size:] = np.concatenate(buffers["sid"], axis=0)
        h5_datasets["hour"][current_size:] = np.concatenate(buffers["hour"], axis=0)

        for k in buffers:
            buffers[k].clear()

    def _process_split(self, df: pl.DataFrame, split_name: str, num_cols: List[str], cat_cols: List[str], cache_dir: Path, prefix: str, stay_metadata: Dict, max_cands: int) -> None:
        """Serialize a single split into an HDF5 file.

        Args:
            df (pl.DataFrame): Split-specific dataframe.
            split_name (str): Name of the split.
            num_cols (List[str]): Numeric feature columns.
            cat_cols (List[str]): Categorical feature columns.
            cache_dir (Path): Output directory for HDF5 artifacts.
            prefix (str): Filename prefix for output.
            stay_metadata (Dict): Per-stay ICD tokenization and candidates.
            max_cands (int): Maximum candidate count for padding.
        """
        h5_path = cache_dir / f"{prefix}_{split_name}.h5"
        self.observer.log("INFO", f"TensorConverter: Writing {split_name} split to {h5_path}")

        df = df.sort(by=["ICUSTAY_ID", "HOUR_IN"])
        seq_len = self.cnn_cfg.seq_len
        bert_max_len = self.icd_cfg.max_length

        norm_exprs = []
        for col in num_cols:
            mean_expr = pl.col(col).mean().over("ICUSTAY_ID")
            std_expr = pl.col(col).std().over("ICUSTAY_ID").fill_null(1.0)
            norm = (pl.col(col) - mean_expr) / (std_expr + 1e-6)
            norm_exprs.append(norm.alias(col))
        df = df.with_columns(norm_exprs)

        all_sids = df["ICUSTAY_ID"].to_numpy()
        all_hours = df["HOUR_IN"].to_numpy()
        all_y_vent = df["target_vent"].to_numpy()
        all_y_icd = df["target_icd"].to_numpy()

        all_x_num = df.select(num_cols).to_numpy().astype(np.float16)
        all_x_cat = df.select(cat_cols).to_numpy().astype(np.int32) if cat_cols else None
        
        unique_sids, start_indices = np.unique(all_sids, return_index=True)
        split_indices = np.append(start_indices, len(all_sids))

        K_ETL = self.etl_cfg.keys
        K_ICD = self.icd_cfg.keys
        K_CNN = self.cnn_cfg.keys

        with h5py.File(h5_path, "w") as f:
            n_num = len(num_cols)
            n_cat = len(cat_cols)

            ds_dynamic = {
                "X_num": f.create_dataset(K_ETL.INPUT_DYN_VITALS, (0, n_num, seq_len), maxshape=(None, n_num, seq_len), dtype=np.float16, compression=hdf5plugin.Blosc()),
                "X_cat": f.create_dataset(K_ETL.INPUT_DYN_CATEGORICAL, (0, n_cat, seq_len), maxshape=(None, n_cat, seq_len), dtype=np.int32, compression=hdf5plugin.Blosc()),
                K_ICD.TARGET_ICD_MULTI: f.create_dataset(K_ICD.TARGET_ICD_MULTI, (0,), maxshape=(None,), dtype=np.int64),
                K_CNN.TARGET_VENT_STATE: f.create_dataset(K_CNN.TARGET_VENT_STATE, (0,), maxshape=(None,), dtype=np.int64),
                "sid": f.create_dataset(K_ETL.STAY_ID, (0,), maxshape=(None,), dtype=np.int64),
                "hour": f.create_dataset(K_ETL.HOUR_IN, (0,), maxshape=(None,), dtype=np.int32),
            }

            f.create_dataset("static_sids", data=unique_sids, dtype=np.int64)

            static_input_ids = []
            static_attn_mask = []
            static_candidates = []

            for sid in unique_sids:
                if sid in stay_metadata:
                    meta = stay_metadata[sid]
                    static_input_ids.append(meta["input_ids"])
                    static_attn_mask.append(meta["attention_mask"])

                    cands = np.array(meta["candidates"], dtype=np.int32)
                    pad_cands = np.full(max_cands, -1, dtype=np.int32)
                    pad_cands[:len(cands)] = cands
                    static_candidates.append(pad_cands)
                else:
                    static_input_ids.append(np.zeros(bert_max_len, dtype=np.int32))
                    static_attn_mask.append(np.zeros(bert_max_len, dtype=np.int32))
                    static_candidates.append(np.full(max_cands, -1, dtype=np.int32))

            f.create_dataset(K_ETL.STATIC_INPUT_IDS, data=np.array(static_input_ids, dtype=np.int32), compression=hdf5plugin.Blosc())
            f.create_dataset(K_ETL.STATIC_ATTN_MASK, data=np.array(static_attn_mask, dtype=np.int32), compression=hdf5plugin.Blosc())
            f.create_dataset(K_ETL.STATIC_CANDS, data=np.array(static_candidates, dtype=np.int32))

            buffers = {
                "X_num": [], "X_cat": [], 
                "y": [],
                "y_vent": [],
                "sid": [], "hour": []
            }

            self.observer.log("INFO", f"TensorConverter: Vectorized processing for {len(unique_sids)} patients...")

            for i in range(len(unique_sids)):
                sid = unique_sids[i]
                start, end = split_indices[i], split_indices[i+1]

                p_x_num = all_x_num[start:end]
                p_x_cat = all_x_cat[start:end] if all_x_cat is not None else None
                p_y_vent = all_y_vent[start:end]
                p_y_icd_static = all_y_icd[start]
                p_hours = all_hours[start:end]

                seq_len_stay = len(p_y_vent)
                pad_width = ((seq_len - 1, 0), (0, 0))

                padded_x_num = np.pad(p_x_num, pad_width, constant_values=np.nan)
                windows_num = sliding_window_view(padded_x_num, window_shape=seq_len, axis=0).transpose(0, 2, 1)

                if p_x_cat is not None:
                    padded_x_cat = np.pad(p_x_cat, pad_width, constant_values=0)
                    windows_cat = sliding_window_view(padded_x_cat, window_shape=seq_len, axis=0).transpose(0, 2, 1)
                else:
                    windows_cat = np.zeros((seq_len_stay, n_cat, seq_len), dtype=np.int32)

                t_y = np.full(seq_len_stay, p_y_icd_static, dtype=np.int64)
                t_sid = np.full(seq_len_stay, sid, dtype=np.int64)

                buffers["X_num"].append(windows_num)
                buffers["X_cat"].append(windows_cat)
                buffers["y"].append(t_y)
                buffers["y_vent"].append(p_y_vent)
                buffers["sid"].append(t_sid)
                buffers["hour"].append(p_hours)

                if len(buffers["y"]) >= 500:
                    self._flush_batch(ds_dynamic, buffers, K_ICD.TARGET_ICD_MULTI, K_CNN.TARGET_VENT_STATE)

            self._flush_batch(ds_dynamic, buffers, K_ICD.TARGET_ICD_MULTI, K_CNN.TARGET_VENT_STATE)
            self.observer.log("INFO", f"TensorConverter: Finished processing {split_name}")
