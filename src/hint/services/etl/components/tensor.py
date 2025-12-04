import h5py
import hdf5plugin
import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Any, Sequence
from sklearn.model_selection import GroupShuffleSplit

from hint.foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from hint.domain.vo import CNNConfig

class TensorConverter(PipelineComponent):
    """
    Converts dataset into HDF5 tensors with sliding windows.
    Fully ported from preprocess.py including windowing logic and batch flushing.
    """
    def __init__(self, config: CNNConfig, registry: Registry, observer: TelemetryObserver):
        self.cfg = config
        self.registry = registry
        self.observer = observer
        self.CLASS_NAMES = ["ONSET", "WEAN", "STAY ON", "STAY OFF"]
        self.CLASS_TO_ID = {name: idx for idx, name in enumerate(self.CLASS_NAMES)}

    def execute(self) -> None:
        self.observer.log("INFO", "TensorConverter: Starting preprocessing pipeline...")
        
        # Load Data
        df = self.registry.load_dataframe("dataset_123_answer.parquet")
        
        # Infer Schema
        id_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"]
        label_col = "VENT_CLASS"
        exclude = self.cfg.exclude_cols
        
        numeric_cols = []
        categorical_cols = []
        
        for name, dtype in zip(df.columns, df.dtypes):
            if name in id_cols + [label_col] + exclude: continue
            if dtype in pl.NUMERIC_DTYPES or dtype == pl.Boolean:
                numeric_cols.append(name)
            elif dtype == pl.Utf8:
                categorical_cols.append(name)
        
        self.observer.log("INFO", f"TensorConverter: Numeric={len(numeric_cols)}, Categorical={len(categorical_cols)}")

        # Encode Categoricals
        vocab_info = {}
        categorical_idx_cols = []
        if categorical_cols:
            encoding_exprs = []
            for col in categorical_cols:
                idx_name = f"{col}_IDX"
                categorical_idx_cols.append(idx_name)
                # Need to calculate vocab size from full dataframe first to be consistent? 
                # In preprocess.py it does it on the full loaded DF before splitting.
                df = df.with_columns(pl.col(col).fill_null("__NULL__").cast(pl.Categorical))
                vocab_size = int(df.select(pl.col(col).n_unique()).item() + 1)
                vocab_info[col] = vocab_size
                encoding_exprs.append((pl.col(col).to_physical() + 1).cast(pl.Int32).alias(idx_name))
            df = df.with_columns(encoding_exprs)

        # Define Feature Names
        feat_names_num = [f"{c}_VAL" for c in numeric_cols] + [f"{c}_MSK" for c in numeric_cols] + [f"{c}_DT" for c in numeric_cols]
        
        # Splitting
        unique_stays = df.select("ICUSTAY_ID").unique().to_numpy().ravel()
        # Seed logic from original code
        gss1 = GroupShuffleSplit(1, test_size=0.2, random_state=42)
        train_idx, teva_idx = next(gss1.split(unique_stays, groups=unique_stays))
        train_ids = unique_stays[train_idx]
        teva_ids = unique_stays[teva_idx]
        
        gss2 = GroupShuffleSplit(1, test_size=0.5, random_state=42)
        test_idx, val_idx = next(gss2.split(teva_ids, groups=teva_ids))
        val_ids = teva_ids[val_idx]
        test_ids = teva_ids[test_idx]
        
        # Calc Stats on Train
        df_tr = df.filter(pl.col("ICUSTAY_ID").is_in(train_ids))
        stats = {}
        v_cols = [c for c in numeric_cols if c.startswith("V__")]
        if v_cols:
            stats_exprs = [pl.col(c).mean().alias(f"{c}_mean") for c in v_cols] + [pl.col(c).std().alias(f"{c}_std") for c in v_cols]
            stats_df = df_tr.select(stats_exprs)
            stats_raw = stats_df.to_dicts()[0]
            for k, v in stats_raw.items():
                if v is None:
                    stats[k] = 1.0 if k.endswith("_std") else 0.0
                    continue
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    stats[k] = 1.0 if k.endswith("_std") else 0.0
                    continue
                
                if k.endswith("_std") and (val == 0.0 or np.isnan(val)):
                    stats[k] = 1.0
                elif np.isnan(val):
                    stats[k] = 0.0
                else:
                    stats[k] = val
        
        self.registry.save_json(stats, "train_stats.json")
        
        # Save Meta
        meta = {
            "base_feats_numeric": numeric_cols,
            "feat_names_numeric": feat_names_num,
            "n_feats_numeric": len(feat_names_num),
            "n_base_feats_numeric": len(numeric_cols),
            "base_feats_categorical": categorical_cols,
            "base_feats_categorical_idx": categorical_idx_cols,
            "vocab_info": vocab_info,
            "seq_len": self.cfg.seq_len,
            "class_names": self.CLASS_NAMES
        }
        self.registry.save_json(meta, "feature_info.json")

        # Process Splits (This will now contain the full logic)
        self._process_split(df_tr, "train", numeric_cols, feat_names_num, categorical_idx_cols, stats)
        self._process_split(df.filter(pl.col("ICUSTAY_ID").is_in(val_ids)), "val", numeric_cols, feat_names_num, categorical_idx_cols, stats)
        self._process_split(df.filter(pl.col("ICUSTAY_ID").is_in(test_ids)), "test", numeric_cols, feat_names_num, categorical_idx_cols, stats)

    def _flush_batch(
        self,
        h5_X_num: h5py.Dataset,
        h5_X_cat: h5py.Dataset,
        h5_y: h5py.Dataset,
        h5_sid: h5py.Dataset,
        X_batch_num: List[np.ndarray],
        X_batch_cat: List[np.ndarray],
        y_batch: List[int],
        sid_batch: List[int],
    ) -> None:
        """Internal helper to flush buffer to HDF5"""
        batch_len = len(y_batch)
        if batch_len == 0:
            return

        current_size = h5_y.shape[0]

        h5_X_num.resize(current_size + batch_len, axis=0)
        h5_X_cat.resize(current_size + batch_len, axis=0)
        h5_y.resize(current_size + batch_len, axis=0)
        h5_sid.resize(current_size + batch_len, axis=0)

        h5_X_num[current_size:] = np.array(X_batch_num, dtype=np.float16)
        h5_X_cat[current_size:] = np.array(X_batch_cat, dtype=np.int32)
        h5_y[current_size:] = np.array(y_batch, dtype=np.int64)
        h5_sid[current_size:] = np.array(sid_batch, dtype=np.int64)

        X_batch_num.clear()
        X_batch_cat.clear()
        y_batch.clear()
        sid_batch.clear()

    def _process_split(self, df: pl.DataFrame, split_name: str, num_cols: List[str], feat_names: List[str], cat_cols: List[str], stats: Dict[str, float]) -> None:
        h5_path = self.registry.dirs["data"] / f"{split_name}.h5"
        self.observer.log("INFO", f"TensorConverter: Writing {split_name} split to {h5_path}")
        
        # Sort by ID and Time
        df = df.sort(by=["ICUSTAY_ID", "HOUR_IN"])
        
        seq_len = self.cfg.seq_len
        n_feats_num = len(feat_names)
        n_cat_feats = len(cat_cols)

        # 1. Build Expressions
        val_exprs = [pl.col(col).forward_fill().over("ICUSTAY_ID").alias(f"{col}_VAL") for col in num_cols]
        mask_exprs = [pl.col(col).is_not_null().cast(pl.Float32).alias(f"{col}_MSK") for col in num_cols]
        
        # Delta Expression Logic
        delta_exprs = []
        for col in num_cols:
            last_time = (
                pl.when(pl.col(col).is_not_null())
                .then(pl.col("HOUR_IN"))
                .otherwise(None)
                .forward_fill()
                .over("ICUSTAY_ID")
            )
            normalized_delta = (
                (pl.col("HOUR_IN") - last_time)
                .fill_null(seq_len)
                .clip(0, seq_len)
                .alias(f"{col}_DT")
            )
            delta_exprs.append((normalized_delta / seq_len))

        # Apply transformations
        df = df.with_columns(val_exprs + mask_exprs + delta_exprs)
        
        # Fill Nulls (Median logic simplified to 0.0 as per preprocess.py median map fallbacks)
        val_cols = [f"{col}_VAL" for col in num_cols]
        df = df.with_columns([pl.col(col).fill_null(0.0) for col in val_cols])

        # Normalize
        norm_exprs = []
        for col in num_cols:
            val_col = f"{col}_VAL"
            if col.startswith("V__"):
                mean = stats.get(f"{col}_mean", 0.0)
                std = stats.get(f"{col}_std", 1.0)
                norm_exprs.append(((pl.col(val_col) - mean) / (std + 1e-6)).alias(val_col))
            else:
                norm_exprs.append(pl.col(val_col))
        df = df.with_columns(norm_exprs)

        # Filter valid labels
        df = df.filter(pl.col("VENT_CLASS").is_in(self.CLASS_NAMES))
        
        final_cols = ["ICUSTAY_ID", "VENT_CLASS"] + list(feat_names) + list(cat_cols)
        df = df.select(final_cols)
        
        # HDF5 Writing
        with h5py.File(h5_path, "w") as f:
            h5_X_num = f.create_dataset(
                "X_num",
                (0, n_feats_num, seq_len),
                maxshape=(None, n_feats_num, seq_len),
                chunks=(64, n_feats_num, seq_len),
                dtype=np.float16,
                compression=hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE),
            )
            h5_X_cat = f.create_dataset(
                "X_cat",
                (0, n_cat_feats, seq_len),
                maxshape=(None, n_cat_feats, seq_len),
                chunks=(64, n_cat_feats, seq_len),
                dtype=np.int32,
                compression=hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE),
            )
            h5_y = f.create_dataset("y", (0,), maxshape=(None,), chunks=(4096,), dtype=np.int64)
            h5_sid = f.create_dataset("sid", (0,), maxshape=(None,), chunks=(4096,), dtype=np.int64)

            # Buffer
            window_buffer_num = np.zeros((n_feats_num, seq_len), dtype=np.float16)
            window_buffer_cat = np.zeros((n_cat_feats, seq_len), dtype=np.int32)

            batch_size = 4096
            X_batch_num: List[np.ndarray] = []
            X_batch_cat: List[np.ndarray] = []
            y_batch: List[int] = []
            sid_batch: List[int] = []

            # Group Iteration
            parts = df.partition_by("ICUSTAY_ID", maintain_order=True)
            
            with self.observer.create_progress(f"Writing {split_name}", total=len(parts)) as progress:
                task = progress.add_task("Streaming", total=len(parts))
                
                for group in parts:
                    pdf = group.to_pandas()
                    sid = int(pdf["ICUSTAY_ID"].iloc[0])
                    labels = pdf["VENT_CLASS"].tolist()

                    arr_num = pdf[feat_names].to_numpy(dtype=np.float32).T
                    if cat_cols:
                        arr_cat = pdf[cat_cols].to_numpy(dtype=np.int32).T
                    else:
                        arr_cat = None

                    for t, label in enumerate(labels):
                        label_id = self.CLASS_TO_ID.get(label)
                        if label_id is None:
                            continue

                        start = max(0, t - seq_len + 1)
                        end = t + 1
                        window_len = end - start

                        window_buffer_num.fill(0)
                        window_buffer_num[:, seq_len - window_len :] = arr_num[:, start:end]

                        if arr_cat is not None:
                            window_buffer_cat.fill(0)
                            window_buffer_cat[:, seq_len - window_len :] = arr_cat[:, start:end]
                            X_batch_cat.append(window_buffer_cat.copy())
                        else:
                            X_batch_cat.append(np.zeros((n_cat_feats, seq_len), dtype=np.int32))

                        X_batch_num.append(window_buffer_num.copy())
                        y_batch.append(label_id)
                        sid_batch.append(sid)

                        if len(y_batch) >= batch_size:
                            self._flush_batch(h5_X_num, h5_X_cat, h5_y, h5_sid, X_batch_num, X_batch_cat, y_batch, sid_batch)
                    
                    progress.advance(task, 1)

            # Final flush
            self._flush_batch(h5_X_num, h5_X_cat, h5_y, h5_sid, X_batch_num, X_batch_cat, y_batch, sid_batch)
