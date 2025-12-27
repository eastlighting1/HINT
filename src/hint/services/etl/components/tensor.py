import h5py
import hdf5plugin
import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Any
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import CNNConfig, ETLConfig, ICDConfig


def _to_python_list(val: Any) -> List[str]:
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


class TensorConverter(PipelineComponent):
    """Convert processed features and labels into HDF5 tensors.
    
    Optimized: Separates static (patient-level) and dynamic (time-level) data to prevent IO bottlenecks.
    """

    def __init__(self, etl_config: ETLConfig, cnn_config: CNNConfig, icd_config: ICDConfig, registry: Registry, observer: TelemetryObserver):
        self.etl_cfg = etl_config
        self.cnn_cfg = cnn_config
        self.icd_cfg = icd_config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        self.observer.log("INFO", "TensorConverter: Preparing processed sources for tensorization")
        proc_dir = Path(self.etl_cfg.proc_dir)
        feat_path = proc_dir / self.etl_cfg.artifacts.features_file
        label_path = proc_dir / self.etl_cfg.artifacts.labels_file

        self.observer.log("INFO", f"TensorConverter: Loading features from {feat_path}")
        df_feat = pl.read_parquet(feat_path)
        df_label = pl.read_parquet(label_path)

        self.observer.log("INFO", "TensorConverter: Joining features and labels")
        df = df_feat.join(df_label, on=["ICUSTAY_ID", "HOUR_IN"], how="inner")
        
        # --- ICD Preprocessing ---
        self.observer.log("INFO", "TensorConverter: Processing ICD codes and Partial Labels")
        icd_col = "ICD9_CODES"
        unique_codes_df = df.select(["ICUSTAY_ID", icd_col]).unique()
        raw_codes_map = {row["ICUSTAY_ID"]: _to_python_list(row[icd_col]) for row in unique_codes_df.iter_rows(named=True)}
        
        all_codes = []
        for codes in raw_codes_map.values(): all_codes.extend(codes)
            
        from collections import Counter
        code_counts = Counter(all_codes)
        top_k = self.icd_cfg.top_k_labels
        
        if top_k and len(code_counts) > top_k:
            keep_codes = set([c for c, _ in code_counts.most_common(top_k)])
        else:
            keep_codes = set(code_counts.keys())
            
        le = LabelEncoder()
        le.fit(list(keep_codes) + ["__OTHER__", "__MISSING__"])
        class_set = set(le.classes_)
        
        tokenizer = AutoTokenizer.from_pretrained(self.icd_cfg.model_name)
        stay_metadata = {} 
        texts = []
        stay_ids_order = []
        
        for sid, codes in raw_codes_map.items():
            raw_label = codes[0] if codes else "__MISSING__"
            cands = []
            for c in codes:
                if c in class_set: cands.append(c)
                else: cands.append("__OTHER__")
            
            if not cands: cands = ["__MISSING__"]
            unique_cands = list(set(cands))
            target_str = raw_label if raw_label in class_set else "__OTHER__"
            
            target_idx = int(le.transform([target_str])[0])
            cand_indices = le.transform(unique_cands).tolist()
            
            texts.append(" ".join(codes))
            stay_ids_order.append(sid)
            stay_metadata[sid] = {"target": target_idx, "candidates": cand_indices}

        encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=self.icd_cfg.max_length, return_tensors="np")
        for idx, sid in enumerate(stay_ids_order):
            stay_metadata[sid]["input_ids"] = encodings["input_ids"][idx]
            stay_metadata[sid]["attention_mask"] = encodings["attention_mask"][idx]

        max_cands_len = max(len(m["candidates"]) for m in stay_metadata.values())
        
        # --- Feature Selection & Encoding ---
        id_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"]
        exclude = self.cnn_cfg.data.exclude_cols + ["VENT_CLASS", "ICD9_CODES", "LABEL"]
        
        numeric_cols = []
        categorical_cols = []
        for name, dtype in zip(df.columns, df.dtypes):
            if name in id_cols or name in exclude: continue
            if dtype in pl.NUMERIC_DTYPES or dtype == pl.Boolean: numeric_cols.append(name)
            elif dtype == pl.Utf8: categorical_cols.append(name)

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

        cache_dir = Path(self.cnn_cfg.data.data_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / "stats.json", "w") as f:
            json.dump({"icd_classes": list(le.classes_)}, f, indent=2)

        prefix = self.etl_cfg.artifacts.output_h5_prefix
        
        self._process_split(df.filter(pl.col("ICUSTAY_ID").is_in(train_ids)), "train", numeric_cols, categorical_cols, cache_dir, prefix, stay_metadata, max_cands_len)
        self._process_split(df.filter(pl.col("ICUSTAY_ID").is_in(val_ids)), "val", numeric_cols, categorical_cols, cache_dir, prefix, stay_metadata, max_cands_len)
        self._process_split(df.filter(pl.col("ICUSTAY_ID").is_in(test_ids)), "test", numeric_cols, categorical_cols, cache_dir, prefix, stay_metadata, max_cands_len)

    def _flush_batch(self, h5_datasets, buffers):
        batch_len = len(buffers["y"])
        if batch_len == 0: return

        current_size = h5_datasets["y"].shape[0]
        for key in h5_datasets:
            h5_datasets[key].resize(current_size + batch_len, axis=0)

        h5_datasets["X_num"][current_size:] = np.concatenate(buffers["X_num"], axis=0)
        h5_datasets["X_cat"][current_size:] = np.concatenate(buffers["X_cat"], axis=0)
        h5_datasets["y"][current_size:] = np.concatenate(buffers["y"], axis=0)
        h5_datasets["y_vent"][current_size:] = np.concatenate(buffers["y_vent"], axis=0)
        h5_datasets["sid"][current_size:] = np.concatenate(buffers["sid"], axis=0)
        h5_datasets["hour"][current_size:] = np.concatenate(buffers["hour"], axis=0)
        
        for k in buffers: buffers[k].clear()

    def _process_split(self, df: pl.DataFrame, split_name: str, num_cols: List[str], cat_cols: List[str], cache_dir: Path, prefix: str, stay_metadata: Dict, max_cands: int) -> None:
        h5_path = cache_dir / f"{prefix}_{split_name}.h5"
        self.observer.log("INFO", f"TensorConverter: Writing {split_name} split to {h5_path}")

        df = df.sort(by=["ICUSTAY_ID", "HOUR_IN"])
        seq_len = self.cnn_cfg.seq_len
        bert_max_len = self.icd_cfg.max_length

        # 1. Instance Normalization
        norm_exprs = []
        for col in num_cols:
            mean_expr = pl.col(col).mean().over("ICUSTAY_ID")
            std_expr = pl.col(col).std().over("ICUSTAY_ID").fill_null(1.0)
            norm = (pl.col(col) - mean_expr) / (std_expr + 1e-6)
            norm_exprs.append(norm.alias(col))
        df = df.with_columns(norm_exprs)
        
        df = df.rename({"LABEL": "y_vent"})
        
        # 2. Extract Columns (Zero-copy / Fast Slicing Prep)
        all_sids = df["ICUSTAY_ID"].to_numpy()
        all_hours = df["HOUR_IN"].to_numpy()
        all_y_vent = df["y_vent"].to_numpy()
        
        all_x_num = df.select(num_cols).to_numpy().astype(np.float16)
        all_x_cat = df.select(cat_cols).to_numpy().astype(np.int32) if cat_cols else None
        
        unique_sids, start_indices = np.unique(all_sids, return_index=True)
        split_indices = np.append(start_indices, len(all_sids))

        with h5py.File(h5_path, "w") as f:
            n_num = len(num_cols)
            n_cat = len(cat_cols)
            
            # --- Dynamic Datasets (Time-Series) ---
            ds_dynamic = {
                "X_num": f.create_dataset("X_num", (0, n_num, seq_len), maxshape=(None, n_num, seq_len), dtype=np.float16, compression=hdf5plugin.Blosc()),
                "X_cat": f.create_dataset("X_cat", (0, n_cat, seq_len), maxshape=(None, n_cat, seq_len), dtype=np.int32, compression=hdf5plugin.Blosc()),
                "y": f.create_dataset("y", (0,), maxshape=(None,), dtype=np.int64),
                "y_vent": f.create_dataset("y_vent", (0,), maxshape=(None,), dtype=np.int64),
                "sid": f.create_dataset("sid", (0,), maxshape=(None,), dtype=np.int64),
                "hour": f.create_dataset("hour", (0,), maxshape=(None,), dtype=np.int32),
            }
            
            # --- Static Datasets (Patient-Level) ---
            # Stored separately to avoid massive duplication
            f.create_dataset("static_sids", data=unique_sids, dtype=np.int64)
            
            # Prepare static arrays
            static_input_ids = []
            static_attn_mask = []
            static_candidates = []
            valid_sids_mask = [] # To filter out SIDs that might not be in metadata

            for sid in unique_sids:
                if sid in stay_metadata:
                    meta = stay_metadata[sid]
                    static_input_ids.append(meta["input_ids"])
                    static_attn_mask.append(meta["attention_mask"])
                    
                    cands = np.array(meta["candidates"], dtype=np.int32)
                    pad_cands = np.full(max_cands, -1, dtype=np.int32)
                    pad_cands[:len(cands)] = cands
                    static_candidates.append(pad_cands)
                    valid_sids_mask.append(True)
                else:
                    # Should not happen typically, but fill defaults
                    static_input_ids.append(np.zeros(bert_max_len, dtype=np.int32))
                    static_attn_mask.append(np.zeros(bert_max_len, dtype=np.int32))
                    static_candidates.append(np.full(max_cands, -1, dtype=np.int32))
                    valid_sids_mask.append(False)

            f.create_dataset("static_input_ids", data=np.array(static_input_ids, dtype=np.int32), compression=hdf5plugin.Blosc())
            f.create_dataset("static_attention_mask", data=np.array(static_attn_mask, dtype=np.int32), compression=hdf5plugin.Blosc())
            f.create_dataset("static_candidates", data=np.array(static_candidates, dtype=np.int32))

            buffers = {k: [] for k in ds_dynamic.keys()}
            self.observer.log("INFO", f"TensorConverter: Vectorized processing for {len(unique_sids)} patients...")

            # 3. Vectorized Sliding Window Loop
            for i in range(len(unique_sids)):
                sid = unique_sids[i]
                if sid not in stay_metadata: continue
                
                start, end = split_indices[i], split_indices[i+1]
                p_x_num = all_x_num[start:end]
                p_x_cat = all_x_cat[start:end] if all_x_cat is not None else None
                p_y_vent = all_y_vent[start:end]
                p_hours = all_hours[start:end]
                
                seq_len_stay = len(p_y_vent)
                
                # Create Windows (Batch, Features, Time)
                pad_width = ((seq_len - 1, 0), (0, 0))
                
                padded_x_num = np.pad(p_x_num, pad_width, constant_values=np.nan)
                # (Time, Features, SeqLen) -> Transpose to (Time, Channels, SeqLen)
                windows_num = sliding_window_view(padded_x_num, window_shape=seq_len, axis=0).transpose(0, 2, 1)
                
                if p_x_cat is not None:
                    padded_x_cat = np.pad(p_x_cat, pad_width, constant_values=0)
                    windows_cat = sliding_window_view(padded_x_cat, window_shape=seq_len, axis=0).transpose(0, 2, 1)
                else:
                    windows_cat = np.zeros((seq_len_stay, n_cat, seq_len), dtype=np.int32)

                t_y = np.full(seq_len_stay, stay_metadata[sid]["target"], dtype=np.int64)
                t_sid = np.full(seq_len_stay, sid, dtype=np.int64)

                buffers["X_num"].append(windows_num)
                buffers["X_cat"].append(windows_cat)
                buffers["y"].append(t_y)
                buffers["y_vent"].append(p_y_vent)
                buffers["sid"].append(t_sid)
                buffers["hour"].append(p_hours)

                if len(buffers["y"]) >= 500: # Flush larger batches
                    self._flush_batch(ds_dynamic, buffers)

            self._flush_batch(ds_dynamic, buffers)
            self.observer.log("INFO", f"TensorConverter: Finished processing {split_name}")