import h5py
import hdf5plugin
import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Any
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import CNNConfig, ETLConfig, ICDConfig


def _to_python_list(val: Any) -> List[str]:
    """Helper to ensure value is a python list of strings."""
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

    Builds base datasets for train, validation, and test splits with numeric, categorical sequences,
    and tokenized ICD inputs for partial label learning.

    Attributes:
        etl_cfg (ETLConfig): ETL configuration.
        cnn_cfg (CNNConfig): CNN configuration (seq_len, etc).
        icd_cfg (ICDConfig): ICD configuration (tokenizer, max_length).
        registry (Registry): Artifact registry.
        observer (TelemetryObserver): Telemetry adapter.
    """

    def __init__(self, etl_config: ETLConfig, cnn_config: CNNConfig, icd_config: ICDConfig, registry: Registry, observer: TelemetryObserver):
        self.etl_cfg = etl_config
        self.cnn_cfg = cnn_config
        self.icd_cfg = icd_config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        """Generate HDF5 tensors for all dataset splits with ICD pre-processing."""
        self.observer.log("INFO", "TensorConverter: Preparing processed sources for tensorization")
        proc_dir = Path(self.etl_cfg.proc_dir)
        if not proc_dir.is_absolute():
            proc_dir = Path.cwd() / proc_dir

        feat_path = proc_dir / self.etl_cfg.artifacts.features_file
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature file not found at: {feat_path}")

        label_path = proc_dir / self.etl_cfg.artifacts.labels_file
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found at: {label_path}")

        self.observer.log("INFO", f"TensorConverter: Loading features from {feat_path}")
        df_feat = pl.read_parquet(feat_path)

        self.observer.log("INFO", f"TensorConverter: Loading labels from {label_path}")
        df_label = pl.read_parquet(label_path)

        self.observer.log("INFO", "TensorConverter: Joining features and labels")
        df = df_feat.join(df_label, on=["ICUSTAY_ID", "HOUR_IN"], how="inner")
        
        # --- ICD Preprocessing Start ---
        self.observer.log("INFO", "TensorConverter: Processing ICD codes and Partial Labels")
        
        # 1. Parse Codes & Frequency Filtering
        icd_col = "ICD9_CODES"
        if icd_col not in df.columns:
             raise KeyError(f"{icd_col} not found in dataset. Ensure it is not dropped earlier.")

        # Optimize: distinct stay-code mapping
        unique_codes_df = df.select(["ICUSTAY_ID", icd_col]).unique()
        raw_codes_map = {row["ICUSTAY_ID"]: _to_python_list(row[icd_col]) for row in unique_codes_df.iter_rows(named=True)}
        
        all_codes = []
        for codes in raw_codes_map.values():
            all_codes.extend(codes)
            
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
        has_other = "__OTHER__" in class_set
        
        # 2. Candidate Generation & Tokenization
        tokenizer = AutoTokenizer.from_pretrained(self.icd_cfg.model_name)
        
        stay_metadata = {} # icustay_id -> {input_ids, attention_mask, target, candidates}
        
        texts = []
        stay_ids_order = []
        
        for sid, codes in raw_codes_map.items():
            # Label Logic
            raw_label = codes[0] if codes else "__MISSING__"
            
            cands = []
            for c in codes:
                if c in class_set:
                    cands.append(c)
                elif has_other:
                    cands.append("__OTHER__")
            
            if not cands:
                cands = [raw_label if raw_label in class_set else "__OTHER__"]
                
            unique_cands = list(set(cands))
            # Fallback for target if not in keep
            target_str = raw_label if raw_label in class_set else "__OTHER__"
            
            target_idx = int(le.transform([target_str])[0])
            cand_indices = le.transform(unique_cands).tolist()
            
            # Text Logic
            text = " ".join(codes)
            texts.append(text)
            stay_ids_order.append(sid)
            
            stay_metadata[sid] = {
                "target": target_idx,
                "candidates": cand_indices
            }

        # Batch Tokenization
        encodings = tokenizer(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=self.icd_cfg.max_length, 
            return_tensors="np"
        )
        
        for idx, sid in enumerate(stay_ids_order):
            stay_metadata[sid]["input_ids"] = encodings["input_ids"][idx]
            stay_metadata[sid]["attention_mask"] = encodings["attention_mask"][idx]

        max_cands_len = max(len(m["candidates"]) for m in stay_metadata.values())
        self.observer.log("INFO", f"TensorConverter: Max candidates per sample: {max_cands_len}")
        # --- ICD Preprocessing End ---

        id_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN"]
        exclude = self.cnn_cfg.data.exclude_cols + ["VENT_CLASS", "ICD9_CODES", "LABEL"]

        numeric_cols = []
        categorical_cols = []

        for name, dtype in zip(df.columns, df.dtypes):
            if name in id_cols or name in exclude:
                continue
            if dtype in pl.NUMERIC_DTYPES or dtype == pl.Boolean:
                numeric_cols.append(name)
            elif dtype == pl.Utf8:
                categorical_cols.append(name)

        vocab_info: Dict[str, int] = {}
        categorical_idx_cols: List[str] = []
        if categorical_cols:
            encoding_exprs = []
            for col in categorical_cols:
                idx_name = f"{col}_IDX"
                categorical_idx_cols.append(idx_name)
                df = df.with_columns(pl.col(col).fill_null("__NULL__").cast(pl.Categorical))
                vocab_size = int(df.select(pl.col(col).n_unique()).item() + 1)
                vocab_info[col] = vocab_size
                encoding_exprs.append((pl.col(col).to_physical() + 1).cast(pl.Int32).alias(idx_name))
            df = df.with_columns(encoding_exprs)

        feat_names_num = [f"{c}_VAL" for c in numeric_cols] + [f"{c}_MSK" for c in numeric_cols] + [f"{c}_DT" for c in numeric_cols]

        unique_stays = df.select("ICUSTAY_ID").unique().to_numpy().ravel()
        
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, teva_idx = next(gss1.split(unique_stays, groups=unique_stays))
        train_ids = unique_stays[train_idx]
        teva_ids = unique_stays[teva_idx]

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        test_idx, val_idx = next(gss2.split(teva_ids, groups=teva_ids))
        val_ids = teva_ids[val_idx]
        test_ids = teva_ids[test_idx]

        df_tr = df.filter(pl.col("ICUSTAY_ID").is_in(train_ids))
        
        # Stats computation
        stats: Dict[str, float] = {}
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
                except Exception:
                    stats[k] = 1.0 if k.endswith("_std") else 0.0
                    continue
                if k.endswith("_std") and (val == 0.0 or np.isnan(val)):
                    stats[k] = 1.0
                elif np.isnan(val):
                    stats[k] = 0.0
                else:
                    stats[k] = val

        # Save stats and ICD classes
        cache_dir = Path(self.cnn_cfg.data.data_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        stats["icd_classes"] = list(le.classes_)
        stats["max_candidates"] = max_cands_len
        
        with open(cache_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        prefix = self.etl_cfg.artifacts.output_h5_prefix
        
        # Helper to pass metadata
        self._process_split(df_tr, "train", numeric_cols, feat_names_num, categorical_idx_cols, stats, cache_dir, prefix, stay_metadata, max_cands_len)
        self._process_split(df.filter(pl.col("ICUSTAY_ID").is_in(val_ids)), "val", numeric_cols, feat_names_num, categorical_idx_cols, stats, cache_dir, prefix, stay_metadata, max_cands_len)
        self._process_split(df.filter(pl.col("ICUSTAY_ID").is_in(test_ids)), "test", numeric_cols, feat_names_num, categorical_idx_cols, stats, cache_dir, prefix, stay_metadata, max_cands_len)

    def _flush_batch(self, h5_datasets, buffers):
        """Append buffered windows into HDF5 datasets."""
        batch_len = len(buffers["y"])
        if batch_len == 0:
            return

        current_size = h5_datasets["y"].shape[0]
        for key in h5_datasets:
            h5_datasets[key].resize(current_size + batch_len, axis=0)

        h5_datasets["X_num"][current_size:] = np.array(buffers["X_num"], dtype=np.float16)
        h5_datasets["X_cat"][current_size:] = np.array(buffers["X_cat"], dtype=np.int32)
        h5_datasets["y"][current_size:] = np.array(buffers["y"], dtype=np.int64)
        h5_datasets["y_vent"][current_size:] = np.array(buffers["y_vent"], dtype=np.int64)
        h5_datasets["sid"][current_size:] = np.array(buffers["sid"], dtype=np.int64)
        h5_datasets["hour"][current_size:] = np.array(buffers["hour"], dtype=np.int32)
        
        # ICD Specific
        h5_datasets["input_ids"][current_size:] = np.array(buffers["input_ids"], dtype=np.int32)
        h5_datasets["attention_mask"][current_size:] = np.array(buffers["attention_mask"], dtype=np.int32)
        h5_datasets["candidates"][current_size:] = np.array(buffers["candidates"], dtype=np.int32)

        for k in buffers:
            buffers[k].clear()

    def _process_split(self, df: pl.DataFrame, split_name: str, num_cols: List[str], feat_names: List[str], cat_cols: List[str], stats: Dict[str, float], cache_dir: Path, prefix: str, stay_metadata: Dict, max_cands: int) -> None:
        """Transform one split into padded sequences and write to HDF5."""
        h5_path = cache_dir / f"{prefix}_{split_name}.h5"
        self.observer.log("INFO", f"TensorConverter: Writing {split_name} split to {h5_path}")

        df = df.sort(by=["ICUSTAY_ID", "HOUR_IN"])
        seq_len = self.cnn_cfg.seq_len
        bert_max_len = self.icd_cfg.max_length

        val_exprs = [pl.col(col).forward_fill().over("ICUSTAY_ID").alias(f"{col}_VAL") for col in num_cols]
        mask_exprs = [pl.col(col).is_not_null().cast(pl.Float32).alias(f"{col}_MSK") for col in num_cols]
        delta_exprs = []
        for col in num_cols:
            last_time = pl.when(pl.col(col).is_not_null()).then(pl.col("HOUR_IN")).otherwise(None).forward_fill().over("ICUSTAY_ID")
            normalized_delta = (pl.col("HOUR_IN") - last_time).fill_null(seq_len).clip(0, seq_len).alias(f"{col}_DT")
            delta_exprs.append((normalized_delta / seq_len))

        df = df.with_columns(val_exprs + mask_exprs + delta_exprs)
        val_cols = [f"{col}_VAL" for col in num_cols]
        df = df.with_columns([pl.col(col).fill_null(0.0) for col in val_cols])

        norm_exprs = []
        for col in num_cols:
            val_col = f"{col}_VAL"
            if col.startswith("V__"):
                mean, std = stats.get(f"{col}_mean", 0.0), stats.get(f"{col}_std", 1.0)
                norm_exprs.append(((pl.col(val_col) - mean) / (std + 1e-6)).alias(val_col))
            else:
                norm_exprs.append(pl.col(val_col))
        df = df.with_columns(norm_exprs)

        # Select LABEL as y_vent
        df = df.rename({"LABEL": "y_vent"})
        df = df.select(["ICUSTAY_ID", "HOUR_IN", "y_vent"] + list(feat_names) + list(cat_cols))

        with h5py.File(h5_path, "w") as f:
            n_feats_num = len(feat_names)
            n_cat_feats = len(cat_cols)

            datasets = {
                "X_num": f.create_dataset("X_num", (0, n_feats_num, seq_len), maxshape=(None, n_feats_num, seq_len), dtype=np.float16, compression=hdf5plugin.Blosc(cname="lz4")),
                "X_cat": f.create_dataset("X_cat", (0, n_cat_feats, seq_len), maxshape=(None, n_cat_feats, seq_len), dtype=np.int32, compression=hdf5plugin.Blosc(cname="lz4")),
                "y": f.create_dataset("y", (0,), maxshape=(None,), dtype=np.int64), # This will be ICD label
                "y_vent": f.create_dataset("y_vent", (0,), maxshape=(None,), dtype=np.int64), # Original Vent label
                "sid": f.create_dataset("sid", (0,), maxshape=(None,), dtype=np.int64),
                "hour": f.create_dataset("hour", (0,), maxshape=(None,), dtype=np.int32),
                
                # New ICD datasets
                "input_ids": f.create_dataset("input_ids", (0, bert_max_len), maxshape=(None, bert_max_len), dtype=np.int32, compression=hdf5plugin.Blosc(cname="lz4")),
                "attention_mask": f.create_dataset("attention_mask", (0, bert_max_len), maxshape=(None, bert_max_len), dtype=np.int32, compression=hdf5plugin.Blosc(cname="lz4")),
                "candidates": f.create_dataset("candidates", (0, max_cands), maxshape=(None, max_cands), dtype=np.int32),
            }

            buffers = {
                "X_num": [], "X_cat": [], "y": [], "y_vent": [], "sid": [], "hour": [],
                "input_ids": [], "attention_mask": [], "candidates": []
            }
            
            window_buffer_num = np.zeros((n_feats_num, seq_len), dtype=np.float16)
            window_buffer_cat = np.zeros((n_cat_feats, seq_len), dtype=np.int32)

            parts = df.partition_by("ICUSTAY_ID", maintain_order=True)
            for group in parts:
                pdf = group.to_pandas()
                sid = int(pdf["ICUSTAY_ID"].iloc[0])
                vent_labels = pdf["y_vent"].tolist()
                hours = pdf["HOUR_IN"].tolist()
                arr_num = pdf[feat_names].to_numpy(dtype=np.float32).T
                arr_cat = pdf[cat_cols].to_numpy(dtype=np.int32).T if cat_cols else None
                
                # Fetch ICD metadata
                meta = stay_metadata.get(sid, None)
                if meta is None: 
                    # Should not happen if data is consistent, but robust fallback
                    continue
                    
                icd_label = meta["target"]
                input_ids = meta["input_ids"]
                attn_mask = meta["attention_mask"]
                cands = np.array(meta["candidates"], dtype=np.int32)
                # Pad candidates
                padded_cands = np.full((max_cands,), -1, dtype=np.int32)
                padded_cands[:len(cands)] = cands

                for t, v_label in enumerate(vent_labels):
                    start = max(0, t - seq_len + 1)
                    end = t + 1
                    window_len = end - start

                    window_buffer_num.fill(0)
                    window_buffer_num[:, seq_len - window_len :] = arr_num[:, start:end]

                    if arr_cat is not None:
                        window_buffer_cat.fill(0)
                        window_buffer_cat[:, seq_len - window_len :] = arr_cat[:, start:end]
                        buffers["X_cat"].append(window_buffer_cat.copy())
                    else:
                        buffers["X_cat"].append(np.zeros((n_cat_feats, seq_len), dtype=np.int32))

                    buffers["X_num"].append(window_buffer_num.copy())
                    buffers["y_vent"].append(v_label)
                    buffers["y"].append(icd_label) # ICD target
                    buffers["sid"].append(sid)
                    buffers["hour"].append(hours[t])
                    
                    buffers["input_ids"].append(input_ids)
                    buffers["attention_mask"].append(attn_mask)
                    buffers["candidates"].append(padded_cands)

                    if len(buffers["sid"]) >= 4096:
                        self._flush_batch(datasets, buffers)

            self._flush_batch(datasets, buffers)