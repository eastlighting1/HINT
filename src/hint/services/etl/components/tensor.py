import h5py
import hdf5plugin
import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Any
from sklearn.model_selection import GroupShuffleSplit
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import CNNConfig, ETLConfig


class TensorConverter(PipelineComponent):
    """Convert processed features and labels into HDF5 tensors.

    Builds base datasets for train, validation, and test splits with numeric and categorical sequences.

    Attributes:
        etl_cfg (ETLConfig): ETL configuration providing processed input paths.
        cnn_cfg (CNNConfig): CNN configuration with cache paths and sequence length.
        registry (Registry): Artifact registry placeholder for interface alignment.
        observer (TelemetryObserver): Telemetry adapter for detailed logging.
    """

    def __init__(self, etl_config: ETLConfig, cnn_config: CNNConfig, registry: Registry, observer: TelemetryObserver):
        self.etl_cfg = etl_config
        self.cnn_cfg = cnn_config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        """Generate HDF5 tensors for all dataset splits.

        Returns:
            None

        Raises:
            FileNotFoundError: When features or labels parquet files are missing.
        """
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

        self.observer.log("INFO", "TensorConverter: Joining features and labels on ICU stay and hour")
        df = df_feat.join(df_label, on=["ICUSTAY_ID", "HOUR_IN"], how="inner")
        self.observer.log("INFO", f"TensorConverter: Joined dataframe rows={df.height}")

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

        self.observer.log("INFO", f"TensorConverter: Numeric columns={len(numeric_cols)}, categorical columns={len(categorical_cols)}")

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
            self.observer.log("INFO", f"TensorConverter: Encoded categorical vocab sizes={vocab_info}")

        feat_names_num = [f"{c}_VAL" for c in numeric_cols] + [f"{c}_MSK" for c in numeric_cols] + [f"{c}_DT" for c in numeric_cols]

        unique_stays = df.select("ICUSTAY_ID").unique().to_numpy().ravel()
        self.observer.log("INFO", f"TensorConverter: Unique ICU stays detected={len(unique_stays)}")

        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, teva_idx = next(gss1.split(unique_stays, groups=unique_stays))
        train_ids = unique_stays[train_idx]
        teva_ids = unique_stays[teva_idx]

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        test_idx, val_idx = next(gss2.split(teva_ids, groups=teva_ids))
        val_ids = teva_ids[val_idx]
        test_ids = teva_ids[test_idx]

        self.observer.log("INFO", f"TensorConverter: Split counts train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

        df_tr = df.filter(pl.col("ICUSTAY_ID").is_in(train_ids))
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
            self.observer.log("INFO", f"TensorConverter: Computed stats for normalization keys={len(stats)}")

        cache_dir = Path(self.cnn_cfg.data.data_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        self.observer.log("INFO", f"TensorConverter: Persisted stats to {cache_dir / 'stats.json'}")

        prefix = self.etl_cfg.artifacts.output_h5_prefix
        self._process_split(df_tr, "train", numeric_cols, feat_names_num, categorical_idx_cols, stats, cache_dir, prefix)
        self._process_split(df.filter(pl.col("ICUSTAY_ID").is_in(val_ids)), "val", numeric_cols, feat_names_num, categorical_idx_cols, stats, cache_dir, prefix)
        self._process_split(df.filter(pl.col("ICUSTAY_ID").is_in(test_ids)), "test", numeric_cols, feat_names_num, categorical_idx_cols, stats, cache_dir, prefix)

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
        h5_datasets["sid"][current_size:] = np.array(buffers["sid"], dtype=np.int64)
        h5_datasets["hour"][current_size:] = np.array(buffers["hour"], dtype=np.int32)

        for k in buffers:
            buffers[k].clear()

    def _process_split(self, df: pl.DataFrame, split_name: str, num_cols: List[str], feat_names: List[str], cat_cols: List[str], stats: Dict[str, float], cache_dir: Path, prefix: str) -> None:
        """Transform one split into padded sequences and write to HDF5."""
        h5_path = cache_dir / f"{prefix}_{split_name}.h5"
        self.observer.log("INFO", f"TensorConverter: Writing {split_name} split to {h5_path}")

        df = df.sort(by=["ICUSTAY_ID", "HOUR_IN"])
        seq_len = self.cnn_cfg.seq_len

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

        df = df.select(["ICUSTAY_ID", "HOUR_IN", "LABEL"] + list(feat_names) + list(cat_cols))

        with h5py.File(h5_path, "w") as f:
            n_feats_num = len(feat_names)
            n_cat_feats = len(cat_cols)

            datasets = {
                "X_num": f.create_dataset("X_num", (0, n_feats_num, seq_len), maxshape=(None, n_feats_num, seq_len), dtype=np.float16, compression=hdf5plugin.Blosc(cname="lz4")),
                "X_cat": f.create_dataset("X_cat", (0, n_cat_feats, seq_len), maxshape=(None, n_cat_feats, seq_len), dtype=np.int32, compression=hdf5plugin.Blosc(cname="lz4")),
                "y": f.create_dataset("y", (0,), maxshape=(None,), dtype=np.int64),
                "sid": f.create_dataset("sid", (0,), maxshape=(None,), dtype=np.int64),
                "hour": f.create_dataset("hour", (0,), maxshape=(None,), dtype=np.int32),
            }

            buffers = {"X_num": [], "X_cat": [], "y": [], "sid": [], "hour": []}
            window_buffer_num = np.zeros((n_feats_num, seq_len), dtype=np.float16)
            window_buffer_cat = np.zeros((n_cat_feats, seq_len), dtype=np.int32)

            parts = df.partition_by("ICUSTAY_ID", maintain_order=True)
            for group in parts:
                pdf = group.to_pandas()
                sid = int(pdf["ICUSTAY_ID"].iloc[0])
                labels = pdf["LABEL"].tolist()
                hours = pdf["HOUR_IN"].tolist()
                arr_num = pdf[feat_names].to_numpy(dtype=np.float32).T
                arr_cat = pdf[cat_cols].to_numpy(dtype=np.int32).T if cat_cols else None

                for t, label in enumerate(labels):
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
                    buffers["y"].append(label)
                    buffers["sid"].append(sid)
                    buffers["hour"].append(hours[t])

                    if len(buffers["sid"]) >= 4096:
                        self._flush_batch(datasets, buffers)

            self._flush_batch(datasets, buffers)
