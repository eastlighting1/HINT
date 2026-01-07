import polars as pl
import numpy as np
import h5py
from pathlib import Path
from typing import List
from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver
from ....domain.vo import ETLConfig, ICDConfig, CNNConfig

class TensorConverter(PipelineComponent):
    """Convert parquet features into HDF5 tensor datasets.

    Attributes:
        etl_cfg (ETLConfig): ETL configuration.
        cnn_cfg (CNNConfig): CNN configuration.
        icd_cfg (ICDConfig): ICD configuration.
        registry (Registry): Artifact registry.
        observer (TelemetryObserver): Logging observer.
    """

    def __init__(self, etl_config: ETLConfig, cnn_config: CNNConfig, icd_config: ICDConfig, registry: Registry, observer: TelemetryObserver):
        """Initialize the tensor converter.

        Args:
            etl_config (ETLConfig): ETL configuration.
            cnn_config (CNNConfig): CNN configuration.
            icd_config (ICDConfig): ICD configuration.
            registry (Registry): Artifact registry.
            observer (TelemetryObserver): Logging observer.
        """
        self.etl_cfg = etl_config
        self.cnn_cfg = cnn_config
        self.icd_cfg = icd_config
        self.registry = registry
        self.observer = observer

    def execute(self) -> None:
        """Run the tensor conversion workflow."""
        proc_dir = Path(self.etl_cfg.proc_dir)
        cache_dir = Path(self.icd_cfg.data.data_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        features_path = proc_dir / self.etl_cfg.artifacts.features_file
        
        icd_targets_path = proc_dir / self.etl_cfg.artifacts.icd_targets_file
        vent_targets_path = proc_dir / self.etl_cfg.artifacts.vent_targets_file

        if not features_path.exists():
            raise FileNotFoundError(f"Missing features artifact: {features_path}")
        
        self.observer.log("INFO", f"TensorConverter: Stage 1/4 Loading features from {features_path}")
        
        df = pl.read_parquet(features_path)
        
        cols = df.columns
        numeric_cols = [c for c in cols if c.startswith("V__")]
        categorical_cols = [c for c in cols if c.startswith("S__")]
        
        self.observer.log("INFO", f"TensorConverter: Found {len(numeric_cols)} dynamic features and {len(categorical_cols)} static features.")

        self.observer.log("INFO", "TensorConverter: Stage 2/4 Joining Targets")
        
        if icd_targets_path.exists():
            self.observer.log("INFO", f"TensorConverter: Loading ICD targets from {icd_targets_path.name}")
            icd_df = pl.read_parquet(icd_targets_path)
            
            if "target_icd" in icd_df.columns:
                icd_df = icd_df.rename({"target_icd": "y"})
            
            df = df.join(icd_df, on="ICUSTAY_ID", how="left")
        else:
            self.observer.log("WARNING", f"TensorConverter: ICD targets not found at {icd_targets_path}. 'y' will be missing.")

        if vent_targets_path.exists():
            self.observer.log("INFO", f"TensorConverter: Loading Vent targets from {vent_targets_path.name}")
            vent_df = pl.read_parquet(vent_targets_path)
            
            if "target_vent" in vent_df.columns:
                vent_df = vent_df.rename({"target_vent": "y_vent"})
            
            join_keys = ["ICUSTAY_ID", "HOUR_IN"]
            vent_df = vent_df.select(join_keys + ["y_vent"])
            
            df = df.join(vent_df, on=join_keys, how="left")

        self.observer.log("INFO", "TensorConverter: Stage 3/4 Splitting dataset")
        
        subjects = df.select("SUBJECT_ID").unique()
        n_subjects = subjects.height
        n_train = int(n_subjects * 0.7)
        n_val = int(n_subjects * 0.1)
        
        subjects = subjects.sample(fraction=1.0, shuffle=True, seed=42)
        
        train_ids = subjects[:n_train]["SUBJECT_ID"]
        val_ids = subjects[n_train:n_train+n_val]["SUBJECT_ID"]
        test_ids = subjects[n_train+n_val:]["SUBJECT_ID"]
        
        prefix = self.icd_cfg.data.input_h5_prefix 
        max_cands_len = 50 

        self.observer.log("INFO", f"TensorConverter: Stage 4/4 Writing HDF5 splits with prefix '{prefix}'")
        
        self._process_split(df.filter(pl.col("SUBJECT_ID").is_in(train_ids)), "train", numeric_cols, categorical_cols, cache_dir, prefix, max_cands_len)
        self.observer.log("INFO", f"TensorConverter: Writing val split to {cache_dir / f'{prefix}_val.h5'}")
        self._process_split(df.filter(pl.col("SUBJECT_ID").is_in(val_ids)), "val", numeric_cols, categorical_cols, cache_dir, prefix, max_cands_len)
        self.observer.log("INFO", f"TensorConverter: Writing test split to {cache_dir / f'{prefix}_test.h5'}")
        self._process_split(df.filter(pl.col("SUBJECT_ID").is_in(test_ids)), "test", numeric_cols, categorical_cols, cache_dir, prefix, max_cands_len)

    def _process_split(self, split_df: pl.DataFrame, split_name: str, num_cols: List[str], cat_cols: List[str], cache_dir: Path, prefix: str, max_cands: int):
        """Process a dataset split and write HDF5 output.

        Args:
            split_df (pl.DataFrame): Input dataframe for the split.
            split_name (str): Split identifier.
            num_cols (List[str]): Numeric feature columns.
            cat_cols (List[str]): Categorical feature columns.
            cache_dir (Path): Output directory for HDF5 files.
            prefix (str): Output file prefix.
            max_cands (int): Maximum candidate list length.
        """
        if split_df.height == 0:
            self.observer.log("WARNING", f"TensorConverter: Split {split_name} is empty. Skipping.")
            return

        if num_cols:
            self.observer.log("INFO", f"TensorConverter: Applying Instance Normalization for split {split_name}")
            norm_exprs = []
            for col in num_cols:
                mean_expr = pl.col(col).mean().over("ICUSTAY_ID")
                std_expr = pl.col(col).std().over("ICUSTAY_ID").fill_null(1.0)
                
                norm = (pl.col(col) - mean_expr) / (std_expr + 1e-6)
                norm_exprs.append(norm.alias(col))
            
            split_df = split_df.with_columns(norm_exprs)

        n_samples = split_df.select("ICUSTAY_ID").n_unique()
        seq_len = self.cnn_cfg.seq_len
        n_features = len(num_cols)
        
        out_path = cache_dir / f"{prefix}_{split_name}.h5"
        
        with h5py.File(out_path, "w") as f:
            f.create_dataset("X_num", (n_samples, n_features, seq_len), dtype='f4')
            
            if cat_cols:
                f.create_dataset("X_cat", (n_samples, len(cat_cols)), dtype='i4')
            
        self.observer.log("INFO", f"TensorConverter: Vectorized processing for {n_samples} patients...")
        
        sorted_df = split_df.sort(["ICUSTAY_ID", "HOUR_IN"])
        unique_stays = sorted_df.select("ICUSTAY_ID").unique(maintain_order=True)
        stay_ids = unique_stays["ICUSTAY_ID"].to_list()
        
        agg_exprs = []
        if "y" in sorted_df.columns:
            agg_exprs.append(pl.col("y").first())
        
        if cat_cols:
            agg_exprs.extend([pl.col(c).first() for c in cat_cols])
            
        grouped = sorted_df.group_by("ICUSTAY_ID", maintain_order=True).agg(agg_exprs)
        
        X_num = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
        y_vent = np.zeros((n_samples, seq_len), dtype=np.float32)
        y_vaso = np.zeros((n_samples, seq_len), dtype=np.float32)
        
        valid_window = sorted_df.filter((pl.col("HOUR_IN") >= 0) & (pl.col("HOUR_IN") < seq_len))
        
        id_map = {sid: i for i, sid in enumerate(stay_ids)}
        
        stay_col = valid_window["ICUSTAY_ID"].to_numpy()
        hour_col = valid_window["HOUR_IN"].to_numpy()
        
        row_indices = np.array([id_map[s] for s in stay_col])
        col_indices = hour_col
        
        for i, col_name in enumerate(num_cols):
            vals = valid_window[col_name].to_numpy().astype(np.float32)
            X_num[row_indices, col_indices, i] = vals
            
        vent_col = "y_vent" if "y_vent" in valid_window.columns else "VENT"
        if vent_col in valid_window.columns:
            vals = valid_window[vent_col].to_numpy().astype(np.float32)
            y_vent[row_indices, col_indices] = vals
            
        if "VASO" in valid_window.columns:
            vals = valid_window["VASO"].to_numpy().astype(np.float32)
            y_vaso[row_indices, col_indices] = vals
            
        X_num_transposed = np.transpose(X_num, (0, 2, 1))
        
        with h5py.File(out_path, "a") as f:
            f["X_num"][:] = X_num_transposed
            
            if cat_cols:
                x_cat_data = grouped.select(cat_cols).to_numpy().astype(np.int32)
                f["X_cat"][:] = x_cat_data
            
            if vent_col in valid_window.columns:
                f.create_dataset("y_vent", data=y_vent)
            if "VASO" in valid_window.columns:
                f.create_dataset("y_vaso", data=y_vaso)
                
            if "y" in grouped.columns:
                try:
                    y_list = grouped["y"].to_list()
                    
                    icd_targets = np.array(y_list)
                    
                    if icd_targets.dtype == object:
                        try:
                            icd_targets = np.vstack(y_list).astype(np.float32)
                            f.create_dataset("y", data=icd_targets)
                        except:
                            self.observer.log("WARNING", "ICD targets are ragged/invalid. Skipping 'y'.")
                    elif np.issubdtype(icd_targets.dtype, np.integer):
                        f.create_dataset("y", data=icd_targets.astype(np.int64), dtype='i8')
                    else:
                        f.create_dataset("y", data=icd_targets.astype(np.float32))
                        
                except Exception as e:
                    self.observer.log("WARNING", f"Could not vectorize ICD targets: {e}")
            
            f.create_dataset("stay_ids", data=np.array(stay_ids, dtype=np.int64))
