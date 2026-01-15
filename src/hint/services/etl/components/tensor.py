"""Summary of the tensor module.

Longer description of the module purpose and usage.
"""

import polars as pl

import numpy as np

import h5py

import shutil

from pathlib import Path

from typing import List, Dict, Tuple

from sklearn.model_selection import train_test_split

from ....foundation.interfaces import PipelineComponent, Registry, TelemetryObserver

from ....domain.vo import ETLConfig, ICDConfig, CNNConfig



class TensorConverter(PipelineComponent):

    """Summary of TensorConverter purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    cnn_cfg (Any): Description of cnn_cfg.
    etl_cfg (Any): Description of etl_cfg.
    icd_cfg (Any): Description of icd_cfg.
    observer (Any): Description of observer.
    registry (Any): Description of registry.
    """



    def __init__(self, etl_config: ETLConfig, cnn_config: CNNConfig, icd_config: ICDConfig, registry: Registry, observer: TelemetryObserver):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        etl_config (Any): Description of etl_config.
        cnn_config (Any): Description of cnn_config.
        icd_config (Any): Description of icd_config.
        registry (Any): Description of registry.
        observer (Any): Description of observer.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.etl_cfg = etl_config

        self.cnn_cfg = cnn_config

        self.icd_cfg = icd_config

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

        cache_dir = Path(self.icd_cfg.data.data_cache_dir)

        cache_dir.mkdir(parents=True, exist_ok=True)



        features_path = proc_dir / self.etl_cfg.artifacts.features_file

        icd_targets_path = proc_dir / self.etl_cfg.artifacts.icd_targets_file

        vent_targets_path = proc_dir / self.etl_cfg.artifacts.vent_targets_file

        icd_meta_path = proc_dir / self.etl_cfg.artifacts.icd_meta_file



        self.observer.log("INFO", f"TensorConverter: Loading features from {features_path}")

        df = pl.read_parquet(features_path)



        cols = df.columns

        val_cols = sorted([c for c in cols if c.startswith("V__")])

        msk_cols = sorted([c for c in cols if c.startswith("M__")])

        delta_cols = sorted([c for c in cols if c.startswith("D__")])



        self.observer.log("INFO", f"TensorConverter: Found {len(val_cols)} features with 3 channels (V, M, D).")



        if icd_targets_path.exists():

            icd_df = pl.read_parquet(icd_targets_path)

            df = df.join(icd_df, on="ICUSTAY_ID", how="left")



        if vent_targets_path.exists():

            vent_df = pl.read_parquet(vent_targets_path)

            df = df.join(vent_df, on=["ICUSTAY_ID", "HOUR_IN"], how="left")



        if icd_meta_path.exists():

            shutil.copy(icd_meta_path, cache_dir / "stats.json")





        self.observer.log("INFO", "TensorConverter: Preparing random split by ICU stay.")



        stay_ids = df.select("ICUSTAY_ID").unique().to_series().to_numpy()



        train_ids, temp_ids = train_test_split(

            stay_ids, test_size=0.3, shuffle=True, random_state=42

        )

        val_ids, test_ids = train_test_split(

            temp_ids, test_size=2 / 3, shuffle=True, random_state=42

        )



        self.observer.log("INFO", f"Split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")



        train_mask = pl.col("ICUSTAY_ID").is_in(list(train_ids))

        stats = df.filter(train_mask).select(val_cols).mean().to_dict(as_series=False)

        std_stats = df.filter(train_mask).select(val_cols).std().to_dict(as_series=False)



        global_stats = {c: (stats[c][0], std_stats[c][0]) for c in val_cols}



        prefixes = (self.icd_cfg.data.input_h5_prefix, self.cnn_cfg.data.input_h5_prefix)



        for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:

            sub_df = df.filter(pl.col("ICUSTAY_ID").is_in(list(ids)))

            self._process_split(sub_df, name, val_cols, msk_cols, delta_cols, cache_dir, prefixes, global_stats)



    def _process_split(self, df, name, v_cols, m_cols, d_cols, out_dir, prefixes, stats):

        """Summary of _process_split.
        
        Longer description of the _process_split behavior and usage.
        
        Args:
        df (Any): Description of df.
        name (Any): Description of name.
        v_cols (Any): Description of v_cols.
        m_cols (Any): Description of m_cols.
        d_cols (Any): Description of d_cols.
        out_dir (Any): Description of out_dir.
        prefixes (Any): Description of prefixes.
        stats (Any): Description of stats.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        if df.height == 0:

            return



        norm_exprs = []

        for c in v_cols:

            mean, std = stats.get(c, (0, 1))

            if std == 0 or std is None:

                std = 1

            norm_exprs.append(((pl.col(c) - mean) / std).alias(c))

        df = df.with_columns(norm_exprs)



        n_samples = df.select("ICUSTAY_ID").n_unique()

        seq_len = self.cnn_cfg.seq_len

        n_feats = len(v_cols)

        cat_cols = [c for c in df.columns if c.startswith("S__")]



        icd_path = out_dir / f"{prefixes[0]}_{name}.h5"

        cnn_path = out_dir / f"{prefixes[1]}_{name}.h5"



        X_val = np.zeros((n_samples, seq_len, n_feats), dtype=np.float32)

        X_msk = np.zeros((n_samples, seq_len, n_feats), dtype=np.float32)

        X_delta = np.zeros((n_samples, seq_len, n_feats), dtype=np.float32)

        y_vent = np.full((n_samples, seq_len), -100, dtype=np.int64)



        sorted_df = df.sort(["ICUSTAY_ID", "HOUR_IN"])

        unique_stays = sorted_df.select("ICUSTAY_ID").unique(maintain_order=True)

        stay_ids = unique_stays["ICUSTAY_ID"].to_list()

        id_map = {sid: i for i, sid in enumerate(stay_ids)}



        max_cands = getattr(self.icd_cfg, "top_k_labels", None) or 50
        max_cands = max(1, int(max_cands))

        y_icd_cands = np.full((n_samples, max_cands), -1, dtype=np.int64)



        cands_df = sorted_df.group_by("ICUSTAY_ID", maintain_order=True).first().select(["ICUSTAY_ID", "candidates"])

        for row in cands_df.iter_rows():

            sid, cands = row

            if cands and sid in id_map:

                idx = id_map[sid]

                trunc = cands[:max_cands]

                y_icd_cands[idx, :len(trunc)] = trunc



        valid_win = sorted_df.filter((pl.col("HOUR_IN") >= 0) & (pl.col("HOUR_IN") < seq_len))



        max_hours = valid_win.group_by("ICUSTAY_ID", maintain_order=True).agg(pl.col("HOUR_IN").max().alias("mh"))

        offset_map = {row[0]: max(0, seq_len - 1 - row[1]) for row in max_hours.iter_rows()}



        v_data = valid_win.select(["ICUSTAY_ID", "HOUR_IN"] + v_cols).to_numpy()

        m_data = valid_win.select(m_cols).to_numpy()

        d_data = valid_win.select(d_cols).to_numpy()

        vent_data = valid_win.select("target_vent").to_numpy().flatten()



        stay_col_idx = 0

        hour_col_idx = 1

        feat_start_idx = 2



        stay_vals = v_data[:, stay_col_idx].astype(int)

        hour_vals = v_data[:, hour_col_idx].astype(int)



        row_idxs = np.array([id_map[s] for s in stay_vals])

        offsets = np.array([offset_map[s] for s in stay_vals])

        col_idxs = hour_vals + offsets



        valid_mask = (col_idxs >= 0) & (col_idxs < seq_len)



        final_rows = row_idxs[valid_mask]

        final_cols = col_idxs[valid_mask]



        val_block = v_data[valid_mask, feat_start_idx:].astype(np.float32)

        val_block = np.nan_to_num(val_block, nan=0.0)

        X_val[final_rows, final_cols, :] = val_block

        X_msk[final_rows, final_cols, :] = m_data[valid_mask, :].astype(np.float32)

        X_delta[final_rows, final_cols, :] = d_data[valid_mask, :].astype(np.float32)



        vent_filled = np.nan_to_num(vent_data.astype(float), nan=-100.0).astype(np.int64)

        y_vent[final_rows, final_cols] = vent_filled[valid_mask]



        X_cat = np.zeros((n_samples, len(cat_cols)), dtype=np.int32)

        cat_df = sorted_df.group_by("ICUSTAY_ID", maintain_order=True).first().select(cat_cols)



        cat_vals = cat_df.fill_null(0).to_numpy()

        X_cat[:] = cat_vals.astype(np.int32)



        X_val = X_val.transpose(0, 2, 1)

        X_msk = X_msk.transpose(0, 2, 1)

        X_delta = X_delta.transpose(0, 2, 1)



        X_num = np.concatenate([X_val, X_msk, X_delta], axis=1)





        valid_icd_mask = (y_icd_cands != -1).any(axis=1)

        icd_stay_ids = np.array(stay_ids)[valid_icd_mask]

        icd_X_num = X_num[valid_icd_mask]

        icd_X_cat = X_cat[valid_icd_mask]

        icd_y = y_icd_cands[valid_icd_mask]









        with h5py.File(icd_path, "w") as f:

            f.create_dataset("X_num", data=icd_X_num)

            f.create_dataset("X_cat", data=icd_X_cat)

            f.create_dataset("stay_ids", data=icd_stay_ids)

            f.create_dataset("y", data=icd_y)









        with h5py.File(cnn_path, "w") as f:

            f.create_dataset("X_num", data=X_num)

            f.create_dataset("X_cat", data=X_cat)

            f.create_dataset("stay_ids", data=np.array(stay_ids))

            f.create_dataset("y", data=y_vent)

            f.create_dataset("y_vent", data=y_vent)
