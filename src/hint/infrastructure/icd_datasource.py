import polars as pl
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from ..foundation.configs import ICDConfig

class CCPDDataset(Dataset):
    """
    Dataset class for ICD coding task (Text + Numeric).
    
    Args:
        df: Polars DataFrame containing data.
        feats: List of numeric feature names.
        label_col: Name of label column.
        list_col: Name of ICD code list column.
        cand_col: Name of candidate indices column.
    """
    def __init__(self, df: pl.DataFrame, feats: List[str], label_col: str, list_col: str, cand_col: str = "candidate_indices"):
        self.X = df.select(feats).to_numpy().astype(np.float32)
        self.y = df.select(label_col).to_numpy().flatten().astype(np.int64)
        self.lst = df.select(list_col).to_series().to_list()
        
        if cand_col in df.columns:
            self.cand = df.select(cand_col).to_series().to_list()
        else:
            self.cand = [[int(l)] for l in self.y]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "num": torch.tensor(self.X[idx], dtype=torch.float32),
            "lab": torch.tensor(self.y[idx], dtype=torch.long),
            "lst": self.lst[idx],
            "cand": self.cand[idx]
        }

class ICDDataModule:
    """
    Data module handling loading, preprocessing, splitting, and tokenization for ICD data.
    
    Args:
        config: ICD configuration.
        data_path: Path to the dataset file.
    """
    def __init__(self, config: ICDConfig, data_path: str):
        self.cfg = config
        self.data_path = Path(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.le = LabelEncoder()
        self.feats: List[str] = []
        self.label_col = "target_label"

    def prepare(self) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
        """
        Prepare DataLoaders for train, val, test.

        Returns:
            Train loader, Val loader, Test loader, and LabelEncoder classes.
        """
        df = pl.read_parquet(self.data_path)
        
        # Parse ICD codes if needed
        if df.schema["ICD9_CODES"] == pl.Utf8:
             pass

        # Label Encoding
        df = df.with_columns(pl.col("ICD9_CODES").list.first().fill_null("__MISSING__").alias("raw_label"))
        
        # Top-K Filtering
        if self.cfg.top_k_labels:
            top_counts = df["raw_label"].value_counts().head(self.cfg.top_k_labels)
            keep_labels = set(top_counts["raw_label"].to_list())
            df = df.with_columns(
                pl.when(pl.col("raw_label").is_in(keep_labels))
                .then(pl.col("raw_label"))
                .otherwise(pl.lit("__OTHER__"))
                .alias("filtered_label")
            )
            target_col = "filtered_label"
        else:
            target_col = "raw_label"

        # Fit Label Encoder
        targets = df[target_col].to_numpy()
        df = df.with_columns(pl.lit(self.le.fit_transform(targets)).alias(self.label_col))
        
        # Feature Selection
        exclude = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN", "ICD9_CODES", self.label_col, "raw_label", "filtered_label"]
        self.feats = [c for c in df.columns if c not in exclude and df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]

        # Filling NaNs
        df = df.with_columns([pl.col(f).fill_null(0.0) for f in self.feats])

        # Splitting
        # Simplified split logic for demonstration
        train_df = df.sample(fraction=1.0 - self.cfg.test_split - self.cfg.val_split, seed=42)
        temp_df = df.filter(~pl.col("HADM_ID").is_in(train_df["HADM_ID"]))
        val_df = temp_df.sample(fraction=0.5, seed=42)
        test_df = temp_df.filter(~pl.col("HADM_ID").is_in(val_df["HADM_ID"]))
        
        train_ds = CCPDDataset(train_df, self.feats, self.label_col, "ICD9_CODES")
        val_ds = CCPDDataset(val_df, self.feats, self.label_col, "ICD9_CODES")
        test_ds = CCPDDataset(test_df, self.feats, self.label_col, "ICD9_CODES")
        
        train_dl = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)
        val_dl = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)
        test_dl = DataLoader(test_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)
        
        return train_dl, val_dl, test_dl, self.le.classes_