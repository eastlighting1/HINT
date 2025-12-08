import h5py
import numpy as np
import polars as pl
from pathlib import Path
from loguru import logger
from hint.domain.vo import ETLConfig, CNNConfig, ICDConfig

class IntegrationFixtures:
    """
    Provides heavy-weight fixtures for integration tests, such as creating real dummy HDF5 files 
    or raw CSV structures on the file system.
    """

    @staticmethod
    def create_dummy_hdf5(path: Path, num_samples: int = 10, seq_len: int = 20, n_features: int = 5) -> None:
        """
        Creates a valid HDF5 file structure required by HDF5StreamingSource.
        
        Args:
            path: Target file path.
            num_samples: Number of samples to generate.
            seq_len: Sequence length of time-series data.
            n_features: Number of numeric features.
        """
        logger.debug(f"Creating dummy HDF5 file at {path}")
        with h5py.File(path, 'w') as f:
            x_num = np.random.randn(num_samples, seq_len, n_features).astype(np.float32)
            x_cat = np.random.randint(0, 5, (num_samples, seq_len, 2)).astype(np.int64)
            y = np.random.randint(0, 4, (num_samples,)).astype(np.int64)
            ids = np.arange(num_samples).astype(np.int64)
            
            f.create_dataset("x_num", data=x_num)
            f.create_dataset("x_cat", data=x_cat)
            f.create_dataset("y", data=y)
            f.create_dataset("ids", data=ids)

    @staticmethod
    def setup_etl_raw_files(raw_dir: Path) -> None:
        """
        Creates minimal raw CSV/Parquet files needed for ETL execution.
        
        Args:
            raw_dir: Directory to place raw files.
        """
        logger.debug(f"Populating raw directory at {raw_dir}")
        pl.DataFrame({
            "SUBJECT_ID": [101, 102],
            "HADM_ID": [1001, 1002],
            "ICD9_CODE": ["4280", "25000"]
        }).write_csv(raw_dir / "DIAGNOSES_ICD.csv")
        pl.DataFrame({
            "SUBJECT_ID": [101, 102],
            "GENDER": ["M", "F"],
            "DOB": ["2100-01-01", "2100-01-02"]
        }).write_csv(raw_dir / "PATIENTS.csv")

    @staticmethod
    def get_integrated_etl_config(tmp_path: Path) -> ETLConfig:
        """
        Returns an ETLConfig pointing to temporary directories.
        """
        return ETLConfig(
            raw_dir=str(tmp_path / "raw"),
            proc_dir=str(tmp_path / "processed"),
            resources_dir=str(tmp_path / "resources")
        )
