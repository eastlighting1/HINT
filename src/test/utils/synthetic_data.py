import torch
import polars as pl
import numpy as np
from hint.foundation.dtos import TensorBatch

class SyntheticDataGenerator:
    """
    Helper class to generate synthetic data for testing purposes.
    """

    @staticmethod
    def create_vitals_labs_parquet(num_rows: int = 100) -> pl.DataFrame:
        """
        Generate a synthetic vitals/labs DataFrame.

        Args:
            num_rows: Number of rows to generate.

        Returns:
            Polars DataFrame containing synthetic vital signs and lab results.
        """
        return pl.DataFrame({
            "SUBJECT_ID": np.random.randint(100, 200, num_rows),
            "HADM_ID": np.random.randint(1000, 2000, num_rows),
            "ICUSTAY_ID": np.random.randint(5000, 6000, num_rows),
            "HOURS_IN": np.random.randint(0, 100, num_rows),
            "LABEL": np.random.choice(["Heart Rate", "Glucose", "pH", "Weight"], num_rows),
            "VALUENUM": np.random.rand(num_rows) * 100,
            "MEAN": np.random.rand(num_rows) * 100
        })

    @staticmethod
    def create_tensor_batch(batch_size: int = 4, seq_len: int = 10, n_num: int = 5, n_cat: int = 2) -> TensorBatch:
        """
        Generate a synthetic TensorBatch object.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            n_num: Number of numerical features.
            n_cat: Number of categorical features.

        Returns:
            Populated TensorBatch object.
        """
        return TensorBatch(
            x_num=torch.randn(batch_size, n_num, seq_len),
            x_cat=torch.randint(0, 5, (batch_size, n_cat, seq_len)),
            y=torch.randint(0, 4, (batch_size,)),
            mask=torch.ones(batch_size, seq_len)
        )

    @staticmethod
    def create_patients_parquet(num_patients: int = 10) -> pl.DataFrame:
        """
        Generate a synthetic patients DataFrame.

        Args:
            num_patients: Number of patients to generate.

        Returns:
            Polars DataFrame containing patient demographics.
        """
        return pl.DataFrame({
            "SUBJECT_ID": np.arange(num_patients),
            "HADM_ID": np.arange(num_patients) + 100,
            "ICUSTAY_ID": np.arange(num_patients) + 1000,
            "INTIME": [f"2150-01-{i+1:02d} 10:00:00" for i in range(num_patients)],
            "AGE": np.random.randint(20, 90, num_patients),
            "GENDER": np.random.choice(["M", "F"], num_patients),
            "ETHNICITY": ["WHITE"] * num_patients,
            "ADMISSION_TYPE": ["EMERGENCY"] * num_patients,
            "INSURANCE": ["Medicare"] * num_patients,
            "STAY_HOURS": np.random.randint(24, 200, num_patients)
        })