import torch
import numpy as np
import pandas as pd
from typing import Union, List, Type, ContextManager
from contextlib import contextmanager

@contextmanager
def assert_raises(expected_exception: Type[Exception]) -> ContextManager:
    """
    Context manager to assert that a block of code raises a specific exception.
    Replaces pytest.raises logic for custom runner.

    Args:
        expected_exception: The exception type expected to be raised.

    Raises:
        AssertionError: If the exception is not raised.
    """
    try:
        yield
    except expected_exception:
        return
    except Exception as e:
        raise AssertionError(f"Expected {expected_exception.__name__} but raised {type(e).__name__}: {e}")
    raise AssertionError(f"Expected {expected_exception.__name__} but no exception was raised.")

def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple) -> None:
    """
    Assert that the tensor has the expected shape.

    Args:
        tensor: Input tensor to check.
        expected_shape: Tuple representing the expected dimensions.

    Raises:
        AssertionError: If shapes do not match.
    """
    if tensor.shape != expected_shape:
        raise AssertionError(f"Expected tensor shape {expected_shape}, but got {tensor.shape}")

def assert_tensor_equal(t1: torch.Tensor, t2: torch.Tensor, atol: float = 1e-6) -> None:
    """
    Assert that two tensors are equal within a given tolerance.

    Args:
        t1: First tensor.
        t2: Second tensor.
        atol: Absolute tolerance.

    Raises:
        AssertionError: If tensors differ beyond tolerance.
    """
    if not torch.allclose(t1, t2, atol=atol):
        diff = (t1 - t2).abs().max()
        raise AssertionError(f"Tensors are not equal (atol={atol}). Max diff: {diff}")

def assert_dataframe_columns(df: Union[pd.DataFrame, 'pl.DataFrame'], expected_columns: List[str]) -> None:
    """
    Assert that the DataFrame contains all expected columns.

    Args:
        df: Pandas or Polars DataFrame.
        expected_columns: List of column names that must exist.

    Raises:
        ValueError: If input is not a DataFrame.
        AssertionError: If columns are missing.
    """
    if hasattr(df, "columns"):
        cols = df.columns
    else:
        raise ValueError("Input must be a Pandas or Polars DataFrame")
    
    missing = [c for c in expected_columns if c not in cols]
    if missing:
        raise AssertionError(f"Missing columns in DataFrame: {missing}")