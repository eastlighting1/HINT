import random

import numpy as np
import torch
from loguru import logger

from src.test.utils.seeding import set_global_seed


def test_set_global_seed_creates_deterministic_streams() -> None:
    """
    [One-line Summary] Verify set_global_seed produces deterministic random streams.

    [Description]
    Seed Python, NumPy, and Torch RNGs with a fixed value twice and assert the sampled
    values match across runs to confirm the utility synchronizes all backends.

    Test Case ID: TEST-INF-SEED-02
    Scenario: Invoke set_global_seed with the same seed before drawing random numbers.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_set_global_seed_creates_deterministic_streams")

    set_global_seed(123)
    first_python = random.random()
    first_numpy = float(np.random.rand())
    first_torch = torch.rand(1).item()

    set_global_seed(123)
    assert random.random() == first_python
    assert float(np.random.rand()) == first_numpy
    assert torch.rand(1).item() == first_torch

    logger.info("Global seed utility verified for deterministic outputs.")
