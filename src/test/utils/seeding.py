import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """
    [One-line Summary] Set global random seeds for deterministic test runs.

    [Description]
    Initialize random number generators for Python, NumPy, and PyTorch (CPU and CUDA)
    to guarantee reproducible outcomes when tests rely on randomness.

    Test Case ID: TEST-INF-SEED-01
    Scenario: Initialize seeds once before running the test session.

    Args:
        seed: Integer seed applied consistently across RNG backends.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
