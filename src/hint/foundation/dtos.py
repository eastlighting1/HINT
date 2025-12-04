from dataclasses import dataclass
import torch
from typing import List

@dataclass
class TensorBatch:
    """
    Data Transfer Object holding a batch of tensors and metadata.

    This container ensures that numeric inputs, categorical inputs, targets,
    and associated metadata travel together through the system.

    Args:
        x_num: Numeric features tensor of shape (B, F_num, T).
        x_cat: Categorical features tensor of shape (B, F_cat, T).
        targets: Target labels tensor of shape (B,).
        stay_ids: List of unique stay identifiers corresponding to the batch.
    """
    x_num: torch.Tensor
    x_cat: torch.Tensor
    targets: torch.Tensor
    stay_ids: List[int]

    def to(self, device: str) -> "TensorBatch":
        """
        Move all tensors in the batch to the specified device.

        Args:
            device: Target device string (e.g., 'cuda', 'cpu').

        Returns:
            A new TensorBatch instance with tensors on the target device.
        """
        return TensorBatch(
            x_num=self.x_num.to(device),
            x_cat=self.x_cat.to(device),
            targets=self.targets.to(device),
            stay_ids=self.stay_ids
        )

    def __len__(self) -> int:
        """
        Return the batch size.

        Returns:
            Number of samples in the batch.
        """
        return self.x_num.size(0)