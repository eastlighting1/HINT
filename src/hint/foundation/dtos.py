from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from hint.domain.vo import ETLConfig, ICDConfig, CNNConfig

@dataclass
class AppContext:
    """Runtime configuration container for the application.

    Aggregates ETL, ICD, and CNN configuration objects alongside run mode
    and seed settings for service initialization.
    """
    etl: ETLConfig
    icd: ICDConfig
    cnn: CNNConfig
    mode: str
    seed: int

@dataclass
class TensorBatch:
    """Unified tensor batch for training and evaluation.

    Bundles numerical features, optional categorical features, and labels,
    with optional fields for partial-label learning and text inputs.

    Attributes:
        x_num (torch.Tensor): Numerical feature tensor.
        x_cat (Optional[torch.Tensor]): Categorical feature tensor.
        y (torch.Tensor): Target labels tensor.
        ids (Optional[torch.Tensor]): Optional sample identifiers.
        mask (Optional[torch.Tensor]): Optional observation mask tensor.
        x_icd (Optional[torch.Tensor]): Optional ICD-related features.
        input_ids (Optional[torch.Tensor]): Optional token ids for text inputs.
        attention_mask (Optional[torch.Tensor]): Optional attention mask.
        candidates (Optional[torch.Tensor]): Optional candidate label indices.
    """
    x_num: torch.Tensor
    x_cat: Optional[torch.Tensor]
    y: torch.Tensor
    ids: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    x_icd: Optional[torch.Tensor] = None
    
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    candidates: Optional[torch.Tensor] = None
    
    def to(self, device: str) -> 'TensorBatch':
        """Move all available tensors in the batch to a target device.

        Args:
            device (str): Device identifier such as "cpu" or "cuda".

        Returns:
            TensorBatch: New batch instance with tensors moved to the device.
        """
        return TensorBatch(
            x_num=self.x_num.to(device),
            x_cat=self.x_cat.to(device) if self.x_cat is not None else None,
            y=self.y.to(device),
            ids=self.ids.to(device) if self.ids is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None,
            x_icd=self.x_icd.to(device) if self.x_icd is not None else None,
            input_ids=self.input_ids.to(device) if self.input_ids is not None else None,
            attention_mask=self.attention_mask.to(device) if self.attention_mask is not None else None,
            candidates=self.candidates.to(device) if self.candidates is not None else None
        )

@dataclass
class PredictionResult:
    """Container for prediction outputs.

    Stores logits, probabilities, predicted labels, and optional targets.
    """
    logits: torch.Tensor
    probs: torch.Tensor
    preds: torch.Tensor
    targets: Optional[torch.Tensor] = None
