from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from hint.domain.vo import ETLConfig, ICDConfig, CNNConfig

@dataclass
class AppContext:
    """Holds all configuration VOs for the application."""
    etl: ETLConfig
    icd: ICDConfig
    cnn: CNNConfig
    mode: str
    seed: int

@dataclass
class TensorBatch:
    """Standard batch format for model training."""
    x_num: torch.Tensor
    x_cat: Optional[torch.Tensor]
    y: torch.Tensor
    ids: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    x_icd: Optional[torch.Tensor] = None
    
    def to(self, device: str) -> 'TensorBatch':
        return TensorBatch(
            x_num=self.x_num.to(device),
            x_cat=self.x_cat.to(device) if self.x_cat is not None else None,
            y=self.y.to(device),
            ids=self.ids.to(device) if self.ids is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None,
            x_icd=self.x_icd.to(device) if self.x_icd is not None else None
        )

@dataclass
class PredictionResult:
    """Result of a prediction step."""
    logits: torch.Tensor
    probs: torch.Tensor
    preds: torch.Tensor
    targets: Optional[torch.Tensor] = None
