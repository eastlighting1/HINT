from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from hint.domain.vo import ETLConfig, ICDConfig, CNNConfig

@dataclass
class AppContext:
    """Container for resolved application configuration.

    This dataclass bundles ETL, ICD, and CNN configuration objects along
    with runtime mode and seed values.

    Attributes:
        etl (ETLConfig): ETL configuration values.
        icd (ICDConfig): ICD training configuration values.
        cnn (CNNConfig): Intervention model configuration values.
        mode (str): Execution mode for the application.
        seed (int): Random seed for reproducibility.
    """
    etl: ETLConfig
    icd: ICDConfig
    cnn: CNNConfig
    mode: str
    seed: int

@dataclass
class TensorBatch:
    """Bundle of tensor inputs and targets for model execution.

    This dataclass groups dynamic features, static features, and labels
    to simplify device transfers and data loading.

    Attributes:
        x_num (torch.Tensor): Numeric time-series features.
        x_cat (Optional[torch.Tensor]): Categorical time-series features.
        y (torch.Tensor): Target labels.
        ids (Optional[torch.Tensor]): Optional identifier tensor.
        mask (Optional[torch.Tensor]): Optional attention or padding mask.
        x_icd (Optional[torch.Tensor]): Optional ICD embedding features.
        input_ids (Optional[torch.Tensor]): Token IDs for static text.
        attention_mask (Optional[torch.Tensor]): Attention mask for tokens.
        candidates (Optional[torch.Tensor]): Optional candidate indices.
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
        """Move all tensors in the batch to the requested device.

        This method returns a new TensorBatch with all non-null tensors
        moved to the given device.

        Args:
            device (str): Target device identifier.

        Returns:
            TensorBatch: New batch with tensors on the target device.
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
    """Result container for model predictions.

    This dataclass carries raw logits, probabilities, and discrete
    predictions with optional targets for evaluation.

    Attributes:
        logits (torch.Tensor): Raw model outputs.
        probs (torch.Tensor): Post-processed probabilities.
        preds (torch.Tensor): Final predicted labels.
        targets (Optional[torch.Tensor]): Optional ground-truth labels.
    """
    logits: torch.Tensor
    probs: torch.Tensor
    preds: torch.Tensor
    targets: Optional[torch.Tensor] = None
