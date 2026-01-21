"""Summary of the dtos module.

Longer description of the module purpose and usage.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from hint.domain.vo import ETLConfig, ICDConfig, InterventionConfig



@dataclass
class AppContext:

    """Summary of AppContext purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        None (None): No documented attributes.
    """

    etl: ETLConfig

    icd: ICDConfig

    intervention: InterventionConfig

    mode: str

    seed: int



@dataclass
class TensorBatch:

    """Summary of TensorBatch purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        None (None): No documented attributes.
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

        """Summary of to.
        
        Longer description of the to behavior and usage.
        
        Args:
            device (Any): Description of device.
        
        Returns:
            'TensorBatch': Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
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

    """Summary of PredictionResult purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        None (None): No documented attributes.
    """

    logits: torch.Tensor

    probs: torch.Tensor

    preds: torch.Tensor

    targets: Optional[torch.Tensor] = None
