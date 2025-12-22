import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional
from ..networks import BaseICDClassifier

class MedBERTClassifier(BaseICDClassifier):
    def __init__(self, num_classes: int, input_dim: int, seq_len: int, dropout: float = 0.3, bert_model_name: str = "Charangan/MedBERT", **kwargs):
        super().__init__(num_classes, input_dim, seq_len, dropout)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden = self.bert.config.hidden_size
        self.dp = nn.Dropout(dropout)
        # Fuse tabular input with BERT pooled output
        self.fc = nn.Linear(hidden + input_dim, num_classes)

    def forward(self, input_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None, 
                x_num: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if input_ids is None or attention_mask is None:
            raise ValueError("MedBERT requires input_ids and attention_mask")

        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state.mean(dim=1)
        
        # Prepare tabular data: (B, C, T) -> Mean Pool -> (B, C)
        if x_num is not None:
            if x_num.dim() == 3:
                x_num_pooled = x_num.mean(dim=2)
            else:
                x_num_pooled = x_num
        else:
            raise ValueError("MedBERT requires numeric input x_num")

        x = torch.cat((pooled, x_num_pooled), dim=1)
        return self.fc(self.dp(x))