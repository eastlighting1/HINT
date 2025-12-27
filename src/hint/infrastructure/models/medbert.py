import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional
from ..networks import BaseICDClassifier

class MedBERTClassifier(BaseICDClassifier):
    def __init__(self, num_classes: int, input_dim: int, seq_len: int, dropout: float = 0.3, bert_model_name: str = "Charangan/MedBERT", **kwargs):
        super().__init__(num_classes, input_dim, seq_len, dropout)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.num_projection = nn.Linear(input_dim, self.hidden_size)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None, 
                x_num: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        
        if x_num is None:
            raise ValueError("MedBERT requires numeric input x_num")

        # x_num: (Batch, Channels, Time) -> (Batch, Time, Channels)
        x = x_num.permute(0, 2, 1) 
        
        # [FIX] Generate correct attention mask for Time-Series
        # The 'attention_mask' from args is for text (len 32), we need one for time (len 120)
        batch_size, seq_len, _ = x.size()
        ts_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=x.device)

        inputs_embeds = self.num_projection(x)
        out = self.bert(inputs_embeds=inputs_embeds, attention_mask=ts_mask)
        
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state.mean(dim=1)
        
        return self.fc(self.dp(pooled))

    def set_backbone_grad(self, requires_grad: bool):
        for p in self.bert.parameters():
            p.requires_grad = requires_grad