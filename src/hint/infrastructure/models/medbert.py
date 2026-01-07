import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional
from ..networks import BaseICDClassifier

class MedBERTClassifier(BaseICDClassifier):
    """BERT-based classifier for ICD prediction from time-series input.

    Attributes:
        bert (nn.Module): Pretrained BERT backbone.
        hidden_size (int): Hidden size of the backbone.
        num_projection (nn.Linear): Projection for numeric features.
        dp (nn.Dropout): Dropout layer for pooled outputs.
        fc (nn.Linear): Output projection to class logits.
    """
    def __init__(self, num_classes: int, input_dim: int, seq_len: int, dropout: float = 0.3, bert_model_name: str = "Charangan/MedBERT", **kwargs):
        """Initialize the MedBERT classifier.

        Args:
            num_classes (int): Number of output classes.
            input_dim (int): Input feature dimension.
            seq_len (int): Sequence length.
            dropout (float): Dropout probability.
            bert_model_name (str): Pretrained model name or path.
            **kwargs (Any): Additional model arguments.
        """
        super().__init__(num_classes, input_dim, seq_len, dropout)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.num_projection = nn.Linear(input_dim, self.hidden_size)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        x_num: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Run the forward pass using numeric time-series embeddings.

        Args:
            input_ids (Optional[torch.Tensor]): Unused token IDs placeholder.
            attention_mask (Optional[torch.Tensor]): Unused attention mask.
            x_num (Optional[torch.Tensor]): Numeric input features.
            **kwargs (Any): Additional model inputs.

        Returns:
            torch.Tensor: Class logits.

        Raises:
            ValueError: If x_num is not provided.
        """
        
        if x_num is None:
            raise ValueError("MedBERT requires numeric input x_num")

                                                                   
        x = x_num.permute(0, 2, 1) 
        
                                                               
                                                                                             
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
        """Enable or disable gradients on the BERT backbone.

        Args:
            requires_grad (bool): Whether to enable gradient updates.
        """
        for p in self.bert.parameters():
            p.requires_grad = requires_grad
