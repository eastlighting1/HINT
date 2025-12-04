import torch
import torch.nn as nn
from transformers import AutoModel

class MedBERTClassifier(nn.Module):
    """
    BERT-based classifier head concatenating text embeddings and numerical features.

    Args:
        model_name: Pretrained transformer model name.
        num_num: Number of numerical features.
        num_cls: Number of output classes.
        dropout: Dropout probability.
    """
    def __init__(self, model_name: str, num_num: int, num_cls: int, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size + num_num, num_cls)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        numerical: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token indices.
            attention_mask: Attention mask.
            numerical: Numerical features tensor.

        Returns:
            Logits tensor.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state.mean(dim=1)
            
        combined = torch.cat((pooled, numerical), dim=1)
        return self.fc(self.dropout(combined))

    def set_backbone_grad(self, requires_grad: bool) -> None:
        """
        Freeze or unfreeze the BERT backbone.

        Args:
            requires_grad: Whether to calculate gradients.
        """
        for param in self.bert.parameters():
            param.requires_grad = requires_grad