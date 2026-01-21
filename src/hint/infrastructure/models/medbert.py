"""Summary of the medbert module.

Longer description of the module purpose and usage.
"""

import torch

import torch.nn as nn

from transformers import AutoModel

from typing import Optional

from ..networks import BaseICDClassifier



class MedBERTClassifier(BaseICDClassifier):

    """Summary of MedBERTClassifier purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        bert (Any): Description of bert.
        dp (Any): Description of dp.
        fc (Any): Description of fc.
        hidden_size (Any): Description of hidden_size.
        num_projection (Any): Description of num_projection.
    """

    def __init__(self, num_classes: int, input_dim: int, seq_len: int, dropout: float = 0.3, bert_model_name: str = "Charangan/MedBERT", **kwargs):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            num_classes (Any): Description of num_classes.
            input_dim (Any): Description of input_dim.
            seq_len (Any): Description of seq_len.
            dropout (Any): Description of dropout.
            bert_model_name (Any): Description of bert_model_name.
            kwargs (Any): Description of kwargs.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
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

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
            input_ids (Any): Description of input_ids.
            attention_mask (Any): Description of attention_mask.
            x_num (Any): Description of x_num.
            kwargs (Any): Description of kwargs.
        
        Returns:
            torch.Tensor: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
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

        """Summary of set_backbone_grad.
        
        Longer description of the set_backbone_grad behavior and usage.
        
        Args:
            requires_grad (Any): Description of requires_grad.
        
        Returns:
            Any: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        for p in self.bert.parameters():

            p.requires_grad = requires_grad
