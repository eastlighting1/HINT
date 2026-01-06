import torch
import torch.nn as nn
from ..networks import BaseICDClassifier

class CrossNetV2(nn.Module):
    """Cross Network V2 component for Deep & Cross models.

    Applies iterative cross layers to model explicit feature interactions.
    """
    def __init__(self, input_dim, num_layers):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
                                                                       
        self.kernels = nn.ParameterList([nn.Parameter(torch.randn(input_dim, 1)) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])
        
                         
        for p in self.kernels:
            nn.init.xavier_normal_(p)

    def forward(self, x):
                               
        x_0 = x.unsqueeze(2)            
        x_l = x_0
        
        for i in range(self.num_layers):
                                                          
                                     
                                 
            xl_w = torch.matmul(x_l.transpose(1, 2), self.kernels[i])            
            
                                 
            x_l = x_0 * xl_w + self.biases[i].unsqueeze(0).unsqueeze(2) + x_l
            
        return x_l.squeeze(2)

class DCNv2ICD(BaseICDClassifier):
    """Deep & Cross Network V2 for ICD coding.

    Combines a cross network with a deep MLP and stacks the outputs.
    """
    def __init__(self, num_classes: int, input_dim: int, seq_len: int, dropout: float = 0.2, 
                 cross_layers=3, deep_layers=[256, 256], **kwargs):
        super().__init__(num_classes, input_dim, seq_len, dropout)
        
                                 
        self.flat_dim = input_dim * seq_len
        self.input_bn = nn.BatchNorm1d(self.flat_dim)
        
                          
        self.cross_net = CrossNetV2(self.flat_dim, num_layers=cross_layers)
        
                         
        layers = []
        in_dim = self.flat_dim
        for hid_dim in deep_layers:
            layers.append(nn.Linear(in_dim, hid_dim))
            layers.append(nn.BatchNorm1d(hid_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hid_dim
        self.deep_net = nn.Sequential(*layers)
        
                           
        stack_dim = self.flat_dim + deep_layers[-1]
        self.final_mapping = nn.Linear(stack_dim, num_classes)
        self.embedding_dim = stack_dim                       

    def forward(self, x_num: torch.Tensor, return_embeddings: bool = False, **kwargs) -> torch.Tensor:
                                      
        x = x_num.reshape(x_num.size(0), -1)
        x = self.input_bn(x)
        
                            
        x_cross = self.cross_net(x)
        x_deep = self.deep_net(x)
        
                        
        x_stack = torch.cat([x_cross, x_deep], dim=1)
        
        if return_embeddings:
            return x_stack
            
        return self.final_mapping(x_stack)
