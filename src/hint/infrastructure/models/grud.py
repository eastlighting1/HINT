import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..networks import BaseICDClassifier  # 사용하시는 경로에 맞춰 유지

class GRUDClassifier(BaseICDClassifier):
    def __init__(self, num_classes: int, input_dim: int, seq_len: int, hidden_dim: int = 64, dropout: float = 0.3, **kwargs):
        super().__init__(num_classes, input_dim, seq_len, dropout=dropout, **kwargs)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # ----------------------------------------------------------------------
        # [FIX 1] Diagonal Decay for Input (Gamma X)
        # 논문 식 (6): W_gamma_x는 대각 행렬이어야 합니다.
        # 따라서 nn.Linear(input_dim, input_dim) 대신, 파라미터 벡터를 사용하여
        # Element-wise 곱셈을 수행해야 합니다.
        # ----------------------------------------------------------------------
        self.w_gamma_x = nn.Parameter(torch.Tensor(input_dim))
        self.b_gamma_x = nn.Parameter(torch.Tensor(input_dim))
        
        # [NOTE] Hidden Decay (Gamma H)
        # 논문 식 (8): 입력 델타(input_dim)를 이용해 히든 스테이트(hidden_dim)의 
        # Decay를 결정하므로, 여기서는 Full Matrix(Linear)를 사용하는 것이 맞습니다.
        self.lin_gamma_h = nn.Linear(input_dim, hidden_dim)
        
        # ----------------------------------------------------------------------
        # [FIX 2] Empirical Mean Buffer
        # 결측이 길어질 때 수렴할 목표값(경험적 평균)을 저장할 버퍼입니다.
        # 학습 전, 데이터셋 전체의 평균으로 이 값을 초기화해주는 것이 좋습니다.
        # (기본값은 0으로 설정됨)
        # ----------------------------------------------------------------------
        self.register_buffer('empirical_mean', torch.zeros(input_dim))
        
        # GRU Cell Components
        # 입력: [x_impute (input_dim), mask (input_dim), h_decay (hidden_dim)]
        input_size = input_dim * 2 + hidden_dim
        
        self.z_layer = nn.Linear(input_size, hidden_dim)
        self.r_layer = nn.Linear(input_size, hidden_dim)
        self.h_layer = nn.Linear(input_size, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        self._init_parameters()

    def _init_parameters(self):
        # 파라미터 초기화 (Decay 관련 가중치는 보통 0 근처 혹은 양수로 초기화하여 학습 유도)
        nn.init.xavier_uniform_(self.z_layer.weight)
        nn.init.xavier_uniform_(self.r_layer.weight)
        nn.init.xavier_uniform_(self.h_layer.weight)
        nn.init.uniform_(self.w_gamma_x, -0.01, 0.01) # 작은 값으로 초기화
        nn.init.constant_(self.b_gamma_x, 0)

    def forward(self, x_num: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None, 
                delta: Optional[torch.Tensor] = None, 
                **kwargs) -> torch.Tensor:
        """
        x_num: (Batch, Channels, Time) -> 내부에서 (Batch, Time, Channels)로 변환
        mask:  (Batch, Channels, Time) -> 결측치 마스크 (1=관측됨, 0=결측)
        delta: (Batch, Channels, Time) -> 마지막 관측 이후 경과 시간
        
        * mask와 delta가 주어지지 않으면, x_num이 모두 관측된 것으로 가정하고
          단순 Time step(1.0)으로 delta를 생성합니다.
        """
        if x_num is None: 
            raise ValueError("GRU-D requires x_num")
        
        # (B, C, T) -> (B, T, C)
        x = x_num.permute(0, 2, 1)
        b, t, c = x.size()
        
        # Mask 및 Delta 처리 (입력으로 들어오지 않았을 경우의 예외처리)
        if mask is None:
            mask = torch.ones_like(x).to(x.device)
        else:
            mask = mask.permute(0, 2, 1) # (B, T, C) 맞춰줌
            
        if delta is None:
            # Delta가 없으면 단순히 매 스텝 1씩 증가한다고 가정 (Regular sampling)
            delta = torch.ones_like(x).to(x.device)
        else:
            delta = delta.permute(0, 2, 1)

        h = torch.zeros(b, self.hidden_dim).to(x.device)
        
        # 마지막 관측값 저장용 (초기값은 0 혹은 평균 등 설정 가능)
        # 논문에서는 관측되지 않은 시작점의 경우 평균을 사용하기도 함
        x_last_obsv = torch.zeros(b, c).to(x.device)
        
        for i in range(t):
            x_t = x[:, i, :]      # (B, C)
            m_t = mask[:, i, :]   # (B, C)
            d_t = delta[:, i, :]  # (B, C)
            
            # 1. Decay Rates 계산
            # [FIX] Gamma X: Diagonal Constraint 적용 (Element-wise)
            # 식: gamma_x = exp(-max(0, w * d + b))
            gamma_x = torch.exp(-torch.relu(d_t * self.w_gamma_x + self.b_gamma_x))
            
            # Gamma H: Linear Projection (Full Matrix)
            gamma_h = torch.exp(-torch.relu(self.lin_gamma_h(d_t)))
            
            # 2. Imputation (Missing Value 처리)
            # [FIX] 논문 식 (5): 관측되면 x_t, 아니면 (gamma * last_obsv + (1-gamma) * mean)
            # 현재 self.empirical_mean은 (C,) 형태이므로 브로드캐스팅됨
            x_impute = m_t * x_t + (1 - m_t) * (
                gamma_x * x_last_obsv + (1 - gamma_x) * self.empirical_mean
            )
            
            # 3. Hidden State Decay
            h_decay = gamma_h * h
            
            # 4. GRU Cell Update
            # 입력으로 Imputed X와 Mask, Decayed Hidden을 함께 사용
            combined = torch.cat([x_impute, m_t, h_decay], dim=1)
            
            z = torch.sigmoid(self.z_layer(combined))
            r = torch.sigmoid(self.r_layer(combined))
            
            # Candidate Hidden State
            # 논문에서는 Reset Gate가 적용된 Hidden에 대해 다시 Decay를 적용하지 않거나
            # r * h_decay 형태로 사용함 (여기서는 r * h_decay 사용)
            combined_new = torch.cat([x_impute, m_t, r * h_decay], dim=1)
            h_tilde = torch.tanh(self.h_layer(combined_new))
            
            h = (1 - z) * h_decay + z * h_tilde
            
            # 5. 마지막 관측값 업데이트
            # 관측되었으면(m_t=1) 현재 값으로, 아니면 기존 값 유지
            x_last_obsv = m_t * x_t + (1 - m_t) * x_last_obsv
            
        return self.fc(self.dropout(h))