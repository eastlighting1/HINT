import torch
import torch.nn as nn
from typing import Optional
from ..networks import BaseICDClassifier

class RK4Solver(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, z0, t_span):
        # Simple RK4 integration
        dt = (t_span[-1] - t_span[0]) / (len(t_span) - 1)
        z = z0
        zs = [z]
        for i in range(len(t_span) - 1):
            t = t_span[i]
            k1 = self.func(t, z)
            k2 = self.func(t + dt/2, z + dt/2 * k1)
            k3 = self.func(t + dt/2, z + dt/2 * k2)
            k4 = self.func(t + dt, z + dt * k3)
            z = z + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            zs.append(z)
        return torch.stack(zs)

class ODEFunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward(self, t, x):
        return self.net(x)

class LatentODEClassifier(BaseICDClassifier):
    def __init__(self, num_classes: int, input_dim: int, seq_len: int, latent_dim: int = 64, rec_dim: int = 128, **kwargs):
        super().__init__(num_classes, input_dim, seq_len)
        
        # ODE-RNN Encoder part (Simplified as RNN encoder -> ODE solver)
        self.rnn = nn.GRU(input_dim, rec_dim, batch_first=True)
        self.fc_mean = nn.Linear(rec_dim, latent_dim)
        self.fc_std = nn.Linear(rec_dim, latent_dim)
        
        self.ode_func = ODEFunc(latent_dim)
        self.solver = RK4Solver(self.ode_func)
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, rec_dim),
            nn.ReLU(),
            nn.Linear(rec_dim, num_classes)
        )
        self.latent_dim = latent_dim

    def forward(self, x_num: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # x_num: (B, C, T) -> (B, T, C)
        if x_num is None: raise ValueError("LatentODE requires x_num")
        x = x_num.permute(0, 2, 1)
        b, t, c = x.size()
        
        # Encode backward to get initial state (common strategy in Latent ODE)
        _, h_n = self.rnn(torch.flip(x, [1]))
        h_n = h_n.squeeze(0)
        
        # Variational parameters
        qm = self.fc_mean(h_n)
        
        # Solve ODE forward (simplified: solving from t=0 to t=1)
        # We use the final state of ODE for classification
        t_span = torch.linspace(0, 1, steps=10).to(x.device)
        z_t = self.solver(qm, t_span) # (Steps, B, Latent)
        
        z_final = z_t[-1]
        
        return self.classifier(z_final)