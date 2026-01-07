import torch
import torch.nn as nn
from typing import Optional
from ..networks import BaseICDClassifier

class RK4Solver(nn.Module):
    """Runge-Kutta 4th order ODE solver.

    Attributes:
        func (nn.Module): ODE function to integrate.
    """
    def __init__(self, func):
        """Initialize the solver with an ODE function.

        Args:
            func (nn.Module): ODE function module.
        """
        super().__init__()
        self.func = func

    def forward(self, z0, t_span):
        """Integrate the ODE across the provided time span.

        Args:
            z0 (torch.Tensor): Initial latent state.
            t_span (torch.Tensor): 1D tensor of time points.

        Returns:
            torch.Tensor: Latent states at each time point.
        """
                                
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
    """Neural ODE function for latent dynamics.

    Attributes:
        net (nn.Sequential): Feed-forward network for derivatives.
    """
    def __init__(self, latent_dim, hidden_dim=64):
        """Initialize the ODE function network.

        Args:
            latent_dim (int): Latent dimension size.
            hidden_dim (int): Hidden layer size.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward(self, t, x):
        """Compute derivatives for the latent state.

        Args:
            t (torch.Tensor): Time value.
            x (torch.Tensor): Latent state.

        Returns:
            torch.Tensor: Time derivative of the state.
        """
        return self.net(x)

class LatentODEClassifier(BaseICDClassifier):
    """Latent ODE classifier for irregular time-series data.

    Attributes:
        rnn (nn.GRU): Encoder RNN for initial state.
        fc_mean (nn.Linear): Latent mean projection.
        fc_std (nn.Linear): Latent std projection.
        ode_func (ODEFunc): Latent dynamics function.
        solver (RK4Solver): ODE integration solver.
        classifier (nn.Sequential): Classifier head.
        latent_dim (int): Latent dimension size.
    """
    def __init__(self, num_classes: int, input_dim: int, seq_len: int, latent_dim: int = 64, rec_dim: int = 128, **kwargs):
        """Initialize the latent ODE classifier.

        Args:
            num_classes (int): Number of output classes.
            input_dim (int): Input feature dimension.
            seq_len (int): Sequence length.
            latent_dim (int): Latent dimension size.
            rec_dim (int): RNN hidden dimension.
            **kwargs (Any): Additional model arguments.
        """
        super().__init__(num_classes, input_dim, seq_len)
        
                                                                        
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
        """Run the forward pass on numeric time-series data.

        Args:
            x_num (Optional[torch.Tensor]): Numeric input features.
            **kwargs (Any): Additional model inputs.

        Returns:
            torch.Tensor: Class logits.

        Raises:
            ValueError: If x_num is not provided.
        """
                                       
        if x_num is None: raise ValueError("LatentODE requires x_num")
        x = x_num.permute(0, 2, 1)
        b, t, c = x.size()
        
                                                                              
        _, h_n = self.rnn(torch.flip(x, [1]))
        h_n = h_n.squeeze(0)
        
                                
        qm = self.fc_mean(h_n)
        
                                                                 
                                                          
        t_span = torch.linspace(0, 1, steps=10).to(x.device)
        z_t = self.solver(qm, t_span)                     
        
        z_final = z_t[-1]
        
        return self.classifier(z_final)
