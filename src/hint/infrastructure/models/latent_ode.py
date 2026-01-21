"""Summary of the latent_ode module.

Longer description of the module purpose and usage.
"""

import torch

import torch.nn as nn

from typing import Optional

from ..networks import BaseICDClassifier



class RK4Solver(nn.Module):

    """Summary of RK4Solver purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        func (Any): Description of func.
    """

    def __init__(self, func):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            func (Any): Description of func.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        super().__init__()

        self.func = func



    def forward(self, z0, t_span):

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
            z0 (Any): Description of z0.
            t_span (Any): Description of t_span.
        
        Returns:
            Any: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
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

    """Summary of ODEFunc purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        net (Any): Description of net.
    """

    def __init__(self, latent_dim, hidden_dim=64):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            latent_dim (Any): Description of latent_dim.
            hidden_dim (Any): Description of hidden_dim.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(latent_dim, hidden_dim),

            nn.Tanh(),

            nn.Linear(hidden_dim, latent_dim)

        )

    def forward(self, t, x):

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
            t (Any): Description of t.
            x (Any): Description of x.
        
        Returns:
            Any: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        return self.net(x)



class LatentODEClassifier(BaseICDClassifier):

    """Summary of LatentODEClassifier purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        classifier (Any): Description of classifier.
        fc_mean (Any): Description of fc_mean.
        fc_std (Any): Description of fc_std.
        latent_dim (Any): Description of latent_dim.
        ode_func (Any): Description of ode_func.
        rnn (Any): Description of rnn.
        solver (Any): Description of solver.
    """

    def __init__(self, num_classes: int, input_dim: int, seq_len: int, latent_dim: int = 64, rec_dim: int = 128, **kwargs):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            num_classes (Any): Description of num_classes.
            input_dim (Any): Description of input_dim.
            seq_len (Any): Description of seq_len.
            latent_dim (Any): Description of latent_dim.
            rec_dim (Any): Description of rec_dim.
            kwargs (Any): Description of kwargs.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
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

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
            x_num (Any): Description of x_num.
            kwargs (Any): Description of kwargs.
        
        Returns:
            torch.Tensor: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
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
