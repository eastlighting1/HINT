import uuid
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage

from ..foundation.configs import TrainingConfig
from ..foundation.dtos import TensorBatch

class TrainableEntity:
    """
    Domain Entity representing a trainable model with state.

    This entity encapsulates the network architecture, optimizer, EMA state,
    and training metadata like current epoch and best metrics. It provides
    methods for prediction, training steps, and state snapshots.

    Attributes:
        id: Unique identifier for the training session.
        network: The underlying neural network module.
        config: Training configuration.
        epoch: Current training epoch.
        best_metric: Best metric value achieved so far.
        current_loss: Most recent loss value.
        state: Lifecycle state of the entity.
        optimizer: Optimizer instance.
        ema: Exponential Moving Average instance.
    """
    def __init__(self, network: nn.Module, config: TrainingConfig):
        self.id = uuid.uuid4().hex[:8]
        self.network = network
        self.config = config
        
        self.epoch = 0
        self.best_metric = 0.0
        self.current_loss = float('inf')
        self.state = "INITIALIZED"
        
        self.optimizer = torch.optim.AdamW(
            network.parameters(), 
            lr=config.lr, 
            weight_decay=1e-5
        )
        self.ema = ExponentialMovingAverage(
            network.parameters(), 
            decay=config.ema_decay
        )

    def predict(self, batch: TensorBatch) -> torch.Tensor:
        """
        Perform inference using EMA weights.

        Args:
            batch: Input data batch.

        Returns:
            Logits tensor.
        """
        self.network.eval()
        with self.ema.average_parameters():
            with torch.no_grad():
                return self.network(batch.x_num, batch.x_cat)

    def step_train(self, batch: TensorBatch, loss_fn: nn.Module) -> float:
        """
        Perform a single training step.

        Args:
            batch: Input data batch.
            loss_fn: Loss function module.

        Returns:
            The scalar loss value.
        """
        self.network.train()
        self.optimizer.zero_grad()
        
        logits = self.network(batch.x_num, batch.x_cat)
        loss = loss_fn(logits, batch.targets)
        
        loss.backward()
        self.optimizer.step()
        self.ema.update(self.network.parameters())
        
        self.current_loss = loss.item()
        return self.current_loss

    def snapshot(self) -> Dict[str, Any]:
        """
        Create a serializable snapshot of the current state.

        Returns:
            Dictionary containing state dictionaries of components and metadata.
        """
        return {
            "id": self.id,
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "ema_state": self.ema.state_dict(),
            "epoch": self.epoch,
            "best_metric": self.best_metric,
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """
        Restore state from a snapshot.

        Args:
            snapshot: Dictionary containing saved state.
        """
        self.id = snapshot["id"]
        self.network.load_state_dict(snapshot["network_state"])
        self.optimizer.load_state_dict(snapshot["optimizer_state"])
        self.ema.load_state_dict(snapshot["ema_state"])
        self.epoch = snapshot["epoch"]
        self.best_metric = snapshot["best_metric"]