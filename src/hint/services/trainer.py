import torch.nn as nn
from sklearn.metrics import accuracy_score
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from ..foundation.interfaces import ModelRegistry, TelemetryObserver, StreamingSource
from ..domain.entities import TrainableEntity

class TrainingService:
    """
    Domain Service responsible for orchestrating the training process.

    Manages the data flow between Source, Entity, and Registry, while reporting
    progress via Telemetry.

    Args:
        registry: Repository for saving models.
        observer: Observer for logging and metrics.
        device: Computation device.
    """
    def __init__(
        self,
        registry: ModelRegistry,
        observer: TelemetryObserver,
        device: str
    ):
        self.registry = registry
        self.observer = observer
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()

    def train_model(
        self,
        entity: TrainableEntity,
        train_source: StreamingSource,
        val_source: StreamingSource,
        epochs: int
    ) -> None:
        """
        Execute the full training loop.

        Args:
            entity: The model entity to train.
            train_source: Source for training data.
            val_source: Source for validation data.
            epochs: Total number of epochs.
        """
        entity.network.to(self.device)
        self.observer.log("INFO", f"Step: Initializing training for entity {entity.id}")

        for epoch in range(1, epochs + 1):
            entity.epoch = epoch
            epoch_loss = 0.0
            steps = 0
            
            self.observer.log("INFO", f"Step: Starting Epoch {epoch}/{epochs}")

            # Rich progress bar for the training loop
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                
                task = progress.add_task(f"[green]Epoch {epoch} Training...", total=len(train_source))
                
                with self.observer.trace(f"train_epoch_{epoch}"):
                    for batch in train_source.stream_batches():
                        batch = batch.to(self.device)
                        loss = entity.step_train(batch, self.loss_fn)
                        epoch_loss += loss
                        steps += 1
                        progress.advance(task)
            
            avg_train_loss = epoch_loss / max(1, steps)
            self.observer.track_metric("train_loss", avg_train_loss, step=epoch)
            
            self.observer.log("INFO", f"Step: Validating Epoch {epoch}")
            val_acc = self._evaluate(entity, val_source)
            self.observer.track_metric("val_acc", val_acc, step=epoch)
            
            self.observer.log("INFO", f"Summary Epoch {epoch}: Loss={avg_train_loss:.4f}, ValAcc={val_acc:.4f}")

            if val_acc > entity.best_metric:
                entity.best_metric = val_acc
                self.registry.save(entity, tag="best")
                self.observer.log("INFO", f"Step: Checkpoint saved (New Best Acc: {val_acc:.4f})")

        self.registry.save(entity, tag="last")
        self.observer.log("INFO", "Step: Training finished successfully.")

    def _evaluate(self, entity: TrainableEntity, source: StreamingSource) -> float:
        """
        Internal method to evaluate the model on a dataset.

        Args:
            entity: The model entity.
            source: Data source for evaluation.

        Returns:
            Accuracy score.
        """
        all_preds, all_targets = [], []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("[blue]Validating...", total=len(source))
            
            for batch in source.stream_batches():
                batch = batch.to(self.device)
                logits = entity.predict(batch)
                preds = logits.argmax(dim=1).cpu()
                all_preds.extend(preds)
                all_targets.extend(batch.targets.cpu())
                progress.advance(task)
        
        return accuracy_score(all_targets, all_preds)