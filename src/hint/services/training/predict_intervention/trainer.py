"""Summary of the trainer module.

Longer description of the module purpose and usage.
"""

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader

from ..common.base import BaseTrainer, BaseEvaluator

from ....domain.entities import InterventionModelEntity

from ....domain.vo import CNNConfig



class InterventionTrainer(BaseTrainer):

    """Summary of InterventionTrainer purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    cfg (Any): Description of cfg.
    class_weights (Any): Description of class_weights.
    entity (Any): Description of entity.
    gamma (Any): Description of gamma.
    """



    def __init__(self, config: CNNConfig, entity: InterventionModelEntity, registry, observer, device, class_weights=None):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        config (Any): Description of config.
        entity (Any): Description of entity.
        registry (Any): Description of registry.
        observer (Any): Description of observer.
        device (Any): Description of device.
        class_weights (Any): Description of class_weights.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super().__init__(registry, observer, device)

        self.cfg = config

        self.entity = entity

        self.class_weights = class_weights

        self.gamma = self.cfg.focal_gamma



    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        """Summary of focal_loss.
        
        Longer description of the focal_loss behavior and usage.
        
        Args:
        logits (Any): Description of logits.
        targets (Any): Description of targets.
        
        Returns:
        torch.Tensor: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=-100)

        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss



        if self.class_weights is not None:

            valid_mask = targets != -100

            weights = torch.ones_like(targets, dtype=torch.float)



            safe_targets = targets.clone()

            safe_targets[~valid_mask] = 0



            weights[valid_mask] = self.class_weights[safe_targets[valid_mask]]

            focal_loss = focal_loss * weights



        return focal_loss.mean()



    def _prepare_inputs(self, batch) -> dict:

        """Summary of _prepare_inputs.
        
        Longer description of the _prepare_inputs behavior and usage.
        
        Args:
        batch (Any): Description of batch.
        
        Returns:
        dict: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        x_num = batch.x_num.to(self.device).float()



        inputs = {"x_num": x_num}



        if batch.x_cat is not None:

            inputs["x_cat"] = batch.x_cat.to(self.device).long()



        if batch.x_icd is not None:

            inputs["x_icd"] = batch.x_icd.to(self.device).float()

        else:

            inputs["x_icd"] = None



        return inputs



    def _select_last_valid(self, y: torch.Tensor) -> torch.Tensor:

        """Summary of _select_last_valid.
        
        Longer description of the _select_last_valid behavior and usage.
        
        Args:
        y (Any): Description of y.
        
        Returns:
        torch.Tensor: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        if y.dim() == 1:

            return y

        valid = y != -100

        has_valid = valid.any(dim=1)

        flipped = torch.flip(valid, dims=[1])

        last_from_end = flipped.int().argmax(dim=1)

        last_idx = y.size(1) - 1 - last_from_end

        rows = torch.arange(y.size(0), device=y.device)

        target = y[rows, last_idx]

        return torch.where(has_valid, target, torch.full_like(target, -100))



    def train(self, train_loader: DataLoader, val_loader: DataLoader, evaluator: BaseEvaluator) -> None:

        """Summary of train.
        
        Longer description of the train behavior and usage.
        
        Args:
        train_loader (Any): Description of train_loader.
        val_loader (Any): Description of val_loader.
        evaluator (Any): Description of evaluator.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.entity.to(self.device)

        self.entity.network.train()



        optimizer = torch.optim.AdamW(self.entity.network.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
        self.observer.log(
            "INFO",
            f"InterventionTrainer: Stage 1/3 setup device={self.device} lr={self.cfg.lr} focal_gamma={self.gamma}.",
        )



        no_improve = 0

        self.observer.log("INFO", "InterventionTrainer: Stage 2/3 starting training loop.")

        for epoch in range(1, self.cfg.epochs + 1):

            self.entity.epoch = epoch

            self.observer.log("INFO", f"InterventionTrainer: Stage 2/3 epoch={epoch} start.")

            self._train_epoch(epoch, train_loader, optimizer)



            self.observer.log("INFO", f"InterventionTrainer: Stage 2/3 epoch={epoch} validation start.")

            metrics = evaluator.evaluate(val_loader)



            val_f1 = metrics["f1_score"]

            val_loss = metrics.get("loss", 0.0)

            macro_auc = metrics.get("macro_auc", 0.0)

            onset_auc = metrics.get("onset_auc", 0.0)

            wean_auc = metrics.get("wean_auc", 0.0)

            macro_auprc = metrics.get("macro_auprc", 0.0)





            self.observer.log(
                "INFO",
                f"InterventionTrainer: Stage 2/3 epoch={epoch} loss={val_loss:.4f} f1={val_f1:.4f} macro_auc={macro_auc:.4f} onset_auc={onset_auc:.4f} wean_auc={wean_auc:.4f}.",
            )



            metric_name = getattr(self.cfg, "early_stop_metric", "f1")

            metric_map = {

                "f1": val_f1,

                "macro_auc": macro_auc,

                "macro_auprc": macro_auprc,

            }

            val_metric = metric_map.get(metric_name, val_f1)

            self.observer.log("INFO", f"InterventionTrainer: Stage 2/3 early_stop_metric={metric_name} value={val_metric:.4f}.")



            if val_metric > self.entity.best_metric:

                self.entity.best_metric = val_metric

                self.entity.update_ema()

                self.registry.save_model(self.entity.state_dict(), self.cfg.artifacts.model_name, "best")

                no_improve = 0

            else:

                no_improve += 1

                if no_improve >= self.cfg.patience:

                    self.observer.log("WARNING", "InterventionTrainer: Stage 3/3 early stopping triggered.")

                    break



    def _train_epoch(self, epoch: int, loader: DataLoader, optimizer: torch.optim.Optimizer) -> None:

        """Summary of _train_epoch.
        
        Longer description of the _train_epoch behavior and usage.
        
        Args:
        epoch (Any): Description of epoch.
        loader (Any): Description of loader.
        optimizer (Any): Description of optimizer.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.entity.network.train()

        total_loss = 0.0

        steps = 0



        with self.observer.create_progress(f"Epoch {epoch} Train", total=len(loader)) as progress:

            task = progress.add_task("Training", total=len(loader))

            for batch in loader:

                optimizer.zero_grad()



                inputs = self._prepare_inputs(batch)

                y = batch.y.to(self.device)

                target = self._select_last_valid(y)



                logits = self.entity.network(**inputs)



                loss = self.focal_loss(logits, target)

                loss.backward()

                optimizer.step()

                self.entity.update_ema()



                total_loss += loss.item()

                steps += 1

                progress.advance(task)



        avg_loss = total_loss / max(1, steps)

        self.observer.track_metric("train_loss", avg_loss, step=epoch)
        self.observer.log("INFO", f"InterventionTrainer: Stage 2/3 epoch={epoch} train_loss={avg_loss:.4f}.")
