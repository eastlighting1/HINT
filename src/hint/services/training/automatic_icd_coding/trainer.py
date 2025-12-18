import torch
import numpy as np
from torch.utils.data import DataLoader

from ..common.base import BaseTrainer, BaseEvaluator
from ....domain.entities import ICDModelEntity
from ....domain.vo import ICDConfig
from ....infrastructure.components import CBFocalLoss

class ICDTrainer(BaseTrainer):
    def __init__(
        self, 
        config: ICDConfig, 
        entity: ICDModelEntity, 
        registry, 
        observer, 
        device,
        class_freq: np.ndarray = None
    ):
        super().__init__(registry, observer, device)
        self.cfg = config
        self.entity = entity
        self.class_freq = class_freq if class_freq is not None else np.ones(1)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, evaluator: BaseEvaluator) -> None:
        self.entity.to(self.device)
        self.observer.log("INFO", f"ICDTrainer: Start training for {self.cfg.epochs} epochs.")

        opt1 = torch.optim.Adam(self.entity.head1.parameters(), lr=self.cfg.lr)
        opt2 = torch.optim.Adam(self.entity.head2.parameters(), lr=self.cfg.lr)
        
        loss_fn = CBFocalLoss(
            class_counts=self.class_freq, 
            beta=self.cfg.cb_beta, 
            gamma=self.cfg.focal_gamma, 
            device=str(self.device)
        )

        for epoch in range(1, self.cfg.epochs + 1):
            self.entity.epoch = epoch
            self.observer.log("INFO", f"ICDTrainer: Epoch {epoch} training started with freeze={epoch <= self.cfg.freeze_bert_epochs}.")
            self.entity.head1.train()
            self.entity.head2.train()
            
            freeze = epoch <= self.cfg.freeze_bert_epochs
            self.entity.head1.set_backbone_grad(not freeze)
            self.entity.head2.set_backbone_grad(not freeze)
            
            with self.observer.create_progress(f"Epoch {epoch} Train", total=len(train_loader)) as progress:
                task = progress.add_task("Training", total=len(train_loader))
                for batch in train_loader:
                    # Fix 1: Use TensorBatch fields directly
                    ids = batch.input_ids.to(self.device)
                    mask = batch.attention_mask.to(self.device)
                    
                    # Fix 2: Dimensionality Mismatch
                    # batch.x_num is (Batch, Features, Time). Model expects (Batch, Features).
                    # We apply Mean Pooling over the time dimension (dim=2).
                    num = batch.x_num.to(self.device).mean(dim=2) 
                    
                    target = batch.y.to(self.device)

                    opt1.zero_grad()
                    logits1 = self.entity.head1(ids, mask, num)
                    loss1 = loss_fn(logits1, target)
                    loss1.backward()
                    opt1.step()

                    opt2.zero_grad()
                    logits2 = self.entity.head2(ids, mask, num)
                    loss2 = loss_fn(logits2, target)
                    loss2.backward()
                    opt2.step()
                    
                    progress.advance(task)

            # Delegate validation and stacking to Evaluator
            metrics = evaluator.evaluate(val_loader, fit_stacker=True)
            val_acc = metrics.get("accuracy", 0.0)
            val_f1 = metrics.get("f1_macro", 0.0)
            
            self.observer.track_metric("icd_val_acc", val_acc, step=epoch)
            self.observer.track_metric("icd_val_f1", val_f1, step=epoch)
            self.observer.log("INFO", f"ICDTrainer: Epoch {epoch} val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

            if val_acc > self.entity.best_metric:
                self.entity.best_metric = val_acc
                self.registry.save_model(self.entity.state_dict(), self.cfg.artifacts.model_name, "best")
                if self.entity.stacker.model:
                     self.registry.save_sklearn(self.entity.stacker.model, f"{self.cfg.artifacts.stacker_name}_best")
                self.observer.log("INFO", f"ICDTrainer: New best accuracy {val_acc:.4f} at epoch {epoch}")

        self.observer.log("INFO", "ICDTrainer: Training complete.")