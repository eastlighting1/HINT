import torch
import numpy as np
from typing import List, Optional
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
        class_freq: np.ndarray = None,
        ignored_indices: Optional[List[int]] = None
    ):
        super().__init__(registry, observer, device)
        self.cfg = config
        self.entity = entity
        self.class_freq = class_freq if class_freq is not None else np.ones(1)
        self.ignored_indices = ignored_indices if ignored_indices else []

    def train(self, train_loader: DataLoader, val_loader: DataLoader, evaluator: BaseEvaluator) -> None:
        self.entity.to(self.device)
        self.observer.log("INFO", f"ICDTrainer: Start training for {self.cfg.epochs} epochs.")

        optimizer = torch.optim.Adam(self.entity.model.parameters(), lr=self.cfg.lr)
        
        loss_fn = CBFocalLoss(
            class_counts=self.class_freq, 
            beta=self.cfg.cb_beta, 
            gamma=self.cfg.focal_gamma, 
            device=str(self.device)
        )
        no_improve = 0
        for epoch in range(1, self.cfg.epochs + 1):
            self.entity.epoch = epoch
            self.observer.log("INFO", f"ICDTrainer: Epoch {epoch} training started.")
            self.entity.model.train()
            
            freeze = epoch <= self.cfg.freeze_bert_epochs
            if hasattr(self.entity.model, "set_backbone_grad"):
                self.entity.model.set_backbone_grad(not freeze)
            
            with self.observer.create_progress(f"Epoch {epoch} Train", total=len(train_loader)) as progress:
                task = progress.add_task("Training", total=len(train_loader))
                for batch in train_loader:
                    ids = batch.input_ids.to(self.device)
                    mask = batch.attention_mask.to(self.device)
                    num = batch.x_num.to(self.device).float()
                    target = batch.y.to(self.device)

                    if self.ignored_indices:
                        valid_mask = torch.ones(target.shape[0], dtype=torch.bool, device=self.device)
                        for idx in self.ignored_indices:
                            valid_mask &= (target != idx)
                        
                        if not valid_mask.any():
                            progress.advance(task)
                            continue
                            
                        ids = ids[valid_mask]
                        mask = mask[valid_mask]
                        num = num[valid_mask]
                        target = target[valid_mask]

                    optimizer.zero_grad()
                    logits = self.entity.model(input_ids=ids, attention_mask=mask, x_num=num)
                    
                    if self.ignored_indices:
                        logits[:, self.ignored_indices] = -1e9

                    loss = loss_fn(logits, target)
                    loss.backward()
                    optimizer.step()
                    
                    progress.advance(task)

            # [FIX] Retrieve and log ALL metrics (Accuracy, Precision, Recall, F1)
            metrics = evaluator.evaluate(val_loader)
            
            val_acc = metrics.get("accuracy", 0.0)
            val_prec = metrics.get("precision", 0.0)
            val_rec = metrics.get("recall", 0.0)
            val_f1 = metrics.get("f1_macro", 0.0)
            
            self.observer.track_metric("icd_val_acc", val_acc, step=epoch)
            self.observer.track_metric("icd_val_prec", val_prec, step=epoch)
            self.observer.track_metric("icd_val_rec", val_rec, step=epoch)
            self.observer.track_metric("icd_val_f1", val_f1, step=epoch)
            
            self.observer.log(
                "INFO", 
                f"ICDTrainer: Epoch {epoch} | "
                f"Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, F1={val_f1:.4f}"
            )

            if val_acc > self.entity.best_metric:
                self.entity.best_metric = val_acc
                self.registry.save_model(self.entity.state_dict(), self.entity.name, "best")
                self.observer.log("INFO", f"ICDTrainer: New best accuracy {val_acc:.4f} at epoch {epoch}")
                no_improve = 0
            else:
                no_improve += 1 # 개선되지 않음
                # patience 도달 시 중단
                if no_improve >= self.cfg.patience:
                    self.observer.log("WARNING", f"ICDTrainer: Early stopping triggered. No improvement for {self.cfg.patience} epochs.")
                    break

        self.observer.log("INFO", "ICDTrainer: Training complete.")