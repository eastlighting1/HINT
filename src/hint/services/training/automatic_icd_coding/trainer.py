import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict
from torch.utils.data import DataLoader
from ..common.base import BaseTrainer, BaseEvaluator
from ....domain.entities import ICDModelEntity
from ....domain.vo import ICDConfig
from ....infrastructure.components import CBFocalLoss, CLPLLoss
from ....foundation.dtos import TensorBatch

class ICDTrainer(BaseTrainer):
    def __init__(self, config: ICDConfig, entity: ICDModelEntity, registry, observer, device, class_freq: np.ndarray = None, ignored_indices: Optional[List[int]] = None):
        super().__init__(registry, observer, device)
        self.cfg = config
        self.entity = entity
        self.class_freq = class_freq if class_freq is not None else np.ones(1)
        self.ignored_indices = ignored_indices if ignored_indices else []
        
                                     
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
                                                  
        self.use_adaptive_softmax = False
        self.adaptive_loss_fn = None
        
        if self.cfg.loss_type == "adaptive_softmax":
            self.use_adaptive_softmax = True
                                                                
                                                                                    
            if hasattr(self.entity.model, "embedding_dim"):
                in_features = self.entity.model.embedding_dim
            elif hasattr(self.entity.model, "fc"):                                                
                in_features = self.entity.model.fc.in_features
            else:
                                                 
                raise AttributeError("Model must have 'embedding_dim' attribute for adaptive_softmax.")
            
                                                        
            n_classes = self.entity.model.num_classes
            if n_classes > 100000:
                cutoffs = [10000, 50000, n_classes - 10000]                      
            elif n_classes > 10000:
                cutoffs = [2000, 8000]
            else:
                cutoffs = [n_classes // 4, n_classes // 2]
                
            self.adaptive_loss_fn = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=in_features,
                n_classes=n_classes,
                cutoffs=cutoffs,
                div_value=4.0
            ).to(self.device)
            
            self.observer.log("INFO", f"Initialized AdaptiveLogSoftmaxWithLoss with cutoffs: {cutoffs}")

    def _prepare_inputs(self, batch: TensorBatch) -> Dict[str, torch.Tensor]:
        inputs = {}
        inputs["x_num"] = batch.x_num.to(self.device).float()
        
        if batch.mask is not None:
            inputs["mask"] = batch.mask.to(self.device).float()
        
        if hasattr(batch, "delta") and batch.delta is not None:
            inputs["delta"] = batch.delta.to(self.device).float()

        if getattr(batch, "input_ids", None) is not None:
            inputs["input_ids"] = batch.input_ids.to(self.device).long()
        
        if getattr(batch, "attention_mask", None) is not None:
            inputs["attention_mask"] = batch.attention_mask.to(self.device).long()
            
        if batch.x_cat is not None:
            inputs["x_cat"] = batch.x_cat.to(self.device).long()
            
        if batch.x_icd is not None:
            inputs["x_icd"] = batch.x_icd.to(self.device).float()

        return inputs

    def train(self, train_loader: DataLoader, val_loader: DataLoader, evaluator: BaseEvaluator) -> None:
        self.entity.to(self.device)
        self.observer.log("INFO", f"ICDTrainer: Start training for {self.cfg.epochs} epochs (AMP + Mode: {self.cfg.loss_type}).")
        
                                                         
        params = list(self.entity.model.parameters())
        if self.use_adaptive_softmax:
            params += list(self.adaptive_loss_fn.parameters())
            
        optimizer = torch.optim.Adam(params, lr=self.cfg.lr, weight_decay=1e-5)
        
                                          
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=3
        )
        
                                                            
        standard_loss_fn = None
        if not self.use_adaptive_softmax:
            if self.cfg.loss_type == "clpl":
                standard_loss_fn = CLPLLoss().to(self.device)
            else:
                standard_loss_fn = CBFocalLoss(class_counts=self.class_freq, beta=self.cfg.cb_beta, gamma=self.cfg.focal_gamma, device=str(self.device))
        
        no_improve = 0
        
        for epoch in range(1, self.cfg.epochs + 1):
            self.entity.epoch = epoch
            self.entity.model.train()
            if self.use_adaptive_softmax:
                self.adaptive_loss_fn.train()
            
            if hasattr(self.entity.model, "set_backbone_grad"):
                self.entity.model.set_backbone_grad(not (epoch <= self.cfg.freeze_bert_epochs))
            
            epoch_loss = 0.0
            batch_count = 0
            
            with self.observer.create_progress(f"Epoch {epoch} Train", total=len(train_loader)) as progress:
                task = progress.add_task("Training", total=len(train_loader))
                
                for batch in train_loader:
                                                                       
                    target = getattr(batch, "y", None)
                    if target is not None:
                        target = target.to(self.device)
                    
                                                                             
                    batch_size = batch.x_num.size(0)
                    valid_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
                    
                                                                                                                                                   
                    if target is not None and self.ignored_indices:
                        for idx in self.ignored_indices: valid_mask &= (target != idx)
                    
                    if not valid_mask.any():
                        progress.advance(task)
                        continue
                    
                    if target is not None:
                        target = target[valid_mask]
                                                                                               
                        if self.use_adaptive_softmax and target.dim() > 1 and target.size(1) > 1:
                                                                                                                
                                                                                                    
                                                                                                             
                                                                                              
                             target = target.squeeze()
                        
                    batch_inputs = self._prepare_inputs(batch)
                    filtered_inputs = {k: v[valid_mask] for k, v in batch_inputs.items()}

                    optimizer.zero_grad()
                    
                                                 
                    with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                        
                        if self.use_adaptive_softmax:
                                                               
                            embeddings = self.entity.model(**filtered_inputs, return_embeddings=True)
                            
                                                                   
                                                                
                            _, loss = self.adaptive_loss_fn(embeddings, target)
                            
                        else:
                                                  
                            logits = self.entity.model(**filtered_inputs)
                            
                                                                                          
                                                                                 
                                                                                          
                            
                                                             
                            if self.cfg.loss_type == "clpl" and getattr(batch, "candidates", None) is not None:
                                cands = batch.candidates.to(self.device).long()
                                cands = cands[valid_mask]
                                
                                                                 
                                cand_mask = torch.zeros_like(logits, dtype=torch.float32)
                                valid_cands_idx = (cands >= 0) & (cands < logits.size(1))
                                rows = torch.arange(cands.size(0), device=self.device).unsqueeze(1).expand_as(cands)
                                
                                cand_mask[rows[valid_cands_idx], cands[valid_cands_idx]] = 1.0
                                
                                                                                            
                                if target is not None:
                                    if target.dim() == 1:
                                         cand_mask.scatter_(1, target.unsqueeze(1), 1.0)
                                    else:
                                                           
                                         cand_mask = torch.max(cand_mask, target.float())
                                
                                if self.ignored_indices:
                                    logits[:, self.ignored_indices] = -1e4

                                                                         
                                loss_main = standard_loss_fn(logits, cand_mask)

                                                                
                                probs = torch.softmax(logits, dim=1)
                                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
                                
                                reg_lambda = getattr(self.cfg, "entropy_reg_lambda", 0.01)
                                loss = loss_main + (reg_lambda * entropy)

                            else:
                                                                                            
                                if target is None:
                                    progress.advance(task)
                                    continue

                                                               
                                if getattr(batch, "candidates", None) is not None:
                                                                   
                                    pass

                                if self.ignored_indices: 
                                    logits[:, self.ignored_indices] = -1e4
                                
                                loss = standard_loss_fn(logits, target)

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                        optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    progress.advance(task)

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            self.observer.track_metric("train_loss", avg_loss, step=epoch)

                        
                                                                                                                         
                                                                                               
                                                                                              
                                                                                                                 
            metrics = evaluator.evaluate(val_loader)
            val_acc = metrics.get("accuracy", 0.0)
            val_f1 = metrics.get("f1_macro", 0.0)
            
            self.observer.track_metric("icd_val_acc", val_acc, step=epoch)
            self.observer.log("INFO", f"Epoch {epoch} | TrLoss={avg_loss:.4f} | Acc={val_acc:.4f} | F1={val_f1:.4f}")

            scheduler.step(val_acc)

            if val_acc > self.entity.best_metric:
                self.entity.best_metric = val_acc
                self.registry.save_model(self.entity.state_dict(), self.entity.name, "best")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience: break
                    
        self.observer.log("INFO", "ICDTrainer: Training complete.")