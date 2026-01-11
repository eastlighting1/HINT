# src/hint/services/training/predict_intervention/evaluator.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from ..common.base import BaseEvaluator
from ....domain.entities import InterventionModelEntity
from ....domain.vo import CNNConfig

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        return logits / self.temperature

class InterventionEvaluator(BaseEvaluator):
    def __init__(self, config: CNNConfig, entity: InterventionModelEntity, registry, observer, device):
        super().__init__(registry, observer, device)
        self.cfg = config
        self.entity = entity
        self.temperature = 1.0

    def calibrate(self, loader) -> None:
        self.entity.network.eval()
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in loader:
                x_num = batch.x_num.to(self.device).float()
                x_cat = batch.x_cat.to(self.device).long() if batch.x_cat is not None else None
                x_icd = batch.x_icd.to(self.device).float() if batch.x_icd is not None else None
                y = batch.y[:, -1].to(self.device)
                
                logits = self.entity.network(x_num=x_num, x_cat=x_cat, x_icd=x_icd)
                logits_list.append(logits)
                labels_list.append(y)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        self.observer.log("INFO", "InterventionEvaluator: Temperature calibration start.")
        temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(logits / temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        self.temperature = temperature.item()
        self.observer.log("INFO", f"Optimal Temperature found: {self.temperature:.4f}")

    def evaluate(self, loader, **kwargs) -> Dict[str, float]:
        self.entity.network.eval()
        
        all_probs = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in loader:
                x_num = batch.x_num.to(self.device).float()
                x_cat = batch.x_cat.to(self.device).long() if batch.x_cat is not None else None
                x_icd = batch.x_icd.to(self.device).float() if batch.x_icd is not None else None
                y = batch.y[:, -1].to(self.device)
                
                logits = self.entity.network(x_num=x_num, x_cat=x_cat, x_icd=x_icd)
                
                scaled_logits = logits / self.temperature
                probs = torch.softmax(scaled_logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                
                loss = nn.CrossEntropyLoss()(scaled_logits, y)
                total_loss += loss.item()

        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        y_pred = np.argmax(y_prob, axis=1)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        # [수정] 클래스별 AUC 안전 계산
        classes = [0, 1, 2, 3] # Onset, Wean, Stay On, Stay Off
        y_true_bin = label_binarize(y_true, classes=classes)
        
        # 존재하는 클래스만 AUC 계산하도록 처리하거나, NaN 발생 시 0.0으로 처리
        # 하지만 roc_auc_score(..., average=None)은 없는 클래스에 대해 에러를 낼 수 있음
        # 안전하게 개별 컬럼별로 계산
        auc_per_class = []
        for i in range(len(classes)):
            try:
                # 해당 클래스가 하나라도 존재해야 AUC 계산 가능
                if np.sum(y_true_bin[:, i]) > 0:
                    score = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                else:
                    score = float('nan')
            except Exception:
                score = float('nan')
            auc_per_class.append(score)
            
        onset_auc = auc_per_class[0]
        wean_auc = auc_per_class[1]
        stay_on_auc = auc_per_class[2]
        stay_off_auc = auc_per_class[3]
        
        # NaN을 제외한 평균으로 Macro AUC 계산
        valid_aucs = [s for s in auc_per_class if not np.isnan(s)]
        macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
        
        try:
            macro_auprc = average_precision_score(y_true_bin, y_prob, average='macro')
        except:
            macro_auprc = 0.0

        return {
            "loss": total_loss / len(loader),
            "accuracy": acc,
            "f1_score": f1,
            "macro_auc": macro_auc,
            "onset_auc": 0.0 if np.isnan(onset_auc) else onset_auc,
            "wean_auc": 0.0 if np.isnan(wean_auc) else wean_auc,
            "stay_on_auc": 0.0 if np.isnan(stay_on_auc) else stay_on_auc,
            "stay_off_auc": 0.0 if np.isnan(stay_off_auc) else stay_off_auc,
            "macro_auprc": macro_auprc
        }