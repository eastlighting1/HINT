"""Summary of the evaluator module.

Longer description of the module purpose and usage.
"""

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

    """Summary of ModelWithTemperature purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    model (Any): Description of model.
    temperature (Any): Description of temperature.
    """

    def __init__(self, model):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        model (Any): Description of model.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super().__init__()

        self.model = model

        self.temperature = nn.Parameter(torch.ones(1) * 1.5)



    def forward(self, input):

        """Summary of forward.
        
        Longer description of the forward behavior and usage.
        
        Args:
        input (Any): Description of input.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        logits = self.model(input)

        return self.temperature_scale(logits)



    def temperature_scale(self, logits):

        """Summary of temperature_scale.
        
        Longer description of the temperature_scale behavior and usage.
        
        Args:
        logits (Any): Description of logits.
        
        Returns:
        Any: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        return logits / self.temperature



class InterventionEvaluator(BaseEvaluator):

    """Summary of InterventionEvaluator purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    cfg (Any): Description of cfg.
    entity (Any): Description of entity.
    temperature (Any): Description of temperature.
    """

    def __init__(self, config: CNNConfig, entity: InterventionModelEntity, registry, observer, device):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        config (Any): Description of config.
        entity (Any): Description of entity.
        registry (Any): Description of registry.
        observer (Any): Description of observer.
        device (Any): Description of device.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super().__init__(registry, observer, device)

        self.cfg = config

        self.entity = entity

        self.temperature = 1.0



    def calibrate(self, loader) -> None:

        """Summary of calibrate.
        
        Longer description of the calibrate behavior and usage.
        
        Args:
        loader (Any): Description of loader.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

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



        self.observer.log("INFO", "InterventionEvaluator: Stage 1/2 temperature calibration start.")

        temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.5)

        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)



        def eval():

            """Summary of eval.
            
            Longer description of the eval behavior and usage.
            
            Args:
            None (None): This function does not accept arguments.
            
            Returns:
            Any: Description of the return value.
            
            Raises:
            Exception: Description of why this exception might be raised.
            """

            optimizer.zero_grad()

            loss = nn.CrossEntropyLoss()(logits / temperature, labels)

            loss.backward()

            return loss



        optimizer.step(eval)

        self.temperature = temperature.item()

        self.observer.log("INFO", f"InterventionEvaluator: Stage 2/2 temperature calibration complete temperature={self.temperature:.4f}.")



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



    def evaluate(self, loader, **kwargs) -> Dict[str, float]:

        """Summary of evaluate.
        
        Longer description of the evaluate behavior and usage.
        
        Args:
        loader (Any): Description of loader.
        kwargs (Any): Description of kwargs.
        
        Returns:
        Dict[str, float]: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.entity.network.eval()
        self.observer.log(
            "INFO",
            f"InterventionEvaluator: Stage 1/3 starting evaluation batches={len(loader)}.",
        )



        all_probs = []

        all_labels = []

        total_loss = 0.0



        with torch.no_grad():

            for batch in loader:

                x_num = batch.x_num.to(self.device).float()

                x_cat = batch.x_cat.to(self.device).long() if batch.x_cat is not None else None

                x_icd = batch.x_icd.to(self.device).float() if batch.x_icd is not None else None

                y = batch.y.to(self.device)

                target = self._select_last_valid(y)



                logits = self.entity.network(x_num=x_num, x_cat=x_cat, x_icd=x_icd)



                scaled_logits = logits / self.temperature

                probs = torch.softmax(scaled_logits, dim=1)



                valid_mask = target != -100

                if not valid_mask.any():

                    continue

                all_probs.append(probs[valid_mask].cpu().numpy())

                all_labels.append(target[valid_mask].cpu().numpy())



                loss = nn.CrossEntropyLoss()(scaled_logits[valid_mask], target[valid_mask])

                total_loss += loss.item()



        y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)

        y_prob = np.concatenate(all_probs) if all_probs else np.zeros((0, 4), dtype=np.float32)

        y_pred = np.argmax(y_prob, axis=1)



        if y_true.size:

            class_counts = np.bincount(y_true, minlength=4)

            self.observer.log(

                "INFO",

                f"InterventionEvaluator: Stage 2/3 class_counts={class_counts.tolist()}",

            )



        acc = accuracy_score(y_true, y_pred) if y_true.size else 0.0

        f1 = f1_score(y_true, y_pred, average='macro') if y_true.size else 0.0





        classes = [0, 1, 2, 3]

        y_true_bin = label_binarize(y_true, classes=classes)









        auc_per_class = []

        for i in range(len(classes)):

            try:



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



        self.observer.log(
            "INFO",
            "InterventionEvaluator: Stage 2/3 auc_raw="
            f"{[None if np.isnan(x) else float(x) for x in auc_per_class]}",
        )





        valid_aucs = [s for s in auc_per_class if not np.isnan(s)]

        macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0



        auprc_per_class = []

        for i in range(len(classes)):

            if np.sum(y_true_bin[:, i]) > 0:

                try:

                    score = average_precision_score(y_true_bin[:, i], y_prob[:, i])

                except Exception:

                    score = float('nan')

            else:

                score = float('nan')

            auprc_per_class.append(score)

        valid_auprc = [s for s in auprc_per_class if not np.isnan(s)]

        macro_auprc = np.mean(valid_auprc) if valid_auprc else 0.0



        self.observer.log(
            "INFO",
            f"InterventionEvaluator: Stage 3/3 complete accuracy={acc:.4f} f1={f1:.4f} macro_auc={macro_auc:.4f} macro_auprc={macro_auprc:.4f}.",
        )

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
