import torch
import numpy as np
import shap
import lime
import lime.lime_tabular
import json
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from functools import partial
from pathlib import Path
from typing import Optional, List, Any

from hint.foundation.interfaces import TelemetryObserver, Registry, StreamingSource
from hint.foundation.dtos import TensorBatch
from hint.foundation.exceptions import ModelError
from hint.domain.entities import ICDModelEntity
from hint.domain.vo import ICDConfig
from hint.infrastructure.components import CBFocalLoss
from hint.infrastructure.networks import MedBERTClassifier
from hint.infrastructure.components import XGBoostStacker

class ICDService:
    """
    Domain service for ICD coding model training and explanation.
    Ported from ICD.py
    """
    def __init__(
        self, 
        config: ICDConfig,
        registry: Registry, 
        observer: TelemetryObserver,
        entity: ICDModelEntity,
        train_source: Optional[StreamingSource] = None,
        val_source: Optional[StreamingSource] = None,
        test_source: Optional[StreamingSource] = None
    ):
        self.cfg = config
        self.registry = registry
        self.observer = observer
        self.entity = entity
        self.train_source = train_source
        self.val_source = val_source
        self.test_source = test_source
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self) -> None:
        self.observer.log("INFO", f"ICD Service: Starting training on {self.device}...")
        self.entity.to(str(self.device))
        
        # Load Data
        tr_df = self.train_source.load().to_pandas()
        
        # Sampler Logic
        target_counts = tr_df["target_label"].value_counts()
        class_weights_map = {cls: 1.0 / (count ** self.cfg.sampler_alpha) for cls, count in target_counts.items()}
        sample_weights = torch.tensor(tr_df["target_label"].map(class_weights_map).values, dtype=torch.double)
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        # Collate Fn
        from hint.infrastructure.icd_datasource import custom_collate
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        collate_fn = partial(custom_collate, tokenizer=tokenizer, max_length=self.cfg.max_length)

        dl_tr = DataLoader(self.train_source, batch_size=self.cfg.batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=self.cfg.num_workers)
        dl_val = DataLoader(self.val_source, batch_size=self.cfg.batch_size, collate_fn=collate_fn, num_workers=self.cfg.num_workers)
        
        opt1 = torch.optim.Adam(self.entity.head1.parameters(), lr=self.cfg.lr)
        opt2 = torch.optim.Adam(self.entity.head2.parameters(), lr=self.cfg.lr)
        
        # Assume 500 classes or derived from data
        num_classes = len(target_counts)
        dummy_counts = np.ones(num_classes)
        loss_fn = CBFocalLoss(dummy_counts, beta=self.cfg.cb_beta, gamma=self.cfg.focal_gamma, device=str(self.device))

        for epoch in range(1, self.cfg.epochs + 1):
            self.entity.epoch = epoch
            self._train_epoch(epoch, dl_tr, opt1, opt2, loss_fn)
            val_acc = self._validate_and_stack(epoch, dl_val)
            
            self.observer.track_metric("icd_val_acc", val_acc, step=epoch)
            
            if val_acc > self.entity.best_metric:
                self.entity.best_metric = val_acc
                self.registry.save_model(self.entity.state_dict(), "icd_model", "best")
                self.registry.save_sklearn(self.entity.stacker.model, "icd_stacker_best")
                self.observer.log("INFO", f"ICD Service: New best accuracy {val_acc:.4f} at epoch {epoch}")

    def _train_epoch(self, epoch: int, loader: DataLoader, opt1, opt2, loss_fn) -> None:
        self.entity.head1.train()
        self.entity.head2.train()
        
        freeze = epoch <= self.cfg.freeze_bert_epochs
        self.entity.head1.set_backbone_grad(not freeze)
        self.entity.head2.set_backbone_grad(not freeze)
        
        with self.observer.create_progress(f"Epoch {epoch} Train", total=len(loader)) as progress:
            task = progress.add_task("Training", total=len(loader))
            for batch in loader:
                ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                num = batch['num'].to(self.device)
                target = batch['lab'].to(self.device)
                
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

    def _validate_and_stack(self, epoch: int, loader: DataLoader) -> float:
        self.entity.head1.eval()
        self.entity.head2.eval()
        
        val_logits_list = []
        val_labels_list = []
        
        with torch.no_grad():
            for batch in loader:
                ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                num = batch['num'].to(self.device)
                y = batch['lab'].cpu().numpy()
                
                o1 = self.entity.head1(ids, mask, num)
                o2 = self.entity.head2(ids, mask, num)
                avg = (o1 + o2) / 2
                
                val_logits_list.append(avg.cpu().numpy())
                val_labels_list.append(y)
        
        X_val_ens = np.vstack(val_logits_list)
        y_val_ens = np.concatenate(val_labels_list)
        
        if self.entity.stacker.pca is None:
            X_pca = self.entity.stacker.fit_pca(X_val_ens, self.cfg.pca_components)
        else:
            X_pca = self.entity.stacker.transform_pca(X_val_ens)
            
        self.entity.stacker.fit(X_pca, y_val_ens)
        preds = self.entity.stacker.predict(X_pca)
        
        return float(accuracy_score(y_val_ens, preds))

    def run_xai(self) -> None:
        """
        Run SHAP and LIME explanations.
        Fully ported from ICD.py: run_xai_core
        """
        self.observer.log("INFO", "ICD Service: Starting XAI pipeline...")
        self.entity.to(str(self.device))
        self.entity.head1.eval()
        self.entity.head2.eval()

        # 1. Prepare Data Loaders
        from hint.infrastructure.icd_datasource import custom_collate
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        collate_fn = partial(custom_collate, tokenizer=tokenizer, max_length=self.cfg.max_length)
        
        # Assuming sources are already set
        dl_val = DataLoader(self.val_source, batch_size=self.cfg.batch_size, collate_fn=collate_fn, num_workers=self.cfg.num_workers)
        dl_te = DataLoader(self.test_source, batch_size=self.cfg.batch_size, collate_fn=collate_fn, num_workers=self.cfg.num_workers)

        # 2. Collect Background Samples (SHAP)
        bg_num_list = []
        bg_sample_count = 0
        shap_bg_size = self.cfg.xai_bg_size
        
        for batch in dl_val:
            bg_num_list.append(batch["num"])
            bg_sample_count += batch["num"].shape[0]
            if bg_sample_count >= shap_bg_size:
                break
        
        if not bg_num_list:
            self.observer.log("ERROR", "ICD Service: No background samples for SHAP.")
            return

        bg_num_tensor = torch.cat(bg_num_list, dim=0)[:shap_bg_size]
        bg_lime = bg_num_tensor.numpy()
        self.observer.log("INFO", f"ICD Service: SHAP background shape={bg_lime.shape}")

        # 3. Collect Test Samples (SHAP/LIME)
        try:
            sample_batch = next(iter(dl_te))
        except StopIteration:
            self.observer.log("ERROR", "ICD Service: No test samples for explanation.")
            return

        shap_sample_size = self.cfg.xai_sample_size
        sample_x = sample_batch["num"][:shap_sample_size]
        sample_ids = sample_batch["input_ids"][:shap_sample_size]
        sample_mask = sample_batch["attention_mask"][:shap_sample_size]
        x_sample_np = sample_x.numpy()

        # 4. Define Predict Wrapper
        def shap_predict_fn(x_shap: np.ndarray) -> np.ndarray:
            num_samples = x_shap.shape[0]
            num_tensor = torch.tensor(x_shap, dtype=torch.float32).to(self.device)
            # Repeat text inputs for numerical perturbation
            # This assumes text is constant while numeric features vary, which is typical for tabular SHAP on multi-modal
            ids_tensor = sample_ids[0:1].to(self.device).expand(num_samples, -1)
            mask_tensor = sample_mask[0:1].to(self.device).expand(num_samples, -1)

            with torch.no_grad():
                o1 = self.entity.head1(input_ids=ids_tensor, mask=mask_tensor, numerical=num_tensor)
                o2 = self.entity.head2(input_ids=ids_tensor, mask=mask_tensor, numerical=num_tensor)
                avg_logits = (o1 + o2) / 2
                probs = torch.softmax(avg_logits, dim=-1).cpu().numpy()
            return probs

        # 5. Run SHAP
        self.observer.log("INFO", "ICD Service: Running KernelExplainer...")
        explainer = shap.KernelExplainer(shap_predict_fn, bg_lime)
        shap_values = explainer.shap_values(x_sample_np, nsamples=self.cfg.xai_nsamples)
        
        self.registry.save_json({"shap_values_shape": str(np.shape(shap_values))}, "icd_shap_meta.json")
        # Save raw numpy
        np.save(self.registry.dirs["metrics"] / "icd_shap_values.npy", shap_values)
        self.observer.log("INFO", "ICD Service: SHAP completed.")

        # 6. Run LIME
        self.observer.log("INFO", "ICD Service: Running LIME...")
        feats = self.train_source.feats # Access feature names from source
        classes = self.train_source.classes
        
        lime_exp = lime.lime_tabular.LimeTabularExplainer(
            training_data=bg_lime,
            feature_names=feats,
            class_names=classes,
            mode="classification"
        )
        
        inst_num = sample_x[0].numpy()
        # For LIME, we use the same prediction function but wrapped slightly if needed
        # Re-using shap_predict_fn is usually compatible if it takes numpy array
        
        lime_explanation = lime_exp.explain_instance(
            data_row=inst_num,
            predict_fn=shap_predict_fn,
            num_features=20
        )
        
        lime_out_path = self.registry.dirs["metrics"] / "icd_lime_explanation.html"
        lime_explanation.save_to_file(str(lime_out_path))
        self.observer.log("INFO", f"ICD Service: LIME explanation saved to {lime_out_path}")
