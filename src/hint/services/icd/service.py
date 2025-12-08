import json
from functools import partial
from pathlib import Path
from typing import Optional, List, Any, Dict

import lime
import lime.lime_tabular
import numpy as np
import polars as pl
import shap
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, WeightedRandomSampler

from ...domain.entities import ICDModelEntity
from ...domain.vo import ICDConfig
from ...foundation.interfaces import TelemetryObserver, Registry, StreamingSource
from ...infrastructure.components import CBFocalLoss, XGBoostStacker
from ...infrastructure.datasource import ICDDataset, custom_collate, _to_python_list
from ...infrastructure.networks import MedBERTClassifier

class ICDService:
    """
    Train and explain the ICD coding models for structured and text inputs.

    Args:
        config: ICD configuration loaded from Hydra.
        registry: Registry used to persist artifacts.
        observer: Telemetry observer for logging and progress.
        train_source: Optional streaming source for training data.
        val_source: Optional streaming source for validation data.
        test_source: Optional streaming source for test data.
    """
    def __init__(
        self, 
        config: ICDConfig,
        registry: Registry, 
        observer: TelemetryObserver,
        train_source: Optional[StreamingSource] = None,
        val_source: Optional[StreamingSource] = None,
        test_source: Optional[StreamingSource] = None
    ):
        self.cfg = config
        self.registry = registry
        self.observer = observer
        
        self.entity: Optional[ICDModelEntity] = None 
        
        self.train_source = train_source 
        self.val_source = val_source
        self.test_source = test_source
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.feats = []
        self.le = None

    def _prepare_data(self):
        self.observer.log("INFO", "ICD Service: Loading data from training source.")

        df_pl = self.train_source.load()
        df = df_pl.to_pandas()

        self.observer.log("INFO", f"ICD Service: Loaded {len(df)} rows. Parsing ICD codes.")

        total_steps = len(df)
        clean_labels = []

        with self.observer.create_progress("Preprocessing ICD Codes", total=total_steps) as progress:
            task = progress.add_task("Parsing", total=total_steps)
            raw_codes = df["ICD9_CODES"].tolist()
            for val in raw_codes:
                clean_labels.append(_to_python_list(val))
                progress.advance(task)

        df["label_list"] = clean_labels
        df["raw_label"] = df["label_list"].apply(lambda lst: lst[0] if len(lst) > 0 else None)
        df["raw_label"] = df["raw_label"].fillna("__MISSING__")

        vc = df["raw_label"].value_counts()
        top_k = self.cfg.top_k_labels

        if top_k and vc.shape[0] > top_k:
            keep = set(vc.head(top_k).index.tolist())
            df["raw_label_filtered"] = df["raw_label"].where(df["raw_label"].isin(keep), "__OTHER__")
            le = LabelEncoder()
            df["target_label"] = le.fit_transform(df["raw_label_filtered"])
        else:
            le = LabelEncoder()
            df["raw_label_filtered"] = df["raw_label"]
            df["target_label"] = le.fit_transform(df["raw_label"])

        self.le = le
        self.observer.log("INFO", f"ICD Service: Final classes={len(le.classes_)}")

        all_names = list(le.classes_)
        class_set = set(all_names)
        has_other = "__OTHER__" in class_set

        label_lists = df["label_list"].tolist()
        filtered_labels = df["raw_label_filtered"].tolist()
        candidate_names = []
        candidate_indices = []

        with self.observer.create_progress("Generating Candidates", total=total_steps) as progress:
            task = progress.add_task("Candidates", total=total_steps)
            for codes, backup in zip(label_lists, filtered_labels):
                out = []
                for c in codes:
                    if c in class_set:
                        out.append(c)
                    elif has_other:
                        out.append("__OTHER__")
                if not out:
                    out = [backup]
                unique_cands = list(set(out))
                candidate_names.append(unique_cands)
                candidate_indices.append(le.transform(unique_cands).tolist())
                progress.advance(task)

        df["candidate_names"] = candidate_names
        df["candidate_indices"] = candidate_indices

        drop_cols = [
            "SUBJECT_ID",
            "HADM_ID",
            "ICUSTAY_ID",
            "HOUR_IN",
            "ICD9_CODES",
            "target_label",
            "label_list",
            "raw_label",
            "raw_label_filtered",
            "candidate_names",
            "candidate_indices",
            "VENT_CLASS",
        ]
        feats = [c for c in df.select_dtypes("number").columns if c not in drop_cols]
        self.feats = feats

        for f_name in feats:
            df[f_name] = df[f_name].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

        df["target_label"] = df["target_label"].astype(np.int64)

        try:
            tr_val, te = train_test_split(df, test_size=self.cfg.test_split_size, random_state=42, stratify=df["target_label"])
        except ValueError:
            tr_val, te = train_test_split(df, test_size=self.cfg.test_split_size, random_state=42, stratify=None)

        try:
            tr, v = train_test_split(tr_val, test_size=self.cfg.val_split_size, random_state=42, stratify=tr_val["target_label"])
        except ValueError:
            tr, v = train_test_split(tr_val, test_size=self.cfg.val_split_size, random_state=42, stratify=None)

        self.observer.log("INFO", f"ICD Service: Split sizes: Train={len(tr)}, Val={len(v)}, Test={len(te)}")
        return tr, v, te, feats, "target_label", le

    def train(self) -> None:
        tr_df, val_df, test_df, feats, label_col, le = self._prepare_data()

        num_feats = len(feats)
        num_classes = len(le.classes_)
        self.observer.log("INFO", f"ICD Service: Initializing model with num_features={num_feats}, num_classes={num_classes}")

        head1 = MedBERTClassifier(self.cfg.model_name, num_num=num_feats, num_cls=num_classes)
        head2 = MedBERTClassifier(self.cfg.model_name, num_num=num_feats, num_cls=num_classes)
        stacker = XGBoostStacker(self.cfg.xgb_params)

        self.entity = ICDModelEntity(head1, head2, stacker)
        self.entity.to(str(self.device))
        self.observer.log("INFO", f"ICD Service: Starting training on {self.device}...")

        target_counts = tr_df[label_col].value_counts()
        class_weights_map = {cls: 1.0 / (count ** self.cfg.sampler_alpha) for cls, count in target_counts.items()}
        sample_weights = torch.tensor(tr_df[label_col].map(class_weights_map).values, dtype=torch.double)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        collate_fn = partial(custom_collate, tokenizer=tokenizer, max_length=self.cfg.max_length)

        ds_tr = ICDDataset(tr_df, feats, label_col, "label_list")
        ds_val = ICDDataset(val_df, feats, label_col, "label_list")
        ds_test = ICDDataset(test_df, feats, label_col, "label_list")

        self.ds_val = ds_val
        self.ds_test = ds_test

        dl_tr = DataLoader(
            ds_tr,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=self.cfg.batch_size,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
        )

        opt1 = torch.optim.Adam(self.entity.head1.parameters(), lr=self.cfg.lr)
        opt2 = torch.optim.Adam(self.entity.head2.parameters(), lr=self.cfg.lr)

        class_freq = np.array([target_counts.get(i, 1) for i in range(num_classes)])
        loss_fn = CBFocalLoss(class_freq, beta=self.cfg.cb_beta, gamma=self.cfg.focal_gamma, device=str(self.device))

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
        
        if not val_logits_list: return 0.0

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
        Run SHAP and LIME explanations for the trained ICD ensemble.
        """
        self.observer.log("INFO", "ICD Service: Starting XAI pipeline...")
        
        if not hasattr(self, 'ds_val') or not hasattr(self, 'ds_test'):
            self.observer.log("WARNING", "ICD Service: Validation/Test sets not found. Run train() first to prepare data.")
            return

        self.entity.to(str(self.device))
        self.entity.head1.eval()
        self.entity.head2.eval()

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        collate_fn = partial(custom_collate, tokenizer=tokenizer, max_length=self.cfg.max_length)
        
        dl_val = DataLoader(self.ds_val, batch_size=self.cfg.batch_size, collate_fn=collate_fn, num_workers=self.cfg.num_workers)
        dl_te = DataLoader(self.ds_test, batch_size=self.cfg.batch_size, collate_fn=collate_fn, num_workers=self.cfg.num_workers)

        bg_num_list = []
        bg_sample_count = 0
        shap_bg_size = self.cfg.xai_bg_size
        
        for batch in dl_val:
            bg_num_list.append(batch["num"])
            bg_sample_count += batch["num"].shape[0]
            if bg_sample_count >= shap_bg_size: break
        
        if not bg_num_list:
            self.observer.log("ERROR", "ICD Service: No background samples for SHAP.")
            return

        bg_num_tensor = torch.cat(bg_num_list, dim=0)[:shap_bg_size]
        bg_lime = bg_num_tensor.numpy()

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

        def shap_predict_fn(x_shap: np.ndarray) -> np.ndarray:
            n_samples = x_shap.shape[0]
            eval_bs = 512
            probs_list = []
            
            if n_samples == 0: return np.array([])
            
            for i in range(0, n_samples, eval_bs):
                batch_x = x_shap[i : i + eval_bs]
                curr_bs = batch_x.shape[0]
                
                num_tensor = torch.tensor(batch_x, dtype=torch.float32).to(self.device)

                ids_tensor = sample_ids[0:1].to(self.device).expand(curr_bs, -1)
                mask_tensor = sample_mask[0:1].to(self.device).expand(curr_bs, -1)

                with torch.no_grad():
                    o1 = self.entity.head1(input_ids=ids_tensor, mask=mask_tensor, numerical=num_tensor)
                    o2 = self.entity.head2(input_ids=ids_tensor, mask=mask_tensor, numerical=num_tensor)
                    avg_logits = (o1 + o2) / 2
                    batch_probs = torch.softmax(avg_logits, dim=-1).cpu().numpy()
                    probs_list.append(batch_probs)
            
            return np.concatenate(probs_list, axis=0)

        self.observer.log("INFO", "ICD Service: Running KernelExplainer...")
        explainer = shap.KernelExplainer(shap_predict_fn, bg_lime)
        shap_values = explainer.shap_values(x_sample_np, nsamples=self.cfg.xai_nsamples)
        
        np.save(self.registry.dirs["metrics"] / "icd_shap_values.npy", shap_values)
        self.observer.log("INFO", "ICD Service: SHAP completed.")

        self.observer.log("INFO", "ICD Service: Running LIME...")
        class_names = list(self.le.classes_) if self.le else [str(i) for i in range(len(bg_lime[0]))]
        
        lime_exp = lime.lime_tabular.LimeTabularExplainer(
            training_data=bg_lime,
            feature_names=self.feats,
            class_names=class_names,
            mode="classification"
        )
        
        inst_num = sample_x[0].numpy()
        lime_explanation = lime_exp.explain_instance(
            data_row=inst_num,
            predict_fn=shap_predict_fn,
            num_features=20
        )
        
        lime_out_path = self.registry.dirs["metrics"] / "icd_lime_explanation.html"
        lime_explanation.save_to_file(str(lime_out_path))
        self.observer.log("INFO", f"ICD Service: LIME explanation saved to {lime_out_path}")
