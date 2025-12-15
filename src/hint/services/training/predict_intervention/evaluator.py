import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import shap
from pathlib import Path
from typing import List, Dict, Any
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from ....foundation.interfaces import TelemetryObserver, Registry, StreamingSource
from ....domain.entities import InterventionModelEntity
from ....domain.vo import CNNConfig
from ....infrastructure.components import TemperatureScaler
from ....infrastructure.datasource import collate_tensor_batch

class EvaluationService:
    """
    Domain service for Calibration, Evaluation and XAI of CNN.
    Fully ported from CNN.py including numeric-only SHAP and tabular LIME logic.
    """
    def __init__(
        self,
        config: CNNConfig,
        registry: Registry,
        observer: TelemetryObserver,
        entity: InterventionModelEntity,
        device: str
    ):
        self.cfg = config
        self.registry = registry
        self.observer = observer
        self.entity = entity
        self.device = device
        self.CLASS_NAMES = ["ONSET", "WEAN", "STAY ON", "STAY OFF"]

    def calibrate(self, val_source: StreamingSource) -> None:
        self.observer.log("INFO", "EvaluationService: Starting calibration (Temperature Scaling)...")
        self.entity.to(self.device)
        self.entity.network.eval()
        
        dl_val = DataLoader(
            val_source, 
            batch_size=self.cfg.batch_size, 
            num_workers=4,
            collate_fn=collate_tensor_batch
        )
        ts = TemperatureScaler().to(self.device)
        
        logits_list, labels_list = [], []
        with torch.no_grad():
            for batch in dl_val:
                batch = batch.to(self.device)
                logits = self.entity.network(batch.x_num, batch.x_cat, batch.x_icd)
                logits_list.append(logits)
                labels_list.append(batch.y)
        
        temp = ts.fit(logits_list, labels_list, self.device)
        self.entity.temperature = temp
        self.observer.log("INFO", f"EvaluationService: Optimal temperature T={temp:.4f}")
        
        self.registry.save_model(self.entity.state_dict(), self.cfg.artifacts.model_name, "calibrated")

    def evaluate(self, test_source: StreamingSource) -> None:
        self.observer.log("INFO", "EvaluationService: Running test evaluation...")
        self.entity.to(self.device)
        self.entity.network.eval()
        
        dl_te = DataLoader(
            test_source, 
            batch_size=self.cfg.batch_size, 
            num_workers=4,
            collate_fn=collate_tensor_batch
        )
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in dl_te:
                batch = batch.to(self.device)
                logits = self.entity.network(batch.x_num, batch.x_cat, batch.x_icd)
                probs = torch.softmax(logits / self.entity.temperature, dim=1)
                preds = probs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "macro_f1": f1_score(all_labels, all_preds, average="macro")
        }
        self.registry.save_json(metrics, "test_metrics.json")
        self.observer.log("INFO", f"EvaluationService: Test Accuracy={metrics['accuracy']:.4f}")

    def run_xai(self, val_source: StreamingSource, test_source: StreamingSource) -> None:
        self.observer.log("INFO", "EvaluationService: Starting XAI pipeline...")
        self.entity.to(self.device)
        self.entity.network.eval()
        
        feature_info_path = Path(self.cfg.data.data_cache_dir) / "feature_info.json"
        
        if feature_info_path.exists():
            with open(feature_info_path, "r", encoding="utf-8") as f:
                feat_info = json.load(f)
        else:
            self.observer.log("WARNING", f"Feature info not found at {feature_info_path}. Using empty info.")
            feat_info = {}

        cat_vocab_sizes = feat_info.get("vocab_info", {})
        seq_len = self.cfg.seq_len
        n_cat_feats = len(cat_vocab_sizes)
        icd_dim = 0
        try:
            probe = val_source[0] if len(val_source) > 0 else test_source[0]
            if probe.x_icd is not None:
                icd_dim = probe.x_icd.shape[0]
        except Exception:
            icd_dim = 0

        self._run_shap_numeric(val_source, test_source, n_cat_feats, seq_len, icd_dim)

        self._run_lime_tabular(test_source, feat_info, n_cat_feats, seq_len, icd_dim)

    def _run_shap_numeric(self, val_source, test_source, n_cat_feats, seq_len, icd_dim):
        """
        Generate SHAP values using numeric features while holding categorical inputs constant.

        Args:
            val_source: Validation data source used to build the SHAP background set.
            test_source: Test data source used to sample records for explanation.
            n_cat_feats: Number of categorical features in the dataset.
            seq_len: Sequence length used by the CNN model.
        """
        self.observer.log("INFO", "EvaluationService: Computing numeric-only SHAP...")
        
        max_bg = 128
        max_samples = 256
        
        dl_val = DataLoader(
            val_source, 
            batch_size=self.cfg.batch_size, 
            shuffle=False,
            collate_fn=collate_tensor_batch
        )
        bg_num_list = []
        total_bg = 0
        for batch in dl_val:
            bg_num_list.append(batch.x_num)
            total_bg += batch.x_num.size(0)
            if total_bg >= max_bg: break
            
        if not bg_num_list:
            self.observer.log("WARNING", "No validation data for SHAP background.")
            return
            
        bg_num = torch.cat(bg_num_list, dim=0)[:max_bg].to(self.device)

        dl_te = DataLoader(
            test_source, 
            batch_size=self.cfg.batch_size, 
            shuffle=False,
            collate_fn=collate_tensor_batch
        )
        samp_num_list = []
        total_samp = 0
        for batch in dl_te:
            samp_num_list.append(batch.x_num)
            total_samp += batch.x_num.size(0)
            if total_samp >= max_samples: break
            
        samp_num = torch.cat(samp_num_list, dim=0)[:max_samples].to(self.device)

        class ProbWrapperNumOnly(nn.Module):
            def __init__(self, base_model, temp, n_cat, length, icd_size):
                super().__init__()
                self.base_model = base_model
                self.temp = temp
                self.n_cat = n_cat
                self.length = length
                self.icd_size = icd_size
            
            def forward(self, x_num):
                batch_size = x_num.size(0)
                if self.n_cat > 0:
                    x_cat_dummy = torch.zeros(batch_size, self.n_cat, self.length, dtype=torch.long, device=x_num.device)
                else:
                    x_cat_dummy = torch.zeros(batch_size, 0, self.length, dtype=torch.long, device=x_num.device)
                if self.icd_size > 0:
                    x_icd_dummy = torch.zeros(batch_size, self.icd_size, dtype=torch.float32, device=x_num.device)
                else:
                    x_icd_dummy = None
                logits = self.base_model(x_num, x_cat_dummy, x_icd_dummy) / self.temp
                return F.softmax(logits, dim=1)

        prob_model = ProbWrapperNumOnly(self.entity.network, self.entity.temperature, n_cat_feats, seq_len, icd_dim).to(self.device)
        prob_model.eval()

        explainer = shap.DeepExplainer(prob_model, bg_num)
        shap_values = explainer.shap_values(samp_num, check_additivity=False)
        
        shap_data = {
            "background_num": bg_num.detach().cpu().numpy().tolist(),
            "sample_num": samp_num.detach().cpu().numpy().tolist(),
            "shap_values": [s.tolist() for s in shap_values],
            "class_names": self.CLASS_NAMES
        }
        self.registry.save_json(shap_data, "cnn_shap_values.json")
        self.observer.log("INFO", "EvaluationService: SHAP values saved.")

    def _run_lime_tabular(self, test_source, feat_info, n_cat_feats, seq_len, icd_dim):
        """
        Generate LIME explanations on flattened tabular views of the latest timestep.

        Args:
            test_source: Test data source providing samples for explanation.
            feat_info: Metadata describing numeric and categorical feature names.
            n_cat_feats: Number of categorical features in the dataset.
            seq_len: Sequence length used by the CNN model.
        """
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            self.observer.log("WARNING", "LIME not installed, skipping.")
            return

        self.observer.log("INFO", "EvaluationService: Computing LIME tabular explanations...")
        
        max_lime_samples = 2000
        num_explanations = 50
        
        dl_te = DataLoader(
            test_source, 
            batch_size=self.cfg.batch_size, 
            shuffle=False,
            collate_fn=collate_tensor_batch
        )
        tab_rows, tab_labels = [], []
        total_rows = 0
        
        for batch in dl_te:
            last_num = batch.x_num[:, :, -1].cpu().numpy()
            if n_cat_feats > 0 and batch.x_cat is not None:
                last_cat = batch.x_cat[:, :, -1].cpu().numpy()
                row = np.concatenate([last_num, last_cat], axis=1)
            else:
                row = last_num
            if icd_dim > 0 and batch.x_icd is not None:
                icd_vals = batch.x_icd.cpu().numpy()
                row = np.concatenate([row, icd_vals], axis=1)
            
            tab_rows.append(row)
            tab_labels.append(batch.y.numpy())
            total_rows += row.shape[0]
            if total_rows >= max_lime_samples: break
            
        if not tab_rows:
            return

        X_tab = np.vstack(tab_rows)
        y_tab = np.concatenate(tab_labels)
        
        if "base_feats_numeric" in feat_info:
            base_feats_num = feat_info["base_feats_numeric"]
            n_feats_num = feat_info["n_feats_numeric"]
            cat_vocab_sizes = feat_info.get("vocab_info", {})
            
            if n_feats_num == len(base_feats_num) * 3:
                suffixes = ["_obs", "_gap", "_pred"]
                num_feat_names = [f"{name}{suf}" for suf in suffixes for name in base_feats_num]
            else:
                num_feat_names = [f"num_{i}" for i in range(n_feats_num)]
                
            cat_feat_names_last = [f"{name}_last" for name in cat_vocab_sizes.keys()]
            icd_feat_names = [f"{self.cfg.data.inferred_col_name}_{i}" for i in range(icd_dim)] if icd_dim > 0 else []
            feat_names_all = num_feat_names + cat_feat_names_last + icd_feat_names
            categorical_features_idx = list(range(len(num_feat_names), len(num_feat_names) + n_cat_feats))
        else:
            self.observer.log("WARNING", "Missing feature info for LIME. Generating generic names.")
            feat_names_all = [f"feat_{i}" for i in range(X_tab.shape[1])]
            categorical_features_idx = []
            cat_vocab_sizes = {}
            n_feats_num = X_tab.shape[1]

        def predict_fn(X):
            X = np.atleast_2d(X)
            n = X.shape[0]
            num_part = X[:, :n_feats_num]
            if n_cat_feats > 0 and (n_feats_num + n_cat_feats <= X.shape[1]):
                cat_part = X[:, n_feats_num : n_feats_num + n_cat_feats]
            else:
                cat_part = None
            icd_part = None
            icd_start = n_feats_num + (n_cat_feats if cat_part is not None else 0)
            if icd_dim > 0 and icd_start < X.shape[1]:
                icd_part = X[:, icd_start : icd_start + icd_dim]
                
            num_seq = np.repeat(num_part[:, :, None], seq_len, axis=2).astype(np.float32)
            
            if cat_part is not None:
                cat_part_int = np.rint(cat_part).astype(np.int64)
                cat_part_clipped_list = []
                for idx, (name, vocab_size) in enumerate(cat_vocab_sizes.items()):
                    vals = np.clip(cat_part_int[:, idx], 0, vocab_size - 1)
                    cat_part_clipped_list.append(vals[:, None])
                cat_part_arr = np.concatenate(cat_part_clipped_list, axis=1)
                cat_seq = np.repeat(cat_part_arr[:, :, None], seq_len, axis=2).astype(np.int64)
            else:
                cat_seq = np.zeros((n, 0, seq_len), dtype=np.int64)
            
            with torch.no_grad():
                t_num = torch.from_numpy(num_seq).to(self.device)
                t_cat = torch.from_numpy(cat_seq).to(self.device)
                t_icd = torch.from_numpy(icd_part.astype(np.float32)).to(self.device) if icd_part is not None else None
                logits = self.entity.network(t_num, t_cat, t_icd) / self.entity.temperature
                probs = F.softmax(logits, dim=1).cpu().numpy()
            return probs

        explainer = LimeTabularExplainer(
            training_data=X_tab,
            feature_names=feat_names_all,
            class_names=self.CLASS_NAMES,
            categorical_features=categorical_features_idx if categorical_features_idx else None,
            mode="classification"
        )
        
        lime_results = []
        limit = min(num_explanations, X_tab.shape[0])
        
        for idx in range(limit):
            exp = explainer.explain_instance(X_tab[idx], predict_fn, num_features=20, top_labels=1)
            probs = predict_fn(X_tab[idx:idx+1])[0]
            top_label = int(exp.top_labels[0])
            
            lime_results.append({
                "sample_index": int(idx),
                "true_label": int(y_tab[idx]),
                "pred_label": int(np.argmax(probs)),
                "probs": probs.tolist(),
                "explanation": exp.as_list(label=top_label)
            })
            
        self.registry.save_json({"explanations": lime_results}, "cnn_lime_explanations.json")
        self.observer.log("INFO", "EvaluationService: LIME explanations saved.")
