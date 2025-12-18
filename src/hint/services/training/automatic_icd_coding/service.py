import torch
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from functools import partial
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..common.base import BaseDomainService
from .trainer import ICDTrainer
from .evaluator import ICDEvaluator
from ....domain.entities import ICDModelEntity
from ....domain.vo import ICDConfig, CNNConfig
from ....foundation.interfaces import TelemetryObserver, Registry, StreamingSource
from ....infrastructure.components import XGBoostStacker
from ....infrastructure.datasource import ICDDataset, custom_collate, _to_python_list
from ....infrastructure.networks import MedBERTClassifier

class ICDService(BaseDomainService):
    def __init__(
        self, 
        config: ICDConfig,
        registry: Registry, 
        observer: TelemetryObserver,
        train_source: Optional[StreamingSource] = None,
        val_source: Optional[StreamingSource] = None,
        test_source: Optional[StreamingSource] = None
    ):
        super().__init__(observer)
        self.cfg = config
        self.registry = registry
        self.train_source = train_source 
        self.val_source = val_source
        self.test_source = test_source
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.entity: Optional[ICDModelEntity] = None
        self.feats = []
        self.le = None

    def _prepare_data(self):
        """
        Prepare encoded ICD dataset, labels, and splits for model training.
        Full logic preserved from original code.
        """
        self.observer.log("INFO", "ICD Service: Starting data preparation with training source load.")
        df_pl = self.train_source.load()
        df = df_pl.to_pandas()
        
        self.observer.log("INFO", f"ICD Service: Loaded {len(df)} rows; parsing ICD codes to label lists.")

        raw_codes = df["ICD9_CODES"].tolist()
        clean_labels = [_to_python_list(val) for val in raw_codes]
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

        # Candidate generation logic (Full Code)
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

        df["candidate_names"] = candidate_names
        df["candidate_indices"] = candidate_indices
        
        drop_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN", "ICD9_CODES", "target_label", "label_list", "raw_label", "raw_label_filtered", "candidate_names", "candidate_indices", "VENT_CLASS"]
        feats = [c for c in df.select_dtypes("number").columns if c not in drop_cols]
        self.feats = feats
        
        for f_name in feats:
            df[f_name] = df[f_name].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        self.observer.log("INFO", f"ICD Service: Selected {len(feats)} numeric features for modeling.")
            
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

    def execute(self) -> None:
        """
        Orchestrates the training process:
        1. Prepare Data
        2. Initialize Models (Entity)
        3. Setup DataLoader & Sampler
        4. Run Trainer
        """
        tr_df, val_df, _test_df, feats, label_col, le = self._prepare_data()
        
        num_feats = len(feats)
        num_classes = len(le.classes_)
        
        head1 = MedBERTClassifier(self.cfg.model_name, num_num=num_feats, num_cls=num_classes)
        head2 = MedBERTClassifier(self.cfg.model_name, num_num=num_feats, num_cls=num_classes)
        stacker = XGBoostStacker(self.cfg.xgb_params)
        
        self.entity = ICDModelEntity(head1, head2, stacker)
        # Entity to device handled in Trainer

        target_counts = tr_df[label_col].value_counts()
        class_weights_map = {cls: 1.0 / (count ** self.cfg.sampler_alpha) for cls, count in target_counts.items()}
        default_weight = 1.0
        sample_weights = torch.tensor([class_weights_map.get(x, default_weight) for x in tr_df[label_col]], dtype=torch.double)
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        collate_fn = partial(custom_collate, tokenizer=tokenizer, max_length=self.cfg.max_length)

        ds_tr = ICDDataset(tr_df, feats, label_col, "label_list")
        ds_val = ICDDataset(val_df, feats, label_col, "label_list")

        dl_tr = DataLoader(ds_tr, batch_size=self.cfg.batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=self.cfg.num_workers)
        dl_val = DataLoader(ds_val, batch_size=self.cfg.batch_size, collate_fn=collate_fn, num_workers=self.cfg.num_workers)

        class_freq = np.array([target_counts.get(i, 1) for i in range(num_classes)])
        
        trainer = ICDTrainer(self.cfg, self.entity, self.registry, self.observer, self.device, class_freq)
        evaluator = ICDEvaluator(self.cfg, self.entity, self.registry, self.observer, self.device)
        
        trainer.train(dl_tr, dl_val, evaluator)

    def _ensure_entity_loaded(self) -> bool:
        if self.entity is not None:
            return True

        try:
            state = self.registry.load_model(self.cfg.artifacts.model_name, "best", str(self.device))
            
            # Re-run prep to ensure features and encoder exist for reconstruction
            if not self.feats or not self.le:
                self._prepare_data()

            if self.feats and self.le:
                num_feats = len(self.feats)
                num_classes = len(self.le.classes_)
                head1 = MedBERTClassifier(self.cfg.model_name, num_num=num_feats, num_cls=num_classes)
                head2 = MedBERTClassifier(self.cfg.model_name, num_num=num_feats, num_cls=num_classes)
                stacker = XGBoostStacker(self.cfg.xgb_params)
                self.entity = ICDModelEntity(head1, head2, stacker)
                self.entity.load_state_dict(state)
                self.entity.to(str(self.device))
                self.observer.log("INFO", "ICD Service: Model state reloaded for inference.")
                return True
            self.observer.log("ERROR", "ICD Service: Missing feature metadata; run training before inference.")
        except Exception as exc:
            self.observer.log("WARNING", f"ICD Service: Unable to load saved model ({exc}); ensure train workflow completed.")
        return False

    def generate_intervention_dataset(self, cnn_config: CNNConfig) -> None:
        """
        Run inference on the coding dataset and emit enriched H5 artifacts for the intervention task.
        """
        self.observer.log("INFO", "ICD Service: Starting intervention dataset generation.")

        if not self._ensure_entity_loaded():
            return

        self.entity.head1.eval()
        self.entity.head2.eval()
        self.entity.to(str(self.device))

        self.observer.log("INFO", "ICD Service: Preparing inference map from training source.")
        df_pl = self.train_source.load()
        df = df_pl.to_pandas()

        drop_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "HOUR_IN", "ICD9_CODES", "target_label", "label_list", "raw_label", "VENT_CLASS"]
        feats = [c for c in df.select_dtypes("number").columns if c not in drop_cols]
        for f_name in feats:
            df[f_name] = df[f_name].fillna(0.0).astype(np.float32)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)

        sid_to_idx = {sid: i for i, sid in enumerate(df["ICUSTAY_ID"].values)}

        raw_codes = df["ICD9_CODES"].tolist()
        texts = [" ".join(_to_python_list(c)) for c in raw_codes]

        self.observer.log("INFO", "ICD Service: Tokenizing texts for full dataset inference.")
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt"
        )
        input_ids_all = encodings["input_ids"]
        mask_all = encodings["attention_mask"]
        nums_all = torch.tensor(df[feats].values, dtype=torch.float32)

        cache_dir = Path(self.cfg.data.data_cache_dir or cnn_config.data.data_cache_dir)
        input_prefix = self.cfg.data.input_h5_prefix
        output_prefix = self.cfg.data.output_h5_prefix

        target_files = [f"{input_prefix}_{split}.h5" for split in ["train", "val", "test"]]
        output_dim = self.entity.head1.fc.out_features

        for fname in target_files:
            src_path = cache_dir / fname
            if not src_path.exists():
                self.observer.log("WARNING", f"ICD Service: Skipping missing cache file {src_path.name}.")
                continue

            split_name = fname.replace(f"{input_prefix}_", "").replace(".h5", "")
            dst_path = cache_dir / f"{output_prefix}_{split_name}.h5"
            self.observer.log("INFO", f"ICD Service: Augmenting {src_path.name} -> {dst_path.name}")

            with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
                for key in src.keys():
                    if key in ("X_icd", self.cfg.data.inferred_col_name):
                        continue
                    dst.copy(src[key], key)

                sids = src["sid"][:]
                total = len(sids)
                prob_ds = dst.create_dataset("X_icd", (total, output_dim), dtype=np.float32)
                code_ds = dst.create_dataset(self.cfg.data.inferred_col_name, (total,), dtype=np.int64)

                chunk_size = 512
                for i in range(0, total, chunk_size):
                    batch_sids = sids[i: i + chunk_size]
                    batch_indices = [sid_to_idx.get(sid, -1) for sid in batch_sids]
                    valid_indices = [idx for idx in batch_indices if idx != -1]

                    if not valid_indices:
                        prob_ds[i: i + len(batch_sids)] = np.zeros((len(batch_sids), output_dim), dtype=np.float32)
                        code_ds[i: i + len(batch_sids)] = -1
                        continue

                    b_ids = input_ids_all[valid_indices].to(self.device)
                    b_mask = mask_all[valid_indices].to(self.device)
                    b_num = nums_all[valid_indices].to(self.device)

                    with torch.no_grad():
                        o1 = self.entity.head1(b_ids, b_mask, b_num)
                        o2 = self.entity.head2(b_ids, b_mask, b_num)
                        avg_logits = (o1 + o2) / 2
                        probs = torch.softmax(avg_logits, dim=-1)

                    res_arr = np.zeros((len(batch_sids), output_dim), dtype=np.float32)
                    argmax_arr = np.zeros((len(batch_sids),), dtype=np.int64)

                    curr_valid = 0
                    for j, idx in enumerate(batch_indices):
                        if idx != -1:
                            prob_vec = probs[curr_valid].cpu().numpy()
                            res_arr[j] = prob_vec
                            argmax_arr[j] = int(prob_vec.argmax())
                            curr_valid += 1
                        else:
                            argmax_arr[j] = -1

                    prob_ds[i: i + len(batch_sids)] = res_arr
                    code_ds[i: i + len(batch_sids)] = argmax_arr

        self.observer.log("INFO", "ICD Service: Augmentation complete.")