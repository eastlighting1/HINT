import torch
import torch.nn as nn
import h5py
import json
import numpy as np
from pathlib import Path
from typing import Optional, List
from torch.utils.data import DataLoader
from ..common.base import BaseDomainService

# 순환 참조 방지를 위해 Top-level import 제거
# from .trainer import ICDTrainer
# from .evaluator import ICDEvaluator

from ....domain.entities import ICDModelEntity
from ....domain.vo import ICDConfig
from ....foundation.interfaces import TelemetryObserver, Registry, StreamingSource
from ....infrastructure.datasource import HDF5StreamingSource, collate_tensor_batch
from ....infrastructure.networks import get_network_class

class ICDService(BaseDomainService):
    """Service that trains and evaluates ICD models with partial label learning."""
    def __init__(self, config: ICDConfig, registry: Registry, observer: TelemetryObserver, **kwargs):
        super().__init__(observer)
        self.cfg = config
        self.registry = registry
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ignored_indices = []

    def execute(self) -> None:
        """Run the ICD training and evaluation workflow."""
        # [수정] 메서드 내부에서 임포트 (Lazy Import)
        from .trainer import ICDTrainer
        from .evaluator import ICDEvaluator

        cache_dir = Path(self.cfg.data.data_cache_dir)
        stats_path = cache_dir / "stats.json"
        
        if not stats_path.exists():
            self.observer.log("ERROR", f"Stats file not found at {stats_path}")
            return

        self.observer.log("INFO", "ICDService: Stage 1/4 loading class statistics.")
        with open(stats_path, "r") as f: 
            stats = json.load(f)
        
        num_classes = len(stats["icd_classes"])
        self.ignored_indices = [i for i, label in enumerate(stats["icd_classes"]) if label in ["__MISSING__", "__OTHER__"]]
        
        prefix = self.cfg.data.input_h5_prefix
        
        self.observer.log("INFO", "ICDService: Stage 2/4 initializing HDF5 sources.")
        try:
            train_source = HDF5StreamingSource(cache_dir / f"{prefix}_train.h5", label_key="candidates")
            val_source = HDF5StreamingSource(cache_dir / f"{prefix}_val.h5", label_key="candidates")
            test_source = HDF5StreamingSource(cache_dir / f"{prefix}_test.h5", label_key="candidates")
        except Exception as e:
            self.observer.log("ERROR", f"Failed to initialize HDF5 sources: {e}")
            return
        
        self.observer.log("INFO", "ICDService: Stage 3/4 preparing training loaders.")
        target_counts = np.ones(num_classes)

        dl_tr = DataLoader(train_source, batch_size=self.cfg.batch_size, collate_fn=collate_tensor_batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        dl_val = DataLoader(val_source, batch_size=self.cfg.batch_size, collate_fn=collate_tensor_batch, num_workers=4, pin_memory=True)
        dl_test = DataLoader(test_source, batch_size=self.cfg.batch_size, collate_fn=collate_tensor_batch, num_workers=4, pin_memory=True)

        models_to_run = self.cfg.models_to_run if self.cfg.models_to_run else ["MedBERT"]
        best_model_name = None
        best_score = float("-inf")

        for model_name in models_to_run:
            self.observer.log("INFO", f"ICDService: Stage 4/4 starting workflow for model={model_name}.")
            try:
                model_cfg = self.cfg.model_configs.get(model_name, {})
                NetworkClass = get_network_class(model_name)
                with h5py.File(train_source.h5_path, 'r') as f:
                    # [수정] X_val 대신 X_num 사용 (ETL 변경 사항 반영)
                    input_dim = f['X_num'].shape[1]
                    seq_len = f['X_num'].shape[2]

                net_kwargs = dict(model_cfg)
                net_kwargs.setdefault("dropout", self.cfg.dropout)
                network = NetworkClass(num_classes=num_classes, input_dim=input_dim, seq_len=seq_len, **net_kwargs)
                
                if hasattr(network, "embedding_dim"):
                    in_features = network.embedding_dim
                elif hasattr(network, "fc"):
                    in_features = network.fc.in_features
                else:
                    in_features = 128
                
                if num_classes > 10000:
                    cutoffs = [2000, 8000]
                else:
                    cutoffs = [num_classes // 4, num_classes // 2]
                
                adaptive_head = nn.AdaptiveLogSoftmaxWithLoss(
                    in_features=in_features, n_classes=num_classes, cutoffs=cutoffs, div_value=4.0
                )
                network.add_module("adaptive_head", adaptive_head)

                entity_name = f"{self.cfg.artifacts.model_name}_{model_name}"
                model_entity = ICDModelEntity(network)
                model_entity.name = entity_name

                use_amp = False if model_name == "DCNv2" else self.cfg.use_amp
                lr_override = model_cfg.get("lr")
                if lr_override is None and model_name == "DCNv2":
                    lr_override = self.cfg.lr * 0.1
                trainer = ICDTrainer(
                    self.cfg,
                    model_entity,
                    self.registry,
                    self.observer,
                    self.device,
                    class_freq=target_counts,
                    use_amp=use_amp,
                    lr_override=lr_override,
                )
                evaluator = ICDEvaluator(self.cfg, model_entity, self.registry, self.observer, self.device)
                
                trainer.train(dl_tr, dl_val, evaluator)
                
                best_state = self.registry.load_model(entity_name, tag="best", device=str(self.device))
                if best_state:
                    model_entity.model.load_state_dict(best_state)
                
                test_metrics = evaluator.evaluate(dl_test)
                self.observer.log("INFO", f"Final Test Metrics: {json.dumps(test_metrics, indent=2)}")
                cand_acc = test_metrics.get("candidate_accuracy")
                if cand_acc is not None and cand_acc > best_score:
                    best_score = cand_acc
                    best_model_name = model_name

            except Exception as e:
                self.observer.log("ERROR", f"Failed to run model {model_name}: {str(e)}")
                import traceback
                self.observer.log("ERROR", traceback.format_exc())

        self.best_model_name = best_model_name or (models_to_run[0] if models_to_run else None)

    def generate_intervention_dataset(self, cnn_config) -> None:
        """Inject X_icd latent context into the intervention dataset."""
        self.observer.log("INFO", "ICDService: Starting feature injection.")
        
        models_to_run = self.cfg.models_to_run if self.cfg.models_to_run else ["MedBERT"]
        if not models_to_run:
            return
        model_name = getattr(self, "best_model_name", None) or models_to_run[0]
        
        cache_dir = Path(self.cfg.data.data_cache_dir)
        with open(cache_dir / "stats.json", "r") as f:
            stats = json.load(f)
        num_classes = len(stats["icd_classes"])
        
        icd_prefix = self.cfg.data.input_h5_prefix
        cnn_prefix = cnn_config.data.input_h5_prefix
        
        sample_file = cache_dir / f"{icd_prefix}_train.h5"
        with h5py.File(sample_file, "r") as f:
            # [수정] X_val 대신 X_num 사용
            num_feats = f["X_num"].shape[1]
            seq_len = f["X_num"].shape[2]

        try:
            NetworkClass = get_network_class(model_name)
            model_cfg = self.cfg.model_configs.get(model_name, {})
            net_kwargs = dict(model_cfg)
            net_kwargs.setdefault("dropout", self.cfg.dropout)
            network = NetworkClass(num_classes=num_classes, input_dim=num_feats, seq_len=seq_len, **net_kwargs)
            
            if hasattr(network, "embedding_dim"):
                in_features = network.embedding_dim
            elif hasattr(network, "fc"):
                in_features = network.fc.in_features
            else:
                in_features = 128
            
            if num_classes > 10000:
                cutoffs = [2000, 8000]
            else:
                cutoffs = [num_classes // 4, num_classes // 2]
            
            adaptive_head = nn.AdaptiveLogSoftmaxWithLoss(in_features, num_classes, cutoffs=cutoffs, div_value=4.0)
            network.add_module("adaptive_head", adaptive_head)
            
            entity_name = f"{self.cfg.artifacts.model_name}_{model_name}"
            best_state = self.registry.load_model(entity_name, tag="best", device=str(self.device))
            if best_state:
                network.load_state_dict(best_state)
            
            network.to(self.device)
            network.eval()
            
        except Exception as e:
            self.observer.log("ERROR", f"Model init failed: {e}")
            return

        splits = ["train", "val", "test"]
        for split in splits:
            src_path = cache_dir / f"{icd_prefix}_{split}.h5"
            tgt_path = cache_dir / f"{cnn_prefix}_{split}.h5"
            
            if not src_path.exists() or not tgt_path.exists():
                continue
            
            self.observer.log("INFO", f"Injecting context into {tgt_path}...")
            source = HDF5StreamingSource(src_path, label_key="candidates")
            dl = DataLoader(source, batch_size=self.cfg.batch_size, collate_fn=collate_tensor_batch, num_workers=4)
            
            all_embeds = []
            
            try:
                with torch.no_grad():
                    for batch in dl:
                        # [수정] 통합된 x_num 사용
                        x_num = batch.x_num.to(self.device).float()
                        embeddings = network(x_num=x_num, return_embeddings=True)
                        all_embeds.append(embeddings.cpu().numpy())
                
                if not all_embeds:
                    continue
                full_embeds = np.concatenate(all_embeds, axis=0)
                
                with h5py.File(tgt_path, "a") as f_tgt:
                    if "X_icd" in f_tgt:
                        del f_tgt["X_icd"]
                    f_tgt.create_dataset("X_icd", data=full_embeds)
                    
                self.observer.log("INFO", f"Injected X_icd (Latent) shape: {full_embeds.shape}")
                
            except Exception as e:
                self.observer.log("ERROR", f"Injection failed for {split}: {e}")
