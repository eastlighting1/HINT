"""Summary of the service module.

Longer description of the module purpose and usage.
"""

import torch

import torch.nn as nn

import h5py

import json

import numpy as np

from pathlib import Path

from typing import Optional, List

from torch.utils.data import DataLoader

from ..common.base import BaseDomainService

from ....domain.entities import ICDModelEntity

from ....domain.vo import ICDConfig

from ....foundation.interfaces import TelemetryObserver, Registry, StreamingSource

from ....infrastructure.datasource import HDF5StreamingSource, collate_tensor_batch

from ....infrastructure.networks import get_network_class



class ICDService(BaseDomainService):

    """Summary of ICDService purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
        best_model_name (Any): Description of best_model_name.
        cfg (Any): Description of cfg.
        device (Any): Description of device.
        ignored_indices (Any): Description of ignored_indices.
        registry (Any): Description of registry.
    """

    def __init__(self, config: ICDConfig, registry: Registry, observer: TelemetryObserver, **kwargs):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
            config (Any): Description of config.
            registry (Any): Description of registry.
            observer (Any): Description of observer.
            kwargs (Any): Description of kwargs.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        super().__init__(observer)

        self.cfg = config

        self.registry = registry

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.ignored_indices = []

    def _summarize_candidates(self, h5_path: Path, num_classes: int, class_labels: Optional[List[str]] = None):
        with h5py.File(h5_path, "r") as f:
            y = f["y"][:]

        valid = y >= 0
        n_samples = int(y.shape[0])
        cand_sizes = valid.sum(axis=1) if n_samples > 0 else np.array([], dtype=np.int64)
        class_inclusion = np.zeros(num_classes, dtype=np.int64)

        for row in y:
            vals = row[row >= 0]
            if vals.size:
                class_inclusion[np.unique(vals)] += 1

        best_class = int(class_inclusion.argmax()) if n_samples > 0 else -1
        best_ratio = float(class_inclusion[best_class] / n_samples) if n_samples > 0 else 0.0

        top_idx = np.argsort(class_inclusion)[-5:][::-1].tolist() if n_samples > 0 else []
        top5 = []
        for idx in top_idx:
            label = class_labels[idx] if class_labels and idx < len(class_labels) else str(idx)
            top5.append({"index": idx, "label": label, "ratio": float(class_inclusion[idx] / n_samples)})

        summary = {
            "samples": n_samples,
            "cand_size_mean": float(cand_sizes.mean()) if n_samples > 0 else 0.0,
            "cand_size_p50": float(np.median(cand_sizes)) if n_samples > 0 else 0.0,
            "cand_size_p90": float(np.percentile(cand_sizes, 90)) if n_samples > 0 else 0.0,
            "cand_size_max": int(cand_sizes.max()) if n_samples > 0 else 0,
            "best_class_index": best_class,
            "best_class_label": class_labels[best_class] if class_labels and best_class >= 0 else None,
            "best_class_inclusion": best_ratio,
            "top5_inclusion": top5,
        }

        return summary, class_inclusion

    def _resolve_model_entries(self):
        if self.cfg.model_testing:
            models = self.cfg.models_to_run or list(self.cfg.model_configs.keys())
        else:
            models = ["DCNv2"]
        return [(name, self.cfg.model_configs.get(name, {})) for name in models]


    def execute(self) -> None:

        """Summary of execute.
        
        Longer description of the execute behavior and usage.
        
        Args:
            None (None): This function does not accept arguments.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """



        from .trainer import ICDTrainer

        from .evaluator import ICDEvaluator



        cache_dir = Path(self.cfg.data.data_cache_dir)

        stats_path = cache_dir / "stats.json"
        self.observer.log(
            "INFO",
            f"[2.1] Device ready. device={self.device} cache_dir={cache_dir}",
        )



        if not stats_path.exists():

            self.observer.log("ERROR", f"[2.1] Stats file missing. path={stats_path}")

            return



        self.observer.log("INFO", "[2.2] Loading ICD class statistics.")

        with open(stats_path, "r") as f:

            stats = json.load(f)



        num_classes = len(stats["icd_classes"])

        self.ignored_indices = [i for i, label in enumerate(stats["icd_classes"]) if label in ["__MISSING__", "__OTHER__"]]
        self.observer.log(
            "INFO",
            f"[2.2] ICD classes loaded. count={num_classes} ignored={len(self.ignored_indices)}",
        )



        prefix = self.cfg.data.input_h5_prefix



        self.observer.log("INFO", "[2.3] Initializing HDF5 sources.")

        try:

            train_source = HDF5StreamingSource(cache_dir / f"{prefix}_train.h5", label_key="y")

            val_source = HDF5StreamingSource(cache_dir / f"{prefix}_val.h5", label_key="y")

            test_source = HDF5StreamingSource(cache_dir / f"{prefix}_test.h5", label_key="y")

        except Exception as e:

            self.observer.log("ERROR", f"[2.3] HDF5 sources init failed. error={e}")

            return



        self.observer.log("INFO", "[2.4] Preparing training loaders.")

        class_labels = stats.get("icd_classes", [])
        report = {"num_classes": num_classes, "splits": {}}
        target_counts = np.ones(num_classes, dtype=np.float32)

        for split_name in ("train", "val", "test"):
            split_path = cache_dir / f"{prefix}_{split_name}.h5"
            if not split_path.exists():
                self.observer.log("WARNING", f"[2.4] Missing split file. path={split_path}")
                continue
            summary, inclusion = self._summarize_candidates(split_path, num_classes, class_labels)
            report["splits"][split_name] = summary
            if split_name == "train":
                target_counts = inclusion.astype(np.float32)

        if report["splits"]:
            report_path = cache_dir / "icd_label_report.json"
            try:
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)
                self.observer.log("INFO", f"[2.4] ICD label report saved. path={report_path}")
                train_summary = report["splits"].get("train", {})
                if train_summary:
                    self.observer.log(
                        "INFO",
                        f"[2.4] Train label stats best_class_inclusion={train_summary.get('best_class_inclusion', 0.0):.4f} "
                        f"cand_size_mean={train_summary.get('cand_size_mean', 0.0):.2f}",
                    )
            except Exception as e:
                self.observer.log("WARNING", f"[2.4] Failed to write ICD label report. error={e}")

        target_counts = np.where(target_counts <= 0, 1.0, target_counts)
        class_weights = None
        if target_counts is not None:
            power = float(getattr(self.cfg, "class_weight_power", 0.5))
            weights = 1.0 / np.power(target_counts, power)
            weights = weights / float(np.mean(weights))
            clip = float(getattr(self.cfg, "class_weight_clip", 0.0))
            if clip > 0:
                weights = np.clip(weights, 0.0, clip)
            class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)



        num_workers = self.cfg.num_workers

        pin_memory = self.cfg.pin_memory



        dl_tr = DataLoader(

            train_source,

            batch_size=self.cfg.batch_size,

            collate_fn=collate_tensor_batch,

            shuffle=True,

            num_workers=num_workers,

            pin_memory=pin_memory,

            drop_last=True,

        )

        dl_val = DataLoader(

            val_source,

            batch_size=self.cfg.batch_size,

            collate_fn=collate_tensor_batch,

            num_workers=num_workers,

            pin_memory=pin_memory,

        )

        dl_test = DataLoader(

            test_source,

            batch_size=self.cfg.batch_size,

            collate_fn=collate_tensor_batch,

            num_workers=num_workers,

            pin_memory=pin_memory,

        )



        model_entries = self._resolve_model_entries()
        models_to_run = [name for name, _ in model_entries]

        best_model_name = None

        best_score = float("-inf")

        self.observer.log("INFO", f"[2.5] Model roster ready. models={models_to_run}")

        for model_name, model_cfg in model_entries:

            self.observer.log("INFO", f"[2.5] Model workflow start. model={model_name}")

            try:

                NetworkClass = get_network_class(model_name)

                with h5py.File(train_source.h5_path, "r") as f:

                    input_dim = f["X_num"].shape[1]

                    seq_len = f["X_num"].shape[2]

                self.observer.log(
                    "INFO",
                    f"[2.5] Model shape resolved. model={model_name} input_dim={input_dim} seq_len={seq_len}",
                )



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



                if self.cfg.loss_type in ("adaptive_softmax", "adaptive_clpl"):
                    adaptive_head = nn.AdaptiveLogSoftmaxWithLoss(
                        in_features=in_features, n_classes=num_classes, cutoffs=cutoffs, div_value=4.0
                    )
                    network.add_module("adaptive_head", adaptive_head)



                entity_name = f"{self.cfg.artifacts.model_name}_{model_name}"

                model_entity = ICDModelEntity(network, num_classes=num_classes)

                model_entity.name = entity_name



                use_amp = False if model_name == "DCNv2" else self.cfg.use_amp

                lr_override = model_cfg.get("lr")

                if lr_override is None and model_name == "DCNv2":

                    lr_override = self.cfg.lr * 0.1

                self.observer.log(
                    "INFO",
                    f"[2.5] Trainer setup. use_amp={use_amp} lr_override={lr_override}",
                )
                trainer = ICDTrainer(

                    self.cfg,

                    model_entity,

                    self.registry,

                    self.observer,

                    self.device,

                    class_freq=target_counts,

                    ignored_indices=self.ignored_indices,

                    num_classes=num_classes,
                    use_amp=use_amp,

                    lr_override=lr_override,

                )

                evaluator = ICDEvaluator(
                    self.cfg,
                    model_entity,
                    self.registry,
                    self.observer,
                    self.device,
                    ignored_indices=self.ignored_indices,
                    class_weights=class_weights,
                )



                self.observer.log("INFO", f"[2.5] Training start. model={model_name}")
                trainer.train(dl_tr, dl_val, evaluator)
                self.observer.log("INFO", f"[2.5] Training complete. model={model_name}")



                best_state = self.registry.load_model(entity_name, tag="best", device=str(self.device))

                if best_state:

                    model_entity.model.load_state_dict(best_state)



                self.observer.log("INFO", f"[2.5] Evaluation start. model={model_name}")
                test_metrics = evaluator.evaluate(dl_test)
                self.observer.log("INFO", f"[2.5] Evaluation complete. model={model_name}")

                self.observer.log("INFO", f"[TEST METRICS] {json.dumps(test_metrics, indent=2)}")

                cand_acc = test_metrics.get("candidate_accuracy")

                if cand_acc is not None and cand_acc > best_score:

                    best_score = cand_acc

                    best_model_name = model_name



            except Exception as e:

                self.observer.log("ERROR", f"[2.5] Model run failed. model={model_name} error={str(e)}")

                import traceback

                self.observer.log("ERROR", traceback.format_exc())



        self.best_model_name = best_model_name or (models_to_run[0] if models_to_run else None)
        self.observer.log("INFO", f"[2.5] Best model selected. model={self.best_model_name}")
    def generate_intervention_dataset(self, intervention_config) -> None:

        """Summary of generate_intervention_dataset.
        
        Longer description of the generate_intervention_dataset behavior and usage.
        
        Args:
            intervention_config (Any): Description of intervention_config.
        
        Returns:
            None: Description of the return value.
        
        Raises:
            Exception: Description of why this exception might be raised.
        """

        self.observer.log("INFO", "[2B.1] Feature injection started")



        model_entries = self._resolve_model_entries()
        models_to_run = [name for name, _ in model_entries]

        if not models_to_run:

            return

        model_name = getattr(self, "best_model_name", None) or models_to_run[0]
        self.observer.log("INFO", f"[2B.1] Using model. model={model_name}")



        cache_dir = Path(self.cfg.data.data_cache_dir)

        with open(cache_dir / "stats.json", "r") as f:

            stats = json.load(f)

        num_classes = len(stats["icd_classes"])



        icd_prefix = self.cfg.data.input_h5_prefix

        intervention_prefix = intervention_config.data.input_h5_prefix



        sample_file = cache_dir / f"{icd_prefix}_train.h5"

        with h5py.File(sample_file, "r") as f:

            num_feats = f["X_num"].shape[1]

            seq_len = f["X_num"].shape[2]

        self.observer.log(
            "INFO",
            f"[2B.1] Extracted input shape. input_dim={num_feats} seq_len={seq_len}",
        )



        try:

            NetworkClass = get_network_class(model_name)

            model_cfg = dict(next((cfg for name, cfg in model_entries if name == model_name), {}))

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



            if self.cfg.loss_type in ("adaptive_softmax", "adaptive_clpl"):
                adaptive_head = nn.AdaptiveLogSoftmaxWithLoss(in_features, num_classes, cutoffs=cutoffs, div_value=4.0)
                network.add_module("adaptive_head", adaptive_head)



            entity_name = f"{self.cfg.artifacts.model_name}_{model_name}"

            best_state = self.registry.load_model(entity_name, tag="best", device=str(self.device))

            if best_state:

                network.load_state_dict(best_state)



            network.to(self.device)

            network.eval()
            self.observer.log("INFO", "[2B.2] Model loaded and set to eval")



        except Exception as e:

            self.observer.log("ERROR", f"[2B.2] Model init failed. error={e}")

            return



        splits = ["train", "val", "test"]
        self.observer.log("INFO", "[2B.3] Embedding extraction start")

        for split in splits:

            src_path = cache_dir / f"{icd_prefix}_{split}.h5"
            tgt_path = cache_dir / f"{intervention_prefix}_{split}.h5"



            if not src_path.exists() or not tgt_path.exists():

                continue



            self.observer.log("INFO", f"[2B.3] Injecting context. target={tgt_path}")

            source = HDF5StreamingSource(src_path, label_key="y")

            dl = DataLoader(source, batch_size=self.cfg.batch_size, collate_fn=collate_tensor_batch, num_workers=4)



            all_embeds = []

            src_stay_ids = None



            try:

                with h5py.File(src_path, "r") as f_src:

                    src_stay_ids = f_src["stay_ids"][:] if "stay_ids" in f_src else None



                with torch.no_grad():

                    for batch in dl:



                        x_num = batch.x_num.to(self.device).float()

                        embeddings = network(x_num=x_num, return_embeddings=True)

                        all_embeds.append(embeddings.cpu().numpy())



                if not all_embeds:

                    self.observer.log("WARNING", f"[2B.3] No embeddings for split. split={split}")
                    continue

                full_embeds = np.concatenate(all_embeds, axis=0)



                with h5py.File(tgt_path, "a") as f_tgt:

                    tgt_stay_ids = f_tgt["stay_ids"][:] if "stay_ids" in f_tgt else None

                    if tgt_stay_ids is None or src_stay_ids is None:

                        raise ValueError("Missing stay_ids for ICD/context alignment.")



                    if full_embeds.shape[0] != src_stay_ids.shape[0]:

                        raise ValueError("ICD embedding count does not match ICD stay_ids count.")



                    embed_dim = full_embeds.shape[1]

                    aligned = np.zeros((tgt_stay_ids.shape[0], embed_dim), dtype=full_embeds.dtype)

                    src_index = {int(sid): i for i, sid in enumerate(src_stay_ids)}

                    matched = 0

                    for i, sid in enumerate(tgt_stay_ids):

                        idx = src_index.get(int(sid))

                        if idx is not None:

                            aligned[i] = full_embeds[idx]

                            matched += 1



                    if "X_icd" in f_tgt:

                        del f_tgt["X_icd"]

                    f_tgt.create_dataset("X_icd", data=aligned)



                self.observer.log(
                    "INFO",
                    f"[2B.3] X_icd injected. shape={aligned.shape} matched={matched}/{aligned.shape[0]}",
                )



            except Exception as e:

                self.observer.log("ERROR", f"[2B.3] Injection failed. split={split} error={e}")

        self.observer.log("INFO", "[2B.4] Feature injection complete")
