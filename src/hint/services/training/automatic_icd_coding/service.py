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
            f"ICDService: Stage 0/5 setup device={self.device} cache_dir={cache_dir}.",
        )



        if not stats_path.exists():

            self.observer.log("ERROR", f"Stats file not found at {stats_path}")

            return



        self.observer.log("INFO", "ICDService: Stage 1/5 loading class statistics.")

        with open(stats_path, "r") as f:

            stats = json.load(f)



        num_classes = len(stats["icd_classes"])

        self.ignored_indices = [i for i, label in enumerate(stats["icd_classes"]) if label in ["__MISSING__", "__OTHER__"]]
        self.observer.log(
            "INFO",
            f"ICDService: Stage 1/5 loaded classes count={num_classes} ignored={len(self.ignored_indices)}.",
        )



        prefix = self.cfg.data.input_h5_prefix



        self.observer.log("INFO", "ICDService: Stage 2/5 initializing HDF5 sources.")

        try:

            train_source = HDF5StreamingSource(cache_dir / f"{prefix}_train.h5", label_key="y")

            val_source = HDF5StreamingSource(cache_dir / f"{prefix}_val.h5", label_key="y")

            test_source = HDF5StreamingSource(cache_dir / f"{prefix}_test.h5", label_key="y")

        except Exception as e:

            self.observer.log("ERROR", f"Failed to initialize HDF5 sources: {e}")

            return



        self.observer.log("INFO", "ICDService: Stage 3/5 preparing training loaders.")

        target_counts = np.ones(num_classes)



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



        models_to_run = self.cfg.models_to_run if self.cfg.models_to_run else ["MedBERT"]

        best_model_name = None

        best_score = float("-inf")

        self.observer.log("INFO", f"ICDService: Stage 4/5 model roster={models_to_run}.")

        for model_name in models_to_run:

            self.observer.log("INFO", f"ICDService: Stage 4/5 starting workflow for model={model_name}.")

            try:

                model_cfg = self.cfg.model_configs.get(model_name, {})

                NetworkClass = get_network_class(model_name)

                with h5py.File(train_source.h5_path, "r") as f:

                    input_dim = f["X_num"].shape[1]

                    seq_len = f["X_num"].shape[2]

                self.observer.log(
                    "INFO",
                    f"ICDService: Stage 4/5 model={model_name} input_dim={input_dim} seq_len={seq_len}.",
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

                self.observer.log(
                    "INFO",
                    f"ICDService: Stage 4/5 trainer setup use_amp={use_amp} lr_override={lr_override}.",
                )
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



                self.observer.log("INFO", f"ICDService: Stage 4/5 training start model={model_name}.")
                trainer.train(dl_tr, dl_val, evaluator)
                self.observer.log("INFO", f"ICDService: Stage 4/5 training complete model={model_name}.")



                best_state = self.registry.load_model(entity_name, tag="best", device=str(self.device))

                if best_state:

                    model_entity.model.load_state_dict(best_state)



                self.observer.log("INFO", f"ICDService: Stage 4/5 evaluation start model={model_name}.")
                test_metrics = evaluator.evaluate(dl_test)
                self.observer.log("INFO", f"ICDService: Stage 4/5 evaluation complete model={model_name}.")

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
        self.observer.log("INFO", f"ICDService: Stage 5/5 selected best_model={self.best_model_name}.")



    def generate_intervention_dataset(self, cnn_config) -> None:

        """Summary of generate_intervention_dataset.
        
        Longer description of the generate_intervention_dataset behavior and usage.
        
        Args:
        cnn_config (Any): Description of cnn_config.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        self.observer.log("INFO", "ICDService: Stage 1/4 starting feature injection.")



        models_to_run = self.cfg.models_to_run if self.cfg.models_to_run else ["MedBERT"]

        if not models_to_run:

            return

        model_name = getattr(self, "best_model_name", None) or models_to_run[0]
        self.observer.log("INFO", f"ICDService: Stage 1/4 using model={model_name}.")



        cache_dir = Path(self.cfg.data.data_cache_dir)

        with open(cache_dir / "stats.json", "r") as f:

            stats = json.load(f)

        num_classes = len(stats["icd_classes"])



        icd_prefix = self.cfg.data.input_h5_prefix

        cnn_prefix = cnn_config.data.input_h5_prefix



        sample_file = cache_dir / f"{icd_prefix}_train.h5"

        with h5py.File(sample_file, "r") as f:

            num_feats = f["X_num"].shape[1]

            seq_len = f["X_num"].shape[2]

        self.observer.log(
            "INFO",
            f"ICDService: Stage 1/4 extracted input_dim={num_feats} seq_len={seq_len}.",
        )



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
            self.observer.log("INFO", "ICDService: Stage 2/4 model loaded and set to eval.")



        except Exception as e:

            self.observer.log("ERROR", f"ICDService: Stage 2/4 model init failed error={e}.")

            return



        splits = ["train", "val", "test"]
        self.observer.log("INFO", "ICDService: Stage 3/4 embedding extraction start.")

        for split in splits:

            src_path = cache_dir / f"{icd_prefix}_{split}.h5"

            tgt_path = cache_dir / f"{cnn_prefix}_{split}.h5"



            if not src_path.exists() or not tgt_path.exists():

                continue



            self.observer.log("INFO", f"ICDService: Stage 3/4 injecting context into {tgt_path}.")

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

                    self.observer.log("WARNING", f"ICDService: Stage 3/4 no embeddings for split={split}.")
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
                    f"ICDService: Stage 3/4 injected X_icd shape={aligned.shape} matched={matched}/{aligned.shape[0]}.",
                )



            except Exception as e:

                self.observer.log("ERROR", f"ICDService: Stage 3/4 injection failed split={split} error={e}.")

        self.observer.log("INFO", "ICDService: Stage 4/4 feature injection complete.")
