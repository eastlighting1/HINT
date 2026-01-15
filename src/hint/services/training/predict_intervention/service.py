"""Summary of the service module.

Longer description of the module purpose and usage.
"""

import torch

import h5py

import numpy as np

import json

from pathlib import Path

from typing import Optional

from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..common.base import BaseDomainService











from ....domain.entities import InterventionModelEntity

from ....domain.vo import CNNConfig

from ....foundation.interfaces import TelemetryObserver, Registry

from ....infrastructure.datasource import collate_tensor_batch, HDF5StreamingSource

from ....infrastructure.networks import get_network_class



class InterventionService(BaseDomainService):

    """Summary of InterventionService purpose.
    
    Longer description of the class behavior and usage.
    
    Attributes:
    cfg (Any): Description of cfg.
    device (Any): Description of device.
    entity (Any): Description of entity.
    registry (Any): Description of registry.
    test_ds (Any): Description of test_ds.
    train_ds (Any): Description of train_ds.
    val_ds (Any): Description of val_ds.
    """



    def __init__(self, config: CNNConfig, registry: Registry, observer: TelemetryObserver, entity: InterventionModelEntity, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Optional[Dataset] = None):

        """Summary of __init__.
        
        Longer description of the __init__ behavior and usage.
        
        Args:
        config (Any): Description of config.
        registry (Any): Description of registry.
        observer (Any): Description of observer.
        entity (Any): Description of entity.
        train_dataset (Any): Description of train_dataset.
        val_dataset (Any): Description of val_dataset.
        test_dataset (Any): Description of test_dataset.
        
        Returns:
        None: Description of the return value.
        
        Raises:
        Exception: Description of why this exception might be raised.
        """

        super().__init__(observer)

        self.cfg = config

        self.registry = registry

        self.entity = entity

        self.train_ds = train_dataset

        self.val_ds = val_dataset

        self.test_ds = test_dataset

        self.device = "cuda" if torch.cuda.is_available() else "cpu"



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



        from .trainer import InterventionTrainer

        from .evaluator import InterventionEvaluator



        self.observer.log(
            "INFO",
            f"InterventionService: Stage 1/4 building dataloaders and class weights device={self.device}.",
        )

        self._ensure_expected_h5_paths()



        class_weights = None
        sample_weights = None

        try:

            if isinstance(self.train_ds, HDF5StreamingSource):

                with h5py.File(self.train_ds.h5_path, 'r') as f:

                    label_key = self.train_ds.label_key

                    if label_key in f:

                        y_train = f[label_key][:]

                        if y_train.ndim > 1:
                            valid = y_train != -100
                            has_valid = valid.any(axis=1)
                            last_from_end = np.argmax(np.flip(valid, axis=1), axis=1)
                            last_idx = y_train.shape[1] - 1 - last_from_end
                            rows = np.arange(y_train.shape[0])
                            y_last = y_train[rows, last_idx]
                            y_last = np.where(has_valid, y_last, -100)
                            y_train = y_last

                        valid_mask = y_train != -100

                        y_valid = y_train[valid_mask].astype(np.int64)



                        if len(y_valid) > 0:

                            classes = np.unique(y_valid)

                            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_valid)

                            weight_map = {c: w for c, w in zip(classes, weights)}

                            final_weights = [weight_map.get(i, 1.0) for i in range(4)]

                            class_weights = torch.FloatTensor(final_weights).to(self.device)

                            class_weights = torch.clamp(class_weights, max=50.0)

                            self.observer.log("INFO", f"Computed Class Weights: {class_weights.tolist()}")

                            if getattr(self.cfg, "use_weighted_sampler", False):
                                sample_weights = np.zeros_like(y_train, dtype=np.float32)
                                if valid_mask.any():
                                    weight_arr = class_weights.detach().cpu().numpy()
                                    sample_weights[valid_mask] = weight_arr[y_train[valid_mask]]

        except Exception as e:

            self.observer.log("WARNING", f"Failed to compute class weights: {e}")

        sampler = None
        if sample_weights is not None:
            sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)

        dl_tr = DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_tensor_batch,
        )

        dl_val = DataLoader(self.val_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=4, collate_fn=collate_tensor_batch)

        dl_test = None

        if self.test_ds:

            dl_test = DataLoader(self.test_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=4, collate_fn=collate_tensor_batch)



        self.observer.log(
            "INFO",
            f"InterventionService: Stage 1/4 loaders ready train={len(dl_tr)} val={len(dl_val)} test={len(dl_test) if dl_test else 0}.",
        )

        self.observer.log("INFO", "InterventionService: Stage 2/4 initializing HINT model and input dimensions.")



        with h5py.File(self.train_ds.h5_path, 'r') as f:
            num_feats = f["X_num"].shape[1]
            icd_dim = 0
            if "X_icd" in f:
                icd_dim = f["X_icd"].shape[1]
                self.observer.log("INFO", f"Found ICD context with dim={icd_dim}")
            else:
                self.observer.log(
                    "WARNING",
                    f"X_icd not found in dataset ({self.train_ds.h5_path}). Gating disabled.",
                )

        self.observer.log("INFO", f"InterventionService: Stage 2/4 input_dim={num_feats} icd_dim={icd_dim}.")




        NetworkClass = get_network_class("TCN")





        vocab_sizes = []

        if hasattr(self.train_ds, "get_real_vocab_sizes"):

             vocab_sizes = self.train_ds.get_real_vocab_sizes()



        network = NetworkClass(

            in_chs=num_feats,

            n_cls=4,

            vocab_sizes=vocab_sizes,

            icd_dim=icd_dim,

            embed_dim=self.cfg.embed_dim if hasattr(self.cfg, "embed_dim") else 128,

            cat_embed_dim=self.cfg.cat_embed_dim if hasattr(self.cfg, "cat_embed_dim") else 32,

            head_drop=self.cfg.dropout,

            tcn_drop=self.cfg.tcn_dropout if hasattr(self.cfg, "tcn_dropout") else 0.2,

            kernel=self.cfg.tcn_kernel_size if hasattr(self.cfg, "tcn_kernel_size") else 5,

            layers=self.cfg.tcn_layers if hasattr(self.cfg, "tcn_layers") else 5

        )



        entity_name = self.cfg.artifacts.model_name

        self.entity = InterventionModelEntity(network)

        self.entity.name = entity_name



        trainer = InterventionTrainer(self.cfg, self.entity, self.registry, self.observer, self.device, class_weights=class_weights)

        evaluator = InterventionEvaluator(self.cfg, self.entity, self.registry, self.observer, self.device)



        self.observer.log("INFO", "InterventionService: Stage 3/4 entering training loop.")

        trainer.train(dl_tr, dl_val, evaluator)

        self.observer.log("INFO", "InterventionService: Stage 3/4 training complete.")

        self.observer.log("INFO", "InterventionService: Stage 4/4 entering test phase.")

        if dl_test:

            best_state = self.registry.load_model(entity_name, tag="best", device=str(self.device))

            if best_state:

                self.entity.load_state_dict(best_state)

                self.entity.to(self.device)



            metrics = evaluator.evaluate(dl_test)

            self.observer.log("INFO", f"Final Test Results: {json.dumps(metrics, indent=2)}")
            self.observer.log("INFO", "InterventionService: Stage 4/4 test evaluation complete.")

        else:

            self.observer.log("WARNING", "InterventionService: No test dataset available, skipping test evaluation.")


    def _ensure_expected_h5_paths(self) -> None:
        if not isinstance(self.train_ds, HDF5StreamingSource):
            return

        cache_dir = Path(self.cfg.data.data_cache_dir)
        prefix = self.cfg.data.input_h5_prefix
        expected = {
            "train": cache_dir / f"{prefix}_train.h5",
            "val": cache_dir / f"{prefix}_val.h5",
            "test": cache_dir / f"{prefix}_test.h5",
        }

        datasets = {
            "train": self.train_ds,
            "val": self.val_ds,
            "test": self.test_ds,
        }

        for split, ds in datasets.items():
            if ds is None or not isinstance(ds, HDF5StreamingSource):
                continue
            expected_path = expected[split]
            if ds.h5_path == expected_path:
                continue
            if expected_path.exists():
                self.observer.log(
                    "WARNING",
                    f"InterventionService: {split} dataset path mismatch ({ds.h5_path}); switching to {expected_path}.",
                )
                datasets[split] = HDF5StreamingSource(
                    expected_path,
                    seq_len=self.cfg.seq_len,
                    label_key=ds.label_key,
                )
            else:
                self.observer.log(
                    "WARNING",
                    f"InterventionService: {split} dataset path mismatch ({ds.h5_path}); expected {expected_path} missing.",
                )

        self.train_ds = datasets["train"]
        self.val_ds = datasets["val"]
        self.test_ds = datasets["test"]
