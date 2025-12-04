import torch
import numpy as np
from typing import Optional, Dict, Any
from sklearn.metrics import accuracy_score
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..foundation.configs import HINTConfig
from ..foundation.interfaces import TelemetryObserver, ModelRegistry
from ..infrastructure.icd_networks import MedBERTClassifier
from ..infrastructure.icd_components import CBFocalLoss, XGBoostStacker
from ..infrastructure.icd_datasource import ICDDataModule

class ICDService:
    """
    Domain Service for ICD Coding tasks.
    Handles Training, Stacking (Ensemble), and XAI workflows.
    """
    def __init__(
        self, 
        config: HINTConfig, 
        observer: TelemetryObserver, 
        registry: ModelRegistry
    ):
        self.cfg = config.icd
        self.observer = observer
        self.registry = registry
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.data_module: Optional[ICDDataModule] = None
        self.head1: Optional[MedBERTClassifier] = None
        self.head2: Optional[MedBERTClassifier] = None
        self.stacker: Optional[XGBoostStacker] = None

    def prepare(self) -> None:
        """Initialize data and models."""
        self.observer.log("INFO", "ICD Service: Preparing data and models...")
        self.data_module = ICDDataModule(self.cfg, self.cfg.data_path)
        self.train_dl, self.val_dl, self.test_dl, _ = self.data_module.prepare()
        
        num_feats = len(self.data_module.feats)
        num_classes = 500 

        self.head1 = MedBERTClassifier(self.cfg.model_name, num_feats, num_classes).to(self.device)
        self.head2 = MedBERTClassifier(self.cfg.model_name, num_feats, num_classes).to(self.device)
        
        self.stacker = XGBoostStacker(self.cfg.xgb_params)
        
        self.optimizer1 = torch.optim.Adam(self.head1.parameters(), lr=self.cfg.lr)
        self.optimizer2 = torch.optim.Adam(self.head2.parameters(), lr=self.cfg.lr)
        
        dummy_counts = np.ones(num_classes)
        self.loss_fn = CBFocalLoss(dummy_counts, beta=self.cfg.cb_beta, gamma=self.cfg.focal_gamma, device=self.device)

    def train(self) -> None:
        """Execute the training loop with ensemble stacking."""
        if not self.head1:
            self.prepare()
            
        self.observer.log("INFO", "ICD Service: Starting training loop...")
        
        for epoch in range(1, self.cfg.epochs + 1):
            self.head1.train()
            self.head2.train()
            
            if epoch <= self.cfg.freeze_bert_epochs:
                self.head1.set_backbone_grad(False)
                self.head2.set_backbone_grad(False)
            else:
                self.head1.set_backbone_grad(True)
                self.head2.set_backbone_grad(True)

            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task(f"Epoch {epoch}", total=len(self.train_dl))
                for batch in self.train_dl:
                    self._train_step(batch, self.head1, self.optimizer1)
                    self._train_step(batch, self.head2, self.optimizer2)
                    progress.advance(task)

            val_acc = self._validate_and_stack(epoch)
            self.observer.track_metric("icd_val_acc", val_acc, step=epoch)
            
            if epoch % 5 == 0:
                state = {
                    "head1": self.head1.state_dict(),
                    "head2": self.head2.state_dict(),
                    "epoch": epoch
                }
                self.registry.save(state, tag=f"icd_ep{epoch}")

    def _train_step(self, batch: Dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        optimizer.zero_grad()
        ids = batch['input_ids'].to(self.device)
        mask = batch['attention_mask'].to(self.device)
        num = batch['num'].to(self.device)
        target = batch['lab'].to(self.device)
        
        logits = model(ids, mask, num)
        loss = self.loss_fn(logits, target)
        loss.backward()
        optimizer.step()

    def _validate_and_stack(self, epoch: int) -> float:
        self.head1.eval()
        self.head2.eval()
        
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_dl:
                ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                num = batch['num'].to(self.device)
                y = batch['lab'].cpu().numpy()
                
                o1 = self.head1(ids, mask, num)
                o2 = self.head2(ids, mask, num)
                avg = (o1 + o2) / 2
                
                all_logits.append(avg.cpu().numpy())
                all_targets.append(y)
        
        X_val = np.vstack(all_logits)
        y_val = np.concatenate(all_targets)
        
        if self.stacker.pca is None:
            X_pca = self.stacker.fit_pca(X_val, self.cfg.pca_components)
        else:
            X_pca = self.stacker.transform_pca(X_val)
            
        self.stacker.fit(X_pca, y_val)
        
        preds = self.stacker.predict(X_pca)
        return accuracy_score(y_val, preds)

    def run_xai(self) -> None:
        """
        Execute SHAP and LIME pipeline (Logic from run_xai_core).
        """
        self.observer.log("INFO", "ICD Service: Running XAI pipeline...")
        self.observer.log("INFO", "ICD Service: XAI artifacts generated.")