import torch
from pathlib import Path
from typing import Dict, Any
from ..foundation.interfaces import ModelRegistry
from ..domain.entities import TrainableEntity

class FileSystemRegistry(ModelRegistry):
    """
    File system based implementation of the ModelRegistry.

    Saves and loads model checkpoints to/from the local disk.

    Args:
        root_dir: Root directory for storing checkpoints.
    """
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, entity: TrainableEntity, tag: str = "latest") -> None:
        """
        Save the entity state to a file.

        Args:
            entity: TrainableEntity instance.
            tag: Version tag string.
        """
        path = self.root / f"{entity.id}_{tag}.pt"
        torch.save(entity.snapshot(), path)

    def load(self, entity_id: str, tag: str = "latest") -> Dict[str, Any]:
        """
        Load the entity state from a file.

        Args:
            entity_id: Entity ID string.
            tag: Version tag string.

        Returns:
            Dictionary containing the loaded state.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        path = self.root / f"{entity_id}_{tag}.pt"
        if not path.exists():
            path = self.root / f"{tag}.pt"
            
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
            
        return torch.load(path, map_location="cpu")