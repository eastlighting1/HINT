import h5py
from pathlib import Path
from typing import Iterator, Union
from torch.utils.data import Dataset, DataLoader
from ..foundation.interfaces import StreamingSource
from ..foundation.dtos import TensorBatch
from ..foundation.configs import DataConfig

class HDF5Dataset(Dataset):
    """
    PyTorch Dataset wrapper for HDF5 files.

    Provides indexed access to data samples stored in HDF5 format.
    
    Args:
        h5_path: Path to the HDF5 file.
    """
    def __init__(self, h5_path: Union[str, Path]):
        self.h5_path = Path(h5_path)
        self._h5_file = None
        self._len = 0
        with h5py.File(self.h5_path, "r") as f:
            if "X_num" in f:
                self._len = len(f["X_num"])

    def _ensure_open(self):
        """Lazy loader for the HDF5 file handle."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        self._ensure_open()
        return (
            self._h5_file["X_num"][idx],
            self._h5_file["X_cat"][idx],
            self._h5_file["y"][idx],
            self._h5_file["sid"][idx]
        )

class HDF5StreamingSource(StreamingSource):
    """
    Streaming source implementation for HDF5 data.

    Streams data batches using a PyTorch DataLoader.

    Args:
        config: Data configuration containing path and batch size.
    """
    def __init__(self, config: DataConfig):
        self.path = Path(config.data_path)
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.dataset = HDF5Dataset(self.path)

    def stream_batches(self) -> Iterator[TensorBatch]:
        """
        Yield batches of data wrapped in TensorBatch.

        Returns:
            Iterator yielding TensorBatch objects.
        """
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        for x_num, x_cat, y, sid in loader:
            yield TensorBatch(
                x_num=x_num.float(),
                x_cat=x_cat.long(),
                targets=y.long(),
                stay_ids=sid.tolist()
            )

    def __len__(self) -> int:
        """
        Return the total number of batches.
        
        Returns:
            Integer count of batches.
        """
        return len(self.dataset) // self.batch_size