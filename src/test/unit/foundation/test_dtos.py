import torch
from loguru import logger
from hint.foundation.dtos import TensorBatch
from ...utils.custom_assertions import assert_raises

def test_tensor_batch_device_transfer() -> None:
    """
    [One-line Summary] Verify TensorBatch moves all tensor fields to the target device.

    [Description]
    Transfer a populated TensorBatch to a target device and assert every tensor attribute
    reports the new device to guarantee downstream components see consistent placement.

    Test Case ID: DTO-01
    Scenario: Transfer a populated TensorBatch to a target device.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_tensor_batch_device_transfer")
    
    batch = TensorBatch(
        x_num=torch.randn(2, 5),
        x_cat=torch.randint(0, 3, (2, 2)),
        y=torch.tensor([0, 1]),
        ids=torch.tensor([100, 101]),
        mask=torch.ones(2, 5)
    )
    
    target_device = "cpu"
    logger.debug(f"Transferring TensorBatch to {target_device}")
    new_batch = batch.to(target_device)
    
    logger.debug("Verifying device of tensor components")
    assert str(new_batch.x_num.device).startswith(target_device)
    assert str(new_batch.x_cat.device).startswith(target_device)
    assert str(new_batch.y.device).startswith(target_device)
    assert str(new_batch.ids.device).startswith(target_device)
    assert str(new_batch.mask.device).startswith(target_device)
    
    logger.info("TensorBatch device transfer verified.")

def test_tensor_batch_optional_fields() -> None:
    """
    [One-line Summary] Confirm TensorBatch handles optional fields during device transfer.

    [Description]
    Move a TensorBatch that includes optional fields set to None and verify the transfer keeps
    absent values untouched while relocating populated tensors to the target device.

    Test Case ID: DTO-02
    Scenario: Transfer a TensorBatch containing optional fields set to None.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_tensor_batch_optional_fields")
    
    batch = TensorBatch(
        x_num=torch.randn(2, 5),
        x_cat=None,
        y=torch.tensor([0, 1]),
        ids=None,
        mask=None
    )
    
    logger.debug("Transferring TensorBatch with None fields")
    new_batch = batch.to("cpu")
    
    assert new_batch.x_cat is None
    assert new_batch.ids is None
    assert new_batch.mask is None
    assert str(new_batch.x_num.device) == "cpu"
    
    logger.info("TensorBatch optional fields handling verified.")
