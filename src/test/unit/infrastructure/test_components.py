import torch
from loguru import logger
from hint.infrastructure.components import FocalLoss, TemperatureScaler

def test_focal_loss_reduction() -> None:
    """
    [One-line Summary] Verify FocalLoss returns a positive scalar when applied to logits.

    [Description]
    Compute FocalLoss with configured gamma and class count on synthetic logits and targets,
    asserting the reduction yields a scalar tensor with a positive value.

    Test Case ID: INF-COMP-01
    Scenario: Compute focal loss on synthetic logits to confirm reduction behavior.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_focal_loss_reduction")
    
    loss_fn = FocalLoss(gamma=2.0, num_classes=3)
    
    logits = torch.randn(10, 3)
    targets = torch.randint(0, 3, (10,))
    
    logger.debug("Computing FocalLoss")
    loss = loss_fn(logits, targets)
    
    logger.debug(f"Loss value: {loss.item()}, Dimension: {loss.dim()}")
    
    assert loss.dim() == 0, f"Expected scalar loss (dim=0), got dim={loss.dim()}"
    assert loss.item() > 0, "Loss should be positive"
    
    logger.info("FocalLoss reduction verified successfully.")

def test_temperature_scaler_scaling() -> None:
    """
    [One-line Summary] Confirm TemperatureScaler scales logits using the configured temperature.

    [Description]
    Manually set the calibration temperature, apply the scaler to known logits, and verify the
    output equals logits divided by the configured value to ensure predictable calibration.

    Test Case ID: INF-COMP-02
    Scenario: Apply temperature scaling to a known tensor and compare with expected output.

    Args:
        None

    Returns:
        None
    """
    logger.info("Starting test: test_temperature_scaler_scaling")
    
    scaler = TemperatureScaler()
    target_temp = 2.0
    
    logger.debug(f"Setting temperature to {target_temp}")
    scaler.temperature.data.fill_(target_temp)
    
    logits = torch.tensor([[2.0, 4.0]])
    logger.debug(f"Input logits: {logits}")
    
    scaled = scaler(logits)
    logger.debug(f"Scaled logits: {scaled}")
    
    expected = torch.tensor([[1.0, 2.0]])
    assert torch.allclose(scaled, expected), f"Expected {expected}, but got {scaled}"
    
    logger.info("TemperatureScaler scaling logic verified successfully.")
