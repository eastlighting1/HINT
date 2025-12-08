import torch
from loguru import logger
from hint.infrastructure.components import FocalLoss, TemperatureScaler

def test_focal_loss_reduction() -> None:
    """
    Validates that FocalLoss correctly reduces to a scalar value.
    
    Test Case ID: INF-COMP-01
    Description:
        Initializes FocalLoss with specific gamma and class count.
        Computes loss for random logits and targets.
        Verifies that the result is a 0-dimensional scalar and is greater than zero.
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
    Validates that TemperatureScaler correctly scales logits by the temperature parameter.
    
    Test Case ID: INF-COMP-02
    Description:
        Sets the temperature parameter manually.
        Passes a known logit tensor.
        Verifies that the output matches the expected scaled values.
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