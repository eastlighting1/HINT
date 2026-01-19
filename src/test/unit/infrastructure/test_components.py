import torch
from loguru import logger
from hint.infrastructure.components import FocalLoss

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
