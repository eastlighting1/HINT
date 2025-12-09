import torch
from loguru import logger
from src.hint.infrastructure.networks import GFINet_CNN

def test_gfinet_cnn_forward_shape() -> None:
    """
    Validates output shape.
    Test Case ID: INF-NET-01
    """
    logger.info("Starting test: test_gfinet_cnn_forward_shape")

    model = GFINet_CNN(
        in_chs=[5, 0, 0], n_cls=4,
        g1=torch.tensor([0, 1, 2, 3, 4]).long(), # [Fix] Explicit LongTensor
        g2=torch.tensor([]).long(), 
        rest=torch.tensor([]).long(),
        cat_vocab_sizes={"cat1": 10, "cat2": 5},
        embed_dim=16, layers=2
    )
    
    x_full = torch.randn(4, 5, 20)
    x_cat = torch.randint(0, 5, (4, 2, 20))
    
    out = model(x_full, x_cat)
    assert out.shape == (4, 4)