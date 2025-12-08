import torch
from loguru import logger
from hint.infrastructure.networks import GFINet_CNN

def test_gfinet_cnn_forward_shape() -> None:
    """
    Validates the output shape of the GFINet_CNN model during a forward pass.
    
    Test Case ID: INF-NET-01
    Description:
        Constructs a GFINet_CNN model with specific channel configurations and mock vocabulary sizes.
        Passes random tensors representing numeric and categorical features.
        Verifies that the output tensor has the expected shape (Batch Size, Num Classes).
    """
    logger.info("Starting test: test_gfinet_cnn_forward_shape")

    batch_size = 4
    seq_len = 20
    embed_dim = 16
    n_num_feats = 5
    
    cat_vocab = {"cat1": 10, "cat2": 5}
    
    logger.debug("Initializing GFINet_CNN model")
    model = GFINet_CNN(
        in_chs=[n_num_feats, 0, 0],
        n_cls=4,
        g1=list(range(n_num_feats)),
        g2=[], 
        rest=[],
        cat_vocab_sizes=cat_vocab,
        embed_dim=embed_dim,
        layers=2
    )
    
    logger.debug(f"Creating input tensors with Batch={batch_size}, SeqLen={seq_len}")
    x_full = torch.randn(batch_size, n_num_feats, seq_len)
    x_cat = torch.randint(0, 5, (batch_size, 2, seq_len))
    
    logger.debug("Executing forward pass")
    out = model(x_full, x_cat)
    
    expected_shape = (batch_size, 4)
    logger.debug(f"Verifying output shape. Expected: {expected_shape}, Got: {out.shape}")
    
    assert out.shape == expected_shape, f"Expected shape {expected_shape}, but got {out.shape}"
    logger.info("GFINet_CNN forward shape verified successfully.")