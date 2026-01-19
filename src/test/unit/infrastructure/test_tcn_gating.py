import torch
from hint.infrastructure.networks import TCNClassifier


def test_tcn_icd_gating_enabled() -> None:
    vocab_sizes = [10]
    embed_dim = 16
    cat_embed_dim = 4
    icd_dim = 8
    model = TCNClassifier(
        in_chs=3,
        n_cls=4,
        vocab_sizes=vocab_sizes,
        icd_dim=icd_dim,
        embed_dim=embed_dim,
        cat_embed_dim=cat_embed_dim,
        layers=2,
        use_icd_gating=True,
    )

    assert model.icd_gate is not None

    p_cat_dim = (len(vocab_sizes) * cat_embed_dim) + embed_dim
    p_cat = torch.randn(2, p_cat_dim)
    gate = model.icd_gate(p_cat)
    assert gate.shape == (2, embed_dim)
    assert torch.all((gate >= 0.0) & (gate <= 1.0))

    x_num = torch.randn(2, 3, 10)
    x_cat = torch.randint(0, 10, (2, 1, 10))
    x_icd = torch.randn(2, icd_dim)
    out = model(x_num, x_cat, x_icd)
    assert out.shape == (2, 4)


def test_tcn_icd_gating_disabled_without_icd() -> None:
    model = TCNClassifier(
        in_chs=3,
        n_cls=4,
        vocab_sizes=[10],
        icd_dim=0,
        embed_dim=16,
        cat_embed_dim=4,
        layers=2,
        use_icd_gating=True,
    )

    assert model.icd_gate is None
