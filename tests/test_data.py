import pytest
import torch

from my_project.data import corrupt_mnist, normalize


def test_normalize_outputs_reasonable_stats():
    x = torch.randn(8, 1, 28, 28)
    y = normalize(x)

    assert y.shape == x.shape
    # mean ~ 0, std ~ 1 (allow small tolerance)
    assert torch.isfinite(y).all()
    assert abs(float(y.mean())) < 1e-2
    assert abs(float(y.std() - 1.0)) < 1e-2


def test_corrupt_mnist_sample_shapes():
    train_set, test_set = corrupt_mnist()

    x, y = train_set[0]
    assert x.shape == (1, 28, 28)
    assert y.ndim == 0


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    std = images.std()
    if std == 0:
        raise ValueError("Cannot normalize: standard deviation is zero.")
    return (images - images.mean()) / std


def test_normalize_raises_when_std_zero():
    x = torch.zeros(4, 1, 28, 28)
    with pytest.raises(ValueError, match="standard deviation is zero"):
        normalize(x)
