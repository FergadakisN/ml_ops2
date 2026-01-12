import pytest
import torch

from my_project.model import MyAwesomeModel


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model_output_shape(batch_size: int) -> None:
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)


def test_model():
    """Test the MyAwesomeModel class."""
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
