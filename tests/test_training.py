"""Tests for the training script."""

import torch

from my_project.model import MyAwesomeModel


def test_model_output_shape():
    """Test that model produces correct output shape for MNIST (10 classes)."""
    model = MyAwesomeModel()
    dummy_input = torch.randn(32, 1, 28, 28)
    output = model(dummy_input)

    assert output.shape == (32, 10), f"Expected shape (32, 10), got {output.shape}"


def test_model_has_trainable_parameters():
    """Test that the model has trainable parameters."""
    model = MyAwesomeModel()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert trainable_params > 0, "Model has no trainable parameters"


def test_model_forward_pass():
    """Test that model forward pass produces valid outputs."""
    model = MyAwesomeModel()
    model.eval()

    with torch.no_grad():
        dummy_input = torch.randn(16, 1, 28, 28)
        output = model(dummy_input)

        assert not torch.isnan(output).any(), "Model output contains NaN"
        assert not torch.isinf(output).any(), "Model output contains Inf"
