import pytest
import torch

from src.pu_loss import nnpu_loss


def test_loss_is_finite_scalar_and_gradients_flow():
    p = torch.tensor([1.2, -0.2, 0.7], requires_grad=True)
    u = torch.tensor([-1.0, 0.3, 0.1, 2.0], requires_grad=True)
    loss, parts = nnpu_loss(p, u, 0.25)
    assert loss.ndim == 0 and torch.isfinite(loss)
    for value in (parts.positive_risk, parts.negative_risk, parts.objective):
        assert value.ndim == 0
    loss.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert u.grad is not None and torch.isfinite(u.grad).all()


@pytest.mark.parametrize("prior", [0.0, 1.0, -0.1, 1.1])
def test_invalid_prior_raises(prior):
    with pytest.raises(ValueError, match="prior"):
        nnpu_loss(torch.ones(2), torch.zeros(3), prior)


def test_invalid_binary_encoding_shape_raises():
    with pytest.raises(ValueError, match="one logit"):
        nnpu_loss(torch.ones(2, 2), torch.zeros(3, 2), 0.3)


def test_empty_and_nonfinite_inputs_raise_useful_errors():
    with pytest.raises(ValueError, match="minibatch"):
        nnpu_loss(torch.tensor([]), torch.ones(2), 0.3)
    with pytest.raises(ValueError, match="finite"):
        nnpu_loss(torch.tensor([float("nan")]), torch.ones(2), 0.3)


def test_nonnegative_correction_branch_is_finite():
    p = torch.full((8,), 20.0, requires_grad=True)
    u = torch.full((8,), -20.0, requires_grad=True)
    loss, parts = nnpu_loss(p, u, 0.9, beta=0.0, gamma=1.0)
    assert parts.correction_active.item()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(u.grad).all()
