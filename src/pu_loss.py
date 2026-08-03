"""Non-negative positive-unlabeled (nnPU) risk estimators."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class NNPUDiagnostics:
    positive_risk: Tensor
    negative_risk: Tensor
    corrected_negative_risk: Tensor
    objective: Tensor
    correction_active: Tensor

    def detached(self) -> dict[str, float | bool]:
        return {
            "positive_risk": float(self.positive_risk.detach()),
            "negative_risk": float(self.negative_risk.detach()),
            "corrected_negative_risk": float(self.corrected_negative_risk.detach()),
            "objective": float(self.objective.detach()),
            "correction_active": bool(self.correction_active.detach()),
        }


def _validate(logits_p: Tensor, logits_u: Tensor, prior: float) -> None:
    if logits_p.ndim not in (1, 2) or logits_u.ndim not in (1, 2):
        raise ValueError("Positive and unlabeled logits must be rank 1 or (n, 1).")
    if logits_p.numel() == 0 or logits_u.numel() == 0:
        raise ValueError("Each nnPU minibatch needs positive and unlabeled samples.")
    if logits_p.shape[-1] != 1 and logits_p.ndim == 2:
        raise ValueError("Binary nnPU expects one logit per example, not class logits.")
    if logits_u.shape[-1] != 1 and logits_u.ndim == 2:
        raise ValueError("Binary nnPU expects one logit per example, not class logits.")
    if not 0.0 < float(prior) < 1.0:
        raise ValueError("The positive class prior must be strictly between 0 and 1.")
    if not torch.isfinite(logits_p).all() or not torch.isfinite(logits_u).all():
        raise ValueError("nnPU logits must be finite.")


def nnpu_loss(
    logits_p: Tensor,
    logits_u: Tensor,
    prior: float,
    *,
    beta: float = 0.0,
    gamma: float = 1.0,
) -> tuple[Tensor, NNPUDiagnostics]:
    """Compute Kiryo et al.'s mini-batch nnPU logistic risk.

    The unbiased risk is pi E_p[l(+f)] + E_u[l(-f)] -
    pi E_p[l(-f)]. When its negative component is below ``-beta``, the
    gradient-reversal correction ``-gamma * negative_risk`` is used.
    """
    _validate(logits_p, logits_u, prior)
    if beta < 0 or gamma <= 0:
        raise ValueError("beta must be non-negative and gamma strictly positive.")
    p = logits_p.reshape(-1)
    u = logits_u.reshape(-1)
    pi = torch.as_tensor(prior, dtype=p.dtype, device=p.device)
    positive_risk = pi * F.softplus(-p).mean()
    negative_risk = F.softplus(u).mean() - pi * F.softplus(p).mean()
    correction_active = negative_risk < -float(beta)
    corrected = torch.where(correction_active, -float(gamma) * negative_risk, negative_risk)
    objective = torch.where(correction_active, corrected, positive_risk + negative_risk)
    if not torch.isfinite(objective):
        raise FloatingPointError("nnPU risk became non-finite; check logits and prior.")
    diagnostics = NNPUDiagnostics(
        positive_risk, negative_risk, corrected, objective, correction_active
    )
    return objective, diagnostics

