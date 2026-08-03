"""Small CPU-friendly classifiers and Mean Teacher utilities."""

from __future__ import annotations

import copy
import math
import random

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def make_teacher(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0, 1).")
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.mul_(decay).add_(s, alpha=1.0 - decay)


def mask_embedding(x: Tensor, probability: float, *, generator=None) -> Tensor:
    if not 0 <= probability < 1:
        raise ValueError("Mask probability must be in [0, 1).")
    mask = torch.rand(x.shape, device=x.device, generator=generator) >= probability
    return x * mask / max(1.0 - probability, 1e-6)


def consistency_loss(student_logits: Tensor, teacher_logits: Tensor,
                     confidence_threshold: float = 0.8) -> tuple[Tensor, Tensor]:
    teacher_p = torch.sigmoid(teacher_logits.detach())
    confidence = torch.maximum(teacher_p, 1 - teacher_p)
    keep = confidence >= confidence_threshold
    if keep.any():
        loss = F.binary_cross_entropy_with_logits(student_logits[keep], teacher_p[keep])
    else:
        loss = student_logits.sum() * 0.0
    return loss, keep.float().mean()


def sigmoid_rampup(step: int, rampup_steps: int) -> float:
    if rampup_steps <= 0:
        return 1.0
    phase = 1.0 - min(max(step, 0), rampup_steps) / rampup_steps
    return float(math.exp(-5.0 * phase * phase))

