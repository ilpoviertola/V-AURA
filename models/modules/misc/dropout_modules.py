"""This code is adapted from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py"""

from copy import deepcopy
from typing import Tuple

import torch
from torch import nn


def nullify_condition(cond: torch.Tensor, dim: int = 1):
    """Transform an input condition to a null condition.
    The way it is done by converting it to a single zero vector similarly
    to how it is done inside WhiteSpaceTokenizer and NoopTokenizer.

    Args:
        condition (ConditionType): A tuple of condition and mask (tuple[torch.Tensor, torch.Tensor])
        dim (int): The dimension that will be truncated (should be the time dimension)
        WARNING!: dim should not be the batch dimension!
    Returns:
        ConditionType: A tuple of null condition and mask
    """
    assert dim != 0, "dim cannot be the batch dimension!"
    assert isinstance(
        cond, torch.Tensor
    ), "'nullify_condition' got an unexpected input type!"
    last_dim = cond.dim() - 1
    out = cond.transpose(dim, last_dim)
    out = 0.0 * out[..., :1]
    out = out.transpose(dim, last_dim)
    assert cond.dim() == out.dim()
    return out


class DropoutModule(nn.Module):
    """Base module for all dropout modules."""

    def __init__(self, seed: int = 666):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)


class ClassifierFreeGuidanceDropout(DropoutModule):
    """Classifier Free Guidance dropout.
    All attributes are dropped with the same probability.

    Args:
        p (float): Probability to apply condition dropout during training.
        seed (int): Random seed.
    """

    def __init__(self, p: float, seed: int = 666):
        super().__init__(seed=seed)
        self.p = p

    def forward(self, samples: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            tuple[torch.Tensor, bool]: Tuple of conditioning tensor and a boolean flag indicating whether it was dropped.
        """
        if not self.training:
            return samples, False

        # decide on which attributes to drop in a batched fashion
        drop = torch.rand(1, generator=self.rng).item() < self.p
        if not drop:
            return samples, False

        # nullify conditions of all attributes
        samples = deepcopy(samples)
        return (
            nullify_condition(torch.zeros_like(samples), dim=(samples.dim() - 3)),
            True,
        )

    def __repr__(self):
        return f"ClassifierFreeGuidanceDropout(p={self.p})"
