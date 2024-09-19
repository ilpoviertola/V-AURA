"""
This code is adapted from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/optim/inverse_sqrt_lr_scheduler.py

Original code is under MIT license.
"""

import math
import typing as tp

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class InverseSquareRootLRScheduler(_LRScheduler):
    """Inverse square root LR scheduler.

    Args:
        optimizer (Optimizer): Torch optimizer.
        warmup_steps (int): Number of warmup steps.
        warmup_init_lr (tp.Optional[float]): Initial learning rate
            during warmup phase. When not set, use the provided learning rate.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        warmup_init_lr: tp.Optional[float] = 0,
    ):
        self.warmup_steps = warmup_steps
        self.warmup_init_lr = warmup_init_lr
        super().__init__(optimizer)

    def _get_sched_lr(self, lr: float, step: int):
        if step < self.warmup_steps:
            warmup_init_lr = self.warmup_init_lr or 0
            lr_step = (lr - warmup_init_lr) / self.warmup_steps
            lr = warmup_init_lr + step * lr_step
        else:
            decay_factor = lr * self.warmup_steps**0.5
            lr = decay_factor * step**-0.5
        return lr

    def get_lr(self):
        return [
            self._get_sched_lr(base_lr, self._step_count) for base_lr in self.base_lrs
        ]


class WarmUpToStaticLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        warmup_init_lr: tp.Optional[float] = 0,
    ):
        self.warmup_steps = warmup_steps
        self.warmup_init_lr = warmup_init_lr
        super().__init__(optimizer)

    def _get_sched_lr(self, lr: float, step: int):
        if step < self.warmup_steps:
            warmup_init_lr = self.warmup_init_lr or 0
            lr_step = (lr - warmup_init_lr) / self.warmup_steps
            lr = warmup_init_lr + step * lr_step
        return lr

    def get_lr(self) -> float:
        return [
            self._get_sched_lr(base_lr, self._step_count) for base_lr in self.base_lrs
        ]


class CosineLRScheduler(_LRScheduler):
    """Cosine LR scheduler.

    Args:
        optimizer (Optimizer): Torch optimizer.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of steps.
        lr_min_ratio (float): Minimum learning rate.
        cycle_length (float): Cycle length.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        lr_min_ratio: float = 0.0,
        cycle_length: float = 1.0,
        **kwargs,
    ):
        self.warmup_steps = warmup_steps
        assert self.warmup_steps >= 0
        self.total_steps = total_steps
        assert self.total_steps >= 0
        self.lr_min_ratio = lr_min_ratio
        self.cycle_length = cycle_length
        super().__init__(optimizer)

    def _get_sched_lr(self, lr: float, step: int):
        if step < self.warmup_steps:
            lr_ratio = step / self.warmup_steps
            lr = lr_ratio * lr
        elif step <= self.total_steps:
            s = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_ratio = self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                1.0 + math.cos(math.pi * s / self.cycle_length)
            )
            lr = lr_ratio * lr
        else:
            lr_ratio = self.lr_min_ratio
            lr = lr_ratio * lr
        return lr

    def get_lr(self):
        return [self._get_sched_lr(lr, self.last_epoch) for lr in self.base_lrs]
