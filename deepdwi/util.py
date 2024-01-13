"""
This module implements utility functions

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import torch

from torch import Tensor

from typing import Tuple, Union


def rss(x: Tensor,
        dim: Tuple[int] = (0, ),
        keepdim: bool = False) -> Tensor:

    return torch.sqrt(torch.sum(abs(x)**2, dim=dim, keepdim=keepdim))


def axpy(y: Tensor, a, x: Tensor):
    """Compute y = a * x + y.

    Args:
        y (Tensor): Output array.
        a (scalar or Tensor): Input scalar.
        x (Tensor): Input array.

    """
    y += a * x


def xpay(y: Tensor, a, x: Tensor):
    """Compute y = x + a * y.

    Args:
        y (Tensor): Output array.
        a (scalar or Tensor): Input scalar.
        x (Tensor): Input array.
    """
    y *= a
    y += x


def estimate_weights(y, coil_dim: int = 0):
    """Compute a binary mask from zero-filled k-space.

    Args:
        y (Tensor): zero-filled k-space.
        coil_dim (int): The coils dimension index. Default is 0.
    """

    weights = (rss(y, dim=(coil_dim, ), keepdims=True) > 0).astype(y.dtype)
    return weights