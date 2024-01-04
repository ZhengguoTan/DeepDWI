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
