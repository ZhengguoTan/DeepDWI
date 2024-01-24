"""
This module implements utility functions

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import numpy as np

import torch
import torch.nn as nn

from torch import Tensor

from typing import Tuple


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

    Note:
        These are inplace operations!!!
    """
    y += a * x


def xpay(y: Tensor, a, x: Tensor):
    """Compute y = x + a * y.

    Args:
        y (Tensor): Output array.
        a (scalar or Tensor): Input scalar.
        x (Tensor): Input array.

    Note:
        These are inplace operations!!!
    """
    y *= a
    y += x


def estimate_weights(y, coil_dim: int = 0):
    """Compute a binary mask from zero-filled k-space.

    Args:
        y (Tensor): zero-filled k-space.
        coil_dim (int): The coils dimension index. Default is 0.
    """

    weights = (rss(y, dim=(coil_dim, ), keepdim=True) > 0).type(y.dtype)
    return weights


class Reshape(nn.Module):
    """Reshape input to given output shape.

    Args:
        oshape (tuple of ints): Output shape.
        ishape (tuple of ints): Input shape.

    Inspired by Linop @ SigPy
    """

    def __init__(self,
                 oshape: Tuple[int, ...],
                 ishape: Tuple[int, ...]):

        self.oshape = oshape
        self.ishape = ishape

        super().__init__()

    def forward(self, input: torch.Tensor):
        return torch.reshape(input, self.oshape)

    def adjoint(self, input: torch.Tensor):
        return torch.reshape(input, self.ishape)

    def normal(self, input: torch.Tensor):
        return input


class Permute(nn.Module):
    """Tranpose input with the given axes.

    Args:
        ishape (tuple of ints): Input shape.
        dims (None or tuple of ints): Axes to transpose input.

    """

    def __init__(self,
                 ishape: Tuple[int, ...],
                 dims: Tuple[int, ...] = None):
        self.dims = dims
        if dims is None:
            self.iaxes = None
            oshape = ishape[::-1]
        else:
            self.iaxes = np.argsort(dims)
            oshape = [ishape[a] for a in dims]

        self.oshape = oshape
        self.ishape = ishape

        super().__init__()

    def forward(self, input: torch.Tensor):
        return torch.permute(input, self.dims)

    def adjoint(self, input: torch.Tensor):

        if self.dims is None:
            iaxes = None
            oshape = self.ishape[::-1]
        else:
            iaxes = np.argsort(self.dims)
            oshape = [self.ishape[a] for a in self.dims]

        return torch.permute(input, tuple(iaxes))

    def normal(self, input: torch.Tensor):
        return input