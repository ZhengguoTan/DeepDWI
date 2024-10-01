"""
This module implements utility functions

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import numpy as np

import torch
import torch.nn as nn

import os

from datetime import datetime
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


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor):
        return input

    def adjoint(self, input: torch.Tensor):
        return input

    def normal(self, input: torch.Tensor):
        return input

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
        return input.permute(self.dims).contiguous()

    def adjoint(self, input: torch.Tensor):
        return input.permute(tuple(self.iaxes)).contiguous()

    def normal(self, input: torch.Tensor):
        return input


class Transpose(nn.Module):
    """swap two axes on the given input tensor

    Args:
        dims (None or tuple of ints): Axes to transpose input.

    """

    def __init__(self,
                 dim0: int = None,
                 dim1: int = None):
        self.dim0 = dim0
        self.dim1 = dim1

        super().__init__()

    def forward(self, input: torch.Tensor):
        return torch.transpose(input, self.dim0, self.dim1)

    def adjoint(self, input: torch.Tensor):
        return torch.transpose(input, self.dim1, self.dim0)

    def normal(self, input: torch.Tensor):
        return input


class C2R(nn.Module):
    """View as Real

    Args:

    """
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor):
        return torch.view_as_real(input)

    def adjoint(self, input: torch.Tensor):
        return torch.view_as_complex(input)

    def normal(self, input: torch.Tensor):
        return input


def set_output_dir(base_dir, config_dict):

    method_conf = config_dict['method']
    data_conf = config_dict.get('data', {})
    model_conf = config_dict.get('model', {})
    optim_conf = config_dict.get('optim', {})
    loss_conf = config_dict['loss']

    data_str = data_conf['kdat'].rsplit('/', 1)[1]
    data_str = data_str.rsplit('.h5', 1)[0]

    now = datetime.now()
    dir_name = now.strftime("%Y-%m-%d_")

    dir_name += method_conf
    dir_name += '_' + data_str
    if data_conf['N_shot_retro'] > 0:
        dir_name += '_shot-retro-' + "{:1d}".format(data_conf['N_shot_retro'])
    dir_name += '_norm-kdat-' + "{:3.1f}".format(data_conf['normalize_kdat'])
    if data_conf['navi'] is not None:
        dir_name += '_navi'
    else:
        dir_name += '_self'

    dir_name += '_' + model_conf['net']
    dir_name += '_ResBlock-' + "{:2d}".format(model_conf['N_residual_block'])
    if model_conf['batch_norm'] is True:
        dir_name += '_BatchNorm'
    dir_name += '_kernel-' + "{:1d}".format(model_conf['kernel_size'])
    dir_name += '_' + model_conf['unrolled_algorithm']
    dir_name += '_' + str(model_conf['N_unroll']).zfill(2)
    dir_name += '_lamda-' + "{:5.3f}".format(model_conf['lamda'])

    dir_name += '_' + optim_conf['method']
    dir_name += '_lr-' + "{:.6f}".format(optim_conf['lr'])
    dir_name += '_' + loss_conf

    return os.path.join(base_dir, dir_name)