"""
This module implements fast fourier transform

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import torch

from torch import Tensor

from typing import Tuple

__all__ = [
    "fft",
    "ifft",
]


def _normalize_dims(dims, ndim):
    if dims is None:
        return tuple(range(ndim))
    else:
        return tuple(a % ndim for a in sorted(dims))

def _fftc(x: Tensor,
          dim: Tuple[int] = None,
          norm: str = 'ortho') -> Tensor:

    ndim = x.ndim
    dim = _normalize_dims(dim, ndim)

    tmp = torch.fft.ifftshift(x, dim=dim)
    tmp = torch.fft.fftn(tmp, dim=dim, norm=norm)
    y = torch.fft.fftshift(tmp, dim=dim)

    return y

def _ifftc(x: Tensor,
          dim: Tuple[int] = None,
          norm: str = 'ortho') -> Tensor:

    ndim = x.ndim
    dim = _normalize_dims(dim, ndim)

    tmp = torch.fft.ifftshift(x, dim=dim)
    tmp = torch.fft.ifftn(tmp, dim=dim, norm=norm)
    y = torch.fft.fftshift(tmp, dim=dim)

    return y


def fft(x: Tensor,
        dim: Tuple[int] = None,
        center: bool = True,
        norm: str = 'ortho') -> Tensor:

    if center:
        y = _fftc(x, dim=dim, norm=norm)
    else:
        y = torch.fft.fftn(x, dim=dim, norm=norm)

    return y


def ifft(x: Tensor,
         dim: Tuple[int] = None,
         center: bool = True,
         norm: str = 'ortho') -> Tensor:

    if center:
        y = _ifftc(x, dim=dim, norm=norm)
    else:
        y = torch.fft.ifftn(x, dim=dim, norm=norm)

    return y