"""
This module implements utility functions

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""


import torch

from torch import Tensor

from typing import Tuple

def rss(x: Tensor,
        dim: Tuple[int] = (0, ),
        keepdim: bool = False) -> Tensor:

    return torch.sqrt(torch.sum(abs(x)**2, dim=dim, keepdim=keepdim))