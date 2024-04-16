"""
This module implements the proximal operator.
"""
import torch
import torch.nn as nn

from typing import Tuple

from deepdwi import util
from deepdwi.dims import *


class VAE(nn.Module):
    def __init__(self,
                 model: nn.Module):
        super(VAE, self).__init__()

        self.model = model

    def forward(self, input: torch.Tensor,
                alpha: float = 1.,
                contrast_dim: int = DIM_TIME):
        baseline = input[0]

        output1 = torch.where(baseline!=0, torch.true_divide(input, baseline), torch.zeros_like(input))
        output1_shape = output1.shape

        output2 = torch.transpose(output1, 0, contrast_dim)
        output2_shape = output2.shape

        output3 = torch.reshape(output2, [output2.shape[0], -1])
        output3_shape = output3.shape

        output4 = torch.transpose(output3, 0, 1)
        output4_shape = output4.shape

        # magnitude and phase
        output4_mag = abs(output4).float()
        output4_phs = torch.angle(output4)

        with torch.no_grad():
            output5, _, _ = self.model(output4_mag)

        output5 = alpha * output5 * output4_phs

        output4b = torch.transpose(output5, 1, 0)

        output3b = torch.reshape(output4b, output2_shape)

        output2b = torch.transpose(output3b, contrast_dim, 0)

        return output2b * baseline
