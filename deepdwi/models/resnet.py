"""
This module implements a Residual Network

Author:
    Burhaneddin Yaman
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import torch
import torch.nn as nn

from typing import Tuple, Union

# %%
def activation_func(activation,is_inplace=False):
    return nn.ModuleDict([['ReLU',  nn.ReLU(inplace=is_inplace)],
                          ['None',  nn.Identity()]])[activation]

def batch_norm(use_batch_norm, features):
    if use_batch_norm:
        return nn.BatchNorm2d(features)
    else:
        return nn.Identity()

def conv_layer(filter_size, padding='same', use_batch_norm=False, activation_type='ReLU'):
    kernel_size, in_c, out_c = filter_size
    return nn.Sequential(nn.Conv2d(in_channels=in_c,
                                   out_channels=out_c,
                                   kernel_size=kernel_size,
                                   padding=padding, bias=True),
                         batch_norm(use_batch_norm, out_c),
                         activation_func(activation_type))

def residual_block(filter_size):
    return nn.Sequential(conv_layer(filter_size, activation_type='ReLU'),
                         conv_layer(filter_size, activation_type='None'))


class ResidualBlockModule(nn.Module):
    def __init__(self, filter_size, num_blocks):
        super().__init__()
        self.layers = nn.ModuleList([ residual_block(filter_size=filter_size) for _ in range(num_blocks)])

    def forward(self, x):
        scale_factor = torch.tensor([0.1], dtype=torch.float32).to(x.device)

        for layer in self.layers:
            x = x + layer(x)*scale_factor

        return x

# %%
class ResNet2D(nn.Module):
    def __init__(self, in_channels: int = 2,
                 N_residual_block: int = 5,
                 kernel_size: int = 3,
                 features: int = 64,
                 use_batch_norm: bool = False):

        super().__init__()
        self.in_channels = in_channels
        self.N_residual_block = N_residual_block
        filter1 = [kernel_size, in_channels, features] #map input to size of feature maps
        filter2 = [kernel_size, features, features] #ResNet Blocks
        filter3 = [kernel_size, features, in_channels] #map output channels to input channels
        self.layer1 = conv_layer(filter_size=filter1,
                                 activation_type='None',
                                 use_batch_norm=use_batch_norm)

        self.layer2 = ResidualBlockModule(filter_size=filter2, num_blocks=N_residual_block)

        self.layer3 =  conv_layer(filter_size=filter2,
                                  activation_type='None',
                                  use_batch_norm=use_batch_norm)

        self.layer4 = conv_layer(filter_size=filter3,
                                 activation_type='None',
                                 use_batch_norm=use_batch_norm)

    def forward(self,input_x):
        l1_out = self.layer1(input_x)
        l2_out = self.layer2(l1_out)
        l3_out = self.layer3(l2_out)
        temp = l3_out + l1_out
        nw_out = self.layer4(temp)

        return nw_out


# %% ResNet 3D
def _conv_layer(in_channels: int,
                out_channels: int,
                kernel_size: Union[int, Tuple[int, int, int]] = 3,
                stride: Union[int, Tuple[int, int, int]] = 1,
                padding: Union[int, Tuple[int, int, int]] = 1,
                activation_type: str = 'ReLU'):

        return nn.Sequential(nn.Conv3d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,bias=False),
                             activation_func(activation_type))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]] = 3,
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 1,
                 scaling: float = 0.1):
        super().__init__()
        self.sequence = nn.Sequential(_conv_layer(in_channels, out_channels, kernel_size,
                                                  stride, padding, activation_type='ReLU'),
                                      _conv_layer(in_channels, out_channels, kernel_size,
                                                  stride, padding, activation_type='None'))

        self.scaling = scaling

    def forward(self, x: torch.Tensor):
        scale = torch.tensor([self.scaling], dtype=torch.float32).to(x.device)

        x = x + self.sequence(x) * scale
        return x


class ResNet3D(nn.Module):
    """
    3D ResNet
    """
    def __init__(self, in_channels: int,
                 features: int = 64,
                 N_residual_block: int = 5,
                 kernel_size: Union[int, Tuple[int, int, int]] = 3,
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 1,
                 scaling: float = 0.1):
        super().__init__()

        # 1st convolutional layer
        self.conv_layer1 = _conv_layer(in_channels, features, kernel_size,
                                       stride, padding,
                                       activation_type='None')

        # 2nd a series of residual blocks
        modules = []
        for _ in range(N_residual_block):
            modules.append(ResidualBlock(features, features, kernel_size,
                                         stride, padding, scaling))

        self.resi_blocks = nn.Sequential(*modules)

        # 3rd convolutional layer
        self.conv_layer3 = _conv_layer(features, features, kernel_size,
                                       stride, padding,
                                       activation_type='None')

        # 4th convolutional layer
        self.conv_layer4 = _conv_layer(features, in_channels, kernel_size,
                                       stride, padding,
                                       activation_type='None')

    def forward(self, x: torch.Tensor):

        x1 = self.conv_layer1(x)
        x2 = self.resi_blocks(x1)
        x3 = self.conv_layer3(x2)
        x4 = x1 + x3
        x5 = self.conv_layer4(x4)

        return x5
