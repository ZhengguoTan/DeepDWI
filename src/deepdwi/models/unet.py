"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F


# %%
class Unet(nn.Module):
    """
    PyTorch implementation of a 2D U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_channels: Number of channels in the input to the U-Net model.
            out_channels: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_channels, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_channels, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_channels, H, W)`.

        Returns:
            Output tensor of shape `(N, out_channels, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, downsample_layer.shape[-1] - output.shape[-1],
                       0, downsample_layer.shape[-2] - output.shape[-2]]
            # if output.shape[-1] != downsample_layer.shape[-1]:
            #     padding[1] = 1  # padding right
            # if output.shape[-2] != downsample_layer.shape[-2]:
            #     padding[3] = 1  # padding bottom
            # if torch.sum(torch.tensor(padding)) != 0:
            output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, drop_prob: float):
        """
        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_channels, H, W)`.

        Returns:
            Output tensor of shape `(N, out_channels, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels in the output.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.Sequential(
            # 1. upsampling nearest neighbor (nn.Module)
            # 2. conv
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_channels, H, W)`.

        Returns:
            Output tensor of shape `(N, out_channels, H*2, W*2)`.
        """
        return self.layers(image)


# %%
class AttenUnet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 conv_channels: int = 128,
                 skip_channels: int = 4,
                 drop_prob: float = 0.02):
        super().__init__()

        self.Enc1a = DropoutConvBlock(in_channels, conv_channels,
                                      kernel_size=3, stride=2,
                                      drop_prob=drop_prob)
        self.Enc1b = DropoutConvBlock(conv_channels, conv_channels,
                                      kernel_size=3, stride=1,
                                      drop_prob=drop_prob)
        self.Enc2a = DropoutConvBlock(conv_channels, conv_channels,
                                      kernel_size=3, stride=2,
                                      drop_prob=drop_prob)
        self.Enc2b = DropoutConvBlock(conv_channels, conv_channels,
                                      kernel_size=3, stride=1,
                                      drop_prob=drop_prob)
        self.Enc3a = DropoutConvBlock(conv_channels, conv_channels,
                                      kernel_size=3, stride=2,
                                      drop_prob=drop_prob)
        self.Enc3b = DropoutConvBlock(conv_channels, conv_channels,
                                      kernel_size=3, stride=1,
                                      drop_prob=drop_prob)
        self.Enc4a = DropoutConvBlock(conv_channels, conv_channels,
                                      kernel_size=3, stride=2,
                                      drop_prob=drop_prob)
        self.Enc4b = DropoutConvBlock(conv_channels, conv_channels,
                                      kernel_size=3, stride=1,
                                      drop_prob=drop_prob)

        self.Dec1a = DropoutConvBlock(skip_channels+conv_channels, conv_channels,
                                      kernel_size=3, stride=1,
                                      drop_prob=drop_prob)
        self.Dec1b = DropoutConvBlock(conv_channels, conv_channels,
                                      kernel_size=1, stride=1,
                                      drop_prob=drop_prob)
        self.Dec2a = DropoutConvBlock(skip_channels+conv_channels, conv_channels,
                                      kernel_size=3, stride=1,
                                      drop_prob=drop_prob)
        self.Dec2b = DropoutConvBlock(conv_channels, conv_channels,
                                      kernel_size=1, stride=1,
                                      drop_prob=drop_prob)
        self.Dec3a = DropoutConvBlock(skip_channels+conv_channels, conv_channels,
                                      kernel_size=3, stride=1,
                                      drop_prob=drop_prob)
        self.Dec3b = DropoutConvBlock(conv_channels, conv_channels,
                                      kernel_size=1, stride=1,
                                      drop_prob=drop_prob)
        self.Dec4a = DropoutConvBlock(skip_channels+conv_channels, conv_channels,
                                      kernel_size=3, stride=1,
                                      drop_prob=drop_prob)
        self.Dec4b = DropoutConvBlock(conv_channels, conv_channels,
                                      kernel_size=1, stride=1,
                                      drop_prob=drop_prob)

        self.Skip1 = DropoutConvBlock(in_channels, skip_channels,
                                      kernel_size=1,
                                      drop_prob=drop_prob)
        self.Skip2 = DropoutConvBlock(conv_channels, skip_channels,
                                      kernel_size=1,
                                      drop_prob=drop_prob)
        self.Skip3 = DropoutConvBlock(conv_channels, skip_channels,
                                      kernel_size=1,
                                      drop_prob=drop_prob)
        self.Skip4 = DropoutConvBlock(conv_channels, skip_channels,
                                      kernel_size=1,
                                      drop_prob=drop_prob)

        self.Atten1 = AttenBlock(skip_channels+conv_channels,
                                 conv_channels,
                                 out_channels=conv_channels)
        self.Atten2 = AttenBlock(skip_channels+conv_channels,
                                 conv_channels,
                                 out_channels=conv_channels)
        self.Atten3 = AttenBlock(skip_channels+conv_channels,
                                 conv_channels,
                                 out_channels=conv_channels)
        self.Atten4 = AttenBlock(skip_channels+conv_channels,
                                 conv_channels,
                                 out_channels=conv_channels)

        self.Up = nn.Upsample(scale_factor=2, mode='nearest')

        self.last_layer = nn.Conv2d(conv_channels, in_channels, 1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip1 = self.Skip1(x)
        x = self.Enc1a(x)
        x = self.Enc1b(x)
        x_skip2 = self.Skip2(x)
        x = self.Enc2a(x)
        x = self.Enc2b(x)
        x_skip3 = self.Skip3(x)
        x = self.Enc3a(x)
        x = self.Enc3b(x)
        x_skip4 = self.Skip4(x)
        x = self.Enc4a(x)
        x = self.Enc4b(x)

        print('> x_skip4 ', x_skip4.shape)
        print('> x ', x.shape)
        print('> up x ', self.Up(x).shape)

        x = self.Atten4(torch.cat((x_skip4, self.Up(x)), dim=-3), x)
        x = self.Dec4a(x)
        x = self.Dec4b(x)

        x = self.Atten3(torch.cat((x_skip3, self.Up(x)), dim=-3), x)
        x = self.Dec3a(x)
        x = self.Dec3b(x)

        x = self.Atten2(torch.cat((x_skip2, self.Up(x)), dim=-3), x)
        x = self.Dec2a(x)
        x = self.Dec2b(x)

        x = self.Atten1(torch.cat((x_skip1, self.Up(x)), dim=-3), x)
        x = self.Dec1a(x)
        x = self.Dec1b(x)

        x = self.last_layer(x)

        return x


class DropoutConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 4,
                 stride: int = 1,
                 drop_prob: float = 0.1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Dropout2d(drop_prob),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=(kernel_size)//2),  # 'same'
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class AttenBlock(nn.Module):
    def __init__(self,
                 in_channels_x: int,
                 in_channels_g: int,
                 out_channels: int = 128,
                 kernel_size: int = 1):
        super().__init__()

        self.in_channels_x = in_channels_x
        self.in_channels_g = in_channels_g
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.phi_g = nn.Conv2d(in_channels_g, out_channels, kernel_size=kernel_size)
        self.theta_x = nn.Conv2d(in_channels_x, out_channels,kernel_size=kernel_size, stride=2)
        self.act = nn.ReLU()
        self.psi = nn.Conv2d(out_channels, 1, kernel_size=kernel_size)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        y = self.act(self.theta_x(x) + self.phi_g(g))
        y = self.psi(y)
        y = torch.sigmoid(y)
        y = self.up(y)
        y = x * y
        return y
