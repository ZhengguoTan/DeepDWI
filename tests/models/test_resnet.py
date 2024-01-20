import unittest

import torch
import torch.nn as nn
import torch.testing as ptt

from deepdwi import fourier, util
from deepdwi.dims import *
from deepdwi.models import resnet


if __name__ == "__main__":
    unittest.main()


devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices.append(torch.device('cuda'))

class TestResnet(unittest.TestCase):

    def test_ResNet2D(self):
        for device in devices:
            img_shape = [1, 1, 1, 1, 3, 16, 24]
            img = torch.randn(img_shape, dtype=torch.cfloat, device=device)

            N_y, N_x = img_shape[-2:]

            img4 = img.view(-1, N_y, N_x)
            img4 = torch.view_as_real(img4)
            img4 = torch.transpose(img4, -1, -3)
            img4 = torch.transpose(img4, -2, -1)
            print('> x shape: ', img4.shape)

            RN2D = resnet.ResNet2D().to(device)

            y = RN2D(img4)
            print('> y shape: ', y.shape)

    def test_ResNet3D(self):
        for device in devices:
            img_shape = [1, 32, 1, 1, 2, 16, 24]
            img = torch.randn(img_shape, dtype=torch.cfloat, device=device)

            N_rep, N_shot, N_echo, N_coil, N_z, N_y, N_x = img_shape

            img5 = img.clone()

            img5 = img5.view(N_rep, N_shot * N_echo * N_coil, N_z, N_y, N_x)

            img5 = img5.permute((0, 2, 1, 3, 4))
            D, H, W = img5.shape[-3:]
            img5 = img5.view(N_rep * N_z, D, H, W)

            img5 = torch.view_as_real(img5)
            img5 = img5.permute((0, 4, 1, 2, 3))

            print('> x shape: ', img5.shape)

            RN3D = resnet.ResNet3D(2).to(device)

            y = RN3D(img5)
            print('> y shape: ', y.shape)
