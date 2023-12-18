import unittest

import torch
import torch.testing as ptt

from deepdwi import fourier
from deepdwi.models import mri
from deepdwi.models.dims import *

if __name__ == "__main__":
    unittest.main()


class TestMri(unittest.TestCase):
    def test_sense_model(self):
        img_shape = [1, 1, 1, 1, 16, 16]
        mps_shape = [1, 1, 8, 1, 16, 16]

        img = torch.randn(img_shape, dtype=torch.cfloat)
        mps = torch.randn(mps_shape, dtype=torch.cfloat)

        y1 = fourier.fft(mps * img, dim=(-2, -1))

        S = mri.Sense(mps, y1)
        y2 = S(img)

        ptt.assert_close(y1, y2)

    def test_sense_model_basis(self):
        subim_shape = [5, 1, 1, 1, 16, 16]
        basis_shape = [35, 5]
        mps_shape = [1, 1, 8, 1, 16, 16]

        subim = torch.randn(subim_shape, dtype=torch.cfloat)
        basis = torch.randn(basis_shape, dtype=torch.cfloat)
        mps = torch.randn(mps_shape, dtype=torch.cfloat)

        totim = basis @ subim.view(subim.shape[0], -1)
        totim = totim.view([basis_shape[0]] + subim_shape[1:])
        y = fourier.fft(mps * totim, dim=(-2, -1))

        S = mri.Sense(mps, y, basis=basis)
        y_fwd = S(subim)

        ptt.assert_close(y, y_fwd)

    def test_sense_model_phase_echo(self):
        img_shape = [1, 7, 1, 1, 16, 16]
        mps_shape = [1, 1, 8, 1, 16, 16]

        img = torch.randn(img_shape, dtype=torch.cfloat)
        phs = torch.randn(img_shape, dtype=torch.cfloat)
        mps = torch.randn(mps_shape, dtype=torch.cfloat)

        y = fourier.fft(mps * phs * img, dim=(-2, -1))

        S = mri.Sense(mps, y, phase_echo=phs)
        y_fwd = S(img)

        ptt.assert_close(y, y_fwd)

    def test_sense_model_phase_slice(self):
        img_shape = [1, 1, 1, 3, 16, 16]
        mps_shape = [1, 1, 8, 3, 16, 16]

        img = torch.randn(img_shape, dtype=torch.cfloat)
        phs = torch.randn(img_shape, dtype=torch.cfloat)
        mps = torch.randn(mps_shape, dtype=torch.cfloat)

        y = torch.sum(phs * fourier.fft(mps * img, dim=(-2, -1)), dim=DIM_Z, keepdim=True)

        S = mri.Sense(mps, y, phase_slice=phs)
        y_fwd = S(img)

        ptt.assert_close(y, y_fwd)
