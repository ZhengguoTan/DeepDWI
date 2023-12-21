import unittest

import numpy as np
import sigpy as sp

import torch
import torch.testing as ptt

from deepdwi import fourier
from deepdwi.models import mri
from deepdwi.dims import *

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

    def test_sense_model_sigpy(self):

        subim_shape = [5, 1, 1, 3, 16, 16]
        basis_shape = [35, 5]

        mps_shape = [1, 1, 8, 1, 16, 16]

        subim = torch.randn(subim_shape, dtype=torch.complex128)
        basis = torch.randn(basis_shape, dtype=torch.complex128)
        mps = torch.randn(mps_shape, dtype=torch.complex128)

        totim = basis @ subim.view(subim.shape[0], -1)
        totim = totim.view([basis_shape[0]] + subim_shape[1:])

        y = fourier.fft(mps * totim, dim=(-2, -1))

        phs = torch.randn(totim.shape, dtype=torch.cfloat)
        y = torch.sum(phs * y, dim=DIM_Z, keepdim=True)

        weights = torch.ones_like(y)

        # deepdwi sense forward model
        SENSE = mri.Sense(mps, y, basis=basis, phase_slice=phs, weights=weights)
        y_fwd = SENSE(subim)

        ptt.assert_close(y, y_fwd)

        # sigpy sense forward model
        B1 = sp.linop.Reshape([subim_shape[0], np.prod(subim_shape[1:])], subim_shape)
        B2 = sp.linop.MatMul(B1.oshape, basis.detach().cpu().numpy())
        B3 = sp.linop.Reshape(totim.shape, B2.oshape)

        B = B3 * B2 * B1
        S = sp.linop.Multiply(B.oshape, mps.detach().cpu().numpy())
        F = sp.linop.FFT(S.oshape, axes=range(-2, 0))

        PHI = sp.linop.Multiply(F.oshape, phs.detach().cpu().numpy())
        SUM = sp.linop.Sum(PHI.oshape, axes=(DIM_Z, ), keepdims=True)
        M = SUM * PHI
        A = M * F * S * B

        y_sp = A * subim.detach().cpu().numpy()
        y_sp_ten = torch.from_numpy(y_sp)

        ptt.assert_close(y, y_sp_ten)

        # test adjoint
        x_sp = A.H * y_sp
        x_sp_ten = torch.from_numpy(x_sp)

        x_adj = SENSE.adjoint(y_fwd)

        ptt.assert_close(x_adj, x_sp_ten)
