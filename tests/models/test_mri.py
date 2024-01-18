import unittest

import numpy as np
import sigpy as sp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.testing as ptt

from deepdwi import fourier, lsqr
from deepdwi.models import mri
from deepdwi.dims import *

if __name__ == "__main__":
    unittest.main()


devices = [torch.device('cpu'), torch.device('cuda')]

class TestMri(unittest.TestCase):
    def solve_mri_lsqr(self, Model: nn.Module,
                       y: torch.Tensor,
                       lamda: float = 1E-5,
                       max_iter: int = 100):

        AHA = lambda x : Model.adjoint(Model.forward(x)) + lamda * x
        AHy = Model.adjoint(y)
        x_init = torch.zeros_like(AHy)
        CG = lsqr.ConjugateGradient(AHA, AHy, x_init, max_iter=max_iter)
        x = CG()
        return x

    # TODO: use this function in test
    def solve_mri_torch(self, Model: nn.Module,
                   y: torch.Tensor,
                   x_gt: torch.Tensor,
                   epochs: int = 400,
                   lr: float = 0.01,
                   weight_decay: float = 0.,
                   verbose: bool = False):
        x = torch.zeros_like(x_gt, requires_grad=True)
        lossf = nn.MSELoss(reduction='sum')
        # lossf = lambda x: torch.sum((y - Model(x))**2) + 1E-5 * torch.sum(x**2)
        optimizer = optim.Adam([x], lr=lr, eps=1E-5,
                               weight_decay=weight_decay)

        rhs = Model.adjoint(Model.y)

        for epoch in range(epochs):
            lhs = Model.adjoint(Model(x)) + 1E-5 * x
            res = lossf(torch.view_as_real(lhs), torch.view_as_real(rhs))

            # res = lossf(x)

            optimizer.zero_grad()
            res.backward()
            optimizer.step()

            if verbose:
                print('> epoch %3d loss %.12f'%(epoch, res.item()))

        return x

    def test_sense_model(self):
        for device in devices:
            img_shape = [1, 1, 1, 1, 16, 16]
            mps_shape = [1, 1, 8, 1, 16, 16]

            img = torch.randn(img_shape, dtype=torch.cfloat, device=device)
            mps = torch.randn(mps_shape, dtype=torch.cfloat, device=device)

            y1 = fourier.fft(mps * img, dim=(-2, -1))

            S = mri.Sense(mps, y1).to(device)
            y2 = S(img)

            ptt.assert_close(y1, y2)

            # solve with CG
            x_lsqr = self.solve_mri_lsqr(S, y2)
            ptt.assert_close(x_lsqr, img, atol=1E-5, rtol=1E-5)

            # TODO: solve with Torch
            # x = self.solve_mri_torch(S, y2, img, lr=0.1, epochs=400)
            # ptt.assert_close(x, img, atol=1E-5, rtol=1E-5)

    def test_sense_model_basis_tensor(self):
        for device in devices:
            subim_shape = [5, 1, 1, 1, 16, 16]
            basis_shape = [35, 5]
            mps_shape = [1, 1, 8, 1, 16, 16]

            subim = torch.randn(subim_shape, dtype=torch.cfloat, device=device)
            basis = torch.randn(basis_shape, dtype=torch.cfloat, device=device)
            mps = torch.randn(mps_shape, dtype=torch.cfloat, device=device)

            totim = basis @ subim.view(subim.shape[0], -1)
            totim = totim.view([basis_shape[0]] + subim_shape[1:])
            y = fourier.fft(mps * totim, dim=(-2, -1))

            S = mri.Sense(mps, y, basis=basis).to(device)
            y_fwd = S(subim)

            assert y.device == y_fwd.device
            ptt.assert_close(y, y_fwd)

            # solve with CG
            x_lsqr = self.solve_mri_lsqr(S, y_fwd, max_iter=100)
            ptt.assert_close(x_lsqr, subim, atol=1E-5, rtol=1E-5)

    def test_sense_model_basis_function(self):
        None # TODO: when basis is Callable

    def test_sense_model_phase_echo(self):
        for device in devices:
            img_shape = [1, 7, 1, 1, 16, 16]
            mps_shape = [1, 1, 8, 1, 16, 16]

            img = torch.randn(img_shape, dtype=torch.cfloat, device=device)
            phs = torch.randn(img_shape, dtype=torch.cfloat, device=device)
            mps = torch.randn(mps_shape, dtype=torch.cfloat, device=device)

            y = fourier.fft(mps * phs * img, dim=(-2, -1))

            S = mri.Sense(mps, y, phase_echo=phs).to(device)
            y_fwd = S(img)

            assert y.device == y_fwd.device
            ptt.assert_close(y, y_fwd)

    def test_sense_model_phase_slice(self):
        for device in devices:
            img_shape = [1, 1, 1, 3, 16, 16]
            mps_shape = [1, 1, 8, 3, 16, 16]

            img = torch.randn(img_shape, dtype=torch.cfloat, device=device)
            phs = torch.randn(img_shape, dtype=torch.cfloat, device=device)
            mps = torch.randn(mps_shape, dtype=torch.cfloat, device=device)

            y = torch.sum(phs * fourier.fft(mps * img, dim=(-2, -1)), dim=DIM_Z, keepdim=True)

            S = mri.Sense(mps, y, phase_slice=phs).to(device)
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

    def test_data_consistency(self):
        for device in devices:
            img_shape = [1, 1, 1, 3, 16, 16]
            mps_shape = [1, 1, 8, 3, 16, 16]

            mps7 = []
            ksp7 = []
            phs7 = []
            img7 = []
            for _ in range(3):  # REPETITION
                img = torch.randn(img_shape, dtype=torch.cfloat, device=device)
                phs = torch.randn(img_shape, dtype=torch.cfloat, device=device)
                mps = torch.randn(mps_shape, dtype=torch.cfloat, device=device)

                ksp = torch.sum(phs * fourier.fft(mps * img, dim=(-2, -1)),
                                dim=DIM_Z, keepdim=True)

                mps7.append(mps)
                ksp7.append(ksp)
                phs7.append(phs)
                img7.append(img)

            mps7 = torch.stack(mps7)
            ksp7 = torch.stack(ksp7)
            phs7 = torch.stack(phs7)
            img7 = torch.stack(img7)

            DC = mri.DataConsistency(mps7, ksp7, phase_slice=phs7, lamda=1E-6)

            print(len(DC.SENSE_ModuleList))

            x = DC()

            ptt.assert_close(x, img7)
