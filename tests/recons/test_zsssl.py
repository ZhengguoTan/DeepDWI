import unittest

import numpy as np

import torch
import torch.nn as nn
import torch.testing as ptt

from torch.utils.data import DataLoader

from deepdwi import fourier, util
from deepdwi.dims import *
from deepdwi.recons import zsssl


if __name__ == "__main__":
    unittest.main()


devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices.append(torch.device('cuda'))

class TestZSSSL(unittest.TestCase):

    def _setup_y_PFCx(self, N_rep: int = 1, verbose: bool = False):
        img_shape = [1, 1, 1, 2, 16, 16]
        eco_shape = [1, 5, 1, 2, 16, 16]
        mps_shape = [1, 1, 8, 2, 16, 16]

        img = []
        mps = []
        phase_shot = []
        phase_slice = []

        for _ in range(N_rep):
            img.append(torch.randn(img_shape, dtype=torch.cfloat))
            mps.append(torch.randn(mps_shape, dtype=torch.cfloat))
            phase_shot.append(torch.rand(eco_shape, dtype=torch.cfloat))
            phase_slice.append(torch.randn(img_shape, dtype=torch.cfloat))

        img = torch.stack(img)
        mps = torch.stack(mps)
        phase_shot = torch.stack(phase_shot)
        phase_slice = torch.stack(phase_slice)

        ksp = torch.sum(phase_slice * fourier.fft(mps * phase_shot * img, dim=(-2, -1)), dim=DIM_Z, keepdim=True)

        mask = util.estimate_weights(ksp, coil_dim=DIM_COIL)

        train_mask, lossf_mask = zsssl.uniform_samp(mask)

        if verbose:
            print('> ksp shape: ', ksp.shape)
            print('> train_mask shape: ', train_mask.shape)
            print('> lossf_mask shape: ', lossf_mask.shape)
            print('> mps shape: ', mps.shape)
            print('> img shape: ', img.shape)
            print('> phase_shot shape: ', phase_shot.shape)
            print('> phase_slice shape: ', phase_slice.shape)

        return ksp, train_mask, lossf_mask, mps, img, phase_shot, phase_slice

    def test_samp(self):
        mask = torch.zeros([32, 2, 1, 1, 16, 16])
        mask[..., ::2, :] = 1

        train_mask, lossf_mask = zsssl.uniform_samp(mask)

        ptt.assert_close(train_mask + lossf_mask, mask)

    def test_dataset(self):
        ksp, train_mask, lossf_mask, mps, img, phase_shot, phase_slice = self._setup_y_PFCx()

        for device in devices:

            Dataset = zsssl.Dataset(mps, ksp, train_mask, lossf_mask, phase_shot, phase_slice)

            Loader = DataLoader(Dataset, batch_size=1, shuffle=True, num_workers=1)

            for i, (mps_b, ksp_b, train_mask_b, lossf_mask_b, phase_shot_b, phase_slice_b) in enumerate(Loader):

                mps_b = mps_b.to(device)
                ksp_b = ksp_b.to(device)
                train_mask_b = train_mask_b.to(device)
                lossf_mask_b = lossf_mask_b.to(device)
                phase_shot_b = phase_shot_b.to(device)
                phase_slice_b = phase_slice_b.to(device)

                ptt.assert_close(mps_b, mps.to(device))
                ptt.assert_close(ksp_b, ksp.to(device))
                ptt.assert_close(train_mask_b, train_mask.to(device))
                ptt.assert_close(lossf_mask_b, lossf_mask.to(device))
                ptt.assert_close(phase_shot_b, phase_shot.to(device))
                ptt.assert_close(phase_slice_b, phase_slice.to(device))

    def test_trafos_normal(self):
        for device in devices:
            img_shape = [10, 5, 1, 1, 3, 16, 24]

            img = torch.randn(img_shape, dtype=torch.cfloat, device=device)
            for contrasts_in_channels in [True, False]:

                T = zsssl.Trafos(tuple(img_shape),
                                 contrasts_in_channels=contrasts_in_channels)

                output = T(img)
                output = T.adjoint(output)

                ptt.assert_close(output, img)

    def test_trafos(self):
        for device in devices:
            dd_shape = [1, 18, 1, 1, 1, 192, 224]

            img_dd = torch.randn(dd_shape, dtype=torch.cfloat, device=device)

            T = zsssl.Trafos(tuple(dd_shape), contrasts_in_channels=True)
            fwd_dd = T(img_dd)
            adj_dd = T.adjoint(fwd_dd)
            print('> fwd_dd shape: ', fwd_dd.shape, ', adj_dd shape: ', adj_dd.shape)

            img_jm = torch.view_as_real(torch.squeeze(img_dd))
            img_jm = img_jm.permute(2, 1, 0, 3).unsqueeze(0)
            jm_shape = img_jm.shape
            print('> img_jm shape: ', jm_shape)

            fwd_jm = img_jm.contiguous().view((jm_shape[0], jm_shape[1], jm_shape[2], jm_shape[3]*jm_shape[4]))
            fwd_jm = fwd_jm.permute(0, 3, 1, 2)

            adj_jm = fwd_jm.permute(0, 2, 3, 1)
            adj_jm = adj_jm.view((jm_shape))
            print('> fwd_jm shape: ', fwd_jm.shape, ', adj_jm shape: ', adj_jm.shape)

            adj_jm = adj_jm.squeeze(0).permute(2, 1, 0, 3)
            adj_jm = torch.view_as_complex(adj_jm)
            adj_jm = adj_jm[None, :, None, None, None, :, :]
            print('> adj_jm shape: ', adj_jm.shape)

            ptt.assert_close(fwd_dd, fwd_jm)
            ptt.assert_close(adj_dd, adj_jm)

    def test_algunroll(self):

        ishape = [1, 18, 1, 1, 1, 192, 224]

        model = zsssl.AlgUnroll(ishape, lamda=0.05, NN='ResNet2D',
                                requires_grad_lamda=True,
                                N_residual_block=12,
                                N_unroll=8,
                                features=128,
                                contrasts_in_channels=True,
                                max_cg_iter=6)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(">>> number of trainable parameters is: ", params)
