import unittest

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

    def test_model(self):
        ksp, train_mask, lossf_mask, mps, img, phase_shot, phase_slice = self._setup_y_PFCx()

        for device in devices:

            Dataset = zsssl.Dataset(mps, ksp, train_mask, lossf_mask, phase_shot, phase_slice)

            # num_workers must be 0 for GPUs
            Loader = DataLoader(Dataset, batch_size=1, shuffle=True, num_workers=1)

            Model = zsssl.UnrollNet(lamda=1E-6, requires_grad_lamda=False, N_unroll=1,
                                    max_cg_iter=50).to(device)

            for i, (mps_b, ksp_b, train_mask_b, lossf_mask_b, phase_shot_b, phase_slice_b) in enumerate(Loader):

                mps_b = mps_b.to(device)
                ksp_b = ksp_b.to(device)
                train_mask_b = train_mask_b.to(device)
                lossf_mask_b = lossf_mask_b.to(device)
                phase_shot_b = phase_shot_b.to(device)
                phase_slice_b = phase_slice_b.to(device)

                train_ksp_b = train_mask_b * ksp_b
                Train_SENSE_ModuleList = zsssl._build_SENSE_ModuleList(mps_b, train_ksp_b, phase_shot_b, phase_slice_b)

                lossf_ksp_b = lossf_mask_b * ksp_b
                Lossf_SENSE_ModuleList = zsssl._build_SENSE_ModuleList(mps_b, lossf_ksp_b, phase_shot_b, phase_slice_b)

                x = torch.zeros_like(img, device=device)

                x, lamda, _ = Model(x, Train_SENSE_ModuleList, Lossf_SENSE_ModuleList)

                ptt.assert_close(x, img.to(device))

    def test_trafos(self):
        for device in devices:
            img_shape = [10, 5, 1, 1, 3, 16, 24]

            img = torch.randn(img_shape, dtype=torch.cfloat, device=device)
            T = zsssl.Trafos(tuple(img_shape + [2]))
            output = T.adjoint(T(img))

            ptt.assert_close(output, img)