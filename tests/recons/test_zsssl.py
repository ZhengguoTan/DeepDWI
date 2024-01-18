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


devices = [torch.device('cpu'), torch.device('cuda')]

class TestZSSSL(unittest.TestCase):

    def _setup_y_PFCx(self, N_rep: int = 1, verbose: bool = False):
        img_shape = [1, 1, 1, 1, 16, 16]
        mps_shape = [1, 1, 8, 1, 16, 16]

        img = []
        mps = []

        for _ in range(N_rep):
            img.append(torch.randn(img_shape, dtype=torch.cfloat))
            mps.append(torch.randn(mps_shape, dtype=torch.cfloat))

        img = torch.stack(img)
        mps = torch.stack(mps)

        ksp = fourier.fft(mps * img, dim=(-2, -1))

        mask = util.estimate_weights(ksp, coil_dim=DIM_COIL)

        train_mask, lossf_mask = zsssl.uniform_samp(mask)

        if verbose:
            print('> ksp shape: ', ksp.shape)
            print('> train_mask shape: ', train_mask.shape)
            print('> lossf_mask shape: ', lossf_mask.shape)
            print('> mps shape: ', mps.shape)
            print('> img shape: ', img.shape)

        return ksp, train_mask, lossf_mask, mps, img

    def test_samp(self):
        mask = torch.zeros([32, 2, 1, 1, 16, 16])
        mask[..., ::2, :] = 1

        train_mask, lossf_mask = zsssl.uniform_samp(mask)

        ptt.assert_close(train_mask + lossf_mask, mask)

    def test_dataset(self):
        ksp, train_mask, lossf_mask, mps, img = self._setup_y_PFCx()

        for device in devices:

            Dataset = zsssl.Dataset(mps, ksp, train_mask, lossf_mask)

            Loader = DataLoader(Dataset, batch_size=1, shuffle=True, num_workers=1)

            for i, (mps_b, ksp_b, train_mask_b, lossf_mask_b) in enumerate(Loader):

                mps_b = mps_b.to(device)
                ksp_b = ksp_b.to(device)
                train_mask_b = train_mask_b.to(device)
                lossf_mask_b = lossf_mask_b.to(device)

                ptt.assert_close(mps_b, mps.to(device))
                ptt.assert_close(ksp_b, ksp.to(device))
                ptt.assert_close(train_mask_b, train_mask.to(device))
                ptt.assert_close(lossf_mask_b, lossf_mask.to(device))

    def test_model(self):
        ksp, train_mask, lossf_mask, mps, img = self._setup_y_PFCx()

        for device in devices:

            Dataset = zsssl.Dataset(mps, ksp, train_mask, lossf_mask)

            # num_workers must be 0 for GPUs
            Loader = DataLoader(Dataset, batch_size=1, shuffle=True, num_workers=1)

            Model = zsssl.UnrollNet(lamda=1E-6, requires_grad_lamda=False, N_unroll=1).to(device)

            for i, (mps_b, ksp_b, train_mask_b, lossf_mask_b) in enumerate(Loader):

                mps_b = mps_b.to(device)
                ksp_b = ksp_b.to(device)
                train_mask_b = train_mask_b.to(device)
                lossf_mask_b = lossf_mask_b.to(device)

                train_ksp_b = train_mask_b * ksp_b
                Train_SENSE_ModuleList = zsssl._build_SENSE_ModuleList(mps_b, train_ksp_b)

                lossf_ksp_b = lossf_mask_b * ksp_b
                Lossf_SENSE_ModuleList = zsssl._build_SENSE_ModuleList(mps_b, lossf_ksp_b)

                x = torch.zeros_like(img, device=device)

                x, lamda, _ = Model(x, Train_SENSE_ModuleList, Lossf_SENSE_ModuleList)

                ptt.assert_close(x, img.to(device))