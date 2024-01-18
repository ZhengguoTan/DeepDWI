import unittest

import torch
import torch.nn as nn
import torch.testing as ptt

from torch.utils.data import DataLoader

if __name__ == "__main__":
    unittest.main()


class TestSet(torch.utils.data.Dataset):
    def __init__(self, ten: torch.Tensor):

        self.ten = ten

    def __len__(self):
        return len(self.ten)

    def __getitem__(self, idx):
        return self.ten[idx]


class TestLoader(unittest.TestCase):

    def test_idx(self):
        mps_shape = [1, 1, 8, 1, 16, 16]

        N_rep = 7
        mps7 = []
        for _ in range(N_rep):
            mps7.append(torch.randn(mps_shape, dtype=torch.cfloat))

        mps7 = torch.stack(mps7)

        DS = TestSet(mps7)

        DL = DataLoader(DS, batch_size=1, shuffle=True, num_workers=6)

        for _, ten_b in enumerate(DL):

            print('> ten_b shape: ', ten_b.shape)
