import numpy as np
import torch
import torch.testing as ptt
import unittest

from deepdwi.recons import linsub
from deepdwi.models import bloch

import h5py
import os
DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]
HOME_DIR = HOME_DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

if __name__ == 'main':
    unittest.main()


devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices.append(torch.device('cuda'))

class testLinsub(unittest.TestCase):

    def test_t2(self):

        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')

        for device in devices:

            TE = np.linspace(0.1, 80.1, 81) * 0.001

            y1 = torch.tensor(bloch.model_t2(TE), device=device)

            N_te, N_atom = y1.shape

            Ut = linsub.learn_linear_subspace(y1, num_coeffs=10)

            y2 = Ut @ Ut.T @ y1.view(N_te, -1)

            ptt.assert_close(y1, y2, rtol=1E-6, atol=1E-6)

    def test_dwi_3scan_trace(self):

        def _set_b_g(list_bvals: list[int] = [0, 100, 800, 1600],
                     list_bavgs: list[int] = [3, 4, 6, 10]):

            trace_array = [[1, 1, -0.5],
                           [1, -0.5, 1],
                           [-0.5, 1, 1]]

            N_diff = len(trace_array)

            b0_array = np.array([0, 0, 0]).reshape((1, 3))

            uvecs = np.array([1, 1, 1]).reshape((3, 1))

            bvals = np.array([]).reshape((0, 1))
            bvecs = np.array([]).reshape((0, 3))

            list_bvecs = [b0_array, trace_array, trace_array, trace_array]

            N_repet = np.sum(list_bavgs)

            cnt = [0] * len(list_bvals)

            cnt_repet = 0

            while cnt_repet < N_repet:

                for n in range(len(list_bvals)):

                    if cnt[n] < list_bavgs[n]:
                        if list_bvals[n] == 0:
                            bval = np.array([list_bvals[n]]).reshape((1, 1))
                        else:
                            bval = np.array([list_bvals[n]] * 3).reshape((3, 1))

                        bvals = np.append(bvals, bval, axis=0)
                        bvecs = np.append(bvecs, list_bvecs[n], axis=0)

                        cnt[n] += 1

                        cnt_repet += 1

            return bvals, bvecs

        bvals, bvecs = _set_b_g()

        print('> bvals shape: ', bvals.shape)
        print('  ', bvals)
        print('> bvecs shape: ', bvecs.shape)
        print('  ', bvecs)


        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')

        for device in devices:

            y1, _ = bloch.model_DTI(bvals, bvecs,
                                    Dxx=(0, 3E-3, 50),
                                    Dyy=(0, 3E-3, 50),
                                    Dzz=(0, 3E-3, 50),
                                    Dxy=(0, 0, 1),
                                    Dxz=(0, 0, 1),
                                    Dyz=(0, 0, 1))

            y1 = torch.tensor(y1, device=device)

            print(y1.shape)

            N_b, N_atom = y1.shape

            Ut = linsub.learn_linear_subspace(y1, num_coeffs=10)

            y2 = Ut @ Ut.T @ y1.view(N_b, -1)

            ptt.assert_close(y1, y2, rtol=1E-6, atol=1E-6)

    def test_dti_64dir(self):

        with h5py.File(HOME_DIR + '/src/deepdwi/models/dvs_bval-0000-0400_bvec-020-064.h5', 'r') as f:
            bvals = np.reshape(f['bvals'][:], (-1,1))
            bvecs = f['bvecs'][:]

        print('> bvals shape: ', bvals.shape)
        print('  ', bvals)
        print('> bvecs shape: ', bvecs.shape)
        print('  ', bvecs)


        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')

        for device in devices:

            y1, _ = bloch.model_DTI(bvals, bvecs,
                                    Dxx=(0, 3E-3, 10),
                                    Dyy=(0, 3E-3, 10),
                                    Dzz=(0, 3E-3, 10),
                                    Dxy=(0, 3E-3, 10),
                                    Dxz=(0, 3E-3, 10),
                                    Dyz=(0, 3E-3, 10))

            y1 = torch.tensor(y1, device=device)

            print(y1.shape)

            N_b, N_atom = y1.shape

            Ut = linsub.learn_linear_subspace(y1, num_coeffs=32)

            y2 = Ut @ Ut.T @ y1.view(N_b, -1)

            ptt.assert_close(y1, y2, rtol=1E-6, atol=1E-6)
