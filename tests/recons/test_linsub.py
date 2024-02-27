import numpy as np
import torch
import torch.testing as ptt
import unittest

from deepdwi.recons import linsub
from deepdwi.models import bloch

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

            Ut = linsub.learn_linear_subspace(y1, num_coeffs=10,
                                              device=device)

            y2 = Ut @ Ut.T @ y1.view(N_te, -1)

            ptt.assert_close(y1, y2, rtol=1E-6, atol=1E-6)