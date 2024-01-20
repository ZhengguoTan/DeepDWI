import unittest

import numpy as np

import torch
import torch.testing as ptt

from deepdwi import util
from deepdwi.dims import *

if __name__ == "__main__":
    unittest.main()


devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices.append(torch.device('cuda'))


class TestUtil(unittest.TestCase):

    def test_reshape(self):
        ishape = [10, 4, 5, 1, 2, 16, 16]
        oshape = [ishape[DIM_REP]] + [np.prod(ishape[DIM_TIME:DIM_Z])] + list(ishape[DIM_Z:])

        for device in devices:
            input = torch.randn(ishape, dtype=torch.cfloat, device=device)

            R = util.Reshape(oshape, ishape)

            output = R(input)

            ptt.assert_close(R.adjoint(output), input)
            ptt.assert_close(R.normal(input), input)

    def test_permute(self):
        ishape = [10, 4, 5, 1, 2, 16, 16]
        dims = [0, 4, 1, 2, 3, 5, 6]

        for device in devices:
            input = torch.randn(ishape, dtype=torch.cfloat, device=device)

            P = util.Permute(ishape, dims)

            output = P(input)

            ptt.assert_close(P.adjoint(output), input)
            ptt.assert_close(P.normal(input), input)
