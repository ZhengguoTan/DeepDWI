import unittest

import torch
import torch.testing as ptt

from deepdwi import lsqr
from deepdwi.dims import *

if __name__ == "__main__":
    unittest.main()


devices = [torch.device('cpu'), torch.device('cuda')]

class TestLsqr(unittest.TestCase):
    def Ax_setup(self, n, dtype=torch.float32):
        A = torch.eye(n) + 0.1 * torch.ones([n, n])
        x = torch.arange(n, dtype=dtype)
        return A, x

    def Ax_y_setup(self, n, lamda):
        A, x = self.Ax_setup(n, dtype=torch.float32)
        y = A @ x
        x_torch = torch.linalg.solve(
            A.T @ A + lamda * torch.eye(n), A.T @ y)

        return A, x_torch, y

    def test_ConjugateGradient(self):
        n = 5
        lamda = 0.1
        A, x_torch, y = self.Ax_y_setup(n, lamda)

        x = torch.zeros_like(x_torch)

        A_func = lambda x: A.T @ A @ x + lamda * x
        CG = lsqr.ConjugateGradient(A_func, A.T @ y, x)

        x = CG()

        ptt.assert_close(x, x_torch)