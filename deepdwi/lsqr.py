"""
this module provides a PyTorch implementation
for the least square althorithm, such as
the conjugate gradient method.

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>

References:
    * https://sigpy.readthedocs.io/en/latest/generated/sigpy.alg.ConjugateGradient.html
    * https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
"""

import torch
import torch.nn as nn

from typing import Optional

from deepdwi import util

class ConjugateGradient(nn.Module):
    """Conjugate gradient method.

    Solves for:

    .. math:: A x = b

    where A is a Hermitian linear operator.

    Args:
        A (nn.Module or function): nn.Module or function to compute A.
        b (tensor): Observation.
        x (tensor): Variable.
        P (function or None): Preconditioner. Default is None.
        damp (float): damping factor. Default is 0.
        x0 (Tensor): initial guess of x. Defaut is None.
        max_iter (int): Maximum number of iterations. Default is 100.
        tol (float): Tolerance for stopping condition. Default is 0.
        device: Default is torch.device('cpu').
        verbose (bool): display debug messages. Default is False.

    """

    def __init__(self, A, b: torch.Tensor, x: torch.Tensor,
                 P=None, damp: float = 0, x0: Optional[torch.Tensor] = None,
                 max_iter: int = 100, tol: float = 0,
                 device = torch.device('cpu'),
                 verbose: bool = False):
        r"""
        initilizes the conjugate gradient method
        """
        super(ConjugateGradient, self).__init__()

        self.A = A
        self.b = b
        self.x = x
        self.P = P

        self.damp = damp
        if x0 is None:
            self.x0 = torch.zeros_like(x)
        else:
            self.x0 = x0

        self.iter = 0
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.verbose = verbose


        self.r = b - self.A(self.x) + damp * self.x0

        if self.P is None:
            z = self.r
        else:
            z = self.P(self.r)

        if max_iter > 1:
            self.p = z.clone()
        else:
            self.p = z

        self.not_positive_definite = False
        self.rzold = torch.real(torch.vdot(self.r.flatten(), z.flatten()))
        self.resid = self.rzold.item()**0.5

    def to(self, device):
        return super(ConjugateGradient, self).to(device)

    def forward(self):
        while not (self.iter >= self.max_iter or self.not_positive_definite or self.resid <= self.tol):
            Ap = self.A(self.p)
            pAp = torch.real(torch.vdot(self.p.flatten(), Ap.flatten())).item()
            if pAp <= 0:
                self.not_positive_definite = True
                return self.x

            self.alpha = self.rzold / pAp
            util.axpy(self.x, self.alpha, self.p)
            if self.iter < self.max_iter - 1:
                util.axpy(self.r, -self.alpha, Ap)
                if self.P is not None:
                    z = self.P(self.r)
                else:
                    z = self.r

                rznew = torch.real(torch.vdot(self.r.flatten(), z.flatten()))
                beta = rznew / self.rzold
                util.xpay(self.p, beta, z)
                self.rzold = rznew

            self.resid = self.rzold.item()**0.5

            if self.verbose:
                print("  cg iter: " + "%2d" % (self.iter)
                      + "; resid: " + "%13.6f" % (self.resid)
                      + "; norm: " + "%13.6f" % (torch.linalg.norm(self.x.flatten())))

            self.iter += 1

        return self.x
