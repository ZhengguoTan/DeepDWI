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
import torch.jit as jit
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
        verbose (bool): display debug messages. Default is False.

    """

    def __init__(self, A, b: torch.Tensor, x: torch.Tensor,
                 P=None, damp: float = 0, x0: Optional[torch.Tensor] = None,
                 max_iter: int = 100, tol: float = 0,
                 verbose: bool = False):
        r"""
        initilizes the conjugate gradient method
        """
        super(ConjugateGradient, self).__init__()

        self.device = b.device
        self.b = b

        if jit.isinstance(A, torch.Tensor) or jit.isinstance(A, nn.Module):
            self.A = A.to(self.device)
        else:
            self.A = A

        self.x = x.to(self.device)

        if jit.isinstance(P, torch.Tensor) or jit.isinstance(P, nn.Module):
            self.P = P.to(self.device)
        else:
            self.P = P

        self.damp = damp
        if x0 is None:
            self.x0 = torch.zeros_like(x).to(self.device)
        else:
            self.x0 = x0.to(self.device)

        self.iter = 0
        self.max_iter = max_iter
        self.tol = tol
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
        r"""
        custom implementation of the `to` function in nn.Module
        """
        self.device = device
        if jit.isinstance(self.A, torch.Tensor) or jit.isinstance(self.A, nn.Module):
            self.A = self.A.to(device)

        if jit.isinstance(self.P, torch.Tensor) or jit.isinstance(self.P, nn.Module):
            self.P = self.P.to(device)

        self.r = self.r.to(device)
        self.p = self.p.to(device)

        return super(ConjugateGradient, self).to(device)

    def forward(self):
        while not (self.iter >= self.max_iter or self.not_positive_definite or self.resid <= self.tol):
            Ap = self.A(self.p)
            pAp = torch.real(torch.vdot(self.p.flatten(), Ap.flatten())).item()
            if pAp <= 0:
                self.not_positive_definite = True
                return self.x

            self.alpha = self.rzold / pAp
            if torch.isnan(self.alpha).any() or torch.isinf(self.alpha).any():
                self.alpha = torch.zeros_like(self.alpha)

            self.x = self.x + self.alpha * self.p
            if self.iter < self.max_iter - 1:
                self.r = self.r - self.alpha * Ap
                if self.P is not None:
                    z = self.P(self.r)
                else:
                    z = self.r

                rznew = torch.real(torch.vdot(self.r.flatten(), z.flatten()))
                beta = rznew / self.rzold
                self.p = beta * self.p + z
                self.rzold = rznew

            self.resid = self.rzold.item()**0.5

            if self.verbose:
                print("  cg iter: " + "%2d" % (self.iter)
                      + "; resid: " + "%13.6f" % (self.resid)
                      + "; norm: " + "%13.6f" % (torch.linalg.norm(self.x.flatten())))

            self.iter += 1

        return self.x
