"""
This module implements linear subspace methods

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import torch

from torch import Tensor
from torch.linalg import norm, svd
# from torch.nn import functional as F


def get_relative_error(reco_sig, orig_sig):
    return norm(orig_sig - reco_sig).item() \
            / norm(orig_sig).item()


def learn_linear_subspace(sig: Tensor,
                          num_coeffs: int = 5,
                          error_bound: float = 1E-5,
                          use_error_bound: bool = False):
    """learn a linear subspace matrix from signal dictionary via singular value decomposition (SVD).

    Args:
        sig (Tensor): MR signal
        num_coeffs (int): expected number of coefficients
        error_bound (float): relative error bound between the subspace signal
                            and the ground-truth signal
        use_error_bound (bool): iteratively increase the number of coefficients
                            in order to reach the error_bound

    Returns:
        U_sub (Tensor): linear subspace basis matrix

    References:
        * Huang C, Graff CG, Clarkson EW, Bilgin A, Altbach MI (2012).
          T2 mapping from highly undersampled data by
          reconstruction of principal component coefficient maps
          using compressed sensing.
          Magn. Reson. Med., 67, 1355-1366.

        * Tamir JI, Uecker M, Chen W, Lai P, Alley MT, Vasanawala SS, Lustig M (2017).
          T2 shuffling: Sharp, multicontrast, volumetric fast spin-echo imaging.
          Magn. Reson. Med., 77, 180-195.

    Author:
        Zhengguo Tan <zhengguo.tan@gmail.com>
    """
    # device
    print('> device: ', sig.device)

    # contrast stored in 0th dim of sig
    sig2 = sig.view(sig.shape[0], -1)

    # singular value decomposition
    U, S, VH = svd(sig2, full_matrices=False)

    while True:
        # truncate U
        U_sub = U[:, :num_coeffs]

        # reconstruction from U_sub
        reco_sig = U_sub @ U_sub.T @ sig2

        err = get_relative_error(reco_sig, sig2)
        # err = F.mse_loss(reco_sig, sig2)

        if (err > error_bound) and use_error_bound:
            num_coeffs += 1

        else:
            print('Eventual number of subspace coefficients: ', num_coeffs)
            print('Eventual relative error: ', err)

            return U_sub