"""
This module implements MRI models

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import torch
import torch.jit as jit
import torch.nn as nn

from typing import Callable, Union

from deepdwi import fourier, lsqr, util
from deepdwi.dims import *


# %%
class Sense(nn.Module):
    """
    Generalized sensitivity encoding (SENSE) forward modeling.
    This class constructs the SENSE forward operator as an nn.Module.

    Args:
        coils (Tensor): coil sensitivity maps.
        y (Tensor): sampled k-space data.
        basis (Tensor or nn.Module or Callable function): basis projection. Default is None.
        N_basis (int): Number of basis. Only used when basis is Callable or nn.Module. Default is 7.
        baseline (Tensor): the baseline image to be multiplied with the basis model output. Default is None.
        phase_echo (Tensor): phase maps of echoes or shots. Default is None.
        combine_echo (bool): reconstruct only one echo or combine shots. Default is None.
        phase_slice (Tensor): multi-band slices phases. Default is None.
        coord (Tensor): non-Cartesian trajectories. Default is None.
        weights (Tensor): k-space weights or sampling masks. Default is None.

    Return:
        A Sense nn.Module.

    References:
    * Pruessmann KP, Weiger M, Börnert P, Boesiger P.
      Advances in sensitivity encoding with arbitrary k-space trajectories.
      Magn Reson Med 2001;46:638-651. doi: 10.1002/mrm.1241.
    """
    def __init__(self,
                 coils: torch.Tensor,
                 y: torch.Tensor,
                 basis: Union[torch.Tensor, nn.Module, Callable] = None,
                 N_basis: int = 7,
                 baseline: torch.Tensor = None,
                 phase_echo: torch.Tensor = None,
                 combine_echo: bool = False,
                 phase_slice: torch.Tensor = None,
                 coord: torch.Tensor = None,
                 weights: torch.Tensor = None):
        r"""
        Initialize Sense as a nn.Module.
        """
        super(Sense, self).__init__()

        self.device = y.device

        self.y = y
        self.coils = coils.to(self.device)

        # k-space data shape in accordance with dims.py
        N_time, N_echo, N_coil, N_z, N_y, N_x = y.shape[-6:]
        extra_shapes = y.shape[:-6]

        # deal with collapsed y even for SMS
        assert(1 == N_z)

        if phase_slice is not None:
            MB = phase_slice.shape[DIM_Z]
            self.phase_slice = phase_slice.to(self.device)
        else:
            MB = 1
            self.phase_slice = None

        # start to construct image shape
        img_shape = [1] + [MB] + [N_y] + [N_x]

        # basis
        if basis is not None:
            if jit.isinstance(basis, torch.Tensor):
                assert(N_time == basis.shape[0])
                x_time = basis.shape[1]
            else:
                x_time = N_basis

            if jit.isinstance(basis, torch.Tensor) or jit.isinstance(basis, nn.Module):
                self.basis = basis.to(self.device)
            else:
                self.basis = basis

            ishape = list(extra_shapes) + [x_time] + [1] + img_shape  # TODO:

        else:
            x_time = N_time
            self.basis = None

            if combine_echo is True:

                assert(phase_echo is not None or N_echo == 1)
                ishape = list(extra_shapes) + [x_time] + [1] + img_shape

            else:

                ishape = list(extra_shapes) + [x_time] + [N_echo] + img_shape

        self.baseline = baseline

        # echo or shot
        if phase_echo is not None:
            self.phase_echo = phase_echo.to(self.device)
        else:
            self.phase_echo = None

        self.ishape = ishape
        self.oshape = y.shape

        if coord is not None:
            self.coord = coord.to(self.device)
        else:
            self.coord = None

        # samling mask
        if weights is None and coord is None:
            weights = (util.rss(y, dim=(DIM_COIL, ), keepdim=True) > 0).type(y.dtype)

        self.weights = weights.to(self.device) if weights is not None else None

    def to(self, device):
        r"""
        custom implementation of the `to` function in nn.Module
        """
        self.device = device

        self.coils.to(self.device)
        self.y.to(self.device)

        if jit.isinstance(self.basis, torch.Tensor) or jit.isinstance(self.basis, nn.Module):
            self.basis = self.basis.to(self.device)

        if self.baseline is not None:
            self.baseline = self.baseline.to(self.device)

        if self.phase_echo is not None:
            self.phase_echo = self.phase_echo.to(self.device)

        if self.phase_slice is not None:
            self.phase_slice = self.phase_slice.to(self.device)

        if self.coord is not None:
            self.coord = self.coord.to(self.device)

        if self.weights is not None:
            self.weights = self.weights.to(self.device)

        return super(Sense, self).to(device)

    def forward(self, x):

        assert torch.is_tensor(x)
        img_post_shape = list(x.shape[-5:])
        img_prev_shape = list(x.shape[:-6])

        # subspace modeling
        if self.basis is not None:

            x2 = torch.swapaxes(x, 0, -6)
            x2 = x2.view(x2.shape[0], -1)

            if jit.isinstance(self.basis, torch.Tensor):
                # linear subspace matrix
                N_ful, N_sub = self.basis.shape

                x1 = self.basis @ x2

            elif jit.isinstance(self.basis, nn.Module):
                # deep nonlinear subspace
                x2 = x2.T
                x1 = self.basis.decode(x2)
                x1 = x1.T

            elif jit.isinstance(self.basis, Callable):
                x1 = self.basis(x2)

            x1 = x1.view(tuple([self.y.shape[-6]] + [*img_prev_shape, *img_post_shape]))
            x_proj = torch.swapaxes(x1, -6, 0)


            if self.baseline is not None:
                x_proj = self.baseline * x_proj

        else:
            x_proj = x

        # phase modeling
        if self.phase_echo is not None:
            x_phase = self.phase_echo * x_proj
        else:
            x_phase = x_proj

        # coil sensitivity maps
        x_coils = self.coils * x_phase

        # FFT
        if self.coord is None:
            x_kspace = fourier.fft(x_coils, dim=(-2, -1))
        else:
            None # TODO: NUFFT

        # SMS
        if self.phase_slice is not None:
            x_kslice = torch.sum(self.phase_slice * x_kspace, dim=DIM_Z, keepdim=True)
        else:
            x_kslice = x_kspace

        # k-space sampling mask
        y = self.weights * x_kslice

        self._check_two_shape(y.shape, self.y.shape)

        return y

    def adjoint(self, input):

        self._check_two_shape(input.shape, self.y.shape)

        # k-space sampling mask
        output = torch.conj(self.weights) * input

        # SMS
        if self.phase_slice is not None:
            N_slice = self.phase_slice.shape[DIM_Z]

            tile_dims = []
            for d in range(-output.dim(), 0):
                tile_dims.append(1 if d != DIM_Z else N_slice)

            output = torch.conj(self.phase_slice) * torch.tile(output, dims=tuple(tile_dims))

        # FFT
        if self.coord is None:
            output = fourier.ifft(output, dim=(-2, -1))
        else:
            None # TODO: NUFFT

        # coil sensitivity maps
        output = torch.sum(torch.conj(self.coils) * output,
                           dim=DIM_COIL, keepdim=True)

        # phase modeling
        if self.phase_echo is not None:
            output = torch.sum(torch.conj(self.phase_echo) * output,
                               dim=DIM_ECHO, keepdim=True)

        # subspace modeling
        if jit.isinstance(self.basis, torch.Tensor):
            N_ful, N_sub = self.basis.shape
            basis_t = self.basis.conj().swapaxes(-1, -2)
            x1 = basis_t @ output.view(output.shape[0], -1)

            output = x1.view([N_sub] + list(output.shape[1:]))

        elif jit.isinstance(self.basis, nn.Module):
            raise RuntimeError('This model is nonlinear.')

        return output

    def _check_two_shape(self, ref_shape, dst_shape):
        for i1, i2 in zip(ref_shape, dst_shape):
            if (i1 != i2):
                raise ValueError('shape mismatch for ref {ref}, got {dst}'.format(
                    ref=ref_shape, dst=dst_shape))


# %%
class DataConsistency(nn.Module):
    def __init__(self, sens: torch.Tensor,
                 kspace: torch.Tensor,
                 basis: Union[torch.Tensor, nn.Module] = None,
                 phase_echo: torch.Tensor = None,
                 phase_slice: torch.Tensor = None,
                 lamda: float = 1E-2,
                 max_iter: int = 100,
                 tol: float = 0):

        super(DataConsistency, self).__init__()

        assert kspace.dim() == 7

        self.kspace = kspace
        self.sens = sens

        SENSE_ModuleList = nn.ModuleList()  # empty ModuleList

        batch_size = kspace.shape[DIM_REP]
        for b in range(batch_size):

            sens_r = sens[b]
            kspace_r = kspace[b]
            basis_r = basis[b] if basis is not None else None
            phase_echo_r = phase_echo[b] if phase_echo is not None else None
            phase_slice_r = phase_slice[b] if phase_slice is not None else None

            SENSE = Sense(sens_r, kspace_r, basis=basis_r,
                          phase_echo=phase_echo_r,
                          phase_slice=phase_slice_r)

            SENSE_ModuleList.append(SENSE)

        self.SENSE_ModuleList = SENSE_ModuleList
        self.lamda = lamda
        self.max_iter = max_iter
        self.tol = tol

    def forward(self):

        x = []
        for A in self.SENSE_ModuleList:

            AHA = lambda x: A.adjoint(A.forward(x)) + self.lamda * x
            AHy = A.adjoint(A.y)

            x_init = torch.zeros_like(AHy)
            CG = lsqr.ConjugateGradient(AHA, AHy, x_init,
                                        max_iter=self.max_iter,
                                        tol=self.tol)

            x.append(CG())

        # convert a list of tensors to one tensor
        x = torch.stack(x)
        return x
