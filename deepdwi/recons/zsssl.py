"""
This module implements Zero-Shot Self-Supervised Learning

"""
import torch
import torch.nn as nn

from deepdwi import lsqr, util
from deepdwi.models import mri, resnet

from typing import Callable, List, Optional, Tuple, Union

# %%
def uniform_samp(mask: torch.Tensor, rho: float = 0.2,
                 acs_block: Tuple[int, int] = (4, 4)):
    r"""
    Perform uniform sampling based on torch.multinomial.
    """

    N_y, N_x = mask.shape[-2:]

    # TODO: find the k-space center instead?
    C_y, C_x = N_y // 2, N_x // 2

    # mask_outer that excludes the ACS region
    outer_mask = mask.clone()
    outer_mask[..., C_y - acs_block[-2] : C_y + acs_block[-2],
               C_x - acs_block[-1] : C_x + acs_block[-1]] = 0

    nonzero_ind = torch.nonzero(outer_mask)  # indices of nonzero points
    N_nonzero = len(nonzero_ind)             # number of nonzero points

    weights = torch.ones([N_nonzero], dtype=torch.float)
    chosen_ones = torch.multinomial(weights, int(N_nonzero * rho), replacement=False)

    chosen_ind = nonzero_ind[chosen_ones].tolist()

    lossf_mask = torch.zeros_like(mask)

    for l in range(len(chosen_ind)):
        lossf_mask[tuple(chosen_ind[l])] = 1

    train_mask = mask - lossf_mask

    return train_mask, lossf_mask


# %% SENSE related functions
def _build_SENSE_ModuleList(sens: torch.Tensor,
                            mask_kspace: torch.Tensor,
                            phase_echo: torch.Tensor = None,
                            phase_slice: torch.Tensor = None) -> nn.ModuleList:

    SENSE_ModuleList = nn.ModuleList()

    N_batch = mask_kspace.shape[0]
    for b in range(N_batch):
        sens_r = sens[b]
        kspace_r = mask_kspace[b]  # masking
        phase_echo_r = phase_echo[b] if phase_echo is not None else None
        phase_slice_r = phase_slice[b] if phase_slice is not None else None

        SENSE = mri.Sense(sens_r, kspace_r,
                          phase_echo=phase_echo_r,
                          phase_slice=phase_slice_r)

        SENSE_ModuleList.append(SENSE)

    return SENSE_ModuleList

def _fwd_SENSE_ModuleList(SENSE_ModuleList: nn.ModuleList,
                          x: torch.Tensor) -> torch.Tensor:
    input = x.clone()

    y = []
    for l in range(len(SENSE_ModuleList)):
        A = SENSE_ModuleList[l]
        y.append(A(input[l]))

    return torch.stack(y)

def _adj_SENSE_ModuleList(SENSE_ModuleList: nn.ModuleList) -> torch.Tensor:
    AHy = []
    for l in range(len(SENSE_ModuleList)):
        A = SENSE_ModuleList[l]
        AHy.append(A.adjoint(A.y))

    return torch.stack(AHy)

def _solve_SENSE_ModuleList(SENSE_ModuleList: nn.ModuleList,
                            x0: torch.Tensor = None,
                            lamda: float = 0.01,
                            max_iter: int = 10,
                            tol: float = 0.) -> torch.Tensor:
    res = []
    for l in range(len(SENSE_ModuleList)):
        A = SENSE_ModuleList[l]

        # normal equation with Tikhonov regu
        AHA = lambda x: A.adjoint(A.forward(x)) + lamda * x
        # adjoint
        AHy = A.adjoint(A.y) + lamda * x0[l]

        CG = lsqr.ConjugateGradient(AHA, AHy, torch.zeros_like(AHy),
                                    max_iter=max_iter,
                                    tol=tol)

        res.append(CG())  # run CG

    return torch.stack(res)


# %%
class Dataset(torch.utils.data.Dataset):
    """
    A Dataset for Zero-Shot Self-Supervised Learning.
    """
    def __init__(self,
                 sens: torch.Tensor,
                 kspace: torch.Tensor,
                 train_mask: torch.Tensor,
                 lossf_mask: torch.Tensor,
                 phase_echo: torch.Tensor,
                 phase_slice: torch.Tensor):
        r"""
        Initializa a Dataset for zero-shot learning.
        """
        self._check_two_shape(train_mask.shape, lossf_mask.shape)
        self._check_tensor_dim(sens.dim(), 7)
        self._check_tensor_dim(kspace.dim(), 7)
        self._check_tensor_dim(train_mask.dim(), 7)
        self._check_tensor_dim(lossf_mask.dim(), 7)

        self._check_tensor_dim(phase_slice.dim(), 7)
        self._check_tensor_dim(phase_slice.dim(), 7)

        self.sens = sens
        self.kspace = kspace
        self.train_mask = train_mask
        self.lossf_mask = lossf_mask
        self.phase_echo = phase_echo
        self.phase_slice = phase_slice

    def __len__(self):
        return len(self.train_mask)

    def __getitem__(self, idx):

        sens_i = self.sens[idx]
        kspace_i = self.kspace[idx]
        train_mask_i = self.train_mask[idx]
        lossf_mask_i = self.lossf_mask[idx]

        phase_echo_i = self.phase_echo[idx]
        phase_slice_i = self.phase_slice[idx]

        return sens_i, kspace_i, train_mask_i, lossf_mask_i, phase_echo_i, phase_slice_i

    def _check_two_shape(self, ref_shape, dst_shape):
        for i1, i2 in zip(ref_shape, dst_shape):
            if (i1 != i2):
                raise ValueError('shape mismatch for ref {ref}, got {dst}'.format(
                    ref=ref_shape, dst=dst_shape))

    def _check_tensor_dim(self, actual_dim: int, expect_dim: int):
        assert actual_dim == expect_dim


# %%
class Trafos(nn.Module):
    def __init__(self, ishape: Tuple[int, ...],
                 contrasts_in_channels: bool = False,
                 verbose: bool = False):
        super(Trafos, self).__init__()

        N_rep, N_diff, N_shot, N_coil, N_z, N_y, N_x = ishape


        R1_oshape = [N_rep] + [N_diff * N_shot * N_coil] + [N_z, N_y, N_x]
        P1_oshape = [R1_oshape[0], R1_oshape[2], R1_oshape[1], R1_oshape[3], R1_oshape[4]]
        D, H, W = R1_oshape[1], R1_oshape[3], R1_oshape[4]

        R2_oshape = [N_rep * N_z, D, H, W]
        P2_oshape = [R2_oshape[0], 2, D, H, W]

        R1 = util.Reshape(tuple(R1_oshape), ishape)
        P1 = util.Permute(tuple(R1_oshape), (0, 2, 1, 3, 4))

        R2 = util.Reshape(tuple(R2_oshape), P1_oshape)

        C2R = util.C2R()

        P2 = util.Permute(tuple(R2_oshape + [2]), (0, 4, 1, 2, 3))

        self.fwd = nn.ModuleList([R1, P1, R2, C2R, P2])

        if contrasts_in_channels is True:
            R3 = util.Reshape(tuple([P2_oshape[0], 2 * D, H, W]), tuple(P2_oshape))
            self.fwd.append(R3)

        self.verbose = verbose

    def forward(self, x: torch.Tensor):
        output = x.clone()

        for ind, module in enumerate(self.fwd):
            if self.verbose:
                print(f"Module {ind + 1}:\n{module}\n")
            output = module(output)

        return output

    def adjoint(self, x: torch.Tensor):
        output = x.clone()

        for ind, module in reversed(list(enumerate(self.fwd))):
            if self.verbose:
                print(f"Module {ind + 1}:\n{module}\n")
            output = module.adjoint(output)

        return output

# %%
class UnrollNet(nn.Module):
    """
    Args:
        TODO: docomentation
    """
    def __init__(self,
                 lamda: float = 0.01,
                 requires_grad_lamda: bool = True,
                 N_unroll: int = 10,
                 NN: str = 'Identity',
                 features: int = 64,
                 max_cg_iter: int = 10):
        super(UnrollNet, self).__init__()

        # neural network part
        if NN == 'ResNet3D':
            self.NN = resnet.ResNet3D(in_channels=2, N_residual_block=5,
                                      features=features)
            print('> Use ResNet3D')
        elif NN == 'ResNet2D':
            self.NN = resnet.ResNet2D(in_channels=2, N_residual_block=5,
                                      features=features)
            print('> Use ResNet2D')
        elif NN == 'Identity':
            self.NN = nn.Identity()

        self.lamda = nn.Parameter(torch.tensor([lamda]), requires_grad=requires_grad_lamda)
        self.N_unroll = N_unroll

        self.max_cg_iter = max_cg_iter

    def forward(self, x: torch.Tensor,
                Train_SENSE_ModuleList: nn.ModuleList,
                Lossf_SENSE_ModuleList: nn.ModuleList):
        """
        Args:
            * ikspace (torch.Tensor): input k-space

        Return:
            * okspace (torch.Tensor): output k-space
        """
        input = x.clone()

        T = Trafos(input.shape)


        for n in range(self.N_unroll):
            input = T.adjoint(self.NN(T(input)))

            input = _solve_SENSE_ModuleList(Train_SENSE_ModuleList, input,
                                            lamda=self.lamda,
                                            max_iter=self.max_cg_iter)

        lossf_kspace = _fwd_SENSE_ModuleList(Lossf_SENSE_ModuleList, input)

        return input, self.lamda, lossf_kspace

# %%
class MixL1L2Loss(nn.Module):
    def __init__(self, eps=1e-6, scalar=1/2):
        super().__init__()
        self.eps = eps
        self.scalar = scalar

    def forward(self, y_est, y_ref):

        y1 = torch.view_as_real(y_est)
        y2 = torch.view_as_real(y_ref)

        scalar = torch.tensor([0.5], dtype=torch.float32).to(y_est.device)

        loss = torch.mul(scalar, torch.linalg.norm(y1 - y2)) / torch.linalg.norm(y2) + torch.mul(scalar, torch.linalg.norm(torch.flatten(y1 - y2), ord=1))  / torch.linalg.norm(torch.flatten(y2),ord=1)

        return loss

class NRMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_est, y_ref):

        # mean = torch.mean(abs(y_ref))
        mean = torch.max(abs(y_ref)) - torch.min(abs(y_ref))
        loss = nn.functional.mse_loss(torch.view_as_real(y_est), torch.view_as_real(y_ref)) / mean

        return loss
