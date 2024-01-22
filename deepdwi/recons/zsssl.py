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

    C_y, C_x = N_y // 2, N_x // 2

    # mask_outer that excludes the ACS region
    outer_mask = mask.clone()
    outer_mask[..., acs_block[-2] - C_y : acs_block[-2] + C_y,
               acs_block[-1] - C_x : acs_block[-1] + C_x] = 0

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
                            xinit: torch.Tensor = None,
                            lamda: float = 0.01,
                            max_iter: int = 100,
                            tol: float = 0.) -> torch.Tensor:
    x = []
    for l in range(len(SENSE_ModuleList)):
        A = SENSE_ModuleList[l]

        # normal equation with Tikhonov regu
        AHA = lambda x: A.adjoint(A.forward(x)) + lamda * x
        # adjoint
        AHy = A.adjoint(A.y)

        if xinit is None:
            x0 = torch.zeros_like(AHy)
        else:
            x0 = xinit[l]

        CG = lsqr.ConjugateGradient(AHA, AHy, x0,
                                    max_iter=max_iter,
                                    tol=tol)

        x.append(CG())  # run CG

    return torch.stack(x)


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
    def __init__(self, ishape: Tuple[int, ...]):
        super(Trafos, self).__init__()

        print('>>> Trafos ishape: ', ishape)

        N_rep, N_diff, N_shot, N_coil, N_z, N_y, N_x, N_channel = ishape

        self.P1 = util.Permute(ishape, (0, 4, 1, 2, 3, 5, 6, 7))
        self.R1 = util.Reshape(tuple([N_rep * N_z] + [N_diff * N_shot * N_coil] + [N_y, N_x, N_channel]), self.P1.oshape)
        self.P2 = util.Permute(self.R1.oshape, (0, 4, 1, 2, 3))

    def forward(self, x: torch.Tensor):
        y = torch.view_as_real(x)
        N_rep, N_diff, N_shot, N_coil, N_z, N_y, N_x, N_channel = y.shape

        output = self.P2(self.R1(self.P1(y)))
        output = output.squeeze(2)

        return output

    def adjoint(self, x: torch.Tensor):
        if x.dim() == 4:
            output = x.unsqueeze(2)
        elif x.dim() == 5:
            output = x.clone()

        output = self.P1.adjoint(self.R1.adjoint(self.P2.adjoint(output)))
        output = torch.view_as_complex(output)

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
                 N_unroll: int = 10):
        super(UnrollNet, self).__init__()

        # neural network part
        self.NN = resnet.ResNet3D(in_channels=2, N_residual_block=5)

        self.lamda = nn.Parameter(torch.tensor([lamda]), requires_grad=requires_grad_lamda)
        self.N_unroll = N_unroll

    def forward(self, x: torch.Tensor,
                Train_SENSE_ModuleList: nn.ModuleList,
                Lossf_SENSE_ModuleList: nn.ModuleList):
        """
        Args:
            * ikspace (torch.Tensor): input k-space

        Return:
            * okspace (torch.Tensor): output k-space
        """
        for n in range(self.N_unroll):
            rhs = x + self.lamda * x
            x = _solve_SENSE_ModuleList(Train_SENSE_ModuleList, rhs, self.lamda)

        lossf_kspace = _fwd_SENSE_ModuleList(Lossf_SENSE_ModuleList, x)

        return x, self.lamda, lossf_kspace

# %%
class MixL1L2Loss(nn.Module):
    def __init__(self, eps=1e-6, scalar=1/2):
        super().__init__()
        self.eps = eps
        self.scalar = scalar

    def forward(self, yhat, y):

        loss = self.scalar*(torch.norm(yhat-y) / torch.norm(y)) +\
                self.scalar*(torch.norm(yhat-y, p=1) / torch.norm(y, p=1))

        return loss

# %%
def train(Model, DataLoader, lossf, optim,
          device=torch.device('cpu')):
    train_lossv = 0
    Model.train()
    for ii, (sens, kspace, train_mask, lossf_mask, phase_shot, phase_slice) in enumerate(DataLoader):

        train_kspace = train_mask * kspace
        Train_SENSE_ModuleList = _build_SENSE_ModuleList(sens,
                                                         train_kspace,
                                                         phase_shot,
                                                         phase_slice)

        x = _adj_SENSE_ModuleList(Train_SENSE_ModuleList)

        lossf_kspace = lossf_mask * kspace
        Lossf_SENSE_ModuleList = _build_SENSE_ModuleList(sens,
                                                         lossf_kspace,
                                                         phase_shot,
                                                         phase_slice)

        # apply Model
        ynet = Model(x, Train_SENSE_ModuleList, Lossf_SENSE_ModuleList)

        # loss
        lossv = lossf(ynet, lossf_kspace)

        # back propagation
        optim.zero_grad()
        lossv.backward()
        optim.step()

        train_lossv += lossv.item()/ len(DataLoader)

    return train_lossv

# %%
def valid(Model, DataLoader, lossf, optim,
          phase_echo: torch.Tensor = None,
          phase_slice: torch.Tensor = None,
          device=torch.device('cpu')):
    valid_lossv = 0
    Model.valid()
    with torch.no_grad():
        for ii, (sens, kspace, train_mask, lossf_mask) in enumerate(DataLoader):

            # to device

            train_kspace = train_mask * kspace
            Train_SENSE_ModuleList = _build_SENSE_ModuleList(sens.to(device),
                                                            train_kspace.to(device),
                                                            phase_echo.to(device),
                                                            phase_slice.to(device))

            x = _adj_SENSE_ModuleList(Train_SENSE_ModuleList)

            lossf_kspace = lossf_mask * kspace
            Lossf_SENSE_ModuleList = _build_SENSE_ModuleList(sens.to(device),
                                                            lossf_kspace.to(device),
                                                            phase_echo.to(device),
                                                            phase_slice.to(device))

            # apply Model
            ynet = Model(x, Train_SENSE_ModuleList, Lossf_SENSE_ModuleList)

            # loss
            lossv = lossf(ynet, lossf_kspace)

            valid_lossv += lossv.item()/ len(DataLoader)

    return valid_lossv

# %%
def test():
    None
