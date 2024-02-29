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
        self.ishape = ishape


        R1_oshape = [N_rep] + [N_diff * N_shot * N_coil] + [N_z, N_y, N_x]
        P1_oshape = [R1_oshape[0], R1_oshape[2], R1_oshape[1], R1_oshape[4], R1_oshape[3]]
        D, H, W = P1_oshape[-3], P1_oshape[-2], P1_oshape[-1]

        R2_oshape = [N_rep * N_z, D, H, W]
        P2_oshape = [R2_oshape[0], 2, D, H, W]

        R1 = util.Reshape(tuple(R1_oshape), ishape)
        P1 = util.Permute(tuple(R1_oshape), (0, 2, 1, 4, 3))

        R2 = util.Reshape(tuple(R2_oshape), P1_oshape)

        C2R = util.C2R()

        P2 = util.Permute(tuple(R2_oshape + [2]), (0, 4, 1, 2, 3))

        self.fwd = nn.ModuleList([R1, P1, R2, C2R, P2])

        if contrasts_in_channels is True:
            P3 = util.Permute(tuple(P2_oshape), (0, 2, 1, 3, 4))
            self.fwd.append(P3)
            R3 = util.Reshape(tuple([P2_oshape[0], 2 * D, H, W]), P3.oshape)
            self.fwd.append(R3)

            self.oshape = R3.oshape
        else:
            self.oshape = P2.oshape

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
def conj_grad(AHA, AHy, max_iter: int = 6, tol: float = 0.):

    x = torch.zeros_like(AHy)
    i, r, p = 0, AHy, AHy
    rTr = torch.sum(r.conj()*r).real
    while i < max_iter and rTr > 1e-10:
        Ap = AHA(p)
        alpha = rTr / torch.sum(p.conj()*Ap).real
        alpha = alpha
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(r.conj()*r).real
        beta = rTrNew / rTr
        beta = beta
        p = r + beta * p
        i += 1
        rTr = rTrNew
    return x

# %%
class UnrollNet(nn.Module):
    """
    Args:
        TODO: docomentation
    """
    def __init__(self,
                 ishape: Tuple[int, ...],
                 lamda: float = 0.01,
                 requires_grad_lamda: bool = True,
                 N_residual_block: int = 5,
                 N_unroll: int = 10,
                 NN: str = 'Identity',
                 features: int = 64,
                 max_cg_iter: int = 10,
                 contrasts_in_channels: bool = False,
                 use_batch_norm: bool = False):
        super(UnrollNet, self).__init__()

        if NN == 'ResNet3D':
            contrasts_in_channels = False
            print('> set contrasts_in_channels to False!')

        self.T = Trafos(ishape,
                        contrasts_in_channels=contrasts_in_channels)

        print('> Trafos oshape: ', self.T.oshape)

        # neural network part
        self.NN = NN
        self.features = features

        if self.NN == 'ResNet3D':
            self.NN_Module = resnet.ResNet3D(in_channels=self.T.oshape[1],
                                             N_residual_block=N_residual_block,
                                             features=self.features)
            print('> Use ResNet3D')
        elif self.NN == 'ResNet2D':
            self.NN_Module = resnet.ResNet2D(in_channels=self.T.oshape[1],
                                             N_residual_block=N_residual_block,
                                             features=self.features,
                                             use_batch_norm=use_batch_norm)
            print('> Use ResNet2D')
        elif self.NN == 'ResNetMAPLE':
            self.NN_Module = resnet.ResNetMAPLE(in_channels=self.T.oshape[1],
                                                N_residual_block=N_residual_block,
                                                features=self.features)
            print('> Use ResNetMAPLE')

        self.lamda = nn.Parameter(torch.tensor([lamda]), requires_grad=requires_grad_lamda)
        self.N_unroll = N_unroll

        self.max_cg_iter = max_cg_iter
        self.contrasts_in_channels = contrasts_in_channels

    def forward(self,
                sens: torch.Tensor,
                kspace: torch.Tensor,
                train_mask: torch.Tensor,
                lossf_mask: torch.Tensor,
                phase_echo: torch.Tensor = None,
                phase_slice: torch.Tensor = None):
        """
        Args:
            * ikspace (torch.Tensor): input k-space

        Return:
            * okspace (torch.Tensor): output k-space
        """

        train_kspace = train_mask * kspace
        SenseT = mri.Sense(sens, train_kspace,
                           phase_echo=phase_echo, combine_echo=True,
                           phase_slice=phase_slice)

        x = SenseT.adjoint(SenseT.y)  # x0: AHy

        refer_kspace = lossf_mask * kspace
        SenseL = mri.Sense(sens, refer_kspace,
                           phase_echo=phase_echo, combine_echo=True,
                           phase_slice=phase_slice)

        for n in range(self.N_unroll):
            x = self.T(x)
            x = x.float()
            x = self.T.adjoint(self.NN_Module(x))

            AHA = lambda x: SenseT.adjoint(SenseT(x)) + self.lamda * x
            AHy = SenseT.adjoint(SenseT.y) + self.lamda * x

            # CG = lsqr.ConjugateGradient(AHA, AHy, torch.zeros_like(AHy),
            #                             max_iter=self.max_cg_iter,
            #                             tol=0., verbose=False)

            # x = CG()
            x = conj_grad(AHA, AHy, max_iter=self.max_cg_iter)

        lossf_kspace = SenseL(x)

        return x, self.lamda, lossf_kspace, refer_kspace

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
