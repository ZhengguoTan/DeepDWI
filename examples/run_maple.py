import argparse
import h5py
import os
import time
import torch
import yaml

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import torch.nn as nn

from deepdwi.dims import *
from deepdwi.recons import zsssl
from torch.utils.data import DataLoader

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]

DAT_DIR = DIR.rsplit('/', 1)[0] + '/data'
print('> data directory: ', DAT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_sp = sp.Device(0 if torch.cuda.is_available() else -1)


# %%
class Dataset(torch.utils.data.Dataset):
    """
    A Dataset for Zero-Shot Self-Supervised Learning.
    """
    def __init__(self,
                 sens: torch.Tensor,
                 kspace: torch.Tensor,
                 train_mask: torch.Tensor,
                 lossf_mask: torch.Tensor):
        r"""
        Initializa a Dataset for zero-shot learning.
        """
        self._check_two_shape(train_mask.shape, lossf_mask.shape)
        self._check_tensor_dim(sens.dim(), 7)
        self._check_tensor_dim(kspace.dim(), 7)
        self._check_tensor_dim(train_mask.dim(), 7)
        self._check_tensor_dim(lossf_mask.dim(), 7)

        self.sens = sens
        self.kspace = kspace
        self.train_mask = train_mask
        self.lossf_mask = lossf_mask

    def __len__(self):
        return len(self.train_mask)

    def __getitem__(self, idx):

        sens_i = self.sens[idx]
        kspace_i = self.kspace[idx]
        train_mask_i = self.train_mask[idx]
        lossf_mask_i = self.lossf_mask[idx]

        return sens_i, kspace_i, train_mask_i, lossf_mask_i

    def _check_two_shape(self, ref_shape, dst_shape):
        for i1, i2 in zip(ref_shape, dst_shape):
            if (i1 != i2):
                raise ValueError('shape mismatch for ref {ref}, got {dst}'.format(
                    ref=ref_shape, dst=dst_shape))

    def _check_tensor_dim(self, actual_dim: int, expect_dim: int):
        assert actual_dim == expect_dim


# %%
def prep_data(data_file: str,
              norm_kdat: bool = False):

    f = h5py.File(HOME_DIR + data_file, 'r')
    coil = f['coil'][:]
    kdat = f['kdat'][:]
    mask = f['mask'][:]
    f.close()

    coil4 = coil[:, None, :, :]


    kdat6 = kdat[..., None, :, :]
    kdat6 = np.reshape(kdat6, [-1] + list(kdat6.shape[2:]))
    kdat6 = kdat6[:, None, ...]

    if norm_kdat:
        kdat6 = kdat6 / np.max(np.abs(kdat6))

    print(' > kdat shape: ', kdat6.shape)


    mask6 = np.reshape(mask, [-1] + list(mask.shape[2:]))
    mask6 = mask6[:, None, None, None, ...]
    mask6 = mask6.astype(kdat6.dtype)
    print(' > mask shape: ', mask6.shape)

    return coil4, kdat6, mask6


# %%
def prep_mask(mask: np.ndarray, N_repeats: int = 12,
              valid_rho: float = 0.2,
              train_rho: float = 0.4):
    mask = torch.from_numpy(mask)
    res_mask, valid_mask = zsssl.uniform_samp(mask, rho=valid_rho, acs_block=(8, 8))
    valid_mask = valid_mask[None, ...]  # 7dim

    train_mask = []
    lossf_mask = []

    for r in range(N_repeats):

        train_mask1, lossf_mask1 = zsssl.uniform_samp(res_mask, rho=train_rho, acs_block=(8, 8))

        train_mask.append(train_mask1)
        lossf_mask.append(lossf_mask1)

    train_mask = torch.stack(train_mask)
    lossf_mask = torch.stack(lossf_mask)

    f = h5py.File(DIR + '/mask.h5', 'w')
    f.create_dataset('train', data=train_mask.detach().cpu().numpy())
    f.create_dataset('lossf', data=lossf_mask.detach().cpu().numpy())
    f.create_dataset('valid', data=valid_mask.detach().cpu().numpy())
    f.close()

    return mask, train_mask, lossf_mask, valid_mask


# %%
def repeat_data(coil4: np.ndarray,
                kdat6: np.ndarray,
                N_repeats: int = 12):

    coil7 = torch.from_numpy(coil4)
    coil7 = coil7[None, None, None, ...]
    coil7 = torch.tile(coil7, tuple([N_repeats] + [1] * (coil7.dim()-1)))

    kdat7 = torch.from_numpy(kdat6)
    kdat7 = kdat7[None, ...]
    kdat7 = torch.tile(kdat7, tuple([N_repeats] + [1] * (kdat7.dim()-1)))

    return coil7, kdat7


# %%
if __name__ == "__main__":

    # %% argument parser
    parser = argparse.ArgumentParser(description='run zsssl.')

    parser.add_argument('--config', type=str,
                        default='/configs/maple.yaml',
                        help='yaml config file for zsssl')

    args = parser.parse_args()


    # %% read in and display the yaml config file
    with open(HOME_DIR + args.config, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # print('> yaml config: ', config_dict)
    data_conf = config_dict.get('data', {})
    print('> data_conf: ')
    print('    data: ', data_conf['data'])
    print('    normalize_kdat: ', data_conf['normalize_kdat'])
    print('    valid_rho: ', data_conf['valid_rho'])
    print('    train_rho: ', data_conf['train_rho'])
    print('    repeats: ', data_conf['repeats'])
    print('    data_size: ', data_conf['batch_size'])

    model_conf = config_dict.get('model', {})
    print('> model_conf: ')
    print('    net: ', model_conf['net'])
    print('    requires_grad_lamda: ', model_conf['requires_grad_lamda'])
    print('    N_residual_block: ', model_conf['N_residual_block'])
    print('    N_unroll: ', model_conf['N_unroll'])
    print('    features: ', model_conf['features'])
    print('    contrasts_in_channels', model_conf['contrasts_in_channels'])
    print('    max_cg_iter: ', model_conf['max_cg_iter'])
    print('    lamda: ', model_conf['lamda'])

    optim_conf = config_dict.get('optim', {})
    print('> optim_conf: ')
    print('    method: ', optim_conf['method'])
    print('    lr: ', optim_conf['lr'])

    loss_conf = config_dict['loss']
    print('> loss: ', loss_conf)

    learn_conf = config_dict['learn']
    print('> learn: ')
    print('    epochs: ', learn_conf['epochs'])
    print('    valid_loss_tracker: ', learn_conf['valid_loss_tracker'])

    # %%
    if data_conf['use_prep_data'] is True:

        f = h5py.File(HOME_DIR + data_conf['data'], 'r')
        kdat6 = f['ksp6'][:]
        coil4 = f['mps4'][:]
        train_mask = torch.from_numpy(f['train_mask7'][:])
        lossf_mask = torch.from_numpy(f['lossf_mask7'][:])
        valid_mask = torch.from_numpy(f['valid_mask7'][:])
        f.close()

        coil7, kdat7 = repeat_data(coil4, kdat6, N_repeats=data_conf['repeats'])

        mask = train_mask[0] + lossf_mask[0] + valid_mask[0]

    else:
        coil4, kdat6, mask = prep_data(data_conf['data'],
                                    norm_kdat=data_conf['normalize_kdat'])

        mask, train_mask, lossf_mask, valid_mask = prep_mask(mask, N_repeats=data_conf['repeats'],
                                                            valid_rho=data_conf['valid_rho'],
                                                            train_rho=data_conf['train_rho'])

        coil7, kdat7 = repeat_data(coil4, kdat6, N_repeats=data_conf['repeats'])

    print('>>> coil7 shape\t: ', coil7.shape, ' type: ', coil7.dtype)
    print('>>> kdat7 shape\t: ', kdat7.shape, ' type: ', kdat7.dtype)

    print('>>> train_mask shape\t: ', train_mask.shape, ' type: ', train_mask.dtype)
    print('>>> lossf_mask shape\t: ', lossf_mask.shape, ' type: ', lossf_mask.dtype)
    print('>>> valid_mask shape\t: ', valid_mask.shape, ' type: ', valid_mask.dtype)

    S = zsssl._build_SENSE_ModuleList(coil7[[0]], mask * kdat7[[0]])

    ishape = [data_conf['batch_size']] + list(S[0].ishape)
    print('>>> ishape to UnrollNet: ', ishape)
    del S

    # %% train and valid
    train_data = Dataset(coil7, kdat7, train_mask, lossf_mask)
    train_load = DataLoader(train_data, batch_size=data_conf['batch_size'])

    res_mask = train_mask + lossf_mask

    valid_data = Dataset(coil7[[0]], kdat7[[0]], res_mask[[0]], valid_mask)
    valid_load = DataLoader(valid_data, batch_size=data_conf['batch_size'])

    if model_conf['net'] == 'ResNet2D' and model_conf['contrasts_in_channels'] is False:
        assert kdat7.shape[DIM_TIME] == 1 and kdat7.shape[DIM_ECHO] == 1

    model = zsssl.UnrollNet(ishape, lamda=model_conf['lamda'], NN=model_conf['net'],
                            requires_grad_lamda=model_conf['requires_grad_lamda'],
                            N_residual_block=model_conf['N_residual_block'],
                            N_unroll=model_conf['N_unroll'],
                            features=model_conf['features'],
                            contrasts_in_channels=model_conf['contrasts_in_channels'],
                            max_cg_iter=model_conf['max_cg_iter']).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(">>> number of trainable parameters is: ", params)

    if loss_conf == 'MixL1L2Loss':
        lossf = zsssl.MixL1L2Loss()
    elif loss_conf == 'MSELoss':
        lossf = nn.MSELoss()
    elif loss_conf == 'NRMSELoss':
        lossf = zsssl.NRMSELoss()

    if optim_conf['method'] == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=optim_conf['lr'])
    else:
        optim = torch.optim.SGD(model.parameters(), lr=optim_conf['lr'])

    # %% train and valid
    total_train_loss, total_valid_loss = [], []
    valid_loss_min = np.inf

    epoch, valid_loss_tracker = 0, 0

    start_time = time.time()

    while epoch < learn_conf['epochs'] and valid_loss_tracker < learn_conf['valid_loss_tracker']:

        tic = time.time()

        train_loss_sum = 0

        # --- train ---
        for ii, (sens, kspace, train_mask, lossf_mask) in enumerate(train_load):

            sens = sens.to(device)
            kspace = kspace.to(device)
            train_mask = train_mask.to(device)
            lossf_mask = lossf_mask.to(device)

            # apply Model
            batch_x, lamda, ynet, yref = model(sens, kspace, train_mask, lossf_mask)

            # loss
            train_loss = lossf(ynet, yref)
            train_loss_sum += train_loss

            # back propagation
            optim.zero_grad()
            train_loss.backward()
            optim.step()

        # --- valid ---
        with torch.no_grad():
            for ii, (sens, kspace, train_mask, lossf_mask) in enumerate(valid_load):

                # to device
                sens = sens.to(device)
                kspace = kspace.to(device)
                train_mask = train_mask.to(device)
                lossf_mask = lossf_mask.to(device)

                # apply Model
                _, lamda, ynet, yref = model(sens, kspace, train_mask, lossf_mask)

                # loss
                valid_loss = lossf(ynet, yref)


        #save the best checkpoint
        checkpoint = {
            "epoch": epoch,
            "valid_loss_min": valid_loss,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict()
        }

        if valid_loss <= valid_loss_min:
            valid_loss_min = valid_loss
            torch.save(checkpoint, os.path.join(DIR, "maple_best.pth"))
            # reset the val loss tracker each time a new lowest val error is achieved
            valid_loss_tracker = 0
        else:
            valid_loss_tracker += 1

        toc = time.time() - tic
        if epoch % 1 == 0:
            print("Epoch:", str(epoch+1).zfill(3), ", elapsed_time = ""{:7.3f}".format(toc),
                  ", train loss = ", "{:18.12f}".format(train_loss.item()),
                  ", valid loss = ", "{:18.12f}".format(valid_loss.item()),
                  ", lamda = ", "{:9.6f}".format(lamda.item()))

        epoch += 1

    end_time = time.time()
    print('Training completed in  ', str(epoch), ' epochs, ',((end_time - start_time) / 60), ' minutes')

    # %% inference
    infer_data = Dataset(coil7[[0]], kdat7[[0]], mask[np.newaxis], mask[np.newaxis])
    infer_load = DataLoader(infer_data, batch_size=1, shuffle=True, num_workers=6)

    best_checkpoint = torch.load(os.path.join(DIR, 'best.pth'))
    model.load_state_dict(best_checkpoint["model_state"])

    # --- valid ---
    x_infer = []
    with torch.no_grad():
        for ii, (sens, kspace, train_mask, lossf_mask) in enumerate(infer_load):

            # to device
            sens = sens.to(device)
            kspace = kspace.to(device)
            train_mask = train_mask.to(device)
            lossf_mask = lossf_mask.to(device)

            x, _, _, _  = model(sens, kspace, train_mask, lossf_mask)
            x_infer.append(x.detach().cpu().numpy())

    x_infer = np.array(x_infer)

    f = h5py.File(DIR + '/maple.h5', 'w')
    f.create_dataset('ZS', data=x_infer)
    f.close()
