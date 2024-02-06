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
from deepdwi.models import mri
from deepdwi.recons import zsssl
from sigpy.mri import app, muse, retro, sms
from torch.utils.data import DataLoader

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]

DAT_DIR = DIR.rsplit('/', 1)[0] + '/data'
print('> data directory: ', DAT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_sp = sp.Device(0 if torch.cuda.is_available() else -1)

torch.manual_seed(0)


# %%
def prep_data(data_file: str,
              coil_file: str,
              slice_idx: int = 0,
              norm_kdat: bool = False):

    f = h5py.File(HOME_DIR + data_file, 'r')
    kdat = f['kdat'][:]
    MB = f['MB'][()]
    N_slices = f['Slices'][()]
    N_segments = f['Segments'][()]
    N_Accel_PE = f['Accel_PE'][()]
    f.close()

    kdat = np.squeeze(kdat)  # 4 dim
    kdat = np.swapaxes(kdat, -2, -3)

    # # split kdat into shots
    N_diff = kdat.shape[-4]
    kdat_prep = []
    for d in range(N_diff):
        k = retro.split_shots(kdat[d, ...], shots=N_segments)
        kdat_prep.append(k)

    kdat_prep = np.array(kdat_prep)
    kdat_prep = kdat_prep[..., None, :, :]  # 6 dim

    if norm_kdat:
        kdat_prep = kdat_prep / np.max(np.abs(kdat_prep[:]))

    N_diff, N_shot, N_coil, _, N_y, N_x = kdat_prep.shape

    print(' > kdat shape: ', kdat_prep.shape)

    # sampling mask
    mask = app._estimate_weights(kdat_prep, None, None, coil_dim=-4)
    mask = abs(mask).astype(float)

    print(' > mask shape: ', mask.shape)

    # coil
    f = h5py.File(HOME_DIR + coil_file, 'r')
    coil = f['coil'][:]
    f.close()

    print(' > coil shape: ', coil.shape)

    N_coil, N_z, N_y, N_x = coil.shape

    # %%
    yshift = []
    for b in range(MB):
        yshift.append(b / N_Accel_PE)

    sms_phase = sms.get_sms_phase_shift([MB, N_y, N_x], MB=MB, yshift=yshift)

    # %%
    slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(slice_idx, N_slices, MB)

    coil2 = coil[:, slice_mb_idx, :, :]
    print('> coil2 shape: ', coil2.shape)

    # %%
    import torchvision.transforms as T

    acs_shape = list([N_y // 4, N_x // 4])
    ksp_acs = sp.resize(kdat_prep, oshape=list(kdat_prep.shape[:-2]) + acs_shape)

    coils_tensor = sp.to_pytorch(coil2)
    TR = T.Resize(acs_shape, antialias=True)
    mps_acs_r = TR(coils_tensor[..., 0]).cpu().detach().numpy()
    mps_acs_i = TR(coils_tensor[..., 1]).cpu().detach().numpy()
    mps_acs = mps_acs_r + 1j * mps_acs_i

    _, dwi_shot = muse.MuseRecon(ksp_acs, mps_acs,
                                MB=MB,
                                acs_shape=acs_shape,
                                lamda=0.01, max_iter=30,
                                yshift=yshift,
                                device=device_sp)

    _, dwi_shot_phase = muse._denoising(dwi_shot, full_img_shape=[N_y, N_x])

    return coil2, kdat_prep, dwi_shot_phase, sms_phase, mask


# %%
def prep_mask(mask: np.ndarray, N_repeats: int = 12,
              valid_rho: float = 0.2,
              train_rho: float = 0.4):
    mask = torch.from_numpy(mask)
    res_mask, valid_mask = zsssl.uniform_samp(mask, rho=valid_rho)
    valid_mask = valid_mask[None, ...]  # 7dim

    train_mask = []
    lossf_mask = []

    for r in range(N_repeats):

        train_mask1, lossf_mask1 = zsssl.uniform_samp(res_mask, rho=train_rho)

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
                phase_shot: np.ndarray,
                phase_slice: np.ndarray,
                N_repeats: int = 12):

    coil7 = torch.from_numpy(coil4)
    coil7 = coil7[None, None, None, ...]
    coil7 = torch.tile(coil7, tuple([N_repeats] + [1] * (coil7.dim()-1)))

    kdat7 = torch.from_numpy(kdat6)
    kdat7 = kdat7[None, ...]
    kdat7 = torch.tile(kdat7, tuple([N_repeats] + [1] * (kdat7.dim()-1)))

    phase_shot7 = torch.from_numpy(phase_shot)
    phase_shot7 = phase_shot7[None, ...]
    phase_shot7 = torch.tile(phase_shot7, tuple([N_repeats] + [1] * (phase_shot7.dim()-1)))

    phase_slice7 = torch.from_numpy(phase_slice)
    phase_slice7 = phase_slice7[None, None, None, None, ...]
    phase_slice7 = torch.tile(phase_slice7, tuple([N_repeats] + [1] * (phase_slice7.dim()-1)))

    return coil7, kdat7, phase_shot7, phase_slice7


# %%
if __name__ == "__main__":

    # %% argument parser
    parser = argparse.ArgumentParser(description='run zsssl.')

    parser.add_argument('--config', type=str,
                        default='/configs/zsssl.yaml',
                        help='yaml config file for zsssl')

    parser.add_argument('--diff_idx', type=int, default=-1,
                        help='reconstruct only one diffuion encoding')

    args = parser.parse_args()


    # %% read in and display the yaml config file
    with open(HOME_DIR + args.config, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # print('> yaml config: ', config_dict)
    data_conf = config_dict.get('data', {})
    print('> data_conf: ')
    print('    kdat: ', data_conf['kdat'])
    print('    slice_idx: ', data_conf['slice_idx'])
    print('    coil: ', data_conf['coil'])
    print('    normalize_kdat: ', data_conf['normalize_kdat'])
    print('    valid_rho: ', data_conf['valid_rho'])
    print('    train_rho: ', data_conf['train_rho'])
    print('    repeats: ', data_conf['repeats'])
    print('    data_size: ', data_conf['batch_size'])

    model_conf = config_dict.get('model', {})
    print('> model_conf: ')
    print('    net: ', model_conf['net'])
    print('    requires_grad_lamda: ', model_conf['requires_grad_lamda'])
    print('    N_unroll: ', model_conf['N_unroll'])
    print('    features: ', model_conf['features'])
    print('    contrasts_in_channels', model_conf['contrasts_in_channels'])

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

    mode_conf = config_dict['mode']
    print('> mode: ', mode_conf)

    # %%
    coil4, kdat6, phase_shot, phase_slice, mask = prep_data(data_conf['kdat'], data_conf['coil'],
                                                            slice_idx=data_conf['slice_idx'],
                                                            norm_kdat=data_conf['normalize_kdat'])

    mask, train_mask, lossf_mask, valid_mask = prep_mask(mask, N_repeats=data_conf['repeats'],
                                                         valid_rho=data_conf['valid_rho'],
                                                         train_rho=data_conf['train_rho'])

    coil7, kdat7, phase_shot7, phase_slice7 = repeat_data(coil4, kdat6, phase_shot, phase_slice,
                                                          N_repeats=data_conf['repeats'])

    # run only one DWI direction and the first 10 coils
    if args.diff_idx >=0 and args.diff_idx < kdat7.shape[DIM_TIME]:
        print('> recon only one diffusion encoding')
        kdat7 = kdat7[:, [args.diff_idx], ...]
        phase_shot7 = phase_shot7[:, [args.diff_idx], ...]
        mask = mask[[args.diff_idx], ...]
        train_mask = train_mask[:, [args.diff_idx], ...]
        lossf_mask = lossf_mask[:, [args.diff_idx], ...]
        valid_mask = valid_mask[:, [args.diff_idx], ...]

    print('>>> coil7 shape\t: ', coil7.shape, ' type: ', coil7.dtype)
    print('>>> kdat7 shape\t: ', kdat7.shape, ' type: ', kdat7.dtype)
    print('>>> phase_shot7 shape\t: ', phase_shot7.shape, ' type: ', phase_shot7.dtype)
    print('>>> phase_slice7 shape\t: ', phase_slice7.shape, ' type: ', phase_slice7.dtype)

    print('>>> train_mask shape\t: ', train_mask.shape, ' type: ', train_mask.dtype)
    print('>>> lossf_mask shape\t: ', lossf_mask.shape, ' type: ', lossf_mask.dtype)
    print('>>> valid_mask shape\t: ', valid_mask.shape, ' type: ', valid_mask.dtype)

    S = mri.Sense(coil7[0], kdat7[0], phase_slice=phase_slice7[0],
                  phase_echo=phase_shot7[0], combine_echo=True)
    ishape = [data_conf['batch_size']] + list(S.ishape)
    print('>>> ishape to UnrollNet: ', ishape)
    del S

    # %% train and valid
    train_data = zsssl.Dataset(coil7, kdat7, train_mask, lossf_mask, phase_shot7, phase_slice7)
    train_load = DataLoader(train_data, batch_size=data_conf['batch_size'])

    res_mask = train_mask + lossf_mask

    valid_data = zsssl.Dataset(coil7[[0]], kdat7[[0]], res_mask[[0]], valid_mask, phase_shot7[[0]], phase_slice7[[0]])
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
    if mode_conf != 'test':
        valid_loss_min = np.inf

        epoch, valid_loss_tracker = 0, 0

        start_time = time.time()

        while epoch < learn_conf['epochs'] and valid_loss_tracker < learn_conf['valid_loss_tracker']:

            tic = time.time()

            train_loss_sum = 0

            epoch_x = []

            # --- train ---
            for ii, (sens, kspace, train_mask, lossf_mask, phase_echo, phase_slice) in enumerate(train_load):

                sens = sens.to(device)
                kspace = kspace.to(device)
                train_mask = train_mask.to(device)
                lossf_mask = lossf_mask.to(device)
                phase_echo = phase_echo.to(device)
                phase_slice = phase_slice.to(device)

                # apply Model
                batch_x, lamda, ynet, yref = model(sens, kspace, train_mask, lossf_mask,
                                                phase_echo, phase_slice)

                epoch_x.append(batch_x)

                # loss
                train_loss = lossf(ynet, yref)
                train_loss_sum += train_loss

                # back propagation
                optim.zero_grad()
                train_loss.backward()
                optim.step()

            epoch_x = torch.stack(epoch_x)
            f = h5py.File(DIR + '/zsssl_epoch_' + str(epoch).zfill(3) + '.h5', 'w')
            f.create_dataset('DWI', data=epoch_x.detach().cpu().numpy())
            f.close()

            # --- valid ---
            with torch.no_grad():
                for ii, (sens, kspace, train_mask, lossf_mask, phase_echo, phase_slice) in enumerate(valid_load):

                    # to device
                    sens = sens.to(device)
                    kspace = kspace.to(device)
                    train_mask = train_mask.to(device)
                    lossf_mask = lossf_mask.to(device)
                    phase_echo = phase_echo.to(device)
                    phase_slice = phase_slice.to(device)

                    # apply Model
                    _, lamda, ynet, yref = model(sens, kspace, train_mask, lossf_mask,
                                                phase_echo, phase_slice)

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
                torch.save(checkpoint, os.path.join(DIR, "zsssl_best.pth"))
                # reset the val loss tracker each time a new lowest val error is achieved
                valid_loss_tracker = 0
            else:
                valid_loss_tracker += 1

            toc = time.time() - tic
            if epoch % 1 == 0:
                print("Epoch:", str(epoch+1).zfill(3), ", elapsed_time = ""{:7.3f}".format(toc),
                    ", train loss = ", "{:12.6f}".format(train_loss.item()),
                    ", valid loss = ", "{:12.6f}".format(valid_loss.item()),
                    ", lamda = ", "{:12.6f}".format(lamda.item()))

            epoch += 1

        end_time = time.time()
        print('Training completed in  ', str(epoch), ' epochs, ',((end_time - start_time) / 60), ' minutes')

    # %% inference
    infer_data = zsssl.Dataset(coil7[[0]], kdat7[[0]], mask[np.newaxis], mask[np.newaxis], phase_shot7[[0]], phase_slice7[[0]])
    infer_load = DataLoader(infer_data, batch_size=1)

    if mode_conf != 'test':
        best_checkpoint = torch.load(os.path.join(DIR, 'zsssl_best.pth'))
    else:
        best_checkpoint = torch.load(HOME_DIR + config_dict['checkpoint'])

    model.load_state_dict(best_checkpoint["model_state"])

    # --- valid ---
    x_infer = []
    with torch.no_grad():
        for ii, (sens, kspace, train_mask, lossf_mask, phase_echo, phase_slice) in enumerate(infer_load):

            # to device
            sens = sens.to(device)
            kspace = kspace.to(device)
            train_mask = train_mask.to(device)
            lossf_mask = lossf_mask.to(device)
            phase_echo = phase_echo.to(device)
            phase_slice = phase_slice.to(device)


            x, _, _, _  = model(sens, kspace, train_mask, lossf_mask, phase_echo, phase_slice)
            x_infer.append(x.detach().cpu().numpy())

    x_infer = np.array(x_infer)

    f = h5py.File(DIR + '/zsssl_slice_' + str(data_conf['slice_idx']).zfill(3) + '.h5', 'w')
    f.create_dataset('ZS', data=x_infer)
    f.close()
