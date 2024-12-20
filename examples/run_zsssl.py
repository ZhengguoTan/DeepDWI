import argparse
import h5py
import os
import pathlib
import time
import torch
import yaml

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

from deepdwi import util, prep
from deepdwi.dims import *
from deepdwi.models import mri
from deepdwi.recons import zsssl
from torch.utils.data import DataLoader

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

DATA_DIR = DIR.rsplit('/', 1)[0] + '/data'
print('> data directory: ', DATA_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torch.manual_seed(0)

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

    if phase_shot is not None:
        phase_shot7 = torch.from_numpy(phase_shot)
        phase_shot7 = phase_shot7[None, ...]
        phase_shot7 = torch.tile(phase_shot7, tuple([N_repeats] + [1] * (phase_shot7.dim()-1)))
    else:
        phase_shot7 = None

    if phase_slice is not None:
        phase_slice7 = torch.from_numpy(phase_slice)
        phase_slice7 = phase_slice7[None, None, None, None, ...]
        phase_slice7 = torch.tile(phase_slice7, tuple([N_repeats] + [1] * (phase_slice7.dim()-1)))
    else:
        phase_slice7 = None

    return coil7, kdat7, phase_shot7, phase_slice7


# %%
if __name__ == "__main__":

    # %% argument parser
    # you can display help messages using `python run_zsssl.py -h`
    parser = argparse.ArgumentParser(description='run zsssl.')

    parser.add_argument('--config', type=str,
                        default='/configs/zsssl.yaml',
                        help='yaml config file for zsssl.')

    parser.add_argument('--slice_idx', type=int, default=-1,
                        help='which slice to train/test.')

    parser.add_argument('--mode', type=str, default='train',
                        choices=('train', 'test'),
                        help="perform 'train' or 'test' of the zsssl model.")

    parser.add_argument('--N_shot_retro', type=int,
                        default=0,
                        help='retro. undersample the number of shots')

    parser.add_argument('--checkpoint', type=str, default='',
                        help="specify the checkpoint to be used in testing.")

    args = parser.parse_args()


    # %% read in and display the yaml config file
    with open(HOME_DIR + args.config, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)


    data_conf = config_dict.get('data', {})
    print('> data_conf: ')
    print('    kdat: ', data_conf['kdat'])
    print('    navi: ', data_conf['navi'])
    print('    slice_idx: ', data_conf['slice_idx'])
    print('    coil: ', data_conf['coil'])
    print('    normalize_kdat: ', data_conf['normalize_kdat'])
    print('    valid_rho: ', data_conf['valid_rho'])
    print('    train_rho: ', data_conf['train_rho'])
    print('    repeats: ', data_conf['repeats'])
    print('    batch_size: ', data_conf['batch_size'])
    print('    N_shot_retro: ', args.N_shot_retro)
    print('    N_diff_split: ', data_conf['N_diff_split'])
    print('    N_diff_split_index: ', data_conf['N_diff_split_index'])

    model_conf = config_dict.get('model', {})
    print('> model_conf: ')
    print('    net: ', model_conf['net'])
    print('    N_residual_block: ', model_conf['N_residual_block'])
    print('    unrolled_algorithm: ', model_conf['unrolled_algorithm'])
    print('    ADMM_rho: ', model_conf['ADMM_rho'])
    print('    N_unroll: ', model_conf['N_unroll'])
    print('    kernel_size: ', model_conf['kernel_size'])
    print('    features: ', model_conf['features'])
    print('    contrasts_in_channels: ', model_conf['contrasts_in_channels'])
    print('    requires_grad_lamda: ', model_conf['requires_grad_lamda'])
    print('    batch_norm: ', model_conf['batch_norm'])

    optim_conf = config_dict.get('optim', {})
    print('> optim_conf: ')
    print('    method: ', optim_conf['method'])
    print('    lr: ', optim_conf['lr'])
    print('    step_size: ', optim_conf['step_size'])
    print('    gamma: ', optim_conf['gamma'])

    loss_conf = config_dict['loss']
    print('> loss: ', loss_conf)

    learn_conf = config_dict['learn']
    print('> learn: ')
    print('    epochs: ', learn_conf['epochs'])
    print('    valid_loss_tracker: ', learn_conf['valid_loss_tracker'])

    test_conf = config_dict['test']
    if args.mode == 'test':
        print('> test_conf: ')
        print('    checkpoint: ', args.checkpoint)


    if args.mode == 'test' and args.slice_idx != -1:
        data_conf['slice_idx'] = args.slice_idx
        data_conf['kdat'] = data_conf['kdat'].split('.h5')[0][:-3] + str(args.slice_idx).zfill(3) + '.h5'

        print('> test slice: ', str(args.slice_idx).zfill(3))

        if data_conf['navi'] is not None:
            data_conf['navi'] = data_conf['navi'].split('.h5')[0][:-3] + str(args.slice_idx).zfill(3) + '.h5'

    if args.mode == 'train':
        RECON_DIR = util.set_output_dir(DIR, config_dict)

        yaml_file = 'zsssl.yaml'

    elif args.mode == 'test':

        relative_path = args.checkpoint.rsplit('/', 1)[0]
        RECON_DIR = HOME_DIR + relative_path

        yaml_file = 'zsssl_slice_' + str(data_conf['slice_idx']).zfill(3) + '.yaml'

    print('> RECON_DIR: ', RECON_DIR)
    # make a new directory if not exist
    pathlib.Path(RECON_DIR).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(RECON_DIR, yaml_file), 'w') as f:
        f.write(yaml.dump(config_dict, sort_keys=False))


    # %%
    coil4, kdat6, kdat_scaling, phase_shot, phase_slice, mask = \
        prep.prep_dwi_data(data_file=data_conf['kdat'],
                           navi_file=data_conf['navi'],
                           coil_file=data_conf['coil'],
                           slice_idx=data_conf['slice_idx'],
                           norm_kdat=data_conf['normalize_kdat'],
                           N_shot_retro=args.N_shot_retro,
                           N_diff_split=data_conf['N_diff_split'],
                           N_diff_split_index=data_conf['N_diff_split_index'])

    mask, train_mask, lossf_mask, valid_mask = \
        prep_mask(mask, N_repeats=data_conf['repeats'],
                  valid_rho=data_conf['valid_rho'],
                  train_rho=data_conf['train_rho'])

    coil7, kdat7, phase_shot7, phase_slice7 = \
        repeat_data(coil4, kdat6, phase_shot, phase_slice,
                    N_repeats=data_conf['repeats'])

    print('>>> coil7 shape\t: ', coil7.shape, ' type: ', coil7.dtype)
    print('>>> kdat7 shape\t: ', kdat7.shape, ' type: ', kdat7.dtype)
    # print('>>> phase_shot7 shape\t: ', phase_shot7.shape, ' type: ', phase_shot7.dtype)
    # print('>>> phase_slice7 shape\t: ', phase_slice7.shape, ' type: ', phase_slice7.dtype)

    print('>>> train_mask shape\t: ', train_mask.shape, ' type: ', train_mask.dtype)
    print('>>> lossf_mask shape\t: ', lossf_mask.shape, ' type: ', lossf_mask.dtype)
    print('>>> valid_mask shape\t: ', valid_mask.shape, ' type: ', valid_mask.dtype)

    S = mri.Sense(coil7[0], kdat7[0],
                  phase_slice=phase_slice7[0] if phase_slice7 is not None else None,
                  phase_echo=phase_shot7[0] if phase_shot7 is not None else None,
                  combine_echo=True)
    ishape = [data_conf['batch_size']] + list(S.ishape)
    print('>>> ishape to AlgUnroll: ', ishape)
    del S

    # %% train and valid
    train_data = zsssl.Dataset(coil7, kdat7, train_mask, lossf_mask, phase_shot7, phase_slice7)
    train_load = DataLoader(train_data, batch_size=data_conf['batch_size'])

    res_mask = train_mask + lossf_mask

    valid_data = zsssl.Dataset(coil7[[0]], kdat7[[0]], res_mask[[0]], valid_mask, phase_shot7[[0]], phase_slice7[[0]])
    valid_load = DataLoader(valid_data, batch_size=data_conf['batch_size'])

    if model_conf['net'] == 'ResNet2D' and model_conf['contrasts_in_channels'] is False:
        assert kdat7.shape[DIM_TIME] == 1 and kdat7.shape[DIM_ECHO] == 1

    model = zsssl.AlgUnroll(ishape, lamda=model_conf['lamda'],
                            NN=model_conf['net'],
                            requires_grad_lamda=model_conf['requires_grad_lamda'],
                            N_residual_block=model_conf['N_residual_block'],
                            unrolled_algorithm=model_conf['unrolled_algorithm'],
                            ADMM_rho=model_conf['ADMM_rho'],
                            N_unroll=model_conf['N_unroll'],
                            kernel_size=model_conf['kernel_size'],
                            features=model_conf['features'],
                            contrasts_in_channels=model_conf['contrasts_in_channels'],
                            max_cg_iter=model_conf['max_cg_iter'],
                            use_batch_norm=model_conf['batch_norm']).to(device)

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
        optimizer = optim.Adam(model.parameters(), lr=optim_conf['lr'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=optim_conf['lr'])

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=optim_conf['step_size'],
                                          gamma=optim_conf['gamma'])


    # %% train and valid
    checkpoint_name = 'zsssl_best.pth'
    if args.mode == 'train':
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
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            scheduler.step()

            epoch_x = torch.stack(epoch_x)
            # f = h5py.File(RECON_DIR + '/zsssl_epoch_' + str(epoch).zfill(3) + '.h5', 'w')
            # f.create_dataset('DWI', data=epoch_x.detach().cpu().numpy())
            # f.close()

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
                "optim_state": optimizer.state_dict()
            }

            if valid_loss <= valid_loss_min:
                valid_loss_min = valid_loss
                torch.save(checkpoint, os.path.join(RECON_DIR, checkpoint_name))
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

    if args.mode == 'train':
        best_checkpoint = torch.load(os.path.join(RECON_DIR, checkpoint_name))
    else:
        best_checkpoint = torch.load(HOME_DIR + args.checkpoint)

    best_epoch = best_checkpoint['epoch']
    print('> loaded best checkpoint at the ' + str(best_epoch+1).zfill(3) + 'th epoch')

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

    x_infer = np.array(x_infer) / kdat_scaling

    recon_file = '/zsssl_slice_' + str(data_conf['slice_idx']).zfill(3)
    if args.mode == 'test':
        recon_file += '_test_shot-retro-' + str(args.N_shot_retro)

    f = h5py.File(RECON_DIR + recon_file + '.h5', 'w')
    f.create_dataset('ZS', data=x_infer)
    f.close()
