import argparse
import h5py
import os
import time
import torch

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp

from deepdwi.dims import *
from deepdwi.recons import zsssl
from sigpy.mri import app, muse, retro, sms
from torch.utils.data import DataLoader

DIR = os.path.dirname(os.path.realpath(__file__))

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

    f = h5py.File(DAT_DIR + '/' + data_file, 'r')
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
    f = h5py.File(DAT_DIR + '/' + coil_file, 'r')
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
def prep_mask(mask: np.ndarray, N_repeats: int = 12):
    mask = torch.from_numpy(mask)
    res_mask, valid_mask = zsssl.uniform_samp(mask, rho=0.2)
    valid_mask = valid_mask[None, ...]  # 7dim

    train_mask = []
    lossf_mask = []

    for r in range(N_repeats):

        train_mask1, lossf_mask1 = zsssl.uniform_samp(res_mask, rho=0.4)

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

    parser.add_argument('--data',
                        default='1.2mm_32-dir_R3x2_kdat_slice_000.h5',
                        help='raw dat file.')

    parser.add_argument('--norm_kdat', action='store_true',
                        help='normalize kspace data')

    parser.add_argument('--coil',
                        default='1.2mm_32-dir_R3x2_coil.h5',
                        help='coil sensitivity maps file.')

    parser.add_argument('--repeats', type=int, default=12,
                        help='number of repeats per epoch')

    parser.add_argument('--diff_idx', type=int, default=2,
                        help='reconstruct only on diffuion encoding')

    args = parser.parse_args()


    # %%
    coil4, kdat6, phase_shot, phase_slice, mask = prep_data(args.data, args.coil, slice_idx=0, norm_kdat=args.norm_kdat)

    mask, train_mask, lossf_mask, valid_mask = prep_mask(mask, N_repeats=args.repeats)

    coil7, kdat7, phase_shot7, phase_slice7 = repeat_data(coil4, kdat6, phase_shot, phase_slice, N_repeats=args.repeats)

    # run only one DWI direction and the first 10 coils
    if args.diff_idx >=0 and args.diff_idx < kdat7.shape[DIM_TIME]:
        kdat7 = kdat7[:, [args.diff_idx], ...]
        phase_shot7 = phase_shot7[:, [args.diff_idx], ...]
        mask = mask[[args.diff_idx], ...]
        train_mask = train_mask[:, [args.diff_idx], ...]
        lossf_mask = lossf_mask[:, [args.diff_idx], ...]
        valid_mask = valid_mask[:, [args.diff_idx], ...]

    print('>>> coil7 shape: ', coil7.shape, ' type: ', coil7.dtype)
    print('>>> kdat7 shape: ', kdat7.shape, ' type: ', kdat7.dtype)
    print('>>> phase_shot7 shape: ', phase_shot7.shape, ' type: ', phase_shot7.dtype)
    print('>>> phase_slice7 shape: ', phase_slice7.shape, ' type: ', phase_slice7.dtype)

    print('>>> train_mask shape: ', train_mask.shape, ' type: ', train_mask.dtype)
    print('>>> lossf_mask shape: ', lossf_mask.shape, ' type: ', lossf_mask.dtype)
    print('>>> valid_mask shape: ', valid_mask.shape, ' type: ', valid_mask.dtype)

    # %% train and valid

    train_data = zsssl.Dataset(coil7, kdat7, train_mask, lossf_mask, phase_shot7, phase_slice7)
    train_load = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=6)

    res_mask = train_mask + lossf_mask

    valid_data = zsssl.Dataset(coil7[[0]], kdat7[[0]], res_mask[[0]], valid_mask, phase_shot7[[0]], phase_slice7[[0]])
    valid_load = DataLoader(valid_data, batch_size=1, shuffle=True, num_workers=6)

    if kdat7.shape[DIM_TIME] == 1:
        model = zsssl.UnrollNet(NN='ResNet2D', requires_grad_lamda=True, N_unroll=8, features=4)
    else:
        model = zsssl.UnrollNet(NN='ResNet3D', requires_grad_lamda=False)


    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(">>> number of trainable parameters is: ", params)

    lossf = zsssl.MixL1L2Loss()
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)

    # %% training
    total_train_loss, total_valid_loss = [], []
    valid_loss_min = np.inf

    epoch, val_loss_tracker = 0, 0

    start_time=time.time()

    while epoch < 50 and val_loss_tracker < 25:

        tic = time.time()
        trn_loss, lamda = zsssl.train(model, train_load, lossf, optim, device=device)
        val_loss = zsssl.valid(model, valid_load, lossf, optim, device=device)

        total_train_loss.append(trn_loss)
        total_valid_loss.append(val_loss)

        #save the best checkpoint
        checkpoint = {
            "epoch": epoch,
            "valid_loss_min":val_loss,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict()
        }

        if val_loss <= valid_loss_min:
            valid_loss_min = val_loss
            torch.save(checkpoint, os.path.join(DIR, "best.pth"))
            val_loss_tracker = 0 #reset the val loss tracker each time a new lowest val error is achieved
        else:
            val_loss_tracker += 1

        toc = time.time() - tic
        if epoch % 1 == 0:
            print("Epoch:", str(epoch+1).zfill(3), ", elapsed_time = ""{:7.3f}".format(toc), ", trn loss = ", "{:9.6f}".format(trn_loss),", val loss = ", "{:9.6f}".format(val_loss), ", lamda = ", "{:9.6f}".format(lamda.item()))

        epoch += 1

    end_time = time.time()
    print('Training completed in  ', str(epoch), ' epochs, ',((end_time - start_time) / 60), ' minutes')

    # %% inference
    infer_data = zsssl.Dataset(coil7[[0]], kdat7[[0]], mask[np.newaxis], mask[np.newaxis], phase_shot7[[0]], phase_slice7[[0]])
    infer_load = DataLoader(infer_data, batch_size=1, shuffle=True, num_workers=6)

    best_checkpoint = torch.load(os.path.join(DIR, 'best.pth'))
    model.load_state_dict(best_checkpoint["model_state"])

    x_infer = zsssl.test(model, infer_load, device=device)
    f = h5py.File(DIR + '/infer_result.h5', 'w')
    f.create_dataset('infer', data=x_infer.detach().cpu().numpy())
    f.close()