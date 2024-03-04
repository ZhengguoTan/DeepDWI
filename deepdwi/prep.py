import h5py
import os
import torch

import numpy as np
import sigpy as sp

from sigpy.mri import app, muse, retro, sms

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

device_sp = sp.Device(0 if torch.cuda.is_available() else -1)


# %%
def retro_usamp_shot(input, N_shot_retro: int = 1, shift: bool = True):

    N_diff, N_shot, N_coil, N_z, N_y, N_x = input.shape

    assert N_shot_retro <= N_shot and N_shot % N_shot_retro == 0

    R = N_shot // N_shot_retro

    output = np.zeros_like(input, shape=[N_diff, N_shot_retro] + list(input.shape[2:]))

    for d in range(N_diff):

        offset = d % R

        shot_ind = [offset + R * s for s in range(N_shot_retro)]

        # print(str(d).zfill(3), shot_ind)

        output[d, ...] = input[d, shot_ind, ...]

    return output


# %%
def prep_dwi_data(data_file: str = '/data/1.0mm_21-dir_R1x3_kdat_slice_010.h5',
                  coil_file: str = '/data/1.0mm_21-dir_R1x3_coils.h5',
                  slice_idx: int = 0,
                  norm_kdat: float = 1.0,
                  N_shot_retro: int = 0,
                  N_diff_retro: int = 0,
                  return_muse: bool = False):

    f = h5py.File(HOME_DIR + data_file, 'r')
    kdat = f['kdat'][:]
    MB = f['MB'][()]
    N_slices = f['Slices'][()]
    N_segments = f['Segments'][()]
    N_Accel_PE = f['Accel_PE'][()]
    # slice_idx = f['slice_idx'][()]
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

    # retro undersampling shots
    if N_shot_retro > 0:
        kdat_prep = retro_usamp_shot(kdat_prep, N_shot_retro)

    if N_diff_retro > 0 and N_diff_retro < N_diff:
        kdat_prep = kdat_prep[:N_diff_retro, ...]

    # normalize kdat
    if norm_kdat > 0:
        print('> norm_kdat: ', norm_kdat)
        kdat_prep = norm_kdat * kdat_prep / np.max(np.abs(kdat_prep[:]))

    N_diff, N_shot, N_coil, _, N_y, N_x = kdat_prep.shape

    print(' > kdat shape: ', kdat_prep.shape)

    # coil
    f = h5py.File(HOME_DIR + coil_file, 'r')
    coil = f['coil'][:]
    f.close()

    print(' > coil shape: ', coil.shape)

    N_coil, N_z, N_y, N_x = coil.shape

    # # sms phase
    yshift = []
    for b in range(MB):
        yshift.append(b / N_Accel_PE)

    sms_phase = sms.get_sms_phase_shift([MB, N_y, N_x], MB=MB, yshift=yshift)

    # # coils
    slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(slice_idx, N_slices, MB)

    coil2 = coil[:, slice_mb_idx, :, :]
    print('> coil2 shape: ', coil2.shape)

    # # shot phase
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

    _, dwi_shot_phase = muse._denoising(dwi_shot, full_img_shape=[N_y, N_x], max_iter=5)

    # # sampling mask
    mask = app._estimate_weights(kdat_prep, None, None, coil_dim=-4)
    mask = abs(mask).astype(float)

    print(' > mask shape: ', mask.shape)

    if return_muse is True:

        DWI_MUSE, _ = muse.MuseRecon(kdat_prep, coil2,
                                     MB=MB,
                                     acs_shape=acs_shape,
                                     lamda=0.01, max_iter=30,
                                     yshift=yshift,
                                     device=device_sp)

        return coil2, kdat_prep, dwi_shot_phase, sms_phase, mask, DWI_MUSE

    else:

        return coil2, kdat_prep, dwi_shot_phase, sms_phase, mask
