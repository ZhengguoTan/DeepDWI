import h5py
import os
import torch

import numpy as np
import sigpy as sp
import torchvision.transforms as T

from sigpy.mri import app, muse, retro, sms

from typing import Optional

DIR = os.path.dirname(os.path.realpath(__file__))

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
def prep_dwi_data(
    data_file: str = '/data/1.0mm_21-dir_R1x3_kdat_slice_010.h5',
    navi_file: Optional[str] = None,
    coil_file: str = '/data/1.0mm_21-dir_R1x3_coils.h5',
    slice_idx: int = 0,
    norm_kdat: float = 1.0,
    N_shot_retro: int = 0,
    N_diff_split: int = 1,
    N_diff_split_index: int = 0,
    return_muse_phase: bool = False,
    HOME_DIR: str = None):

    # %%
    if HOME_DIR is None:
        HOME_DIR = DIR.rsplit('/', 1)[0]
        HOME_DIR = HOME_DIR.rsplit('/', 1)[0]

    print('> HOME: ', HOME_DIR)

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
    # TODO: retro shots after normalize the kdat?
    if N_shot_retro > 0:
        kdat_prep = retro_usamp_shot(kdat_prep, N_shot_retro)

    if N_diff_split > 1:
        N_diff_sub = N_diff // N_diff_split
        diff_idx = range(N_diff_split_index * N_diff_sub,
                         (N_diff_split_index+1) * N_diff_sub if N_diff_split_index < N_diff_split else N_diff)
        kdat_prep = kdat_prep[diff_idx, ...]

    # normalize kdat
    if norm_kdat > 0:
        print('> norm_kdat: ', norm_kdat)
        kdat_scaling = norm_kdat / np.max(np.abs(kdat_prep[:]))
        kdat_prep = kdat_prep * kdat_scaling
    else:
        kdat_scaling = 1.

    N_diff, N_shot, N_coil, _, N_y, N_x = kdat_prep.shape

    print(' > kdat shape: ', kdat_prep.shape)


    # %% navi
    if navi_file is not None:
        f = h5py.File(HOME_DIR + navi_file, 'r')
        navi = f['navi'][:]
        f.close()

        navi = np.squeeze(navi)
        navi = np.swapaxes(navi, -2, -3)
        navi = np.swapaxes(navi, 0, 1)
        navi_prep = navi[..., None, :, :]  # 6 dim

    else:
        navi_prep = None

    # %% coil
    f = h5py.File(HOME_DIR + coil_file, 'r')
    coil = f['coil'][:]
    f.close()

    print(' > coil shape: ', coil.shape)
    N_coil, N_z, N_y, N_x = coil.shape

    # # MB coils
    slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(slice_idx, N_slices, MB)

    coil2 = coil[:, slice_mb_idx, :, :]
    print('> coil2 shape: ', coil2.shape)

    # %% sms phase
    yshift = []
    for b in range(MB):
        yshift.append(b / N_Accel_PE)

    sms_phase = sms.get_sms_phase_shift([MB, N_y, N_x], MB=MB, yshift=yshift)

    # %% shot phase
    if navi_prep is None:

        acs_shape = list([N_y // 2, N_x // 2])
        ksp_acs = sp.resize(kdat_prep,
                            oshape=list(kdat_prep.shape[:-2]) +\
                                acs_shape)

    else:

        N_navi_y, N_navi_x = navi_prep.shape[-2:]
        acs_shape = [N_navi_y, N_navi_x * 2]

        ksp_acs = sp.resize(navi_prep,
                            oshape=list(navi_prep.shape[:-2]) +\
                                acs_shape)

    coils_tensor = sp.to_pytorch(coil2)
    TR = T.Resize(acs_shape, antialias=True)
    c_small_r = TR(coils_tensor[..., 0]).cpu().detach().numpy()
    c_small_i = TR(coils_tensor[..., 1]).cpu().detach().numpy()
    c_small = c_small_r + 1j * c_small_i

    if N_shot > 1:

        _, dwi_shot = muse.MuseRecon(ksp_acs, c_small,
                                    MB=MB,
                                    acs_shape=acs_shape,
                                    lamda=0.01, max_iter=30,
                                    yshift=yshift,
                                    device=device_sp)


        _, dwi_shot_phase = muse._denoising(dwi_shot, full_img_shape=[N_y, N_x], max_iter=5)

        if navi_prep is not None:  # IMPORTANT
            dwi_shot_phase = np.conj(dwi_shot_phase)

    else:

        kdat_prep_dev = sp.to_device(kdat_prep, device=device_sp)
        c_small_dev = sp.to_device(c_small, device=device_sp)

        dwi_shot_phase = []
        for d in range(N_diff):

            k = kdat_prep_dev[d, ...]
            k_small = sp.resize(k, oshape=list(k.shape[:-2]) +\
                                acs_shape)

            A = muse.sms_sense_linop(k_small, c_small_dev, yshift)
            R = muse.sms_sense_solve(A, k_small, lamda=0.01, max_iter=30)
            _, R_phase = muse._denoising(R, full_img_shape=[N_y, N_x], max_iter=7)

            dwi_shot_phase.append(sp.to_device(R_phase))

        dwi_shot_phase = np.array(dwi_shot_phase)
        dwi_shot_phase = dwi_shot_phase[:, None, ...]  # 6 dim

    print(' > dwi_shot_phase shape: ', dwi_shot_phase.shape)

    # %% sampling mask
    mask = app._estimate_weights(kdat_prep, None, None, coil_dim=-4)
    mask = abs(mask).astype(float)

    print(' > mask shape: ', mask.shape)

    if return_muse_phase is True:

        if N_shot > 1:

            DWI_MUSE, _ = muse.MuseRecon(kdat_prep, coil2,
                                        MB=MB,
                                        acs_shape=acs_shape,
                                        lamda=0.01, max_iter=30,
                                        yshift=yshift,
                                        device=device_sp)

        else:

            kdat_prep_dev = sp.to_device(kdat_prep, device=device_sp)
            coil2_dev = sp.to_device(coil2, device=device_sp)

            DWI_MUSE = []

            for d in range(N_diff):
                k = kdat_prep_dev[d]

                A = muse.sms_sense_linop(k, coil2_dev, yshift)
                R = muse.sms_sense_solve(A, k, lamda=0.01, max_iter=30)

                DWI_MUSE.append(sp.to_device(R))

            DWI_MUSE = np.array(DWI_MUSE)

        DWI_MUSE_PHASE = np.exp(1j * np.angle(DWI_MUSE[:, None, ...]))

        with h5py.File(HOME_DIR + '/muse.h5', 'w') as f:
            f.create_dataset('muse', data=DWI_MUSE)
            print('> saved muse')

    else:
        DWI_MUSE_PHASE = None

    return coil2, kdat_prep, kdat_scaling, dwi_shot_phase, sms_phase, mask, DWI_MUSE_PHASE
