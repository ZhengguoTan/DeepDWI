import argparse
import h5py
import os

import numpy as np
import sigpy as sp
import torchvision.transforms as T

from sigpy.mri import retro, app, sms, muse, mussels
from os.path import exists

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

DATA_DIR = HOME_DIR + '/data/'
print('> DATA: ', DATA_DIR)

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

# %% options
parser = argparse.ArgumentParser(description='run reconstruction.')

parser.add_argument('--prefix',
                    default='1.0mm_126-dir_R3x3',
                    help='dat file prefix')

parser.add_argument('--slice_idx', type=int, default=-1,
                    help='recon only one slice given its index [default: -1].')

parser.add_argument('--slice_inc', type=int, default=1,
                    help='slice increments [default: 1].')

parser.add_argument('--pi', action='store_true',
                    help='run paralel imaging recon.')

parser.add_argument('--muse', action='store_true',
                    help='run MUSE recon.')

parser.add_argument('--mussels', action='store_true',
                    help='run MUSSELS recon.')

parser.add_argument('--jets', action='store_true',
                    help='run JETS recon.')

parser.add_argument('--admm_rho', type=float, default=0.05,
                    help='admm rho [default: 0.05].')

parser.add_argument('--admm_lamda', type=float, default=0.04,
                    help='admm lamda [default: 0.04].')

parser.add_argument('--split', type=int, default=1,
                    help='split diffusion encodings in recon [default: 1]')

parser.add_argument('--shot', type=int, default=0,
                    help='number of shots to use in recon')

parser.add_argument('--device', type=int, default=0,
                    help='which device to run recon [default: 0]')

args = parser.parse_args()

print('> data: ', args.prefix)

device = sp.Device(args.device)
xp = device.xp

# %% read in raw data
instr = DATA_DIR + '/' + args.prefix

# read in coils
coil_file = instr + '_coils'
print('> coil: ' + coil_file)
f = h5py.File(coil_file + '.h5', 'r')
coil = f['coil'][:]
f.close()

print('> coil shape: ', coil.shape)

N_y, N_x = coil.shape[-2:]

# read in other parameters
f = h5py.File(instr + '_kdat_slice_000.h5', 'r')
MB = f['MB'][()]
N_slices = f['Slices'][()]
N_segments = f['Segments'][()]
N_Accel_PE = f['Accel_PE'][()]
f.close()

# number of collapsed slices
N_slices_collap = N_slices // MB

# SMS phase shift
yshift = []
for b in range(MB):
    yshift.append(b / N_Accel_PE)

sms_phase = sms.get_sms_phase_shift([MB, N_y, N_x], MB=MB, yshift=yshift)

# %% run reconstruction
if args.slice_idx >= 0:
    slice_loop = range(args.slice_idx, args.slice_idx + args.slice_inc, 1)
else:
    slice_loop = range(N_slices_collap)

for s in slice_loop:

    slice_str = str(s).zfill(3)
    print('> collapsed slice idx: ', slice_str)

    # read in k-space data
    f = h5py.File(instr + '_kdat_slice_' + slice_str + '.h5', 'r')
    kdat = f['kdat'][:]
    f.close()

    kdat = np.squeeze(kdat)  # 4 dim
    kdat = np.swapaxes(kdat, -2, -3)

    # split kdat into shots
    N_diff = kdat.shape[-4]
    kdat_prep = []
    for d in range(N_diff):
        k = retro.split_shots(kdat[d, ...], shots=N_segments)
        kdat_prep.append(k)

    kdat_prep = np.array(kdat_prep)
    kdat_prep = kdat_prep[..., None, :, :]  # 6 dim

    print('>> kdat_prep shape: ', kdat_prep.shape)
    N_diff, N_shot, N_coil, N_z, N_y, N_x = kdat_prep.shape


    shot_str = ''
    if (args.shot > 0) and (args.shot < N_shot):
        kdat_prep = retro_usamp_shot(kdat_prep, args.shot)

        N_diff, N_shot, N_coil, N_z, N_y, N_x = kdat_prep.shape
        print('>> kdat_prep shape: ', kdat_prep.shape)

        shot_str = '_%1dshot'%(N_shot)


    slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(s, N_slices, MB)

    print('>> slice_mb_idx: ', slice_mb_idx)
    coil2 = coil[:, slice_mb_idx, :, :]

    # %% Phase estimation from full FOV or not
    RECON_ON_FULL_FOV = False

    if RECON_ON_FULL_FOV:
        kdat_resi = kdat_prep
        coil_resi = coil2
        sms_phase_resi = sms_phase

        acs_shape = list(np.array([N_y, N_x]) // 4)

    else:
        shot_fov = list(np.array([N_y, N_x]) // 4)
        kdat_resi = sp.resize(kdat_prep,
                              oshape=list(kdat_prep.shape[:-2]) +\
                                  shot_fov)

        coil_tensor = sp.to_pytorch(coil2)
        TR = T.Resize(shot_fov)
        coil_resi_r = TR(coil_tensor[..., 0]).cpu().detach().numpy()
        coil_resi_i = TR(coil_tensor[..., 1]).cpu().detach().numpy()
        coil_resi = coil_resi_r + 1j * coil_resi_i

        sms_phase_resi = sms.get_sms_phase_shift([MB] + shot_fov,
                                                 MB=MB, yshift=yshift)

        acs_shape = shot_fov

    # %% PI
    if (args.pi is True) and (N_segments == 1):

        kdat_prep_dev = sp.to_device(kdat_prep, device=device)
        coil2_dev = sp.to_device(coil2, device=device)

        dwi_pi = []

        for d in range(N_diff):
            k = kdat_prep_dev[d, ...]

            A = muse.sms_sense_linop(k, coil2_dev, yshift)
            R = muse.sms_sense_solve(A, k, lamda=0.01, max_iter=30)

            dwi_pi.append(sp.to_device(R))

        dwi_pi = np.array(dwi_pi)

        print('>>> dwi_pi shape: ', dwi_pi.shape)

        # store output
        f = h5py.File(instr + '_PI_slice_' + slice_str + '.h5', 'w')
        f.create_dataset('dwi_pi', data=dwi_pi)
        f.close()


    # %% JETS
    if (args.jets is True) and (N_segments > 1):

        N_diff_split = N_diff // args.split

        for s in range(args.split):

            if args.split == 1:
                split_str = ""
            else:
                split_str = "_part_" + str(s).zfill(1)

            diff_idx = range(s * N_diff_split, (s+1) * N_diff_split if s < args.split else N_diff)

            kdat_prep_split = kdat_prep[diff_idx, ...]

            USE_SHOT_LLR = False

            if USE_SHOT_LLR is True:

                dwi_shot = app.HighDimensionalRecon(
                                    kdat_resi[diff_idx, ...],
                                    coil_resi,
                                    phase_sms=sms_phase_resi,
                                    combine_echo=False,
                                    regu='LLR',
                                    blk_shape=(1, 6, 6),
                                    blk_strides=(1, 1, 1),
                                    normalization=True,
                                    solver='ADMM',
                                    lamda=args.admm_lamda,
                                    rho=args.admm_rho,
                                    max_iter=15,
                                    show_pbar=False, verbose=True,
                                    device=device).run()

                dwi_shot = sp.to_device(dwi_shot_llr)

            else: # use parallel imaging

                _, dwi_shot = muse.MuseRecon(kdat_prep_split, coil2,
                                    MB=MB,
                                    acs_shape=acs_shape,
                                    lamda=0.01, max_iter=30,
                                    yshift=yshift,
                                    device=device)

            print('shot images shape: ', dwi_shot.shape)

            dwi_shot_full, dwi_shot_phase = muse._denoising(dwi_shot, full_img_shape=[N_y, N_x])

            # # shot-combined recon

            dwi_comb_llr_imag = app.HighDimensionalRecon(
                                    kdat_prep[diff_idx, ...], coil2,
                                    phase_sms=sms_phase,
                                    combine_echo=True,
                                    phase_echo=dwi_shot_phase,
                                    regu='LLR', blk_shape=(1, 6, 6), blk_strides=(1, 1, 1),
                                    normalization=True,
                                    solver='ADMM',
                                    lamda=args.admm_lamda,
                                    rho=args.admm_rho,
                                    max_iter=15,
                                    show_pbar=False, verbose=True,
                                    device=device).run()

            dwi_comb_llr_imag = sp.to_device(dwi_comb_llr_imag)

            # store output
            f = h5py.File(instr + '_JETS_slice_' + slice_str + split_str + '.h5', 'w')
            f.create_dataset('dwi_shot', data=dwi_shot_full)
            f.create_dataset('dwi_comb', data=dwi_comb_llr_imag)
            f.create_dataset('admm_lamda', data=args.admm_lamda)
            f.create_dataset('admm_rho', data=args.admm_rho)
            f.close()

    if (args.jets is True) and (N_segments == 1):

        dwi_shot_llr = app.HighDimensionalRecon(kdat_prep, coil2,
                                    phase_sms=sms_phase,
                                    combine_echo=False,
                                    regu='LLR',
                                    blk_shape=(1, 6, 6),
                                    blk_strides=(1, 1, 1),
                                    normalization=True,
                                    solver='ADMM', lamda=0.005, rho=0.05,
                                    max_iter=15,
                                    show_pbar=False, verbose=True,
                                    device=device).run()

        dwi_shot_llr = sp.to_device(dwi_shot_llr)

        # store output
        f = h5py.File(instr + '_JETS_slice_' + slice_str + '.h5', 'w')
        f.create_dataset('dwi_shot', data=dwi_shot_llr)
        f.close()


    # %% MUSE
    if (args.muse is True) and (N_segments > 1):

        dwi_muse, dwi_shot = muse.MuseRecon(kdat_prep, coil2, MB=MB,
                                acs_shape=acs_shape,
                                lamda=0.01, max_iter=30,
                                yshift=yshift,
                                device=device)

        dwi_muse = sp.to_device(dwi_muse)

        # store output
        recon_file = '_MUSE_slice_' + slice_str + '.h5'
        f = h5py.File(instr + recon_file, 'w')
        f.create_dataset('DWI', data=dwi_muse)
        f.create_dataset('DWI_shot', data=dwi_shot)
        f.close()


    # %% MUSSELS
    if (args.mussels is True) and (N_segments > 1):

        dwi_mussels = mussels.MusselsRecon(kdat_prep, coil2, MB=MB,
                                lamda=0.02, rho=0.05, max_iter=50,
                                yshift=yshift,
                                device=device)

        dwi_mussels = sp.to_device(dwi_mussels)

        # store output
        recon_file = '_MUSSELS_slice_' + slice_str + '.h5'
        f = h5py.File(instr + recon_file, 'w')
        f.create_dataset('DWI', data=dwi_mussels)
        f.close()
