import argparse
import h5py
import os

import numpy as np
import sigpy as sp

from sigpy.mri import sms

DIR = os.path.dirname(os.path.realpath(__file__))

# %% options
parser = argparse.ArgumentParser(description='Read in slice files, append them, and save in correct order.')

parser.add_argument('--dir', default='2024-05-13_zsssl_0.7mm_21-dir_R2x2_vol2_scan2_kdat_slice_040_norm-kdat-1.0_navi_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss',
                    help='directory in which the data are read.')

parser.add_argument('--method_pre', default='zsssl',
                    help='file name before the slice number.')

parser.add_argument('--method_post', default='_test_shot-retro-0',
                    help='file name after the slice number.')

parser.add_argument('--key', default='ZS',
                    help='prefix of the file name to look for.')

parser.add_argument('--MB', type=int, default=2,
                    help='multi-band factor.')

parser.add_argument('--slices', type=int, default=88,
                    help='total number of slices.')

parser.add_argument('--slice_idx', type=int, default=0,
                    help='slice index to start with.')

parser.add_argument('--order', type=str,
                    choices=['interleaved', 'sequential'],
                    default='interleaved',
                    help='slice order.')

parser.add_argument('--sum', metavar='N', type=int, nargs='+',
                    help='sum over the dims')

parser.add_argument('--rss', metavar='N', type=int, nargs='+',
                    help='rss over the dims')

parser.add_argument('--resize', action='store_true',
                    help='resize the image.')

parser.add_argument('--zf', type=int, default=12,
                    help='zero-fill in the kx and ky dimensions.')

args = parser.parse_args()

# %%
acq_slice = []

if args.slices == 1:
    slice_loop = range(args.slice_idx, args.slice_idx + 1)
else:
    slice_loop = range(args.slices)

for s in slice_loop:

    print('> slice ' + str(s).zfill(3))

    fstr = DIR + '/' + args.dir + '/' + args.method_pre + '_slice_' + str(s).zfill(3) + args.method_post

    with h5py.File(fstr + '.h5', 'r') as f:
        I = f[args.key][:]

    if args.sum is not None:
        print('  - sum over dims ' + str(args.sum))
        I = np.sum(I, axis=tuple(args.sum))

    if args.rss is not None:
        print('  - rss over dims ' + str(args.rss))
        I = sp.rss(I, axes=tuple(args.rss))

    print('  - I shape: ', I.shape)

    acq_slice.append(I)

acq_slice = np.array(acq_slice)
print('> acq_slice shape: ', acq_slice.shape)

# total number of slices
N_slices = args.slices * args.MB

if args.order == 'interleaved':
    reo_slice = sms.reorder_slices_mbx(acq_slice, args.MB, N_slices)
elif args.order == 'sequential':
    reo_slice = acq_slice

reo_slice = np.squeeze(reo_slice)

if args.slices == 1:
    reo_slice = reo_slice[:, None, :, :]

print('> reo_slice shape: ', reo_slice.shape)

if args.resize is True:
    N_diff, N_z, N_y, N_x = reo_slice.shape
    N_i = min(N_y, N_x)
    reo_slice = sp.resize(reo_slice, [N_diff, N_z, N_i, N_i])
    print('> reo_slice shape: ', reo_slice.shape)

N_y, N_x = reo_slice.shape[-2:]

if args.zf > N_y and args.zf > N_x:
    reo_slice = sp.fft(reo_slice, axes=(-2, -1))
    reo_slice = sp.resize(reo_slice, list(reo_slice.shape[:-2]) + [args.zf, args.zf])
    reo_slice = sp.ifft(reo_slice, axes=(-2, -1))
    print('> reo_slice shape: ', reo_slice.shape)

f = h5py.File(DIR + '/' + args.dir + '/' + args.method_pre + args.method_post + '.h5', 'w')
f.create_dataset('DWI', data=reo_slice)
f.close()
