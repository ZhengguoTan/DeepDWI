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

args = parser.parse_args()

# %%
acq_slice = []

for s in range(args.slices):

    print('> slice ' + str(s).zfill(3))

    fstr = DIR + '/' + args.dir + '/' + args.method_pre + '_slice_' + str(s).zfill(3) + args.method_post

    f = h5py.File(fstr + '.h5', 'r')
    acq_slice.append(f[args.key][:])
    f.close()


acq_slice = np.array(acq_slice)
print('> acq_slice shape: ', acq_slice.shape)

# total number of slices
N_slices = args.slices * args.MB

reo_slice = sms.reorder_slices_mbx(acq_slice, args.MB, N_slices)
reo_slice = np.squeeze(reo_slice)

f = h5py.File(DIR + '/' + args.dir + '/' + args.method_pre + args.method_post + '.h5', 'w')
f.create_dataset('DWI', data=reo_slice)
f.close()
