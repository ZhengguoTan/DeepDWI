import argparse
import h5py
import os

import matplotlib.pyplot as plt
import numpy as np

# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import normalized_root_mse as nrmse

DIR = os.path.dirname(os.path.realpath(__file__))

# %%
parser = argparse.ArgumentParser(description='plot brain results.')

parser.add_argument('--diff_idx', type=int, default=19,
                    help='diffusion encoding index')

args = parser.parse_args()

# %%
f = h5py.File(DIR + '/results.h5', 'r')
DWI_MUSE = f['MUSE'][:]
DWI_VAE = f['VAE'][:]
DWI_LINSUB = f['LINSUB'][:]
f.close()

DWI_MUSE = np.flip(DWI_MUSE, axis=(-2))
DWI_VAE = np.flip(DWI_VAE, axis=(-2))
DWI_LINSUB = np.flip(DWI_LINSUB, axis=(-2))

N_diff, N_z, N_y, N_x = DWI_MUSE.shape


DWI_MUSE_1 = DWI_MUSE[:, 1, :, :]
DWI_VAE_1 = DWI_VAE[:, 1, :, :]
DWI_LINSUB_1 = DWI_LINSUB[:, 1, :, :]

row, col = 2, 3

diff_idx = [args.diff_idx]


for d in diff_idx:

    print('> diff %3d'%(d))

    fig, ax = plt.subplots(row, col, figsize=(col*4, row*4))

    muse = abs(DWI_MUSE_1[d])
    vmin = 0
    vmax = np.amax(muse)
    ax[0][0].imshow(muse, cmap='gray', interpolation=None,
                    vmin=vmin, vmax=vmax)
    ax[0][0].text(0.02*N_x, 0.08*N_y, 'MUSE',
                  fontsize=16, color='w')


    linsub = abs(DWI_LINSUB_1[d])
    ax[0][1].imshow(linsub, cmap='gray', interpolation=None,
                    vmin=vmin, vmax=vmax)
    ax[0][1].text(0.02*N_x, 0.08*N_y, 'SVD',
                  fontsize=16, color='w')


    vae = abs(DWI_VAE_1[d])
    ax[0][2].imshow(vae, cmap='gray', interpolation=None,
                    vmin=vmin, vmax=vmax)
    ax[0][2].text(0.02*N_x, 0.08*N_y, 'VAE',
                  fontsize=16, color='w')


    diff_l = muse - linsub
    ax[1][1].imshow(muse - linsub, cmap='gray', interpolation=None,
                    vmin=vmin, vmax=np.amax(diff_l)*0.2)
    ax[1][1].set_ylabel('diff X 5', fontsize=16)

    diff_v = muse - vae
    ax[1][2].imshow(diff_v, cmap='gray', interpolation=None,
                    vmin=vmin, vmax=np.amax(diff_l)*0.2)

    for m in range(row):
        for n in range(col):
            ax[m][n].axes.xaxis.set_ticks([])
            ax[m][n].axes.yaxis.set_ticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(DIR + '/vae_vs_linsub_brain_diff_' + str(d).zfill(2) + '.png',
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()