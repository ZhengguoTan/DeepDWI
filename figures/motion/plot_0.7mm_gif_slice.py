import h5py
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]
HOME_DIR = HOME_DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

DATA_DIR = HOME_DIR + '/data/'
print('> DATA: ', DATA_DIR)

# %% mkdir
GIF_DIR = DIR + '/gif_slice'
pathlib.Path(GIF_DIR).mkdir(parents=True, exist_ok=True)

# %%
f = h5py.File(DATA_DIR + '/0.7mm_21-dir_R2x2_vol1_scan1_JETS.h5', 'r')
DWI_JETS_NAVI = f['DWI'][:]
f.close()

f = h5py.File(HOME_DIR + '/examples/2024-05-23_zsssl_0.7mm_21-dir_R2x2_vol1_scan1_kdat_slice_040_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_test_shot-retro-0.h5', 'r')
DWI_ZSSSL_SELF = f['DWI'][:]
f.close()

print('> DWI_ZSSSL_SELF shape: ', DWI_ZSSSL_SELF.shape)
print('> DWI_JETS_NAVI shape: ', DWI_JETS_NAVI.shape)

N_diff, N_z, N_y, N_x = DWI_JETS_NAVI.shape

# %% axial
vmax = np.amax(abs(DWI_JETS_NAVI)) * 0.04

diff_idx = 5

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

for z in range(N_z):

    N_row, N_col = 1, 4
    fig = plt.figure(figsize=(N_col*4, N_row*4))

    fontsize = 6

    fig_size = fig.get_size_inches()
    fig_width = fig_size[0]
    fontsize = fontsize * (fig_width / 6)

    gs = fig.add_gridspec(2, N_col, width_ratios=[1] * N_col)
    ax = []
    ax.append(fig.add_subplot(gs[:,0]))
    ax.append(fig.add_subplot(gs[0,1]))
    ax.append(fig.add_subplot(gs[1,1]))

    ax.append(fig.add_subplot(gs[:,2]))
    ax.append(fig.add_subplot(gs[0,3]))
    ax.append(fig.add_subplot(gs[1,3]))

    # self-gating
    ax[0].imshow(abs(np.flip(DWI_JETS_NAVI[diff_idx, z, :, :], axis=-2)),
                    cmap='gray', interpolation=None,
                    vmin=0, vmax=vmax)

    ax[1].imshow(abs(np.flip(DWI_JETS_NAVI[diff_idx, (N_z - int(N_x/2)):, (N_y - N_z)//2 + z, :], axis=-2)),
                    cmap='gray', interpolation=None,
                    vmin=0, vmax=vmax)

    ax[2].imshow(abs(np.flip(DWI_JETS_NAVI[diff_idx, (N_z - int(N_x/2)):, :, (N_x - N_z)//2 + z], axis=-2)),
                    cmap='gray', interpolation=None,
                    vmin=0, vmax=vmax)

    # self-gating
    ax[3].imshow(abs(np.flip(DWI_ZSSSL_SELF[diff_idx, z, :, :], axis=-2)),
                    cmap='gray', interpolation=None,
                    vmin=0, vmax=vmax)

    ax[4].imshow(abs(np.flip(DWI_ZSSSL_SELF[diff_idx, (N_z - int(N_x/2)):, (N_y - N_z)//2 + z, :], axis=-2)),
                    cmap='gray', interpolation=None,
                    vmin=0, vmax=vmax)

    ax[5].imshow(abs(np.flip(DWI_ZSSSL_SELF[diff_idx, (N_z - int(N_x/2)):, :, (N_x - N_z)//2 + z], axis=-2)),
                    cmap='gray', interpolation=None,
                    vmin=0, vmax=vmax)


    ax[0].text(0.03*N_x, 0.08*N_y, 'Navigated LLR', bbox=props,
                color='w', fontsize=fontsize)

    ax[3].text(0.03*N_x, 0.08*N_y, 'Self-Gated ZSSSL', bbox=props,
               color='w', fontsize=fontsize)

    for m in range(6):
        ax[m].axes.xaxis.set_ticks([])
        ax[m].axes.yaxis.set_ticks([])


    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig(GIF_DIR + '/slice-' + str(z).zfill(3) + '.png',
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
