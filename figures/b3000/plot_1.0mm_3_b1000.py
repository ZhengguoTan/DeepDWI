import h5py
import os

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]
HOME_DIR = HOME_DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

DATA_DIR = HOME_DIR + '/data/'
print('> DATA: ', DATA_DIR)

# %%
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
for met in ['llr', 'zsssl']:

    if met == 'llr':
        f = h5py.File('/home/atuin/b143dc/b143dc15/Experiments/2023-09-26_Terra_Diffusion_iEPI/meas_MID00083_FID00084_Seg2_1p0_126dir_BW1086/JETS.h5', 'r')

    elif met == 'zsssl':
        f = h5py.File('/home/atuin/b143dc/b143dc15/Softwares/DeepDWI/examples/2024-08-07_zsssl_1.0mm_126-dir_R3x3_kdat_slice_000_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_test_shot-retro-0.h5', 'r')  # 128 channels

    DWI = f['DWI'][:]
    f.close()

    N_diff, N_z, N_y, N_x = DWI.shape
    # N_cube = int(N_x * 0.75)
    # DWI_crop = sp.resize(DWI, [N_diff, N_z] + [N_cube] * 2)

    # # #
    N_row, N_col = 1, 2
    fig = plt.figure(figsize=(N_col*4, N_row*4))

    fontsize = 9

    fig_size = fig.get_size_inches()
    fig_width = fig_size[0]
    fontsize = fontsize * (fig_width / 6)

    gs = fig.add_gridspec(2, 2, width_ratios=[1,1])
    ax = []
    ax.append(fig.add_subplot(gs[:,0]))
    ax.append(fig.add_subplot(gs[0,1]))
    ax.append(fig.add_subplot(gs[1,1]))

    diff_idx = 2

    vmax_scale = 0.6

    # axial
    tra_slice_idx = 61
    img = np.flip(abs(DWI[diff_idx, tra_slice_idx, :, :]), axis=(-2))
    ax[0].imshow(img, cmap='gray',
                 interpolation=None, vmin=0, vmax=np.amax(img)*vmax_scale)

    ax[0].text(0.03*N_x, 0.08*N_y, met.upper(), bbox=props,
                    color='w', fontsize=16)

    ax[0].text(0.03*N_x, 0.97*N_y, 'b1000', color='w', fontsize=16)

    # coronal
    cor_slice_idx = 88
    img = np.flip(abs(DWI[diff_idx, (N_z - int(N_x/2)):, cor_slice_idx, :]), axis=(-2))
    ax[1].imshow(img, cmap='gray',
                 interpolation=None, vmin=0, vmax=np.amax(img)*vmax_scale)

    # sagittal
    sag_slice_idx = 93
    img = np.flip(abs(DWI[diff_idx, (N_z - int(N_x/2)):, :, sag_slice_idx]), axis=(-2))

    ax[2].imshow(img, cmap='gray',
                 interpolation=None, vmin=0, vmax=np.amax(img)*vmax_scale)


    for n in range(3):
        ax[n].axes.xaxis.set_ticks([])
        ax[n].axes.yaxis.set_ticks([])

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig(DIR + '/1.0mm_dwi_' + met + '_b1000.png',
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()