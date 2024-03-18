import h5py
import os

import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]
HOME_DIR = HOME_DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

DATA_DIR = HOME_DIR + '/data/'
print('> DATA: ', DATA_DIR)

# %%

for met in ['jets', 'ss']:

    print('> ' + met)

    # %% read in dwi
    if met == 'jets':
        recon_list = [DATA_DIR + '/0.7mm_21-dir_R2x2_JETS_PHASE-IMAG_slice_000.h5',
                      DATA_DIR + '/0.7mm_21-dir_R2x2_JETS_PHASE-NAVI_slice_000.h5']

        key = 'dwi_comb_jets'

    elif met == 'ss':
        recon_list = [HOME_DIR + '/examples/2024-03-16_zsssl_0.7mm_21-dir_R2x2_kdat_slice_000_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_slice_000.h5',
                      HOME_DIR + '/examples/2024-03-16_zsssl_0.7mm_21-dir_R2x2_kdat_slice_000_norm-kdat-1.0_navi_ResNet2D_ResBlock-12_kernel-3_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_slice_000.h5']

        key = 'ZS'


    f = h5py.File(recon_list[0], 'r')
    DWI_IMAG = np.squeeze(f[key][:])
    f.close()

    f = h5py.File(recon_list[1], 'r')
    DWI_NAVI = np.squeeze(f[key][:])
    f.close()

    print('> DWI_IMAG shape: ', DWI_IMAG.shape)
    print('> DWI_NAVI shape: ', DWI_NAVI.shape)

    N_diff, N_z, N_y, N_x = DWI_NAVI.shape

    # %%
    N_row = 2
    N_col = 4

    fig = plt.figure(constrained_layout=True, figsize=(N_col*4, N_row*4+0.5))
    subfigs = fig.subfigures(1, 2, wspace=0.01, width_ratios=[0.50, 0.50])

    fontsize = 9

    fig_size = fig.get_size_inches()
    fig_width = fig_size[0]
    fontsize = fontsize * (fig_width / 6)

    vmax = np.amax(abs(DWI_NAVI)) * 0.08

    diff_idx_motion0 = 18
    diff_idx_motion1 =  5

    ax0 = subfigs[0].subplots(2, 2)
    ax1 = subfigs[1].subplots(2, 2)

    # self-gating
    ax0[0][0].imshow(abs(np.flip(DWI_IMAG[diff_idx_motion0, 1, :, :], axis=-2)),
                    cmap='gray', interpolation=None,
                    vmin=0, vmax=vmax)

    ax0[0][1].imshow(np.angle(np.flip(DWI_IMAG[diff_idx_motion0, 1, :, :], axis=-2)),
                    cmap='RdBu_r', interpolation=None,
                    vmin=-np.pi, vmax=np.pi)

    ax0[0][0].set_title('Magnitude', fontsize=fontsize-4)
    ax0[0][1].set_title('Phase', fontsize=fontsize-4)


    ax0[1][0].imshow(abs(np.flip(DWI_IMAG[diff_idx_motion1, 1, :, :], axis=-2)),
                    cmap='gray', interpolation=None,
                    vmin=0, vmax=vmax)

    ax0[1][1].imshow(np.angle(np.flip(DWI_IMAG[diff_idx_motion1, 1, :, :], axis=-2)),
                    cmap='RdBu_r', interpolation=None,
                    vmin=-np.pi, vmax=np.pi)

    # navigator
    ax1[0][0].imshow(abs(np.flip(DWI_NAVI[diff_idx_motion0, 1, :, :], axis=-2)),
                    cmap='gray', interpolation=None,
                    vmin=0, vmax=vmax)

    ax1[0][1].imshow(np.angle(np.flip(DWI_NAVI[diff_idx_motion0, 1, :, :], axis=-2)),
                    cmap='RdBu_r', interpolation=None,
                    vmin=-np.pi, vmax=np.pi)

    ax1[0][0].set_title('Magnitude', fontsize=fontsize-4)
    ax1[0][1].set_title('Phase', fontsize=fontsize-4)


    ax1[1][0].imshow(abs(np.flip(DWI_NAVI[diff_idx_motion1, 1, :, :], axis=-2)),
                    cmap='gray', interpolation=None,
                    vmin=0, vmax=vmax)

    ax1[1][1].imshow(np.angle(np.flip(DWI_NAVI[diff_idx_motion1, 1, :, :], axis=-2)),
                    cmap='RdBu_r', interpolation=None,
                    vmin=-np.pi, vmax=np.pi)


    for m in range(2):
        for n in range(2):
            ax0[m][n].axes.xaxis.set_ticks([])
            ax0[m][n].axes.yaxis.set_ticks([])

            ax1[m][n].axes.xaxis.set_ticks([])
            ax1[m][n].axes.yaxis.set_ticks([])

    subfigs[0].suptitle('Self-gating', fontsize=fontsize, weight='bold')
    subfigs[1].suptitle('Navigator', fontsize=fontsize, weight='bold')

    ax0[0][0].set_ylabel('w/o motion', fontsize=fontsize, weight='bold')
    ax0[1][0].set_ylabel('w motion', fontsize=fontsize, weight='bold')

    ax1[0][0].set_ylabel('w/o motion', fontsize=fontsize, weight='bold', color='w')
    ax1[1][0].set_ylabel('w motion', fontsize=fontsize, weight='bold', color='w')


    plt.savefig(DIR + '/0.7mm_dwi_' + met + '.png',
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
