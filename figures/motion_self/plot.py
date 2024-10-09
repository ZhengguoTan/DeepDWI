import argparse
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
parser = argparse.ArgumentParser(description='plot.')

parser.add_argument('--ave', action='store_true')

args = parser.parse_args()


if args.ave is True:
    ave_str = '_ave'
else:
    ave_str = ''

# %%
props = dict(boxstyle='round', facecolor='black',
             edgecolor='wheat', linewidth=1, alpha=1.0)

for met in ['muse', 'jets', 'zsssl']:

    if met == 'muse':
        f = h5py.File('/home/atuin/b143dc/b143dc15/Experiments/2024-06-10_Terra_Diffusion_iEPI/meas_MID00329_FID25293_ep2d_diff_ms_mddw_0_7mm_self/MUSE_PHASE-IMAG-REDU.h5', 'r')

    elif met == 'jets':
        f = h5py.File('/home/atuin/b143dc/b143dc15/Experiments/2024-06-10_Terra_Diffusion_iEPI/meas_MID00329_FID25293_ep2d_diff_ms_mddw_0_7mm_self/JETS_PHASE-IMAG-REDU.h5', 'r')

    elif met == 'zsssl':
        f = h5py.File('/home/atuin/b143dc/b143dc15/Softwares/DeepDWI/examples/2024-06-14_zsssl_0.7mm_21-dir_R2x2_vol2_scan2_kdat_slice_040_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_test_shot-retro-0.h5', 'r')

    DWI = f['DWI'][:]
    f.close()

    N_diff, N_z, N_y, N_x = DWI.shape


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

    diff_idx = 15

    # axial
    tra_slice_idx = 108
    if args.ave is True:
        img = np.mean(abs(DWI[1:, tra_slice_idx, :, :]), axis=0)
    else:
        img = abs(DWI[diff_idx, tra_slice_idx, :, :])
    img = np.flip(img, axis=(-2))
    ax[0].imshow(img, cmap='gray',
                 interpolation=None, vmin=0, vmax=np.amax(img)*0.25)

    # if args.ave is False and met != 'muse':
    #     ax[0].annotate("", xy=(0.60*N_x, 0.65*N_y), xytext=(0.70*N_x, 0.60*N_y),
    #                 arrowprops=dict(arrowstyle="->", color='r', linewidth=3,
    #                                 mutation_scale=15))
    #     ax[0].annotate("", xy=(0.36*N_x, 0.38*N_y), xytext=(0.26*N_x, 0.43*N_y),
    #                 arrowprops=dict(arrowstyle="->", color='r', linewidth=3,
    #                                 mutation_scale=15))

    # coronal
    cor_slice_idx = 124
    if args.ave is True:
        img = np.mean(abs(DWI[1:, (N_z - int(N_x/2)):, cor_slice_idx, :]), axis=0)
    else:
        img = abs(DWI[diff_idx, (N_z - int(N_x/2)):, cor_slice_idx, :])
    img = np.flip(img, axis=(-2))
    ax[1].imshow(img, cmap='gray',
                 interpolation=None, vmin=0, vmax=np.amax(img)*0.40)

    if met != 'muse':
        if args.ave is False:
            ax[1].annotate("", xy=(0.72*N_x, 0.28*N_x/2), xytext=(0.72*N_x, 0.08*N_x/2),
                        arrowprops=dict(arrowstyle="->", color='r', linewidth=3,
                                        mutation_scale=15))

            ax[1].annotate("", xy=(0.55*N_x, 0.47*N_x/2), xytext=(0.65*N_x, 0.47*N_x/2),
                        arrowprops=dict(arrowstyle="->", color='r', linewidth=3,
                                        mutation_scale=15))

        else:
            ax[1].annotate("", xy=(0.56*N_x, 0.68*N_x/2), xytext=(0.66*N_x, 0.68*N_x/2),
                        arrowprops=dict(arrowstyle="->", color='b', linewidth=3,
                                        mutation_scale=15))


    # sagittal
    sag_slice_idx = 150
    if args.ave is True:
        img = np.mean(abs(DWI[1:, (N_z - int(N_x/2)):, :, sag_slice_idx]), axis=0)
    else:
        img = abs(DWI[diff_idx, (N_z - int(N_x/2)):, :, sag_slice_idx])
    img = np.flip(img, axis=(-2))
    ax[2].imshow(img, cmap='gray',
                 interpolation=None, vmin=0, vmax=np.amax(img)*0.30)

    if met != 'muse':
        if args.ave is False:
            ax[2].annotate("", xy=(0.50*N_x, 0.35*N_x/2), xytext=(0.50*N_x, 0.15*N_x/2),
                        arrowprops=dict(arrowstyle="->", color='r', linewidth=3,
                                        mutation_scale=15))

        else:
            ax[2].annotate("", xy=(0.22*N_x, 0.88*N_x/2), xytext=(0.10*N_x, 0.98*N_x/2),
                        arrowprops=dict(arrowstyle="->", color='b', linewidth=3,
                                        mutation_scale=15))


    for n in range(3):
        ax[n].axes.xaxis.set_ticks([])
        ax[n].axes.yaxis.set_ticks([])


    if args.ave is False:
        if met == 'muse':
            met_str = 'Self-Gated MUSE'
        elif met == 'jets':
            met_str = 'Self-Gated LLR'
        elif met == 'zsssl':
            met_str = 'Self-Gated ADMM Unrolling'

        ax[0].text(0.03*N_x, 0.08*N_x, met_str, bbox=props,
                   color='y', fontsize=fontsize, weight='bold')

        # ax[0].text(0.02*N_x, 0.06*N_x, 'Axial', bbox=props,
        #            color='w', fontsize=fontsize)
        # ax[1].text(0.02*N_x, 0.06*N_x, 'Coronal', bbox=props,
        #            color='w', fontsize=fontsize)
        # ax[2].text(0.02*N_x, 0.06*N_x, 'Sagittal', bbox=props,
        #            color='w', fontsize=fontsize)


    # fig.suptitle('Self-Gated ' + met.upper(), fontsize=fontsize, weight='bold')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig(DIR + '/0.7mm_dwi_sg_' + met + ave_str + '.png',
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()