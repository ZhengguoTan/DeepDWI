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
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
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

    diff_idx = 11

    # axial
    tra_slice_idx = 109
    img = np.flip(abs(DWI[diff_idx, tra_slice_idx, :, :]), axis=(-2))
    ax[0].imshow(img, cmap='gray',
                 interpolation=None, vmin=0, vmax=np.amax(img)*0.4)

    # coronal
    cor_slice_idx = 124
    img = np.flip(abs(DWI[diff_idx, (N_z - int(N_x/2)):, cor_slice_idx, :]), axis=(-2))
    ax[1].imshow(img, cmap='gray',
                 interpolation=None, vmin=0, vmax=np.amax(img)*0.4)

    # sagittal
    sag_slice_idx = 150
    img = np.flip(abs(DWI[diff_idx, (N_z - int(N_x/2)):, :, sag_slice_idx]), axis=(-2))

    ax[2].imshow(img, cmap='gray',
                 interpolation=None, vmin=0, vmax=np.amax(img)*0.4)


    for n in range(3):
        ax[n].axes.xaxis.set_ticks([])
        ax[n].axes.yaxis.set_ticks([])
    

    if met == 'muse':
        ax[0].text(0.02*N_x, 0.06*N_x, 'Axial', bbox=props,
                   color='w', fontsize=fontsize)
        ax[1].text(0.02*N_x, 0.06*N_x, 'Coronal', bbox=props,
                   color='w', fontsize=fontsize)
        ax[2].text(0.02*N_x, 0.06*N_x, 'Sagittal', bbox=props,
                   color='w', fontsize=fontsize)


    # fig.suptitle('Self-Gated ' + met.upper(), fontsize=fontsize, weight='bold')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig(DIR + '/0.7mm_dwi_sg_' + met + '.png',
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()