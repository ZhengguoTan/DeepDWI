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
f = h5py.File('/home/atuin/b143dc/b143dc15/Experiments/2023-10-18_Terra_Diffusion_iEPI/meas_MID00124_FID00287_ep2d_diff_20dir_MB2/JETS_PHASE-NAVI-REDU.h5', 'r')
DWI_JETS_NAVI = f['DWI'][:]
f.close()

print('DWI_JETS_NAVI shape: ', DWI_JETS_NAVI.shape)


f = h5py.File('/home/atuin/b143dc/b143dc15/Softwares/DeepDWI/examples/2024-05-23_zsssl_0.7mm_21-dir_R2x2_vol1_scan1_kdat_slice_040_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_test_shot-retro-0.h5', 'r')
DWI_ZS_SELF = f['DWI'][:]
f.close()

print('DWI_ZS_SELF shape: ', DWI_ZS_SELF.shape)


N_diff, N_z, N_y, N_x = DWI_JETS_NAVI.shape

# %%
N_row, N_col = 1, 2

f, ax = plt.subplots(N_row, N_col, figsize=(N_col*4, N_row*4*(N_z/N_x)))

diff_idx = 11

cor_slice_idx = 124
sag_slice_idx = 150

fontsize = 12

# cor
img = abs(np.flip(DWI_JETS_NAVI[diff_idx, :, cor_slice_idx, :], axis=-2))
ax[0].imshow(img,
                cmap='gray', interpolation=None,
                vmin=0, vmax=np.amax(img)*0.6)

img = abs(np.flip(DWI_ZS_SELF[diff_idx, :, cor_slice_idx, :], axis=-2))
ax[1].imshow(img,
                cmap='gray', interpolation=None,
                vmin=0, vmax=np.amax(img)*0.6)


for n in range(N_col):
    ax[n].axes.xaxis.set_ticks([])
    ax[n].axes.yaxis.set_ticks([])


ax[0].set_title('Navigated LLR', fontsize=fontsize, weight='bold')
ax[1].set_title('Self-Gated ZSSSL', fontsize=fontsize, weight='bold')

ax[0].set_ylabel('Coronal', fontsize=fontsize, weight='bold')

plt.subplots_adjust(wspace=0, hspace=0.01)
plt.savefig(DIR + '/0.7mm_dwi_cor.png',
            bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()
