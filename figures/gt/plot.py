import h5py
import os

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import peak_signal_noise_ratio as psnr

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]
HOME_DIR = HOME_DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

DATA_DIR = HOME_DIR + '/data/'
print('> DATA: ', DATA_DIR)

props = dict(boxstyle='round', facecolor='black',
             edgecolor='black', linewidth=1, alpha=1.0)

# %%
# HOME_DIR + '/examples/2025-08-13_zsssl_1.0mm_21-dir_R1x3_kdat_slice_010_self_ResNet2D_PC0_reim-conv_ReLU_reim-acti_ResBlock-12_features-128_kernel-3_ADMM-12_lamda-0.050_rho-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_test_shot-retro-0.h5'
with h5py.File(DATA_DIR + '/gt_1.0mm_21-dir_R1x3_4shot_zsssl_test.h5', 'r') as f:
    DWI_ZS_4SHOT = f['DWI'][:]
print('ZS 4shot: ', DWI_ZS_4SHOT.shape)

# HOME_DIR + '/examples/2025-08-17_zsssl_1.0mm_21-dir_R1x3_kdat_slice_010_shot-retro-2_self_ResNet2D_PC0_reim-conv_ReLU_reim-acti_ResBlock-12_features-128_kernel-3_ADMM-12_lamda-0.050_rho-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_test_shot-retro-2.h5'
with h5py.File(DATA_DIR + '/gt_1.0mm_21-dir_R1x3_2shot_zsssl_test.h5', 'r') as f:
    DWI_ZS_2SHOT = f['DWI'][:]
print('ZS 2shot: ', DWI_ZS_2SHOT.shape)

# '/home/atuin/b143dc/b143dc15/Experiments/2023-09-26_Terra_Diffusion_iEPI/meas_MID00081_FID00082_Seg4_20_1p0iso/JETS.h5'
with h5py.File(DATA_DIR + '/gt_1.0mm_21-dir_R1x3_4shot_LLR.h5', 'r') as f:
    DWI_LR_4SHOT = f['DWI'][:]
print('LLR 4shot: ', DWI_LR_4SHOT.shape)

# '/home/atuin/b143dc/b143dc15/Experiments/2023-09-26_Terra_Diffusion_iEPI/meas_MID00081_FID00082_Seg4_20_1p0iso/JETS_2shot.h5'
with h5py.File(DATA_DIR + '/gt_1.0mm_21-dir_R1x3_2shot_LLR.h5', 'r') as f:
    DWI_LR_2SHOT = f['DWI'][:]
print('LLR 2shot: ', DWI_LR_2SHOT.shape)

# '/home/atuin/b143dc/b143dc15/Experiments/2023-09-26_Terra_Diffusion_iEPI/meas_MID00081_FID00082_Seg4_20_1p0iso/MUSE.h5'
with h5py.File(DATA_DIR + '/gt_1.0mm_21-dir_R1x3_4shot_MUSE.h5', 'r') as f:
    DWI_PI_4SHOT = f['DWI'][:]
print('MUSE 4shot: ', DWI_PI_4SHOT.shape)

# '/home/atuin/b143dc/b143dc15/Experiments/2023-09-26_Terra_Diffusion_iEPI/meas_MID00081_FID00082_Seg4_20_1p0iso/MUSE_2shot.h5'
with h5py.File(DATA_DIR + '/gt_1.0mm_21-dir_R1x3_2shot_MUSE.h5', 'r') as f:
    DWI_PI_2SHOT = f['DWI'][:]
print('MUSE 2shot: ', DWI_PI_2SHOT.shape)

# %%
DWI_ZS_4SHOT = np.flip(abs(DWI_ZS_4SHOT), axis=(-2))
DWI_ZS_2SHOT = np.flip(abs(DWI_ZS_2SHOT), axis=(-2))

DWI_LR_4SHOT = np.flip(abs(DWI_LR_4SHOT), axis=(-2))
DWI_LR_2SHOT = np.flip(abs(DWI_LR_2SHOT), axis=(-2))

DWI_PI_4SHOT = np.flip(abs(DWI_PI_4SHOT), axis=(-2))
DWI_PI_2SHOT = np.flip(abs(DWI_PI_2SHOT), axis=(-2))

N_diff, N_z, N_y, N_x = DWI_ZS_4SHOT.shape


# # #
DWI_ZS_4SHOT = sp.resize(DWI_ZS_4SHOT, (N_diff, N_z, int(N_y//1.2), int(N_x//1.2)))
DWI_ZS_2SHOT = sp.resize(DWI_ZS_2SHOT, (N_diff, N_z, int(N_y//1.2), int(N_x//1.2)))
DWI_LR_4SHOT = sp.resize(DWI_LR_4SHOT, (N_diff, N_z, int(N_y//1.2), int(N_x//1.2)))
DWI_LR_2SHOT = sp.resize(DWI_LR_2SHOT, (N_diff, N_z, int(N_y//1.2), int(N_x//1.2)))
DWI_PI_4SHOT = sp.resize(DWI_PI_4SHOT, (N_diff, N_z, int(N_y//1.2), int(N_x//1.2)))
DWI_PI_2SHOT = sp.resize(DWI_PI_2SHOT, (N_diff, N_z, int(N_y//1.2), int(N_x//1.2)))

N_y, N_x = DWI_ZS_4SHOT.shape[-2:]


diff_idx = 16
slice_idx = 62
scale = 0.65

N_row, N_col = 3, 2
fig, ax = plt.subplots(N_row, N_col, figsize=(N_col*4, N_row*4))


# %%
DWI_ZS_4SHOT_1 = DWI_ZS_4SHOT[:, slice_idx, :, :]
DWI_ZS_2SHOT_1 = DWI_ZS_2SHOT[:, slice_idx, :, :]

ssim_zs = ssim(DWI_ZS_4SHOT_1, DWI_ZS_2SHOT_1,
               data_range=DWI_ZS_2SHOT_1.max() - DWI_ZS_2SHOT_1.min())

mse_zs = nrmse(DWI_ZS_4SHOT_1, DWI_ZS_2SHOT_1)

psnr_zs = psnr(DWI_ZS_4SHOT_1, DWI_ZS_2SHOT_1,
               data_range=DWI_ZS_2SHOT_1.max() - DWI_ZS_2SHOT_1.min())

img = abs(DWI_ZS_4SHOT[diff_idx, slice_idx, :, :])
ax[0, 0].imshow(img, cmap='gray', interpolation=None,
                vmin=0, vmax=np.amax(img) * scale)
ax[0, 0].set_title('4-shot fully sampled', fontsize=18)
ax[0, 0].set_ylabel('ADMM Unrolling', fontsize=16)


img = abs(DWI_ZS_2SHOT[diff_idx, slice_idx, :, :])
ax[0, 1].imshow(img, cmap='gray', interpolation=None,
                vmin=0, vmax=np.amax(img) * scale)
ax[0, 1].set_title('retro. 2-shot', fontsize=18)

ax[0, 1].text(0.40 * N_x, 0.96 * N_y, '%5.3f, %6.3f'%(ssim_zs, psnr_zs),
              bbox=props, fontsize=16, color='w', weight='bold')


# %%
DWI_LR_4SHOT_1 = DWI_LR_4SHOT[:, slice_idx, :, :]
DWI_LR_2SHOT_1 = DWI_LR_2SHOT[:, slice_idx, :, :]

ssim_lr = ssim(DWI_LR_4SHOT_1, DWI_LR_2SHOT_1,
               data_range=DWI_LR_2SHOT_1.max() - DWI_LR_2SHOT_1.min())

mse_lr = nrmse(DWI_LR_4SHOT_1, DWI_LR_2SHOT_1)

psnr_lr = psnr(DWI_LR_4SHOT_1, DWI_LR_2SHOT_1,
               data_range=DWI_LR_2SHOT_1.max() - DWI_LR_2SHOT_1.min())

img = abs(DWI_LR_4SHOT[diff_idx, slice_idx, :, :])
ax[1, 0].imshow(img, cmap='gray', interpolation=None,
                vmin=0, vmax=np.amax(img) * scale)
ax[1, 0].set_ylabel('LLR', fontsize=16)

img = abs(DWI_LR_2SHOT[diff_idx, slice_idx, :, :])
ax[1, 1].imshow(img, cmap='gray', interpolation=None,
                vmin=0, vmax=np.amax(img) * scale)

ax[1, 1].text(0.40 * N_x, 0.96 * N_y, '%5.3f, %6.3f'%(ssim_lr, psnr_lr),
              bbox=props, fontsize=16, color='w', weight='bold')

# %%
DWI_PI_4SHOT_1 = DWI_PI_4SHOT[:, slice_idx, :, :]
DWI_PI_2SHOT_1 = DWI_PI_2SHOT[:, slice_idx, :, :]

ssim_pi = ssim(DWI_PI_4SHOT_1, DWI_PI_2SHOT_1,
               data_range=DWI_PI_2SHOT_1.max() - DWI_PI_2SHOT_1.min())

mse_pi = nrmse(DWI_PI_4SHOT_1, DWI_PI_2SHOT_1)

psnr_pi = psnr(DWI_PI_4SHOT_1, DWI_PI_2SHOT_1,
               data_range=DWI_PI_2SHOT_1.max() - DWI_PI_2SHOT_1.min())

img = abs(DWI_PI_4SHOT[diff_idx, slice_idx, :, :])
ax[2, 0].imshow(img, cmap='gray', interpolation=None,
                vmin=0, vmax=np.amax(img) * scale)
ax[2, 0].set_ylabel('MUSE', fontsize=16)

img = abs(DWI_PI_2SHOT[diff_idx, slice_idx, :, :])
ax[2, 1].imshow(img, cmap='gray', interpolation=None,
                vmin=0, vmax=np.amax(img) * scale)

ax[2, 1].text(0.40 * N_x, 0.96 * N_y, '%5.3f, %6.3f'%(ssim_pi, psnr_pi),
              bbox=props, fontsize=16, color='w', weight='bold')


print(' ssim: zs - %.3f, lr - %.3f, pi - %.3f'%(ssim_zs, ssim_lr, ssim_pi))
print('nrmse: zs - %.3f, lr - %.3f, pi - %.3f'%(mse_zs, mse_lr, mse_pi))
print(' psnr: zs - %.3f, lr - %.3f, pi - %.3f'%(psnr_zs, psnr_lr, psnr_pi))

# %%
for m in range(N_row):
    for n in range(N_col):
        ax[m,n].set_xticks([])
        ax[m,n].set_yticks([])

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(DIR + '/gt.png',
            bbox_inches='tight', pad_inches=0, dpi=300)
