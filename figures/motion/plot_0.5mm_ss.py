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

recon_list = [
    '/examples/2024-03-06_zsssl_0.5x0.5x2.0mm_R3x2_kdat_slice_000_norm-kdat-1.0_ResNet2D_ResBlock-12_kernel-3_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_slice_000.h5',
    '/examples/2024-03-06_zsssl_0.5x0.5x2.0mm_R3x2_kdat_slice_000_norm-kdat-1.0_navi_ResNet2D_ResBlock-12_kernel-3_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_slice_000.h5']


# %%
f = h5py.File(HOME_DIR + recon_list[0], 'r')
DWI_SS_IMAG = np.squeeze(f['ZS'][:])
f.close()

f = h5py.File(HOME_DIR + recon_list[1], 'r')
DWI_SS_NAVI = np.squeeze(f['ZS'][:])
f.close()

print('> DWI_SS_IMAG shape: ', DWI_SS_IMAG.shape)
print('> DWI_SS_NAVI shape: ', DWI_SS_NAVI.shape)

N_diff, N_z, N_y, N_x = DWI_SS_NAVI.shape

# %%
N_row = 1
N_col = 2
fig, ax = plt.subplots(N_row, N_col, figsize=(N_col*4, N_row*4))
fontsize = 16

vmax = np.amax(abs(DWI_SS_NAVI)) * 0.15

diff_idx_motion0 = 2

# self-gating
ax[0].imshow(abs(np.flip(DWI_SS_IMAG[diff_idx_motion0, 1, :, :], axis=-2)),
                cmap='gray', interpolation=None,
                vmin=0, vmax=vmax)

# navigator
ax[1].imshow(abs(np.flip(DWI_SS_NAVI[diff_idx_motion0, 1, :, :], axis=-2)),
                cmap='gray', interpolation=None,
                vmin=0, vmax=vmax)

ax[1].text(0.70 * N_x, 0.96 * N_y,
              str(diff_idx_motion0).zfill(2) + '. diff',
              fontsize=fontsize, color='w')

ax[0].set_ylabel('w/o motion', fontsize=fontsize)

for n in range(N_col):
    ax[n].axes.xaxis.set_ticks([])
    ax[n].axes.yaxis.set_ticks([])

ax[0].set_title('self-gating', fontsize=fontsize)
ax[1].set_title('navigator', fontsize=fontsize)

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(DIR + '/motion_0.5mm_ss.png',
            bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()
