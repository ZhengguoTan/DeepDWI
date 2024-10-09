import h5py
import os

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]
HOME_DIR = HOME_DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

DATA_DIR = HOME_DIR + '/data/'
print('> DATA: ', DATA_DIR)

# %%

f = h5py.File(DATA_DIR + '/0.7mm_21-dir_R2x2_vol1_scan1_JETS_NAVI.h5', 'r')
DWI_JETS_NAVI = f['DWI'][:]
f.close()

f = h5py.File(DATA_DIR + '/0.7mm_21-dir_R2x2_vol1_scan1_JETS_IMAG.h5', 'r')
DWI_JETS_SELF = f['DWI'][:]
f.close()

f = h5py.File(HOME_DIR + '/examples/2024-05-22_zsssl_0.7mm_21-dir_R2x2_vol1_scan1_kdat_slice_040_norm-kdat-1.0_navi_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_test_shot-retro-0.h5', 'r')
DWI_ZSSSL_NAVI = f['DWI'][:]
f.close()

f = h5py.File(HOME_DIR + '/examples/2024-05-23_zsssl_0.7mm_21-dir_R2x2_vol1_scan1_kdat_slice_040_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_test_shot-retro-0.h5', 'r')
DWI_ZSSSL_SELF = f['DWI'][:]
f.close()

print('> DWI_JETS_NAVI shape: ', DWI_JETS_NAVI.shape)
print('> DWI_JETS_SELF shape: ', DWI_JETS_SELF.shape)
print('> DWI_ZSSSL_NAVI shape: ', DWI_ZSSSL_NAVI.shape)
print('> DWI_ZSSSL_SELF shape: ', DWI_ZSSSL_SELF.shape)


N_diff, N_z, N_y, N_x = DWI_JETS_NAVI.shape

# %% axial
N_row, N_col = 2, 3

fig, ax = plt.subplots(N_row, N_col, figsize=(N_col*4, N_row*4))

fontsize = 9

fig_size = fig.get_size_inches()
fig_width = fig_size[0]
fontsize = fontsize * (fig_width / 6)

vmax = np.amax(abs(DWI_JETS_NAVI)) * 0.04

slice_idx = 89

diff_idx_motion0 = 19
diff_idx_motion1 = 11


# motion 1
ax[0][0].imshow(abs(np.flip(DWI_JETS_NAVI[diff_idx_motion1, slice_idx, :, :], axis=-2)),
                cmap='gray', interpolation=None,
                vmin=0, vmax=vmax)

ax[0][1].imshow(abs(np.flip(DWI_JETS_SELF[diff_idx_motion1, slice_idx, :, :], axis=-2)),
                cmap='gray', interpolation=None,
                vmin=0, vmax=vmax)

ax[0][0].set_title('Navigated', fontsize=fontsize, weight='bold')
ax[0][1].set_title('Self-Gated', fontsize=fontsize, weight='bold')


ax[1][0].imshow(abs(np.flip(DWI_ZSSSL_NAVI[diff_idx_motion1, slice_idx, :, :], axis=-2)),
                cmap='gray', interpolation=None,
                vmin=0, vmax=vmax)

ax[1][1].imshow(abs(np.flip(DWI_ZSSSL_SELF[diff_idx_motion1, slice_idx, :, :], axis=-2)),
                cmap='gray', interpolation=None,
                vmin=0, vmax=vmax)


rect_x0 = int(N_x//2.3)
rect_y0 = int(N_y//2.8)

rect_w = 100


Rect = Rectangle((rect_x0, rect_y0), rect_w, rect_w, edgecolor='y', facecolor='none', linewidth=2)
ax[0][1].add_patch(Rect)

Rect = Rectangle((rect_x0, rect_y0), rect_w, rect_w, edgecolor='y', facecolor='none', linewidth=2)
ax[1][1].add_patch(Rect)


img = abs(np.flip(DWI_JETS_SELF[diff_idx_motion1, slice_idx, :, :], axis=-2))
ax[0][2].imshow(img[rect_y0 : rect_y0+rect_w, rect_x0 : rect_x0+rect_w],
                cmap='gray', interpolation=None,
                vmin=0, vmax=vmax)

ax[0][2].annotate("", xy=(0.10*rect_w, 0.58*rect_w), xytext=(0.02*rect_w, 0.64*rect_w),
                   arrowprops=dict(arrowstyle="->", color='r', linewidth=3,
                                   mutation_scale=25))

img = abs(np.flip(DWI_ZSSSL_SELF[diff_idx_motion1, slice_idx, :, :], axis=-2))
ax[1][2].imshow(img[rect_y0 : rect_y0+rect_w, rect_x0 : rect_x0+rect_w],
                cmap='gray', interpolation=None,
                vmin=0, vmax=vmax)

ax[1][2].annotate("", xy=(0.10*rect_w, 0.58*rect_w), xytext=(0.02*rect_w, 0.64*rect_w),
                   arrowprops=dict(arrowstyle="->", color='r', linewidth=3,
                                   mutation_scale=25))


for m in range(N_row):
    for n in range(N_col):
        ax[m][n].axes.xaxis.set_ticks([])
        ax[m][n].axes.yaxis.set_ticks([])

        ax[m][n].axes.xaxis.set_ticks([])
        ax[m][n].axes.yaxis.set_ticks([])

ax[0][0].set_ylabel('LLR', fontsize=fontsize, weight='bold')
ax[1][0].set_ylabel('ADMM Unrolling', fontsize=fontsize, weight='bold')

plt.subplots_adjust(wspace=0, hspace=0.01)
plt.savefig(DIR + '/0.7mm_dwi_tra_4.png',
            bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()
