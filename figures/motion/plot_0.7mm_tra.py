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
N_row = 2
N_col = 4

fig = plt.figure(constrained_layout=True, figsize=(N_col*4, N_row*4+0.7))
subfigs = fig.subfigures(1, 2, wspace=0.01, width_ratios=[0.50, 0.50])

fontsize = 9

fig_size = fig.get_size_inches()
fig_width = fig_size[0]
fontsize = fontsize * (fig_width / 6)

vmax = np.amax(abs(DWI_JETS_NAVI)) * 0.04

slice_idx = 96

diff_idx_motion0 = 19
diff_idx_motion1 =  5

ax0 = subfigs[0].subplots(2, 2)
ax1 = subfigs[1].subplots(2, 2)


# navigator
ax0[0][0].imshow(abs(np.flip(DWI_JETS_NAVI[diff_idx_motion0, slice_idx, :, :], axis=-2)),
                cmap='gray', interpolation=None,
                vmin=0, vmax=vmax)

ax0[0][1].imshow(np.angle(np.flip(DWI_JETS_NAVI[diff_idx_motion0, slice_idx, :, :], axis=-2)),
                cmap='RdBu_r', interpolation=None,
                vmin=-np.pi, vmax=np.pi)

ax0[0][0].annotate("", xy=(0.65*N_x, 0.83*N_y), xytext=(0.70*N_x, 0.95*N_y),
                   arrowprops=dict(arrowstyle="->", color='r', linewidth=3,
                                   mutation_scale=25))

ax0[0][0].set_title('Magnitude', fontsize=fontsize-4)
ax0[0][1].set_title('Phase', fontsize=fontsize-4)


ax0[1][0].imshow(abs(np.flip(DWI_JETS_NAVI[diff_idx_motion1, slice_idx, :, :], axis=-2)),
                cmap='gray', interpolation=None,
                vmin=0, vmax=vmax)

nav = ax0[1][1].imshow(np.angle(np.flip(DWI_JETS_NAVI[diff_idx_motion1, slice_idx, :, :], axis=-2)),
                cmap='RdBu_r', interpolation=None,
                vmin=-np.pi, vmax=np.pi)

cbar = fig.colorbar(nav, ax=ax0[1][1], location='bottom',
                    ticks=[-3, 0, 3], shrink=0.8)
cbar.ax.tick_params(labelsize=fontsize)

# self-gating
ax1[0][0].imshow(abs(np.flip(DWI_ZSSSL_SELF[diff_idx_motion0, slice_idx, :, :], axis=-2)),
                cmap='gray', interpolation=None,
                vmin=0, vmax=vmax)

ax1[0][1].imshow(np.angle(np.flip(DWI_ZSSSL_SELF[diff_idx_motion0, slice_idx, :, :], axis=-2)),
                cmap='RdBu_r', interpolation=None,
                vmin=-np.pi, vmax=np.pi)

ax1[0][0].annotate("", xy=(0.65*N_x, 0.83*N_y), xytext=(0.70*N_x, 0.95*N_y),
                   arrowprops=dict(arrowstyle="->", color='r', linewidth=3,
                                   mutation_scale=25))

ax1[0][0].set_title('Magnitude', fontsize=fontsize-4)
ax1[0][1].set_title('Phase', fontsize=fontsize-4)


ax1[1][0].imshow(abs(np.flip(DWI_ZSSSL_SELF[diff_idx_motion1, slice_idx, :, :], axis=-2)),
                cmap='gray', interpolation=None,
                vmin=0, vmax=vmax)

sel = ax1[1][1].imshow(np.angle(np.flip(DWI_ZSSSL_SELF[diff_idx_motion1, slice_idx, :, :], axis=-2)),
                cmap='RdBu_r', interpolation=None,
                vmin=-np.pi, vmax=np.pi)

cbar = fig.colorbar(sel, ax=ax1[1][1], location='bottom',
                    ticks=[-3, 0, 3], shrink=0.8)
cbar.ax.tick_params(labelsize=fontsize)



for m in range(2):
    for n in range(2):
        ax0[m][n].axes.xaxis.set_ticks([])
        ax0[m][n].axes.yaxis.set_ticks([])

        ax1[m][n].axes.xaxis.set_ticks([])
        ax1[m][n].axes.yaxis.set_ticks([])

subfigs[0].suptitle('Navigated LLR', fontsize=fontsize, weight='bold')
subfigs[1].suptitle('Self-Gated ZSSSL', fontsize=fontsize, weight='bold')

ax0[0][0].set_ylabel('w/o motion', fontsize=fontsize, weight='bold')
ax0[1][0].set_ylabel('w motion', fontsize=fontsize, weight='bold')

ax1[0][0].set_ylabel('w/o motion', fontsize=fontsize, weight='bold', color='w')
ax1[1][0].set_ylabel('w motion', fontsize=fontsize, weight='bold', color='w')


plt.savefig(DIR + '/0.7mm_dwi_tra.png',
            bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()
