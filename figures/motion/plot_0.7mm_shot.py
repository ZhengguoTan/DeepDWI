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
f = h5py.File(DATA_DIR + '/0.7mm_21-dir_R2x2_MUSE_PHASE-IMAG_slice_000.h5', 'r')
PHS_IMAG = f['nav_shot_muse_phase'][:]
f.close()

f = h5py.File(DATA_DIR + '/0.7mm_21-dir_R2x2_MUSE_PHASE-NAVI_slice_000.h5', 'r')
PHS_NAVI = f['nav_shot_muse_phase'][:]
DWI_NAVI = f['dwi_comb_muse'][:]
f.close()

print('> PHS_IMAG shape: ', PHS_IMAG.shape)
print('> PHS_NAVI shape: ', PHS_NAVI.shape)

N_diff, N_shot, _, N_z, N_y, N_x = PHS_NAVI.shape

slice_idx = 1

b0 = np.squeeze(DWI_NAVI)
b0 = abs(b0[0, slice_idx, :, :])  # b0
mask = b0 > np.max(b0) * 0.002

# %%
N_row = 2
N_col = 6

fig = plt.figure(constrained_layout=True, figsize=(N_col*4, N_row*4+0.5))
subfigs = fig.subfigures(1, 3, wspace=0.01, width_ratios=[0.49, 0.49, 0.01])

fontsize = 9

fig_size = fig.get_size_inches()
fig_width = fig_size[0]
fontsize = fontsize * (fig_width / 6)

diff_idx_motion0 = 19
diff_idx_motion1 = 11

ax0 = subfigs[0].subplots(2, 3)
ax1 = subfigs[1].subplots(2, 3)

for s in range(N_shot):

    # self-gating
    img = np.angle(PHS_IMAG[diff_idx_motion0, s, 0, slice_idx, :, :]) * mask
    pcm = ax0[0][s].imshow(np.flip(img, axis=-2),
                cmap='RdBu_r', interpolation=None,
                vmin=-np.pi, vmax=np.pi)

    img = np.angle(PHS_IMAG[diff_idx_motion1, s, 0, slice_idx, :, :]) * mask
    pcm = ax0[1][s].imshow(np.flip(img, axis=-2),
                cmap='RdBu_r', interpolation=None,
                vmin=-np.pi, vmax=np.pi)

    ax0[0][s].set_title('shot #' + str(s+1), fontsize=fontsize-4)

    # navigator
    img = np.angle(PHS_NAVI[diff_idx_motion0, s, 0, slice_idx, :, :]) * mask
    pcm = ax1[0][s].imshow(np.flip(img, axis=-2),
                    cmap='RdBu_r', interpolation=None,
                    vmin=-np.pi, vmax=np.pi)

    img = np.angle(PHS_NAVI[diff_idx_motion1, s, 0, slice_idx, :, :]) * mask
    pcm = ax1[1][s].imshow(np.flip(img, axis=-2),
                    cmap='RdBu_r', interpolation=None,
                    vmin=-np.pi, vmax=np.pi)

    ax1[0][s].set_title('shot #' + str(s+1), fontsize=fontsize-4)

for m in range(2):
    for n in range(N_shot):
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


cbar_ax = fig.add_axes([1.0, 0.02, 0.01, 0.85])
cbar = subfigs[2].colorbar(pcm, cax=cbar_ax, ticks=[-3, 0, 3])
cbar.ax.tick_params(labelsize=24)


plt.savefig(DIR + '/0.7mm_shot_phase.png',
            bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()
