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
f = h5py.File(HOME_DIR + '/examples/2024-09-20_zsssl_0.7mm_21-dir_R2x2_vol3_scan1_kdat_slice_000_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_test_shot-retro-0.h5', 'r')
DWI_ZS_SELF_One4All = f['DWI'][:]
f.close()

print('DWI_ZS_SELF_One4All shape: ', DWI_ZS_SELF_One4All.shape)

f = h5py.File(HOME_DIR + '/examples/2024-09-24_zsssl_0.7mm_21-dir_R2x2_vol3_scan1_kdat_slice_001_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_slice_001.h5', 'r')
DWI_ZS_SELF_One4One = np.squeeze(f['ZS'][:])
f.close()

print('DWI_ZS_SELF_One4One shape: ', DWI_ZS_SELF_One4One.shape)

N_diff = DWI_ZS_SELF_One4One.shape[-4]
N_y, N_x = DWI_ZS_SELF_One4One.shape[-2:]

# %%
props = dict(boxstyle='round', facecolor='black',
             edgecolor='black', linewidth=1, alpha=1.0)

N_row, N_col = 1, 3
fig, ax = plt.subplots(N_row, N_col, figsize=(N_col*4, N_row*4))

fontsize = 6

fig_size = fig.get_size_inches()
fig_width = fig_size[0]
fontsize = fontsize * (fig_width / 6)

scale = np.amax(abs(DWI_ZS_SELF_One4One[:, 1, :, :])) * 0.06
diff_idx = 10

img1 = np.flip(abs(DWI_ZS_SELF_One4One[diff_idx, 1, :, :]), axis=(-2))
ax[0].imshow(img1, cmap='gray',
            interpolation=None, vmin=0, vmax=scale)

ax[0].text(0.03*N_x, 0.06*N_x, "Slice-by-Slice Training", bbox=props,
           color='w', fontsize=fontsize, weight='bold')

rect_x0 = int(N_x//1.78)
rect_y0 = int(N_y*0.62)
rect_w = int(N_x*0.03)

Rect = Rectangle((rect_x0, rect_y0), rect_w, rect_w, edgecolor='y',
                 facecolor='none', linewidth=2)
ax[0].add_patch(Rect)


img2 = np.flip(abs(DWI_ZS_SELF_One4All[diff_idx, 78, :, :]), axis=(-2))
ax[1].imshow(img2, cmap='gray',
            interpolation=None, vmin=0, vmax=scale)

ax[1].text(0.03*N_x, 0.06*N_x, "Single-Slice Traning", bbox=props,
           color='w', fontsize=fontsize, weight='bold')

Rect = Rectangle((rect_x0, rect_y0), rect_w, rect_w, edgecolor='g',
                 facecolor='none', linewidth=2)
ax[1].add_patch(Rect)

ax[2].imshow(abs(img1 - img2)*5, cmap='gray',
                interpolation=None, vmin=0, vmax=scale)

ax[2].text(0.03*N_x, 0.06*N_x, "Diff X5", bbox=props,
           color='w', fontsize=fontsize, weight='bold')


for m in range(N_col):
    ax[m].set_axis_off()

plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(DIR + '/training.png',
            bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()


# %%
plt.style.use('dark_background')
fig, ax = plt.subplots(1, 1, figsize=(N_col*4, N_row*4))

roi_One4One = np.flip(abs(DWI_ZS_SELF_One4One[1:, 1, :, :]), axis=(-2))
roi_One4One = roi_One4One[:, rect_y0 : rect_y0+rect_w, rect_x0 : rect_x0+rect_w]
roi_One4One = roi_One4One.reshape((N_diff-1, -1)) * 1e5

ave1 = np.squeeze(np.mean(roi_One4One, axis=1))
std1 = np.squeeze(np.std(roi_One4One, axis=1))

print(roi_One4One.shape, ave1.shape, std1.shape)

roi_One4All = np.flip(abs(DWI_ZS_SELF_One4All[1:, 78, :, :]), axis=(-2))
roi_One4All = roi_One4All[:, rect_y0 : rect_y0+rect_w, rect_x0 : rect_x0+rect_w]
roi_One4All = roi_One4All.reshape((N_diff-1, -1)) * 1e5

ave2 = np.squeeze(np.mean(roi_One4All, axis=1))
std2 = np.squeeze(np.std(roi_One4All, axis=1))

x = np.arange(1, N_diff, 1)


plt.errorbar(x, ave1, yerr=std1, color='y', fmt='-o', capsize=6, linewidth=3)
plt.errorbar(x, ave2, yerr=std2, color='g', fmt='-o', capsize=6, linewidth=3)

plt.xlim([0, N_diff])
plt.xticks(x)
plt.yticks(np.arange(1, 5, 1))
plt.xlabel('Diffusion Encoding', fontsize=16)
plt.ylabel('Signal (a.u.)', fontsize=16)

plt.tick_params(axis='both', which='major', labelsize=15)

plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(DIR + '/training_line.png',
            bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()
