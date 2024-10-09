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
props = dict(boxstyle='round', facecolor='black',
             edgecolor='wheat', linewidth=1, alpha=1.0)

f = h5py.File(DATA_DIR + '0.7mm_21-dir_R2x2_vol3_scan1_JETS.h5', 'r')
DWI_JETS_SELF= f['DWI'][:]
f.close()

f = h5py.File(HOME_DIR + '/examples/2024-09-20_zsssl_0.7mm_21-dir_R2x2_vol3_scan1_kdat_slice_000_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_test_shot-retro-0.h5', 'r')
DWI_ZS_SELF = f['DWI'][:]
f.close()

N_diff, N_z, N_y, N_x = DWI_JETS_SELF.shape


# # #
N_row, N_col = 2, 4
fig, ax = plt.subplots(N_row, N_col, figsize=(N_col*4, N_row*4))

fontsize = 9

fig_size = fig.get_size_inches()
fig_width = fig_size[0]
fontsize = fontsize * (fig_width / 9)

tra_slice_idx = 77
diff_list = [1, 7, 13, 19]

cnt = 0
for d in diff_list:

    # axial
    img = abs(DWI_JETS_SELF[d, tra_slice_idx, :, :])
    img = np.flip(img, axis=(-2))
    ax[0, cnt].imshow(img, cmap='gray',interpolation=None,
                      vmin=0, vmax=np.amax(img)*0.5)

    img = abs(DWI_ZS_SELF[d, tra_slice_idx, :, :])
    img = np.flip(img, axis=(-2))
    ax[1, cnt].imshow(img, cmap='gray',interpolation=None,
                      vmin=0, vmax=np.amax(img)*0.5)

    ax[0, cnt].set_axis_off()
    ax[1, cnt].set_axis_off()

    if cnt == 0:
        ax[0, cnt].text(0.03*N_x, 0.08*N_x, 'LLR', bbox=props,
                        color='y', fontsize=fontsize, weight='bold')

        ax[1, cnt].text(0.03*N_x, 0.08*N_x, 'ADMM Unrolling', bbox=props,
                        color='y', fontsize=fontsize, weight='bold')

    cnt += 1

# fig.suptitle('0.7 mm DWI at four different directions', fontsize=16)

plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig(DIR + '/0.7mm_dwi_vol3.png',
            bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()
