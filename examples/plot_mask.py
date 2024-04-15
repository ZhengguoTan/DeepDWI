import h5py
import os
import pathlib

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.realpath(__file__))
print('>> file directory: ', DIR)

OUT_DIR = DIR + '/mask'
# make a new directory if not exist
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# %%
f = h5py.File(DIR + '/mask.h5', 'r')
train_mask = f['train'][:]
lossf_mask = f['lossf'][:]
f.close()

print('> train_mask shape: ', train_mask.shape)
print('> lossf_mask shape: ', lossf_mask.shape)


N_rep, N_diff, N_shot, N_coil, N_z, N_y, N_x = train_mask.shape

diff_ind = 2
shot_ind = 0

for r in range(N_rep):
    print('> ' + str(r).zfill(3))

    t_mask = train_mask[r, diff_ind, shot_ind, 0, 0, :, :]
    l_mask = lossf_mask[r, diff_ind, shot_ind, 0, 0, :, :]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(abs(t_mask), cmap='gray')
    ax[1].imshow(abs(l_mask), cmap='gray')

    for n in range(2):
        ax[n].set_axis_off()

    ax[0].text(0.02 * N_x, 0.08 * N_y, 'train',
               color='w', fontsize=16)

    ax[1].text(0.02 * N_x, 0.08 * N_y, 'lossf',
               color='w', fontsize=16)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax[1].text(0.50 * N_x, 0.97 * N_y,
               str(r).zfill(3) + 'th repeat',
               color='w', fontsize=16, bbox=props)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(OUT_DIR + '/mask_diff_' + str(diff_ind).zfill(3) + '_shot_' + str(shot_ind) + '_repeat_' + str(r).zfill(3) + '.png',
                bbox_inches='tight', pad_inches=0, dpi=300)