import h5py
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp

from matplotlib.patches import Rectangle

import os
DIR = os.path.dirname(os.path.realpath(__file__))
print('> DIR: ', DIR)

# %% read in RGB
fit_files = ['MUSE_DENOISE_fit.h5',
             'JETS_fit.h5',
             'ZSSSL_fit.h5']

title_str = ['MUSE + Denoiser', 'LLR', 'ADMM Unrolling']

RGB_arrays = []
for fit_file in fit_files:

    with h5py.File(DIR + '/' + fit_file, 'r') as f:
        RGB = f['RGB'][:]
        RGB = RGB[:, 50:, ...]
        N_c, N_z, N_y, N_x = RGB.shape
        N_r = int(min(N_y, N_x) * 0.80)
        RGB = sp.resize(RGB, oshape=(N_c, N_z, N_r, N_r)).T
        RGB = np.swapaxes(RGB, 0, 1)
        print('> RGB shape: ', RGB.shape)
        RGB_arrays.append(RGB)

# %% plot
N_x, N_y, N_z, N_c = RGB.shape

tra_slice_idx = [29, 42]
# cor_slice_idx = 111
# sag_slice_idx = 137

N_row, N_col = len(tra_slice_idx), 3
fig, ax = plt.subplots(N_row, N_col, figsize=(N_col*4, N_row*4))

for n in range(N_col):

    RGB = RGB_arrays[n]

    for m in range(N_row):

        # transverse
        dsp = np.flip(RGB[:, :, tra_slice_idx[m], :], axis=0)
        ax[m,n].imshow(dsp, interpolation=None, vmin=0, vmax=1)

        # title
        if m == 0:
            ax[m,n].set_title(title_str[n], fontsize=16,
                              weight='bold')
        # arrow
        if m == 0:
            ax[m,n].annotate("", xy=(0.20*N_x, 0.30*N_y),
                             xytext=(0.20*N_x, 0.46*N_y),
                             arrowprops=dict(arrowstyle="->",
                                             color='#FFCB05',
                                             linewidth=4,
                                             mutation_scale=25)
                            )

        else:
            ax[m,n].annotate("", xy=(0.75*N_x, 0.22*N_y),
                             xytext=(0.80*N_x, 0.37*N_y),
                             arrowprops=dict(arrowstyle="->",
                                             color='#FFCB05',
                                             linewidth=4,
                                             mutation_scale=25)
                            )

        ax[m,n].set_axis_off()

plt.subplots_adjust(wspace=0.01, hspace=0.04)
plt.savefig(DIR + '/dti.png',
            bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()
