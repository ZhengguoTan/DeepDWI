import h5py
import os

import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))

# %%
N_FA = 3
N_echo = 6

f = h5py.File(DIR + '/maple.h5', 'r')
R = f['ZS'][:]
f.close()

R = np.squeeze(R)
N_contrast, N_y, N_x = R.shape

R = np.reshape(R, [N_FA, N_echo, N_y, N_x])
R = np.swapaxes(R, -1, -2)

print('R shape', R.shape)

vmax = np.max(abs(R)) * 0.6

fig, ax = plt.subplots(N_FA, N_echo, figsize=(N_echo*3.4, N_FA*4))

for m in range(N_FA):
    for n in range(N_echo):

        ax[m][n].imshow(abs(R[m, n]), cmap='gray', vmin=0, vmax=vmax)
        ax[m][n].set_axis_off()
        ax[m][n].text(0.36*N_y, 0.97*N_x,
                      str(m+1) + '. FA ' + str(n+1) + '. Echo',
                      color='w', fontsize=16)

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(DIR + '/recon_maple.png',
            bbox_inches='tight', pad_inches=0, dpi=300)