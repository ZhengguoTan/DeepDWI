import h5py
import os
import yaml

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0].rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

# %% load images
f = h5py.File(DIR + '/llr_regu.h5', 'r')
DWI_MUSE = f['MUSE'][:]
DWI_LLR = np.squeeze(f['LLR'][:])
f.close()

f = h5py.File(DIR + '/vae_regu.h5', 'r')
DWI_VAE = np.squeeze(f['VAE'][:])
f.close()

print('> MUSE shape: ', DWI_MUSE.shape)
print('> LLR shape: ', DWI_LLR.shape)
print('> VAE shape: ', DWI_VAE.shape)

# read in zsssl
with open(DIR + '/config_zsssl.yaml', 'r') as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)

test_conf = config_dict.get('test', {})
checkpoint_dir = test_conf['checkpoint']
ZSSSL_DIR = checkpoint_dir.rsplit('/', 1)[0]

data_conf = config_dict.get('data', {})
slice_idx = data_conf['slice_idx']

f = h5py.File(HOME_DIR + ZSSSL_DIR + '/zsssl_slice_' + str(slice_idx).zfill(3) + '.h5')
DWI_ZSSSL = np.squeeze(f['ZS'][:])
f.close()

print('> ZS shape: ', DWI_ZSSSL.shape)

N_diff, N_z, N_y, N_x = DWI_MUSE.shape

# %%
def normalize_image(input):
    output = abs(input)
    output = np.flip(output, axis=[-2])
    return output / (np.amax(output) - np.amin(output))

# %% plot
diff_idx = [2, 16]
disp_order = ['MUSE', 'LLR', 'VAE', 'ZSSSL']

N_row = len(diff_idx)
N_col = len(disp_order)
fig, ax = plt.subplots(N_row, N_col, figsize=(N_col*4, N_row*4))


slice_idx = 1

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

for m in range(N_row):
    for n in range(N_col):

        if disp_order[n] == 'MUSE':
            disp_image = normalize_image(DWI_MUSE[diff_idx[m], slice_idx])
        elif disp_order[n] == 'LLR':
            disp_image = normalize_image(DWI_LLR[diff_idx[m], slice_idx])
        elif disp_order[n] == 'VAE':
            disp_image = normalize_image(DWI_VAE[diff_idx[m], slice_idx])
        elif disp_order[n] == 'ZSSSL':
            disp_image = normalize_image(DWI_ZSSSL[diff_idx[m], slice_idx])

        ax[m,n].imshow(disp_image, cmap='gray', vmin=0, vmax=0.7,
                interpolation=None)

        if m == 0:
            ax[m,n].text(0.03*N_x, 0.08*N_y, disp_order[n], bbox=props,
                    color='w', fontsize=16)

        # ax[n].set_axis_off()
        ax[m,n].set_xticks([])
        ax[m,n].set_yticks([])


        if n == 0:
            ax[m,n].set_ylabel(str(diff_idx[m]) + '. diffusion direction',
                               fontsize=16)

plt.suptitle('1.0 mm ISO 4-shot fully-sampled iEPI', fontsize=16, fontweight='bold')
plt.subplots_adjust(wspace=0, hspace=0.01)
plt.savefig(DIR + '/regularizations.png',
            bbox_inches='tight', pad_inches=0.05, dpi=300)
plt.close()