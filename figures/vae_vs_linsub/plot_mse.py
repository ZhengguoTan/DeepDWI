
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))

plt.rcParams['font.family'] = 'monospace'

# %% define helper functions

def get_indices_of_noise(noiseParam):
    indices_sorted_noise = []
    for noiseValue in range(int(max(noiseParam))+1):
        indices = np.argwhere(noiseParam == noiseValue).flatten()
        indices_sorted_noise.append(indices)
    return indices_sorted_noise

# Calculate MSE for each noise distribution
def get_mse_of_sorted_noise(indices_sorted_by_noise, fReconEpochs, fCleanEpochs):
    mse_over_noise = []
    sample_amount = []
    for noise_parameter in range(len(indices_sorted_by_noise)):
        mse = np.mean((fCleanEpochs[indices_sorted_by_noise[noise_parameter]] - fReconEpochs[indices_sorted_by_noise[noise_parameter]])**2)
        mse_over_noise.append(mse)
        sample_amount.append(len(indices_sorted_by_noise[noise_parameter]))
    return mse_over_noise, sample_amount


# %%

with open(DIR + '/mse.pkl', 'rb') as f:
    sample_struct = pickle.load(f)

indices_sorted_by_noise = get_indices_of_noise(np.squeeze(sample_struct['noise_amount'][:]))

print(len(indices_sorted_by_noise))

mses, sample_amount = get_mse_of_sorted_noise(indices_sorted_by_noise, np.squeeze(sample_struct['recon_VAE'][:]), np.squeeze(sample_struct['clean_samples'][:]))
mse_linsub, sample_amount2 = get_mse_of_sorted_noise(indices_sorted_by_noise, np.squeeze(sample_struct['recon_linsub'][:]), np.squeeze(sample_struct['clean_samples'][:]))
mse_noisy, _ = get_mse_of_sorted_noise(indices_sorted_by_noise, np.squeeze(sample_struct['noised_samples'][:]), np.squeeze(sample_struct['clean_samples'][:]))

plt.bar(np.arange(len(indices_sorted_by_noise[3:]))+3,
        mse_noisy[3:],
        label= f'Noisy',  alpha=0.7)
plt.bar(np.arange(len(indices_sorted_by_noise[3:]))+3,
        mse_linsub[3:],
        label= f'SVD recon',  alpha=0.7)
plt.bar(np.arange(len(indices_sorted_by_noise[3:]))+3,
        mses[3:],
        label= f'VAE recon',  alpha=0.7)

# plt.title('MSE over noise amount')
plt.xlabel('Noise Level')
plt.ylabel('MSE')
plt.legend()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(DIR + '/vae_vs_linsub_mse.png',
            bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.close()
