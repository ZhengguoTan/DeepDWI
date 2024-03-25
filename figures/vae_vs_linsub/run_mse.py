import copy
import h5py
import os
import pickle
import torch
import yaml

from deepdwi.models import bloch
from deepdwi.models import autoencoder as ae
from deepdwi.recons import linsub

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]
HOME_DIR = HOME_DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% read in and display the yaml config file
with open(DIR + '/config.yaml', 'r') as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)

data_conf = config_dict.get('data', {})
print('> data_conf: ')
print('    dvs: ', data_conf['dvs'])

model_conf = config_dict.get('model', {})
print('> model_conf: ')
print('    checkpoint: ', model_conf['checkpoint'])
print('    N_latent: ', model_conf['N_latent'])

N_latent = model_conf['N_latent']


# %%
def create_data_loader(x_clean, original_D, min_noise, max_noise, batch_size_train,sample_struct):

    noise_amount = np.zeros(shape=(1, x_clean.shape[1]))
    noise_amount_full = np.transpose(noise_amount)

    x_clean = np.transpose(x_clean)
    x_no_noise_yet = copy.deepcopy(x_clean)
    original_D = np.transpose(original_D)

    q_dataset = bloch.DwiDataset(x_clean, x_no_noise_yet, noise_amount_full)

    q_datalen = len(q_dataset)

    train_siz = int(q_datalen * 0.8)
    test_siz = q_datalen - train_siz

    train_set, test_set = data.random_split(q_dataset, [train_siz, test_siz]) #get_D with test_set

    train_set_noised, test_set_noised = create_noised_dataset(train_set, test_set, min_noise, max_noise)

    loader_train = data.DataLoader(train_set_noised, batch_size=batch_size_train, shuffle=True)

    loader_test = data.DataLoader(test_set_noised, batch_size=1, shuffle=False)

    test_samples_noised = []
    test_samples_clean = []
    test_samples_D = []
    test_samples_noise = []
    for i in range(len(test_set_noised)):
        sample = test_set_noised[i]  # Assuming each sample is a tuple
        test_samples_noised.append(sample[0])
        test_samples_clean.append(sample[1])
        # test_samples_D.append(sample[2])
        test_samples_noise.append(sample[2])

    sample_struct['noised_samples'] = test_samples_noised
    sample_struct['clean_samples'] = test_samples_clean
    # sample_struct['D_samples'] = test_samples_D
    sample_struct['noise_amount'] = test_samples_noise

    # sets_file = h5py.File('test_set.h5', 'w')

    # sets_file.create_dataset('noisy', data=np.array(test_samples_noised))
    # sets_file.create_dataset('clean', data=np.array(test_samples_clean))
    # sets_file.create_dataset('D', data=np.array(test_samples_D))
    # sets_file.create_dataset('noise', data=np.array(test_samples_noise))
    # sets_file.close()

    return loader_train, loader_test

def create_noised_dataset(train_set, test_set, min_noise, max_noise):

    train_set_full = copy.deepcopy(train_set)
    test_set_full = copy.deepcopy(test_set)


    for id in range(min_noise, max_noise):
        train_set_copy = copy.deepcopy(train_set)
        test_set_copy = copy.deepcopy(test_set)

        sd = 0.01 + id * 0.03

        for index in range(len(train_set_copy)):
            train_set_copy[index][0][:] = bloch.add_noise(train_set_copy[index][1], sd)
            train_set_copy[index][2][:] = id

        for index in range(len(test_set_copy)):
            test_set_copy[index][0][:] = bloch.add_noise(test_set_copy[index][1], sd)
            test_set_copy[index][2][:] = id

        train_set_full = data.ConcatDataset([train_set_full, train_set_copy])
        test_set_full =  data.ConcatDataset([test_set_full, test_set_copy])

    print('train set size ', len(train_set))
    print('test set size ', len(test_set))

    return train_set_full, test_set_full

# %% define helper functions

def get_indices_of_noise(noiseParam):
    indices_sorted_noise = []
    for noiseValue in range(int(max(noiseParam))+1):
        indices = np.argwhere(noiseParam == noiseValue).flatten()
        indices_sorted_noise.append(indices)
    return indices_sorted_noise

#Calculate MSE for each noise distribution
def get_mse_of_sorted_noise(indices_sorted_by_noise, fReconEpochs, fCleanEpochs):
    mse_over_noise = []
    sample_amount = []
    for noise_parameter in range(len(indices_sorted_by_noise)):
        mse = np.mean((fCleanEpochs[indices_sorted_by_noise[noise_parameter]] - fReconEpochs[indices_sorted_by_noise[noise_parameter]])**2)
        mse_over_noise.append(mse)
        sample_amount.append(len(indices_sorted_by_noise[noise_parameter]))
    return mse_over_noise, sample_amount


# %%
# define some parameters

noise_max = 10
noise_min = 1
batch_size_train = 64
N_diff = 21

test_samples_noised = []
test_samples_clean = []
test_samples_D = []
test_samples_noise = []
recon_VAE = []
recon_linsub = []

sample_struct = {
    'noised_samples': test_samples_noised,
    'clean_samples': test_samples_clean,
    # 'D_samples': test_samples_D,
    'noise_amount': test_samples_noise,
    'recon_VAE': recon_VAE,
    'recon_linsub': recon_linsub
}

# create dataset (has to be in same directory)

f = h5py.File(HOME_DIR + data_conf['dvs'], 'r')
bvals = f['bvals'][:]
bvecs = f['bvecs'][:]
f.close()

if bvals.ndim == 1:
    bvals = bvals[:, np.newaxis]

x_clean, original_D = bloch.model_DTI(bvals, bvecs)

loader_train, loader_test = create_data_loader(x_clean, original_D, noise_min, noise_max, batch_size_train, sample_struct)

#train linsub and load VAE (has do be in same directory)

x_clean_tensor = torch.tensor(x_clean).to(device).to(torch.float)
linsub_basis_tensor = linsub.learn_linear_subspace(x_clean_tensor, num_coeffs=N_latent, use_error_bound=False)

model = ae.VAE(input_features=N_diff, latent_features=N_latent)
model.load_state_dict(torch.load(HOME_DIR + model_conf['checkpoint'], map_location=torch.device('cpu')))
model.to(device)

#Test VAE and Linsub

test_linsub = []
test_VAE = []

for batch_idx, (noisy_t, clean_t, noise) in enumerate(loader_test):

    noisy_t = noisy_t.type(torch.FloatTensor).to(device)
    clean_t = clean_t.type(torch.FloatTensor).to(device)


    linsub_tensor = linsub_basis_tensor @ linsub_basis_tensor.T @ noisy_t.view(21, -1)

    linsub_recon = linsub_tensor.view(noisy_t.shape)
    test_linsub.append(linsub_recon.detach().cpu().numpy())

    recon_t, _, _ = model(noisy_t)
    test_VAE.append(recon_t.detach().cpu().numpy())

sample_struct['recon_VAE'] = test_VAE
sample_struct['recon_linsub'] = test_linsub


with open(DIR + '/mse.pkl', 'wb') as f:
    pickle.dump(sample_struct, f)

# %% Create MSE figure

indices_sorted_by_noise = get_indices_of_noise(np.squeeze(sample_struct['noise_amount'][:]))

mses, sample_amount = get_mse_of_sorted_noise(indices_sorted_by_noise, np.squeeze(sample_struct['recon_VAE'][:]), np.squeeze(sample_struct['clean_samples'][:]))
mse_linsub, sample_amount2 = get_mse_of_sorted_noise(indices_sorted_by_noise, np.squeeze(sample_struct['recon_linsub'][:]), np.squeeze(sample_struct['clean_samples'][:]))
mse_noisy, _ = get_mse_of_sorted_noise(indices_sorted_by_noise, np.squeeze(sample_struct['noised_samples'][:]), np.squeeze(sample_struct['clean_samples'][:]))
plt.bar(np.arange(len(indices_sorted_by_noise)), mse_noisy ,label= f'Noisy',  alpha=0.7)
plt.bar(np.arange(len(indices_sorted_by_noise)), mse_linsub ,label= f'Linsub recon',  alpha=0.7)
plt.bar(np.arange(len(indices_sorted_by_noise)), mses ,label= f'VAE recon',  alpha=0.7)

plt.title('MSE over noise amount')
plt.xlabel('Noise amount')
plt.ylabel('MSE')
plt.legend()
plt.savefig(DIR + '/vae_vs_linsub_mse.png',
            bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()
