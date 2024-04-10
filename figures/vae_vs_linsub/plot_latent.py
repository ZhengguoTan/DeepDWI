import copy
import h5py
import os
import pickle
import torch
import yaml

import scipy.stats as stats
from deepdwi.models import autoencoder as ae

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
# define some parameters

noise_max = 10
noise_min = 1
batch_size_train = 64
N_diff = 21

#load VAE (has do be in same directory)

model = ae.VAE(input_features=N_diff, latent_features=N_latent)
model.load_state_dict(torch.load(HOME_DIR + model_conf['checkpoint'], map_location=torch.device('cpu')))
model.to(device)

#Test VAE and Linsub

_, means, logvars = model(torch.zeros(N_diff,))
stds = torch.exp(0.5 * logvars)

stds = stds.detach().numpy()
means = means.detach().numpy()

# %% Create MSE figure

for i in range(N_latent):
     std = stds[i]
     mean = means[i]
     x = np.linspace(mean - 5*std, mean + 5*std, 100)
     plt.plot(x, stats.norm.pdf(x, mean, std))
title = 'Latent distributions'
plt.title(title)

plt.savefig(DIR + '/vae_latent.png',
            bbox_inches='tight', pad_inches=0, dpi=300)
plt.close()
