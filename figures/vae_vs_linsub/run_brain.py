import h5py
import os
import torch
import yaml

import numpy as np

from deepdwi import prep, util
from deepdwi.models import bloch, prox
from deepdwi.models import autoencoder as ae
from deepdwi.recons import linsub

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
print('    kdat: ', data_conf['kdat'])
print('    navi: ', data_conf['navi'])
print('    slice_idx: ', data_conf['slice_idx'])
print('    coil: ', data_conf['coil'])
print('    dvs: ', data_conf['dvs'])

model_conf = config_dict.get('model', {})
print('> model_conf: ')
print('    checkpoint: ', model_conf['checkpoint'])
print('    N_latent: ', model_conf['N_latent'])

N_latent = model_conf['N_latent']

# %% MUSE reconstruction
print('>>> MUSE reconstruction ...')
coil4, kdat6, phase_shot, phase_slice, mask, DWI_MUSE = \
    prep.prep_dwi_data(data_file=data_conf['kdat'],
                       navi_file=data_conf['navi'],
                       coil_file=data_conf['coil'],
                       slice_idx=data_conf['slice_idx'],
                       norm_kdat=False,
                       return_muse=True)

print('> MUSE shape: ', DWI_MUSE.shape)

DWI_MUSE = np.squeeze(DWI_MUSE)
N_diff, N_z, N_y, N_x = DWI_MUSE.shape

# %% VAE Decoder Reconstruction
print('>>> VAE reconstruction ...')
model = ae.VAE(input_features=N_diff, latent_features=N_latent)
model.load_state_dict(torch.load(HOME_DIR + model_conf['checkpoint'], map_location=torch.device('cpu')))
model.to(device)

DWI_MUSE_tensor = torch.from_numpy(DWI_MUSE).to(device)

prox_vae = prox.VAE(model)

DWI_VAE_tensor = prox_vae(DWI_MUSE_tensor, contrast_dim=0)
DWI_VAE = DWI_VAE_tensor.detach().cpu().numpy()


# %% LINSUB reconstruction
print('>>> LINSUB reconstruction ...')
f = h5py.File(HOME_DIR + data_conf['dvs'], 'r')
bvals = f['bvals'][:]
bvecs = f['bvecs'][:]
f.close()

if bvals.ndim == 1:
    bvals = bvals[:, np.newaxis]

x_clean, _ = bloch.model_DTI(bvals, bvecs)
x_clean_tensor = torch.tensor(x_clean).to(device).to(torch.float)
print('> dictionary shape ', x_clean_tensor.shape, x_clean_tensor.dtype)

linsub_basis_tensor = linsub.learn_linear_subspace(x_clean_tensor,
                                                   num_coeffs=N_latent,
                                                   use_error_bound=False)

DWI_LINSUB_tensor = linsub_basis_tensor @ linsub_basis_tensor.T @ abs(DWI_MUSE_tensor).view(N_diff, -1)

DWI_LINSUB_tensor = DWI_LINSUB_tensor.view(DWI_MUSE_tensor.shape)

DWI_LINSUB = DWI_LINSUB_tensor.detach().cpu().numpy()
DWI_LINSUB = DWI_LINSUB * DWI_MUSE[0]

# %% save outputs
f = h5py.File(DIR + '/results.h5', 'w')
f.create_dataset('MUSE', data=DWI_MUSE)
f.create_dataset('VAE', data=DWI_VAE)
f.create_dataset('LINSUB', data=DWI_LINSUB)
f.close()
