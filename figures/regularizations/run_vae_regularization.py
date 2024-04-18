import h5py
import os
import torch
import yaml

import numpy as np

from deepdwi import prep
from deepdwi.models import mri, prox
from deepdwi.models import autoencoder as ae
from deepdwi.recons import zsssl

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]
HOME_DIR = HOME_DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% read in and display the yaml config file
with open(DIR + '/config_vae.yaml', 'r') as f:
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

admm_conf = config_dict.get('ADMM', {})
print('> ADMM_conf: ')
print('    iteration: ', admm_conf['iteration'])
print('    max_cg_iter: ', admm_conf['max_cg_iter'])
print('    rho: ', admm_conf['rho'])
print('    lamda: ', admm_conf['lamda'])

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

# %% VAE as Regularization for Reconstruction
print('>>> VAE as regularization for reconstruction ...')

# load VAE model
model = ae.VAE(input_features=N_diff, latent_features=N_latent)
model.load_state_dict(torch.load(HOME_DIR + model_conf['checkpoint'], map_location=torch.device('cpu')))
model.to(device)

prox_vae = prox.VAE(model)


coil4_tensor = torch.from_numpy(coil4).to(device)
kdat6_tensor = torch.from_numpy(kdat6).to(device)
phase_shot_tensor = torch.from_numpy(phase_shot).to(device)
phase_slice_tensor = torch.from_numpy(phase_slice).to(device)
mask_tensor = torch.from_numpy(mask).to(device)


A = mri.Sense(coil4_tensor, kdat6_tensor,
              phase_echo=phase_shot_tensor, combine_echo=True,
              phase_slice=phase_slice_tensor)

# ADMM

lamda = admm_conf['lamda']
rho = admm_conf['rho']
ABSTOL = 1E-4
RELTOL = 1E-3
verbose = True

x = A.adjoint(A.y)
v = x.clone()
u = torch.zeros_like(x)

for n in range(admm_conf['iteration']):

    AHA = lambda x: A.adjoint(A(x)) + rho * x
    AHy = A.adjoint(A.y) + rho * (v - u)

    # update x
    x = zsssl.conj_grad(AHA, AHy, max_iter=admm_conf['max_cg_iter'])
    # update v
    v_old = v.clone()
    v = prox_vae(x + u, alpha=lamda / rho)

    if verbose:
        r_norm = torch.linalg.norm(x - v).item()
        s_norm = torch.linalg.norm(-rho * (v - v_old)).item()

        r_scaling = max(torch.linalg.norm(x).item(),
                        torch.linalg.norm(v).item())
        s_scaling = rho * torch.linalg.norm(u).item()

        eps_pri = ABSTOL * (np.prod(v.shape)**0.5) + RELTOL * r_scaling
        eps_dual = ABSTOL * (np.prod(v.shape)**0.5) + RELTOL * s_scaling

        print('admm iter: ' + "%2d" % (n+1) +
              ', r norm: ' + "%10.4f" % (r_norm) +
              ', eps pri: ' + "%10.4f" % (eps_pri) +
              ', s norm: ' + "%10.4f" % (s_norm) +
              ', eps dual: ' + "%10.4f" % (eps_dual))

    # update u
    u = u + x - v

# %% save output
f = h5py.File(DIR + '/vae_regu.h5', 'w')
f.create_dataset('MUSE', data=DWI_MUSE)
f.create_dataset('VAE', data=x.detach().cpu().numpy())
f.close()
