import argparse
import h5py
import os
import torch
import yaml

import numpy as np
import sigpy as sp

from deepdwi import prep
from deepdwi.models import mri, prox
from deepdwi.models import autoencoder as ae
from deepdwi.recons import zsssl

from sigpy.mri import app

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]
HOME_DIR = HOME_DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

device = sp.Device(0 if torch.cuda.is_available() else -1)

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

# %% user options
parser = argparse.ArgumentParser(description='run zsssl.')

parser.add_argument('--N_shot_retro', type=int,
                    default=0,
                    help='retro. undersample the number of shots')

args = parser.parse_args()

# %% MUSE reconstruction
print('>>> MUSE reconstruction ...')
coil4, kdat6, kdat_scaling, phase_shot, phase_slice, mask, DWI_MUSE = \
    prep.prep_dwi_data(data_file=data_conf['kdat'],
                       navi_file=data_conf['navi'],
                       coil_file=data_conf['coil'],
                       slice_idx=data_conf['slice_idx'],
                       norm_kdat=False,
                       N_shot_retro=args.N_shot_retro,
                       return_muse=True)

print('> MUSE shape: ', DWI_MUSE.shape)

DWI_MUSE = np.squeeze(DWI_MUSE)
N_diff, N_z, N_y, N_x = DWI_MUSE.shape

# %% VAE as Regularization for Reconstruction
print('>>> LLR as regularization for reconstruction ...')

kdat6 = sp.to_device(kdat6, device=device)
coil4 = sp.to_device(coil4, device=device)
phase_slice = sp.to_device(phase_slice, device=device)
phase_shot = sp.to_device(phase_shot, device=device)

dwi_comb_jets = app.HighDimensionalRecon(
                                kdat6, coil4,
                                phase_sms=phase_slice,
                                combine_echo=True,
                                phase_echo=phase_shot,
                                regu='LLR',
                                blk_shape=(1, 6, 6),
                                blk_strides=(1, 1, 1),
                                solver='ADMM',
                                normalization=True,
                                lamda=0.01,
                                rho=0.05,
                                max_iter=15,
                                show_pbar=False, verbose=True,
                                device=device).run()

# %% save output
if args.N_shot_retro > 0:
    SHOT_STR = '_shot-retro-' + str(args.N_shot_retro)
else:
    SHOT_STR = ''

f = h5py.File(DIR + '/llr_regu' + SHOT_STR + '.h5', 'w')
f.create_dataset('MUSE', data=DWI_MUSE)
f.create_dataset('LLR', data=sp.to_device(dwi_comb_jets))
f.close()
