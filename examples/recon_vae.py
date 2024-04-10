import argparse
import h5py
import os
import pathlib
import torch
import yaml

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from sigpy.mri import muse

import torch.nn as nn
import torch.optim as optim

from deepdwi import prep
from deepdwi.models import mri
from deepdwi.models import autoencoder as ae

DIR = os.getcwd()

HOME_DIR = DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

DATA_DIR = DIR.rsplit('/', 1)[0] + '/data'
print('> data directory: ', DATA_DIR)

# Switch on GPU if your GPU has more than 4 GB memory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('> device: ', device)


# %%
if __name__ == "__main__":
    # %% argument parser
    # you can display help messages using `python run_zsssl.py -h`
    parser = argparse.ArgumentParser(description='run zsssl.')

    parser.add_argument('--config', type=str,
                        default='/configs/recon_vae.yaml',
                        help='yaml config file for zsssl')

    args = parser.parse_args()

    # %% read in and display the yaml config file
    with open(HOME_DIR + args.config, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    data_conf = config_dict.get('data', {})
    print('> data_conf: ')
    print('    kdat: ', data_conf['kdat'])
    print('    slice_idx: ', data_conf['slice_idx'])
    print('    coil: ', data_conf['coil'])
    print('    normalize_kdat: ', data_conf['normalize_kdat'])

    test_conf = config_dict['test']
    print('> test_conf: ')
    print('    checkpoint: ', test_conf['checkpoint'])

    relative_path = test_conf['checkpoint'].rsplit('/', 1)[0]
    RECON_DIR = HOME_DIR + relative_path

    yaml_file = 'vae_slice_' + str(data_conf['slice_idx']).zfill(3) + '.yaml'

    print('> RECON_DIR: ', RECON_DIR)
    # make a new directory if not exist
    pathlib.Path(RECON_DIR).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(RECON_DIR, yaml_file), 'w') as f:
        f.write(yaml.dump(config_dict, sort_keys=False))


    # %%
    coil4, kdat6, phase_shot, phase_slice, mask, DWI_MUSE = \
        prep.prep_dwi_data(data_file=data_conf['kdat'],
                           coil_file=data_conf['coil'],
                           slice_idx=data_conf['slice_idx'],
                           norm_kdat=data_conf['normalize_kdat'],
                           return_muse=True)

    # %%
    N_latent = test_conf['N_latent']
    model = ae.VAE(input_features=21, latent_features=N_latent)
    model.load_state_dict(torch.load(HOME_DIR + test_conf['checkpoint']))
    model.to(device)

    # %%
    coil_tensor = torch.from_numpy(coil4).to(device).type(torch.complex64)
    kdat_tensor = torch.from_numpy(kdat6).to(device).type(torch.complex64)
    shot_phase_tensor = torch.from_numpy(phase_shot).to(device).type(torch.complex64)
    sms_phase_tensor = torch.from_numpy(phase_slice).to(device).type(torch.complex64)

    S = mri.Sense(coil_tensor, kdat_tensor,
                  phase_echo=shot_phase_tensor,
                  combine_echo=True,
                  phase_slice=sms_phase_tensor,
                  N_basis=N_latent,
                  basis=model,
                  baseline=torch.from_numpy(DWI_MUSE[[0]]).to(device))

    print('S ishape: ', S.ishape)
    print('S oshape: ', S.oshape)
    print('S.y shape: ', S.y.shape)


    latents = torch.zeros(S.ishape, dtype=torch.float32,
                        device=device,
                        requires_grad=True)

    print('> x device: ', latents.device)


    lossf = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam([latents], lr=0.1)

    for epoch in range(100):
        fwd = S(latents)
        res = lossf(torch.view_as_real(fwd), torch.view_as_real(S.y))

        optimizer.zero_grad()
        res.backward()
        optimizer.step()

        print('> epoch %3d loss %.12f'%(epoch, res.item()))


    reco_file = '/vae_slice_' + str(data_conf['slice_idx']).zfill(3)

    f = h5py.File(RECON_DIR + reco_file + '.h5', 'w')
    f.create_dataset('MUSE', data=DWI_MUSE)
    f.create_dataset('latents', data=latents.detach().cpu().numpy())
    f.close()
