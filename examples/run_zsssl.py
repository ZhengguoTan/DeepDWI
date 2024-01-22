import h5py
import os
import time
import torch

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp

from deepdwi.dims import *
from deepdwi.recons import zsssl
from sigpy.mri import app, muse, retro, sms
from torch.utils.data import DataLoader

DIR = os.path.dirname(os.path.realpath(__file__))

DAT_DIR = DIR.rsplit('/', 1)[0] + '/data'
print('> data directory: ', DAT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
slice_idx = 0

f = h5py.File(DAT_DIR + '/1.2mm_32-dir_R3x2_kdat_slice_000.h5', 'r')
kdat = f['kdat'][:]
MB = f['MB'][()]
N_slices = f['Slices'][()]
N_segments = f['Segments'][()]
N_Accel_PE = 3  # f['Accel_PE'][()]
f.close()

kdat = np.squeeze(kdat)  # 4 dim
kdat = np.swapaxes(kdat, -2, -3)

# # split kdat into shots
N_diff = kdat.shape[-4]
kdat_prep = []
for d in range(N_diff):
    k = retro.split_shots(kdat[d, ...], shots=N_segments)
    kdat_prep.append(k)

kdat_prep = np.array(kdat_prep)
kdat_prep = kdat_prep[..., None, :, :]  # 6 dim

N_diff, N_shot, N_coil, _, N_y, N_x = kdat_prep.shape

print(' > kdat shape: ', kdat_prep.shape)

# sampling mask
mask = app._estimate_weights(kdat_prep, None, None, coil_dim=-4)
mask = abs(mask).astype(float)

print(' > mask shape: ', mask.shape)

# coil
f = h5py.File(DAT_DIR + '/1.2mm_32-dir_R3x2_coil.h5', 'r')
coil = f['coil'][:]
f.close()

print(' > coil shape: ', coil.shape)

N_coil, N_z, N_y, N_x = coil.shape

# %%
yshift = []
for b in range(MB):
    yshift.append(b / N_Accel_PE)

sms_phase = sms.get_sms_phase_shift([MB, N_y, N_x], MB=MB, yshift=yshift)

# %%
slice_mb_idx = sms.map_acquire_to_ordered_slice_idx(0, N_slices, MB)

coil2 = coil[:, slice_mb_idx, :, :]
print('> coil2 shape: ', coil2.shape)

# %%
import torchvision.transforms as T

acs_shape = list([N_y // 4, N_x // 4])
ksp_acs = sp.resize(kdat_prep, oshape=list(kdat_prep.shape[:-2]) + acs_shape)

coils_tensor = sp.to_pytorch(coil2)
TR = T.Resize(acs_shape, antialias=True)
mps_acs_r = TR(coils_tensor[..., 0]).cpu().detach().numpy()
mps_acs_i = TR(coils_tensor[..., 1]).cpu().detach().numpy()
mps_acs = mps_acs_r + 1j * mps_acs_i

sms_phase_acs = sms.get_sms_phase_shift([MB] + acs_shape, MB=MB, yshift=yshift)

_, dwi_shot = muse.MuseRecon(ksp_acs, mps_acs,
                             MB=MB,
                             acs_shape=acs_shape,
                             lamda=0.01, max_iter=30,
                             yshift=yshift,
                             device=sp.Device(0))

_, dwi_shot_phase = muse._denoising(dwi_shot, full_img_shape=[N_y, N_x])

# %%
res_mask, valid_mask = zsssl.uniform_samp(torch.from_numpy(mask), rho=0.2)
valid_mask = valid_mask[None, ...]  # 7dim

N_repeats = 3

train_mask = []
lossf_mask = []

for r in range(N_repeats):

    train_mask1, lossf_mask1 = zsssl.uniform_samp(res_mask, rho=0.4)

    train_mask.append(train_mask1)
    lossf_mask.append(lossf_mask1)

train_mask = torch.stack(train_mask)
lossf_mask = torch.stack(lossf_mask)

print('> train_mask shape: ', train_mask.shape)
print('> lossf_mask shape: ', lossf_mask.shape)

print('> valid_mask shape: ', valid_mask.shape)


coil7 = torch.from_numpy(coil2)
coil7 = coil7[None, None, None, ...]
coil7 = torch.tile(coil7, tuple([N_repeats] + [1] * (coil7.dim()-1)))
print('> coil7 shape: ', coil7.shape)

kdat7 = torch.from_numpy(kdat_prep)
kdat7 = kdat7[None, ...]
kdat7 = torch.tile(kdat7, tuple([N_repeats] + [1] * (kdat7.dim()-1)))
print('> kdat7 shape: ', kdat7.shape)

phase_shot7 = torch.from_numpy(dwi_shot_phase)
phase_shot7 = phase_shot7[None, ...]
phase_shot7 = torch.tile(phase_shot7, tuple([N_repeats] + [1] * (phase_shot7.dim()-1)))
print('> phase_shot7 shape: ', phase_shot7.shape)

phase_slice7 = torch.from_numpy(sms_phase)
phase_slice7 = phase_slice7[None, None, None, None, ...]
phase_slice7 = torch.tile(phase_slice7, tuple([N_repeats] + [1] * (phase_slice7.dim()-1)))
print('> phase_slice7 shape: ', phase_slice7.shape)


# %% run only one DWI direction and the first 10 coils
kdat7 = kdat7[:, [0], ...]
phase_shot7 = phase_shot7[:, [0], ...]
res_mask = res_mask[[0], ...]
train_mask = train_mask[:, [0], ...]
lossf_mask = lossf_mask[:, [0], ...]
valid_mask = valid_mask[:, [0], ...]

kdat7 = kdat7[..., :10, :, :, :]
coil7 = coil7[..., :10, :, :, :]

print('> phase_shot7 shape: ', phase_shot7.shape)

# %%
from deepdwi.recons import zsssl

train_data = zsssl.Dataset(coil7, kdat7, train_mask, lossf_mask, phase_shot7, phase_slice7)
train_load = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=6)

valid_data = zsssl.Dataset(coil7[[0]], kdat7[[0]], res_mask[None, ...], valid_mask, phase_shot7[[0]], phase_slice7[[0]])
valid_load = DataLoader(valid_data, batch_size=1, shuffle=True, num_workers=6)

if kdat7.shape[DIM_TIME] == 1:
    model = zsssl.UnrollNet(NN='Identity', requires_grad_lamda=False)
else:
    model = zsssl.UnrollNet(NN='ResNet3D', requires_grad_lamda=False)

lossf = zsssl.MixL1L2Loss()
optim = torch.optim.Adam(model.parameters(), lr=5e-4)

# %% training
total_train_loss, total_valid_loss = [], []
valid_loss_min = np.inf

epoch, val_loss_tracker = 0, 0

start_time=time.time()

while epoch < 10 and val_loss_tracker < 25:

    tic = time.time()
    trn_loss = zsssl.train(model, train_load, lossf, optim, device=device)
    val_loss = zsssl.valid(model, valid_load, lossf, optim, device=device)

    total_train_loss.append(trn_loss)
    total_valid_loss.append(val_loss)

    #save the best checkpoint
    checkpoint = {
        "epoch": epoch,
        "valid_loss_min":val_loss,
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict()
    }

    if val_loss <= valid_loss_min:
        valid_loss_min = val_loss
        torch.save(checkpoint, os.path.join(DIR, "best.pth"))
        val_loss_tracker = 0 #reset the val loss tracker each time a new lowest val error is achieved
    else:
        val_loss_tracker += 1

    toc = time.time() - tic
    if epoch % 1 == 0:
        print("Epoch:", epoch+1, ", elapsed_time = ""{:f}".format(toc), ", trn loss = ", "{:.9f}".format(trn_loss),", val loss = ", "{:.9f}".format(val_loss))

    epoch += 1

end_time = time.time()
print('Training completed in  ', str(epoch), ' epochs, ',((end_time - start_time) / 60), ' minutes')
