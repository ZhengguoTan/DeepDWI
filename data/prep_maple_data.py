import h5py
import os

import numpy as np
import scipy.io as sio

DIR = os.path.dirname(os.path.realpath(__file__))
print('DIR: ', DIR)

# %%
f = sio.loadmat(DIR + '/cmap_slice_25.mat')
mps = f['maps']

f = sio.loadmat(DIR + '/raw_slice_25.mat')
ksp = f['kData']
print(f.keys())

mask = np.single(np.load(DIR + '/kMask_16x_Uniform_complementary.npy'))

print('> original ...')
print('mps shape: ', mps.shape, mps.flags['C_CONTIGUOUS'])
print('ksp shape: ', ksp.shape, ksp.flags['C_CONTIGUOUS'])
print('mask shape: ', mask.shape, mask.flags['C_CONTIGUOUS'])


mps_c = np.ascontiguousarray(mps.T)
ksp_c = np.ascontiguousarray(ksp.T)
mask_c = np.ascontiguousarray(mask.T)

print('> converted ...')
print('mps_c shape: ', mps_c.shape, mps_c.flags['C_CONTIGUOUS'])
print('ksp_c shape: ', ksp_c.shape, ksp_c.flags['C_CONTIGUOUS'])
print('mask_c shape: ', mask_c.shape, mask_c.flags['C_CONTIGUOUS'])

f = h5py.File(DIR + '/maple_data.h5', 'w')
f.create_dataset('coil', data=mps_c)
f.create_dataset('kdat', data=ksp_c)
f.create_dataset('mask', data=mask_c)
f.close()
