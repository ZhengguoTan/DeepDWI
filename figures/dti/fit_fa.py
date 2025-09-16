import os
import h5py

import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = DIR.rsplit('/', 1)[0]
DATA_DIR = DATA_DIR.rsplit('/', 1)[0] + '/data'

# %% b-values and vectors
with h5py.File(DATA_DIR + '/0.7mm_21-dir_R1x3_dvs.h5', 'r') as f:
    bvals = f['bvals'][:]
    bvecs = f['bvecs'][:]

from dipy.core.gradients import gradient_table
gtab = gradient_table(bvals, bvecs, atol=0.1)

import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa
tenmodel = dti.TensorModel(gtab)

# %% loop over all source files to fit MD, FA, and RGB
src_files = ['MUSE_DENOISE.h5',
             'JETS.h5',
             'ZSSSL.h5']

for cnt in range(len(src_files)):

    src_file = DIR + '/' + src_files[cnt]
    print('> src: ' + src_file)
    outprefstr = src_file.split('.h5')[0]

    f = h5py.File(src_file, 'r')
    DWI = f['DWI'][:]
    f.close()

    DWI_prep = abs(np.squeeze(DWI)) * 1E4  # scale DWI

    # DTI from self implementation
    # bvals = np.reshape(bvals, [-1, 1])
    # B = epi.get_B(bvals, bvecs)
    # D = epi.get_D(B, DWI_prep)
    # evals, evecs = epi.get_eig(D, B=B)

    # FA = epi.get_FA(evals)
    # RGB = epi.get_cFA(FA, evecs)
    # MD = epi.get_MD(evals)

    # f = h5py.File(outprefstr + '_fit.h5', 'w')
    # f.create_dataset('FA', data=FA)
    # f.create_dataset('RGB', data=RGB)
    # f.create_dataset('MD', data=MD)
    # f.close()


    # DTI from dipy
    # if cnt != 1:
    DWI_pret = DWI_prep.T
    # else:
    #     DWI_pret = DWI_prep

    print('  shape: ', DWI_pret.shape)

    tenfit = tenmodel.fit(DWI_pret)

    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)
    RGB = np.squeeze(color_fa(FA, tenfit.evecs))
    MD = tenfit.md

    f = h5py.File(outprefstr + '_fit.h5', 'w')
    f.create_dataset('FA', data=FA.T)
    f.create_dataset('RGB', data=RGB.T)
    f.create_dataset('MD', data=MD.T)
    f.close()
