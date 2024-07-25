import os
import h5py

import numpy as np

from scipy.ndimage.filters import gaussian_filter

DIR = os.path.dirname(os.path.realpath(__file__))

HOME_DIR = DIR.rsplit('/', 1)[0]
HOME_DIR = HOME_DIR.rsplit('/', 1)[0]
print('> HOME: ', HOME_DIR)

DATA_DIR = HOME_DIR + '/data/'
print('> DATA: ', DATA_DIR)

# %% b-values and vectors
f = h5py.File(DATA_DIR + '/1shell_21dir_diff_encoding.h5', 'r')
bvals = f['bvals'][:]
bvecs = f['bvecs'][:]
f.close()

from dipy.core.gradients import gradient_table
gtab = gradient_table(bvals, bvecs, atol=0.1)

import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa
tenmodel = dti.TensorModel(gtab)

# %% loop over all source files to fit MD, FA, and RGB
src_files = ['/home/atuin/b143dc/b143dc15/Softwares/DeepDWI/examples/2024-06-14_zsssl_0.7mm_21-dir_R2x2_vol2_scan2_kdat_slice_040_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_test_shot-retro-0.h5']

for cnt in range(len(src_files)):

    src_file = src_files[cnt]
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

    # filtering
    fwhm = 1.25
    gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std
    DWI_pref = np.zeros(DWI_pret.shape)
    for v in range(DWI_pret.shape[-1]):
        DWI_pref[..., v] = gaussian_filter(DWI_pret[..., v], sigma=gauss_std)


    print('  shape: ', DWI_pret.shape)

    tenfit = tenmodel.fit(DWI_pref)

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
