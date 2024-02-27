"""
This module simulates MRI signal based on Bloch equations.

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
    Soundarya Soundarresan <soundarya.soundarresan@fau.de>
    Julius Glaser <julius.glaser@fau.de>
"""

import numpy as np


def get_B(b, g):
    num_g, num_axis = g.shape

    assert num_axis == 3
    assert num_g == len(b)

    gx = g[:, 0]
    gy = g[:, 1]
    gz = g[:, 2]

    return - b * np.array([gx**2, 2*gx*gy, gy**2,
                           2*gx*gz, 2*gy*gz, gz**2]).transpose()


def get_D_linspace(D):
    return np.linspace(D[0], D[1], D[2])


def model_DTI(b, g,
              Dxx=(0    , 3E-3, 10),
              Dxy=(-1E-3, 1E-3, 10),
              Dyy=(0    , 3E-3, 10),
              Dxz=(-1E-3, 1E-3, 10),
              Dyz=(-1E-3, 1E-3, 10),
              Dzz=(0    , 3E-3, 10)
              ):

    B = get_B(b, g)  #(32,6)

    Dxx_grid = get_D_linspace(Dxx)  # 0
    Dxy_grid = get_D_linspace(Dxy)  # 1
    Dyy_grid = get_D_linspace(Dyy)  # 2
    Dxz_grid = get_D_linspace(Dxz)  # 3
    Dyz_grid = get_D_linspace(Dyz)  # 4
    Dzz_grid = get_D_linspace(Dzz)  # 5

    D = np.meshgrid(Dxx_grid, Dxy_grid, Dyy_grid,
                    Dxz_grid, Dyz_grid, Dzz_grid)
    D = np.array(D).reshape((6, -1))

    # diagonal tensors (Dxx, Dyy, Dzz) should be
    # larger than or equal to
    # off-diagonal tensors (Dxy, Dxz, Dyz).
    D_pick1 = []
    for n in range(D.shape[1]):
        d = D[:, n]
        if (d[0] >= d[1] and d[0] >= d[3] and d[0] >= d[4] and
            d[2] >= d[1] and d[2] >= d[3] and d[2] >= d[4] and
            d[5] >= d[1] and d[5] >= d[3] and d[5] >= d[4]):
            D_pick1.append(d)

    D_pick1 = np.transpose(np.array(D_pick1))  # [6, N_D_atoms]

    y = np.exp(np.matmul(B, D_pick1)) # [32, N_D_atoms]

    # q-space signal larger than 1 (b0) violates MR physics.
    y_pick = []
    D_pick2 = []
    for n in range(y.shape[1]): #N_D_atoms
        q = y[:, n]
        if not np.any(q > 1):
            y_pick.append(q)
            D_pick2.append(D_pick1[:, n])

    y_pick = np.transpose(np.array(y_pick))  # [N_diff, N_q_atoms]
    D_pick2 = np.transpose(np.array(D_pick2))  # [6, N_q_atoms]

    return y_pick, D_pick2


def model_ball_and_stick(b, g):
    # TODO:
    None


def _linspace_to_array(linspace_list):

    val0 = linspace_list[0]
    val1 = linspace_list[1]
    leng = linspace_list[2]

    if leng == 1:
        return np.array([val0])
    else:
        step = (val1 - val0) / (leng - 1)
        return val0 + step * np.arange(leng)


def model_t2(TE,
             T2=(0.001, 0.200, 100)):

    T2_array = _linspace_to_array(T2)

    sig = np.zeros((len(TE), len(T2_array)), dtype=complex)

    for T2_ind in np.arange(T2[2]):
        T2_val = T2_array[T2_ind]
        z = (-1. / T2_val + 1j * 0.)

        sig[:, T2_ind] = np.exp(z * TE)

    return sig