"""
This script demonstrates
  * the kx-ky sampling pattern in diffusion MRI,
  * the ky-diff sampling pattern in diffusion MRI.

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

import argparse

# %%
parser = argparse.ArgumentParser(description='plot sampling pattern.')

parser.add_argument('--seg', type=int, default=2,
                    help='number of segments (shots) per diffusion encoding.')

parser.add_argument('--seg_idx', type=int, default=1,
                    help='segment (shot) index.')

parser.add_argument('--pat', type=int, default=3,
                    help='undersampling factor per diffusion encoding.')

parser.add_argument('--plot_kxky', action='store_true',
                    help='plot kx-ky sampling pattern.')

args = parser.parse_args()


fontsize = 16

# %%
def plot_kxky(seg=2, pat=3, xmin=1, xmax=13, ymin=-10, ymax=8):
    """
    plot ky and t sampling pattern

    Arguments:

    """
    xvec = range(xmin, xmax)
    yvec = range(ymin, ymax)

    colors = plt.cm.rainbow(np.linspace(1, 0, 1))

    f, ax = plt.subplots(figsize=(4,4))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # plot circles
    X, Y = np.meshgrid(xvec, yvec)
    ax.scatter(X, Y, color='none', edgecolor='gray')

    # plot y and x axes with arrows
    ax.plot(1, ymin-1, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1     , "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    for t in range(0, len(yvec), pat * seg):  # line-by-line scan

        xarr = xvec
        yarr = np.ones_like(xarr) * (yvec[t] + (args.seg_idx - 1) * pat)

        ax.scatter(xarr, yarr, color=colors[0], edgecolor='none')

    ax.set_xlim(0, xmax)
    ax.set_xticks([])
    ax.set_xlabel('$k_x$', fontsize=fontsize)
    ax.set_ylim(int(ymin)-1, int(ymax))
    ax.set_yticks([])
    ax.set_ylabel('$k_y$', fontsize=fontsize)
    # ax.set_title('Shot #' + str(args.seg_idx))

    ax.set_aspect('equal', adjustable='box')

    plt.savefig('sampling_pattern_kxky' + '_Seg' + str(args.seg_idx) + 'of' + str(seg) + '_Pat' + str(pat) + '.png',
                bbox_inches='tight', pad_inches=0, dpi=300)


# %%
def plot_ky_t(seg=2, pat=3, xmin=1, xmax=13, ymin=-10, ymax=8):
    """
    plot ky and t sampling pattern

    Arguments:

    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xvec = range(xmin, xmax)
    yvec = range(ymin, ymax)

    # plot circles
    X, Y = np.meshgrid(xvec, yvec)
    ax.scatter(X, Y, color='none', edgecolor='gray')

    # plot y and x axes with arrows
    ax.plot(1, ymin-1, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1     , "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    denc = len(xvec) // seg + 1

    colors = plt.cm.rainbow(np.linspace(1, 0, denc))

    for t in range(len(xvec)):

        ind_shot = t % seg
        ind_denc = t // seg
        ky_shift = ind_denc % pat

        yarr = range(ymin + ky_shift + ind_shot * pat, ymax, pat * seg)
        xarr = np.ones_like(yarr) * xvec[t]
        ax.scatter(xarr, yarr, color=colors[ind_denc], edgecolor='none')

    ax.set_xlim(0, xmax)
    ax.set_xticks([])
    ax.set_xlabel('Diffusion encoding', fontsize=fontsize)

    ax.set_ylim(ymin-1, ymax)
    ax.set_yticks([])
    ax.set_ylabel('$k_y$', fontsize=fontsize)

    # ax.set_title('$k_y$-shifted diffusion encoding')

    ax.set_aspect('equal', adjustable='box')

    plt.savefig('sampling_pattern_ky_t' + '_Seg' + str(seg) + '_Pat' + str(pat) + '.png',
                bbox_inches='tight', pad_inches=0, dpi=300)


# %%
if args.plot_kxky is True:
    plot_kxky(seg=args.seg, pat=args.pat)
else:
    plot_ky_t(seg=args.seg, pat=args.pat)