# Generalized Deep Diffusion-Weighted Imaging (DeepDWI) Reconstruction Framework

## Introduction

In the world of [PyTorch](https://pytorch.org/), deep learning reconstruction for magnetic resonance imaging consists of these ingredients:

* **Dataset**, inherited from `torch.utils.data.Dataset`,
* **DataLoader**, inherited from `torch.utils.data.DataLoader`,
* **Model**, inherited from `torch.nn.Module`, and
* **Train**, **Validation**, and **Test** stages.


## Installation

DeepDWI requires Python version >= 3.10.13.

* please follow the instructions to install `cupy`: https://docs.cupy.dev/en/stable/install.html


### Via `conda`


### Via `pip`


### Installation for Developers

```
cd /path/to/DeepDWI
python -m pip install -e .
```

## Features


## Data

Before running the scripts in `DeepDWI`, you need to download the following data to the `/data/` folder:

| Spatial Resolution (mm3) | Diffusion Mode | Acceleration (in-plane x slice) | Shots | Navigator | Link |
|---|---|---|---|---|---|
| 0.7 x 0.7 x 0.7 | MDDW 20 directions with b-value of 1000 s/mm2 | 2 x 2 | 3 | Yes | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10781347.svg)](https://doi.org/10.5281/zenodo.10781347) |
| 1.0 x 1.0 x 1.0 | MDDW 20 directions with b-value of 1000 s/mm2 | 1 x 3 | 4 | No  | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10668487.svg)](https://doi.org/10.5281/zenodo.10668487) |
| 1.0 x 1.0 x 1.0 | 3-shell 126 directions with b-value up to 3000 s/mm2 | 3 x 3 | 2 | No  |  |

## References

* Liang ZP. [Spatiotemporal imaging with partially separable functions](https://doi.org/10.1109/ISBI.2007.357020). Proc IEEE Int Symp Biomed Imaging 2007;4:988-991.

* Doneva M, Börnert P, Eggers H, Stehning C, Sénégas J, Mertins A. [Compressed sensing reconstruction for magnetic resonance parameter mapping](https://doi.org/10.1002/mrm.22483). Magn Reson Med 2010;64:1114-1120.

* Huang C, Graff CG, Clarkson EW, Bilgin A, Altbach MI. [T2 mapping from highly undersampled data by reconstruction of principal component coefficient maps using compressed sensing](https://doi.org/10.1002/mrm.23128). Magn Reson Med 2012;67:1355-1366.

* Dan M, Gulani V, Seiberlich N, Liu K, Sunshine JL, Duerk JL, Griswold MA. [Magnetic resonance fingerprinting](https://doi.org/10.1038/nature11971). Nature 2013;495:187-192.

* McGivney DF, Pierre E, Ma D, Jiang Y, Saybasili H, Gulani V, Griswold MA. [SVD compression for magnetic resonance fingerprinting in the time domain](https://doi.org/10.1109/TMI.2014.2337321). IEEE Trans Med Imaging 2014;33:2311-2322.

* Tamir JI, Uecker M, Chen W, Lai P, Alley MT, Vasanawala SS, Lustig M. [T2 shuffling: Sharp, multicontrast, volumetric fast spin-echo imaging](https://doi.org/10.1002/mrm.26102). Magn Reson Med 2017;77:180-195.

* Dong Z, Wang F, Chan KS, Reese TG, Bilgic B, Marques JP, Setsompop K. [Variable flip angle echo planar time-resolved imaging (vFA-EPTI) for fast high-resolution gradient echo myelin water imaging](https://doi.org/10.1016/j.neuroimage.2021.117897). NeuroImage 2021;232.

* Hammernik K, Klatzer T, Kobler E, Recht MP, Sodickson DK, Pock T, Knoll F. [Learning a variational network for reconstruction of accelerated MRI data](https://doi.org/10.1002/mrm.26977). Magn Reson Med 2018;79:3055-3071.

* Aggarwal HK, Mani MP, Jacob M. [MoDL: Model-Based Deep Learning Architecture for Inverse Problems](https://doi.org/10.1109/TMI.2018.2865356). IEEE Trans Med Imaging 2019;38:394-405.

* Yaman B, Hosseini SAH, Moeller S, Ellermann J, Uğurbil K, Akçakaya M. [Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference data](https://doi.org/10.1002/mrm.28378). Magn Reson Med 2020;84:3172-3191.

* Lam F, Li Y, Peng X. [Constrained Magnetic Resonance Spectroscopic Imaging by Learning Nonlinear Low-Dimensional Models](https://doi.org/10.1109/TMI.2019.2930586). IEEE Trans Med Imaging 2020;39:545-555.

* Mani M, Magnotta VA, Jacob M. [qModeL: A plug-and-play model-based reconstruction for highly accelerated multi-shot diffusion MRI using learned priors](https://doi.org/10.1002/mrm.28756). Magn Reson Med. 2021;86:835-851.

* Arefeen Y, Xu J, Zhang M, Dong Z, Wang F, White J, Bilgic B, Adalsteinsson E. [Latent signal models: Learning compact representations of signal evolution for improved time-resolved, multi-contrast MRI](https://doi.org/10.1002/mrm.29657). Magn Reson Med 2023;90:483-501.

* Heydari A, Ahmadi A, Kim TH, Bilgic B. [Joint MAPLE: Accelerated joint T1 and T2* mapping with scan-specific self-supervised networks](https://doi.org/10.1002/mrm.29989). Magn Reson Med 2024.


## TODO:

* add VAE examples
* add transformer and diffusion models
* use sphinx to create documentation