# This folder creates figures for regularzations:

* Locally Low Rank (LLR)
* Zero-Shot Self-Supervised Learning (ZSSSL)

## 0. setup the directory of the `DeepDWI` folder in the terminal:

```bash
export DWIDIR=/path/to/DeepDWI
cd ${DWIDIR}/figures/motion_self
```

## x. plot results

```bash
python plot.py
```
<p align="center">
  <img alt="Light" src="0.7mm_dwi_sg_muse.png" width="45%">
  <img alt="Light" src="0.7mm_dwi_sg_muse_ave.png" width="45%">
  <img alt="Light" src="0.7mm_dwi_sg_jets.png" width="45%">
  <img alt="Light" src="0.7mm_dwi_sg_jets_ave.png" width="45%">
  <img alt="Light" src="0.7mm_dwi_sg_zsssl.png" width="45%">
  <img alt="Light" src="0.7mm_dwi_sg_zsssl_ave.png" width="45%">
</p>

```bash
python plot_vol3.py
```
<p align="center">
  <img alt="Light" src="0.7mm_dwi_vol3.png" width="100%">
</p>