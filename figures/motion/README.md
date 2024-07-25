# This folder creates figures for regularzations:

* Locally Low Rank (LLR)
* Variatinal AutoEncoder (VAE)
* Zero-Shot Self-Supervised Learning (ZSSSL)

## 0. setup the directory of the `DeepDWI` folder in the terminal:

```bash
export DWIDIR=/path/to/DeepDWI
cd ${DWIDIR}/figures/motion
```

## 1. run the reconstruction with the LLR regularization:

```bash
python run_llr_regularization.py
```

## 2. run the reconstruction with the learned VAE model as regularization:

```bash
python run_vae_regularization.py
```

## 3. run ZSSSL

### train:

```bash
python ../../examples/run_zsssl.py --config /figures/motion/config_zsssl_navi.yaml --mode train
python ../../examples/run_zsssl.py --config /figures/motion/config_zsssl_navi.yaml --mode train
```

### test:

```bash
bash test.sh
```

## 4. plot results

```bash
python plot.py
```

<!-- <p align="center">
  <img alt="Light" src="regularizations.png" width="100%">
</p> -->
