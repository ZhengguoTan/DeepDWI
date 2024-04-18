# This folder creates figures for regularzations:

* VAE
* Zero-Shot Self-Supervised Learning (ZSSSL)

## 0. setup the directory of the `DeepDWI` folder in the terminal:

```bash
export DWIDIR=/path/to/DeepDWI
cd ${DWIDIR}/figures/regularizations
```

## 1. run the reconstruction with the learned VAE model as regularization:

```bash
python run_vae_regularization.py
```

## 2. run ZSSSL

```bash
python ../../examples/run_zsssl.py --config /figures/regularizations/config_zsssl.yaml --mode train
python ../../examples/run_zsssl.py --config /figures/regularizations/config_zsssl.yaml --mode test
```

## 3. plot results

```bash
python plot.py
```