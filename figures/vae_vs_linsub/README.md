
# This folder creates figures for variational autoencoder (VAE).

The following bash blocks assume that you are under the `DeepDWI` directory.

## 1. train a VAE model

```bash
cd examples
python vae_training.py
```

## 2. comapre VAE and SVD

```bash
cd figures/vae_vs_linsub
gedit config.yaml  # modify 'checkpoint' and 'N_latent' to those from Step 1
python run_mse.py
python plot_mse.py  # this outputs 'vae_vs_linsub_mse.png'
```

## 3. compare VAE and SVD on in vivo data

```bash
cd figures/vae_vs_linsub
python run_brain.py
python plot_brain.py  # this outputs 'vae_vs_linsub_brain_diff_xx.png'
```

## 4. plot the learned data distribution

```bash
cd figures/vae_vs_linsub
python plot_latent.py  # this outputs 'vae_latent.png'
```