#!/bin/bash

for S in {0..87}; do
    echo $S
    python ../../examples/run_zsssl.py --config /figures/motion/config_zsssl_navi.yaml --mode test --checkpoint /examples/2024-05-22_zsssl_0.7mm_21-dir_R2x2_vol1_scan1_kdat_slice_040_norm-kdat-1.0_navi_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_best.pth --slice_idx $S
done

for S in {0..87}; do
    echo $S
    python ../../examples/run_zsssl.py --config /figures/motion/config_zsssl_self.yaml --mode test --checkpoint /examples/2024-05-23_zsssl_0.7mm_21-dir_R2x2_vol1_scan1_kdat_slice_040_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_best.pth --slice_idx $S
done