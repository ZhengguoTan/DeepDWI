#!/bin/bash

for S in {0..37}; do
    echo $S
    python ../../examples/run_zsssl.py --config /figures/b3000/config_zsssl.yaml --mode test --checkpoint /examples/2024-08-26_zsssl_1.0mm_126-dir_R3x3_kdat_slice_000_norm-kdat-1.0_self_ResNet2D_ResBlock-12_kernel-3_ADMM_08_lamda-0.050_Adam_lr-0.000500_MixL1L2Loss/zsssl_best.pth --slice_idx $S
done