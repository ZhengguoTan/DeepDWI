#!/bin/bash

for S in {0..87}; do
    echo $S
    python ../../examples/run_zsssl.py --config /figures/motion/config_zsssl_navi.yaml --mode test --slice_idx $S
done

for S in {0..87}; do
    echo $S
    python ../../examples/run_zsssl.py --config /figures/motion/config_zsssl_self.yaml --mode test --slice_idx $S
done