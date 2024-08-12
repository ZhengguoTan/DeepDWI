#!/bin/bash

# use navigator data to estimate shot phase
python ../../examples/run_zsssl.py --config /figures/motion/config_zsssl_navi.yaml --mode train

# use self-gating data to estimate shot phase
python ../../examples/run_zsssl.py --config /figures/motion/config_zsssl_self.yaml --mode train