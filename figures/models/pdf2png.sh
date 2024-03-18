#!/bin/bash -l

pdftoppm -r 300 plot_sms_diff_model/fwd.pdf fwd -png
pdftoppm -r 300 plot_nn/vae.pdf vae -png