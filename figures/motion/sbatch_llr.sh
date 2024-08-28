#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --job-name=JETS_0.7mm_21dir
#SBATCH --time=23:59:59
#SBATCH --mail-user=zgtan@med.umich.edu
#SBATCH --mail-type=ALL
#
# do not export environment variables
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load cuda
module load python/3.9-anaconda
# conda init bash
eval "$(conda shell.bash hook)"
conda activate sigpy

python ../../examples/run_llr.py --prefix 0.7mm_21-dir_R2x2_vol1_scan1 --slice_idx 0 --slice_inc 88 --jets --admm_lamda 0.002 --admm_rho 0.05
