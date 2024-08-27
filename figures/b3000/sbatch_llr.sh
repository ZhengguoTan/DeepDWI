#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --job-name=JETS_1.0mm_126dir
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


# 1.0 mm - 126 dir
# python run_llr.py --prefix 1.0mm_126-dir_R3x3_ --slice_idx 0 --slice_inc 38 --muse

python run_llr.py --prefix 1.0mm_126-dir_R3x3 --slice_idx 0 --slice_inc 38 --jets --split 3 --admm_lamda 0.002 --admm_rho 0.05
