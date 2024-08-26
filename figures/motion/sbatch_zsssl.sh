#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --job-name=zsssl
#SBATCH --time=23:59:59
#SBATCH --mail-user=zgtan@med.umich.edu
#SBATCH --mail-type=ALL
#SBATCH --output=%x.%j.out
#
# do not export environment variables
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load cuda
module load python/3.9-anaconda
# conda init bash
eval "$(conda shell.bash hook)"
conda activate deepdwi

bash train.sh

bash test.sh