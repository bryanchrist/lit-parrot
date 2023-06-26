#!/bin/bash
#SBATCH -A sds-phd-2022
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -c 1
#SBATCH -t 3-00:00:00
#SBATCH -J falcon40b_run1
#SBATCH -o job1-%A.out
#SBATCH -e job1-%A.err
module purge
module load singularity pytorch/1.8.1  # 2

singularity run --nv $CONTAINERDIR/pytorch-1.8.1.sif finetune/adapter.py # 3
