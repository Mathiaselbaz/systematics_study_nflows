#!/bin/env bash

#SBATCH --time=12:00:00
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --gres=gpu:1,VramPerGpu:10G
#SBATCH --mem=50G
#SBATCH --output=/home/shares/sanchezf/gundam_n_flow/project/logs_slurm/jupyter_baobab_gpu.out
#SBATCH --job-name='interactive'
#SBATCH --chdir=/home/shares/sanchezf/gundam_n_flow

export IMAGE_PATH="/home/shares/sanchezf/gundam_n_flow/project/ml_image.simg"
module load GCC/9.3.0 Singularity/3.7.3-Go-1.14

echo Write in local terminal  " ssh -N -t username@login1.baobab.hpc.unige.ch -L 8888:$SLURMD_NODENAME:8888 "

srun singularity exec --nv -B /home,/srv $IMAGE_PATH jupyter notebook --no-browser --allow-root --ip=$SLURMD_NODENAME --notebook-dir=$RUNDIR


