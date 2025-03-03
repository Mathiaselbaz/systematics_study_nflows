#!/bin/env bash
#SBATCH --job-name=train_nf
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1,VramPerGpu:40G
#SBATCH --array=1-20%10
#SBATCH --output=logs_slurm/train_nf_%a.out
#SBATCH --mem=200G

module load GCCcore/8.2.0
module load Singularity/3.4.0-Go-1.12

export IMAGE_PATH="/home/shares/sanchezf/gundam_n_flow/project/ml_image.simg"


SCRIPT=train_nf.py
srun nvidia-smi

echo "Starting job: " `date`
srun singularity exec --nv -B /home,/srv $IMAGE_PATH  python ${SCRIPT}
echo "Job done: " `date`
