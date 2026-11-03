#!/bin/bash
#SBATCH --job-name=optional_A
#SBATCH --account=ece_gy_7123-2025sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --export=ALL
#SBATCH --requeue

# Submit from the project folder:  cd /scratch/$USER/DL && sbatch hpc/optional_A.sbatch
# Check the account / partition names above against the ones your course gave you.
ENV_DIR=/scratch/$USER/env
PROJECT_DIR=/scratch/$USER/DL

module purge

singularity exec --nv --bind /scratch \
    --overlay $ENV_DIR/overlay-15GB-500K.ext3:ro \
    $ENV_DIR/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; cd $PROJECT_DIR; nvidia-smi; bash hpc/run_optional_A.sh"
