#!/usr/bin/env bash
#SBATCH --job-name=looped-wiki
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Scheduler owns CUDA_VISIBLE_DEVICES; do not override it here.
./scripts/check.sh --require-cuda
./scripts/train.sh "$@"
