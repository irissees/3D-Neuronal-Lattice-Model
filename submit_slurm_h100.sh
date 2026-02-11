#!/bin/bash
#SBATCH --job-name=toroidal-ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

#
# SLURM Script for Toroidal Token Generation
# Optimized for 8x H100 GPUs on HPC clusters
#
# Usage:
#   sbatch submit_slurm_h100.sh /path/to/embeddings.txt
#

# ============================================================================
# Configuration
# ============================================================================

EMBEDDINGS_PATH="${1:-/path/to/dolma_embeddings.txt}"

# Model
N_THETA=24
N_PHI=48
MAX_STEPS=12
HIDDEN_DIM=512

# Training
EPOCHS=100
BATCH_SIZE=256
LR=1e-3
WEIGHT_DECAY=0.01

# Data
MAX_EMBEDDINGS=50000
N_PAIRS=500000

# ============================================================================
# Environment
# ============================================================================

# Load modules (adjust for your cluster)
module load cuda/12.1
module load pytorch/2.1

# Or activate conda environment
# source activate pytorch

# H100 optimizations
export NVIDIA_TF32_OVERRIDE=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ============================================================================
# Create directories
# ============================================================================

mkdir -p checkpoints_ddp
mkdir -p logs

# ============================================================================
# Print info
# ============================================================================

echo "============================================================"
echo "SLURM Job: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS_ON_NODE}"
echo "============================================================"
echo ""

# ============================================================================
# Run Training
# ============================================================================

srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train_toroidal_ddp.py \
    --embeddings-path "${EMBEDDINGS_PATH}" \
    --max-embeddings ${MAX_EMBEDDINGS} \
    --n-pairs ${N_PAIRS} \
    --n-theta ${N_THETA} \
    --n-phi ${N_PHI} \
    --max-steps ${MAX_STEPS} \
    --hidden-dim ${HIDDEN_DIM} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --num-workers 4 \
    --checkpoint-dir checkpoints_ddp

echo "Training complete!"
