#!/bin/bash
#SBATCH --job-name=toroidal-hpo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/toroidal_hpo_%j.out
#SBATCH --error=slurm_logs/toroidal_hpo_%j.err

#
# SLURM Script for PyTorch Lightning HPO on 8x H100 GPUs
#
# Usage:
#   sbatch submit_lightning_hpo.sh                    # Standard training
#   sbatch submit_lightning_hpo.sh --hpo              # HPO mode
#   sbatch submit_lightning_hpo.sh --best             # Use best HPO params
#

set -e

# ============================================================================
# Configuration
# ============================================================================

MODE="${1:-train}"
EMBEDDINGS_FILE="${EMBEDDINGS_FILE:-dolma_300_2024_1.2M.100_combined.txt}"

# HPO Settings
N_TRIALS=${N_TRIALS:-50}
HPO_EPOCHS=${HPO_EPOCHS:-10}

# Training Settings  
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-256}

# ============================================================================
# Environment Setup
# ============================================================================

echo "============================================================"
echo "SLURM Job: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS}"
echo "Start: $(date)"
echo "============================================================"

# Create log directory
mkdir -p slurm_logs

# Load modules (adjust based on your cluster)
module load cuda/12.1 2>/dev/null || true
module load pytorch/2.0 2>/dev/null || true

# H100 specific optimizations
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# NCCL settings
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_GID_INDEX=3
export NCCL_ALGO=Ring
export NCCL_P2P_DISABLE=0

export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

# ============================================================================
# Setup Directories
# ============================================================================

mkdir -p checkpoints_lightning
mkdir -p logs_lightning
mkdir -p hpo_results

# ============================================================================
# Run Training
# ============================================================================

echo ""
echo "Starting PyTorch Lightning training..."
echo "Mode: ${MODE}"
echo "Embeddings: ${EMBEDDINGS_FILE}"
echo ""

case "${MODE}" in
    "--hpo"|"hpo")
        echo "Running Hyperparameter Optimization with ${N_TRIALS} trials"
        python train_toroidal_lightning.py \
            --embeddings-path "${EMBEDDINGS_FILE}" \
            --devices 8 \
            --hpo \
            --n-trials ${N_TRIALS} \
            --hpo-epochs ${HPO_EPOCHS}
        ;;
    
    "--best"|"best")
        echo "Training with best HPO parameters"
        python train_toroidal_lightning.py \
            --embeddings-path "${EMBEDDINGS_FILE}" \
            --devices 8 \
            --use-best-params \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH_SIZE}
        ;;
    
    *)
        echo "Standard training"
        python train_toroidal_lightning.py \
            --embeddings-path "${EMBEDDINGS_FILE}" \
            --devices 8 \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH_SIZE}
        ;;
esac

# ============================================================================
# Cleanup
# ============================================================================

echo ""
echo "============================================================"
echo "Job Complete!"
echo "End: $(date)"
echo "============================================================"
echo ""
echo "Results:"
echo "  Checkpoints: ./checkpoints_lightning/"
echo "  Logs: ./logs_lightning/"
echo "  HPO Results: ./hpo_results/"
