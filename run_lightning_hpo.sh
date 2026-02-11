#!/bin/bash
#
# PyTorch Lightning HPO Training Script for 8x H100 GPUs
# Usage:
#   ./run_lightning_hpo.sh <embeddings_file>              # Normal training
#   ./run_lightning_hpo.sh <embeddings_file> --hpo        # HPO mode
#   ./run_lightning_hpo.sh <embeddings_file> --best       # Use best params from HPO

set -e

# ============================================================================
# Configuration
# ============================================================================

NUM_GPUS=${NUM_GPUS:-8}
EMBEDDINGS_PATH="${1:-dolma_300_2024_1.2M.100_combined.txt}"
MODE="${2:-train}"  # train, hpo, or best

# Lattice parameters (used if not in HPO mode)
N_THETA=${N_THETA:-24}
N_PHI=${N_PHI:-48}
MAX_STEPS=${MAX_STEPS:-12}

# Network architecture parameters
HIDDEN_DIM=${HIDDEN_DIM:-512}
ENCODER_LAYERS=${ENCODER_LAYERS:-2}
OUTPUT_LAYERS=${OUTPUT_LAYERS:-2}
DROPOUT=${DROPOUT:-0.1}

# Training parameters
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_STEPS=${WARMUP_STEPS:-100}
PATIENCE=${PATIENCE:-10}
ACCUMULATE_GRAD=${ACCUMULATE_GRAD:-1}
NUM_WORKERS=${NUM_WORKERS:-4}

# Data parameters
MAX_EMBEDDINGS=${MAX_EMBEDDINGS:-50000}
N_PAIRS=${N_PAIRS:-500000}

# HPO parameters
N_TRIALS=${N_TRIALS:-50}
HPO_EPOCHS=${HPO_EPOCHS:-10}
HPO_TIMEOUT=${HPO_TIMEOUT:-}  # Empty = no timeout

# ============================================================================
# Environment Setup for H100
# ============================================================================

echo "============================================================"
echo "PyTorch Lightning Toroidal Training"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  GPUs: ${NUM_GPUS}"
echo "  Embeddings: ${EMBEDDINGS_PATH}"
echo "  Mode: ${MODE}"
echo ""

# H100 specific optimizations
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# NCCL settings for H100
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_GID_INDEX=3
export NCCL_ALGO=Ring
export NCCL_P2P_DISABLE=0

# Memory settings
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

# ============================================================================
# Create Directories
# ============================================================================

mkdir -p checkpoints_lightning
mkdir -p logs_lightning
mkdir -p hpo_results

# ============================================================================
# Build Command
# ============================================================================

# Base command
CMD="python train_toroidal_lightning.py \
    --embeddings-path ${EMBEDDINGS_PATH} \
    --max-embeddings ${MAX_EMBEDDINGS} \
    --n-pairs ${N_PAIRS} \
    --devices ${NUM_GPUS} \
    --num-workers ${NUM_WORKERS}"

# Add mode-specific arguments
case "${MODE}" in
    "--hpo"|"hpo")
        echo "Mode: Hyperparameter Optimization"
        echo "  Trials: ${N_TRIALS}"
        echo "  Epochs per trial: ${HPO_EPOCHS}"
        CMD="${CMD} \
            --hpo \
            --n-trials ${N_TRIALS} \
            --hpo-epochs ${HPO_EPOCHS}"
        if [ -n "${HPO_TIMEOUT}" ]; then
            CMD="${CMD} --timeout ${HPO_TIMEOUT}"
        fi
        ;;
    
    "--best"|"best")
        echo "Mode: Training with best HPO parameters"
        CMD="${CMD} \
            --use-best-params \
            --epochs ${EPOCHS} \
            --patience ${PATIENCE} \
            --accumulate-grad ${ACCUMULATE_GRAD}"
        ;;
    
    *)
        echo "Mode: Standard training"
        CMD="${CMD} \
            --n-theta ${N_THETA} \
            --n-phi ${N_PHI} \
            --max-steps ${MAX_STEPS} \
            --hidden-dim ${HIDDEN_DIM} \
            --encoder-layers ${ENCODER_LAYERS} \
            --output-layers ${OUTPUT_LAYERS} \
            --dropout ${DROPOUT} \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH_SIZE} \
            --lr ${LR} \
            --weight-decay ${WEIGHT_DECAY} \
            --warmup-steps ${WARMUP_STEPS} \
            --patience ${PATIENCE} \
            --accumulate-grad ${ACCUMULATE_GRAD}"
        ;;
esac

echo ""
echo "============================================================"
echo "Starting Training..."
echo "============================================================"
echo ""
echo "Command: ${CMD}"
echo ""

# ============================================================================
# Run Training
# ============================================================================

${CMD}

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  Checkpoints: ./checkpoints_lightning/"
echo "  Logs: ./logs_lightning/"
echo "  HPO Results: ./hpo_results/"
