#!/bin/bash
#
# DDP Training Script for Toroidal Token Generation
# Optimized for 8x H100 GPUs
#
# Usage:
#   ./run_ddp_h100.sh [embeddings_path]
#
# Example:
#   ./run_ddp_h100.sh /path/to/dolma_embeddings.txt
#

set -e

# ============================================================================
# Configuration
# ============================================================================

# Number of GPUs
NUM_GPUS=8

# Embeddings path (can be overridden by argument)
EMBEDDINGS_PATH="${1:-/path/to/dolma_300_2024_1.2M.100_combined.txt}"

# Model configuration
N_THETA=24          # Cells around tube
N_PHI=48            # Cells around torus (total cells = N_THETA * N_PHI = 1152)
MAX_STEPS=12        # Propagation steps
HIDDEN_DIM=512      # Hidden dimension

# Training configuration
EPOCHS=100
BATCH_SIZE=256      # Per GPU (effective batch = 256 * 8 = 2048)
LR=1e-3
WEIGHT_DECAY=0.01
NUM_WORKERS=4       # DataLoader workers per GPU

# Data configuration
MAX_EMBEDDINGS=50000
N_PAIRS=500000

# Output
CHECKPOINT_DIR="checkpoints_ddp"
LOG_DIR="logs"

# ============================================================================
# Environment Setup for H100 GPUs
# ============================================================================

# Enable TF32 for faster matrix operations on H100
export NVIDIA_TF32_OVERRIDE=1

# Optimize NCCL for multi-GPU
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Use all available H100 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ============================================================================
# Create directories
# ============================================================================

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}

# ============================================================================
# Print configuration
# ============================================================================

echo "============================================================"
echo "Toroidal Token Generation - DDP Training on 8x H100"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  GPUs: ${NUM_GPUS}"
echo "  Embeddings: ${EMBEDDINGS_PATH}"
echo "  Max embeddings: ${MAX_EMBEDDINGS}"
echo "  Token pairs: ${N_PAIRS}"
echo ""
echo "Model:"
echo "  Torus size: ${N_THETA} x ${N_PHI} = $((N_THETA * N_PHI)) cells"
echo "  Propagation steps: ${MAX_STEPS}"
echo "  Hidden dim: ${HIDDEN_DIM}"
echo ""
echo "Training:"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size per GPU: ${BATCH_SIZE}"
echo "  Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "  Learning rate: ${LR}"
echo "  Weight decay: ${WEIGHT_DECAY}"
echo ""
echo "============================================================"
echo ""

# ============================================================================
# Check for embeddings file
# ============================================================================

if [ ! -f "${EMBEDDINGS_PATH}" ]; then
    echo "ERROR: Embeddings file not found: ${EMBEDDINGS_PATH}"
    echo "Please provide path to embeddings file as first argument"
    echo ""
    echo "Usage: ./run_ddp_h100.sh /path/to/embeddings.txt"
    exit 1
fi

# ============================================================================
# Run DDP Training
# ============================================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "Starting training at $(date)"
echo "Logging to: ${LOG_FILE}"
echo ""

# Launch with torchrun for DDP
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
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
    --num-workers ${NUM_WORKERS} \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "Training complete at $(date)"
echo "Checkpoints saved to: ${CHECKPOINT_DIR}"
echo "Log saved to: ${LOG_FILE}"
