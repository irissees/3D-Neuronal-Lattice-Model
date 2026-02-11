#!/bin/bash
#
# Training script for Lattice Autoencoder with MNIST using PyTorch Lightning
#
# Features:
#   - Configurable hyperparameters
#   - Automatic checkpoint saving
#   - Post-training evaluation with reconstruction visualizations
#   - Supports MPS (Apple Silicon), CUDA, and CPU
#
# Usage:
#   ./train_mnist.sh                    # Run with defaults
#   ./train_mnist.sh --epochs 200       # Custom epochs
#   ./train_mnist.sh --devices 2 --strategy ddp  # Multi-GPU DDP
#

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Default values (can be overridden via command line)
LAYERS=64
HEX_RADIUS=14
TRAIN_SAMPLES=""         # Empty = use all (60k)
TEST_SAMPLES=""          # Empty = use all (10k)
BATCH_SIZE=64
LEARNING_RATE=0.001
PROPAGATION_STEPS=5
MAX_EPOCHS=1000
ACCELERATOR="gpu"       # auto, mps, gpu, cpu
DEVICES=8
STRATEGY="ddp"          # auto, ddp, ddp_spawn
NUM_WORKERS=8
PRECISION="bf16-mixed"           # 32, 16-mixed, bf16-mixed

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints"
EVAL_OUTPUT_DIR="${SCRIPT_DIR}/evaluation_results"
LOG_DIR="${SCRIPT_DIR}/logs"

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="lattice_mnist_${TIMESTAMP}"

# =============================================================================
# Parse command line arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --layers)
            LAYERS="$2"
            shift 2
            ;;
        --hex_radius)
            HEX_RADIUS="$2"
            shift 2
            ;;
        --train_samples)
            TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --test_samples)
            TEST_SAMPLES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --propagation_steps)
            PROPAGATION_STEPS="$2"
            shift 2
            ;;
        --epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --accelerator)
            ACCELERATOR="$2"
            shift 2
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --layers N              Lattice layers (default: 2)"
            echo "  --hex_radius N          Hexagonal radius (default: 8)"
            echo "  --train_samples N       Training samples (default: all)"
            echo "  --test_samples N        Test samples (default: all)"
            echo "  --batch_size N          Batch size (default: 64)"
            echo "  --lr FLOAT              Learning rate (default: 0.001)"
            echo "  --propagation_steps N   Propagation steps (default: 5)"
            echo "  --epochs N              Max epochs (default: 100)"
            echo "  --accelerator TYPE      auto, mps, gpu, cpu (default: auto)"
            echo "  --devices N             Number of devices (default: 1)"
            echo "  --strategy TYPE         auto, ddp, ddp_spawn (default: auto)"
            echo "  --num_workers N         Data loading workers (default: 4)"
            echo "  --precision TYPE        32, 16-mixed, bf16-mixed (default: 32)"
            echo "  --checkpoint_dir DIR    Checkpoint directory"
            echo "  --run_name NAME         Run name for logging"
            echo "  --help                  Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

echo "============================================================"
echo "Lattice Autoencoder - MNIST Training"
echo "============================================================"
echo ""
echo "Run name: ${RUN_NAME}"
echo "Timestamp: ${TIMESTAMP}"
echo ""
echo "Configuration:"
echo "  Lattice: ${LAYERS} layers, hex_radius=${HEX_RADIUS}"
echo "  Training: ${MAX_EPOCHS} epochs, batch_size=${BATCH_SIZE}, lr=${LEARNING_RATE}"
echo "  Propagation steps: ${PROPAGATION_STEPS}"
echo "  Accelerator: ${ACCELERATOR}, Devices: ${DEVICES}, Strategy: ${STRATEGY}"
echo "  Precision: ${PRECISION}"
echo ""

# Create directories
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${EVAL_OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Activate virtual environment if it exists
if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "${SCRIPT_DIR}/.venv/bin/activate"
fi

# Check Python
PYTHON_CMD="python3"
if ! command -v ${PYTHON_CMD} &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Using Python: $(${PYTHON_CMD} --version)"

# Install dependencies from requirements.txt
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    if command -v uv &> /dev/null; then
        uv pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet
    else
        ${PYTHON_CMD} -m pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet
    fi
    echo "Dependencies installed."
fi
echo ""

# =============================================================================
# Training
# =============================================================================

echo "============================================================"
echo "Starting Training"
echo "============================================================"

# Build training command
TRAIN_CMD="${PYTHON_CMD} ${SCRIPT_DIR}/lightning_trainer.py"
TRAIN_CMD+=" --layers ${LAYERS}"
TRAIN_CMD+=" --hex_radius ${HEX_RADIUS}"
TRAIN_CMD+=" --batch_size ${BATCH_SIZE}"
TRAIN_CMD+=" --lr ${LEARNING_RATE}"
TRAIN_CMD+=" --propagation_steps ${PROPAGATION_STEPS}"
TRAIN_CMD+=" --epochs ${MAX_EPOCHS}"
TRAIN_CMD+=" --accelerator ${ACCELERATOR}"
TRAIN_CMD+=" --devices ${DEVICES}"
TRAIN_CMD+=" --strategy ${STRATEGY}"
TRAIN_CMD+=" --num_workers ${NUM_WORKERS}"

if [ -n "${TRAIN_SAMPLES}" ]; then
    TRAIN_CMD+=" --train_samples ${TRAIN_SAMPLES}"
fi

if [ -n "${TEST_SAMPLES}" ]; then
    TRAIN_CMD+=" --test_samples ${TEST_SAMPLES}"
fi

echo "Command: ${TRAIN_CMD}"
echo ""

# Run training
TRAIN_START_TIME=$(date +%s)
${TRAIN_CMD} 2>&1 | tee "${LOG_DIR}/${RUN_NAME}_training.log"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
TRAIN_END_TIME=$(date +%s)
TRAIN_DURATION=$((TRAIN_END_TIME - TRAIN_START_TIME))

if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed with exit code ${TRAIN_EXIT_CODE}"
    exit ${TRAIN_EXIT_CODE}
fi

echo ""
echo "Training completed in ${TRAIN_DURATION} seconds"
echo ""

# =============================================================================
# Find best checkpoint
# =============================================================================

echo "============================================================"
echo "Finding Best Checkpoint"
echo "============================================================"

# Find the latest/best checkpoint
BEST_CHECKPOINT=""

# Try to find checkpoint with lowest val_loss in filename
if ls ${CHECKPOINT_DIR}/lattice-*.ckpt 1> /dev/null 2>&1; then
    BEST_CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/lattice-*.ckpt | head -n 1)
fi

# Fallback to last.ckpt
if [ -z "${BEST_CHECKPOINT}" ] && [ -f "${CHECKPOINT_DIR}/last.ckpt" ]; then
    BEST_CHECKPOINT="${CHECKPOINT_DIR}/last.ckpt"
fi

if [ -z "${BEST_CHECKPOINT}" ]; then
    echo "WARNING: No checkpoint found. Skipping evaluation."
else
    echo "Best checkpoint: ${BEST_CHECKPOINT}"
    echo ""

    # =============================================================================
    # Post-Training Evaluation
    # =============================================================================

    echo "============================================================"
    echo "Running Post-Training Evaluation"
    echo "============================================================"

    EVAL_CMD="${PYTHON_CMD} ${SCRIPT_DIR}/evaluate_model.py"
    EVAL_CMD+=" --checkpoint ${BEST_CHECKPOINT}"
    EVAL_CMD+=" --output_dir ${EVAL_OUTPUT_DIR}/${RUN_NAME}"
    EVAL_CMD+=" --layers ${LAYERS}"
    EVAL_CMD+=" --hex_radius ${HEX_RADIUS}"
    EVAL_CMD+=" --propagation_steps ${PROPAGATION_STEPS}"
    EVAL_CMD+=" --test_samples 1000"
    EVAL_CMD+=" --device cpu"  # Use CPU for evaluation to avoid memory issues

    echo "Command: ${EVAL_CMD}"
    echo ""

    ${EVAL_CMD} 2>&1 | tee "${LOG_DIR}/${RUN_NAME}_evaluation.log"
    EVAL_EXIT_CODE=${PIPESTATUS[0]}

    if [ ${EVAL_EXIT_CODE} -ne 0 ]; then
        echo ""
        echo "WARNING: Evaluation failed with exit code ${EVAL_EXIT_CODE}"
    fi
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo "Training Run Complete"
echo "============================================================"
echo ""
echo "Run name: ${RUN_NAME}"
echo "Training duration: ${TRAIN_DURATION} seconds"
echo ""
echo "Outputs:"
echo "  Checkpoints: ${CHECKPOINT_DIR}/"
if [ -n "${BEST_CHECKPOINT}" ]; then
    echo "  Best checkpoint: ${BEST_CHECKPOINT}"
fi
echo "  Evaluation results: ${EVAL_OUTPUT_DIR}/${RUN_NAME}/"
echo "  Training log: ${LOG_DIR}/${RUN_NAME}_training.log"
echo "  Evaluation log: ${LOG_DIR}/${RUN_NAME}_evaluation.log"
echo ""

# List checkpoint files
echo "Checkpoint files:"
ls -lh ${CHECKPOINT_DIR}/*.ckpt 2>/dev/null || echo "  No checkpoints found"
echo ""

# List evaluation outputs
if [ -d "${EVAL_OUTPUT_DIR}/${RUN_NAME}" ]; then
    echo "Evaluation outputs:"
    ls -lh ${EVAL_OUTPUT_DIR}/${RUN_NAME}/ 2>/dev/null || echo "  No outputs found"
fi

echo ""
echo "Done!"
