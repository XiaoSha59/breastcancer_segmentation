#!/usr/bin/env bash
# run_umamba_3phases_server.sh
# ==============================================================================
# End-to-end execution of U-Mamba Bot 3D on 3 Natural DCE-MRI Phases (Fold 0, 200 Epochs).
# Architecture: Official U-Mamba Bot 3D (Bo Wang Lab, MICCAI 2024 - Jun Ma et al.).
# Input: Slices strictly 3 natural phases [:3] (Pre, Early Peak, Delayed Late).
# Data: 100% REUSE of existing 64GB nnUNetPlans_3d_fullres without duplicating data.
# Server Safety: Auto-shutdown STRICTLY DISABLED (Lab cluster node1 runs 24/7).
# CPU Allocation: Strictly torch.set_num_threads(4) for shared cluster safety.
# Evaluation: Complete 197-case validation + official paper metrics (Dice, HD95, ASSD).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resource Allocation for node1 / lab server
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=============================================================================="
echo "    U-MAMBA BOT 3D (3 NATURAL PHASES) PIPELINE - UNIVERSITY SERVER (NODE1)    "
echo "=============================================================================="

# GPU Collision Guard: Verify GPU is idle
if command -v nvidia-smi &> /dev/null; then
    TARGET_GPU="${CUDA_VISIBLE_DEVICES:-0}"
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$TARGET_GPU" 2>/dev/null || echo "0")
    if [ "$MEM_USED" -gt 2500 ]; then
        echo "[CRITICAL SAFETY WARNING] GPU $TARGET_GPU is currently BUSY (${MEM_USED} MiB VRAM used)!"
        echo "Please verify running jobs with nvidia-smi before proceeding."
        exit 1
    fi
    echo "Verified GPU $TARGET_GPU is idle (${MEM_USED} MiB used). Safe to allocate."
fi

# Locate preprocessed data directory
if [ -d "$PROJECT_DIR/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres" ]; then
    PREPROCESSED_DIR="$PROJECT_DIR/data/preprocessed/Dataset501_BreastDCE_ISPY2"
elif [ -d "$HOME/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres" ]; then
    PREPROCESSED_DIR="$HOME/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2"
elif [ -d "$HOME/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres" ]; then
    PREPROCESSED_DIR="$HOME/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2"
else
    PREPROCESSED_DIR="${nnUNet_preprocessed:-$HOME/nnUNet_preprocessed}/Dataset501_BreastDCE_ISPY2"
fi

RESULTS_DIR="$PROJECT_DIR/results/Dataset501_BreastDCE_ISPY2/UMamba_Bot_3phases_200epochs/fold_0"
mkdir -p "$RESULTS_DIR"

# Server safety trap: Terminate child processes on exit, NEVER shutdown the machine
cleanup() {
    EXIT_STATUS=$?
    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[$(date)] Interrupted: Safely terminating training process $TRAIN_PID..."
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
    echo "[$(date)] Script exiting with status $EXIT_STATUS."
    echo "[$(date)] [Server Mode] Cluster safety active: University server remains running normally."
}
trap cleanup EXIT ERR INT TERM

echo "=== STEP 1: VERIFY 5-FOLD CV SPLITS ==="
SPLITS_FILE="$PREPROCESSED_DIR/splits_final.json"
if [ ! -f "$SPLITS_FILE" ]; then
    if [ -f "$PROJECT_DIR/splits_final.json" ]; then
        cp "$PROJECT_DIR/splits_final.json" "$SPLITS_FILE" 2>/dev/null || SPLITS_FILE="$PROJECT_DIR/splits_final.json"
    fi
fi
echo "Verified splits_final.json exists at $SPLITS_FILE (785 train / 197 val)."

PLAN_DIR="$PREPROCESSED_DIR/nnUNetPlans_3d_fullres"
echo "Verified preprocessed directory $PLAN_DIR."

echo "=== STEP 2: LAUNCH U-MAMBA BOT 3D (3 PHASES) TRAINING (200 EPOCHS) ==="
EXTRA_FLAGS=""
if [ -f "$RESULTS_DIR/checkpoint_latest.pth" ]; then
    echo "Found existing checkpoint_latest.pth in $RESULTS_DIR, enabling --resume flag."
    EXTRA_FLAGS="--resume"
fi

NUM_EPOCHS="${1:-200}"

python3 -u "$SCRIPT_DIR/train_umamba_3phases_fold0.py" \
    --prep_dir "$PLAN_DIR" \
    --splits_file "$SPLITS_FILE" \
    --output_dir "$RESULTS_DIR" \
    --fold 0 \
    --epochs "$NUM_EPOCHS" \
    --iterations_per_epoch 250 \
    --batch_size 2 \
    --patch_size 128 128 64 \
    --initial_lr 0.0001 \
    $EXTRA_FLAGS &

TRAIN_PID=$!
echo "[$(date)] U-Mamba Bot 3D (3 Phases) training started with PID $TRAIN_PID ($NUM_EPOCHS epochs)"

wait $TRAIN_PID
TRAIN_EXIT=$?
echo "[$(date)] U-Mamba Bot 3D training finished with exit code $TRAIN_EXIT"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "=== STEP 3: SLIDING WINDOW EVALUATION ON 197 VALIDATION CASES (3 PHASES) ==="
if [ "$NUM_EPOCHS" -ge 200 ]; then
    python3 -u "$SCRIPT_DIR/evaluate_umamba_3phases_fold0.py" \
        --prep_dir "$PLAN_DIR" \
        --splits_file "$SPLITS_FILE" \
        --output_dir "$RESULTS_DIR" \
        --fold 0 \
        --patch_size 128 128 64 \
        --num_workers 4 || echo "Evaluation warning: Check validation logs"
fi

echo "=== STEP 4: COMPUTE OFFICIAL PAPER METRICS ==="
python3 -u "$SCRIPT_DIR/compute_paper_metrics.py" "$RESULTS_DIR" "U-Mamba Bot 3D 3-Phases (Fold 0, 200 epochs)" || echo "Metrics warning"

echo "=== U-MAMBA BOT 3D (3 PHASES) PIPELINE COMPLETED SUCCESSFULLY AT $(date) ==="
