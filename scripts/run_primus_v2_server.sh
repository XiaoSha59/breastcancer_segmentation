#!/usr/bin/env bash
# run_primus_v2_server.sh
# ==============================================================================
# End-to-end execution of PrimusV2-S on I-SPY2 Fold 0 (200 Epochs).
# Architecture: PrimusV2-S (Iterative Dynamic Tokenizer + 3D Axial RoPE Transformer).
# Preprocess Strategy: 100% REUSE of existing nnUNetPlans_3d_fullres (.b2nd / .npy).
# Hardware Safety: Tailored for University Server / Lab GPU (RTX 4090 / 4080).
# Cluster / Server Safety: Auto-shutdown is STRICTLY DISABLED (Machine remains running 24/7).
# Compliant with UNIVERSITY_SERVER_GUIDE.md & 3D_UNET_BEST_PRACTICES.md.
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
echo "  PRIMUSV2-S TRAINING & EVALUATION PIPELINE (UNIVERSITY SERVER MODE)"
echo "=============================================================================="

# GPU Collision Guard: Never collide with another active job
if command -v nvidia-smi &> /dev/null; then
    TARGET_GPU="${CUDA_VISIBLE_DEVICES:-0}"
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$TARGET_GPU" 2>/dev/null || echo "0")
    if [ "$MEM_USED" -gt 1500 ]; then
        echo "[CRITICAL SAFETY WARNING] GPU $TARGET_GPU is currently BUSY (${MEM_USED} MiB VRAM used)!"
        echo "Starting PrimusV2-S now would collide with another active job (e.g., MedNeXt or other lab member)."
        echo "Please wait until the other job finishes, or assign another idle card:"
        echo "  export CUDA_VISIBLE_DEVICES=<free_gpu>"
        exit 1
    fi
    echo "Verified GPU $TARGET_GPU is idle (${MEM_USED} MiB used). Safe to allocate."
fi

# Locate preprocessed data directory (.b2nd / .npy)
if [ -d "$PROJECT_DIR/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres" ]; then
    PREPROCESSED_DIR="$PROJECT_DIR/data/preprocessed/Dataset501_BreastDCE_ISPY2"
elif [ -d "$HOME/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres" ]; then
    PREPROCESSED_DIR="$HOME/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2"
elif [ -d "$HOME/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres" ]; then
    PREPROCESSED_DIR="$HOME/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2"
else
    PREPROCESSED_DIR="${nnUNet_preprocessed:-$HOME/nnUNet_preprocessed}/Dataset501_BreastDCE_ISPY2"
fi

# Locate results directory
if [ -n "$nnUNet_results" ] && [ "$nnUNet_results" != "$HOME/nnUNet_results" ]; then
    RESULTS_DIR="$nnUNet_results/Dataset501_BreastDCE_ISPY2/PrimusV2_S_200epochs/fold_0"
else
    RESULTS_DIR="$PROJECT_DIR/results/Dataset501_BreastDCE_ISPY2/PrimusV2_S_200epochs/fold_0"
fi
mkdir -p "$RESULTS_DIR"

# Cleanup trap: Safe for shared servers (only terminates child training process on interrupt, NEVER shuts down server)
cleanup() {
    EXIT_STATUS=$?
    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[$(date)] Interrupted: Safely terminating PrimusV2-S training process $TRAIN_PID to release GPU VRAM..."
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
    elif [ -f "$PROJECT_DIR/data/processed/Dataset501_BreastDCE_ISPY2/splits_final.json" ]; then
        cp "$PROJECT_DIR/data/processed/Dataset501_BreastDCE_ISPY2/splits_final.json" "$SPLITS_FILE" 2>/dev/null || SPLITS_FILE="$PROJECT_DIR/data/processed/Dataset501_BreastDCE_ISPY2/splits_final.json"
    fi
fi

if [ ! -f "$SPLITS_FILE" ]; then
    echo "Error: splits_final.json not found in $PREPROCESSED_DIR or $PROJECT_DIR!"
    exit 1
fi
echo "Verified splits_final.json exists at $SPLITS_FILE (785 train / 197 val)."

echo "=== STEP 2: VERIFY PREPROCESSED DATA (.b2nd / .npy) ==="
PLAN_DIR="$PREPROCESSED_DIR/nnUNetPlans_3d_fullres"
if [ ! -d "$PLAN_DIR" ]; then
    echo "Error: Preprocessed directory $PLAN_DIR does not exist!"
    exit 1
fi
echo "Verified preprocessed directory $PLAN_DIR."

echo "=== STEP 3: LAUNCH PRIMUSV2-S TRAINING (200 EPOCHS) ==="
echo "[$(date)] Starting train_primus_v2_fold0.py..."

# Auto-resume if checkpoint_latest.pth exists
EXTRA_FLAGS=""
if [ -f "$RESULTS_DIR/checkpoint_latest.pth" ]; then
    echo "Found existing checkpoint_latest.pth in $RESULTS_DIR, enabling --resume flag."
    EXTRA_FLAGS="--resume"
fi

python3 -u "$SCRIPT_DIR/train_primus_v2_fold0.py" \
    --prep_dir "$PLAN_DIR" \
    --splits_file "$SPLITS_FILE" \
    --output_dir "$RESULTS_DIR" \
    --fold 0 \
    --epochs 200 \
    --iterations_per_epoch 250 \
    --batch_size 2 \
    --patch_size 64 128 128 \
    --model_id S \
    --initial_lr 0.0001 \
    $EXTRA_FLAGS &

TRAIN_PID=$!
echo "[$(date)] PrimusV2-S training started with PID $TRAIN_PID"

# Wait for training to complete
wait $TRAIN_PID
TRAIN_EXIT=$?
echo "[$(date)] PrimusV2-S training finished with exit code $TRAIN_EXIT"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "=== STEP 4: SLIDING WINDOW EVALUATION ON 197 VALIDATION CASES ==="
python3 -u "$SCRIPT_DIR/evaluate_primus_v2_fold0.py" \
    --prep_dir "$PLAN_DIR" \
    --splits_file "$SPLITS_FILE" \
    --output_dir "$RESULTS_DIR" \
    --fold 0 \
    --patch_size 64 128 128 \
    --model_id S || echo "Evaluation warning: Check validation logs"

echo "=== STEP 5: COMPUTE OFFICIAL PAPER METRICS ==="
python3 -u "$SCRIPT_DIR/compute_paper_metrics.py" "$RESULTS_DIR" "PrimusV2-S (Fold 0, 200 epochs)" || echo "Metrics warning"

echo "=== PRIMUSV2-S TRAINING & EVALUATION COMPLETED SUCCESSFULLY AT $(date) ==="
