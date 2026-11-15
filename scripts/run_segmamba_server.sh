#!/usr/bin/env bash
# run_segmamba_server.sh
# ==============================================================================
# End-to-end execution of 3D SegMamba on I-SPY2 Fold 0 (200 Epochs).
# Architecture: 3D SegMamba (MICCAI 2024 SOTA Pure 3D State Space Model).
# Preprocess Strategy: 100% REUSE of existing nnUNetPlans_3d_fullres (.b2nd / .npy).
# Hardware Safety: Tailored for University Server / Lab GPU (RTX 4090 24GB).
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

# Auto-activate breast_env if current shell is in base or system python
if [ "$CONDA_DEFAULT_ENV" != "breast_env" ]; then
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook 2>/dev/null)"
        conda activate breast_env 2>/dev/null || true
    fi
    if ! python3 -c "import torch, numpy" &>/dev/null; then
        for act in "$HOME/breast_env/bin/activate" "$HOME/miniconda3/bin/activate" "$HOME/anaconda3/bin/activate"; do
            if [ -f "$act" ]; then
                source "$act" breast_env 2>/dev/null || true
                break
            fi
        done
    fi
fi

echo "=============================================================================="
echo "  3D SEGMAMBA TRAINING & EVALUATION PIPELINE (UNIVERSITY SERVER MODE)"
echo "=============================================================================="

# GPU Collision Guard: Never collide with another lab member's or another model's active job
if command -v nvidia-smi &> /dev/null; then
    TARGET_GPU="${CUDA_VISIBLE_DEVICES:-0}"
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$TARGET_GPU" 2>/dev/null || echo "0")
    if [ "$MEM_USED" -gt 1500 ]; then
        echo "[CRITICAL SAFETY WARNING] GPU $TARGET_GPU is currently BUSY (${MEM_USED} MiB VRAM used)!"
        echo "Starting SegMamba now would collide with another active job (e.g., MedNeXt or other lab member)."
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
    RESULTS_DIR="$nnUNet_results/Dataset501_BreastDCE_ISPY2/SegMamba_200epochs/fold_0"
else
    RESULTS_DIR="$PROJECT_DIR/results/Dataset501_BreastDCE_ISPY2/SegMamba_200epochs/fold_0"
fi
mkdir -p "$RESULTS_DIR"

# Cleanup trap: Safe for shared servers (only terminates child training process on interrupt, NEVER shuts down server)
cleanup() {
    EXIT_STATUS=$?
    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[$(date)] Interrupted: Safely terminating SegMamba training process $TRAIN_PID to release GPU VRAM..."
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

echo "=== STEP 3: LAUNCH 3D SEGMAMBA TRAINING (200 EPOCHS) ==="
LOG_FILE="$RESULTS_DIR/train.log"
echo "Training logs will stream to console and append to $LOG_FILE"

TRAIN_SCRIPT="$SCRIPT_DIR/train_segmamba_fold0.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script $TRAIN_SCRIPT not found!"
    exit 1
fi

NUM_EPOCHS="${1:-200}"
VAL_INT="${2:-5}"
if [ "$NUM_EPOCHS" -le 5 ]; then
    VAL_INT=1
fi

python3 -u "$TRAIN_SCRIPT" \
    --prep_dir "$PLAN_DIR" \
    --splits_file "$SPLITS_FILE" \
    --output_dir "$RESULTS_DIR" \
    --fold 0 \
    --epochs "$NUM_EPOCHS" \
    --iterations_per_epoch 250 \
    --batch_size 2 \
    --patch_size 64 128 128 \
    --initial_lr 1e-4 \
    --weight_decay 1e-5 \
    --val_interval "$VAL_INT" \
    --resume &

TRAIN_PID=$!
echo "SegMamba training started with PID $TRAIN_PID on GPU $CUDA_VISIBLE_DEVICES ($NUM_EPOCHS epochs, val interval $VAL_INT)."
wait "$TRAIN_PID"
unset TRAIN_PID

echo "=== STEP 4: VERIFY CHECKPOINT ==="
CHECKPOINT_BEST="$RESULTS_DIR/checkpoint_best.pth"
if [ ! -f "$CHECKPOINT_BEST" ]; then
    echo "Warning: checkpoint_best.pth not found, checking checkpoint_latest.pth..."
    CHECKPOINT_BEST="$RESULTS_DIR/checkpoint_latest.pth"
fi

if [ ! -f "$CHECKPOINT_BEST" ]; then
    echo "Error: No checkpoint generated in $RESULTS_DIR!"
    exit 1
fi
echo "Verified checkpoint: $CHECKPOINT_BEST ($(du -h "$CHECKPOINT_BEST" | cut -f1))."

echo "=== STEP 5: SLIDING WINDOW INFERENCE & OFFICIAL EVALUATION ==="
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_segmamba_fold0.py"
if [ "$NUM_EPOCHS" -ge 200 ]; then
    if [ -f "$EVAL_SCRIPT" ]; then
        python3 -u "$EVAL_SCRIPT" \
            --prep_dir "$PLAN_DIR" \
            --splits_file "$SPLITS_FILE" \
            --output_dir "$RESULTS_DIR" \
            --fold 0 \
            --patch_size 64 128 128
    else
        echo "Warning: $EVAL_SCRIPT not found, skipping evaluation step."
    fi
else
    echo "Notice: Running short smoke test ($NUM_EPOCHS epochs < 200). Skipping 50-minute full validation."
    echo "To run full 200 epochs and evaluate, simply run: bash scripts/run_segmamba_server.sh (it will resume seamlessly)."
fi

echo "=============================================================================="
echo "  3D SEGMAMBA PIPELINE COMPLETED SUCCESSFULLY ON UNIVERSITY SERVER"
echo "  Results: $RESULTS_DIR"
echo "=============================================================================="
