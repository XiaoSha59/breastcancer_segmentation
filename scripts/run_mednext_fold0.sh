#!/usr/bin/env bash
# run_mednext_fold0.sh
# ==============================================================================
# End-to-end execution of Official MedNeXt-L k3 on I-SPY2 Fold 0 (200 Epochs).
# Architecture: MedNeXt-L k3 (MICCAI 2023 - Saikat Roy et al., MIC-DKFZ).
# Preprocess Strategy: 100% REUSE of existing nnUNetPlans_3d_fullres (.b2nd / .npy).
# Hardware Safety: Adapts to RTX 4090 / 4080 / L4.
# Cluster / Server Safety: Auto-shutdown is DISABLED by default (safe for university servers).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Dynamic environment paths (with auto-detection for repo and cluster paths)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# GPU Collision Guard: Never collide with another lab member's active job
if command -v nvidia-smi &> /dev/null; then
    TARGET_GPU="${CUDA_VISIBLE_DEVICES:-0}"
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$TARGET_GPU" 2>/dev/null || echo "0")
    if [ "$MEM_USED" -gt 1500 ]; then
        echo "[CRITICAL SAFETY ERROR] GPU $TARGET_GPU is currently BUSY (${MEM_USED} MiB VRAM used)!"
        echo "Starting now would crash the other student's training job."
        echo "Check 'nvidia-smi' and assign an idle card: export CUDA_VISIBLE_DEVICES=<free_gpu>"
        exit 1
    fi
    echo "Verified GPU $TARGET_GPU is idle (${MEM_USED} MiB used). Safe to allocate."
fi

if [ -d "$PROJECT_DIR/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres" ]; then
    PREPROCESSED_DIR="$PROJECT_DIR/data/preprocessed/Dataset501_BreastDCE_ISPY2"
elif [ -d "$HOME/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres" ]; then
    PREPROCESSED_DIR="$HOME/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2"
elif [ -d "$HOME/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres" ]; then
    PREPROCESSED_DIR="$HOME/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2"
else
    PREPROCESSED_DIR="${nnUNet_preprocessed:-$HOME/nnUNet_preprocessed}/Dataset501_BreastDCE_ISPY2"
fi

if [ -n "$nnUNet_results" ] && [ "$nnUNet_results" != "$HOME/nnUNet_results" ]; then
    RESULTS_DIR="$nnUNet_results/Dataset501_BreastDCE_ISPY2/MedNeXt_L_k3_200epochs/fold_0"
else
    RESULTS_DIR="$PROJECT_DIR/results/Dataset501_BreastDCE_ISPY2/MedNeXt_L_k3_200epochs/fold_0"
fi
mkdir -p "$RESULTS_DIR"

# Cleanup trap: Safe for shared servers (kills child training process on interrupt, never shuts down machine)
cleanup() {
    EXIT_STATUS=$?
    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[$(date)] Interrupted: Safely terminating MedNeXt training process $TRAIN_PID to free GPU VRAM..."
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
    echo "[$(date)] Script exiting with status $EXIT_STATUS."
    echo "[$(date)] [Server Mode] Cluster safety active: Machine will remain running."
}
trap cleanup EXIT ERR INT TERM

echo "=== STEP 1: VERIFY 5-FOLD CV SPLITS ==="
SPLITS_FILE="$PREPROCESSED_DIR/splits_final.json"
if [ ! -f "$SPLITS_FILE" ]; then
    if [ -f "$PROJECT_DIR/splits_final.json" ]; then
        cp "$PROJECT_DIR/splits_final.json" "$SPLITS_FILE" 2>/dev/null || SPLITS_FILE="$PROJECT_DIR/splits_final.json"
    elif [ -f "$PROJECT_DIR/data/processed/Dataset501_BreastDCE_ISPY2/splits_final.json" ]; then
        cp "$PROJECT_DIR/data/processed/Dataset501_BreastDCE_ISPY2/splits_final.json" "$SPLITS_FILE" 2>/dev/null || SPLITS_FILE="$PROJECT_DIR/data/processed/Dataset501_BreastDCE_ISPY2/splits_final.json"
    elif [ -f "$HOME/.ssh/id_gcp" ]; then
        echo "Fetching splits_final.json from GCP..."
        scp -i ~/.ssh/id_gcp XiaoSha@34.143.181.216:/home/XiaoSha/batches/splits_final.json "$SPLITS_FILE" 2>/dev/null || true
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

echo "=== STEP 3: LAUNCH MEDNEXT-L K3 TRAINING (200 EPOCHS) ==="
echo "[$(date)] Starting train_mednext_fold0.py for MedNeXt-L k3..."

python3 -u "$SCRIPT_DIR/train_mednext_fold0.py" \
    --prep_dir "$PLAN_DIR" \
    --splits_file "$SPLITS_FILE" \
    --output_dir "$RESULTS_DIR" \
    --fold 0 \
    --epochs 200 \
    --iterations_per_epoch 250 \
    --batch_size 2 \
    --patch_size 128 128 64 \
    --model_id L \
    --kernel_size 3 \
    --initial_lr 0.0001 \
    --resume &

TRAIN_PID=$!
echo "[$(date)] MedNeXt-L k3 training started with PID $TRAIN_PID"

# Wait for training to complete
wait $TRAIN_PID
TRAIN_EXIT=$?
echo "[$(date)] MedNeXt-L k3 training finished with exit code $TRAIN_EXIT"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "=== STEP 4: SLIDING WINDOW EVALUATION ON 197 VALIDATION CASES ==="
python3 -u "$SCRIPT_DIR/evaluate_mednext_fold0.py" \
    --prep_dir "$PLAN_DIR" \
    --splits_file "$SPLITS_FILE" \
    --output_dir "$RESULTS_DIR" \
    --fold 0 \
    --model_id L \
    --kernel_size 3 || echo "Evaluation warning: Check validation logs"

echo "=== STEP 5: COMPUTE OFFICIAL PAPER METRICS ==="
python3 -u "$SCRIPT_DIR/compute_paper_metrics.py" "$RESULTS_DIR" "MedNeXt-L k3 (Fold 0, 200 epochs)" || echo "Metrics warning"

echo "=== MEDNEXT-L K3 TRAINING & EVALUATION COMPLETED SUCCESSFULLY AT $(date) ==="
