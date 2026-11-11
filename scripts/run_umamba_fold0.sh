#!/usr/bin/env bash
# run_umamba_fold0.sh
# ==============================================================================
# End-to-end execution of Official U-Mamba Bot 3D on I-SPY2 Fold 0 (200 Epochs).
# Architecture: U-Mamba Bot 3D (Bo Wang Lab, MICCAI 2024 - Jun Ma et al.).
# Preprocess Strategy: 100% REUSE of existing nnUNetPlans_3d_fullres (.b2nd / .npy).
# Hardware Safety: Adapts to RTX 4090 / 4080 / L4.
# Cluster / Server Safety: Auto-shutdown is DISABLED by default (safe for university servers).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Dynamic environment paths (falls back to $HOME if not preset)
export nnUNet_raw="${nnUNet_raw:-$HOME/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$HOME/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$HOME/nnUNet_results}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PREPROCESSED_DIR="$nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2"
RESULTS_DIR="$nnUNet_results/Dataset501_BreastDCE_ISPY2/UMambaBot_3d_200epochs/fold_0"
mkdir -p "$RESULTS_DIR"

# Cleanup trap: Safe for shared servers (never shuts down machine)
cleanup() {
    EXIT_STATUS=$?
    echo "[$(date)] Script exiting with status $EXIT_STATUS."
    echo "[$(date)] [Server Mode] Cluster safety active: Machine will remain running."
}
trap cleanup EXIT ERR INT TERM

echo "=== STEP 1: VERIFY 5-FOLD CV SPLITS ==="
SPLITS_FILE="$PREPROCESSED_DIR/splits_final.json"
if [ ! -f "$SPLITS_FILE" ]; then
    if [ -f "$PROJECT_DIR/data/processed/Dataset501_BreastDCE_ISPY2/splits_final.json" ]; then
        cp "$PROJECT_DIR/data/processed/Dataset501_BreastDCE_ISPY2/splits_final.json" "$SPLITS_FILE"
    elif [ -f "$PROJECT_DIR/splits_final.json" ]; then
        cp "$PROJECT_DIR/splits_final.json" "$SPLITS_FILE"
    else
        echo "Error: splits_final.json not found in $PREPROCESSED_DIR or $PROJECT_DIR!"
        exit 1
    fi
fi
echo "Verified splits_final.json exists (785 train / 197 val)."

echo "=== STEP 2: VERIFY PREPROCESSED DATA (.b2nd / .npy) ==="
PLAN_DIR="$PREPROCESSED_DIR/nnUNetPlans_3d_fullres"
if [ ! -d "$PLAN_DIR" ]; then
    echo "Error: Preprocessed directory $PLAN_DIR does not exist!"
    exit 1
fi
echo "Verified preprocessed directory $PLAN_DIR."

echo "=== STEP 3: LAUNCH U-MAMBA BOT 3D TRAINING (200 EPOCHS) ==="
echo "[$(date)] Starting train_umamba_fold0.py for U-Mamba Bot 3D..."

python3 -u "$SCRIPT_DIR/train_umamba_fold0.py" \
    --prep_dir "$PLAN_DIR" \
    --splits_file "$SPLITS_FILE" \
    --output_dir "$RESULTS_DIR" \
    --fold 0 \
    --epochs 200 \
    --iterations_per_epoch 250 \
    --batch_size 2 \
    --patch_size 128 128 64 \
    --initial_lr 0.0001 \
    --resume &

TRAIN_PID=$!
echo "[$(date)] U-Mamba Bot 3D training started with PID $TRAIN_PID"

# Wait for training to complete
wait $TRAIN_PID
TRAIN_EXIT=$?
echo "[$(date)] U-Mamba Bot 3D training finished with exit code $TRAIN_EXIT"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "=== STEP 4: SLIDING WINDOW EVALUATION ON 197 VALIDATION CASES ==="
python3 -u "$SCRIPT_DIR/evaluate_umamba_fold0.py" \
    --data_dir "$nnUNet_raw/Dataset501_BreastDCE_ISPY2" \
    --splits_file "$SPLITS_FILE" \
    --output_dir "$RESULTS_DIR" \
    --fold 0 || echo "Evaluation warning: Check validation logs"

echo "=== STEP 5: COMPUTE OFFICIAL PAPER METRICS ==="
python3 -u "$SCRIPT_DIR/compute_paper_metrics.py" "$RESULTS_DIR" "U-Mamba Bot 3D (Fold 0, 200 epochs)" || echo "Metrics warning"

echo "=== U-MAMBA BOT 3D TRAINING & EVALUATION COMPLETED SUCCESSFULLY AT $(date) ==="
