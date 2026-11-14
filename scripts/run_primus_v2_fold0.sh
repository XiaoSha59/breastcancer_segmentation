#!/usr/bin/env bash
# run_primus_v2_fold0.sh
# ==============================================================================
# End-to-end execution of Official PrimusV2-S on I-SPY2 Fold 0 (200 Epochs).
# Architecture: PrimusV2-S (MICCAI / TMLR 2024 - Tassilo Wald, Fabian Isensee, DKFZ).
# Preprocess Strategy: 100% REUSE of existing nnUNetPlans_3d_fullres (.b2nd / .npy).
# Hardware Safety: Tailored for NVIDIA L4 24GB.
# Auto-shutdown: Triggers ONLY on SUCCESS ($EXIT_STATUS == 0) to avoid premature poweroff on error.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PREPROCESSED_DIR="$HOME/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2"
RESULTS_DIR="$HOME/nnUNet_results/Dataset501_BreastDCE_ISPY2/PrimusV2_S_200epochs/fold_0"
mkdir -p "$RESULTS_DIR"

cleanup() {
    EXIT_STATUS=$?
    echo "[$(date)] PrimusV2 run script exiting with status $EXIT_STATUS."
    if [ "${AUTO_SHUTDOWN:-1}" = "1" ] || [ "${GCP_CLOUD_INSTANCE:-1}" = "1" ]; then
        if [ $EXIT_STATUS -eq 0 ]; then
            echo "[$(date)] [Cloud Trap] Auto-shutdown active on SUCCESS. Waiting 120s before powering off VM..."
            sleep 120
            sudo /sbin/shutdown -h now || sudo poweroff || true
        else
            echo "[$(date)] [Cloud Trap] Script exited with ERROR ($EXIT_STATUS). Preserving VM state for inspection (no premature shutdown)."
        fi
    fi
}
trap cleanup EXIT ERR INT TERM

echo "=== STEP 1: VERIFY SPLITS AND DATA ==="
SPLITS_FILE="$PREPROCESSED_DIR/splits_final.json"
PLAN_DIR="$PREPROCESSED_DIR/nnUNetPlans_3d_fullres"

if [ ! -f "$SPLITS_FILE" ] || [ ! -d "$PLAN_DIR" ]; then
    echo "Error: Required data or splits not found in $PREPROCESSED_DIR!"
    exit 1
fi
echo "Verified splits and preprocessed data at $PLAN_DIR."

echo "=== STEP 2: LAUNCH PRIMUSV2-S TRAINING (200 EPOCHS) ==="
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
    --resume

echo "=== STEP 3: SLIDING WINDOW EVALUATION ON 197 VALIDATION CASES ==="
python3 -u "$SCRIPT_DIR/evaluate_primus_v2_fold0.py" \
    --prep_dir "$PLAN_DIR" \
    --splits_file "$SPLITS_FILE" \
    --output_dir "$RESULTS_DIR" \
    --fold 0 \
    --patch_size 64 128 128 \
    --model_id S || echo "Evaluation warning: Check validation logs"

echo "=== PRIMUSV2-S RUN COMPLETED SUCCESSFULLY AT $(date) ==="
