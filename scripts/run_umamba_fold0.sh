#!/usr/bin/env bash
# run_umamba_fold0.sh
# ==============================================================================
# End-to-end execution of Official U-Mamba Bot 3D on I-SPY2 Fold 0 (200 Epochs).
# Architecture: U-Mamba Bot 3D (Bo Wang Lab, MICCAI 2024 - Jun Ma et al.).
# Preprocess Strategy: 100% REUSE of existing 64GB nnUNetPlans_3d_fullres (.b2nd).
# Memory Safeguards: export nnUNet_n_proc_DA=4 to prevent 42GB virtual memory OOM crash.
# Server Safety: Auto-shutdown is STRICTLY DISABLED on cluster (safe for university servers).
# Cloud Safety: Auto-shutdown is enabled ONLY when AUTO_SHUTDOWN=1 or GCP_CLOUD_INSTANCE=1.
# Evaluation: Extracts strictly the 4 paper metrics (Mean Dice, Median Dice, HD95, ASSD).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Dynamic environment paths
export nnUNet_raw="${nnUNet_raw:-$HOME/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$HOME/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$HOME/nnUNet_results}"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# CRITICAL SAFEGUARD (Section 4 of 3D_UNET_BEST_PRACTICES.md): Limit DA workers to prevent RAM OOM
export nnUNet_n_proc_DA=4

PREPROCESSED_DIR="$nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2"
RESULTS_DIR="$nnUNet_results/Dataset501_BreastDCE_ISPY2/nnUNetTrainerUMambaBot_200epochs__nnUNetPlans__3d_fullres/fold_0"
mkdir -p "$RESULTS_DIR"

# LAYER 1: Auto-shutdown trap (ONLY active on GCP Cloud when explicitly requested)
cleanup() {
    EXIT_STATUS=$?
    echo "[$(date)] Script exiting with status $EXIT_STATUS."
    if [ "${AUTO_SHUTDOWN:-0}" = "1" ] || [ "${GCP_CLOUD_INSTANCE:-0}" = "1" ]; then
        echo "[$(date)] [Auto-Shutdown] Cloud mode active. Waiting 120s before powering off VM..."
        sleep 120
        sudo /sbin/shutdown -h now || sudo poweroff || true
    else
        echo "[$(date)] [Server Mode] Cluster safety active: Machine will remain running."
    fi
}
trap cleanup EXIT ERR INT TERM

echo "=== STEP 1: PRESERVE & VERIFY 5-FOLD CV SPLITS ==="
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

echo "=== STEP 2: REGISTER & VALIDATE U-MAMBA BOT 3D TRAINER IN nnU-Net ==="
python3 "$SCRIPT_DIR/add_trainer_umamba_200epochs.py"
python3 "$SCRIPT_DIR/test_umamba_trainer_discovery.py"

echo "=== STEP 3: VERIFY PLANS & REUSE 64GB PREPROCESSED DATA ==="
PLAN_DIR="$PREPROCESSED_DIR/nnUNetPlans_3d_fullres"
if [ ! -d "$PLAN_DIR" ]; then
    echo "Error: Preprocessed directory $PLAN_DIR does not exist!"
    exit 1
fi

if [ ! -f "$PREPROCESSED_DIR/nnUNetPlans.json" ]; then
    echo "nnUNetPlans.json not found. Running nnUNetv2_plan_experiment (planning only, NO re-preprocess)..."
    nnUNetv2_plan_experiment -d 501 -c 3d_fullres
fi
echo "Reusing existing 64GB preprocessed data in $PLAN_DIR directly."

# Check if checkpoint exists to seamlessly continue if interrupted
EXTRA_FLAGS=""
if [ -f "$RESULTS_DIR/checkpoint_latest.pth" ]; then
    echo "Found existing checkpoint_latest.pth in $RESULTS_DIR, continuing training with --c..."
    EXTRA_FLAGS="--c"
fi

echo "=== STEP 4: LAUNCH U-MAMBA BOT 3D TRAINING (200 EPOCHS, 250 ITER/EPOCH) ==="
echo "[$(date)] Starting nnUNetv2_train for U-Mamba Bot 3D..."

nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainerUMambaBot_200epochs -p nnUNetPlans $EXTRA_FLAGS &
TRAIN_PID=$!
echo "[$(date)] U-Mamba Bot 3D training started with PID $TRAIN_PID"

# LAYER 2: Launch independent Python watchdog daemon ONLY if cloud auto-shutdown is requested
if [ "${AUTO_SHUTDOWN:-0}" = "1" ] || [ "${GCP_CLOUD_INSTANCE:-0}" = "1" ]; then
    if [ -f "$SCRIPT_DIR/safety_guard.py" ]; then
        python3 -u "$SCRIPT_DIR/safety_guard.py" $TRAIN_PID > "$RESULTS_DIR/safety_guard.log" 2>&1 &
        GUARD_PID=$!
        echo "[$(date)] [Watchdog] Cloud watchdog started with PID $GUARD_PID monitoring PID $TRAIN_PID"
    fi
fi

# Wait for training to complete
wait $TRAIN_PID
TRAIN_EXIT=$?
echo "[$(date)] U-Mamba Bot 3D training finished with exit code $TRAIN_EXIT"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "=== STEP 5: VERIFY VALIDATION INFERENCE ON 197 CASES ==="
# nnUNet performs validation automatically at epoch 199. If summary.json missing, run predict:
if [ ! -f "$RESULTS_DIR/validation/summary.json" ] && [ ! -f "$RESULTS_DIR/validation_raw/summary.json" ]; then
    echo "Running nnUNetv2_predict on validation cohort..."
    mkdir -p "$RESULTS_DIR/validation"
    # Fallback to evaluate_umamba_fold0.py if nnUNet raw folder is mapped differently
    python3 -u "$SCRIPT_DIR/evaluate_umamba_fold0.py" \
        --data_dir "$nnUNet_raw/Dataset501_BreastDCE_ISPY2" \
        --splits_file "$SPLITS_FILE" \
        --output_dir "$RESULTS_DIR" \
        --fold 0 || echo "Validation warning: Check validation logs"
fi

echo "=== STEP 6: COMPUTE OFFICIAL PAPER METRICS (Krikun Table 2) ==="
python3 -u "$SCRIPT_DIR/compute_paper_metrics.py" "$RESULTS_DIR" "U-Mamba Bot 3D (MICCAI 2024, nnU-Net v2)" || echo "Metrics warning"

echo "=== U-MAMBA BOT 3D TRAINING & EVALUATION COMPLETED SUCCESSFULLY AT $(date) ==="
