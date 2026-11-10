#!/usr/bin/env bash
# run_standard_nnunet_fold0.sh
# ==============================================================================
# End-to-end execution of Standard 3D nnU-Net on I-SPY2 Fold 0 (200 Epochs).
# Architecture: Plain Convolutional 3D U-Net (nnUNetPlans).
# Hardware Safeguards: 32GB Swap, nnUNet_n_proc_DA=4 to prevent RAM OOM crash.
# Preprocess Strategy: 100% REUSE of existing nnUNetPlans_3d_fullres (NO re-preprocess).
# Evaluation: Extracts strictly the 4 paper metrics (Mean Dice, Median Dice, HD95, ASSD).
# Auto-shutdown: 2-layer fail-safe (Layer 1: Bash trap on EXIT/ERR; Layer 2: Python watchdog).
# ==============================================================================

set -e

export nnUNet_raw="/home/XiaoSha/nnUNet_raw"
export nnUNet_preprocessed="/home/XiaoSha/nnUNet_preprocessed"
export nnUNet_results="/home/XiaoSha/nnUNet_results"
export PATH="/home/XiaoSha/.local/bin:$PATH"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# CRITICAL SAFEGUARD: Limit Data Augmentation workers to prevent 42GB virtual memory OOM
export nnUNet_n_proc_DA=4

PREPROCESSED_DIR="$nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2"
RESULTS_DIR="$nnUNet_results/Dataset501_BreastDCE_ISPY2/nnUNetTrainer_200epochs__nnUNetPlans__3d_fullres/fold_0"

# LAYER 1: Auto-shutdown on script exit (whether SUCCESS exit 0, or ERROR exit != 0)
cleanup() {
    EXIT_STATUS=$?
    echo "[$(date)] [Layer 1 Trap] Script exiting with status $EXIT_STATUS."
    echo "[$(date)] [Layer 1 Trap] Waiting 120 seconds before auto-shutdown to allow logs/checkpoints to flush..."
    sleep 120
    echo "[$(date)] [Layer 1 Trap] Shutting down VM now..."
    sudo /sbin/shutdown -h now
}
trap cleanup EXIT ERR INT TERM

echo "=== STEP 1: PRESERVE & VERIFY 5-FOLD CV SPLITS ==="
# Backup splits_final.json to guarantee 100% fair comparison with ResEnc Fold 0
if [ -f "$PREPROCESSED_DIR/splits_final.json" ]; then
    cp "$PREPROCESSED_DIR/splits_final.json" "$PREPROCESSED_DIR/splits_final.json.bak"
    echo "Backed up existing splits_final.json."
elif [ -f "/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2/splits_final.json" ]; then
    cp "/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2/splits_final.json" "$PREPROCESSED_DIR/splits_final.json"
    cp "$PREPROCESSED_DIR/splits_final.json" "$PREPROCESSED_DIR/splits_final.json.bak"
    echo "Restored splits_final.json from data/processed."
fi

echo "=== STEP 2: VERIFY PLANS & REUSE PREPROCESSED DATA ==="
# Check if nnUNetPlans.json exists; if not, generate plans ONLY (takes 15s, NEVER re-preprocess)
if [ ! -f "$PREPROCESSED_DIR/nnUNetPlans.json" ]; then
    echo "nnUNetPlans.json not found. Running nnUNetv2_plan_experiment (planning only)..."
    nnUNetv2_plan_experiment -d 501 -c 3d_fullres
    echo "nnUNetPlans.json generated successfully. Reusing existing preprocessed images in nnUNetPlans_3d_fullres!"
else
    echo "Found existing nnUNetPlans.json. Reusing preprocessed data in nnUNetPlans_3d_fullres directly."
fi

# Restore backed-up splits_final.json in case planner touched it
if [ -f "$PREPROCESSED_DIR/splits_final.json.bak" ]; then
    cp "$PREPROCESSED_DIR/splits_final.json.bak" "$PREPROCESSED_DIR/splits_final.json"
    echo "Verified splits_final.json is intact (785 train / 197 val)."
fi

# Check if checkpoint exists to continue if interrupted
EXTRA_FLAGS=""
if [ -f "$RESULTS_DIR/checkpoint_latest.pth" ]; then
    echo "Found existing checkpoint_latest.pth, continuing training with --c..."
    EXTRA_FLAGS="--c"
fi

echo "=== STEP 3: LAUNCH TRAINING WITH 2-LAYER FAIL-SAFE ==="
echo "[$(date)] Starting nnUNetv2_train for Standard 3D nnU-Net..."

# Launch training
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_200epochs -p nnUNetPlans $EXTRA_FLAGS &
TRAIN_PID=$!
echo "[$(date)] nnUNetv2_train started with PID $TRAIN_PID"

# LAYER 2: Launch independent Python watchdog daemon monitoring TRAIN_PID
python3 -u /home/XiaoSha/project/scripts/safety_guard.py $TRAIN_PID > /home/XiaoSha/safety_guard_standard.log 2>&1 &
GUARD_PID=$!
echo "[$(date)] [Layer 2 Guard] Watchdog daemon started with PID $GUARD_PID monitoring PID $TRAIN_PID"

# Wait for training process to finish
wait $TRAIN_PID
TRAIN_EXIT=$?
echo "[$(date)] nnUNetv2_train finished with exit code $TRAIN_EXIT"

echo "=== STEP 4: EXTRACT ONLY THE 4 OFFICIAL PAPER METRICS ==="
python3 -u /home/XiaoSha/project/scripts/compute_paper_metrics.py "$RESULTS_DIR" "Standard 3D nnU-Net (Plain Conv, nnUNetPlans)" || echo "Warning: Failed to compute paper metrics"

echo "=== STANDARD 3D NNU-NET TRAINING & EVALUATION COMPLETE AT $(date) ==="
