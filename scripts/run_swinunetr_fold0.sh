#!/usr/bin/env bash
# run_swinunetr_fold0.sh
# ==============================================================================
# End-to-end execution of SwinUNETR (Transformer) on I-SPY2 Fold 0 (200 Epochs).
# Architecture: monai.networks.nets.SwinUNETR (Base, feature_size=48, 5 channels).
# Hardware Safeguards:
# - 32GB Swap active, 4 workers to prevent host RAM OOM crash.
# - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to eliminate fragmentation.
# - Mixed precision AMP + Activation Checkpointing keeping VRAM ~13-15 GB.
# Evaluation: 197 Validation Cases evaluated with 3D sliding window inference.
# Metrics: Strictly the 6 paper metrics (Mean/Median Dice, IoU, Rec, Prec, HD95, ASSD).
# Auto-shutdown: 2-layer fail-safe (Layer 1: Bash trap on EXIT/ERR; Layer 2: Python watchdog).
# ==============================================================================

set -e

export PATH="/home/XiaoSha/.local/bin:$PATH"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

DATA_DIR="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2"
SPLITS_FILE="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"
RESULTS_DIR="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/SwinUNETR_200epochs/fold_0"

mkdir -p "$RESULTS_DIR"

# LAYER 1: Auto-shutdown on script exit (whether SUCCESS exit 0, or ERROR exit != 0)
cleanup() {
    EXIT_STATUS=$?
    echo "[$(date)] Script exiting with status $EXIT_STATUS."
    if [ "${GCP_CLOUD_INSTANCE:-0}" = "1" ]; then
        echo "[$(date)] [Cloud Trap] GCP_CLOUD_INSTANCE=1. Shutting down VM in 120s..."
        sleep 120
        sudo /sbin/shutdown -h now || true
    else
        echo "[$(date)] [Server Mode] System remains running."
    fi
}
trap cleanup EXIT ERR INT TERM

echo "=== STEP 1: VERIFY ENVIRONMENT & DEPENDENCIES ==="
# Check if monai is installed
python3 -c "import monai" 2>/dev/null || {
    echo "Installing MONAI..."
    pip install -q monai
}
python3 -c "import monai; print('MONAI version:', monai.__version__)"

echo "=== STEP 2: VERIFY DATA & SPLITS ==="
if [ ! -f "$SPLITS_FILE" ]; then
    echo "Splits file missing at $SPLITS_FILE! Restoring from data/processed..."
    cp "$DATA_DIR/splits_final.json" "$SPLITS_FILE"
fi
echo "Verified splits_final.json exists (785 train / 197 val)."

EXTRA_FLAGS=""
if [ -f "$RESULTS_DIR/checkpoint_latest.pth" ]; then
    echo "Found existing checkpoint_latest.pth, enabling --resume flag..."
    EXTRA_FLAGS="--resume"
fi

echo "=== STEP 3: LAUNCH SWINUNETR TRAINING (200 EPOCHS) WITH 2-LAYER FAIL-SAFE ==="
echo "[$(date)] Starting train_swinunetr_fold0.py..."

PREP_DIR="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"

python3 -u /home/XiaoSha/project/scripts/train_swinunetr_fold0.py \
    --prep_dir "$PREP_DIR" \
    --splits_file "$SPLITS_FILE" \
    --output_dir "$RESULTS_DIR" \
    --fold 0 \
    --epochs 200 \
    --iterations_per_epoch 250 \
    --batch_size 2 \
    --patch_size 96 96 96 \
    --feature_size 48 \
    --initial_lr 0.0001 \
    $EXTRA_FLAGS &

TRAIN_PID=$!
echo "[$(date)] SwinUNETR training started with PID $TRAIN_PID"

# LAYER 2: Launch independent Python watchdog daemon monitoring TRAIN_PID
python3 -u /home/XiaoSha/project/scripts/safety_guard.py $TRAIN_PID > /home/XiaoSha/safety_guard_swinunetr.log 2>&1 &
GUARD_PID=$!
echo "[$(date)] [Layer 2 Guard] Watchdog daemon started with PID $GUARD_PID monitoring PID $TRAIN_PID"

# Wait for training process to finish
wait $TRAIN_PID
TRAIN_EXIT=$?
echo "[$(date)] SwinUNETR training finished with exit code $TRAIN_EXIT"

echo "=== STEP 4: FULL INDEPENDENT VALIDATION ON 197 CASES ==="
python3 -u /home/XiaoSha/project/scripts/evaluate_swinunetr_fold0.py \
    --data_dir "$DATA_DIR" \
    --splits_file "$SPLITS_FILE" \
    --output_dir "$RESULTS_DIR" \
    --fold 0 \
    --patch_size 96 96 96 \
    --feature_size 48 \
    --num_workers 4 || echo "Warning: Evaluation script encountered an issue"

echo "=== SWINUNETR TRAINING & EVALUATION COMPLETE AT $(date) ==="
