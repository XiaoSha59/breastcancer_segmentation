#!/usr/bin/env bash
# run_segmamba_3phases_gcp.sh
# ==============================================================================
# End-to-end execution of 3D SegMamba on 3 Natural DCE-MRI Phases (Fold 0, 200 Epochs) on GCP.
# Architecture: 3D SegMamba (MICCAI 2024 SOTA Pure 3D State Space Model - ge-xing/SegMamba).
# Hardware: Tailored for GCP Instance (g2-standard-8, NVIDIA L4 24GB VRAM).
# Preprocess: 100% REUSE of existing nnUNetPlans_3d_fullres (.b2nd / .npy) on GCP disk.
# Auto-shutdown: STRICTLY ENABLED (VM automatically powers off after 200 ep + 197 val evaluation).
# ==============================================================================

set -e

export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:/home/XiaoSha/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export AUTO_SHUTDOWN=1
export GCP_CLOUD_INSTANCE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2"
SPLITS_FILE="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"
PLAN_DIR="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"
RESULTS_DIR="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/SegMamba_3phases_200epochs/fold_0"

mkdir -p "$RESULTS_DIR"

# Auto-shutdown fail-safe on script exit (whether SUCCESS or ERROR to prevent billing leaks)
cleanup() {
    EXIT_STATUS=$?
    echo "[$(date)] Script exiting with status $EXIT_STATUS."
    if [ "${AUTO_SHUTDOWN:-0}" = "1" ] || [ "${GCP_CLOUD_INSTANCE:-0}" = "1" ]; then
        echo "[$(date)] [Cloud Trap] Auto-shutdown active. Waiting 120s before powering off GCP VM to prevent unwanted billing..."
        sleep 120
        sudo /sbin/shutdown -h now || sudo poweroff || true
    else
        echo "[$(date)] [Server Mode] System remains running."
    fi
}
trap cleanup EXIT ERR INT TERM

echo "=============================================================================="
echo "  3D SEGMAMBA (3 NATURAL PHASES) PIPELINE (GCP CLOUD MODE)"
echo "  Target: 200 Epochs x 250 Iterations | Hardware: NVIDIA L4 24GB"
echo "=============================================================================="

# STEP 1: VERIFY HARDWARE & CUDA MAMBA ACCELERATION
echo "=== STEP 1: VERIFY HARDWARE & CUDA MAMBA KERNEL ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi

python3 -c "import mamba_ssm; print('mamba_ssm loaded successfully:', mamba_ssm.__file__)" 2>/dev/null || {
    echo "Installing packaging, ninja, causal-conv1d and mamba-ssm with CUDA support..."
    pip install -q packaging ninja einops
    pip install -q causal-conv1d>=1.4.0 --no-build-isolation
    pip install -q mamba-ssm --no-build-isolation
    python3 -c "import mamba_ssm; print('Verified mamba_ssm installed!')"
}

# STEP 2: VERIFY PREPROCESSED DATA & SPLITS ON GCP
echo "=== STEP 2: VERIFY DATA & SPLITS ==="
if [ ! -f "$SPLITS_FILE" ]; then
    if [ -f "$DATA_DIR/splits_final.json" ]; then
        echo "Copying splits_final.json to $SPLITS_FILE..."
        cp "$DATA_DIR/splits_final.json" "$SPLITS_FILE"
    else
        echo "Error: splits_final.json missing on GCP!"
        exit 1
    fi
fi
echo "Verified splits_final.json exists (785 train / 197 val)."

if [ ! -d "$PLAN_DIR" ]; then
    echo "Error: Preprocessed directory $PLAN_DIR does not exist on GCP!"
    exit 1
fi
echo "Verified preprocessed directory $PLAN_DIR."

# STEP 3: LAUNCH 3D SEGMAMBA (3 PHASES) TRAINING (200 EPOCHS)
echo "=== STEP 3: LAUNCH 3D SEGMAMBA (3 PHASES) TRAINING (200 EPOCHS) ==="
LOG_FILE="$RESULTS_DIR/train.log"
echo "Training logs will stream to console and append to $LOG_FILE"

NUM_EPOCHS="${1:-200}"
VAL_INT="${2:-5}"
if [ "$NUM_EPOCHS" -le 5 ]; then
    VAL_INT=1
fi

EXTRA_FLAGS=""
if [ -f "$RESULTS_DIR/checkpoint_latest.pth" ]; then
    echo "Found existing checkpoint_latest.pth, enabling --resume flag..."
    EXTRA_FLAGS="--resume"
fi

python3 -u "$SCRIPT_DIR/train_segmamba_3phases_fold0.py" \
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
    $EXTRA_FLAGS

echo "=== STEP 4: VERIFY CHECKPOINT ==="
CHECKPOINT_BEST="$RESULTS_DIR/checkpoint_best.pth"
if [ ! -f "$CHECKPOINT_BEST" ]; then
    CHECKPOINT_BEST="$RESULTS_DIR/checkpoint_latest.pth"
fi
if [ ! -f "$CHECKPOINT_BEST" ]; then
    echo "Error: No checkpoint generated in $RESULTS_DIR!"
    exit 1
fi
echo "Verified checkpoint: $CHECKPOINT_BEST ($(du -h "$CHECKPOINT_BEST" | cut -f1))."

# STEP 5: SLIDING WINDOW INFERENCE & OFFICIAL METRICS (3 PHASES)
echo "=== STEP 5: SLIDING WINDOW EVALUATION ON 197 VALIDATION CASES (3 PHASES) ==="
if [ "$NUM_EPOCHS" -ge 200 ]; then
    python3 -u "$SCRIPT_DIR/evaluate_segmamba_3phases_fold0.py" \
        --prep_dir "$PLAN_DIR" \
        --splits_file "$SPLITS_FILE" \
        --output_dir "$RESULTS_DIR" \
        --fold 0 \
        --patch_size 64 128 128
    
    echo "Flagging completion..."
    touch "$RESULTS_DIR/results_ready.flag"
else
    echo "Notice: Running short smoke test ($NUM_EPOCHS epochs < 200). Skipping full validation."
fi

echo "=============================================================================="
echo "  3D SEGMAMBA (3 PHASES) PIPELINE COMPLETED SUCCESSFULLY ON GCP!"
echo "  Results ready in: $RESULTS_DIR"
echo "  Shutting down VM in 120 seconds..."
echo "=============================================================================="
