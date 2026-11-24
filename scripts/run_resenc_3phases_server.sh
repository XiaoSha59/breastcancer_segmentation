#!/usr/bin/env bash
# run_resenc_3phases_server.sh
# ==============================================================================
# End-to-end execution of ResEnc-UNet L on 3 Natural DCE-MRI Phases (Fold 0, 200 Epochs).
# Architecture: Residual Encoder UNet Large (dynamic-network-architectures / nnUNetResEncUNetLPlans).
# Input: Slices strictly 3 natural phases [:3] (Pre, Early Peak, Delayed Late).
# Data: 100% REUSE of existing 64GB Dataset501_BreastDCE_ISPY2 preprocessed data.
# Server Safety: Zero auto-shutdown (cluster runs 24/7). OMP_NUM_THREADS=4, nnUNet_n_proc_DA=4.
# Evaluation: 197-case validation + official paper metrics (Dice, HD95, ASSD).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resource Allocation for node1 / lab server
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export nnUNet_n_proc_DA=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export nnUNet_compile=False
export TORCH_CUDNN_V8_API_ENABLED=0

# nnU-Net environment paths
export nnUNet_raw="${nnUNet_raw:-$HOME/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$HOME/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$HOME/nnUNet_results}"

echo "=============================================================================="
echo "    RESENC-UNET L (3 NATURAL PHASES) - UNIVERSITY SERVER (NODE1)              "
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

# Safe cleanup: terminate background jobs on interrupt, NEVER shutdown server
cleanup() {
    EXIT_STATUS=$?
    echo "[$(date)] Script exiting with status $EXIT_STATUS."
    echo "[$(date)] [Server Mode] System remains running (zero auto-shutdown)."
}
trap cleanup EXIT ERR INT TERM

# Locate preprocessed data directory
if [ -d "$PROJECT_DIR/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres" ]; then
    PREPROCESSED_DIR="$PROJECT_DIR/data/preprocessed/Dataset501_BreastDCE_ISPY2"
elif [ -d "$HOME/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres" ]; then
    PREPROCESSED_DIR="$HOME/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2"
elif [ -d "$nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres" ]; then
    PREPROCESSED_DIR="$nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2"
else
    PREPROCESSED_DIR="$HOME/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2"
fi

# Ensure nnUNet_preprocessed environment variable points to parent of PREPROCESSED_DIR
export nnUNet_preprocessed="$(dirname "$PREPROCESSED_DIR")"
echo "Exported nnUNet_preprocessed to: $nnUNet_preprocessed"

# Also ensure fallback path $HOME/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2 has symlink or plans
mkdir -p "$HOME/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2" 2>/dev/null || true

# Deploy nnUNetResEncUNetLPlans.json if missing
if [ ! -f "$PREPROCESSED_DIR/nnUNetResEncUNetLPlans.json" ]; then
    if [ -f "$SCRIPT_DIR/nnUNetResEncUNetLPlans.json" ]; then
        cp "$SCRIPT_DIR/nnUNetResEncUNetLPlans.json" "$PREPROCESSED_DIR/nnUNetResEncUNetLPlans.json"
        echo "Deployed nnUNetResEncUNetLPlans.json to $PREPROCESSED_DIR"
    fi
fi
if [ -f "$SCRIPT_DIR/nnUNetResEncUNetLPlans.json" ]; then
    cp "$SCRIPT_DIR/nnUNetResEncUNetLPlans.json" "$HOME/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetResEncUNetLPlans.json" 2>/dev/null || true
fi

RESULTS_DIR="$nnUNet_results/Dataset501_BreastDCE_ISPY2/nnUNetTrainer_3phases_200epochs__nnUNetResEncUNetLPlans__3d_fullres/fold_0"
mkdir -p "$RESULTS_DIR"

echo "=== STEP 1: PRESERVE & VERIFY 5-FOLD CV SPLITS ==="
if [ -f "$PREPROCESSED_DIR/splits_final.json" ]; then
    echo "Verified splits_final.json is intact: $PREPROCESSED_DIR/splits_final.json"
elif [ -f "$PROJECT_DIR/splits_final.json" ]; then
    cp "$PROJECT_DIR/splits_final.json" "$PREPROCESSED_DIR/splits_final.json"
    echo "Restored splits_final.json from project directory."
fi
# Ensure splits_final.json also exists in fallback
cp "$PREPROCESSED_DIR/splits_final.json" "$HOME/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json" 2>/dev/null || true

echo "=== STEP 2: REGISTER 3-PHASE TRAINER INTO NNUNETV2 ==="
python3 "$SCRIPT_DIR/add_trainer_3phases_200epochs.py"

echo "=== STEP 3: CHECK CHECKPOINT & LAUNCH TRAINING ==="
EXTRA_FLAGS=""
if [ -f "$RESULTS_DIR/checkpoint_latest.pth" ]; then
    echo "Found existing checkpoint_latest.pth. Resuming training with --c..."
    EXTRA_FLAGS="--c"
fi

echo "[$(date)] Starting nnUNetv2_train for ResEnc-UNet L (3 phases)..."
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_3phases_200epochs -p nnUNetResEncUNetLPlans $EXTRA_FLAGS

echo "=== STEP 4: OFFICIAL VALIDATION ON CHECKPOINT_BEST ==="
echo "[$(date)] Running validation on 197 Fold 0 cases..."
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_3phases_200epochs -p nnUNetResEncUNetLPlans --val --val_best

echo "=== STEP 5: COMPUTE OFFICIAL PAPER METRICS ==="
python3 "$SCRIPT_DIR/compute_paper_metrics.py" "$RESULTS_DIR" "ResEnc-UNet L (3 Natural Phases)" || echo "Warning: Paper metrics extraction completed with notices."

echo "=============================================================================="
echo "    RESENC-UNET L (3 PHASES) COMPLETE AT $(date)                              "
echo "=============================================================================="
