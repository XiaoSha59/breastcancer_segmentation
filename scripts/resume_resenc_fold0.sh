#!/usr/bin/env bash
# resume_resenc_fold0.sh
# Resume ResEnc-UNet L training on Fold 0 from last checkpoint with memory safeguards and auto-shutdown.

export nnUNet_raw="/home/XiaoSha/nnUNet_raw"
export nnUNet_preprocessed="/home/XiaoSha/nnUNet_preprocessed"
export nnUNet_results="/home/XiaoSha/nnUNet_results"
export PATH="/home/XiaoSha/.local/bin:$PATH"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Crucial: limit data augmentation workers to prevent RAM OOM
export nnUNet_n_proc_DA=4

FOLD_DIR="$nnUNet_results/Dataset501_BreastDCE_ISPY2/nnUNetTrainer_200epochs__nnUNetResEncUNetLPlans__3d_fullres/fold_0"

# Auto-shutdown on script exit (whether successful or error)
cleanup() {
    EXIT_STATUS=$?
    echo "[$(date)] Training script exiting with status $EXIT_STATUS."
    if [ "${GCP_CLOUD_INSTANCE:-0}" = "1" ]; then
        echo "[$(date)] [Cloud Trap] GCP_CLOUD_INSTANCE=1. Shutting down VM in 120s..."
        sleep 120
        sudo /sbin/shutdown -h now || true
    else
        echo "[$(date)] [Server Mode] System remains running."
    fi
}
trap cleanup EXIT

echo "=== CHECKING CHECKPOINTS ==="
if [ ! -f "$FOLD_DIR/checkpoint_latest.pth" ]; then
    if [ -f "$FOLD_DIR/checkpoint_best.pth" ]; then
        echo "checkpoint_latest.pth not found, copying checkpoint_best.pth -> checkpoint_latest.pth"
        cp "$FOLD_DIR/checkpoint_best.pth" "$FOLD_DIR/checkpoint_latest.pth"
    else
        echo "ERROR: Neither checkpoint_latest.pth nor checkpoint_best.pth found in $FOLD_DIR"
        exit 1
    fi
else
    echo "Found checkpoint_latest.pth: $(ls -lh "$FOLD_DIR/checkpoint_latest.pth")"
fi

echo "=== RESUMING TRAINING (Fold 0, ResEnc L, nnUNet_n_proc_DA=4) ==="
echo "[$(date)] Starting nnUNetv2_train..."
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_200epochs -p nnUNetResEncUNetLPlans --c

echo "=== TRAINING FINISHED AT $(date) ==="
