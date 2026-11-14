#!/usr/bin/env bash
# run_re_evaluate_umamba_swinunetr.sh
# ==============================================================================
# Precision-Fixed Re-Evaluation for U-Mamba (Bot 3D) and SwinUNETR V1 on I-SPY2 Fold 0.
# Re-evaluates on all 197 validation cases with corrected boundary precision logic.
# Automatically powers down the GCP VM upon completion to prevent idle costs.
# ==============================================================================

set -e

export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export AUTO_SHUTDOWN=1
export GCP_CLOUD_INSTANCE=1

pkill -f safety_guard || true
sudo pkill -f shutdown || true

cleanup() {
    EXIT_STATUS=$?
    echo "[$(date)] Re-evaluation runner exiting with status $EXIT_STATUS."
    sync
    if [ "${AUTO_SHUTDOWN:-1}" = "1" ] || [ "${GCP_CLOUD_INSTANCE:-1}" = "1" ]; then
        if [ $EXIT_STATUS -ne 0 ]; then
            echo "[$(date)] [Cloud Trap] Re-eval exited with ERROR ($EXIT_STATUS). Syncing filesystem and waiting 180s before auto-shutdown..."
            sleep 180
        else
            echo "[$(date)] [Cloud Trap] Re-eval completed successfully ($EXIT_STATUS). Waiting 120s before powering off VM..."
            sleep 120
        fi
        sudo /sbin/shutdown -h now || sudo poweroff || true
    fi
}
trap cleanup EXIT ERR INT TERM

DATA_DIR="/home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2"
PREP_DIR="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"
SPLITS_FILE="/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"

UMAMBA_DIR="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/UMambaBot_3d_200epochs/fold_0"
if [ ! -d "$UMAMBA_DIR" ]; then
    UMAMBA_DIR="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/UMamba_Bot_200epochs/fold_0"
fi

SWIN_DIR="/home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/SwinUNETR_200epochs/fold_0"

echo "================================================================================"
echo "[$(date)] STARTING ACCURACY-FIXED RE-EVALUATION: U-MAMBA & SWINUNETR V1"
echo "================================================================================"

echo ">>> [1/2] RE-EVALUATING U-MAMBA BOT 3D (197 CASES)..."
python3 -u /home/XiaoSha/project/scripts/evaluate_umamba_fold0.py \
    --data_dir "$DATA_DIR" \
    --splits_file "$SPLITS_FILE" \
    --output_dir "$UMAMBA_DIR" \
    --fold 0 \
    --patch_size 128 128 64 \
    --num_workers 4

echo ">>> [2/2] RE-EVALUATING SWINUNETR V1 (197 CASES)..."
python3 -u /home/XiaoSha/project/scripts/evaluate_swinunetr_fold0.py \
    --data_dir "$DATA_DIR" \
    --prep_dir "$PREP_DIR" \
    --splits_file "$SPLITS_FILE" \
    --output_dir "$SWIN_DIR" \
    --fold 0 \
    --patch_size 96 96 96 \
    --num_workers 4

echo "================================================================================"
echo "[$(date)] BOTH RE-EVALUATIONS COMPLETED SUCCESSFULLY AT $(date)"
echo "================================================================================"
