#!/usr/bin/env bash
# run_resenc_fold0.sh
# End-to-end execution of Krikun et al. ResEnc-UNet L on I-SPY2 Fold 0.

set -e

export nnUNet_raw="/home/XiaoSha/nnUNet_raw"
export nnUNet_preprocessed="/home/XiaoSha/nnUNet_preprocessed"
export nnUNet_results="/home/XiaoSha/nnUNet_results"
export PATH="/home/XiaoSha/.local/bin:$PATH"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "=== STEP 1: LINKING DATASET ==="
if [ ! -e "$nnUNet_raw/Dataset501_BreastDCE_ISPY2" ]; then
    ln -s /home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2 "$nnUNet_raw/Dataset501_BreastDCE_ISPY2"
    echo "Linked Dataset501_BreastDCE_ISPY2 to $nnUNet_raw"
fi

echo "=== STEP 2: RUNNING PLANNING & PREPROCESSING (ResEnc L) ==="
# -pl nnUNetPlannerResEncL generates nnUNetResEncUNetLPlans
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres -pl nnUNetPlannerResEncL

echo "=== STEP 3: INJECTING 5-FOLD CV SPLITS ==="
cp /home/XiaoSha/data/processed/Dataset501_BreastDCE_ISPY2/splits_final.json "$nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"
echo "Injected splits_final.json successfully."

echo "=== STEP 4: LAUNCHING TRAINING (Fold 0, ResEnc L, 200 Epochs) ==="
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_200epochs -p nnUNetResEncUNetLPlans

echo "=== TRAINING COMPLETE! ==="
