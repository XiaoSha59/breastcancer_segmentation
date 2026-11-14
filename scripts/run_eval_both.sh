#!/usr/bin/env bash
set -e
echo "=================================================="
echo "[$(date)] STARTING SWINUNETR V1 AND V2 EVALUATION"
echo "=================================================="

echo ">>> [1/2] EVALUATING SWINUNETR V1..."
python3 -u /home/XiaoSha/project/scripts/evaluate_swinunetr_fold0.py \
    --splits_file /home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json \
    --prep_dir /home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres \
    --output_dir /home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/SwinUNETR_200epochs/fold_0

echo ">>> [2/2] EVALUATING SWINUNETR V2..."
python3 -u /home/XiaoSha/project/scripts/evaluate_swinunetr_v2_fold0.py \
    --splits_file /home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json \
    --prep_dir /home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres \
    --output_dir /home/XiaoSha/nnUNet_results/Dataset501_BreastDCE_ISPY2/SwinUNETR_v2_200epochs/fold_0

echo "=================================================="
echo "[$(date)] FINISHED BOTH EVALUATIONS SUCCESSFULLY"
echo "=================================================="
