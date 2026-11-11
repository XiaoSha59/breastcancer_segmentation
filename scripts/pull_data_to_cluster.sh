#!/usr/bin/env bash
# pull_data_to_cluster.sh
# ==============================================================================
# Automated utility to pull 64GB preprocessed Breast DCE-MRI data (.b2nd / .npy)
# to university server / cluster storage WITHOUT re-preprocessing from scratch.
# ==============================================================================

set -e

DEST_DIR="${1:-$HOME/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2}"
mkdir -p "$DEST_DIR"

echo "========================================================================"
echo " PULL PREPROCESSED DATA TO UNIVERSITY SERVER"
echo " Destination: $DEST_DIR"
echo "========================================================================"

echo ""
echo "Select source method:"
echo "  1) Direct SCP / Rsync from GCP VM (Fastest if GCP VM is RUNNING)"
echo "  2) Download from Kaggle Dataset (Private, 100MB/s)"
echo "  3) Download from Google Drive / Cloud Link (via gdown)"
echo "========================================================================"

# Auto-detect or ask
if [ -n "$TRANSFER_MODE" ]; then
    CHOICE="$TRANSFER_MODE"
else
    read -p "Enter choice [1, 2, or 3] (default 1): " CHOICE
    CHOICE="${CHOICE:-1}"
fi

case "$CHOICE" in
    1)
        GCP_IP="${GCP_IP:-34.87.101.121}"
        GCP_USER="${GCP_USER:-XiaoSha}"
        echo "[Mode 1] Pulling directly from GCP VM ($GCP_USER@$GCP_IP)..."
        echo "Source: /home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/"
        
        rsync -avzP --exclude '*.tmp' "$GCP_USER@$GCP_IP:/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/" "$DEST_DIR/" || {
            echo "Rsync failed or SSH key needed. Falling back to SCP..."
            scp -r "$GCP_USER@$GCP_IP:/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres" "$DEST_DIR/"
            scp "$GCP_USER@$GCP_IP:/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json" "$DEST_DIR/"
        }
        ;;
    2)
        echo "[Mode 2] Pulling from Kaggle..."
        if ! command -v kaggle &> /dev/null; then
            echo "Installing kaggle CLI..."
            pip install kaggle
        fi
        kaggle datasets download -d xiaosha/ispy2-breast-dce-mri-preprocessed -p "$DEST_DIR" --unzip
        ;;
    3)
        echo "[Mode 3] Pulling via gdown from Google Drive..."
        if ! command -v gdown &> /dev/null; then
            pip install gdown
        fi
        read -p "Enter Google Drive File ID / Link: " DRIVE_URL
        gdown "$DRIVE_URL" -O "$DEST_DIR/preprocessed.tar.gz"
        tar -xvf "$DEST_DIR/preprocessed.tar.gz" -C "$DEST_DIR"
        rm -f "$DEST_DIR/preprocessed.tar.gz"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=== VERIFYING DOWNLOADED FILES ==="
if [ -d "$DEST_DIR/nnUNetPlans_3d_fullres" ] && [ -f "$DEST_DIR/splits_final.json" ]; then
    NUM_CASES=$(ls -1 "$DEST_DIR/nnUNetPlans_3d_fullres/"*.b2nd 2>/dev/null | wc -l || ls -1 "$DEST_DIR/nnUNetPlans_3d_fullres/"*.npy 2>/dev/null | wc -l)
    echo "SUCCESS: Found $NUM_CASES preprocessed files and splits_final.json!"
    echo "Ready for training MedNeXt-L k3 and U-Mamba Bot without re-preprocessing."
else
    echo "Warning: Please check $DEST_DIR to ensure nnUNetPlans_3d_fullres and splits_final.json are present."
fi
