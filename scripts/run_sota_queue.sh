#!/usr/bin/env bash
# run_sota_queue.sh
# ==============================================================================
# Sequential Automated Queue for Remaining SOTA Models:
# 1. SegMamba 3D (MICCAI 2024 Pure 3D State Space Model)
# 2. SwinUNETR v2 (Hierarchical Vision Transformer, MONAI)
# 3. PrimusV2-S (EVA ViT with 3D Axial RoPE, DKFZ 2024)
#
# Server Safety:
# - Runs sequentially to strictly avoid VRAM collision on RTX 4090 24GB.
# - Zero auto-shutdown (cluster runs 24/7).
# - CPU threads constrained (OMP_NUM_THREADS=4).
# - Bounded checkpoints only (checkpoint_best.pth & checkpoint_latest.pth).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================================================="
echo "  STARTING SOTA SEQUENTIAL AUTO-QUEUE (SEGMAMBA -> SWINUNETR V2 -> PRIMUSV2)"
echo "  Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

# STAGE 1: SegMamba 3D
echo ""
echo "[STAGE 1/3] Launching SegMamba 3D..."
START_T1=$(date +%s)
bash "$SCRIPT_DIR/run_segmamba_server.sh"
END_T1=$(date +%s)
DUR_T1=$(( (END_T1 - START_T1) / 60 ))
echo "[STAGE 1/3 COMPLETED] SegMamba 3D finished in $DUR_T1 minutes!"

# Cooldown buffer (30 seconds to release PyTorch CUDA context)
sleep 30

# STAGE 2: SwinUNETR v2
echo ""
echo "[STAGE 2/3] Launching SwinUNETR v2..."
START_T2=$(date +%s)
bash "$SCRIPT_DIR/run_swinunetr_v2_server.sh"
END_T2=$(date +%s)
DUR_T2=$(( (END_T2 - START_T2) / 60 ))
echo "[STAGE 2/3 COMPLETED] SwinUNETR v2 finished in $DUR_T2 minutes!"

# Cooldown buffer
sleep 30

# STAGE 3: PrimusV2-S
echo ""
echo "[STAGE 3/3] Launching PrimusV2-S..."
START_T3=$(date +%s)
bash "$SCRIPT_DIR/run_primus_v2_server.sh"
END_T3=$(date +%s)
DUR_T3=$(( (END_T3 - START_T3) / 60 ))
echo "[STAGE 3/3 COMPLETED] PrimusV2-S finished in $DUR_T3 minutes!"

echo ""
echo "=============================================================================="
echo "  ALL SOTA MODELS IN QUEUE COMPLETED SUCCESSFULLY!"
echo "  Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="
