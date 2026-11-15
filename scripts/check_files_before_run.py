#!/usr/bin/env python3
"""
check_files_before_run.py
=========================
Fast pre-flight check to verify that all required scripts, architectures, 
datasets, and splits are 100% present before starting training on node1.
"""

import os
import sys

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, ".."))
    os.chdir(project_dir)

    print("==============================================================================")
    print("  PRE-FLIGHT AUDIT: VERIFYING ALL SOTA MODEL SCRIPTS ON UNIVERSITY SERVER")
    print("==============================================================================")

    required_scripts = [
        # SegMamba 3D
        "scripts/segmamba_3d.py",
        "scripts/train_segmamba_fold0.py",
        "scripts/evaluate_segmamba_fold0.py",
        "scripts/run_segmamba_server.sh",
        # SwinUNETR v2
        "scripts/train_swinunetr_v2_fold0.py",
        "scripts/evaluate_swinunetr_v2_fold0.py",
        "scripts/run_swinunetr_v2_server.sh",
        # PrimusV2-S
        "scripts/primus_v2.py",
        "scripts/train_primus_v2_fold0.py",
        "scripts/evaluate_primus_v2_fold0.py",
        "scripts/run_primus_v2_server.sh",
        # Automation & Evaluation
        "scripts/run_sota_queue.sh",
        "scripts/compute_paper_metrics.py",
    ]

    all_scripts_ok = True
    print("\n[1/3] KIỂM TRA MÃ NGUỒN VÀ SHELL SCRIPTS:")
    for path in required_scripts:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  [OK] {path:<38} ({size:,} bytes)")
        else:
            print(f"  [MISSING] {path:<38} <-- THIẾU FILE NÀY!")
            all_scripts_ok = False

    print("\n[2/3] KIỂM TRA FILE CHIA TẬP (splits_final.json):")
    split_candidates = [
        "splits_final.json",
        "data/preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json",
        os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
        os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/splits_final.json"),
    ]
    split_found = None
    for p in split_candidates:
        if os.path.exists(p):
            split_found = p
            break

    if split_found:
        print(f"  [OK] Tìm thấy splits_final.json tại: {split_found}")
    else:
        print("  [MISSING] Chưa tìm thấy splits_final.json!")
        all_scripts_ok = False

    print("\n[3/3] KIỂM TRA THƯ MỤC TIỀN XỬ LÝ (.npy / .b2nd):")
    prep_candidates = [
        "data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres",
        os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
        os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2/nnUNetPlans_3d_fullres"),
    ]
    prep_found = None
    for p in prep_candidates:
        if os.path.isdir(p):
            files = [f for f in os.listdir(p) if f.endswith(".npy") or f.endswith(".b2nd")]
            if len(files) > 0:
                prep_found = (p, len(files))
                break

    if prep_found:
        print(f"  [OK] Tìm thấy thư mục preprocessed: {prep_found[0]} ({prep_found[1]} files)")
    else:
        print("  [MISSING] Không tìm thấy thư mục preprocessed chứa .npy/.b2nd!")
        all_scripts_ok = False

    print("\n==============================================================================")
    if all_scripts_ok:
        print("  KET QUA: TOÀN BỘ FILE & DỮ LIỆU ĐÃ ĐẦY ĐỦ 100%, SẴN SÀNG CHẠY!")
    else:
        print("  CANH BAO: VẪN CÒN FILE BỊ THIẾU Ở TRÊN, HÃY KIỂM TRA LẠI!")
    print("==============================================================================")

if __name__ == "__main__":
    main()
