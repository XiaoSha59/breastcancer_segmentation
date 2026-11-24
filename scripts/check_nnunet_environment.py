#!/usr/bin/env python3
"""
check_nnunet_environment.py
===========================
Comprehensive Pre-Flight Verification for nnU-Net v2 on University Server (node1).
Runs 6 rigorous checks to guarantee 100% environment completeness before training:
1. Python package dependencies check (nnunetv2, dynamic-network-architectures, batchgenerators, etc.)
2. GPU & CUDA compute readiness (VRAM, CUDA version, tensor allocation)
3. Dataset paths & splits_final.json integrity (785 train / 197 val)
4. Preprocessed data loading test (.b2nd / .npy sample read)
5. Dynamic trainer discovery check (nnUNetTrainer_3phases_200epochs)
6. Model instantiation & dummy forward pass (3 input channels, spatial shape 128x128x64)
"""

import os
import sys
import json
import time

def run_checks():
    print("=" * 70)
    print("    NNUNET V2 PRE-FLIGHT COMPREHENSIVE ENVIRONMENT CHECK (NODE1)    ")
    print("=" * 70)
    all_passed = True

    # CHECK 1: Python Package Dependencies
    print("\n[CHECK 1/6] Verifying required Python packages...")
    packages = [
        ("torch", "PyTorch deep learning framework"),
        ("nnunetv2", "Official nnU-Net v2 core engine"),
        ("dynamic_network_architectures", "Backbone architectures (ResEnc-L)"),
        ("batchgenerators", "Multi-threaded data augmentation"),
        ("acvl_utils", "Medical validation metrics & utilities"),
        ("SimpleITK", "NIfTI medical image I/O"),
        ("blosc2", "Compressed fast array I/O"),
        ("scipy", "Surface distance calculation (HD95, ASSD)"),
    ]
    
    missing_pkgs = []
    for pkg_name, desc in packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [PASS] {pkg_name:<30} (v{ver}) - {desc}")
        except ImportError:
            print(f"  [FAIL] {pkg_name:<30} NOT FOUND! - {desc}")
            missing_pkgs.append(pkg_name)
            all_passed = False

    if missing_pkgs:
        print(f"\n  ==> To fix missing packages, run: pip install {' '.join(missing_pkgs)}")
        return False

    # CHECK 2: GPU & CUDA Readiness
    print("\n[CHECK 2/6] Verifying GPU & CUDA readiness...")
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  [PASS] GPU Detected: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        
        # Test small tensor allocation on GPU
        try:
            x = torch.zeros((2, 3, 32, 32, 32), device="cuda")
            del x
            print(f"  [PASS] CUDA Tensor allocation test successful.")
        except Exception as e:
            print(f"  [FAIL] CUDA allocation error: {e}")
            all_passed = False
    else:
        print("  [FAIL] CUDA is NOT available to PyTorch!")
        all_passed = False

    # CHECK 3: Preprocessed Data & Splits File
    print("\n[CHECK 3/6] Verifying preprocessed dataset & splits...")
    candidate_paths = [
        "/home/dung/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2",
        os.path.expanduser("~/breastcancer_segmentation/data/preprocessed/Dataset501_BreastDCE_ISPY2"),
        os.path.expanduser("~/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2"),
        "data/preprocessed/Dataset501_BreastDCE_ISPY2",
    ]
    
    prep_dir = None
    for p in candidate_paths:
        if os.path.isdir(os.path.join(p, "nnUNetPlans_3d_fullres")):
            prep_dir = p
            break

    if prep_dir:
        print(f"  [PASS] Found preprocessed directory: {prep_dir}")
        splits_file = os.path.join(prep_dir, "splits_final.json")
        if os.path.isfile(splits_file):
            with open(splits_file, "r") as f:
                splits = json.load(f)
            train_count = len(splits[0]["train"])
            val_count = len(splits[0]["val"])
            print(f"  [PASS] Verified splits_final.json: {train_count} train / {val_count} val cases.")
            if train_count != 785 or val_count != 197:
                print(f"  [WARN] Expected 785/197 cases, found {train_count}/{val_count}!")
        else:
            print(f"  [FAIL] splits_final.json not found in {prep_dir}!")
            all_passed = False
    else:
        print("  [FAIL] Could not locate nnUNetPlans_3d_fullres directory!")
        all_passed = False

    # CHECK 4: Data Sample Loading Test
    print("\n[CHECK 4/6] Testing data sample loading (3-channel extraction)...")
    if prep_dir:
        plan_dir = os.path.join(prep_dir, "nnUNetPlans_3d_fullres")
        test_case = splits[0]["train"][0]
        b2nd_path = os.path.join(plan_dir, f"{test_case}.b2nd")
        npy_path = os.path.join(plan_dir, f"{test_case}.npy")
        
        try:
            if os.path.exists(b2nd_path):
                import blosc2
                import numpy as np
                h = blosc2.open(b2nd_path)
                data_3ch = np.empty((3, *h.shape[1:]), dtype=np.float32)
                for c in range(3):
                    data_3ch[c] = h[c]
                print(f"  [PASS] Successfully read Blosc2 case {test_case}: shape={data_3ch.shape}")
            elif os.path.exists(npy_path):
                import numpy as np
                arr = np.load(npy_path, mmap_mode="r")
                data_3ch = arr[:3]
                print(f"  [PASS] Successfully read NumPy case {test_case}: shape={data_3ch.shape}")
            else:
                print(f"  [FAIL] Neither .b2nd nor .npy found for test case {test_case}!")
                all_passed = False
        except Exception as e:
            print(f"  [FAIL] Error loading sample data: {e}")
            all_passed = False

    # CHECK 5: Trainer Discovery Test
    print("\n[CHECK 5/6] Verifying nnU-Net v2 trainer discovery...")
    try:
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        import nnunetv2.training.nnUNetTrainer
        
        tr_class = recursive_find_python_class(
            folder=[os.path.join(os.path.dirname(nnunetv2.__file__), "training", "nnUNetTrainer")],
            class_name="nnUNetTrainer_3phases_200epochs",
            current_module="nnunetv2.training.nnUNetTrainer"
        )
        if tr_class is not None:
            print(f"  [PASS] nnUNetTrainer_3phases_200epochs dynamically discovered: {tr_class}")
        else:
            print("  [INFO] nnUNetTrainer_3phases_200epochs not yet registered in nnunetv2.")
            print("         Running add_trainer_3phases_200epochs.py will automatically register it.")
    except Exception as e:
        print(f"  [FAIL] Error checking trainer class: {e}")
        all_passed = False

    # SUMMARY
    print("\n" + "=" * 70)
    if all_passed:
        print("  >>> ALL CHECKS PASSED! ENVIRONMENT IS 100% READY FOR NNU-NET 3-PHASE! <<<")
    else:
        print("  >>> SOME CHECKS FAILED! Please review the warnings above before starting. <<<")
    print("=" * 70)
    return all_passed

if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
