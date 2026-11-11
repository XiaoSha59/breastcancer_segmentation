#!/usr/bin/env python3
"""
test_umamba_trainer_discovery.py
================================
Validates that nnUNetTrainerUMambaBot_200epochs can be discovered and instantiated
properly by nnU-Net v2's recursive trainer resolver using real Dataset501 plans.
"""

import os
import sys
import json
import torch

try:
    from nnunetv2.run.run_training import recursive_find_trainer_class_by_name
    HAS_NNUNET = True
except ImportError:
    HAS_NNUNET = False
    print("nnunetv2 not installed locally. This test is intended for execution on the GCP VM.")

def main():
    if not HAS_NNUNET:
        print("Skipping dynamic nnUNet discovery test on local machine without nnunetv2.")
        return 0

    cls_name = "nnUNetTrainerUMambaBot_200epochs"
    trainer_cls = recursive_find_trainer_class_by_name(cls_name)
    if trainer_cls is None:
        print(f"FAILED: Could not find trainer class '{cls_name}'!")
        sys.exit(1)
    print(f"SUCCESS: Found trainer class -> {trainer_cls}")

    prep_dir = "/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2"
    plans_path = os.path.join(prep_dir, "nnUNetPlans.json")
    dataset_path = os.path.join(prep_dir, "dataset.json")

    if not os.path.exists(plans_path):
        print(f"Warning: {plans_path} not found. Testing class definition only.")
        return 0

    with open(plans_path, "r") as f:
        real_plans = json.load(f)
    real_plans["continue_training"] = False

    with open(dataset_path, "r") as f:
        real_dataset_json = json.load(f)

    try:
        instance = trainer_cls(
            plans=real_plans,
            configuration="3d_fullres",
            fold=0,
            dataset_json=real_dataset_json,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        print(f"SUCCESS: Instantiated {cls_name} with REAL Dataset501 plans!")
        print(f"num_epochs: {instance.num_epochs}")
        assert instance.num_epochs == 200, "num_epochs must be 200!"

        print("Testing instance.initialize()...")
        instance.initialize()
        print("SUCCESS: Trainer initialized network architecture!")

        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dummy_x = torch.randn(2, 5, 128, 128, 64, device=dev)
        print(f"Testing network forward pass with input shape {dummy_x.shape} on {dev}...")
        with torch.no_grad():
            outputs = instance.network(dummy_x)
        if isinstance(outputs, list):
            print(f"SUCCESS: Deep supervision outputs ({len(outputs)} stages): {[o.shape for o in outputs]}")
        else:
            print(f"SUCCESS: Network output shape: {outputs.shape}")

        print("ALL U-MAMBA BOT 3D ASSERTIONS PASSED PERFECTLY!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Initialization or forward pass failed: {e}")
        sys.exit(1)

    return 0

if __name__ == "__main__":
    sys.exit(main())
