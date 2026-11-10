#!/usr/bin/env python3
import sys
import os
import json
import torch
from nnunetv2.run.run_training import recursive_find_trainer_class_by_name

cls_name = "nnUNetTrainer_Vanilla3DUNet_200epochs"
trainer_cls = recursive_find_trainer_class_by_name(cls_name)
print(f"SUCCESS: Found trainer class -> {trainer_cls}")

# Load REAL plans and dataset.json from Dataset501
prep_dir = "/home/XiaoSha/nnUNet_preprocessed/Dataset501_BreastDCE_ISPY2"
plans_path = os.path.join(prep_dir, "nnUNetPlans.json")
dataset_path = os.path.join(prep_dir, "dataset.json")

with open(plans_path, "r") as f:
    real_plans = json.load(f)
real_plans["continue_training"] = False

with open(dataset_path, "r") as f:
    real_dataset_json = json.load(f)

try:
    instance = trainer_cls(plans=real_plans, configuration="3d_fullres", fold=0, dataset_json=real_dataset_json, device=torch.device('cuda:0'))
    print(f"SUCCESS: Instantiated {cls_name} with REAL Dataset501 plans!")
    print(f"num_epochs: {instance.num_epochs}")
    print(f"enable_deep_supervision: {instance.enable_deep_supervision}")
    assert instance.num_epochs == 200, "num_epochs should be 200"
    assert instance.enable_deep_supervision == False, "enable_deep_supervision should be False"
    print("ALL REAL ASSERTIONS PASSED PERFECTLY!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Instantiation failed: {e}")
    sys.exit(1)
