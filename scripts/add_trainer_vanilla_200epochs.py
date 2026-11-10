#!/usr/bin/env python3
"""
add_trainer_vanilla_200epochs.py
================================
Registers nnUNetTrainer_Vanilla3DUNet_200epochs into the installed nnunetv2 package.
Architecture: Plain Convolutional 3D U-Net without Deep Supervision (Vanilla 3D U-Net, Özgün Çiçek et al. 2016).
Training Length: 200 epochs (same as ResEnc-UNet L and Standard nnU-Net).
"""

import os
import shutil

nnunet_pkg = __import__("nnunetv2")
base_dir = os.path.dirname(nnunet_pkg.__file__)

target_file = os.path.join(
    base_dir,
    "training", "nnUNetTrainer", "variants", "network_architecture",
    "nnUNetTrainerNoDeepSupervision.py"
)

print(f"Target file for registration: {target_file}")

with open(target_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Clean up any previously added nnUNetTrainer_Vanilla3DUNet_200epochs
cleaned_lines = []
skip = False
for line in lines:
    if "class nnUNetTrainer_Vanilla3DUNet_200epochs" in line or "# Added for Vanilla 3D U-Net" in line:
        skip = True
    if not skip:
        cleaned_lines.append(line)
    elif line.strip() == "" and skip:
        # after class definition
        pass

cleaned_content = "".join(cleaned_lines).rstrip()

TRAINER_CLASS_NAME = "nnUNetTrainer_Vanilla3DUNet_200epochs"

addition = f"""

# Added for Vanilla 3D U-Net Ablation Baseline (200 Epochs, No Deep Supervision)
class {TRAINER_CLASS_NAME}(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
"""

with open(target_file, "w", encoding="utf-8") as f:
    f.write(cleaned_content + addition + "\n")

print(f"Successfully wrote clean {TRAINER_CLASS_NAME} to:\n  {target_file}")

# Clear __pycache__
cache_dir = os.path.join(os.path.dirname(target_file), "__pycache__")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Cleared cache at: {cache_dir}")

print("Done registration update.")
