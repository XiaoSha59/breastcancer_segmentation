#!/usr/bin/env python3
"""
add_trainer_3phases_200epochs.py
================================
Registers nnUNetTrainer_3phases_200epochs into the installed nnunetv2 package.
Ablation Study: 3 Natural DCE-MRI Phases (Pre, Early, Late) on Fold 0 (200 Epochs).
Reuses 100% of existing 64GB Dataset501_BreastDCE_ISPY2 preprocessed data without duplicating files.
Compatible with BOTH:
  - Standard 3D nnU-Net: -p nnUNetPlans
  - ResEnc-UNet L:       -p nnUNetResEncUNetLPlans
"""

import os
import shutil
import sys

TRAINER_CODE = '''import os
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainer_3phases_200epochs(nnUNetTrainer):
    """
    Official nnU-Net Trainer for 3 Natural Phases (Pre, Early, Late) Ablation Study.
    - 200 Epochs (consistent with 5-phase baseline and 3D_UNET_BEST_PRACTICES.md).
    - Automatically slices 5-phase preprocessed data to 3 natural phases [:3].
    - 100% REUSES existing 64GB Dataset501_BreastDCE_ISPY2 preprocessed data without duplicating files.
    - Zero disk duplication, zero auto-shutdown.
    - Compatible with BOTH Standard 3D nnU-Net (-p nnUNetPlans) and ResEnc-UNet L (-p nnUNetResEncUNetLPlans).
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # Restrict channel_names in dataset_json to 3 natural phases:
        # 0: Pre-contrast (I_0), 1: Early post-contrast (I_e), 2: Late post-contrast (I_l)
        if "channel_names" in dataset_json:
            dataset_json = dict(dataset_json)
            dataset_json["channel_names"] = {
                "0": dataset_json["channel_names"].get("0", "pre"),
                "1": dataset_json["channel_names"].get("1", "early"),
                "2": dataset_json["channel_names"].get("2", "late"),
            }
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200

    def initialize(self):
        super().initialize()
        # Wrap dataset_class so load_case always returns only 3 natural phases [:3]
        base_dataset_cls = self.dataset_class
        import numpy as np

        class nnUNetDataset3Phases(base_dataset_cls):
            def load_case(self, identifier):
                data, seg, seg_prev, properties = super().load_case(identifier)
                # Slices strictly 3 natural phases with contiguous memory layout for fast Cython DA
                return np.ascontiguousarray(data[:3]), seg, seg_prev, properties

        self.dataset_class = nnUNetDataset3Phases

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        # Strictly enforce 3 input channels for the 3 natural phases ablation
        return plans_manager.get_network_architecture(
            architecture_class_name=configuration_manager.network_arch_class_name,
            arch_init_kwargs=configuration_manager.network_arch_init_kwargs,
            arch_init_kwargs_req_import=configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels=3,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision
        )

    def train_step(self, batch: dict) -> dict:
        if batch['data'].shape[1] > 3:
            batch['data'] = batch['data'][:, :3].contiguous()
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        if batch['data'].shape[1] > 3:
            batch['data'] = batch['data'][:, :3].contiguous()
        return super().validation_step(batch)
'''


def register_trainer():
    try:
        import nnunetv2
        base_dir = os.path.dirname(nnunetv2.__file__)
    except ImportError:
        print("[Warning] nnunetv2 is not installed in the active environment.")
        return False

    trainer_dir = os.path.join(base_dir, "training", "nnUNetTrainer", "variants", "network_architecture")
    os.makedirs(trainer_dir, exist_ok=True)
    trainer_file = os.path.join(trainer_dir, "nnUNetTrainer_3phases_200epochs.py")

    with open(trainer_file, "w", encoding="utf-8") as f:
        f.write(TRAINER_CODE)
    print(f"Successfully registered nnUNetTrainer_3phases_200epochs at:\n  {trainer_file}")

    # Clear __pycache__ in variants and parent to ensure clean dynamic discovery
    for cache_sub in [
        os.path.join(trainer_dir, "__pycache__"),
        os.path.join(base_dir, "training", "nnUNetTrainer", "__pycache__")
    ]:
        if os.path.exists(cache_sub):
            shutil.rmtree(cache_sub, ignore_errors=True)
            print(f"Cleared cache at: {cache_sub}")

    print("Trainer registration complete! Discoverable by nnUNetv2_train via: -tr nnUNetTrainer_3phases_200epochs")
    return True


if __name__ == "__main__":
    success = register_trainer()
    sys.exit(0 if success else 1)
