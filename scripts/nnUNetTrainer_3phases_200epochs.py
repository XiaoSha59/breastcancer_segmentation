import os
import torch
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class nnUNetTrainer_3phases_200epochs(nnUNetTrainer):
    """
    Official nnU-Net Trainer for 3 Natural Phases (Pre, Early, Late) Ablation Study.
    - 200 Epochs (consistent with 5-phase baseline and 3D_UNET_BEST_PRACTICES.md).
    - Automatically slices 5-phase preprocessed data to 3 natural phases [:3].
    - 100% REUSES existing 64GB Dataset501_BreastDCE_ISPY2 preprocessed data without duplicating files.
    - Zero disk duplication, zero auto-shutdown.
    - Compatible with BOTH Standard 3D nnU-Net (-p nnUNetPlans) and ResEnc-UNet L (-p nnUNetResEncUNetLPlans).
    - Preserves exact 100% precision settings as GCP baseline (AMP FP16 + PyTorch default float32 precision).
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
        return get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            3,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision
        )

    def train_step(self, batch: dict) -> dict:
        if batch['data'].shape[1] > 3:
            batch['data'] = batch['data'][:, :3].contiguous()
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        if batch['data'].shape[1] > 3:
            batch['data'] = batch['data'][:, :3].contiguous()
        return super().validation_step(batch)
