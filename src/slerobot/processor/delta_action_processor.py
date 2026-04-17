from dataclasses import dataclass
from typing import Dict

import torch

from slerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from .core import PolicyAction, RobotAction
from .pipeline import PolicyActionProcessorStep, ProcessorStepRegistry, RobotActionProcessorStep


@ProcessorStepRegistry.register("map_tensor_to_delta_action_dict")
@dataclass
class MapTensorToDeltaActionDictStep(PolicyActionProcessorStep):
    use_gripper: bool = True

    def action(self, action: PolicyAction) -> RobotAction:
        ...

    def transform_features(
        self,
        features: Dict[PipelineFeatureType, Dict[str, PolicyFeature]],
    ) -> Dict[PipelineFeatureType, Dict[str, PolicyFeature]]:
        ...


@ProcessorStepRegistry.register("map_delta_action_to_robot_action")
@dataclass
class MapDeltaActionToRobotActionStep(RobotActionProcessorStep):
    position_scale: float = 1.0
    noise_threshold: float = 1e-3

    def action(self, action: RobotAction) -> RobotAction:
        ...

    def transform_features(
        self,
        features: Dict[PipelineFeatureType, Dict[str, PolicyFeature]],
    ) -> Dict[PipelineFeatureType, Dict[str, PolicyFeature]]:
        ...