from __future__ import annotations

from dataclasses import dataclass, field

from slerobot.configs.types import PipelineFeatureType, PolicyFeature

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("rename_observations_processor")
@dataclass
class RenameObservationsProcessorStep(ProcessorStep):
    rename_map: dict[str, str] = field(default_factory=dict)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return new_transition
        if not isinstance(observation, dict):
            raise ValueError(f"Observation should be a RobotObservation type (dict), but got {type(observation)}")

        renamed_observation = {
            self.rename_map.get(key, key): value for key, value in observation.items()
        }
        new_transition[TransitionKey.OBSERVATION] = renamed_observation
        return new_transition

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = {
            feature_type: feature_map.copy() for feature_type, feature_map in features.items()
        }

        observation_features = transformed.get(PipelineFeatureType.OBSERVATION)
        if observation_features is None:
            return transformed

        transformed[PipelineFeatureType.OBSERVATION] = {
            self.rename_map.get(key, key): value for key, value in observation_features.items()
        }
        return transformed
