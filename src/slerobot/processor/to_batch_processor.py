from __future__ import annotations

from dataclasses import dataclass

from slerobot.configs.types import PipelineFeatureType, PolicyFeature

from .core import EnvTransition
from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("to_batch_processor")
@dataclass
class ToBatchProcessorStep(ProcessorStep):
    """Legacy compatibility step for old processor configs.

    Older saved processor pipelines may reference `to_batch_processor` as a final
    structural step. In the current pipeline design, batch/transition conversion is
    handled by the pipeline's `to_transition` / `to_output` callables rather than by
    an in-pipeline processor step, so this step is intentionally a no-op.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        self._current_transition = transition.copy()
        return self._current_transition

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
