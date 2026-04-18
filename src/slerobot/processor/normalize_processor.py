from __future__ import annotations

from typing import Any

import torch

from slerobot.configs.types import PipelineFeatureType, PolicyFeature

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry


class _BaseNormalizationProcessorStep(ProcessorStep):
    def __init__(self, **kwargs):
        self.config = dict(kwargs)
        self.stats: dict[str, torch.Tensor] = {}
        self.eps = float(kwargs.get("eps", 1e-6))
        self.target = kwargs.get("target") or kwargs.get("transition_key") or kwargs.get("field")

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.stats.copy()

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.stats = dict(state)

    def _extract_stats(self, prefix: str | None = None) -> dict[str, torch.Tensor]:
        if prefix:
            scoped = {
                name.split(".")[-1]: value
                for name, value in self.stats.items()
                if name.startswith(f"{prefix}.")
            }
            if scoped:
                return scoped
        return {
            key: value
            for key, value in self.stats.items()
            if key in {"mean", "std", "min", "max", "q01", "q99"}
        }

    def _apply_norm(self, value: Any, stats: dict[str, torch.Tensor]) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        if "mean" in stats and "std" in stats:
            return self._mean_std(value, stats["mean"], stats["std"])
        if "min" in stats and "max" in stats:
            return self._min_max(value, stats["min"], stats["max"])
        return value

    def _process_mapping(self, mapping: dict[str, Any]) -> dict[str, Any]:
        processed: dict[str, Any] = {}
        for key, value in mapping.items():
            processed[key] = self._apply_norm(value, self._extract_stats(key))
        return processed

    def _process_transition_value(self, value: Any, prefix: str | None = None) -> Any:
        if isinstance(value, dict):
            return self._process_mapping(value)

        stats = self._extract_stats(prefix)
        if not stats:
            stats = self._extract_stats()
        return self._apply_norm(value, stats)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        self._current_transition = transition.copy()
        new_transition = self._current_transition.copy()

        targets: list[TransitionKey]
        if self.target == TransitionKey.OBSERVATION.value:
            targets = [TransitionKey.OBSERVATION]
        elif self.target == TransitionKey.ACTION.value:
            targets = [TransitionKey.ACTION]
        else:
            targets = [TransitionKey.OBSERVATION, TransitionKey.ACTION]

        for target in targets:
            value = new_transition.get(target)
            if value is None:
                continue
            new_transition[target] = self._process_transition_value(value, target.value)

        return new_transition

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def _mean_std(self, value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _min_max(self, value: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@ProcessorStepRegistry.register("normalizer_processor")
class NormalizerProcessorStep(_BaseNormalizationProcessorStep):
    def _mean_std(self, value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        mean = mean.to(device=value.device, dtype=value.dtype)
        std = std.to(device=value.device, dtype=value.dtype)
        return (value - mean) / torch.clamp(std, min=self.eps)

    def _min_max(self, value: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
        min_value = min_value.to(device=value.device, dtype=value.dtype)
        max_value = max_value.to(device=value.device, dtype=value.dtype)
        return (value - min_value) / torch.clamp(max_value - min_value, min=self.eps)


@ProcessorStepRegistry.register("unnormalizer_processor")
class UnnormalizerProcessorStep(_BaseNormalizationProcessorStep):
    def _mean_std(self, value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        mean = mean.to(device=value.device, dtype=value.dtype)
        std = std.to(device=value.device, dtype=value.dtype)
        return value * torch.clamp(std, min=self.eps) + mean

    def _min_max(self, value: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
        min_value = min_value.to(device=value.device, dtype=value.dtype)
        max_value = max_value.to(device=value.device, dtype=value.dtype)
        return value * torch.clamp(max_value - min_value, min=self.eps) + min_value
