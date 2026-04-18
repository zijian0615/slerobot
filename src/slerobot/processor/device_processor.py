from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import logging

from slerobot.configs.types import PipelineFeatureType, PolicyFeature
from slerobot.utils.utils import auto_select_torch_device, get_safe_torch_device, is_torch_device_available

from .core import EnvTransition
from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("device_processor")
@dataclass
class DeviceProcessorStep(ProcessorStep):
    """Move transition payloads containing tensors onto a target device.

    This provides compatibility for older saved processor pipelines that included a
    dedicated device step.
    """

    device: str | None = None
    float_dtype: str | None = None

    def __post_init__(self) -> None:
        if self.device is not None:
            requested_device = str(self.device)
            if requested_device != "cpu" and not is_torch_device_available(requested_device):
                fallback_device = str(auto_select_torch_device())
                logging.warning(
                    "Requested processor device '%s' is unavailable, falling back to %s.",
                    requested_device,
                    fallback_device,
                )
                self.device = fallback_device
            else:
                self.device = str(get_safe_torch_device(requested_device))

    def _move(self, value: Any) -> Any:
        if self.device is None:
            return value
        if isinstance(value, torch.Tensor):
            kwargs: dict[str, Any] = {"device": self.device}
            if self.float_dtype is not None and value.is_floating_point():
                kwargs["dtype"] = getattr(torch, self.float_dtype)
            return value.to(**kwargs)
        if isinstance(value, dict):
            return {key: self._move(sub_value) for key, sub_value in value.items()}
        if isinstance(value, list):
            return [self._move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move(item) for item in value)
        return value

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        self._current_transition = transition.copy()
        return {key: self._move(value) for key, value in self._current_transition.items()}

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
