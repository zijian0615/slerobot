
from __future__ import annotations

import importlib
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeAlias, TypedDict, TypeVar, cast

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from slerobot.configs.types import PipelineFeatureType, PolicyFeature
from slerobot.utils.hub import HubMixin

from .converters import batch_to_transition, create_transition, transition_to_batch
from .core import EnvAction, EnvTransition, PolicyAction, RobotAction, RobotObservation, TransitionKey


class ProcessorStepRegistry:
    """A registry for ProcessorStep classes to allow instantiation from a string name.

    This class provides a way to map string identifiers to `ProcessorStep` classes,
    which is useful for deserializing pipelines from configuration files without

    hardcoding class imports.
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str | None = None):
        """A class decorator to register a ProcessorStep.

        Args:
            name: The name to register the class under. If None, the class's `__name__` is used.

        Returns:
            A decorator function that registers the class and returns it.

        Raises:
            ValueError: If a step with the same name is already registered.
        """

        def decorator(step_class: type) -> type:
            """The actual decorator that performs the registration."""
            registration_name = name if name is not None else step_class.__name__

            if registration_name in cls._registry:
                raise ValueError(
                    f"Processor step '{registration_name}' is already registered. "
                    f"Use a different name or unregister the existing one first."
                )

            cls._registry[registration_name] = step_class
            # Store the registration name on the class for easy lookup during serialization.
            step_class._registry_name = registration_name
            return step_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Retrieves a processor step class from the registry by its name.

        Args:
            name: The name of the step to retrieve.

        Returns:
            The processor step class corresponding to the given name.

        Raises:
            KeyError: If the name is not found in the registry.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Processor step '{name}' not found in registry. "
                f"Available steps: {available}. "
                f"Make sure the step is registered using @ProcessorStepRegistry.register()"
            )
        return cls._registry[name]

    @classmethod
    def unregister(cls, name: str) -> None:
        """Removes a processor step from the registry.

        Args:
            name: The name of the step to unregister.
        """
        cls._registry.pop(name, None)

    @classmethod
    def list(cls) -> list[str]:
        """Returns a list of all registered processor step names."""
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clears all processor steps from the registry."""
        cls._registry.clear()

class ProcessorStep(ABC):
    """Abstract base class for a single step in a data processing pipeline.

    Each step must implement the `__call__` method to perform its transformation
    on a data transition and the `transform_features` method to describe how it
    alters the shape or type of data features.

    Subclasses can optionally be stateful by implementing `state_dict` and `load_state_dict`.
    """

    _current_transition: EnvTransition | None = None

    @property
    def transition(self) -> EnvTransition:
        """Provides access to the most recent transition being processed.

        This is useful for steps that need to access other parts of the transition
        data beyond their primary target (e.g., an action processing step that
        needs to look at the observation).

        Raises:
            ValueError: If accessed before the step has been called with a transition.
        """
        if self._current_transition is None:
            raise ValueError("Transition is not set. Make sure to call the step with a transition first.")
        return self._current_transition

    @abstractmethod
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Processes an environment transition.

        This method should contain the core logic of the processing step.

        Args:
            transition: The input data transition to be processed.

        Returns:
            The processed transition.
        """
        return transition

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the step for serialization.

        Returns:
            A JSON-serializable dictionary of configuration parameters.
        """
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Returns the state of the step (e.g., learned parameters, running means).

        Returns:
            A dictionary mapping state names to tensors.
        """
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Loads the step's state from a state dictionary.

        Args:
            state: A dictionary of state tensors.
        """
        return None

    def reset(self) -> None:
        """Resets the internal state of the processor step, if any."""
        return None

    @abstractmethod
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Defines how this step modifies the description of pipeline features.

        This method is used to track changes in data shapes, dtypes, or modalities
        as data flows through the pipeline, without needing to process actual data.

        Args:
            features: A dictionary describing the input features for observations, actions, etc.

        Returns:
            A dictionary describing the output features after this step's transformation.
        """
        return features
    
class RobotActionProcessorStep(ProcessorStep, ABC):
    """An abstract `ProcessorStep` for processing a `RobotAction` (a dictionary)."""

    @abstractmethod
    def action(self, action: RobotAction) -> RobotAction:
        """Processes a `RobotAction`. Subclasses must implement this method.

        Args:
            action: The input `RobotAction` dictionary.

        Returns:
            The processed `RobotAction`.
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Applies the `action` method to the transition's action, ensuring it's a `RobotAction`."""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is None or not isinstance(action, dict):
            raise ValueError(f"Action should be a RobotAction type (dict), but got {type(action)}")

        processed_action = self.action(action.copy())
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition


class PolicyActionProcessorStep(ProcessorStep, ABC):
    """An abstract `ProcessorStep` for processing a `PolicyAction` (a tensor or dict of tensors)."""

    @abstractmethod
    def action(self, action: PolicyAction) -> PolicyAction:
        """Processes a `PolicyAction`. Subclasses must implement this method.

        Args:
            action: The input `PolicyAction`.

        Returns:
            The processed `PolicyAction`.
        """
        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Applies the `action` method to the transition's action, ensuring it's a `PolicyAction`."""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type (tensor), but got {type(action)}")

        processed_action = self.action(action)
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition


class ActionProcessorStep(PolicyActionProcessorStep, ABC):
    """Backward-compatible alias for `PolicyActionProcessorStep`."""

    pass