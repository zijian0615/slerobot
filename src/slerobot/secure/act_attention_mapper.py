"""Backward-compatible wrapper around ACTPolicy attention visualization."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from slerobot.policies.act.modeling_act import ACTPolicy, create_act_attention_helper


class ACTPolicyWithAttention:
    """Deprecated wrapper. Prefer `ACTPolicy` with `enable_attention_visualization=True`."""

    def __init__(
        self,
        policy: ACTPolicy,
        image_shapes: list[tuple[int, int]] | None = None,
        specific_decoder_token_index: int | None = None,
    ):
        if not isinstance(policy, ACTPolicy):
            raise TypeError("ACTPolicyWithAttention expects an ACTPolicy instance.")

        policy.config.enable_attention_visualization = True
        policy.config.grad_cam_target_action_index = (
            specific_decoder_token_index
            if specific_decoder_token_index is not None
            else policy.config.grad_cam_target_action_index
        )
        policy._attention_helper = create_act_attention_helper(policy)
        self.policy = policy
        self.config = policy.config
        self.specific_decoder_token_index = specific_decoder_token_index
        self.last_observation = None
        self.last_attention_maps = None

    @property
    def num_images(self) -> int:
        return len(self.config.image_features) if self.config.image_features else 0

    def select_action(
        self, observation: Dict[str, torch.Tensor], **kwargs
    ) -> Tuple[torch.Tensor, List[np.ndarray | None]]:
        action = self.policy.select_action(observation, **kwargs)
        attention_maps = None
        if self.policy.last_attention_maps is not None:
            attention_maps = list(self.policy.last_attention_maps.values())
            self.last_attention_maps = attention_maps
        else:
            attention_maps = [None] * self.num_images
        self.last_observation = observation.copy()
        return action, attention_maps

    def visualize_attention(self, **kwargs) -> List[np.ndarray | None]:
        return self.policy.visualize_attention(**kwargs)

    def __getattr__(self, name):
        if name not in self.__dict__:
            return getattr(self.policy, name)
        return self.__dict__[name]
