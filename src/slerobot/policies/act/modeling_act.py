#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import logging
import math
import time
from collections import deque
from collections.abc import Callable
from itertools import chain
import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from slerobot.policies.act.configuration_act import ACTConfig, resolve_warning_camera_keys
from slerobot.policies.pretrained import PreTrainedPolicy
from slerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

ATTENTION_CAM_METHODS = frozenset({"eigen_cam", "grad_cam_pp"})
ATTENTION_ROI_MODES = frozenset({"detect", "mitigation"})


def attention_roi_mode_banner_label(mode: str) -> str:
    if mode == "detect":
        return "WARNING! Anomaly Detected"
    return "WARNING! Automatic Mitigation"


def attention_roi_mode_speech_text(mode: str) -> str:
    if mode == "detect":
        return "Warning, anomaly detected in region of interest"
    return "Warning, automatic mitigation adopted"


def attention_cam_method_display_name(method: str) -> str:
    return {"eigen_cam": "Eigen-CAM", "grad_cam_pp": "Grad-CAM++"}.get(method, method)


def create_act_attention_helper(policy: "ACTPolicy"):
    """Instantiate the configured attention/CAM helper for an ACT policy."""
    method = policy.config.attention_cam_method
    if method == "eigen_cam":
        return ACTEigenCAMHelper(policy)
    if method == "grad_cam_pp":
        return ACTGradCAMPlusPlusHelper(policy)
    raise ValueError(
        f"Unknown attention_cam_method={method!r}. Choose from {sorted(ATTENTION_CAM_METHODS)}."
    )


class ACTEigenCAMHelper:
    """Eigen-CAM on ResNet backbone feature maps, one heatmap per camera (gradient-free attention visualization)."""

    CAM_METHOD_LABEL = "EIGEN-CAM"

    def __init__(self, policy: "ACTPolicy"):
        self.policy = policy
        self.config = policy.config
        self.last_observation: dict[str, Tensor] | None = None
        self.last_attention_maps: dict[str, np.ndarray] | None = None
        self.last_edge_warnings: dict[str, bool] = {}
        self.last_edge_warning_stats: dict[str, float] = {}
        self.warning_image_keys = resolve_warning_camera_keys(self.config)
        self._warning_active_since: float | None = None
        self._warning_speech_announced: bool = False
        self._last_cam_compute_time: float | None = None

        if not hasattr(self.policy.model, "backbone"):
            raise AttributeError(
                "Eigen-CAM requires a vision backbone. Enable image inputs in ACTConfig."
            )

        target_action_index = self.config.grad_cam_target_action_index
        if not (0 <= target_action_index < self.config.chunk_size):
            raise ValueError(
                f"grad_cam_target_action_index ({target_action_index}) must be in "
                f"[0, {self.config.chunk_size - 1}]."
            )

    def _activation_for_image_key(
        self, image_key: str, backbone_activations: list[Tensor]
    ) -> Tensor | None:
        all_keys = list(self.config.image_features.keys())
        if image_key not in all_keys:
            return None
        activation_index = all_keys.index(image_key)
        if activation_index >= len(backbone_activations):
            return None
        return backbone_activations[activation_index]

    def predict_action_chunk_with_attention(self, batch: dict[str, Tensor]) -> Tensor:
        """Run inference and compute per-camera Eigen-CAM maps (stored in last_attention_maps)."""
        self.last_observation = dict(batch)
        backbone_activations: list[Tensor] = []

        def backbone_hook(_module, _inputs, output) -> None:
            feature_map = output["feature_map"]
            backbone_activations.append(feature_map)

        handle = self.policy.model.backbone.register_forward_hook(backbone_hook)

        try:
            with torch.no_grad():
                actions = self.policy.model(batch)[0]
        finally:
            handle.remove()

        attention_maps: dict[str, np.ndarray] = {}
        self.last_edge_warnings = {}
        self.last_edge_warning_stats = {}

        for image_key in self.config.image_features:
            activation = self._activation_for_image_key(image_key, backbone_activations)
            if activation is None:
                continue

            cam_map = self._compute_eigen_cam(activation)
            if cam_map is None:
                continue

            attention_maps[image_key] = cam_map
            if image_key not in self.warning_image_keys:
                self.last_edge_warnings[image_key] = False
                continue

            image_hw = self._image_hw_from_batch(batch, image_key)
            if image_hw is None:
                continue

            self._record_cam_warning(image_key, cam_map, image_hw)

        self.last_attention_maps = attention_maps or None
        if self.last_attention_maps is None:
            logging.warning("%s did not produce any heatmaps.", self.CAM_METHOD_LABEL)

        self._last_cam_compute_time = time.perf_counter()
        return actions.detach()

    def should_refresh_attention(self) -> bool:
        interval_s = self.config.attention_cam_interval_s
        if interval_s <= 0:
            return True
        if self._last_cam_compute_time is None:
            return True
        return (time.perf_counter() - self._last_cam_compute_time) >= interval_s

    def update_last_observation(self, batch: dict[str, Tensor]) -> None:
        self.last_observation = dict(batch)

    def _prepare_batch_for_grad_cam(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Enable gradients on image inputs so Grad-CAM works with a frozen backbone."""
        batch = dict(batch)
        for key in self.config.image_features:
            if key not in batch:
                continue
            image = batch[key]
            if not image.requires_grad:
                batch[key] = image.detach().requires_grad_(True)
        if self.config.image_features:
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch

    def predict_action_chunk_without_attention(self, batch: dict[str, Tensor]) -> Tensor:
        """Forward-only action chunk; reuse cached heatmaps between CAM refreshes."""
        self.update_last_observation(batch)
        with torch.no_grad():
            return self.policy.model(batch)[0]

    def _record_cam_warning(self, image_key: str, heatmap: np.ndarray, image_hw: tuple[int, int]) -> None:
        warning, red_fraction = self.check_top_right_red_activation(
            heatmap,
            image_hw=image_hw,
            roi_size_px=self.config.grad_cam_edge_margin_px,
            red_fraction_threshold=self.config.grad_cam_edge_mean_threshold,
            high_activation_threshold=self.config.cam_warning_high_activation_threshold,
        )
        self.last_edge_warnings[image_key] = warning
        self.last_edge_warning_stats[image_key] = red_fraction
        if warning:
            logging.warning(
                "%s top-right ROI red-area high for '%s': red_frac=%.3f (threshold=%.3f, "
                "activation_thr=%.2f, roi=%dpx)",
                self.CAM_METHOD_LABEL,
                image_key,
                red_fraction,
                self.config.grad_cam_edge_mean_threshold,
                self.config.cam_warning_high_activation_threshold,
                self.config.grad_cam_edge_margin_px,
            )

    @property
    def image_keys_needing_roi_mask(self) -> set[str]:
        """Image keys to whiten when mitigation mode and warning was active on the previous step."""
        if self.config.attention_roi_mode != "mitigation":
            return set()
        return {key for key, active in self.last_edge_warnings.items() if active}

    def edge_warning_banner_label(self) -> str:
        return attention_roi_mode_banner_label(self.config.attention_roi_mode)

    def edge_warning_speech_text(self) -> str:
        return attention_roi_mode_speech_text(self.config.attention_roi_mode)

    def clear_warning_masks(self) -> None:
        self.last_edge_warnings.clear()
        self.last_edge_warning_stats.clear()
        self._warning_active_since = None
        self._warning_speech_announced = False
        self._last_cam_compute_time = None

    def _any_warning_active(self) -> bool:
        return any(self.last_edge_warnings.get(key, False) for key in self.warning_image_keys)

    def announce_new_edge_warnings(
        self,
        play_sounds: bool = True,
        voice: str | None = None,
        rate: int | None = None,
    ) -> None:
        """Speak once after ROI warning stays active for `warning_speech_min_duration_s`."""
        any_active = self._any_warning_active()
        now = time.perf_counter()
        min_duration_s = self.config.warning_speech_min_duration_s

        if not any_active:
            self._warning_active_since = None
            self._warning_speech_announced = False
            return

        if self._warning_active_since is None:
            self._warning_active_since = now

        if self._warning_speech_announced:
            return

        if now - self._warning_active_since < min_duration_s:
            return

        self._warning_speech_announced = True
        if not play_sounds or not self.config.warning_speech_enabled:
            return

        from slerobot.utils.utils import log_say

        speech_voice = voice if voice is not None else self.config.warning_speech_voice
        speech_rate = rate if rate is not None else self.config.warning_speech_rate
        log_say(
            self.edge_warning_speech_text(),
            play_sounds=True,
            blocking=False,
            voice=speech_voice,
            rate=speech_rate,
        )

    @staticmethod
    def camera_short_name(image_key: str) -> str:
        return image_key.split(".")[-1]

    @staticmethod
    def apply_warning_roi_mask_to_observation(
        observation: dict[str, np.ndarray],
        image_keys: set[str],
        roi_size_px: int,
    ) -> None:
        """Set the top-right ROI to white on cameras with an active warning (in-place)."""
        if not image_keys:
            return

        masked_short_names = {ACTEigenCAMHelper.camera_short_name(k) for k in image_keys}
        for obs_key, image in observation.items():
            if "image" not in obs_key:
                continue
            short_name = ACTEigenCAMHelper.camera_short_name(obs_key)
            if obs_key not in image_keys and short_name not in masked_short_names:
                continue
            if image.ndim != 3:
                continue

            image_h, image_w = int(image.shape[0]), int(image.shape[1])
            roi_h = min(roi_size_px, image_h)
            roi_w = min(roi_size_px, image_w)
            if roi_h <= 0 or roi_w <= 0:
                continue

            white_value: float | int = 255 if image.dtype == np.uint8 else 1.0
            observation[obs_key][:roi_h, image_w - roi_w : image_w] = white_value

    def _image_hw_from_batch(self, batch: dict[str, Tensor], image_key: str) -> tuple[int, int] | None:
        if image_key not in batch:
            return None
        img = batch[image_key]
        if img.dim() == 4:
            img = img[0]
        if img.dim() != 3:
            return None
        return int(img.shape[1]), int(img.shape[2])

    @staticmethod
    def check_top_right_red_activation(
        heatmap: np.ndarray,
        image_hw: tuple[int, int],
        roi_size_px: int,
        red_fraction_threshold: float,
        high_activation_threshold: float,
    ) -> tuple[bool, float]:
        """Return (warning_triggered, red_pixel_fraction) in the top-right ROI on full-resolution image."""
        image_h, image_w = image_hw
        roi_h = min(roi_size_px, image_h)
        roi_w = min(roi_size_px, image_w)
        if roi_h <= 0 or roi_w <= 0:
            return False, 0.0

        heatmap_resized = cv2.resize(heatmap, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
        roi = heatmap_resized[:roi_h, image_w - roi_w :]
        red_fraction = float((roi >= high_activation_threshold).mean())
        return red_fraction >= red_fraction_threshold, red_fraction

    @staticmethod
    def check_edge_activation(
        heatmap: np.ndarray,
        image_hw: tuple[int, int],
        margin_px: int,
        mean_threshold: float,
    ) -> tuple[bool, float]:
        """Backward-compatible alias for top-right ROI red-area check."""
        return ACTEigenCAMHelper.check_top_right_red_activation(
            heatmap,
            image_hw=image_hw,
            roi_size_px=margin_px,
            red_fraction_threshold=mean_threshold,
            high_activation_threshold=0.65,
        )

    @staticmethod
    def _build_top_right_roi_mask(image_hw: tuple[int, int], roi_size_px: int) -> np.ndarray:
        image_h, image_w = image_hw
        roi_h = min(roi_size_px, image_h)
        roi_w = min(roi_size_px, image_w)
        roi_mask = np.zeros((image_h, image_w), dtype=bool)
        if roi_h <= 0 or roi_w <= 0:
            return roi_mask
        roi_mask[:roi_h, image_w - roi_w :] = True
        return roi_mask

    @staticmethod
    def _build_edge_region_mask(image_hw: tuple[int, int], margin_px: int) -> np.ndarray:
        """Backward-compatible alias for the top-right ROI mask."""
        return ACTEigenCAMHelper._build_top_right_roi_mask(image_hw, margin_px)

    @staticmethod
    def _compute_eigen_cam(activation: Tensor) -> np.ndarray | None:
        """Compute Eigen-CAM from feature map using principal component analysis."""
        # activation shape: (batch_size, channels, height, width)
        b, c, h, w = activation.shape
        
        if c < 1 or h < 1 or w < 1:
            return None
        
        # Reshape to (batch_size, channels, h*w)
        x = activation.reshape(b, c, -1).detach().cpu()
        
        # For batch processing, take first sample
        x = x[0]  # (channels, h*w)
        
        # Compute mean for centering
        x_mean = x.mean(dim=1, keepdim=True)
        x_centered = x - x_mean
        
        # Compute covariance matrix (channels x channels)
        cov = torch.mm(x_centered, x_centered.t()) / max(1, x_centered.shape[1] - 1)
        
        # Compute eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            # Use eigenvector corresponding to largest eigenvalue
            principal_component = eigenvectors[:, -1]
        except Exception:
            return None
        
        # Project feature map onto principal component
        cam = torch.mm(principal_component.unsqueeze(0), x)  # (1, h*w)
        cam = cam.reshape(h, w).numpy()
        
        return ACTEigenCAMHelper._normalize_heatmap(cam)

    @staticmethod
    def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
        heatmap = np.maximum(heatmap, 0)
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        if heatmap_max > heatmap_min:
            return ((heatmap - heatmap_min) / (heatmap_max - heatmap_min)).astype(np.float32)
        return np.zeros_like(heatmap, dtype=np.float32)

    def _extract_images(self, observation: dict[str, Tensor]) -> list[Tensor]:
        images = []
        for key in self.config.image_features:
            if key in observation:
                images.append(observation[key])
        return images

    @classmethod
    def overlay_attention_on_image(
        cls,
        image: np.ndarray,
        attention_map: np.ndarray,
        overlay_alpha: float = 0.5,
        use_rgb: bool = True,
        edge_warning: bool = False,
        edge_margin_px: int = 50,
        edge_warning_banner_px: int = 48,
        edge_mask_alpha: float = 0.55,
        edge_warning_mean: float | None = None,
        edge_warning_threshold: float | None = None,
        edge_warning_roi_mode: str = "mitigation",
    ) -> np.ndarray:
        img_np = image
        if img_np.ndim == 3 and img_np.shape[0] in (3, 4):
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.dtype != np.float32 and img_np.dtype != np.float64:
            img_np = img_np.astype(np.float32)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0

        h, w = img_np.shape[:2]
        attn_map_resized = cv2.resize(attention_map, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        vis = cv2.addWeighted(
            np.uint8(255 * img_np),
            1 - overlay_alpha,
            heatmap,
            overlay_alpha,
            0,
        )

        if edge_warning:
            vis = cls._apply_edge_warning_overlay(
                vis,
                image_hw=(h, w),
                margin_px=edge_margin_px,
                banner_px=edge_warning_banner_px,
                mask_alpha=edge_mask_alpha,
                edge_mean=edge_warning_mean,
                edge_threshold=edge_warning_threshold,
                edge_warning_roi_mode=edge_warning_roi_mode,
            )

        return vis

    @classmethod
    def _format_edge_warning_label(
        cls,
        roi_mode: str,
        red_fraction: float | None,
        fraction_threshold: float | None,
        roi_size_px: int,
    ) -> str:
        banner = attention_roi_mode_banner_label(roi_mode)
        if red_fraction is not None and fraction_threshold is not None:
            return (
                f"{banner}  red_frac={red_fraction:.3f}  "
                f"thr={fraction_threshold:.3f}  roi={roi_size_px}px"
            )
        return banner

    @classmethod
    def _apply_edge_warning_overlay(
        cls,
        vis: np.ndarray,
        image_hw: tuple[int, int],
        margin_px: int,
        banner_px: int,
        mask_alpha: float,
        edge_mean: float | None = None,
        edge_threshold: float | None = None,
        edge_warning_roi_mode: str = "mitigation",
    ) -> np.ndarray:
        image_h, image_w = image_hw
        output = vis.copy()

        label = cls._format_edge_warning_label(
            edge_warning_roi_mode, edge_mean, edge_threshold, margin_px
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (_text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        banner_px = max(min(banner_px, image_h), text_h + baseline + 12)

        banner = output.copy()
        cv2.rectangle(banner, (0, 0), (image_w, banner_px), (180, 0, 0), thickness=-1)
        output = cv2.addWeighted(output, 0.35, banner, 0.65, 0)
        cv2.putText(
            output,
            label,
            (12, text_h + 8),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        return output

    def visualize_attention(
        self,
        images: list[Tensor] | None = None,
        attention_maps: list[np.ndarray | None] | dict[str, np.ndarray] | None = None,
        observation: dict[str, Tensor] | None = None,
        use_rgb: bool = True,
        overlay_alpha: float = 0.5,
    ) -> list[np.ndarray | None]:
        if images is None:
            if observation is not None:
                images = self._extract_images(observation)
            elif self.last_observation is not None:
                images = self._extract_images(self.last_observation)
            else:
                raise ValueError("No images provided and no stored observation available")

        if attention_maps is None:
            if self.last_attention_maps is not None:
                attention_maps = [
                    self.last_attention_maps.get(key) for key in self.config.image_features
                ]
            else:
                raise ValueError("No attention maps provided and no stored attention maps available")
        elif isinstance(attention_maps, dict):
            attention_maps = [attention_maps.get(key) for key in self.config.image_features]

        image_keys = list(self.config.image_features.keys())
        visualizations: list[np.ndarray | None] = []
        for image_key, img, attn_map in zip(image_keys, images, attention_maps):
            if img is None or attn_map is None:
                visualizations.append(None)
                continue

            if isinstance(img, Tensor):
                if img.dim() == 4:
                    img = img.squeeze(0)
                img_np = img.permute(1, 2, 0).detach().cpu().numpy()
            else:
                img_np = img

            edge_warning = self.last_edge_warnings.get(image_key, False)
            visualizations.append(
                self.overlay_attention_on_image(
                    img_np,
                    attn_map,
                    overlay_alpha=overlay_alpha,
                    use_rgb=use_rgb,
                    edge_warning=edge_warning,
                    edge_margin_px=self.config.grad_cam_edge_margin_px,
                    edge_warning_mean=self.last_edge_warning_stats.get(image_key),
                    edge_warning_threshold=self.config.grad_cam_edge_mean_threshold,
                    edge_warning_roi_mode=self.config.attention_roi_mode,
                )
            )

        return visualizations


class ACTGradientCAMHelperBase(ACTEigenCAMHelper):
    """Shared forward/backward hook path for gradient-based CAM methods."""

    def __init__(self, policy: "ACTPolicy"):
        super().__init__(policy)

    def _compute_grad_cam_for_image(self, image: Tensor) -> np.ndarray | None:
        """Run Grad-CAM on the vision backbone only (matches ACT inference; actions use a separate forward)."""
        if not image.requires_grad:
            image = image.detach().requires_grad_(True)

        activation_holder: list[Tensor] = []

        def backbone_hook(_module, _inputs, output) -> None:
            feature_map = output["feature_map"]
            if not feature_map.requires_grad:
                logging.warning(
                    "%s: backbone feature_map has requires_grad=False; skipping retain_grad.",
                    self.CAM_METHOD_LABEL,
                )
                return
            feature_map.retain_grad()
            activation_holder.append(feature_map)

        handle = self.policy.model.backbone.register_forward_hook(backbone_hook)
        self.policy.model.zero_grad(set_to_none=True)

        try:
            with torch.enable_grad():
                self.policy.model.backbone(image)
                if not activation_holder:
                    return None
                activation_holder[0].sum().backward()
        finally:
            handle.remove()

        if not activation_holder:
            return None
        return self._compute_cam(activation_holder[0])

    def predict_action_chunk_with_attention(self, batch: dict[str, Tensor]) -> Tensor:
        batch = self._prepare_batch_for_grad_cam(batch)
        self.last_observation = dict(batch)

        with torch.no_grad():
            actions = self.policy.model(batch)[0]

        attention_maps: dict[str, np.ndarray] = {}
        self.last_edge_warnings = {}
        self.last_edge_warning_stats = {}

        for image_key in self.config.image_features:
            if image_key not in batch:
                continue

            cam_map = self._compute_grad_cam_for_image(batch[image_key])
            if cam_map is None:
                continue

            attention_maps[image_key] = cam_map
            if image_key not in self.warning_image_keys:
                self.last_edge_warnings[image_key] = False
                continue

            image_hw = self._image_hw_from_batch(batch, image_key)
            if image_hw is None:
                continue

            self._record_cam_warning(image_key, cam_map, image_hw)

        self.last_attention_maps = attention_maps or None
        if self.last_attention_maps is None:
            logging.warning("%s did not produce any heatmaps.", self.CAM_METHOD_LABEL)

        self._last_cam_compute_time = time.perf_counter()
        return actions.detach()

    @staticmethod
    def _compute_cam(activation: Tensor) -> np.ndarray | None:
        raise NotImplementedError


class ACTGradCAMHelper(ACTGradientCAMHelperBase):
    """Grad-CAM on ResNet backbone feature maps, one heatmap per camera."""

    CAM_METHOD_LABEL = "GRAD-CAM"

    @staticmethod
    def _compute_cam(activation: Tensor) -> np.ndarray | None:
        gradients = activation.grad
        if gradients is None:
            return None

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1))
        cam_np = cam[0].detach().float().cpu().numpy()
        return ACTEigenCAMHelper._normalize_heatmap(cam_np)


class ACTGradCAMPlusPlusHelper(ACTGradientCAMHelperBase):
    """Grad-CAM++ on ResNet backbone feature maps, one heatmap per camera."""

    CAM_METHOD_LABEL = "GRAD-CAM++"

    @staticmethod
    def _compute_cam(activation: Tensor) -> np.ndarray | None:
        gradients = activation.grad
        if gradients is None:
            return None

        grads_power_2 = gradients.pow(2)
        grads_power_3 = gradients.pow(3)
        sum_activations = activation.sum(dim=(2, 3), keepdim=True)
        eps = 1e-8
        alpha_denom = 2.0 * grads_power_2 + sum_activations * grads_power_3 + eps
        alpha = grads_power_2 / alpha_denom

        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1))
        cam_np = cam[0].detach().float().cpu().numpy()
        return ACTEigenCAMHelper._normalize_heatmap(cam_np)


# Backward-compatible alias (default CAM is Eigen-CAM; use create_act_attention_helper in policy code).
ACTAttentionHelper = ACTEigenCAMHelper


class ACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = ACTConfig
    name = "act"

    def __init__(
        self,
        config: ACTConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self._attention_helper: ACTEigenCAMHelper | ACTGradCAMPlusPlusHelper | None = None
        if config.enable_attention_visualization and config.image_features:
            self._attention_helper = create_act_attention_helper(self)
            cam_name = attention_cam_method_display_name(config.attention_cam_method)
            all_cameras = [k.split(".")[-1] for k in config.image_features]
            warning_cameras = [k.split(".")[-1] for k in resolve_warning_camera_keys(config)]
            roi_mode = config.attention_roi_mode
            cam_interval = config.attention_cam_interval_s
            if cam_interval > 0:
                cam_refresh = f"every {cam_interval:g}s (wall clock)"
            else:
                cam_refresh = "every control step"
            if config.n_action_steps > 1:
                logging.info(
                    "%s will refresh %s (n_action_steps=%d); "
                    "CAM on %s, ROI %s on %s; "
                    "robot actions are still consumed from the chunk queue.",
                    cam_name,
                    cam_refresh,
                    config.n_action_steps,
                    all_cameras,
                    roi_mode,
                    warning_cameras,
                )
            else:
                logging.info(
                    "ACT attention visualization: %s on %s, ROI %s on %s; refresh %s.",
                    cam_name,
                    all_cameras,
                    roi_mode,
                    warning_cameras,
                    cam_refresh,
                )

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self._attention_helper is not None:
            self._attention_helper.clear_warning_masks()
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.

        CAM refresh is throttled by `attention_cam_interval_s` (wall clock). Between refreshes, actions use
        forward-only inference and the last heatmap is reused. Executed actions continue to be consumed from
        the action queue when `n_action_steps > 1`.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_chunk: Tensor | None = None
        if self._attention_helper is not None:
            if self._attention_helper.should_refresh_attention():
                actions_chunk = self._attention_helper.predict_action_chunk_with_attention(batch)
            else:
                self._attention_helper.update_last_observation(batch)

        if self.config.temporal_ensemble_coeff is not None:
            if actions_chunk is not None:
                actions = actions_chunk
            elif self._attention_helper is not None:
                actions = self._attention_helper.predict_action_chunk_without_attention(batch)
            else:
                actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            if actions_chunk is not None:
                actions = actions_chunk[:, : self.config.n_action_steps]
            elif self._attention_helper is not None:
                actions = self._attention_helper.predict_action_chunk_without_attention(batch)[
                    :, : self.config.n_action_steps
                ]
            else:
                with torch.no_grad():
                    actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        with torch.no_grad():
            return self._action_queue.popleft()

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        if self._attention_helper is not None:
            return self._attention_helper.predict_action_chunk_with_attention(batch)

        with torch.no_grad():
            actions = self.model(batch)[0]
        return actions

    @property
    def last_attention_maps(self) -> dict[str, np.ndarray] | None:
        if self._attention_helper is None:
            return None
        return self._attention_helper.last_attention_maps

    @property
    def last_grad_cam_edge_warnings(self) -> dict[str, bool]:
        if self._attention_helper is None:
            return {}
        return self._attention_helper.last_edge_warnings

    @property
    def last_grad_cam_edge_stats(self) -> dict[str, float]:
        if self._attention_helper is None:
            return {}
        return self._attention_helper.last_edge_warning_stats

    def announce_attention_warnings(
        self,
        play_sounds: bool = True,
        voice: str | None = None,
        rate: int | None = None,
    ) -> None:
        if self._attention_helper is not None:
            self._attention_helper.announce_new_edge_warnings(
                play_sounds=play_sounds,
                voice=voice,
                rate=rate,
            )

    @property
    def attention_overlay_helper(self) -> type[ACTEigenCAMHelper]:
        """Helper class used for overlaying heatmaps (matches the configured CAM method)."""
        if self._attention_helper is None:
            return ACTEigenCAMHelper
        return type(self._attention_helper)

    def visualize_attention(
        self,
        images: list[Tensor] | None = None,
        attention_maps: list[np.ndarray | None] | dict[str, np.ndarray] | None = None,
        observation: dict[str, Tensor] | None = None,
        use_rgb: bool = True,
        overlay_alpha: float = 0.5,
    ) -> list[np.ndarray | None]:
        if self._attention_helper is None:
            raise RuntimeError(
                "Attention visualization is disabled. Set `enable_attention_visualization=True` in ACTConfig."
            )
        return self._attention_helper.visualize_attention(
            images=images,
            attention_maps=attention_maps,
            observation=observation,
            use_rgb=use_rgb,
            overlay_alpha=overlay_alpha,
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: ACTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # Extend immediately instead of accumulating and concatenating
                # Convert to list to extend properly
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            encoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            decoder_pos_embed: (DS, 1, C) positional embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
