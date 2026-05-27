# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
import os
from typing import Any

import numpy as np
import rerun as rr

from slerobot.policies.act.modeling_act import ACTEigenCAMHelper

logger = logging.getLogger(__name__)


def _set_rerun_control_step(step: int) -> None:
    """Set timeline step (compatible with rerun-sdk 0.32+ and older APIs)."""
    if hasattr(rr, "set_time"):
        rr.set_time("control_step", sequence=int(step))
    elif hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("control_step", int(step))


def _init_rerun(
    session_name: str = "lerobot_control_loop",
    connect_ip: str | None = None,
    connect_port: int | None = None,
) -> None:
    """Initialize Rerun and open (or connect to) the viewer for live streaming."""
    os.environ["RERUN_FLUSH_NUM_BYTES"] = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    try:
        if connect_ip:
            rr.init(session_name)
            port = connect_port if connect_port is not None else 9876
            rr.connect(f"rerun+http://{connect_ip}:{port}/proxy")
            logger.info(
                "Rerun SDK connected to gRPC proxy at %s:%d (use web UI at http://127.0.0.1:9090)",
                connect_ip,
                port,
            )
        else:
            try:
                rr.init(session_name, spawn=True, memory_limit=memory_limit)
            except TypeError:
                rr.init(session_name)
                rr.spawn(memory_limit=memory_limit)
            logger.info(
                "Rerun native viewer spawned (gRPC :9876). For browser UI run: "
                "rerun --serve-web --web-viewer --web-viewer-port 9090"
            )
    except Exception as exc:
        logger.warning("Failed to start Rerun: %s", exc)
        raise


def shutdown_rerun() -> None:
    try:
        rr.disconnect()
    except Exception:
        pass


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    img_np = image
    if img_np.ndim == 3 and img_np.shape[0] in (3, 4):
        img_np = np.transpose(img_np, (1, 2, 0))
    if img_np.dtype != np.uint8:
        if img_np.max() > 1.1:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        else:
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    return img_np


def log_rerun_data(
    observation: dict[str | Any, Any],
    action: dict[str | Any, Any],
    attention_maps: dict[str, np.ndarray] | None = None,
    grad_cam_edge_warnings: dict[str, bool] | None = None,
    grad_cam_edge_stats: dict[str, float] | None = None,
    grad_cam_edge_threshold: float | None = None,
    grad_cam_edge_margin_px: int | None = None,
    grad_cam_edge_roi_mode: str = "mitigation",
    attention_overlay_helper: type = ACTEigenCAMHelper,
    control_step: int | None = None,
):
    if control_step is not None:
        _set_rerun_control_step(control_step)

    for obs, val in observation.items():
        if isinstance(val, float):
            rr.log(f"observation.{obs}", rr.Scalars(val))
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                for i, v in enumerate(val):
                    rr.log(f"observation.{obs}_{i}", rr.Scalars(float(v)))
            else:
                rr.log(f"observation.{obs}", rr.Image(_to_uint8_image(val)))
                if attention_maps and obs in attention_maps:
                    edge_warning = bool(grad_cam_edge_warnings.get(obs, False)) if grad_cam_edge_warnings else False
                    edge_mean = grad_cam_edge_stats.get(obs) if grad_cam_edge_stats else None
                    attn_overlay = attention_overlay_helper.overlay_attention_on_image(
                        val,
                        attention_maps[obs],
                        overlay_alpha=0.5,
                        use_rgb=True,
                        edge_warning=edge_warning,
                        edge_margin_px=grad_cam_edge_margin_px or 100,
                        edge_warning_mean=edge_mean,
                        edge_warning_threshold=grad_cam_edge_threshold,
                        edge_warning_roi_mode=grad_cam_edge_roi_mode,
                    )
                    rr.log(f"observation.{obs}_attention", rr.Image(attn_overlay))

    for act, val in action.items():
        if isinstance(val, float):
            rr.log(f"action.{act}", rr.Scalars(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"action.{act}_{i}", rr.Scalars(float(v)))
