"""Push camera frames and actions to the Attack/Defense UI (HTTP, no Rerun)."""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _numpy_to_jpeg_b64(image: np.ndarray, quality: int = 72) -> str:
    import cv2

    img = image
    if img.ndim == 3 and img.shape[0] in (3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        if img.max() <= 1.1:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def push_live_telemetry(
    observation: dict[str, Any],
    action: dict[str, Any],
    *,
    step: int | None = None,
    attention_maps: dict[str, np.ndarray] | None = None,
    grad_cam_edge_warnings: dict[str, bool] | None = None,
    attention_overlay_helper: type | None = None,
    grad_cam_edge_margin_px: int = 100,
    grad_cam_edge_threshold: float | None = None,
    grad_cam_edge_stats: dict[str, float] | None = None,
    grad_cam_edge_roi_mode: str = "mitigation",
    camera_keys: tuple[str, ...] = ("front", "side"),
) -> None:
    """POST one frame to the local Attack/Defense UI telemetry endpoint."""
    url = os.getenv("SLEROBOT_TELEMETRY_URL", "http://127.0.0.1:8765/api/telemetry/push")
    if not url:
        return

    cameras_b64: dict[str, str] = {}
    for key in camera_keys:
        if key not in observation:
            continue
        val = observation[key]
        if isinstance(val, np.ndarray) and val.ndim >= 2:
            cameras_b64[key] = _numpy_to_jpeg_b64(val)

    session_mode = os.getenv("SLEROBOT_SESSION_MODE", "").lower()
    push_attention = session_mode == "detect"

    attention_b64: dict[str, str] = {}
    if push_attention and attention_maps and attention_overlay_helper is not None:
        for key, attn_map in attention_maps.items():
            if key not in observation or attn_map is None:
                continue
            edge_warning = bool(grad_cam_edge_warnings.get(key, False)) if grad_cam_edge_warnings else False
            edge_mean = grad_cam_edge_stats.get(key) if grad_cam_edge_stats else None
            overlay = attention_overlay_helper.overlay_attention_on_image(
                observation[key],
                attn_map,
                overlay_alpha=0.5,
                use_rgb=True,
                edge_warning=edge_warning,
                edge_margin_px=grad_cam_edge_margin_px,
                edge_warning_mean=edge_mean,
                edge_warning_threshold=grad_cam_edge_threshold,
                edge_warning_roi_mode=grad_cam_edge_roi_mode,
            )
            attention_b64[key] = _numpy_to_jpeg_b64(overlay)

    payload = {
        "step": step,
        "mode": session_mode,
        "cameras": cameras_b64,
        "attention": attention_b64,
        "action": {k: float(v) for k, v in action.items() if isinstance(v, (int, float, np.floating))},
        "warnings": grad_cam_edge_warnings or {},
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=0.15) as resp:
            resp.read()
    except urllib.error.URLError as exc:
        logger.debug("Telemetry push skipped: %s", exc)
