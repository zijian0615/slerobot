"""In-memory telemetry store for live camera + action charts (no Rerun)."""

from __future__ import annotations

import math
import threading
import time
from collections import deque


DEFAULT_ACTION_KEYS = ("j0", "j1", "j2", "j3", "j4", "j5", "j7")
TRIG_PAIRS = (("j3_sin", "j3_cos", "j3"), ("j4_sin", "j4_cos", "j4"), ("j5_sin", "j5_cos", "j5"))


def _decode_trig_action(action: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in action.items():
        if key in {"j3", "j4", "j5"}:
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue

    for sin_k, cos_k, angle_k in TRIG_PAIRS:
        if sin_k in action and cos_k in action:
            s = float(action[sin_k])
            c = float(action[cos_k])
            norm = math.hypot(s, c)
            if norm > 1e-8:
                out[angle_k] = math.degrees(math.atan2(s / norm, c / norm))
            else:
                out[angle_k] = 0.0
        elif angle_k in action:
            out[angle_k] = float(action[angle_k])
    return out


class TelemetryHub:
    def __init__(self, history_len: int = 400, action_keys: tuple[str, ...] = DEFAULT_ACTION_KEYS) -> None:
        self._lock = threading.Lock()
        self.history_len = history_len
        self.action_keys = action_keys
        self._step = 0
        self._timestamp = 0.0
        self._mode: str | None = None
        self._cameras: dict[str, str] = {}
        self._attention: dict[str, str] = {}
        self._warnings: dict[str, bool] = {}
        self._history: dict[str, deque[float]] = {k: deque(maxlen=history_len) for k in action_keys}
        self._steps: deque[int] = deque(maxlen=history_len)

    def reset(self) -> None:
        with self._lock:
            self._step = 0
            self._cameras.clear()
            self._attention.clear()
            self._warnings.clear()
            self._steps.clear()
            for q in self._history.values():
                q.clear()

    def push(
        self,
        *,
        step: int | None = None,
        cameras: dict[str, str] | None = None,
        attention: dict[str, str] | None = None,
        action: dict | None = None,
        warnings: dict[str, bool] | None = None,
        mode: str | None = None,
    ) -> None:
        decoded = _decode_trig_action(action or {})
        with self._lock:
            self._step = int(step if step is not None else self._step + 1)
            self._timestamp = time.time()
            if mode is not None:
                self._mode = mode
            if cameras:
                self._cameras.update(cameras)
            if attention:
                self._attention.update(attention)
            if warnings:
                self._warnings.update(warnings)
            self._steps.append(self._step)
            for key in self.action_keys:
                self._history[key].append(float(decoded.get(key, float("nan"))))

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "step": self._step,
                "timestamp": self._timestamp,
                "mode": self._mode,
                "cameras": dict(self._cameras),
                "attention": dict(self._attention),
                "warnings": dict(self._warnings),
                "action_keys": list(self.action_keys),
                "steps": list(self._steps),
                "series": {k: list(self._history[k]) for k in self.action_keys},
            }


TELEMETRY = TelemetryHub()
