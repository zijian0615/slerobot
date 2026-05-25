"""Optional debug prints for LeKiwi Quest → send_action pipeline.

Enable on Mac (record) or Pi (host):
  export SLEROBOT_DEBUG_ACTION=1
  # optional: max prints per second (default 5)
  export SLEROBOT_DEBUG_ACTION_HZ=5
"""

from __future__ import annotations

import os
import time
from typing import Any

_ARM_POS_KEYS = (
    "arm_shoulder_pan.pos",
    "arm_shoulder_lift.pos",
    "arm_elbow_flex.pos",
    "arm_wrist_flex.pos",
    "arm_wrist_roll.pos",
    "arm_gripper.pos",
)
_BASE_VEL_KEYS = ("x.vel", "y.vel", "theta.vel")


def action_debug_enabled() -> bool:
    return os.getenv("SLEROBOT_DEBUG_ACTION", "").lower() in ("1", "true", "yes")


class _RateLimiter:
    def __init__(self) -> None:
        self._last_t = 0.0
        try:
            hz = float(os.getenv("SLEROBOT_DEBUG_ACTION_HZ", "5"))
        except ValueError:
            hz = 5.0
        self._interval = 1.0 / max(hz, 0.1)

    def ok(self) -> bool:
        now = time.perf_counter()
        if now - self._last_t < self._interval:
            return False
        self._last_t = now
        return True


_limiter = _RateLimiter()


def _fmt_pose(quest_action: dict[str, Any] | None) -> str:
    if not quest_action:
        return "quest=None"
    pos = quest_action.get("position", {})
    rot = quest_action.get("rotation", {})
    btn = quest_action.get("buttons", {})
    return (
        f"quest pos=({float(pos.get('x', 0)):.1f},{float(pos.get('y', 0)):.1f},{float(pos.get('z', 0)):.1f}) "
        f"wpr=({float(rot.get('w', 0)):.1f},{float(rot.get('p', 0)):.1f},{float(rot.get('r', 0)):.1f}) "
        f"btn(A={int(btn.get('a', 0))} trig={int(btn.get('trigger', 0))} grip={int(btn.get('grip', 0))})"
    )


def _fmt_arm(action: dict[str, Any]) -> str:
    parts = [f"{k.split('.')[0].removeprefix('arm_')}={float(action.get(k, 0)):.2f}" for k in _ARM_POS_KEYS]
    base = [f"{k}={float(action.get(k, 0)):.3f}" for k in _BASE_VEL_KEYS if k in action]
    return "arm[" + ", ".join(parts) + "]" + (" base[" + ", ".join(base) + "]" if base else "")


def _fmt_delta(obs: dict[str, Any], action: dict[str, Any]) -> str:
    deltas = []
    for k in _ARM_POS_KEYS:
        o = float(obs.get(k, 0.0))
        a = float(action.get(k, 0.0))
        d = a - o
        if abs(d) > 0.05:
            deltas.append(f"{k}:{d:+.2f}")
    if not deltas:
        return "delta_arm=~hold"
    return "delta_arm " + ", ".join(deltas)


def log_lekiwi_action_debug(
    stage: str,
    *,
    quest_action: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    mapped_action: dict[str, Any] | None = None,
    zmq_payload: dict[str, Any] | None = None,
    extra: str | None = None,
) -> None:
    if not action_debug_enabled() or not _limiter.ok():
        return

    lines = [f"[LeKiwi send_action debug] {stage}"]
    if quest_action is not None:
        lines.append(f"  {_fmt_pose(quest_action)}")
    if observation is not None:
        lines.append(f"  obs  {_fmt_arm(observation)}")
    if mapped_action is not None:
        lines.append(f"  map  {_fmt_arm(mapped_action)}")
        if observation is not None:
            lines.append(f"  {_fmt_delta(observation, mapped_action)}")
    if zmq_payload is not None:
        lines.append(f"  zmq  {_fmt_arm(zmq_payload)}")
    if extra:
        lines.append(f"  {extra}")
    print("\n".join(lines), flush=True)
