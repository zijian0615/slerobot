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

import numpy as np

from slerobot.utils.lekiwi_ik import (
    ARM_JOINT_KEYS,
    LEKIWI_ARM_MOTOR_TO_URDF,
    observation_to_joint_degrees,
    transform_to_fanuc_mm_wpr,
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


def _fmt_xyz_wpr(x: float, y: float, z: float, w: float, p: float, r: float) -> str:
    return f"xyz_mm=({x:.1f},{y:.1f},{z:.1f}) wpr_deg=({w:.1f},{p:.1f},{r:.1f})"


def _fmt_controller(quest_action: dict[str, Any]) -> str:
    pos = quest_action.get("position", {})
    rot = quest_action.get("rotation", {})
    btn = quest_action.get("buttons", {})
    return (
        f"controller     {_fmt_xyz_wpr(float(pos.get('x', 0)), float(pos.get('y', 0)), float(pos.get('z', 0)), float(rot.get('w', 0)), float(rot.get('p', 0)), float(rot.get('r', 0)))} "
        f"btn(A={int(btn.get('a', 0))} trig={int(btn.get('trigger', 0))} grip={int(btn.get('grip', 0))})"
    )


def _arm_norm_from_dict(data: dict[str, Any]) -> np.ndarray:
    return np.array([float(data.get(k, 0.0)) for k in ARM_JOINT_KEYS], dtype=float)


def _fmt_joints_norm(label: str, joints: np.ndarray) -> str:
    names = [m.removeprefix("arm_") for m in LEKIWI_ARM_MOTOR_TO_URDF]
    parts = [f"{n}={joints[i]:.2f}" for i, n in enumerate(names)]
    return f"{label:14} norm[{','.join(parts)}]"


def build_teleop_alignment_lines(
    quest_action: dict[str, Any],
    observation: dict[str, Any],
    mapped_action: dict[str, Any],
    quest_mapper: Any | None = None,
) -> list[str]:
    """Controller pose, RA EE pose (FK), RA joint norm — same timestep."""
    lines = [_fmt_controller(quest_action)]

    ik = getattr(quest_mapper, "_ik", None) if quest_mapper is not None else None
    cal = getattr(ik, "calibration", None) if ik is not None else None
    use_degrees = bool(getattr(ik, "use_degrees", False)) if ik is not None else False

    present_norm = _arm_norm_from_dict(observation)
    cmd_norm = _arm_norm_from_dict(mapped_action)
    lines.append(_fmt_joints_norm("ra_joints_present", present_norm))
    lines.append(
        _fmt_joints_norm("ra_joints_command", cmd_norm)
        + "  "
        + _fmt_joint_delta(present_norm, cmd_norm)
    )

    if ik is not None and getattr(ik, "kinematics", None) is not None:
        present_deg = observation_to_joint_degrees(
            observation, cal, use_degrees=use_degrees
        )
        cmd_deg = observation_to_joint_degrees(
            {k: mapped_action.get(k, observation.get(k, 0.0)) for k in ARM_JOINT_KEYS},
            cal,
            use_degrees=use_degrees,
        )
        t_present = ik.kinematics.forward_kinematics(present_deg)
        t_cmd = ik.kinematics.forward_kinematics(cmd_deg)
        px, py, pz, pw, pp, pr = transform_to_fanuc_mm_wpr(t_present)
        cx, cy, cz, cw, cp, cr = transform_to_fanuc_mm_wpr(t_cmd)
        lines.insert(
            1,
            f"ra_ee_present  {_fmt_xyz_wpr(px, py, pz, pw, pp, pr)}",
        )
        lines.insert(2, f"ra_ee_command  {_fmt_xyz_wpr(cx, cy, cz, cw, cp, cr)}")

        if getattr(ik, "_vr_zeroed", False):
            pos = quest_action.get("position", {})
            rot = quest_action.get("rotation", {})
            qx = float(pos.get("x", 0.0))
            qy = float(pos.get("y", 0.0))
            qz = float(pos.get("z", 0.0))
            qw = float(rot.get("w", 0.0))
            qp = float(rot.get("p", 0.0))
            qr = float(rot.get("r", 0.0))
            dx, dy, dz, dw, dp, dr = ik._quest_offsets_from_zero(qx, qy, qz, qw, qp, qr)
            lines[0] += f"  vs_neutral_mm=({dx:.2f},{dy:.2f},{dz:.2f}) d_wpr=({dw:.1f},{dp:.1f},{dr:.1f})"
            if ik._filtered_delta is not None:
                fd = ik._filtered_delta
                lines[0] += f"  smoothed_mm=({fd[0]:.2f},{fd[1]:.2f},{fd[2]:.2f})"
    else:
        lines.insert(1, "ra_ee_present  (IK/placo unavailable — joints only)")

    return lines


def _fmt_joint_delta(present: np.ndarray, cmd: np.ndarray) -> str:
    deltas = []
    names = [m.removeprefix("arm_") for m in LEKIWI_ARM_MOTOR_TO_URDF]
    for i, n in enumerate(names):
        d = float(cmd[i] - present[i])
        if abs(d) > 0.05:
            deltas.append(f"{n}:{d:+.2f}")
    return "delta=" + (",".join(deltas) if deltas else "~0")


def log_lekiwi_teleop_alignment(
    stage: str,
    *,
    quest_action: dict[str, Any],
    observation: dict[str, Any],
    mapped_action: dict[str, Any],
    quest_mapper: Any | None = None,
) -> None:
    if not action_debug_enabled() or not _limiter.ok():
        return
    lines = [f"[LeKiwi teleop align] {stage}"] + [
        f"  {line}" for line in build_teleop_alignment_lines(
            quest_action, observation, mapped_action, quest_mapper
        )
    ]
    print("\n".join(lines), flush=True)


def _fmt_arm(action: dict[str, Any]) -> str:
    parts = [
        f"{k.split('.')[0].removeprefix('arm_')}={float(action.get(k, 0)):.2f}"
        for k in ARM_JOINT_KEYS
    ]
    base = [f"{k}={float(action.get(k, 0)):.3f}" for k in _BASE_VEL_KEYS if k in action]
    return "arm[" + ", ".join(parts) + "]" + (" base[" + ", ".join(base) + "]" if base else "")


def log_lekiwi_action_debug(
    stage: str,
    *,
    quest_action: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    mapped_action: dict[str, Any] | None = None,
    zmq_payload: dict[str, Any] | None = None,
    quest_mapper: Any | None = None,
    extra: str | None = None,
) -> None:
    if not action_debug_enabled():
        return

    # Mac mapper stage: print controller / RA EE / RA joints in one aligned block.
    if (
        stage == "mac_after_mapper"
        and quest_action is not None
        and observation is not None
        and mapped_action is not None
    ):
        log_lekiwi_teleop_alignment(
            stage,
            quest_action=quest_action,
            observation=observation,
            mapped_action=mapped_action,
            quest_mapper=quest_mapper,
        )
        return

    if not _limiter.ok():
        return

    lines = [f"[LeKiwi send_action debug] {stage}"]
    if zmq_payload is not None:
        lines.append(f"  zmq  {_fmt_arm(zmq_payload)}")
    if extra:
        lines.append(f"  {extra}")
    print("\n".join(lines), flush=True)
