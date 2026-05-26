"""Quest EE poses -> LeKiwi SO-101 joint targets via placo IK."""

from __future__ import annotations

import enum
import logging
from pathlib import Path
from typing import Any

import numpy as np

from slerobot.motors import MotorCalibration, MotorNormMode
from slerobot.utils.robot_kinematics import (
    LEKIWI_ARM_MOTOR_TO_URDF,
    SO101_ARM_JOINT_NAMES,
    RobotKinematics,
    default_so101_urdf_path,
)

logger = logging.getLogger(__name__)

ARM_JOINT_KEYS = tuple(f"{motor}.pos" for motor in LEKIWI_ARM_MOTOR_TO_URDF)
ARM_GRIPPER_KEY = "arm_gripper.pos"

STS3215_MAX_RES = 4095


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def _q_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-10 else q


def _q_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of unit quaternion [x, y, z, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def _q_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, [x, y, z, w] convention."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=float)


def _q_to_mat3(q: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix from unit quaternion [x, y, z, w]."""
    q = _q_normalize(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ], dtype=float)


def _quest_quat(quest_action: dict[str, Any]) -> np.ndarray | None:
    """Extract [x, y, z, w] quaternion from quest_action, or None if missing."""
    qx = quest_action.get("qx")
    qy = quest_action.get("qy")
    qz = quest_action.get("qz")
    qw = quest_action.get("qw")
    if any(v is None for v in (qx, qy, qz, qw)):
        return None
    q = np.array([float(qx), float(qy), float(qz), float(qw)], dtype=float)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-10 else None


# ── Calibration helpers ────────────────────────────────────────────────────────

def normalized_pos_to_degrees(
    norm: float,
    calibration: MotorCalibration | None,
    *,
    norm_mode: MotorNormMode = MotorNormMode.RANGE_M100_100,
    drive_mode: bool = False,
) -> float:
    if norm_mode is MotorNormMode.DEGREES:
        return float(norm)
    if calibration is None:
        return float(norm) * 1.8
    min_ = calibration.range_min
    max_ = calibration.range_max
    val = -norm if drive_mode and norm_mode is MotorNormMode.RANGE_M100_100 else norm
    bounded = min(100.0, max(-100.0, val))
    raw = int(((bounded + 100) / 200) * (max_ - min_) + min_)
    mid = (min_ + max_) / 2
    return (raw - mid) * 360.0 / STS3215_MAX_RES


def degrees_to_normalized_pos(
    deg: float,
    calibration: MotorCalibration | None,
    *,
    norm_mode: MotorNormMode = MotorNormMode.RANGE_M100_100,
    drive_mode: bool = False,
) -> float:
    if norm_mode is MotorNormMode.DEGREES:
        return float(deg)
    if calibration is None:
        return float(deg) / 1.8
    min_ = calibration.range_min
    max_ = calibration.range_max
    mid = (min_ + max_) / 2
    raw = int((deg * STS3215_MAX_RES / 360.0) + mid)
    bounded = min(max_, max(min_, raw))
    norm = (((bounded - min_) / (max_ - min_)) * 200) - 100
    return -norm if drive_mode else norm


def observation_to_joint_degrees(
    observation: dict[str, Any],
    calibration: dict[str, MotorCalibration] | None,
    *,
    use_degrees: bool = False,
) -> np.ndarray:
    norm_mode = MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    joints = []
    for motor in LEKIWI_ARM_MOTOR_TO_URDF:
        key = f"{motor}.pos"
        norm = float(observation.get(key, 0.0))
        cal = calibration.get(motor) if calibration else None
        drive = bool(cal.drive_mode) if cal else False
        joints.append(
            normalized_pos_to_degrees(norm, cal, norm_mode=norm_mode, drive_mode=drive)
        )
    return np.array(joints, dtype=float)


def joint_degrees_to_arm_action(
    joint_deg: np.ndarray,
    calibration: dict[str, MotorCalibration] | None,
    *,
    use_degrees: bool = False,
) -> dict[str, float]:
    norm_mode = MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    action: dict[str, float] = {}
    for i, motor in enumerate(LEKIWI_ARM_MOTOR_TO_URDF):
        cal = calibration.get(motor) if calibration else None
        drive = bool(cal.drive_mode) if cal else False
        action[f"{motor}.pos"] = degrees_to_normalized_pos(
            float(joint_deg[i]), cal, norm_mode=norm_mode, drive_mode=drive
        )
    return action


# ── Quest button helpers ───────────────────────────────────────────────────────

def quest_a_button_pressed(quest_action: dict[str, Any]) -> bool:
    """True if Quest A / primary button is held (covers multiple MQTT field names)."""
    buttons = quest_action.get("buttons", {})
    if int(buttons.get("a", 0)):
        return True
    for key in ("aButton", "buttonA", "primaryButton", "A", "button_a"):
        if int(quest_action.get(key, 0)):
            return True
    return False


def quest_trigger_pressed(quest_action: dict[str, Any]) -> bool:
    buttons = quest_action.get("buttons", {})
    if int(buttons.get("trigger", 0)):
        return True
    return bool(int(quest_action.get("triggerButton", 0)))


# ── Axis remap ─────────────────────────────────────────────────────────────────

def parse_quest_axis_remap(spec: str) -> np.ndarray:
    """Build 3×3 permutation/sign matrix from spec like ``z,-x,y`` (LeKiwi default)."""
    axes = {"x": 0, "y": 1, "z": 2}
    mat = np.zeros((3, 3), dtype=float)
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"quest_axis_remap must have 3 comma-separated axes, got {spec!r}")
    for out_i, token in enumerate(parts):
        sign = -1.0 if token.startswith("-") else 1.0
        axis = token.lstrip("+-").lower()
        if axis not in axes:
            raise ValueError(f"Invalid axis {axis!r} in quest_axis_remap {spec!r}")
        mat[out_i, axes[axis]] = sign
    return mat


# ── Debug helper (used by lekiwi_action_debug) ────────────────────────────────

def rotation_matrix_to_wpr_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    """Intrinsic ZYX (Fanuc W,P,R) from a 3×3 rotation matrix."""
    r = rotation
    p_rad = float(np.arcsin(np.clip(-r[2, 0], -1.0, 1.0)))
    cp = np.cos(p_rad)
    if abs(cp) > 1e-6:
        w_rad = float(np.arctan2(r[1, 0], r[0, 0]))
        r_rad = float(np.arctan2(r[2, 1], r[2, 2]))
    else:
        w_rad = float(np.arctan2(-r[0, 1], r[1, 1]))
        r_rad = 0.0
    w, p, r_deg = np.rad2deg([w_rad, p_rad, r_rad])
    return float(w), float(p), float(r_deg)


def transform_to_fanuc_mm_wpr(
    transform: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """EE pose as Fanuc-style mm + W,P,R degrees (for debug / telemetry)."""
    x_mm, y_mm, z_mm = (transform[:3, 3] * 1000.0).tolist()
    w, p, r = rotation_matrix_to_wpr_deg(transform[:3, :3])
    return x_mm, y_mm, z_mm, w, p, r


# ── Simple IK controller ───────────────────────────────────────────────────────

class _State(enum.Enum):
    IDLE = "idle"
    SETTLING = "settling"
    ARMED = "armed"


class SimpleLeKiwiQuestIK:
    """Quest position + orientation delta → LeKiwi SO-101 joint targets via placo IK.

    State machine:
      IDLE  ──[A press]──>  SETTLING  ──[settle_frames elapsed]──>  ARMED
                                                                       │
                                                           trigger held → IK + send
                                                           trigger released → freeze
                                                           A press → re-zero (→ SETTLING)

    Position: target = neutral_ee_pos + remap(quest_pos - quest_zero) in base frame.
    Orientation: target = R_delta_remapped @ neutral_ee_rot, where R_delta is the
    relative rotation from the Quest quaternion at arm time.
    Set orientation_weight=0 to disable rotation tracking.
    """

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        *,
        quest_axis_remap: str = "z,-x,y",
        position_scale: float = 1.0,
        orientation_weight: float = 0.3,
        max_delta_m: float = 0.12,
        settle_frames: int = 12,
        max_joint_step_deg: float = 8.0,
        ema_alpha: float = 0.6,
        deadzone_mm: float = 3.0,
        use_degrees: bool = False,
        calibration: dict[str, MotorCalibration] | None = None,
    ) -> None:
        path = Path(urdf_path) if urdf_path else default_so101_urdf_path()
        self.kinematics = RobotKinematics(path, joint_names=list(SO101_ARM_JOINT_NAMES))
        self._remap = parse_quest_axis_remap(quest_axis_remap)
        self.position_scale = position_scale
        self.orientation_weight = orientation_weight
        self.max_delta_m = max_delta_m
        self.settle_frames = settle_frames
        self.max_joint_step_deg = max_joint_step_deg
        self.ema_alpha = ema_alpha
        self.deadzone_mm = deadzone_mm
        self.use_degrees = use_degrees
        self.calibration = calibration or {}

        self._state: _State = _State.IDLE
        self._quest_zero: np.ndarray | None = None   # (3,) mm — Quest pos at arm time
        self._q_zero: np.ndarray | None = None       # (4,) [x,y,z,w] — Quest quat at arm time
        self._neutral_T: np.ndarray | None = None    # (4,4) robot EE FK at arm time
        self._neutral_joints: np.ndarray | None = None
        self._last_joints: np.ndarray | None = None  # last *commanded* joints
        self._settle_buf: list[np.ndarray] = []
        self._settle_qbuf: list[np.ndarray] = []     # quaternion settle buffer
        self._prev_a: bool = False
        self._prev_trigger: bool = False
        self._logged_idle: bool = False
        self._idle_log_countdown: int = 0

    def reset(self) -> None:
        self._state = _State.IDLE
        self._quest_zero = None
        self._q_zero = None
        self._neutral_T = None
        self._neutral_joints = None
        self._last_joints = None
        self._settle_buf.clear()
        self._settle_qbuf.clear()
        self._prev_a = False
        self._prev_trigger = False
        self._logged_idle = False
        self._idle_log_countdown = 0

    # ── public property for debug compatibility ──────────────────────────────

    @property
    def is_armed(self) -> bool:
        return self._state == _State.ARMED

    def quest_delta_mm(
        self, quest_action: dict[str, Any]
    ) -> tuple[float, float, float] | None:
        """Current Quest-space delta from zero, or None if not armed."""
        if self._quest_zero is None:
            return None
        pos = quest_action.get("position", {})
        q = np.array([float(pos.get(k, 0.0)) for k in ("x", "y", "z")], dtype=float)
        d = q - self._quest_zero
        return float(d[0]), float(d[1]), float(d[2])

    # ── helpers ──────────────────────────────────────────────────────────────

    def _hold_last(self, current_joints: np.ndarray) -> dict[str, float]:
        """Return last commanded joint action (or current obs if not available)."""
        joints = self._last_joints if self._last_joints is not None else current_joints
        return joint_degrees_to_arm_action(joints, self.calibration, use_degrees=self.use_degrees)

    def _clamp_step(self, raw: np.ndarray) -> np.ndarray:
        assert self._last_joints is not None
        delta = raw - self._last_joints
        max_abs = float(np.max(np.abs(delta)))
        if max_abs > self.max_joint_step_deg:
            raw = self._last_joints + delta * (self.max_joint_step_deg / max_abs)
        return raw

    # ── main entry point ─────────────────────────────────────────────────────

    def solve(
        self,
        quest_action: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, float]:
        pos = quest_action.get("position", {})
        q_pos = np.array([float(pos.get(k, 0.0)) for k in ("x", "y", "z")], dtype=float)

        current_joints = observation_to_joint_degrees(
            observation, self.calibration, use_degrees=self.use_degrees
        )

        a_now = quest_a_button_pressed(quest_action)
        trig_now = quest_trigger_pressed(quest_action)
        a_rising = a_now and not self._prev_a
        trig_falling = (not trig_now) and self._prev_trigger
        self._prev_a = a_now
        self._prev_trigger = trig_now

        # ── A button: (re-)arm — start settle ────────────────────────────
        if a_rising:
            self._state = _State.SETTLING
            self._settle_buf.clear()
            self._settle_qbuf.clear()
            self._last_joints = current_joints.copy()
            self._logged_idle = False
            logger.info(
                "Quest A — settling %d frames (~%.1fs), keep hand still.",
                self.settle_frames,
                self.settle_frames / 20.0,
            )
            return self._hold_last(current_joints)

        # ── SETTLING ─────────────────────────────────────────────────────
        if self._state == _State.SETTLING:
            self._settle_buf.append(q_pos.copy())
            q_now = _quest_quat(quest_action)
            if q_now is not None:
                self._settle_qbuf.append(q_now)
            if len(self._settle_buf) >= self.settle_frames:
                # Lock Quest zero and robot FK at the same instant
                self._quest_zero = np.mean(self._settle_buf, axis=0)
                if self._settle_qbuf:
                    q_avg = np.mean(self._settle_qbuf, axis=0)
                    self._q_zero = _q_normalize(q_avg)
                else:
                    self._q_zero = None
                self._neutral_T = self.kinematics.forward_kinematics(current_joints)
                self._neutral_joints = current_joints.copy()
                self._last_joints = current_joints.copy()
                self._state = _State.ARMED
                self._settle_buf.clear()
                self._settle_qbuf.clear()
                logger.info(
                    "Teleop ARMED — quest_zero=(%.1f, %.1f, %.1f) mm | "
                    "neutral_ee=(%.3f, %.3f, %.3f) m",
                    *self._quest_zero,
                    *self._neutral_T[:3, 3],
                )
            return self._hold_last(current_joints)

        # ── IDLE: wait for A ──────────────────────────────────────────────
        if self._state != _State.ARMED:
            if not self._logged_idle:
                logger.info("Arm teleop idle — press Quest A to arm.")
                self._logged_idle = True
                self._idle_log_countdown = 0
            # Periodically log button state so the user can confirm A is detected
            self._idle_log_countdown -= 1
            if self._idle_log_countdown <= 0:
                btns = quest_action.get("buttons", {})
                pos = quest_action.get("position", {})
                logger.info(
                    "Quest state — A(primary)=%s secondary=%s trig=%s grip=%s "
                    "| pos_mm=(%.1f, %.1f, %.1f) — press A or B to arm",
                    btns.get("a", "?"),
                    quest_action.get("secondaryButton", "n/a"),
                    btns.get("trigger", "?"),
                    btns.get("grip", "?"),
                    float(pos.get("x", 0)),
                    float(pos.get("y", 0)),
                    float(pos.get("z", 0)),
                )
                self._idle_log_countdown = 100  # log every ~5s at 20Hz
            return self._hold_last(current_joints)

        # ── ARMED: trigger release → freeze ──────────────────────────────
        if trig_falling:
            logger.info("Trigger released — arm frozen at last commanded pose.")
            return self._hold_last(current_joints)

        # ── ARMED: trigger not held → hold ───────────────────────────────
        if not trig_now:
            return self._hold_last(current_joints)

        # ── ARMED + trigger held: compute IK target ───────────────────────
        assert self._quest_zero is not None
        assert self._neutral_T is not None
        assert self._last_joints is not None

        delta_quest = q_pos - self._quest_zero  # (3,) mm in Quest/Fanuc space

        # Deadzone: L-inf in Quest space
        if float(np.max(np.abs(delta_quest))) < self.deadzone_mm:
            return self._hold_last(current_joints)

        # ── Position target ──────────────────────────────────────────────
        # Remap Quest/Fanuc axes → robot base frame, scale mm → m.
        delta_base = self._remap @ delta_quest * (self.position_scale * 1e-3)
        norm = float(np.linalg.norm(delta_base))
        if norm > self.max_delta_m:
            delta_base *= self.max_delta_m / norm

        target_T = self._neutral_T.copy()
        target_T[:3, 3] = self._neutral_T[:3, 3] + delta_base

        # ── Orientation target ───────────────────────────────────────────
        # Compute relative rotation from Quest quaternion since arm time.
        # R_delta is in Quest world space; remap axes to robot base space:
        #   R_robot = P @ R_quest_delta @ P^T   (P = axis-remap matrix)
        ori_w = 0.0
        q_now = _quest_quat(quest_action)
        if q_now is not None and self._q_zero is not None and self.orientation_weight > 0:
            q_delta = _q_multiply(q_now, _q_conjugate(self._q_zero))
            R_delta_quest = _q_to_mat3(q_delta)
            P = self._remap
            R_delta_robot = P @ R_delta_quest @ P.T
            target_T[:3, :3] = R_delta_robot @ self._neutral_T[:3, :3]
            ori_w = float(self.orientation_weight)

        # IK — warm start from last commanded joints for smooth, consistent solutions
        try:
            raw = self.kinematics.inverse_kinematics(
                self._last_joints,
                target_T,
                position_weight=1.0,
                orientation_weight=ori_w,
            )
        except Exception as exc:
            logger.warning("IK failed (%s) — holding joints.", exc)
            return self._hold_last(current_joints)

        if not hasattr(self, "_ik_log_cnt"):
            self._ik_log_cnt = 0
        self._ik_log_cnt += 1
        if self._ik_log_cnt % 20 == 1:
            logger.info(
                "IK: quest_delta=(%.0f,%.0f,%.0f)mm ori_w=%.2f "
                "| Δjoints=%s°",
                *delta_quest,
                ori_w,
                [round(float(v), 1) for v in (raw - self._last_joints)],
            )

        # Per-joint step limit
        raw = self._clamp_step(raw)

        # EMA smoothing
        a = float(self.ema_alpha)
        out = a * raw + (1.0 - a) * self._last_joints
        self._last_joints = out.copy()

        return joint_degrees_to_arm_action(out, self.calibration, use_degrees=self.use_degrees)
