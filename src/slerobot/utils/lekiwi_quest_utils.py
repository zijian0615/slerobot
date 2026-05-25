"""Map Quest 3s MQTT EE poses to LeKiwi arm + base command dicts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slerobot.motors import MotorCalibration

from .lekiwi_ik import ARM_GRIPPER_KEY, ARM_JOINT_KEYS, LeKiwiQuestIK

logger = logging.getLogger(__name__)


@dataclass
class LeKiwiQuestMapperConfig:
    """Quest teleop mapping for LeKiwi."""

    use_ik: bool = True
    urdf_path: str | None = None
    # Fanuc VR: x,y,z,w,p,r are offsets from A-button zero (not absolute cell pose).
    quest_pose_mode: str = "relative_to_a"
    use_degrees: bool = False
    quest_position_scale: float = 0.5
    ee_position_scale_mm: float = 0.001
    position_weight: float = 1.0
    orientation_weight: float = 0.05
    max_delta_translation_m: float = 0.12
    require_trigger: bool = True
    settle_frames_after_zero: int = 20
    warmup_frames_after_settle: int = 10
    ramp_frames: int = 0
    max_joint_step_deg: float = 5.0
    position_deadzone_mm: float = 1.5
    rotation_deadzone_deg: float = 1.5
    quest_delta_ema_alpha: float = 0.55
    resync_zero_during_settle: bool = True
    gripper_open: float = 0.0
    gripper_closed: float = 100.0
    joystick_xy_speed: float = 0.15
    joystick_theta_speed: float = 45.0
    # Legacy heuristic (only if use_ik=False or placo missing)
    position_scale_pan: float = 80.0
    position_scale_lift: float = 80.0
    position_scale_elbow: float = 60.0
    orientation_scale_wrist_flex: float = 40.0
    orientation_scale_wrist_roll: float = 40.0


def _read_joystick(quest_action: dict[str, Any]) -> tuple[float, float, float]:
    """Return (x, y, theta) stick values in [-1, 1] from Quest MQTT payload."""
    keys_xy = (
        ("joystickX", "joystickY"),
        ("joystick_x", "joystick_y"),
        ("thumbstickX", "thumbstickY"),
        ("leftThumbstickX", "leftThumbstickY"),
        ("axisX", "axisY"),
        ("moveX", "moveY"),
    )
    jx, jy = 0.0, 0.0
    for kx, ky in keys_xy:
        if kx in quest_action or ky in quest_action:
            jx = float(quest_action.get(kx, 0.0))
            jy = float(quest_action.get(ky, 0.0))
            break
    jtheta = float(
        quest_action.get(
            "rightThumbstickX",
            quest_action.get("rotateStickX", quest_action.get("joystickRX", 0.0)),
        )
    )
    return jx, jy, jtheta


def quest_base_action(quest_action: dict[str, Any], config: LeKiwiQuestMapperConfig) -> dict[str, float]:
    jx, jy, jtheta = _read_joystick(quest_action)
    if abs(jx) < 0.05 and abs(jy) < 0.05 and abs(jtheta) < 0.05:
        return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
    return {
        "x.vel": jy * config.joystick_xy_speed,
        "y.vel": jx * config.joystick_xy_speed,
        "theta.vel": jtheta * config.joystick_theta_speed,
    }


def base_action_is_active(base_action: dict[str, float]) -> bool:
    return any(abs(float(v)) > 1e-6 for k, v in base_action.items() if k.endswith(".vel"))


def _heuristic_arm_action(
    quest_action: dict[str, Any],
    observation: dict[str, Any],
    config: LeKiwiQuestMapperConfig,
    reference_position: tuple[float, float, float] | None,
    reference_orientation: tuple[float, float, float] | None,
) -> tuple[dict[str, float], tuple[float, float, float], tuple[float, float, float]]:
    position = quest_action.get("position", {})
    rotation = quest_action.get("rotation", {})

    px = float(position.get("x", 0.0))
    py = float(position.get("y", 0.0))
    pz = float(position.get("z", 0.0))
    rw = float(rotation.get("w", 0.0))
    rp = float(rotation.get("p", 0.0))
    rr = float(rotation.get("r", 0.0))

    if reference_position is None:
        ref_pos = (px, py, pz)
        ref_ori = (rw, rp, rr)
        arm_action = {key: float(observation.get(key, 0.0)) for key in ARM_JOINT_KEYS}
        return arm_action, ref_pos, ref_ori

    ref_x, ref_y, ref_z = reference_position
    ref_w, ref_p, ref_r = reference_orientation or (rw, rp, rr)
    dx, dy, dz = px - ref_x, py - ref_y, pz - ref_z
    dw, dp, dr = rw - ref_w, rp - ref_p, rr - ref_r

    arm_action = {
        "arm_shoulder_pan.pos": float(observation.get("arm_shoulder_pan.pos", 0.0))
        + dx * config.position_scale_pan,
        "arm_shoulder_lift.pos": float(observation.get("arm_shoulder_lift.pos", 0.0))
        + dy * config.position_scale_lift,
        "arm_elbow_flex.pos": float(observation.get("arm_elbow_flex.pos", 0.0))
        + dz * config.position_scale_elbow,
        "arm_wrist_flex.pos": float(observation.get("arm_wrist_flex.pos", 0.0))
        + dp * config.orientation_scale_wrist_flex,
        "arm_wrist_roll.pos": float(observation.get("arm_wrist_roll.pos", 0.0))
        + dr * config.orientation_scale_wrist_roll,
        ARM_GRIPPER_KEY: float(observation.get(ARM_GRIPPER_KEY, 0.0)),
    }
    return arm_action, reference_position, reference_orientation or (rw, rp, rr)


@dataclass
class LeKiwiQuestMapper:
    """Quest EE (Fanuc-style) -> LeKiwi joint targets via placo IK (default)."""

    config: LeKiwiQuestMapperConfig = field(default_factory=LeKiwiQuestMapperConfig)
    calibration: dict[str, MotorCalibration] | None = None
    reference_position: tuple[float, float, float] | None = None
    reference_orientation: tuple[float, float, float] | None = None
    _ik: LeKiwiQuestIK | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.config.use_ik:
            return
        try:
            urdf = Path(self.config.urdf_path) if self.config.urdf_path else None
            self._ik = LeKiwiQuestIK(
                urdf,
                quest_pose_mode=self.config.quest_pose_mode,  # type: ignore[arg-type]
                position_weight=self.config.position_weight,
                orientation_weight=self.config.orientation_weight,
                max_delta_translation_m=self.config.max_delta_translation_m,
                require_trigger=self.config.require_trigger,
                settle_frames_after_zero=self.config.settle_frames_after_zero,
                warmup_frames_after_settle=self.config.warmup_frames_after_settle,
                ramp_frames=self.config.ramp_frames,
                max_joint_step_deg=self.config.max_joint_step_deg,
                position_deadzone_mm=self.config.position_deadzone_mm,
                rotation_deadzone_deg=self.config.rotation_deadzone_deg,
                quest_delta_ema_alpha=self.config.quest_delta_ema_alpha,
                resync_zero_during_settle=self.config.resync_zero_during_settle,
                quest_position_scale=self.config.quest_position_scale,
                ee_position_scale_mm=self.config.ee_position_scale_mm,
                use_degrees=self.config.use_degrees,
                calibration=self.calibration,
            )
            logger.info("LeKiwi Quest mapper: using placo IK (SO-101 URDF).")
        except ImportError as exc:
            logger.warning(
                "placo not available (%s). Falling back to heuristic joint mapping. "
                "Install: pip install 'slerobot[kinematics]'",
                exc,
            )
            self._ik = None

    def reset(self) -> None:
        self.reference_position = None
        self.reference_orientation = None
        if self._ik is not None:
            self._ik.reset()

    def map_action(
        self,
        quest_action: dict[str, Any],
        observation: dict[str, Any],
        base_action: dict[str, float] | None = None,
    ) -> dict[str, float]:
        buttons = quest_action.get("buttons", {})

        if self._ik is not None:
            arm_action = self._ik.solve(quest_action, observation)
        else:
            arm_action, self.reference_position, self.reference_orientation = _heuristic_arm_action(
                quest_action,
                observation,
                self.config,
                self.reference_position,
                self.reference_orientation,
            )

        if int(buttons.get("grip", 0)):
            arm_action[ARM_GRIPPER_KEY] = self.config.gripper_closed
        else:
            arm_action[ARM_GRIPPER_KEY] = self.config.gripper_open

        merged_base = quest_base_action(quest_action, self.config)
        if base_action and base_action_is_active(base_action):
            merged_base = {**merged_base, **base_action}

        return {**arm_action, **merged_base}
