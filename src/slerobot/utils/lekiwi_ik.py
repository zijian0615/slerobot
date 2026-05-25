"""Quest Fanuc-style EE poses -> LeKiwi joint targets via placo IK."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

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


def wpr_deg_to_rotation_matrix(w_deg: float, p_deg: float, r_deg: float) -> np.ndarray:
    """Fanuc W,P,R (degrees) as intrinsic ZYX Euler angles."""
    w, p, r = np.deg2rad([w_deg, p_deg, r_deg])
    cw, sw = np.cos(w), np.sin(w)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    rz = np.array([[cw, -sw, 0.0], [sw, cw, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def fanuc_pose_to_matrix(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    w_deg: float,
    p_deg: float,
    r_deg: float,
    *,
    position_scale: float = 0.001,
) -> np.ndarray:
    """Build 4x4 pose; Quest/Fanuc position is typically millimeters."""
    t = np.eye(4, dtype=float)
    t[:3, :3] = wpr_deg_to_rotation_matrix(w_deg, p_deg, r_deg)
    t[:3, 3] = [x_mm * position_scale, y_mm * position_scale, z_mm * position_scale]
    return t


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
        # Rough fallback when Mac client has no calibration file (IK still usable).
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


def quest_a_button_pressed(quest_action: dict[str, Any]) -> bool:
    """True if Quest A / primary button is held (MQTT field names differ by app)."""
    buttons = quest_action.get("buttons", {})
    if int(buttons.get("a", 0)):
        return True
    for key in ("aButton", "buttonA", "primaryButton", "A", "button_a"):
        if int(quest_action.get(key, 0)):
            return True
    return False


QuestPoseMode = Literal["relative_to_a", "absolute_pair"]


class LeKiwiQuestIK:
    """Map Quest EE commands to joint targets via placo IK.

    ``relative_to_a`` (default, Fanuc VR app): MQTT x,y,z,w,p,r are already offsets from
    the last A-button zero — do not subtract a second reference frame.
    ``absolute_pair``: legacy mode, delta between latched reference pose and current pose.
    """

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        *,
        quest_pose_mode: QuestPoseMode = "relative_to_a",
        position_weight: float = 1.0,
        orientation_weight: float = 0.05,
        quest_position_scale: float = 0.5,
        ee_position_scale_mm: float = 0.001,
        max_delta_translation_m: float = 0.08,
        use_degrees: bool = False,
        calibration: dict[str, MotorCalibration] | None = None,
    ) -> None:
        path = Path(urdf_path) if urdf_path else default_so101_urdf_path()
        self.kinematics = RobotKinematics(path, joint_names=list(SO101_ARM_JOINT_NAMES))
        self.quest_pose_mode = quest_pose_mode
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.quest_position_scale = quest_position_scale
        self.ee_position_scale_mm = ee_position_scale_mm
        self.max_delta_translation_m = max_delta_translation_m
        self.use_degrees = use_degrees
        self.calibration = calibration or {}

        self._ref_position: tuple[float, float, float] | None = None
        self._ref_orientation: tuple[float, float, float] | None = None
        self._prev_a_pressed: bool = False
        self._armed: bool = False

    def reset(self) -> None:
        self._ref_position = None
        self._ref_orientation = None
        self._prev_a_pressed = False
        self._armed = False

    def _realign_reference(
        self, px: float, py: float, pz: float, rw: float, rp: float, rr: float
    ) -> None:
        self._ref_position = (px, py, pz)
        self._ref_orientation = (rw, rp, rr)

    def _on_a_button_edge(
        self,
        quest_action: dict[str, Any],
        px: float,
        py: float,
        pz: float,
        rw: float,
        rp: float,
        rr: float,
    ) -> bool:
        """A-button rising edge: Fanuc-style VR zero (arm teleop armed, no motion this frame)."""
        pressed = quest_a_button_pressed(quest_action)
        rising = pressed and not self._prev_a_pressed
        self._prev_a_pressed = pressed
        if rising:
            self._armed = True
            if self.quest_pose_mode == "absolute_pair":
                self._realign_reference(px, py, pz, rw, rp, rr)
            logger.info(
                "Quest A pressed — arm teleop armed (mode=%s, VR offsets from zero).",
                self.quest_pose_mode,
            )
            return True
        return False

    def _delta_matrix_from_quest(
        self, px: float, py: float, pz: float, rw: float, rp: float, rr: float
    ) -> np.ndarray:
        scale = self.quest_position_scale * self.ee_position_scale_mm
        return fanuc_pose_to_matrix(px, py, pz, rw, rp, rr, position_scale=scale)

    def _solve_relative_to_a(
        self,
        px: float,
        py: float,
        pz: float,
        rw: float,
        rp: float,
        rr: float,
        current_deg: np.ndarray,
    ) -> np.ndarray:
        """Quest values are already offset from A-zero; apply once on robot FK."""
        t_delta = self._delta_matrix_from_quest(px, py, pz, rw, rp, rr)
        delta_m = float(np.linalg.norm(t_delta[:3, 3]))
        if delta_m > self.max_delta_translation_m:
            logger.warning(
                "Quest offset %.3f m exceeds cap %.3f m — clamping (check mm vs m units).",
                delta_m,
                self.max_delta_translation_m,
            )
            if delta_m > 1e-9:
                t_delta[:3, 3] *= self.max_delta_translation_m / delta_m

        t_current = self.kinematics.forward_kinematics(current_deg)
        t_target = t_current @ t_delta
        return self.kinematics.inverse_kinematics(
            current_deg,
            t_target,
            position_weight=self.position_weight,
            orientation_weight=self.orientation_weight,
        )

    def _solve_absolute_pair(
        self,
        px: float,
        py: float,
        pz: float,
        rw: float,
        rp: float,
        rr: float,
        current_deg: np.ndarray,
    ) -> np.ndarray:
        if self._ref_position is None:
            self._realign_reference(px, py, pz, rw, rp, rr)
            return current_deg

        ref_x, ref_y, ref_z = self._ref_position
        ref_w, ref_p, ref_r = self._ref_orientation or (rw, rp, rr)
        t_ref = self._delta_matrix_from_quest(ref_x, ref_y, ref_z, ref_w, ref_p, ref_r)
        t_now = self._delta_matrix_from_quest(px, py, pz, rw, rp, rr)
        t_delta = t_now @ np.linalg.inv(t_ref)

        delta_m = float(np.linalg.norm(t_delta[:3, 3]))
        if delta_m > self.max_delta_translation_m:
            logger.warning(
                "Quest pose jump %.3f m — realigning reference.",
                delta_m,
            )
            self._realign_reference(px, py, pz, rw, rp, rr)
            return current_deg

        t_current = self.kinematics.forward_kinematics(current_deg)
        t_target = t_current @ t_delta
        return self.kinematics.inverse_kinematics(
            current_deg,
            t_target,
            position_weight=self.position_weight,
            orientation_weight=self.orientation_weight,
        )

    def solve(
        self,
        quest_action: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, float]:
        position = quest_action.get("position", {})
        rotation = quest_action.get("rotation", {})
        px = float(position.get("x", 0.0))
        py = float(position.get("y", 0.0))
        pz = float(position.get("z", 0.0))
        rw = float(rotation.get("w", 0.0))
        rp = float(rotation.get("p", 0.0))
        rr = float(rotation.get("r", 0.0))

        current_deg = observation_to_joint_degrees(
            observation, self.calibration, use_degrees=self.use_degrees
        )

        hold = joint_degrees_to_arm_action(
            current_deg, self.calibration, use_degrees=self.use_degrees
        )

        if self._on_a_button_edge(quest_action, px, py, pz, rw, rp, rr):
            return hold

        if self.quest_pose_mode == "relative_to_a" and not self._armed:
            return hold

        try:
            if self.quest_pose_mode == "relative_to_a":
                target_deg = self._solve_relative_to_a(px, py, pz, rw, rp, rr, current_deg)
            else:
                target_deg = self._solve_absolute_pair(px, py, pz, rw, rp, rr, current_deg)
        except Exception as exc:
            logger.warning("IK failed (%s); holding current joints.", exc)
            return hold

        return joint_degrees_to_arm_action(
            target_deg, self.calibration, use_degrees=self.use_degrees
        )
