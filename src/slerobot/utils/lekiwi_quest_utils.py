"""Map Quest 3s MQTT actions to LeKiwi arm + base command dicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ARM_JOINT_KEYS = (
    "arm_shoulder_pan.pos",
    "arm_shoulder_lift.pos",
    "arm_elbow_flex.pos",
    "arm_wrist_flex.pos",
    "arm_wrist_roll.pos",
    "arm_gripper.pos",
)


@dataclass
class LeKiwiQuestMapperConfig:
    position_scale_pan: float = 80.0
    position_scale_lift: float = 80.0
    position_scale_elbow: float = 60.0
    orientation_scale_wrist_flex: float = 40.0
    orientation_scale_wrist_roll: float = 40.0
    gripper_open: float = 0.0
    gripper_closed: float = 100.0
    joystick_xy_speed: float = 0.15
    joystick_theta_speed: float = 45.0


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


@dataclass
class LeKiwiQuestMapper:
    """Incremental Quest pose -> LeKiwi joint targets using the current observation as state."""

    config: LeKiwiQuestMapperConfig = field(default_factory=LeKiwiQuestMapperConfig)
    reference_position: tuple[float, float, float] | None = None
    reference_orientation: tuple[float, float, float] | None = None

    def reset(self) -> None:
        self.reference_position = None
        self.reference_orientation = None

    def map_action(
        self,
        quest_action: dict[str, Any],
        observation: dict[str, Any],
        base_action: dict[str, float] | None = None,
    ) -> dict[str, float]:
        position = quest_action.get("position", {})
        rotation = quest_action.get("rotation", {})
        buttons = quest_action.get("buttons", {})

        px = float(position.get("x", 0.0))
        py = float(position.get("y", 0.0))
        pz = float(position.get("z", 0.0))
        rw = float(rotation.get("w", 0.0))
        rp = float(rotation.get("p", 0.0))
        rr = float(rotation.get("r", 0.0))

        if self.reference_position is None:
            self.reference_position = (px, py, pz)
            self.reference_orientation = (rw, rp, rr)
            arm_action = {key: float(observation.get(key, 0.0)) for key in ARM_JOINT_KEYS}
        else:
            ref_x, ref_y, ref_z = self.reference_position
            ref_w, ref_p, ref_r = self.reference_orientation or (rw, rp, rr)
            dx, dy, dz = px - ref_x, py - ref_y, pz - ref_z
            dw, dp, dr = rw - ref_w, rp - ref_p, rr - ref_r

            arm_action = {
                "arm_shoulder_pan.pos": float(observation.get("arm_shoulder_pan.pos", 0.0))
                + dx * self.config.position_scale_pan,
                "arm_shoulder_lift.pos": float(observation.get("arm_shoulder_lift.pos", 0.0))
                + dy * self.config.position_scale_lift,
                "arm_elbow_flex.pos": float(observation.get("arm_elbow_flex.pos", 0.0))
                + dz * self.config.position_scale_elbow,
                "arm_wrist_flex.pos": float(observation.get("arm_wrist_flex.pos", 0.0))
                + dp * self.config.orientation_scale_wrist_flex,
                "arm_wrist_roll.pos": float(observation.get("arm_wrist_roll.pos", 0.0))
                + dr * self.config.orientation_scale_wrist_roll,
                "arm_gripper.pos": float(observation.get("arm_gripper.pos", 0.0)),
            }

        if int(buttons.get("grip", 0)):
            arm_action["arm_gripper.pos"] = self.config.gripper_closed
        else:
            arm_action["arm_gripper.pos"] = self.config.gripper_open

        merged_base = quest_base_action(quest_action, self.config)
        if base_action and base_action_is_active(base_action):
            merged_base = {**merged_base, **base_action}

        return {**arm_action, **merged_base}
