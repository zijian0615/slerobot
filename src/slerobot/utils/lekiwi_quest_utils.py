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

        if base_action:
            return {**arm_action, **base_action}

        joystick_x = float(quest_action.get("joystick_x", quest_action.get("joystickX", 0.0)))
        joystick_y = float(quest_action.get("joystick_y", quest_action.get("joystickY", 0.0)))
        if joystick_x != 0.0 or joystick_y != 0.0:
            return {
                **arm_action,
                "x.vel": joystick_y * self.config.joystick_xy_speed,
                "y.vel": joystick_x * self.config.joystick_xy_speed,
                "theta.vel": 0.0,
            }

        return {
            **arm_action,
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
        }
