"""SO-ARM / LeKiwi placo kinematics (adapted from HuggingFace LeRobot)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import placo  # type: ignore[import-not-found]

_placo_runtime_error: ImportError | None = None

try:
    import placo  # type: ignore[import-not-found]
except ImportError as _placo_import_err:
    placo = None  # type: ignore[assignment]
    _placo_runtime_error = _placo_import_err


def _raise_if_placo_unusable() -> None:
    if placo is None and _placo_runtime_error is not None:
        raise ImportError(
            f"placo is required for Quest EE teleop. Install with: pip install 'slerobot[kinematics]'. "
            f"Import error: {_placo_runtime_error!s}"
        ) from _placo_runtime_error
    if placo is None:
        raise ImportError(
            "placo is required for Quest EE teleop. Install with: pip install 'slerobot[kinematics]'."
        )


def default_so101_urdf_path() -> Path:
    """Mesh-free URDF for placo (no STL assets required on Mac/Pi)."""
    urdf_dir = Path(__file__).resolve().parents[1] / "assets" / "urdf"
    kinematics_urdf = urdf_dir / "so101_kinematics.urdf"
    if kinematics_urdf.is_file():
        return kinematics_urdf
    return urdf_dir / "so101_new_calib.urdf"


# SO-101 URDF joint names (without LeKiwi `arm_` prefix).
SO101_ARM_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

LEKIWI_ARM_MOTOR_TO_URDF = {
    "arm_shoulder_pan": "shoulder_pan",
    "arm_shoulder_lift": "shoulder_lift",
    "arm_elbow_flex": "elbow_flex",
    "arm_wrist_flex": "wrist_flex",
    "arm_wrist_roll": "wrist_roll",
}


class RobotKinematics:
    """Forward / inverse kinematics via placo."""

    def __init__(
        self,
        urdf_path: str | Path,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ) -> None:
        _raise_if_placo_unusable()
        urdf_path = Path(urdf_path)
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self.robot = placo.RobotWrapper(str(urdf_path))
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)

        self.target_frame_name = target_frame_name
        self.joint_names = list(SO101_ARM_JOINT_NAMES) if joint_names is None else list(joint_names)
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, float(joint_pos_rad[i]))
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.05,
        iterations: int = 8,
    ) -> np.ndarray:
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, float(current_joint_rad[i]))

        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)
        for _ in range(iterations):
            self.tip_frame.T_world_frame = desired_ee_pose
            self.solver.solve(True)
            self.robot.update_kinematics()

        joint_pos_deg = np.rad2deg(
            [self.robot.get_joint(name) for name in self.joint_names]
        )
        if len(current_joint_pos) > len(self.joint_names):
            result = np.array(current_joint_pos, dtype=float, copy=True)
            result[: len(self.joint_names)] = joint_pos_deg
            return result
        return joint_pos_deg
