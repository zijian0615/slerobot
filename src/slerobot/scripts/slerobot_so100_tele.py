"""
SO100 单臂遥操：Quest 3s MQTT → PyBullet IK → SOFollower。

MQTT 字段（与 Quest3sController 一致）：
  px/py/pz  Unity LH 手部位置（m），X=右 Y=上 Z=前
  qx/qy/qz/qw  Unity LH 四元数
  x/y/z, w/p/r  Fanuc TCP（mm / °），本脚本不使用
  triggerButton  >0.5 时启用位置跟踪并张开夹爪（与 telegrip VR 逻辑一致）

坐标映射与 IK 参考 telegrip/telegrip/core/robot_interface.py 与 telegrip/URDF/SO100。

Example:
```shell
python -m slerobot.scripts.slerobot_so100_tele \\
    --robot.port=/dev/ttyACM0 \\
    --mqtt_broker=10.22.9.10 \\
    --mqtt_topic=quest/data

# 依赖: pip install -e ".[pybullet]"
```
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from slerobot.configs import parser
from slerobot.robots.so100.config_so_follower import SOFollowerRobotConfig
from slerobot.robots.so100.so_follower import SOFollower
from slerobot.teleoperators.quest3s import Quest3sController
from slerobot.utils.lekiwi_ik import parse_quest_axis_remap
from slerobot.utils.rotation import Rotation
from slerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

# telegrip 常量（与 telegrip/config.py 保持一致）
NUM_JOINTS = 6
NUM_IK_JOINTS = 3
WRIST_FLEX_INDEX = 3
WRIST_ROLL_INDEX = 4
GRIPPER_INDEX = 5
GRIPPER_OPEN_ANGLE = 0.0
GRIPPER_CLOSED_ANGLE = 45.0
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
URDF_TO_INTERNAL_NAME_MAP = {
    "1": "shoulder_pan",
    "2": "shoulder_lift",
    "3": "elbow_flex",
    "4": "wrist_flex",
    "5": "wrist_roll",
    "6": "gripper",
}
END_EFFECTOR_LINK_NAME = "Fixed_Jaw_tip"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_telegrip_importable() -> None:
    telegrip_root = _repo_root() / "telegrip"
    if telegrip_root.is_dir():
        root_str = str(telegrip_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)


def _load_telegrip_module(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 telegrip 模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _import_telegrip_kinematics():
    """加载 telegrip IK，跳过 telegrip/__init__.py 与 core/__init__.py（避免 lerobot 依赖）。"""
    import types

    _ensure_telegrip_importable()
    pkg_root = _repo_root() / "telegrip" / "telegrip"
    if not pkg_root.is_dir():
        raise ImportError(f"未找到 telegrip 包目录: {pkg_root}")

    if "telegrip" not in sys.modules:
        telegrip_pkg = types.ModuleType("telegrip")
        telegrip_pkg.__path__ = [str(pkg_root)]
        sys.modules["telegrip"] = telegrip_pkg

    if "telegrip.core" not in sys.modules:
        core_pkg = types.ModuleType("telegrip.core")
        core_pkg.__path__ = [str(pkg_root / "core")]
        sys.modules["telegrip.core"] = core_pkg

    if "telegrip.config" not in sys.modules:
        config_mod = _load_telegrip_module("telegrip.config", pkg_root / "config.py")
        # 遥操只用当前姿态 IK，避免 reference_1 导致关节跳变
        config_mod.USE_REFERENCE_POSES = False
        config_mod.IK_HYSTERESIS_THRESHOLD = 999.0

    if "telegrip.utils" not in sys.modules:
        _load_telegrip_module("telegrip.utils", pkg_root / "utils.py")

    mod = _load_telegrip_module("telegrip.core.kinematics", pkg_root / "core" / "kinematics.py")
    return mod.ForwardKinematics, mod.IKSolver


def default_so100_urdf_path() -> Path:
    telegrip_urdf = _repo_root() / "telegrip" / "URDF" / "SO100" / "so100.urdf"
    if telegrip_urdf.is_file():
        return telegrip_urdf
    bundled = _repo_root() / "src" / "slerobot" / "assets" / "urdf" / "so101_kinematics.urdf"
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(
        "未找到 SO100 URDF。请确认 telegrip/URDF/SO100/so100.urdf 存在，或通过 --urdf_path 指定。"
    )


def unity_delta_to_robot(
    delta_unity: np.ndarray,
    remap: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Unity 位移 (m) → 机器人基座系。默认 remap=z,-x,y：Unity Z(前)→机器人 X(前)。"""
    return remap @ np.asarray(delta_unity, dtype=float) * scale


def compute_relative_position(
    current: dict[str, float],
    origin: dict[str, float],
    remap: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    delta_unity = np.array(
        [
            current["x"] - origin["x"],
            current["y"] - origin["y"],
            current["z"] - origin["z"],
        ],
        dtype=float,
    )
    return unity_delta_to_robot(delta_unity, remap, scale)


# 兼容旧名（telegrip 映射，Unity Z 前不会映射到机器人 X 前，勿用于 Quest 遥操）
def vr_to_robot_coordinates(vr_pos: dict[str, float], scale: float = 1.0) -> np.ndarray:
    legacy = np.array(
        [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
        dtype=float,
    )
    return unity_delta_to_robot(
        np.array([vr_pos["x"], vr_pos["y"], vr_pos["z"]], dtype=float),
        legacy,
        scale,
    )


def _extract_roll_deg(current_quat: np.ndarray, origin_quat: np.ndarray) -> float:
    relative = Rotation.from_quat(current_quat) * Rotation.from_quat(origin_quat).inv()
    return float(-np.degrees(relative.as_rotvec()[2]))


def _extract_pitch_deg(current_quat: np.ndarray, origin_quat: np.ndarray) -> float:
    relative = Rotation.from_quat(current_quat) * Rotation.from_quat(origin_quat).inv()
    return float(np.degrees(relative.as_rotvec()[0]))


def _unity_hand_position(quest_action: dict[str, Any]) -> dict[str, float] | None:
    """仅使用 Unity px/py/pz（米）。不用 Fanuc x/y/z（毫米），避免单位/坐标系混用。"""
    if all(k in quest_action for k in ("px", "py", "pz")):
        return {
            "x": float(quest_action["px"]),
            "y": float(quest_action["py"]),
            "z": float(quest_action["pz"]),
        }
    return None


def _unity_hand_pose_valid(hand_pos: dict[str, float] | None) -> bool:
    """拒绝 (0,0,0) 等未初始化手部数据。"""
    if hand_pos is None:
        return False
    return abs(hand_pos["x"]) + abs(hand_pos["y"]) + abs(hand_pos["z"]) > 0.05


def _latch_teleop_origin(
    state: SO100TeleopState,
    hand_pos: dict[str, float],
    hand_quat: np.ndarray | None,
    kinematics: SO100PyBulletIK,
    *,
    reason: str,
) -> None:
    state.origin_hand_pos = hand_pos.copy()
    state.origin_hand_quat = hand_quat.copy() if hand_quat is not None else None
    state.origin_robot_position = kinematics.forward_ee_position(state.actual_angles)
    state.origin_wrist_flex = float(state.actual_angles[WRIST_FLEX_INDEX])
    state.origin_wrist_roll = float(state.actual_angles[WRIST_ROLL_INDEX])
    state.target_position = state.origin_robot_position.copy()
    state.last_relative_delta = np.zeros(3, dtype=float)
    state.engaged = True
    state.arming = False
    logger.info(
        "锁定遥操原点 (%s) hand=%s ee=%s joints=%s",
        reason,
        (round(hand_pos["x"], 4), round(hand_pos["y"], 4), round(hand_pos["z"], 4)),
        np.round(state.origin_robot_position, 4).tolist(),
        np.round(state.actual_angles, 1).tolist(),
    )


def _unity_hand_quaternion(quest_action: dict[str, Any]) -> np.ndarray | None:
    if all(k in quest_action for k in ("qx", "qy", "qz", "qw")):
        return np.array(
            [
                float(quest_action["qx"]),
                float(quest_action["qy"]),
                float(quest_action["qz"]),
                float(quest_action["qw"]),
            ],
            dtype=float,
        )
    return None


def _trigger_active(quest_action: dict[str, Any]) -> bool:
    buttons = quest_action.get("buttons", {})
    raw = buttons.get("trigger", quest_action.get("triggerButton", 0))
    try:
        return float(raw) > 0.5
    except (TypeError, ValueError):
        return bool(raw)


def _arr3(v: np.ndarray | None) -> list[float] | None:
    if v is None:
        return None
    return [round(float(x), 4) for x in v]


@dataclass
class IkDebugInfo:
    target_ee: np.ndarray | None = None
    relative_delta: np.ndarray | None = None
    unity_delta: np.ndarray | None = None
    ik_joints_deg: np.ndarray | None = None
    fk_ee: np.ndarray | None = None
    fk_error_m: float | None = None
    fk_ee_cmd: np.ndarray | None = None
    fk_error_cmd_m: float | None = None
    ik_seed_source: str = ""


class SO100TeleopDebugger:
    """分通道调试输出，各通道独立限频。"""

    def __init__(self, enabled: bool, interval_s: float = 0.2):
        self.enabled = enabled
        self.interval_s = max(interval_s, 0.05)
        self._last_log: dict[str, float] = {}

    def _due(self, channel: str, force: bool = False) -> bool:
        if not self.enabled:
            return False
        if force:
            return True
        now = time.perf_counter()
        last = self._last_log.get(channel, 0.0)
        if now - last >= self.interval_s:
            self._last_log[channel] = now
            return True
        return False

    def log_mqtt(self, quest_action: dict[str, Any], *, force: bool = False) -> None:
        if not self._due("mqtt", force):
            return
        hand = _unity_hand_position(quest_action)
        quat = _unity_hand_quaternion(quest_action)
        trigger = _trigger_active(quest_action)
        pos = quest_action.get("position", {})
        rot = quest_action.get("rotation", {})
        parts = [f"trigger={trigger}"]
        if hand is None:
            parts.append("unity_hand(m)=<missing px/py/pz>")
        else:
            parts.append(f"unity_hand(m)=({hand['x']:+.4f},{hand['y']:+.4f},{hand['z']:+.4f})")
        if quat is not None:
            parts.append(
                f"quat=({quat[0]:+.4f},{quat[1]:+.4f},{quat[2]:+.4f},{quat[3]:+.4f})"
            )
        if any(k in quest_action for k in ("px", "py", "pz")):
            parts.append(
                f"raw_pxpyz=({quest_action.get('px')},{quest_action.get('py')},{quest_action.get('pz')})"
            )
        if pos:
            parts.append(
                f"fanuc_pos(mm)=({pos.get('x')},{pos.get('y')},{pos.get('z')})"
            )
        if rot:
            parts.append(f"fanuc_wpr(deg)=({rot.get('w')},{rot.get('p')},{rot.get('r')})")
        buttons = quest_action.get("buttons", {})
        if buttons:
            parts.append(f"buttons={buttons}")
        logger.info("[MQTT] %s", " | ".join(parts))

    def log_ik(
        self,
        *,
        engaged: bool,
        arming: bool,
        hand_pos: dict[str, float] | None,
        origin_hand: dict[str, float] | None,
        ik: IkDebugInfo,
        wrist_flex: float,
        wrist_roll: float,
        force: bool = False,
    ) -> None:
        if not self._due("ik", force):
            return
        if not engaged:
            if arming:
                logger.info("[IK] arming=True，等待/锁定原点中… hand=%s", hand_pos)
            else:
                logger.info("[IK] engaged=False（松开 trigger 或未开始跟踪）")
            return
        hand_s = (
            f"({hand_pos['x']:+.4f},{hand_pos['y']:+.4f},{hand_pos['z']:+.4f})"
            if hand_pos
            else "—"
        )
        origin_s = (
            f"({origin_hand['x']:+.4f},{origin_hand['y']:+.4f},{origin_hand['z']:+.4f})"
            if origin_hand
            else "—"
        )
        err_cmd = ik.fk_error_cmd_m if ik.fk_error_cmd_m is not None else float("nan")
        logger.info(
            "[IK] hand=%s origin=%s unity_d=%s delta_robot=%s target_ee=%s "
            "fk_ik=%s err_ik=%.4fm fk_cmd=%s err_cmd=%.4fm "
            "ik123=%s wrist=(flex=%.1f,roll=%.1f) seed=%s",
            hand_s,
            origin_s,
            _arr3(getattr(ik, "unity_delta", None)),
            _arr3(ik.relative_delta),
            _arr3(ik.target_ee),
            _arr3(ik.fk_ee),
            ik.fk_error_m if ik.fk_error_m is not None else float("nan"),
            _arr3(ik.fk_ee_cmd),
            err_cmd,
            _arr3(ik.ik_joints_deg),
            wrist_flex,
            wrist_roll,
            ik.ik_seed_source or "?",
        )

    def log_cmd(
        self,
        commanded: np.ndarray,
        *,
        target_ee: np.ndarray | None,
        engaged: bool,
        force: bool = False,
    ) -> None:
        if not self._due("cmd", force):
            return
        joints = {name: round(float(commanded[i]), 2) for i, name in enumerate(JOINT_NAMES)}
        logger.info(
            "[CMD] engaged=%s target_ee=%s joints_deg=%s",
            engaged,
            _arr3(target_ee),
            joints,
        )

    def log_ik_warning(self, message: str, *args: object, force: bool = False) -> None:
        if not self._due("ik_warn", force):
            return
        logger.warning(message, *args)

    def log_robot(
        self,
        actual: np.ndarray,
        commanded: np.ndarray,
        *,
        actual_ee: np.ndarray | None,
        commanded_ee: np.ndarray | None,
        tracking_err_deg: np.ndarray | None,
        force: bool = False,
    ) -> None:
        if not self._due("robot", force):
            return
        cmd_j = {name: round(float(commanded[i]), 2) for i, name in enumerate(JOINT_NAMES)}
        delta_j = {
            name: round(float(commanded[i] - actual[i]), 2) for i, name in enumerate(JOINT_NAMES)
        }
        logger.info(
            "[ROBOT] actual_joints=%s cmd_joints=%s delta_cmd-actual=%s",
            actual,
            cmd_j,
            delta_j,
        )
        logger.info(
            "[ROBOT] actual_ee=%s cmd_ee=%s ee_err=%s",
            _arr3(actual_ee),
            _arr3(commanded_ee),
            _arr3(
                (commanded_ee - actual_ee)
                if commanded_ee is not None and actual_ee is not None
                else None
            ),
        )
        if tracking_err_deg is not None:
            logger.info(
                "[ROBOT] joint_tracking_err(deg) max=%.2f %s",
                float(np.max(np.abs(tracking_err_deg))),
                {name: round(float(tracking_err_deg[i]), 2) for i, name in enumerate(JOINT_NAMES)},
            )


class SO100PyBulletIK:
    """单臂 PyBullet IK，封装 telegrip IKSolver / ForwardKinematics。"""

    def __init__(self, urdf_path: Path, use_gui: bool = False):
        try:
            import pybullet as p
            import pybullet_data
        except ImportError as exc:
            raise ImportError(
                "SO100 遥操需要 PyBullet。请安装: pip install -e \".[pybullet]\""
            ) from exc

        _ensure_telegrip_importable()
        ForwardKinematics, IKSolver = _import_telegrip_kinematics()

        self.p = p
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        if self.physics_client < 0:
            raise RuntimeError("PyBullet 连接失败")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        urdf_path = urdf_path.resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF 不存在: {urdf_path}")

        self.robot_id = p.loadURDF(str(urdf_path), [0.2, 0.0, 0.0], [0, 0, 0, 1], useFixedBase=1)
        self.joint_indices: list[int | None] = [None] * NUM_JOINTS
        self._map_joints()
        self.end_effector_link_index = self._find_end_effector()
        self.joint_limits_min_deg, self.joint_limits_max_deg = self._read_joint_limits()

        self.fk = ForwardKinematics(
            self.physics_client,
            self.robot_id,
            self.joint_indices,
            self.end_effector_link_index,
        )
        self.ik = IKSolver(
            self.physics_client,
            self.robot_id,
            self.joint_indices,
            self.end_effector_link_index,
            self.joint_limits_min_deg,
            self.joint_limits_max_deg,
            arm_name="left",
        )

    def _map_joints(self) -> None:
        p = self.p
        name_to_index: dict[str, int] = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            if info[2] != p.JOINT_FIXED:
                name_to_index[info[1].decode("UTF-8")] = i
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL, force=0)

        for urdf_name, internal_name in URDF_TO_INTERNAL_NAME_MAP.items():
            if internal_name in JOINT_NAMES and urdf_name in name_to_index:
                self.joint_indices[JOINT_NAMES.index(internal_name)] = name_to_index[urdf_name]

        missing = [name for i, name in enumerate(JOINT_NAMES) if self.joint_indices[i] is None]
        if missing:
            raise RuntimeError(f"URDF 关节映射失败，缺少: {missing}")

    def _find_end_effector(self) -> int:
        p = self.p
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            if info[12].decode("UTF-8") == END_EFFECTOR_LINK_NAME:
                return i
        raise RuntimeError(f"未找到末端连杆: {END_EFFECTOR_LINK_NAME}")

    def _read_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.p
        mins = np.full(NUM_JOINTS, -180.0)
        maxs = np.full(NUM_JOINTS, 180.0)
        for i in range(NUM_JOINTS):
            pb_index = self.joint_indices[i]
            if pb_index is None:
                continue
            info = p.getJointInfo(self.robot_id, pb_index)
            lower, upper = info[8], info[9]
            if lower < upper:
                mins[i] = math.degrees(lower)
                maxs[i] = math.degrees(upper)
        return mins, maxs

    def forward_ee_position(self, joint_angles_deg: np.ndarray) -> np.ndarray:
        position, _ = self.fk.compute(joint_angles_deg)
        return position

    def solve_position_ik(self, target_position: np.ndarray, current_angles_deg: np.ndarray) -> np.ndarray:
        return self.ik.solve(target_position, None, current_angles_deg)

    def disconnect(self) -> None:
        if self.p.isConnected(self.physics_client):
            self.p.disconnect(self.physics_client)


def clamp_joint_angles(
    joint_angles: np.ndarray,
    joint_limits_min_deg: np.ndarray,
    joint_limits_max_deg: np.ndarray,
) -> np.ndarray:
    processed = joint_angles.copy()
    shoulder_pan = processed[0]
    min_limit = joint_limits_min_deg[0]
    max_limit = joint_limits_max_deg[0]
    if shoulder_pan < min_limit or shoulder_pan > max_limit:
        for offset in (-360.0, 360.0):
            wrapped = shoulder_pan + offset
            if min_limit <= wrapped <= max_limit:
                processed[0] = wrapped
                break
    return np.clip(processed, joint_limits_min_deg, joint_limits_max_deg)


@dataclass
class SO100TeleopState:
    engaged: bool = False
    arming: bool = False
    trigger_prev: bool = False
    origin_hand_pos: dict[str, float] | None = None
    origin_hand_quat: np.ndarray | None = None
    origin_robot_position: np.ndarray | None = None
    origin_wrist_flex: float = 0.0
    origin_wrist_roll: float = 0.0
    target_position: np.ndarray | None = None
    last_relative_delta: np.ndarray | None = None
    last_unity_delta: np.ndarray | None = None
    wrist_flex: float = 0.0
    wrist_roll: float = 0.0
    arm_angles: np.ndarray = field(default_factory=lambda: np.array([0.0, -100.0, 100.0, 60.0, 0.0, 0.0]))
    actual_angles: np.ndarray = field(default_factory=lambda: np.array([0.0, -100.0, 100.0, 60.0, 0.0, 0.0]))
    last_ik_debug: IkDebugInfo = field(default_factory=IkDebugInfo)


@dataclass
class SO100TeleConfig:
    robot_port: str = "/dev/ttyACM0"
    robot_id: str = "so100_follower"
    use_degrees: bool = True
    max_relative_target: float | None = None
    mqtt_broker: str = "10.22.9.10"
    mqtt_port: int = 1883
    mqtt_topic: str = "quest/data"
    urdf_path: str | None = None
  # Unity LH: X右 Y上 Z前 → 机器人 X前 Y左 Z上（与 LeKiwi Quest 一致）
    quest_axis_remap: str = "z,-x,y"
    vr_to_robot_scale: float = 1.0
    deadzone_m: float = 0.003
    max_joint_step_deg: float = 12.0
    track_wrist_orientation: bool = False
    ik_warn_error_m: float = 0.08
    send_interval: float = 0.05
    fps: int = 30
    pybullet_gui: bool = False
    display_data: bool = False
    debug: bool = True
    debug_hz: float = 5.0
    ik_use_actual_joints: bool = True
    max_hand_delta_m: float = 0.25


def _clamp_delta(delta: np.ndarray, max_norm_m: float) -> np.ndarray:
    n = float(np.linalg.norm(delta))
    if max_norm_m > 0 and n > max_norm_m:
        return delta * (max_norm_m / n)
    return delta


def _wrap_angle_deg(angle: float, lo: float, hi: float) -> float:
    """将角度绕到 [lo, hi] 附近，避免 wrist_roll 飞到 300°+。"""
    width = hi - lo
    if width <= 0:
        return float(np.clip(angle, lo, hi))
    for _ in range(4):
        if lo <= angle <= hi:
            return angle
        if angle > hi:
            angle -= 360.0
        else:
            angle += 360.0
    return float(np.clip(angle, lo, hi))


def _observation_to_angles(observation: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            float(observation["shoulder_pan.pos"]),
            float(observation["shoulder_lift.pos"]),
            float(observation["elbow_flex.pos"]),
            float(observation["wrist_flex.pos"]),
            float(observation["wrist_roll.pos"]),
            float(observation["gripper.pos"]),
        ],
        dtype=float,
    )


def _angles_to_action(angles: np.ndarray) -> dict[str, float]:
    return {
        "shoulder_pan.pos": float(angles[0]),
        "shoulder_lift.pos": float(angles[1]),
        "elbow_flex.pos": float(angles[2]),
        "wrist_flex.pos": float(angles[3]),
        "wrist_roll.pos": float(angles[4]),
        "gripper.pos": float(angles[5]),
    }


def _clamp_joint_step(target: np.ndarray, current: np.ndarray, max_step_deg: float) -> np.ndarray:
    out = target.copy()
    for i in range(len(out)):
        d = float(target[i] - current[i])
        if abs(d) > max_step_deg:
            out[i] = current[i] + np.sign(d) * max_step_deg
    return out


def _update_from_mqtt(
    quest_action: dict[str, Any],
    state: SO100TeleopState,
    kinematics: SO100PyBulletIK,
    remap: np.ndarray,
    scale: float,
    max_hand_delta_m: float,
    deadzone_m: float,
    track_wrist_orientation: bool,
) -> bool:
    """返回 True 表示本帧更新了末端目标。"""
    trigger = _trigger_active(quest_action)
    hand_pos = _unity_hand_position(quest_action)
    hand_quat = _unity_hand_quaternion(quest_action)

    if trigger and not state.trigger_prev:
        state.arming = True
        state.engaged = False
        logger.info("Trigger 按下：等待有效 Unity 手部数据 (px/py/pz)...")

    if not trigger and state.trigger_prev:
        state.engaged = False
        state.arming = False
        state.origin_hand_pos = None
        state.origin_hand_quat = None
        state.origin_robot_position = None
        state.target_position = None
        logger.info("Trigger 松开：停止位置跟踪")

    state.trigger_prev = trigger
    state.arm_angles[GRIPPER_INDEX] = GRIPPER_OPEN_ANGLE if trigger else GRIPPER_CLOSED_ANGLE

    # 按住 trigger：收到有效 px/py/pz 后锁定原点（含启动时已按住 trigger 的情况）
    if trigger:
        if not state.engaged:
            state.arming = True
        if state.arming or not _unity_hand_pose_valid(state.origin_hand_pos):
            if _unity_hand_pose_valid(hand_pos):
                reason = "trigger" if state.arming else "re-latch-invalid-origin"
                _latch_teleop_origin(state, hand_pos, hand_quat, kinematics, reason=reason)
            elif hand_pos is None and state.arming:
                return False

    if not state.engaged or state.origin_hand_pos is None or state.origin_robot_position is None:
        return False
    if hand_pos is None:
        return False

    relative_delta = compute_relative_position(hand_pos, state.origin_hand_pos, remap, scale)
    if float(np.linalg.norm(relative_delta)) < deadzone_m:
        return False

    relative_delta = _clamp_delta(relative_delta, max_hand_delta_m)
    state.target_position = state.origin_robot_position + relative_delta
    state.last_relative_delta = relative_delta.copy()
    state.last_unity_delta = np.array(
        [
            hand_pos["x"] - state.origin_hand_pos["x"],
            hand_pos["y"] - state.origin_hand_pos["y"],
            hand_pos["z"] - state.origin_hand_pos["z"],
        ],
        dtype=float,
    )

    if track_wrist_orientation and hand_quat is not None and state.origin_hand_quat is not None:
        roll_delta = _extract_roll_deg(hand_quat, state.origin_hand_quat)
        pitch_delta = _extract_pitch_deg(hand_quat, state.origin_hand_quat)
        state.wrist_roll = state.origin_wrist_roll + (-roll_delta)
        state.wrist_flex = state.origin_wrist_flex + (-pitch_delta)
        state.wrist_roll = _wrap_angle_deg(
            state.wrist_roll,
            kinematics.joint_limits_min_deg[WRIST_ROLL_INDEX],
            kinematics.joint_limits_max_deg[WRIST_ROLL_INDEX],
        )
        state.wrist_flex = _wrap_angle_deg(
            state.wrist_flex,
            kinematics.joint_limits_min_deg[WRIST_FLEX_INDEX],
            kinematics.joint_limits_max_deg[WRIST_FLEX_INDEX],
        )
    else:
        # 仅位置遥操：手腕保持锁原点，避免腕部把末端拉偏导致 err 很大
        state.wrist_roll = state.origin_wrist_roll
        state.wrist_flex = state.origin_wrist_flex
    return True


def _angles_for_fk(
    ik_joints_deg: np.ndarray,
    wrist_flex: float,
    wrist_roll: float,
    gripper_angle: float,
) -> np.ndarray:
    angles = np.zeros(NUM_JOINTS, dtype=float)
    angles[:NUM_IK_JOINTS] = ik_joints_deg[:NUM_IK_JOINTS]
    angles[WRIST_FLEX_INDEX] = wrist_flex
    angles[WRIST_ROLL_INDEX] = wrist_roll
    angles[GRIPPER_INDEX] = gripper_angle
    return angles


def _apply_ik(
    state: SO100TeleopState,
    kinematics: SO100PyBulletIK,
    *,
    ik_use_actual_joints: bool,
    max_joint_step_deg: float,
    ik_warn_error_m: float,
) -> IkDebugInfo:
    info = IkDebugInfo()
    if not state.engaged or state.target_position is None:
        state.last_ik_debug = info
        return info

    info.target_ee = state.target_position.copy()
    info.relative_delta = (
        state.last_relative_delta.copy() if state.last_relative_delta is not None else None
    )
    info.unity_delta = (
        state.last_unity_delta.copy() if state.last_unity_delta is not None else None
    )

    if ik_use_actual_joints:
        ik_seed = state.actual_angles.copy()
        info.ik_seed_source = "actual"
    else:
        ik_seed = state.arm_angles.copy()
        info.ik_seed_source = "commanded"

    ik_angles = kinematics.solve_position_ik(state.target_position, ik_seed)
    info.ik_joints_deg = ik_angles.copy()

    ideal_angles = clamp_joint_angles(
        _angles_for_fk(
            ik_angles,
            state.wrist_flex,
            state.wrist_roll,
            state.arm_angles[GRIPPER_INDEX],
        ),
        kinematics.joint_limits_min_deg,
        kinematics.joint_limits_max_deg,
    )
    info.fk_ee = kinematics.forward_ee_position(ideal_angles)
    info.fk_error_m = float(np.linalg.norm(info.fk_ee - info.target_ee))

    state.arm_angles[:NUM_IK_JOINTS] = ik_angles
    state.arm_angles[WRIST_FLEX_INDEX] = state.wrist_flex
    state.arm_angles[WRIST_ROLL_INDEX] = state.wrist_roll
    state.arm_angles = clamp_joint_angles(
        state.arm_angles,
        kinematics.joint_limits_min_deg,
        kinematics.joint_limits_max_deg,
    )
    state.arm_angles = _clamp_joint_step(
        state.arm_angles,
        state.actual_angles,
        max_joint_step_deg,
    )
    state.arm_angles[GRIPPER_INDEX] = np.clip(
        state.arm_angles[GRIPPER_INDEX],
        GRIPPER_OPEN_ANGLE,
        GRIPPER_CLOSED_ANGLE,
    )

    info.fk_ee_cmd = kinematics.forward_ee_position(state.arm_angles)
    info.fk_error_cmd_m = float(np.linalg.norm(info.fk_ee_cmd - info.target_ee))
    state.last_ik_debug = info
    return info


def teleop_loop(
    controller: Quest3sController,
    robot: SOFollower,
    kinematics: SO100PyBulletIK,
    cfg: SO100TeleConfig,
) -> None:
    state = SO100TeleopState()
    observation = robot.get_observation()
    state.arm_angles = _observation_to_angles(observation)
    state.actual_angles = state.arm_angles.copy()

    dbg = SO100TeleopDebugger(cfg.debug, interval_s=1.0 / max(cfg.debug_hz, 0.1))
    period = 1.0 / cfg.fps
    logger.info(
        "SO100 遥操已启动。按住 trigger (>0.5) 跟踪手部运动；松开停止并闭合夹爪。"
    )
    remap = parse_quest_axis_remap(cfg.quest_axis_remap)
    logger.info("Quest 轴映射 %s → 机器人 [X前,Y左,Z上]", cfg.quest_axis_remap)
    if cfg.debug:
        logger.info(
            "调试日志已开启 (%.1f Hz)。通道: [MQTT] [IK] [CMD] [ROBOT]。"
            "关闭: --debug=false",
            cfg.debug_hz,
        )

    while True:
        loop_start = time.perf_counter()

        # 每帧先读实际关节（供 IK 初值与 [ROBOT] 对比）
        try:
            obs = robot.get_observation()
            state.actual_angles = _observation_to_angles(obs)
        except Exception as exc:
            logger.warning("[ROBOT] 读取实际关节失败: %s", exc)

        action = controller.get_action()
        if action is not None:
            force_log = False
            hand_pos = _unity_hand_position(action)
            trigger = _trigger_active(action)
            if trigger != state.trigger_prev:
                force_log = True

            dbg.log_mqtt(action, force=force_log)

            _update_from_mqtt(
                action,
                state,
                kinematics,
                remap,
                cfg.vr_to_robot_scale,
                cfg.max_hand_delta_m,
                cfg.deadzone_m,
                cfg.track_wrist_orientation,
            )
            if state.engaged:
                ik_info = _apply_ik(
                    state,
                    kinematics,
                    ik_use_actual_joints=cfg.ik_use_actual_joints,
                    max_joint_step_deg=cfg.max_joint_step_deg,
                    ik_warn_error_m=cfg.ik_warn_error_m,
                )
                if ik_info.fk_error_m is not None and ik_info.fk_error_m > cfg.ik_warn_error_m:
                    dbg.log_ik_warning(
                        "[IK] IK 解 FK 误差 %.3fm > %.3fm（腕部姿态或工作空间）",
                        ik_info.fk_error_m,
                        cfg.ik_warn_error_m,
                        force=force_log,
                    )
                elif (
                    ik_info.fk_error_cmd_m is not None
                    and ik_info.fk_error_m is not None
                    and ik_info.fk_error_cmd_m > cfg.ik_warn_error_m
                    and ik_info.fk_error_m <= cfg.ik_warn_error_m
                ):
                    dbg.log_ik_warning(
                        "[IK] 步进限幅后末端偏差 %.3fm（IK 可行，可增大 max_joint_step_deg）",
                        ik_info.fk_error_cmd_m,
                        force=force_log,
                    )
                robot.send_action(_angles_to_action(state.arm_angles))
            elif trigger:
                robot.send_action(_angles_to_action(state.arm_angles))
                ik_info = IkDebugInfo()
            else:
                ik_info = IkDebugInfo()

            dbg.log_ik(
                engaged=state.engaged,
                arming=state.arming,
                hand_pos=hand_pos,
                origin_hand=state.origin_hand_pos,
                ik=ik_info,
                wrist_flex=state.wrist_flex,
                wrist_roll=state.wrist_roll,
                force=force_log,
            )

            if state.engaged:
                dbg.log_cmd(
                    state.arm_angles,
                    target_ee=state.target_position,
                    engaged=True,
                    force=force_log,
                )
                actual_ee = kinematics.forward_ee_position(state.actual_angles)
                cmd_ee = kinematics.forward_ee_position(state.arm_angles)
                dbg.log_robot(
                    state.actual_angles,
                    state.arm_angles,
                    actual_ee=actual_ee,
                    commanded_ee=cmd_ee,
                    tracking_err_deg=state.arm_angles - state.actual_angles,
                    force=force_log,
                )

                if cfg.display_data and not cfg.debug:
                    logger.info(
                        "engaged=%s ee_cmd=%s ee_act=%s gripper=%.1f",
                        state.engaged,
                        np.round(cmd_ee, 4).tolist(),
                        np.round(actual_ee, 4).tolist(),
                        state.arm_angles[GRIPPER_INDEX],
                    )

        elapsed = time.perf_counter() - loop_start
        sleep_s = max(period - elapsed, 0.0)
        if sleep_s > 0:
            time.sleep(sleep_s)


@parser.wrap()
def teleoperate(cfg: SO100TeleConfig) -> None:
    init_logging()
    if cfg.debug:
        logging.getLogger().setLevel(logging.INFO)
    urdf_path = Path(cfg.urdf_path) if cfg.urdf_path else default_so100_urdf_path()

    controller = Quest3sController(
        mqtt_broker=cfg.mqtt_broker,
        mqtt_port=cfg.mqtt_port,
        mqtt_topic=cfg.mqtt_topic,
    )
    robot = SOFollower(
        SOFollowerRobotConfig(
            port=cfg.robot_port,
            id=cfg.robot_id,
            use_degrees=cfg.use_degrees,
            max_relative_target=cfg.max_relative_target,
        )
    )
    kinematics = SO100PyBulletIK(urdf_path=urdf_path, use_gui=cfg.pybullet_gui)

    logger.info("URDF: %s", urdf_path)
    logger.info("Robot port: %s", cfg.robot_port)
    logger.info("MQTT: %s:%s topic=%s", cfg.mqtt_broker, cfg.mqtt_port, cfg.mqtt_topic)

    controller.connect()
    robot.connect()

    try:
        teleop_loop(controller, robot, kinematics, cfg)
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        try:
            robot.bus.disable_torque()
        except Exception:
            pass
        robot.disconnect()
        controller.disconnect()
        kinematics.disconnect()
        logger.info("SO100 遥操已停止")


if __name__ == "__main__":
    teleoperate()
