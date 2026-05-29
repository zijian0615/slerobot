"""
xArm 数据采集脚本
=================
参考 slerobot_record.py，针对 xArm 的通信特点做了以下精简：

  - 移除 Fanuc Ethernet/IP 专用逻辑（ACK 队列、pending 缓冲、encode_fanuc_pose_dict）
  - 移除 Fanuc 夹爪 LCB/DOUT 端口注入逻辑
  - xArm SDK 为同步通信，循环速率直接由 target_dt sleep 控制
  - 夹爪归一化值 j7 (0.0=张开, 1.0=闭合) 直接传入 XArmRobot.send_action()
  - Quest3s 笛卡尔遥操作结果通过 _teleop_to_xarm_action() 转换为 xArm 动作格式

旋转坐标系变换（Unity LH → xArm RH）：
  Unity 左手系：X=right, Y=up, Z=forward
  xArm 右手系：X=forward, Y=left, Z=up
  轴对应：xArm X ↔ Unity Z, xArm Y ↔ -Unity X, xArm Z ↔ Unity Y
  LH→RH 旋转方向取反，结合轴重排：
    xArm qx = -qz_u   (Z→X, LH→RH反向)
    xArm qy = +qx_u   (X→-Y, 两次取反相消)
    xArm qz = -qy_u   (Y→Z, LH→RH反向)

用法示例（笛卡尔模式 + Quest3s 遥操作）：
    python slerobot_xarm_record.py \
        --dataset.repo_id=zijian2022/xarm_demo \
        --dataset.single_task="pick and place" \
        --robot.robot_ip=192.168.1.204 \
        --robot.robot_dof=6 \
        --robot.robot_mode=7 \
        --robot.gripper_type=1 \
        --teleop.mqtt_broker=10.22.9.10

用法示例（回放策略）：
    python slerobot_xarm_record.py \
        --dataset.repo_id=my/xarm_demo \
        --dataset.single_task="pick and place" \
        --robot.robot_ip=192.168.1.29 \
        --policy.path=path/to/checkpoint
"""

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from slerobot.configs import parser
from slerobot.configs.policies import PreTrainedConfig
from slerobot.teleoperators import Teleoperator
from slerobot.teleoperators.quest3s import Quest3sController
from slerobot.robots import Robot
from slerobot.robots.xarm import XArmConfig, XArmRobot

from slerobot.utils.constants import ACTION, OBS_STR
from slerobot.datasets.slerobot_datasets import sLerobotDataset
from slerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from slerobot.datasets.video_utils import VideoEncodingManager

from slerobot.policies.pretrained import PreTrainedPolicy
from slerobot.policies.utils import make_robot_action
from slerobot.policies.act.modeling_act import ACTEigenCAMHelper, ACTPolicy, attention_cam_method_display_name
from slerobot.policies.factory import get_policy_class, make_pre_post_processors

from slerobot.utils.control_utils import init_keyboard_listener, is_headless, predict_action
from slerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from slerobot.utils.live_telemetry import push_live_telemetry
from slerobot.utils.visualization_utils import _init_rerun, log_rerun_data, shutdown_rerun

from slerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from slerobot.processor.converters import policy_action_to_transition, transition_to_policy_action


def _ui_telemetry_enabled() -> bool:
    return os.getenv("SLEROBOT_TELEMETRY_PUSH", "").lower() in ("1", "true", "yes")


# ======================================================================== #
#  配置数据类                                                                 #
# ======================================================================== #

@dataclass
class DatasetRecordConfig:
    """数据集录制配置。"""

    repo_id: str
    single_task: str
    root: str | None = None
    fps: int = 20
    episode_time_s: int | float = 60
    reset_time_s: int | float = 60
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = True
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_process: int = 0
    num_image_write_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    vcodec: str = "libsvtav1"
    streaming_encoding: bool = False
    encoder_queue_maxsize: int = 30
    encoder_threads: int = 2

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class XArmRobotConfig:
    """
    xArm 机器人连接配置。

    robot_mode:
        6  -> 关节伺服模式（低延迟，适合 GELLO 等关节遥操作）
        7  -> 笛卡尔在线轨迹规划（适合 Quest3s 笛卡尔遥操作）

    gripper_type:
        0  -> 无夹爪
        1  -> xArm Gripper
        2  -> xArm Gripper G2
        3  -> Bio Gripper G2
        10 -> Pika Gripper（需配合 gripper_port）
        11 -> Robotiq 2F-85
    """

    robot_ip: str = "192.168.1.127"
    robot_dof: int = 6
    robot_mode: int = 7
    # mode=7 遥操作时建议 300-500 mm/s；mode=6 关节伺服时单位为 deg/s
    robot_speed: float = 300.0
    robot_acc: float = 2000.0
    gripper_type: int = 0
    gripper_port: str | None = None
    gripper_speed: int = -1
    gripper_force: int = -1
    start_joints: tuple = (0.0, 0.0, -1.5708, 0.0, 1.5708, 0.0)
    # 设为 False 可阻止 connect() 时机械臂自动运动到 start_joints
    move_to_start_on_connect: bool = True
    cameras: dict[str, Any] | None = None

    # ---- 遥操作坐标映射（Delta Remapping）----
    # 解决不同机器人 TCP home 不同的问题：
    #   xarm_pos = robot_home + (teleop_pos - teleop_home)
    #
    # teleop_home_pos: Unity 脚本里 robotHome [x,y,z]（mm），即遥操作坐标系原点
    # robot_home_pos : xArm 实际 start_joints 对应的 TCP [x,y,z]（mm）
    #                  设为 None 则在第一帧自动从 get_observation() 读取
    # teleop_home_rot_deg: Unity 脚本里 robotHome [W,P,R]（度），遥操作旋转原点（暂未使用）
    # robot_home_rot_deg : xArm 对应旋转 [rx,ry,rz]（度），None = 自动读取
    #
    # 若不需要映射（遥操作坐标已匹配机器人）：保持全部为 None
    teleop_home_pos: tuple | None = (400.802, -33.385, -97.255)  # Unity 脚本里的 robotHome
    teleop_home_rot_deg: tuple | None = (160.115, 19.658, 142.221)
    robot_home_pos: tuple | None = None   # None = 自动从连接后第一帧 obs 读取
    robot_home_rot_deg: tuple | None = None  # None = 自动从连接后第一帧 obs 读取


@dataclass
class Quest3sConfig:
    """Quest 3s MQTT 遥操作配置。"""
    # CSI LAB: 10.22.9.10
    mqtt_broker: str = "10.22.55.72"
    mqtt_port: int = 1883
    mqtt_topic: str = "quest/data"


@dataclass
class RecordConfig:
    """xArm 录制总配置。"""

    dataset: DatasetRecordConfig
    robot: XArmRobotConfig = field(default_factory=XArmRobotConfig)
    teleop: Quest3sConfig | None = None
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    tts_voice: str | None = None
    tts_rate: int | None = None
    resume: bool = False
    enable_attention_visualization: bool = False
    realtime_attention_display: bool = False

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ======================================================================== #
#  四元数工具                                                                 #
# ======================================================================== #

def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法，格式均为 [qx, qy, qz, qw]。"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def _qconj(q: np.ndarray) -> np.ndarray:
    """单位四元数的共轭（即逆）。"""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qnorm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([0., 0., 0., 1.])


# Unity LH (X=right, Y=up, Z=forward) → xArm RH (X=forward, Y=left, Z=up)
# 与 lekiwi quest_axis_remap="z,-x,y" 一致，用于旋转矩阵相似变换。
_QUEST_TO_XARM = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=float)


def _unity_quat(qx_u: float, qy_u: float, qz_u: float, qw_u: float) -> np.ndarray:
    """归一化 Unity 原始四元数 [qx, qy, qz, qw]。"""
    return _qnorm(np.array([qx_u, qy_u, qz_u, qw_u], dtype=float))


def _q_to_mat3(q: np.ndarray) -> np.ndarray:
    """单位四元数 [qx, qy, qz, qw] → 3×3 旋转矩阵。"""
    q = _qnorm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _mat3_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 旋转矩阵 → 单位四元数 [qx, qy, qz, qw]。"""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return _qnorm(np.array([
            (m21 - m12) / s,
            (m02 - m20) / s,
            (m10 - m01) / s,
            0.25 * s,
        ]))
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return _qnorm(np.array([
            0.25 * s,
            (m01 + m10) / s,
            (m02 + m20) / s,
            (m21 - m12) / s,
        ]))
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return _qnorm(np.array([
            (m01 + m10) / s,
            0.25 * s,
            (m12 + m21) / s,
            (m02 - m20) / s,
        ]))
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return _qnorm(np.array([
        (m02 + m20) / s,
        (m12 + m21) / s,
        0.25 * s,
        (m10 - m01) / s,
    ]))


def _mat3_to_aa(R: np.ndarray) -> tuple[float, float, float]:
    """3×3 旋转矩阵 → xArm 轴角向量 [rx, ry, rz]（弧度）。"""
    return _quat_to_aa(_mat3_to_quat(R))


def _aa_to_mat3(rx: float, ry: float, rz: float) -> np.ndarray:
    """xArm 轴角向量 [rx, ry, rz]（弧度）→ 3×3 旋转矩阵。"""
    return _q_to_mat3(_aa_to_quat(rx, ry, rz))


def _unity_delta_quat_to_xarm(q_delta_u: np.ndarray) -> np.ndarray:
    """
    Unity 左手系增量四元数 → xArm 右手系增量四元数。
    Step1: LH→RH，翻转 qx、qz
    Step2: 轴重映射 Unity(X=right,Y=up,Z=fwd) → xArm(X=fwd,Y=left,Z=up)
    """
    qx, qy, qz, qw = q_delta_u
    # LH→RH: 翻转所有虚部分量（det=-1 变换，确保 Unity Y → xArm Z 正确取反）
    qx_rh, qy_rh, qz_rh = -qx, -qy, -qz
    # 轴重映射
    return _qnorm(np.array([
         qz_rh,   # xArm X ← Unity Z
        -qx_rh,   # xArm Y ← -Unity X
         qy_rh,   # xArm Z ← Unity Y
         qw,
    ]))


def _quest_local_rot_delta_to_xarm(q_cur_u: np.ndarray, q_home_u: np.ndarray) -> np.ndarray:
    q_delta_u = _qnorm(_qmul(_qconj(q_home_u), q_cur_u))
    q_delta_xarm = _unity_delta_quat_to_xarm(q_delta_u)

    return _q_to_mat3(q_delta_xarm)



def _aa_to_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    轴角旋转向量 [rx, ry, rz]（弧度，向量方向=旋转轴，模长=旋转角）→ 四元数 [qx, qy, qz, qw]。
    与 xArm get_position_aa / set_servo_cartesian_aa 的 _aa 格式完全一致。
    """
    angle = math.sqrt(rx*rx + ry*ry + rz*rz)
    if angle < 1e-9:
        return np.array([0., 0., 0., 1.])
    s = math.sin(angle / 2) / angle
    return np.array([rx * s, ry * s, rz * s, math.cos(angle / 2)])


def _quat_to_aa(q: np.ndarray) -> tuple:
    """
    四元数 [qx, qy, qz, qw] → 轴角旋转向量 [rx, ry, rz]（弧度）。
    与 xArm get_position_aa / set_servo_cartesian_aa 的 _aa 格式完全一致。
    旋转角规范化到 [0, π]（qw >= 0 分支），模长 ∈ [0, π]。
    """
    qx, qy, qz, qw = q
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    qw = max(-1.0, min(1.0, qw))
    angle = 2.0 * math.acos(qw)
    sin_half = math.sqrt(max(0.0, 1.0 - qw * qw))
    if sin_half < 1e-9:
        return 0.0, 0.0, 0.0
    return qx / sin_half * angle, qy / sin_half * angle, qz / sin_half * angle


# ======================================================================== #
#  xArm 动作格式转换                                                          #
# ======================================================================== #

def _teleop_to_xarm_action(
    act: dict,
    robot_mode: int,
    robot_dof: int,
    teleop_home_pos: tuple | None = None,
    teleop_home_rot_deg: tuple | None = None,
    robot_home_pos: tuple | None = None,
    robot_home_rot_deg: tuple | None = None,
    teleop_home_quat: np.ndarray | None = None,
) -> dict:
    """
    将 Quest3s 的输出动作转换为 XArmRobot.send_action() 需要的格式。

    Quest3s MQTT payload 字段（来自 Unity VRHandCubeTest.cs）：
        px,py,pz    : Quest 手柄世界坐标（Unity 坐标系，mm 级别）
        qx,qy,qz,qw : Quest 手柄旋转四元数（Unity 左手系，原始值）
        x,y,z       : 经 Unity 脚本计算的 Fanuc TCP 目标位置（mm）
        w,p,r       : 对应 Fanuc WPR 欧拉角（度）
        triggerButton, primaryButton, secondaryButton, gripButton

    Quest3sController.get_action() 将上述 payload 解析为：
        {"position": {"x","y","z"}, "rotation": {"w","p","r"},
         "buttons": {"trigger","grip","a"}, "qx","qy","qz","qw"}

    xArm 模式 7（笛卡尔）需要：j0(x), j1(y), j2(z), j3(rx), j4(ry), j5(rz), j7(可选)
    """
    xarm_act: dict = {}

    # 若上游 processor 已将 act 转成 j0…j5 格式，直接使用
    has_j_keys = any(f"j{i}" in act for i in range(max(robot_dof, 6)))
    if has_j_keys:
        for i in range(max(robot_dof, 6)):
            if f"j{i}" in act:
                xarm_act[f"j{i}"] = float(act[f"j{i}"])

    elif robot_mode in (1, 7):
        # ── 位置：从 Quest3s position 字段读取（即 Unity 脚本算好的 Fanuc TCP 坐标）──
        pos = act.get("position", {})
        if isinstance(pos, dict):
            raw_x = float(pos.get("x", 0.0))
            raw_y = float(pos.get("y", 0.0))
            raw_z = float(pos.get("z", 0.0))
        elif hasattr(pos, "__len__") and len(pos) >= 3:
            raw_x, raw_y, raw_z = float(pos[0]), float(pos[1]), float(pos[2])
        else:
            raw_x, raw_y, raw_z = 0.0, 0.0, 0.0

        # ── Delta Remapping：robot_pos = robot_home + (teleop_pos - teleop_home) ──
        if teleop_home_pos is not None and robot_home_pos is not None:
            tx, ty, tz = teleop_home_pos
            rhx, rhy, rhz = robot_home_pos
            xarm_act["j0"] = rhx + (raw_x - tx)
            xarm_act["j1"] = rhy + (raw_y - ty)
            xarm_act["j2"] = rhz + (raw_z - tz)
        else:
            xarm_act["j0"] = raw_x
            xarm_act["j1"] = raw_y
            xarm_act["j2"] = raw_z

        # ── 旋转：Unity 原始四元数 → 本地 delta → P 轴映射 → xArm 轴角 ──
        #
        # MQTT 中的 qx/qy/qz/qw 是 Quest 手柄原始 Unity 四元数（左手系）。
        # 不能用分量重排 (-qz, qx, -qy, qw) 代替坐标系变换：手柄 home 有倾角时
        # 会把绕前后/左右轴的旋转耦合成 rx/ry/rz 混合（表现为轴串扰）。
        qx_u = float(act.get("qx", 0.0))
        qy_u = float(act.get("qy", 0.0))
        qz_u = float(act.get("qz", 0.0))
        qw_u = float(act.get("qw", 1.0))
        has_valid_quat = (abs(qx_u) + abs(qy_u) + abs(qz_u) + abs(qw_u)) > 0.5
        q_cur_u = _unity_quat(qx_u, qy_u, qz_u, qw_u)

        print(f"[DBG] robot_home_rot_deg={robot_home_rot_deg}")
        # if has_valid_quat and teleop_home_quat is not None and robot_home_rot_deg is not None:
        #     rrx, rry, rrz = (math.radians(d) for d in robot_home_rot_deg)
        #     q_delta_u = _qnorm(_qmul(_qconj(teleop_home_quat), q_cur_u))
        #     q_delta_xarm = _unity_delta_quat_to_xarm(q_delta_u)   # ← 新函数
        #     R_delta_x = _q_to_mat3(q_delta_xarm)
        #     R_robot_home = _aa_to_mat3(rrx, rry, rrz)
        #     R_tgt = R_robot_home @ R_delta_x
        #     j3, j4, j5 = _mat3_to_aa(R_tgt)
        if has_valid_quat and teleop_home_quat is not None and robot_home_rot_deg is not None:
            rrx, rry, rrz = (math.radians(d) for d in robot_home_rot_deg)
            q_delta_u = _qnorm(_qmul(_qconj(teleop_home_quat), q_cur_u))
            q_delta_xarm = _unity_delta_quat_to_xarm(q_delta_u)
            R_delta_x = _q_to_mat3(q_delta_xarm)
            R_robot_home = _aa_to_mat3(rrx, rry, rrz)
            R_tgt = R_delta_x @ R_robot_home   # ← 改这里，顺序对调
            j3, j4, j5 = _mat3_to_aa(R_tgt)
        elif robot_home_rot_deg is not None:
            rrx, rry, rrz = robot_home_rot_deg
            j3, j4, j5 = math.radians(rrx), math.radians(rry), math.radians(rrz)
        else:
            j3, j4, j5 = 0.0, 0.0, 0.0

        # 轴角 → 6D 旋转（连续无跳变），供 send_action 和数据集使用
        from slerobot.robots.xarm.xarm import _aa_to_rot6d, _ROT6D_KEYS
        rot6d = _aa_to_rot6d(j3, j4, j5)
        for key, val in zip(_ROT6D_KEYS, rot6d):
            xarm_act[key] = val

    else:
        # 关节模式（mode=6）：teleop 应直接提供关节角（如 GELLO）
        logging.warning(
            "[xarm_record] Teleop action does not contain j0…j%d keys. "
            "For joint-space mode, use a joint-space teleop (e.g. GELLO). Skipping action.",
            robot_dof - 1,
        )
        return {}

    # 夹爪归一化值（0.0=张开, 1.0=闭合）
    if "j7" in act:
        xarm_act["j7"] = float(act["j7"])
    elif "buttons" in act:
        buttons = act["buttons"]
        if isinstance(buttons, dict):
            xarm_act["j7"] = float(bool(buttons.get("grip", 0)))

    return xarm_act


def _log_processor_pipeline(name: str, pipeline: PolicyProcessorPipeline | None) -> None:
    if pipeline is None:
        logging.info("[%s] None", name)
        return
    for idx, step in enumerate(pipeline.steps):
        config = getattr(step, "config", None) or {
            k: v for k, v in vars(step).items() if k not in {"_current_transition", "stats"}
        }
        logging.info("[%s] step=%d class=%s config=%s", name, idx, step.__class__.__name__, config)


# ======================================================================== #
#  核心录制循环                                                               #
# ======================================================================== #

def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    robot_mode: int = 7,
    robot_dof: int = 6,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
    dataset: sLerobotDataset | None = None,
    teleop: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
    record_data: bool = True,
    attention_visualization_enabled: bool = False,
    teleop_home_pos: tuple | None = None,
    teleop_home_rot_deg: tuple | None = None,
    robot_home_pos: tuple | None = None,
    robot_home_rot_deg: tuple | None = None,
    attention_output_dir: Path | str | None = None,
    realtime_attention_display: bool = False,
    play_sounds: bool = True,
    tts_voice: str | None = None,
    tts_rate: int | None = None,
) -> None:
    """
    xArm 录制主循环。

    与 Fanuc 版本的关键差异：
        - 无 ACK/pending 队列：xArm SDK 直接同步（或伺服）通信
        - 无 encode_fanuc_pose_dict：动作格式为 j0…j(n) + j7
        - 无夹爪 LCB/DOUT 元数据注入
        - 速率控制：每帧末尾 time.sleep(max(target_dt - elapsed, 0))
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"Dataset fps mismatch: {dataset.fps} != {fps}")
    if policy is not None and dataset is None:
        raise ValueError("A dataset is required when using a policy for recording.")
    if policy is not None:
        policy.reset()

    if teleop_action_processor is None or robot_action_processor is None or robot_observation_processor is None:
        default_teleop, default_robot_act, default_robot_obs = make_default_processors()
        teleop_action_processor = teleop_action_processor or default_teleop
        robot_action_processor   = robot_action_processor  or default_robot_act
        robot_observation_processor = robot_observation_processor or default_robot_obs

    target_dt = 1.0 / fps
    timestamp = 0.0
    start_episode_t = time.perf_counter()

    action_sent_count = 0
    no_action_count = 0
    rerun_step = 0
    last_diagnostic_time = time.perf_counter()
    diagnostic_interval_s = 1.0

    _robot_home_pos     = robot_home_pos
    _robot_home_rot_deg = robot_home_rot_deg

    # ── teleop_home_quat：首帧自动从 MQTT 读取，无需手动配置 ──
    # 保存 Unity 原始四元数；旋转 delta 在 _quest_local_rot_delta_to_xarm() 中做轴映射。
    _teleop_home_quat: np.ndarray | None = None

    # home 捕获后跳过若干帧，等待手势稳定，避免首帧大幅跳变
    _home_capture_skip = 0
    _HOME_SKIP_FRAMES  = 10   # 增加到 10 帧，给 servo 模式稳定更多时间

    while True:
        loop_start = time.perf_counter()
        if control_time_s is not None and timestamp >= control_time_s:
            break
        if events["exit_early"]:
            events["exit_early"] = False
            break

        # ── 获取观测 ──
        obs = robot.get_observation()
        obs = robot_observation_processor(obs)
        observation_frame = (
            build_dataset_frame(dataset.features, obs, prefix=OBS_STR) if dataset is not None else {}
        )

        # ── 计算动作 ──
        act_processed_policy: RobotAction | None = None
        act_processed_teleop: RobotAction | None = None

        if policy is not None and preprocessor is not None and postprocessor is not None:
            robot_type = getattr(robot, "robot_type", getattr(robot, "name", robot.__class__.__name__.lower()))
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot_type,
            )

            if isinstance(policy, ACTPolicy):
                warning_voice = policy.config.warning_speech_voice or tts_voice
                warning_rate  = policy.config.warning_speech_rate  or tts_rate
                policy.announce_attention_warnings(play_sounds=play_sounds, voice=warning_voice, rate=warning_rate)

            if (
                record_data
                and attention_visualization_enabled
                and isinstance(policy, ACTPolicy)
                and policy.last_attention_maps is not None
                and attention_output_dir is not None
            ):
                try:
                    import cv2
                    frame_idx  = int(timestamp * fps)
                    output_dir = Path(attention_output_dir) / "attention_maps" / f"frame_{frame_idx}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    for cam_idx, (image_key, attn_map) in enumerate(policy.last_attention_maps.items()):
                        camera_key = image_key.split(".")[-1]
                        if camera_key not in obs or attn_map is None:
                            continue
                        overlay_helper = (
                            policy.attention_overlay_helper
                            if isinstance(policy, ACTPolicy)
                            else ACTEigenCAMHelper
                        )
                        vis = overlay_helper.overlay_attention_on_image(
                            obs[camera_key], attn_map,
                            overlay_alpha=0.5, use_rgb=True,
                            edge_warning=policy.last_grad_cam_edge_warnings.get(image_key, False),
                            edge_margin_px=policy.config.grad_cam_edge_margin_px,
                            edge_warning_mean=policy.last_grad_cam_edge_stats.get(image_key),
                            edge_warning_threshold=policy.config.grad_cam_edge_mean_threshold,
                            edge_warning_roi_mode=policy.config.attention_roi_mode,
                        )
                        cv2.imwrite(
                            str(output_dir / f"camera_{cam_idx}_attention.png"),
                            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                        )
                except Exception as exc:
                    logging.warning("Failed to save attention visualization: %s", exc)

            act_processed_policy = make_robot_action(action_values, dataset.features)

        elif teleop is not None:
            raw_act = teleop.get_action()
            if raw_act is None:
                time.sleep(0.005)
                timestamp = time.perf_counter() - start_episode_t
                continue

            # ── 调试打印（每秒最多 1 次）──
            _mqtt_dbg_last = getattr(record_loop, "_mqtt_dbg_last", 0.0)
            if time.perf_counter() - _mqtt_dbg_last > 1.0:
                pos = raw_act.get("position", {})
                rot = raw_act.get("rotation", {})
                btn = raw_act.get("buttons", {})
                px, py, pz = pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)
                _th = teleop_home_pos
                _th_str = (
                    f"teleop_home=({_th[0]:+.1f},{_th[1]:+.1f},{_th[2]:+.1f})"
                    if _th else "teleop_home=None"
                )
                print(
                    f"[MQTT] pos=({px:+.1f},{py:+.1f},{pz:+.1f})  "
                    f"rot=({rot.get('w',0):+.1f},{rot.get('p',0):+.1f},{rot.get('r',0):+.1f})  "
                    f"trigger={btn.get('trigger',0)} grip={btn.get('grip',0)}  "
                    f"quat=({raw_act.get('qx',0):+.3f},{raw_act.get('qy',0):+.3f},"
                    f"{raw_act.get('qz',0):+.3f},{raw_act.get('qw',0):+.3f})  "
                    f"{_th_str}",
                    flush=True,
                )
                record_loop._mqtt_dbg_last = time.perf_counter()

            # ── 首帧自动记录 teleop_home 四元数（Unity 原始值）──
            if _teleop_home_quat is None:
                qx_h = float(raw_act.get("qx", 0.0))
                qy_h = float(raw_act.get("qy", 0.0))
                qz_h = float(raw_act.get("qz", 0.0))
                qw_h = float(raw_act.get("qw", 1.0))
                if abs(qx_h) + abs(qy_h) + abs(qz_h) + abs(qw_h) > 0.5:
                    _teleop_home_quat = _unity_quat(qx_h, qy_h, qz_h, qw_h)
                    _home_capture_skip = _HOME_SKIP_FRAMES
                    logging.info(
                        "[xArm Record] teleop_home_quat(Unity) 已从首帧读取："
                        "[%.4f, %.4f, %.4f, %.4f]，跳过前 %d 帧等待手势稳定",
                        *_teleop_home_quat, _HOME_SKIP_FRAMES,
                    )

            # home 刚捕获后的稳定跳过帧
            if _home_capture_skip > 0:
                _home_capture_skip -= 1
                timestamp = time.perf_counter() - start_episode_t
                elapsed   = time.perf_counter() - loop_start
                time.sleep(max(target_dt - elapsed, 0.0))
                continue

            act_processed_teleop = teleop_action_processor((raw_act, obs))

        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning("No policy or teleoperator provided, skipping action generation.")
            time.sleep(0.01)
            timestamp = time.perf_counter() - start_episode_t
            continue

        no_action_count = 0

        # ── 构造 xArm 动作 ──
        if policy is not None and act_processed_policy is not None:
            robot_action_to_send   = robot_action_processor((act_processed_policy, obs))
            action_values_for_dataset = dict(act_processed_policy)
        else:
            raw_teleop_action = dict(act_processed_teleop or {})
            xarm_action = _teleop_to_xarm_action(
                raw_teleop_action, robot_mode, robot_dof,
                teleop_home_pos=teleop_home_pos,
                teleop_home_rot_deg=teleop_home_rot_deg,
                robot_home_pos=_robot_home_pos,
                robot_home_rot_deg=_robot_home_rot_deg,
                teleop_home_quat=_teleop_home_quat,
            )
            if not xarm_action:
                timestamp = time.perf_counter() - start_episode_t
                elapsed   = time.perf_counter() - loop_start
                time.sleep(max(target_dt - elapsed, 0.0))
                continue
            robot_action_to_send      = robot_action_processor((xarm_action, obs))
            action_values_for_dataset = dict(robot_action_to_send)

        # ── 调试打印最终发送值（每秒最多 1 次）──
        _send_dbg_last = getattr(record_loop, "_send_dbg_last", 0.0)
        if time.perf_counter() - _send_dbg_last > 1.0:
            keys = sorted([k for k in robot_action_to_send if k.startswith("j")])
            vals = "  ".join(f"{k}={robot_action_to_send[k]:+.3f}" for k in keys)
            print(f"[SEND] {vals}", flush=True)
            record_loop._send_dbg_last = time.perf_counter()

        # ── 发送动作 ──
        robot.send_action(robot_action_to_send)
        action_sent_count += 1

        # ── 记录数据帧 ──
        if record_data and dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values_for_dataset, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        # ── 可视化 ──
        if display_data:
            attention_maps_for_obs          = None
            grad_cam_edge_warnings_for_obs  = None
            grad_cam_edge_stats_for_obs     = None

            if isinstance(policy, ACTPolicy) and policy.last_attention_maps is not None:
                attention_maps_for_obs         = {}
                grad_cam_edge_warnings_for_obs = {}
                grad_cam_edge_stats_for_obs    = {}
                for image_key, attn_map in policy.last_attention_maps.items():
                    camera_key = image_key.split(".")[-1]
                    if camera_key in obs:
                        attention_maps_for_obs[camera_key]         = attn_map
                        grad_cam_edge_warnings_for_obs[camera_key] = (
                            policy.last_grad_cam_edge_warnings.get(image_key, False)
                        )
                        if image_key in policy.last_grad_cam_edge_stats:
                            grad_cam_edge_stats_for_obs[camera_key] = (
                                policy.last_grad_cam_edge_stats[image_key]
                            )

            act_policy = policy if isinstance(policy, ACTPolicy) else None
            common_kwargs = dict(
                attention_maps=attention_maps_for_obs,
                grad_cam_edge_warnings=grad_cam_edge_warnings_for_obs,
                grad_cam_edge_stats=grad_cam_edge_stats_for_obs,
                grad_cam_edge_threshold=act_policy.config.grad_cam_edge_mean_threshold if act_policy else None,
                grad_cam_edge_margin_px=act_policy.config.grad_cam_edge_margin_px       if act_policy else None,
                grad_cam_edge_roi_mode=act_policy.config.attention_roi_mode              if act_policy else "mitigation",
                attention_overlay_helper=(
                    act_policy.attention_overlay_helper if act_policy else ACTEigenCAMHelper
                ),
            )

            if _ui_telemetry_enabled():
                push_live_telemetry(obs, robot_action_to_send, step=rerun_step, **common_kwargs)
            else:
                log_rerun_data(obs, robot_action_to_send, control_step=rerun_step, **common_kwargs)
            rerun_step += 1

        # ── 帧率控制 ──
        timestamp = time.perf_counter() - start_episode_t
        elapsed   = time.perf_counter() - loop_start
        time.sleep(max(target_dt - elapsed, 0.0))

        # ── 诊断输出 ──
        now = time.perf_counter()
        if now - last_diagnostic_time >= diagnostic_interval_s:
            actual_fps = action_sent_count / timestamp if timestamp > 0 else 0.0
            print(
                f"[xArm FPS] t={timestamp:.1f}s  actual={actual_fps:.2f}  "
                f"target={fps}  sent={action_sent_count}"
            )
            last_diagnostic_time = now


# ======================================================================== #
#  主入口                                                                     #
# ======================================================================== #

@parser.wrap()
def record(cfg: RecordConfig) -> sLerobotDataset:
    init_logging()

    # ── 构建 XArmRobot ──
    xarm_cfg = XArmConfig(
        robot_ip=cfg.robot.robot_ip,
        robot_dof=cfg.robot.robot_dof,
        robot_mode=cfg.robot.robot_mode,
        robot_speed=cfg.robot.robot_speed,
        robot_acc=cfg.robot.robot_acc,
        gripper_type=cfg.robot.gripper_type,
        gripper_port=cfg.robot.gripper_port,
        gripper_speed=cfg.robot.gripper_speed,
        gripper_force=cfg.robot.gripper_force,
        start_joints=tuple(cfg.robot.start_joints),
        move_to_start_on_connect=cfg.robot.move_to_start_on_connect,
        cameras=cfg.robot.cameras or {},
    )
    robot = XArmRobot(xarm_cfg)

    # ── 构建 Teleop ──
    teleop: Teleoperator | None = None
    if cfg.teleop is not None:
        teleop = Quest3sController(
            mqtt_broker=cfg.teleop.mqtt_broker,
            mqtt_port=cfg.teleop.mqtt_port,
            mqtt_topic=cfg.teleop.mqtt_topic,
        )

    # ── 构建 Policy ──
    policy = None
    preprocessor = None
    postprocessor = None
    attention_visualization_enabled = False

    if cfg.policy is not None:
        if cfg.policy.pretrained_path is None:
            raise ValueError("Policy config is missing `pretrained_path`. Pass `--policy.path=...`.")
        try:
            policy_cls = get_policy_class(cfg.policy.type)
            if cfg.enable_attention_visualization and cfg.policy.type == "act":
                cam_name     = attention_cam_method_display_name(cfg.policy.attention_cam_method)
                warning_label = cfg.policy.attention_camera or "all"
                logging.info(
                    "Enabling ACT %s on all cameras; ROI mode=%s on: %s",
                    cam_name, cfg.policy.attention_roi_mode, warning_label,
                )
                cfg.policy.enable_attention_visualization = True
                attention_visualization_enabled = True

            policy = policy_cls.from_pretrained(
                pretrained_name_or_path=cfg.policy.pretrained_path,
                config=cfg.policy,
            )
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=str(cfg.policy.pretrained_path),
            )
            _log_processor_pipeline("POLICY_PREPROCESSOR", preprocessor)
            _log_processor_pipeline("POLICY_POSTPROCESSOR", postprocessor)
        except Exception as exc:
            raise ValueError(f"Unsupported policy type: {cfg.policy.type}") from exc

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset  = None
    listener = None
    events   = None

    try:
        # ── 先连接机器人，相机初始化后才能获取正确的特征（含相机列）──
        robot.connect()

        # ── 数据集特征（connect 后读取，包含相机）──
        dataset_features = combine_feature_dicts(robot.observation_features, robot.action_features)
        num_cameras = len(robot.cameras) if robot.cameras else 1

        if cfg.resume:
            dataset = sLerobotDataset(
                repo_id=cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )
            if robot.cameras:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_process,
                    num_threads=cfg.dataset.num_image_write_threads_per_camera * len(robot.cameras),
                )
        else:
            dataset = sLerobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=getattr(robot, "name", "xarm"),
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_process,
                image_writer_threads=cfg.dataset.num_image_write_threads_per_camera * num_cameras,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        # robot_home：连接后立即读取（此时机器人在 start_joints 位置）
        resolved_robot_home_pos     = cfg.robot.robot_home_pos
        resolved_robot_home_rot_deg = cfg.robot.robot_home_rot_deg

        if (
            cfg.robot.teleop_home_pos is not None
            and resolved_robot_home_pos is None
            and cfg.robot.robot_mode in (1, 7)
        ):
            init_obs = robot.get_observation()
            resolved_robot_home_pos = (
                float(init_obs.get("j0", 0)),
                float(init_obs.get("j1", 0)),
                float(init_obs.get("j2", 0)),
            )
            import math as _math
            from slerobot.robots.xarm.xarm import _rot6d_to_aa, _ROT6D_KEYS
            if all(k in init_obs for k in _ROT6D_KEYS):
                # get_observation 返回 6D 旋转，转回轴角再转度数
                _rx, _ry, _rz = _rot6d_to_aa(
                    float(init_obs["r0"]), float(init_obs["r1"]), float(init_obs["r2"]),
                    float(init_obs["r3"]), float(init_obs["r4"]), float(init_obs["r5"]),
                )
            else:
                _rx = float(init_obs.get("j3", 0))
                _ry = float(init_obs.get("j4", 0))
                _rz = float(init_obs.get("j5", 0))
            resolved_robot_home_rot_deg = (
                _math.degrees(_rx), _math.degrees(_ry), _math.degrees(_rz),
            )
            logging.info(
                "[xArm Record] robot_home_pos=%s  robot_home_rot_deg=%s",
                resolved_robot_home_pos, resolved_robot_home_rot_deg,
            )

        if teleop is not None:
            try:
                teleop.connect()
                time.sleep(2.0)
            except ConnectionError as exc:
                logging.error(
                    "\n[xArm Record] 遥操作设备连接失败，录制终止。\n%s\n"
                    "提示：若只想回放策略（无需遥操作），请去掉 --teleop 参数。",
                    exc,
                )
                raise

        listener, events = init_keyboard_listener()

        if cfg.display_data and not _ui_telemetry_enabled():
            if os.environ.get("SLEROBOT_RERUN_CONNECT_ONLY") == "1":
                grpc_port = int(os.environ.get("SLEROBOT_RERUN_GRPC_PORT", "9876"))
                _init_rerun(session_name="slerobot_xarm_record", connect_ip="127.0.0.1", connect_port=grpc_port)
            else:
                _init_rerun(
                    session_name="slerobot_xarm_record",
                    connect_ip=cfg.display_ip,
                    connect_port=cfg.display_port,
                )

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. "
                "Consider enabling: --dataset.streaming_encoding=true --dataset.encoder_threads=2"
            )

        attention_output_dir = None
        if attention_visualization_enabled:
            attention_output_dir = (
                Path(cfg.dataset.root or "./") / cfg.dataset.repo_id / "attention_visualizations"
            )
            logging.info("Attention visualizations will be saved to: %s", attention_output_dir)

        # ── 录制循环 ──
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(
                    f"Recording episode {recorded_episodes + 1}",
                    cfg.play_sounds, voice=cfg.tts_voice, rate=cfg.tts_rate,
                )

                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    robot_mode=cfg.robot.robot_mode,
                    robot_dof=cfg.robot.robot_dof,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dataset=dataset,
                    teleop=teleop,
                    policy=policy,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=cfg.display_compressed_images,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    record_data=True,
                    attention_visualization_enabled=attention_visualization_enabled,
                    attention_output_dir=attention_output_dir,
                    realtime_attention_display=cfg.realtime_attention_display,
                    play_sounds=cfg.play_sounds,
                    tts_voice=cfg.tts_voice,
                    tts_rate=cfg.tts_rate,
                    teleop_home_pos=cfg.robot.teleop_home_pos,
                    teleop_home_rot_deg=cfg.robot.teleop_home_rot_deg,
                    robot_home_pos=resolved_robot_home_pos,
                    robot_home_rot_deg=resolved_robot_home_rot_deg,
                )

                need_reset = not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                )
                if need_reset:
                    log_say(
                        "Reset the environment",
                        cfg.play_sounds, voice=cfg.tts_voice, rate=cfg.tts_rate,
                    )
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        robot_mode=cfg.robot.robot_mode,
                        robot_dof=cfg.robot.robot_dof,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        dataset=dataset,
                        teleop=teleop,
                        policy=policy,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_compressed_images=cfg.display_compressed_images,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        record_data=False,
                        attention_visualization_enabled=attention_visualization_enabled,
                        attention_output_dir=attention_output_dir,
                        realtime_attention_display=cfg.realtime_attention_display,
                        play_sounds=cfg.play_sounds,
                        tts_voice=cfg.tts_voice,
                        tts_rate=cfg.tts_rate,
                        teleop_home_pos=cfg.robot.teleop_home_pos,
                        teleop_home_rot_deg=cfg.robot.teleop_home_rot_deg,
                        robot_home_pos=resolved_robot_home_pos,
                        robot_home_rot_deg=resolved_robot_home_rot_deg,
                    )

                if events["rerecord_episode"]:
                    log_say(
                        "Re-record episode",
                        cfg.play_sounds, voice=cfg.tts_voice, rate=cfg.tts_rate,
                    )
                    events["rerecord_episode"] = False
                    events["exit_early"]       = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True, voice=cfg.tts_voice, rate=cfg.tts_rate)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()

        if teleop is not None and getattr(teleop, "is_connected", False):
            teleop.disconnect()

        if listener is not None and not is_headless():
            listener.stop()

        if cfg.display_data and not _ui_telemetry_enabled():
            shutdown_rerun()

        if dataset is not None and cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        log_say("Exiting, goodbye!", cfg.play_sounds, voice=cfg.tts_voice, rate=cfg.tts_rate)

    return dataset


if __name__ == "__main__":
    record()