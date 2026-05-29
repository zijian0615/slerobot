import logging
import math
import struct
import time
from enum import IntEnum
from typing import Dict, Optional

import numpy as np

from slerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from .config_xarm import XArmConfig

logger = logging.getLogger(__name__)


class GripperType(IntEnum):
    NoGripper = 0
    xArmGripper = 1
    xArmGripperG2 = 2
    BioGripperG2 = 3
    PikaGripper = 10
    RobotiqGripper = 11


class XArmRobot(Robot):
    """
    xArm 机器人接口，实现 slerobot Robot 协议。

    支持两种运动模式：
        mode=6 (joint servo)    : action/observation 均为关节角度，键名 j0…j(dof-1) + j7(gripper)
        mode=7 (cartesian online): action/observation 为笛卡尔位姿 [x,y,z,rx,ry,rz] (radians)，
                                   键名 j0(x)…j5(rz) + j7(gripper)

    夹爪归一化值约定：0.0 = 完全张开，1.0 = 完全闭合
    """

    config_class = XArmConfig
    name = "xarm"

    def __init__(self, config: XArmConfig) -> None:
        super().__init__(config)
        self.config: XArmConfig = config

        self._dof = config.robot_dof
        self._mode = config.robot_mode
        self._gripper_type = GripperType(config.gripper_type)
        self._cmd_cnt = 0

        self._joint_speed = math.radians(config.robot_speed)
        self._joint_acc = math.radians(config.robot_acc)

        self._gripper_param = self._build_gripper_param()

        self._connected = False
        self.real_arm = None
        self.cameras: dict = {}

        # 保存上一次关节 / 笛卡尔观测（线程安全近似）
        self._latest_joints: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None  # [x,y,z,rx,ry,rz] in radians
        self._latest_t: Optional[float] = None
        self._latest_gripper_norm: float = 0.0

    # ------------------------------------------------------------------ #
    #  夹爪参数构建                                                          #
    # ------------------------------------------------------------------ #

    def _build_gripper_param(self) -> Dict:
        """返回夹爪参数字典 {open_pos, close_pos, speed, force}"""
        gt = self._gripper_type
        cfg = self.config

        if gt == GripperType.xArmGripper:
            speed = 5000 if cfg.gripper_speed < 0 else max(50, min(cfg.gripper_speed, 5000))
            force = 50 if cfg.gripper_force < 0 else cfg.gripper_force
            return dict(name="xArmGripper", open_pos=800, close_pos=0, speed=speed, force=force)

        elif gt == GripperType.xArmGripperG2:
            spd_deg = 225 if cfg.gripper_speed < 0 else max(15, min(cfg.gripper_speed, 225))
            speed = int(((spd_deg * 60) / 9.88235 + 140) / 0.4)
            force = 50 if cfg.gripper_force < 0 else max(1, min(cfg.gripper_force, 100))
            return dict(name="xArmGripperG2", open_pos=84, close_pos=0, speed=speed, force=force)

        elif gt == GripperType.BioGripperG2:
            speed = 2000 if cfg.gripper_speed < 0 else max(500, min(cfg.gripper_speed, 4500))
            force = 100 if cfg.gripper_force < 0 else max(1, min(cfg.gripper_force, 100))
            return dict(name="BioGripperG2", open_pos=150, close_pos=71, speed=speed, force=force)

        elif gt == GripperType.PikaGripper:
            speed = 0 if cfg.gripper_speed < 0 else cfg.gripper_speed
            force = 0 if cfg.gripper_force < 0 else cfg.gripper_force
            return dict(name="PikaGripper", open_pos=100, close_pos=0, speed=speed, force=force)

        elif gt == GripperType.RobotiqGripper:
            speed = 255 if cfg.gripper_speed < 0 else max(1, min(cfg.gripper_speed, 255))
            force = 255 if cfg.gripper_force < 0 else max(1, min(cfg.gripper_force, 255))
            return dict(name="RobotiqGripper", open_pos=0, close_pos=0xFF, speed=speed, force=force)

        else:
            return dict(name="NoGripper", open_pos=0, close_pos=0, speed=0, force=0)

    def _gripper_norm_to_pos(self, gripper_norm: float) -> int:
        open_p = self._gripper_param["open_pos"]
        close_p = self._gripper_param["close_pos"]
        pos = open_p + gripper_norm * (close_p - open_p)
        lo, hi = min(open_p, close_p), max(open_p, close_p)
        return int(min(max(lo, pos), hi))

    def _gripper_pos_to_norm(self, pos: Optional[int]) -> float:
        if pos is None:
            return self._latest_gripper_norm
        open_p = self._gripper_param["open_pos"]
        close_p = self._gripper_param["close_pos"]
        denom = open_p - close_p
        if denom == 0:
            return 0.0
        norm = (open_p - pos) / denom
        return float(np.clip(norm, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    #  连接 / 断开                                                          #
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        if self._connected:
            logger.warning("XArmRobot: already connected, skipping.")
            return

        try:
            from xarm.wrapper import XArmAPI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "xArm Python SDK not found. Install via: pip install xArm-Python-SDK"
            ) from exc

        logger.info("Connecting to xArm at %s …", self.config.robot_ip)
        self.real_arm = XArmAPI(self.config.robot_ip)
        time.sleep(0.2)

        if not self.real_arm.connected:
            raise ConnectionError(
                f"xArm connection failed. Check hardware at IP: {self.config.robot_ip}"
            )

        # 初始化 Pika 夹爪（如需）
        if self._gripper_type == GripperType.PikaGripper:
            from ufactory_devices.pika import PikaDevice  # type: ignore

            self._pika_device = PikaDevice(2, pika_gripper_port=self.config.gripper_port)
            self._pika_gripper = self._pika_device.pika_gripper

        # 初始使能序列（与 uf_robot.py 保持一致）
        self._clear_and_enable()

        if self.config.move_to_start_on_connect:
            logger.info("XArmRobot: moving to start_joints …")
            code = self.real_arm.set_servo_angle(
                angle=list(self.config.start_joints),
                is_radian=True,
                wait=True,
            )
            if code != 0:
                # 运动中再次触发错误（常见于 error 19），清除后继续
                logger.warning(
                    "Move to start_joints failed (code=%d), clearing error and retrying once.", code
                )
                self._clear_and_enable()
                code = self.real_arm.set_servo_angle(
                    angle=list(self.config.start_joints),
                    is_radian=True,
                    wait=True,
                )
                if code != 0:
                    logger.warning(
                        "Move to start_joints still failed (code=%d), skipping.", code
                    )
                    self._clear_and_enable()
        else:
            logger.info("XArmRobot: move_to_start_on_connect=False, skipping initial joint movement.")

        # enable=False：不重复调用 clean_error/motion_enable，只切目标模式并初始化夹爪
        self._robot_init(enable=False, init_gripper=True)
        self.real_arm.set_linear_spd_limit_factor(2.0)
        self.real_arm.set_collision_sensitivity(0)

        # SDK 的 error_code 是异步更新的，等待缓存值刷新为 0
        for _ in range(20):
            if self.real_arm.error_code == 0:
                break
            time.sleep(0.1)
        if self.real_arm.error_code != 0:
            logger.warning(
                "xArm error_code=%d still present after connect, "
                "will attempt auto-recovery in send_action.",
                self.real_arm.error_code,
            )

        self._connected = True

        # 初始化相机（若 record 脚本已提前 make_cameras_from_configs，则只 connect）
        if self.config.cameras:
            if not self.cameras:
                self.cameras = make_cameras_from_configs(self.config.cameras)
            for cam in self.cameras.values():
                cam.connect()

        logger.info("XArmRobot: connected (DOF=%d, mode=%d).", self._dof, self._mode)

    def _clear_and_enable(self) -> None:
        """清除控制器错误并使能机械臂，切换到 mode=0 就绪状态。"""
        self.real_arm.motion_enable()
        self.real_arm.clean_error()
        self.real_arm.clean_warn()
        self.real_arm.set_mode(0)
        self.real_arm.set_state(0)
        time.sleep(0.3)

    def disconnect(self) -> None:
        if not self._connected:
            return

        try:
            if self.real_arm is not None:
                self.real_arm.disconnect()
        except Exception as exc:
            logger.warning("Error during xArm disconnect: %s", exc)

        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception as exc:
                logger.warning("Error disconnecting camera: %s", exc)

        self.cameras = {}
        self._connected = False
        logger.info("XArmRobot: disconnected.")

    def _robot_init(self, enable: bool = True, init_gripper: bool = False) -> None:
        """切换到目标运动模式并初始化夹爪。

        enable=True  : 重新执行 clean_error / clean_warn / motion_enable（用于运行中恢复错误）
        enable=False : 跳过上述步骤，仅切模式+就绪状态（connect() 内首次调用时使用，
                       避免与 connect() 中的初始使能序列重复，防止触发 error code 19）
        """
        if enable:
            self.real_arm.clean_error()
            self.real_arm.clean_warn()
            self.real_arm.motion_enable(True)
        self.real_arm.set_mode(self._mode)
        self.real_arm.set_state(0)
        time.sleep(0.1)

        _, err_warn = self.real_arm.get_err_warn_code()
        if err_warn[0] != 0:
            # error 19 (emergency stop) 等可能在模式切换后短暂残留，尝试再次清除
            logger.warning("xArm error code=%d after mode switch, attempting to clear …", err_warn[0])
            self.real_arm.clean_error()
            self.real_arm.set_state(0)
            time.sleep(0.2)
            _, err_warn = self.real_arm.get_err_warn_code()
            if err_warn[0] != 0:
                raise RuntimeError(
                    f"xArm controller error persists after mode switch, code={err_warn[0]}. "
                    "Check hardware E-stop button or teach pendant."
                )

        if self._gripper_type == GripperType.NoGripper or not init_gripper:
            return

        self.real_arm._arm._baud_checkset = True
        gp = self._gripper_param

        if self._gripper_type == GripperType.xArmGripper:
            self.real_arm.set_gripper_enable(True)
            self.real_arm.set_gripper_mode(0)
            self.real_arm.set_gripper_speed(gp["speed"])
            if init_gripper:
                self.real_arm.set_gripper_position(gp["open_pos"])

        elif self._gripper_type == GripperType.xArmGripperG2:
            self.real_arm.set_gripper_enable(True)
            self.real_arm.set_gripper_mode(0)
            if init_gripper:
                self.real_arm.set_gripper_g2_position(gp["open_pos"])

        elif self._gripper_type == GripperType.BioGripperG2:
            _, mode = self.real_arm.get_bio_gripper_control_mode()
            if mode != 1:
                self.real_arm.set_bio_gripper_control_mode(1)
            self.real_arm.set_bio_gripper_enable(True)
            if init_gripper:
                self.real_arm.open_bio_gripper()

        elif self._gripper_type == GripperType.PikaGripper:
            self._pika_gripper.enable()
            time.sleep(0.5)
            if init_gripper:
                self._pika_gripper.set_gripper_distance(gp["open_pos"])

        elif self._gripper_type == GripperType.RobotiqGripper:
            self.real_arm.robotiq_reset()
            self.real_arm.robotiq_set_activate(wait=True)
            if init_gripper:
                self.real_arm.robotiq_set_position(gp["open_pos"], wait=True)

        self.real_arm._arm._baud_checkset = False

        _, err_warn = self.real_arm.get_err_warn_code()
        if err_warn[0] != 0:
            # 夹爪初始化后仍有错误（可能是 error 19 残留），尝试清除后继续
            logger.warning("xArm error code=%d after gripper init, attempting to clear …", err_warn[0])
            self.real_arm.clean_error()
            self.real_arm.set_state(0)
            time.sleep(0.2)
            _, err_warn = self.real_arm.get_err_warn_code()
            if err_warn[0] != 0:
                raise RuntimeError(
                    f"xArm gripper error persists after init, code={err_warn[0]}. "
                    "Check hardware E-stop button or teach pendant."
                )

        self._latest_gripper_norm = 0.0

    # ------------------------------------------------------------------ #
    #  核心接口：send_action / get_observation                              #
    # ------------------------------------------------------------------ #

    def send_action(self, action: Dict) -> Dict:
        """
        发送控制指令。

        action 字典格式（关节模式 mode=6）：
            {"j0": ..., "j1": ..., ..., "j(dof-1)": ..., "j7": gripper_norm}

        action 字典格式（笛卡尔模式 mode=7）：
            {"j0": x, "j1": y, "j2": z, "j3": rx, "j4": ry, "j5": rz, "j7": gripper_norm}
            单位：位置 mm，姿态 radians

        j7 (gripper_norm)：0.0 = 完全张开，1.0 = 完全闭合。可省略。
        """
        if not self._connected or self.real_arm is None:
            raise RuntimeError("XArmRobot is not connected. Call connect() first.")

        if self.real_arm.error_code != 0:
            # 限速自动恢复：每 2 秒尝试一次 clean_error + set_state
            now = time.perf_counter()
            last_recovery = getattr(self, "_last_error_recovery_t", 0.0)
            if now - last_recovery > 2.0:
                logger.warning(
                    "xArm error code=%d, attempting auto-recovery …",
                    self.real_arm.error_code,
                )
                self.real_arm.clean_error()
                self.real_arm.set_state(0)
                self._last_error_recovery_t = now
                time.sleep(0.1)
            if self.real_arm.error_code != 0:
                logger.error("xArm error code=%d persists, skipping action.", self.real_arm.error_code)
                return action

        gripper_norm: Optional[float] = action.get("j7", None)

        if self._mode == 6:
            joints = np.array(
                [float(action[f"j{i}"]) for i in range(self._dof)], dtype=np.float64
            )
            jnt_spd = 0.2 if self._cmd_cnt < 20 else self._joint_speed

            if self._cmd_cnt == 0:
                # 第一条指令用普通模式同步执行，确保机器人先到位再切伺服
                if self.real_arm.mode != 0:
                    self.real_arm.set_mode(0)
                    self.real_arm.set_state(0)
                    time.sleep(0.1)
                self.real_arm.set_servo_angle(
                    angle=joints.tolist(),
                    speed=jnt_spd,
                    mvacc=self._joint_acc,
                    is_radian=True,
                    wait=True,
                )
                self.real_arm.set_mode(6)
                self.real_arm.set_state(0)
                time.sleep(0.1)
            else:
                if self.real_arm.mode != 6:
                    self.real_arm.set_mode(6)
                    self.real_arm.set_state(0)
                    time.sleep(0.1)
                self.real_arm.set_servo_angle(
                    angle=joints.tolist(),
                    speed=jnt_spd,
                    mvacc=self._joint_acc,
                    is_radian=True,
                    wait=False,
                )

        elif self._mode == 7:
            pose = [float(action[f"j{i}"]) for i in range(6)]
            # 命令队列满时跳过本帧，避免积压导致 FPS 下降和跟手延迟
            cmd_num = getattr(self.real_arm, "cmd_num", 0)
            if cmd_num > 3:
                logger.debug("[xArm mode=7] cmd_num=%d > 3, skip frame to drain queue", cmd_num)
            else:
                self.real_arm.set_position_aa(
                    pose,
                    is_radian=True,
                    speed=self.config.robot_speed,
                    mvacc=self.config.robot_acc,
                    wait=False,
                )
        else:
            # mode=1: Cartesian servo（笛卡尔伺服，每帧直接覆盖目标位置，无轨迹队列）
            pose = [float(action[f"j{i}"]) for i in range(6)]

            if self.real_arm.mode != 1:
                # 切换到 Cartesian servo 模式
                self.real_arm.set_mode(1)
                self.real_arm.set_state(0)
                time.sleep(0.1)
                # 进入 servo 后，先以当前真实位置"原地踏步"若干帧
                # 防止 servo 高增益控制器因首帧微小偏差产生剧烈抖动
                _, cur_pose = self.real_arm.get_position_aa(is_radian=True)
                if cur_pose is not None:
                    for _ in range(5):
                        self.real_arm.set_servo_cartesian_aa(
                            cur_pose[:6],
                            is_radian=True,
                            speed=self.config.robot_speed,
                            mvacc=self.config.robot_acc,
                        )
                        time.sleep(0.01)
            self.real_arm.set_servo_cartesian_aa(
                pose,
                is_radian=True,
                speed=self.config.robot_speed,
                mvacc=self.config.robot_acc,
            )

        if self._cmd_cnt < 99999:
            self._cmd_cnt += 1

        self._send_gripper(gripper_norm)
        return action

    def _send_gripper(self, gripper_norm: Optional[float]) -> None:
        if self._gripper_type == GripperType.NoGripper or gripper_norm is None:
            return

        self._latest_gripper_norm = float(gripper_norm)
        grippos = self._gripper_norm_to_pos(gripper_norm)
        gp = self._gripper_param

        if self._gripper_type == GripperType.xArmGripper:
            modbus = [0x08, 0x10, 0x07, 0x00, 0x00, 0x02, 0x04]
            modbus.extend(list(struct.pack(">i", grippos)))
            self.real_arm.getset_tgpio_modbus_data(modbus)

        elif self._gripper_type == GripperType.xArmGripperG2:
            pos_raw = int(
                (math.degrees(math.asin((grippos - 16) / 110)) + 8.33) * 18.28
            )
            modbus = [0x08, 0x10, 0x0C, 0x00, 0x00, 0x05, 0x0A, 0x00, 0x01]
            modbus.extend(list(struct.pack(">h", gp["speed"])))
            modbus.extend(list(struct.pack(">h", gp["force"])))
            modbus.extend(list(struct.pack(">i", pos_raw)))
            self.real_arm.getset_tgpio_modbus_data(modbus)

        elif self._gripper_type == GripperType.BioGripperG2:
            pos_raw = int(grippos * 3.7342 - 265.13)
            modbus = [0x08, 0x10, 0x0C, 0x00, 0x00, 0x05, 0x0A, 0x00, 0x01]
            modbus.extend(list(struct.pack(">h", gp["speed"])))
            modbus.extend(list(struct.pack(">h", gp["force"])))
            modbus.extend(list(struct.pack(">i", pos_raw)))
            self.real_arm.getset_tgpio_modbus_data(modbus)

        elif self._gripper_type == GripperType.PikaGripper:
            self._pika_gripper.set_gripper_distance(grippos)

        elif self._gripper_type == GripperType.RobotiqGripper:
            modbus = [
                0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                0x09, 0x00, 0x00,
                grippos, gp["speed"], gp["force"],
            ]
            self.real_arm.getset_tgpio_modbus_data(modbus)

    def get_observation(self) -> Dict:
        """
        获取机器人当前状态。

        返回字典（关节模式 mode=6）：
            {"j0": ..., ..., "j(dof-1)": ..., "j7": gripper_norm, "timestamp": ...}

        返回字典（笛卡尔模式 mode=7）：
            {"j0": x, "j1": y, "j2": z, "j3": rx, "j4": ry, "j5": rz,
             "j7": gripper_norm, "timestamp": ...}
        """
        if not self._connected or self.real_arm is None:
            raise RuntimeError("XArmRobot is not connected. Call connect() first.")

        obs: Dict = {}

        if self._mode == 6:
            code, joints = self.real_arm.get_servo_angle(is_radian=True)
            if code != 0 or joints is None:
                logger.warning("get_servo_angle returned code=%d", code)
                joints = [0.0] * self._dof
            joints = list(joints[: self._dof])
            for i, v in enumerate(joints):
                obs[f"j{i}"] = float(v)
        else:
            code, pose = self.real_arm.get_position_aa(is_radian=True)
            if code != 0 or pose is None:
                logger.warning("get_position_aa returned code=%d", code)
                pose = [0.0] * 6
            for i, v in enumerate(pose[:6]):
                obs[f"j{i}"] = float(v)

        # 夹爪状态
        gripper_norm = self._read_gripper_norm()
        obs["j7"] = gripper_norm
        obs["timestamp"] = time.perf_counter()

        # 相机
        for cam_name, camera in self.cameras.items():
            frame = camera.read()
            if frame is not None:
                obs[cam_name] = frame

        return obs

    def _read_gripper_norm(self) -> float:
        """读取夹爪归一化开合度，失败时返回上一次记录值。"""
        try:
            if self._gripper_type == GripperType.xArmGripper:
                code, pos = self.real_arm.get_gripper_position()
                if code == 0 and pos is not None:
                    return self._gripper_pos_to_norm(int(pos))

            elif self._gripper_type == GripperType.xArmGripperG2:
                code, pos = self.real_arm.get_gripper_g2_position()
                if code == 0 and pos is not None:
                    return self._gripper_pos_to_norm(int(pos))

            elif self._gripper_type == GripperType.BioGripperG2:
                # Bio 夹爪无直接位置查询，返回上次缓存值
                pass

            elif self._gripper_type == GripperType.PikaGripper:
                pos = self._pika_gripper.get_gripper_distance()
                if pos is not None:
                    return self._gripper_pos_to_norm(int(pos))

            elif self._gripper_type == GripperType.RobotiqGripper:
                _, data = self.real_arm.robotiq_get_status()
                if data is not None:
                    pos = data.get("gPO", None)
                    if pos is not None:
                        return self._gripper_pos_to_norm(int(pos))

        except Exception as exc:
            logger.debug("Could not read gripper position: %s", exc)

        return self._latest_gripper_norm

    # ------------------------------------------------------------------ #
    #  特征描述（协议必要属性）                                               #
    # ------------------------------------------------------------------ #

    @property
    def observation_features(self) -> dict:
        if self._mode == 6:
            feats = {f"j{i}": float for i in range(self._dof)}
        else:
            feats = {f"j{i}": float for i in range(6)}

        feats["j7"] = float  # gripper norm

        if self.cameras:
            for cam_name, camera in self.cameras.items():
                h = camera.height if hasattr(camera, "height") and camera.height else 480
                w = camera.width if hasattr(camera, "width") and camera.width else 640
                feats[cam_name] = (h, w, 3)
        elif self.config.cameras:
            for cam_name, cam_cfg in self.config.cameras.items():
                if isinstance(cam_cfg, dict):
                    h = cam_cfg.get("height") or 480
                    w = cam_cfg.get("width") or 640
                else:
                    h = cam_cfg.height or 480
                    w = cam_cfg.width or 640
                feats[cam_name] = (h, w, 3)

        return feats

    @property
    def action_features(self) -> dict:
        if self._mode == 6:
            feats = {f"j{i}": float for i in range(self._dof)}
        else:
            feats = {f"j{i}": float for i in range(6)}

        feats["j7"] = float  # gripper norm

        return feats

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ #
    #  上下文管理器支持                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "XArmRobot":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()
