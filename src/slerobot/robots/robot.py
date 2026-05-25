from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from slerobot.motors import MotorCalibration
from slerobot.utils.constants import HF_LEROBOT_CALIBRATION

logger = logging.getLogger(__name__)

ROBOTS = "robots"


class Robot(ABC):
    """
    Robot Abstract Base Class
    ========================
    定义所有机器人必须实现的接口。
    
    核心功能：
    - Lifecycle Management: connect, disconnect
    - Communication: send_action, get_observation
    - Status: is_connected
    """

    config_class: type | None = None
    name: str = ""

    def __init__(self, config: Any | None = None) -> None:
        if config is None:
            return
        self.config = config
        self.robot_type = getattr(self, "name", "")
        self.id = getattr(config, "id", "") or self.robot_type or "default"
        calibration_dir = getattr(config, "calibration_dir", None)
        self.calibration_dir = (
            Path(calibration_dir)
            if calibration_dir is not None
            else HF_LEROBOT_CALIBRATION / ROBOTS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        self.calibration: dict[str, MotorCalibration] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()

    def _load_calibration(self) -> None:
        with open(self.calibration_fpath, encoding="utf-8") as f:
            data = json.load(f)
        self.calibration = {
            name: MotorCalibration(**values) for name, values in data.items()
        }

    def _save_calibration(self) -> None:
        data = {
            name: {
                "id": cal.id,
                "drive_mode": cal.drive_mode,
                "homing_offset": cal.homing_offset,
                "range_min": cal.range_min,
                "range_max": cal.range_max,
            }
            for name, cal in self.calibration.items()
        }
        with open(self.calibration_fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @abstractmethod
    def connect(self) -> None:
        """
        连接到机器人并初始化所有子系统
        
        Raises:
            RuntimeError: 连接失败
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """断开与机器人的连接并清理资源"""
        pass

    @abstractmethod
    def send_action(self, action: Dict) -> Dict:
        """
        发送控制命令到机器人
        
        Args:
            action: 动作字典，包含控制信息
                - position: (x, y, z, w, p, r) 目标位姿
                - speed: 运动速度（可选）
                - term_type: 终止类型（可选）
                等等
        
        Returns:
            发送的动作字典
        
        Raises:
            RuntimeError: 如果机器人未连接
        """
        pass

    @abstractmethod
    def get_observation(self) -> Dict:
        """
        获取机器人的当前状态观测
        
        Returns:
            观测字典，包含：
            - position: (x, y, z, w, p, r) 当前位姿
            - timestamp: 观测时间戳
            - 其他传感器数据（如图像、力传感器等）
        
        Raises:
            RuntimeError: 如果机器人未连接
        """
        pass
    
    @property
    @abstractmethod
    def observation_features(self) -> dict:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Its structure (keys) should match the structure of what is returned by :pymeth:`get_observation`.
        Values for the dict should either be:
            - The type of the value if it's a simple value, e.g. `float` for single proprioceptive value (a joint's position/velocity)
            - A tuple representing the shape if it's an array-type value, e.g. `(height, width, channel)` for images

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        pass

    @property
    @abstractmethod
    def action_features(self) -> dict:
        """
        A dictionary describing the structure and types of the actions expected by the robot. Its structure
        (keys) should match the structure of what is passed to :pymeth:`send_action`. Values for the dict
        should be the type of the value if it's a simple value, e.g. `float` for single proprioceptive value
        (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """返回机器人是否已连接"""
        pass
