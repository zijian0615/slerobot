from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


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
