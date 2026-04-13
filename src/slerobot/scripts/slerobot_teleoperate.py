"""
Simple script to control a robot from teleoperation controller.

Example:

```shell
slerobot-teleoperate \
    --robot.type=fanuc \
    --robot.port=16001 \
    --robot.cameras="{ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=robot-2 \
    --teleop.type=quest3s \
    --teleop.port=1883 \
    --teleop.id=quest3s \
    --display_data=true
```
"""
# from datetime import datetime
# from slerobot.controller import TeleoperationController
# from slerobot.robot import Robot

# class TeleoperateConfig:
# def teleop_loop(self, teleop, robot):
#     """
#     This function continuously reads actions from a teleoperation device, sends them to a robot, 
#     and optionally displays the robot's state. The loop runs at a
#     specified frequency until a set duration is reached or it is manually interrupted.
#     Args:
#         teleop: The teleoperator device instance providing control actions.
#         robot: The robot instance being controlled
#         fps: int  
#         duration: Optional[float] - Maximum duration to run the loop in seconds. If None, runs indefinitely.
#         display_data: bool - Whether to display the robot's state and actions in each loop
#     """
# 
#       start = time.perf_counter()
#       while True:
#           loop_start = time.perf_counter()
#           obs = robot.get_observation()  # Get initial observation
#           actions = teleop.get_action(obs)  # Get action from teleoperation controller 
#           _ = robot.send_action(actions)  # Send action to robot   
#           if self.display_data:
#               self.display(obs, actions)  # Optionally display data
#           dt_s = time.perf_counter() - loop_start

#          precise_sleep(max(1 / fps - dt_s, 0.0))
#          loop_s = time.perf_counter() - loop_start
#          print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
#          move_cursor_up(1)

#.         if duration is not None and time.perf_counter() - start >= duration:
#            return
#
#
# def teleoperate:
#   1. Initialize teleoperation controller and robot instances based on configuration.
#.     init_logging()
#.     contorller = TeleoperationController(config.teleop) 
#      robot = Robot(config.robot)
#.     
#      controller.connect()  # Connect to teleoperation device (e.g., MQTT broker)
#      robot.connect()  # Connect to robot (e.g., open TCP connection)
#.     
#      try:
#           teleop_loop(controller, robot)  # Start the teleoperation loop
#.     except KeyboardInterrupt:
#.         pass
#
#      robot.disconnect()  # Clean up robot connection
#      controller.disconnect()  # Clean up teleoperation controller connection
#
#
#
# def main():
#   teleoperate()
# teleoperate.py
import sys
import time
import logging
import dataclasses
from typing import Optional

import draccus

sys.path.insert(0, '../')
from robots.fanuc.fanuc import Fanuc
from teleoperators.quest3s import Quest3sController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 配置类 ──────────────────────────────────────────

@dataclasses.dataclass
class FanucConfig:
    type: str = "fanuc"
    host: str = "172.30.109.22"
    port: int = 16001
    speed: int = 250
    term_type: str = "CNT"
    term_value: int = 100
    id: str = "robot-1"


@dataclasses.dataclass
class Quest3sConfig:
    type: str = "quest3s"
    broker: str = "10.22.9.10"
    port: int = 1883
    topic: str = "quest/data"
    id: str = "quest3s"


@dataclasses.dataclass
class TeleoperateConfig:
    robot: FanucConfig = dataclasses.field(default_factory=FanucConfig)
    teleop: Quest3sConfig = dataclasses.field(default_factory=Quest3sConfig)
    fps: int = 30
    duration: Optional[float] = None
    display_data: bool = False
    #dry_run: bool = False


# ── 核心逻辑 ─────────────────────────────────────────

def precise_sleep(duration: float) -> None:
    if duration <= 0:
        return
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        remaining = deadline - time.perf_counter()
        if remaining > 0.001:
            time.sleep(remaining * 0.5)


def teleop_loop(controller, robot, cfg: TeleoperateConfig):
    BUFFER_SIZE = 2
    MIN_DIST_MM = 0.5
    INTER_PACKET_DELAY = 0.002

    pending: set = set()
    last_sent_pose = None
    buffer_full = False
    start = time.perf_counter()
    

    while True:
        t_start = time.perf_counter()

        # 1. 消费已完成的 ACK
        while True:
            seq_id, err = robot.check_ack()
            if seq_id is None:
                break
            pending.discard(seq_id)
            if err != 0:
                logger.warning(f"seq {seq_id} ErrorID={err}")

        # 2. buffer 满则等待
        if len(pending) >= BUFFER_SIZE:
            time.sleep(0.001)
            continue

        # 3. 获取控制器数据
        action = controller.get_action()
        if action is None:
            time.sleep(0.001)
            continue

        target_pose = (
            action["position"]["x"],
            action["position"]["y"],
            action["position"]["z"],
            action["rotation"]["w"],
            action["rotation"]["p"],
            action["rotation"]["r"],
        )

        # 4. 空间滤波
        if last_sent_pose is not None:
            dx = target_pose[0] - last_sent_pose[0]
            dy = target_pose[1] - last_sent_pose[1]
            dz = target_pose[2] - last_sent_pose[2]
            if (dx**2 + dy**2 + dz**2) ** 0.5 < MIN_DIST_MM:
                time.sleep(0.001)
                continue
        
        # 5. 发送
        fut = robot.send_action({
            "position": target_pose,
            "speed": cfg.robot.speed,
            "term_type": cfg.robot.term_type,
            "term_value": cfg.robot.term_value,
        })
        seq_id = robot.seq_id - 1
        pending.add(seq_id)
        last_sent_pose = target_pose

        if not buffer_full and len(pending) >= BUFFER_SIZE:
            time.sleep(INTER_PACKET_DELAY)

        if cfg.display_data:
            print(f"pose={target_pose}  pending={len(pending)}")

        loop_s = time.perf_counter() - t_start
        print(f"\rLoop: {loop_s*1e3:.1f}ms ({1/loop_s:.0f} Hz) pending={len(pending)}  ", end="", flush=True)

        if cfg.duration is not None and time.perf_counter() - start >= cfg.duration:
            return


@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    controller = Quest3sController(
        mqtt_broker=cfg.teleop.broker,
        mqtt_port=cfg.teleop.port,
        mqtt_topic=cfg.teleop.topic,
    )
    robot = Fanuc(
        host=cfg.robot.host,
        port=cfg.robot.port,
        speed=cfg.robot.speed,
        term_type=cfg.robot.term_type,
        term_value=cfg.robot.term_value,
    )

    controller.connect()
    robot.connect()
    robot.get_observation()

    #logger.info(f"Teleop started {'[DRY RUN]' if cfg.dry_run else ''}")
    try:
        teleop_loop(controller, robot, cfg)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        robot.disconnect()
        controller.disconnect()
        logger.info("Teleop stopped")


if __name__ == "__main__":
    teleoperate()