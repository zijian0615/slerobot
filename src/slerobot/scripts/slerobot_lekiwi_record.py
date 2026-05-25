"""
Record LeKiwi demonstrations with Quest 3s (arm) + keyboard (base).

Example:
```shell
python -m slerobot.scripts.slerobot_lekiwi_record \
    --dataset.repo_id=user/lekiwi_quest \
    --dataset.single_task="pick and place" \
    --robot.remote_ip=10.22.26.30  \
    --teleop.mqtt_broker=10.22.9.10 \
    --teleop.mqtt_topic=quest/data

# Pi runs without cameras:
#   python -m slerobot.robots.lekiwi.lekiwi_host --no-cameras
# Mac record (default robot.use_cameras=false):
#   pip install -e ".[kinematics]"   # placo IK: Quest EE -> arm joints
#   ... same command, no extra flags
# Lower arm sensitivity (hand moves less on robot):
#   --quest_map.position_scale=0.5
#   --quest_map.orientation_weight=0.02
#   --quest_map.max_joint_step_deg=8
#   --quest_map.quest_delta_ema_alpha=0.4
# Copy Pi motor calibration to Mac (same joint ranges for IK):
#   ~/.cache/huggingface/lerobot/calibration/robots/lekiwi/<pi-id>.json
#   -> .../robots/lekiwi_client/lekiwi_client.json
# With cameras on both sides:
#   add --robot.use_cameras=true
#
# Debug send_action (Mac + Pi):
#   export SLEROBOT_DEBUG_ACTION=1
#   export SLEROBOT_DEBUG_ACTION_HZ=5   # optional, max prints per second
```
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from slerobot.configs import parser
from slerobot.datasets.slerobot_datasets import sLerobotDataset
from slerobot.datasets.utils import combine_feature_dicts
from slerobot.datasets.video_utils import VideoEncodingManager
from slerobot.processor import make_default_processors
from slerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig, lekiwi_cameras_config
from slerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from slerobot.scripts.slerobot_record import (
    Quest3sConfig,
    record_loop,
)
from slerobot.teleoperators.lekiwi_keyboard import LeKiwiKeyboardTeleop
from slerobot.teleoperators.quest3s import Quest3sController
from slerobot.utils.control_utils import init_keyboard_listener, is_headless
from slerobot.utils.lekiwi_quest_utils import LeKiwiQuestMapper, LeKiwiQuestMapperConfig
from slerobot.utils.utils import init_logging, log_say
from slerobot.utils.visualization_utils import _init_rerun, shutdown_rerun


def _ui_telemetry_enabled() -> bool:
    return os.getenv("SLEROBOT_TELEMETRY_PUSH", "").lower() in ("1", "true", "yes")


@dataclass
class LeKiwiDatasetRecordConfig:
    """Dataset settings for LeKiwi Quest recording (`single_task` has a default)."""

    repo_id: str
    single_task: str = "pick and place"
    root: str | None = None
    fps: int = 20
    episode_time_s: int | float = 60
    reset_time_s: int | float = 60
    num_episodes: int = 50
    video: bool = False
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_process: int = 0
    num_image_write_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    vcodec: str = "libsvtav1"
    streaming_encoding: bool = False
    encoder_queue_maxsize: int = 30
    encoder_threads: int = 2


@dataclass
class LeKiwiRobotRecordConfig:
    """Plain dataclass for draccus CLI (same pattern as FanucConfig in slerobot_record)."""

    remote_ip: str = "127.0.0.1"
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    polling_timeout_ms: int = 100
    connect_timeout_s: int = 10
    id: str = "lekiwi_client"
    # Set false when Pi host runs with `--no-cameras` (proprio-only dataset).
    use_cameras: bool = False

    def to_client_config(self) -> LeKiwiClientConfig:
        cameras = lekiwi_cameras_config() if self.use_cameras else {}
        return LeKiwiClientConfig(
            id=self.id,
            remote_ip=self.remote_ip,
            port_zmq_cmd=self.port_zmq_cmd,
            port_zmq_observations=self.port_zmq_observations,
            polling_timeout_ms=self.polling_timeout_ms,
            connect_timeout_s=self.connect_timeout_s,
            cameras=cameras,
        )


@dataclass
class LeKiwiQuestMapConfig:
    """Quest → arm IK sensitivity (lower = smaller arm motion per hand move)."""

    # MQTT position is mm; effective meters = mm * ee_position_scale_mm * position_scale.
    position_scale: float = 0.5
    orientation_weight: float = 0.05
    max_joint_step_deg: float = 5.0
    quest_delta_ema_alpha: float = 0.55
    joint_output_alpha: float = 0.4
    position_deadzone_mm: float = 1.5
    rotation_deadzone_deg: float = 1.5


@dataclass
class LeKiwiRecordConfig:
    dataset: LeKiwiDatasetRecordConfig
    robot: LeKiwiRobotRecordConfig = field(default_factory=LeKiwiRobotRecordConfig)
    teleop: Quest3sConfig = field(default_factory=Quest3sConfig)
    quest_map: LeKiwiQuestMapConfig = field(default_factory=LeKiwiQuestMapConfig)
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    play_sounds: bool = True
    tts_voice: str | None = None
    tts_rate: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        if self.teleop is None:
            raise ValueError("Quest3s teleop config is required for LeKiwi recording.")


@parser.wrap()
def record(cfg: LeKiwiRecordConfig) -> sLerobotDataset:
    init_logging()

    client_cfg = cfg.robot.to_client_config()
    if not client_cfg.remote_ip:
        raise ValueError("Set the Pi host IP: --robot.remote_ip=10.22.26.30")

    robot = LeKiwiClient(client_cfg)
    teleop_quest = Quest3sController(
        mqtt_broker=cfg.teleop.mqtt_broker,
        mqtt_port=cfg.teleop.mqtt_port,
        mqtt_topic=cfg.teleop.mqtt_topic,
    )
    teleop_keyboard = LeKiwiKeyboardTeleop(teleop_keys=client_cfg.teleop_keys)
    teleop = [teleop_quest, teleop_keyboard]
    quest_mapper = LeKiwiQuestMapper(
        config=LeKiwiQuestMapperConfig(
            quest_position_scale=cfg.quest_map.position_scale,
            orientation_weight=cfg.quest_map.orientation_weight,
            max_joint_step_deg=cfg.quest_map.max_joint_step_deg,
            quest_delta_ema_alpha=cfg.quest_map.quest_delta_ema_alpha,
            joint_output_alpha=cfg.quest_map.joint_output_alpha,
            position_deadzone_mm=cfg.quest_map.position_deadzone_mm,
            rotation_deadzone_deg=cfg.quest_map.rotation_deadzone_deg,
        )
    )
    logging.info(
        "Quest arm sensitivity: position_scale=%.3f orientation_weight=%.3f "
        "max_joint_step_deg=%.1f quest_ema=%.2f joint_ema=%.2f (use max_joint_step<=4 if jitter)",
        cfg.quest_map.position_scale,
        cfg.quest_map.orientation_weight,
        cfg.quest_map.max_joint_step_deg,
        cfg.quest_map.quest_delta_ema_alpha,
        cfg.quest_map.joint_output_alpha,
    )

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(robot.observation_features, robot.action_features)
    if not cfg.robot.use_cameras:
        logging.info(
            "Recording without cameras (robot.use_cameras=false). "
            "Pi host must use: python -m slerobot.robots.lekiwi.lekiwi_host --no-cameras"
        )

    dataset = None
    listener = None
    events = None

    try:
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
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_process,
                    num_threads=cfg.dataset.num_image_write_threads_per_camera * len(robot.cameras),
                )
        else:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") and robot.cameras else 0
            dataset = sLerobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
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

        logging.info(
            "Connecting LeKiwi client -> tcp://%s:%s (cmd) / :%s (obs)",
            client_cfg.remote_ip,
            client_cfg.port_zmq_cmd,
            client_cfg.port_zmq_observations,
        )
        robot.connect()
        if robot.calibration:
            quest_mapper.calibration = robot.calibration
            if quest_mapper._ik is not None:
                quest_mapper._ik.calibration = robot.calibration
            logging.info(
                "Loaded motor calibration (%d motors) for Quest IK.",
                len(robot.calibration),
            )
        else:
            logging.warning(
                "No calibration file on this machine — IK uses approximate joint scaling. "
                "Copy Pi calibration to %s",
                robot.calibration_fpath,
            )
        teleop_quest.connect()
        teleop_keyboard.connect()
        time.sleep(2.0)

        listener, events = init_keyboard_listener()
        if cfg.display_data and not _ui_telemetry_enabled():
            _init_rerun(
                session_name="slerobot_lekiwi_record",
                connect_ip=cfg.display_ip,
                connect_port=cfg.display_port,
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                quest_mapper.reset()
                log_say(
                    f"Recording episode {recorded_episodes + 1}",
                    cfg.play_sounds,
                    voice=cfg.tts_voice,
                    rate=cfg.tts_rate,
                )
                record_loop(
                    robot=robot,
                    teleop=teleop,
                    dataset=dataset,
                    events=events,
                    fps=cfg.dataset.fps,
                    robot_backend="lekiwi",
                    quest_mapper=quest_mapper,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    record_data=True,
                    play_sounds=cfg.play_sounds,
                    tts_voice=cfg.tts_voice,
                    tts_rate=cfg.tts_rate,
                )

                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds, voice=cfg.tts_voice, rate=cfg.tts_rate)
                    quest_mapper.reset()
                    record_loop(
                        robot=robot,
                        teleop=teleop,
                        dataset=dataset,
                        events=events,
                        fps=cfg.dataset.fps,
                        robot_backend="lekiwi",
                        quest_mapper=quest_mapper,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        record_data=False,
                        play_sounds=cfg.play_sounds,
                        tts_voice=cfg.tts_voice,
                        tts_rate=cfg.tts_rate,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds, voice=cfg.tts_voice, rate=cfg.tts_rate)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
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

        if teleop_quest.is_connected:
            teleop_quest.disconnect()
        if teleop_keyboard.is_connected:
            teleop_keyboard.disconnect()

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
