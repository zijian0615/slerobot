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
    DatasetRecordConfig,
    Quest3sConfig,
    record_loop,
)
from slerobot.teleoperators.lekiwi_keyboard import LeKiwiKeyboardTeleop
from slerobot.teleoperators.quest3s import Quest3sController
from slerobot.utils.control_utils import init_keyboard_listener, is_headless
from slerobot.utils.lekiwi_quest_utils import LeKiwiQuestMapper
from slerobot.utils.utils import init_logging, log_say
from slerobot.utils.visualization_utils import _init_rerun, shutdown_rerun


def _ui_telemetry_enabled() -> bool:
    return os.getenv("SLEROBOT_TELEMETRY_PUSH", "").lower() in ("1", "true", "yes")


@dataclass
class LeKiwiRobotRecordConfig:
    """Plain dataclass for draccus CLI (same pattern as FanucConfig in slerobot_record)."""

    remote_ip: str = "127.0.0.1"
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
    id: str = "lekiwi_client"

    def to_client_config(self) -> LeKiwiClientConfig:
        return LeKiwiClientConfig(
            id=self.id,
            remote_ip=self.remote_ip,
            port_zmq_cmd=self.port_zmq_cmd,
            port_zmq_observations=self.port_zmq_observations,
            polling_timeout_ms=self.polling_timeout_ms,
            connect_timeout_s=self.connect_timeout_s,
            cameras=lekiwi_cameras_config(),
        )


@dataclass
class LeKiwiRecordConfig:
    dataset: DatasetRecordConfig
    robot: LeKiwiRobotRecordConfig = field(default_factory=LeKiwiRobotRecordConfig)
    teleop: Quest3sConfig = field(default_factory=Quest3sConfig)
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
    quest_mapper = LeKiwiQuestMapper()

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(robot.observation_features, robot.action_features)

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
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") and robot.cameras else 1
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

        robot.connect()
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
