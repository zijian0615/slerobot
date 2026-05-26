"""
Record LeKiwi demonstrations with Quest 3s (arm) + keyboard (base).

Example:
```shell
python -m slerobot.scripts.slerobot_lekiwi_record \
    --dataset.repo_id=user/lekiwi_quest \
    --dataset.single_task="pick and place" \
    --robot.remote_ip=10.22.26.30 \
    --mqtt_broker=10.22.9.10 \
    --mqtt_topic=quest/data

# Pi: python -m slerobot.robots.lekiwi.lekiwi_host --no-cameras
# Mac IK: pip install -e ".[kinematics]"
# Arm: press A (keep hand still ~settle_frames/20s) → ARMED → hold trigger to move
# Tune: --position_scale=0.7  --quest_axis_remap=z,-x,y  (try x,y,z if axes feel wrong)
#       --settle_frames=12  --ema_alpha=0.6  --deadzone_mm=3.0
```
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from slerobot.configs import parser
from slerobot.datasets.slerobot_datasets import sLerobotDataset
from slerobot.datasets.utils import combine_feature_dicts
from slerobot.datasets.video_utils import VideoEncodingManager
from slerobot.processor import make_default_processors
from slerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig, lekiwi_cameras_config
from slerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from slerobot.teleoperators.lekiwi_keyboard import LeKiwiKeyboardTeleop
from slerobot.teleoperators.quest3s import Quest3sController
from slerobot.utils.control_utils import init_keyboard_listener, is_headless
from slerobot.utils.lekiwi_quest_utils import LeKiwiQuestMapper, LeKiwiQuestMapperConfig
from slerobot.utils.lekiwi_record_loop import lekiwi_record_loop
from slerobot.utils.utils import init_logging, log_say
from slerobot.utils.visualization_utils import _init_rerun, shutdown_rerun


def _ui_telemetry_enabled() -> bool:
    return os.getenv("SLEROBOT_TELEMETRY_PUSH", "").lower() in ("1", "true", "yes")


@dataclass
class LeKiwiDatasetRecordConfig:
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
    remote_ip: str = "127.0.0.1"
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    polling_timeout_ms: int = 100
    connect_timeout_s: int = 10
    id: str = "lekiwi_client"
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
class LeKiwiRecordConfig:
    dataset: LeKiwiDatasetRecordConfig
    robot: LeKiwiRobotRecordConfig = field(default_factory=LeKiwiRobotRecordConfig)
    mqtt_broker: str = "10.22.9.10"
    mqtt_port: int = 1883
    mqtt_topic: str = "quest/data"
    quest_axis_remap: str = "z,-x,y"
    position_scale: float = 1.0
    orientation_weight: float = 0.3
    max_joint_step_deg: float = 8.0
    ema_alpha: float = 0.6
    deadzone_mm: float = 3.0
    settle_frames: int = 12
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    play_sounds: bool = True
    tts_voice: str | None = None
    tts_rate: int | None = None
    resume: bool = False


def _make_quest_mapper(cfg: LeKiwiRecordConfig) -> LeKiwiQuestMapper:
    return LeKiwiQuestMapper(
        config=LeKiwiQuestMapperConfig(
            quest_axis_remap=cfg.quest_axis_remap,
            position_scale=cfg.position_scale,
            orientation_weight=cfg.orientation_weight,
            max_joint_step_deg=cfg.max_joint_step_deg,
            ema_alpha=cfg.ema_alpha,
            deadzone_mm=cfg.deadzone_mm,
            settle_frames=cfg.settle_frames,
        )
    )


def _open_dataset(cfg: LeKiwiRecordConfig, robot: LeKiwiClient) -> sLerobotDataset:
    features = combine_feature_dicts(robot.observation_features, robot.action_features)
    num_cameras = len(robot.cameras) if getattr(robot, "cameras", None) else 0
    threads = cfg.dataset.num_image_write_threads_per_camera * num_cameras

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
        if num_cameras > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_process,
                num_threads=threads,
            )
        return dataset

    return sLerobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_process,
        image_writer_threads=threads if num_cameras else 0,
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        vcodec=cfg.dataset.vcodec,
        streaming_encoding=cfg.dataset.streaming_encoding,
        encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
        encoder_threads=cfg.dataset.encoder_threads,
    )


@parser.wrap()
def record(cfg: LeKiwiRecordConfig) -> sLerobotDataset:
    init_logging()

    client_cfg = cfg.robot.to_client_config()
    if not client_cfg.remote_ip or client_cfg.remote_ip == "127.0.0.1":
        logging.warning(
            "robot.remote_ip is %s — set Pi IP, e.g. --robot.remote_ip=10.22.26.30",
            client_cfg.remote_ip,
        )

    robot = LeKiwiClient(client_cfg)
    quest = Quest3sController(
        mqtt_broker=cfg.mqtt_broker,
        mqtt_port=cfg.mqtt_port,
        mqtt_topic=cfg.mqtt_topic,
    )
    keyboard = LeKiwiKeyboardTeleop(teleop_keys=client_cfg.teleop_keys)
    mapper = _make_quest_mapper(cfg)
    teleop_proc, robot_proc, obs_proc = make_default_processors()

    if not cfg.robot.use_cameras:
        logging.info("No cameras: Pi host must use --no-cameras")

    dataset = None
    listener = None
    events = None

    try:
        dataset = _open_dataset(cfg, robot)
        logging.info(
            "ZMQ -> tcp://%s:%s (cmd) / :%s (obs)",
            client_cfg.remote_ip,
            client_cfg.port_zmq_cmd,
            client_cfg.port_zmq_observations,
        )
        robot.connect()
        if robot.calibration:
            mapper.calibration = robot.calibration
            if mapper._ik is not None:
                mapper._ik.calibration = robot.calibration
            logging.info("Loaded calibration (%d motors) for Quest IK.", len(robot.calibration))
        else:
            logging.warning("No calibration on Mac — copy Pi file to %s", robot.calibration_fpath)

        quest.connect()
        keyboard.connect()

        listener, events = init_keyboard_listener()
        if cfg.display_data and not _ui_telemetry_enabled():
            _init_rerun(
                session_name="slerobot_lekiwi_record",
                connect_ip=cfg.display_ip,
                connect_port=cfg.display_port,
            )

        loop_kw = dict(
            robot=robot,
            quest=quest,
            keyboard=keyboard,
            mapper=mapper,
            events=events,
            fps=cfg.dataset.fps,
            teleop_action_processor=teleop_proc,
            robot_action_processor=robot_proc,
            robot_observation_processor=obs_proc,
            dataset=dataset,
            single_task=cfg.dataset.single_task,
            display_data=cfg.display_data,
        )

        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                mapper.reset()
                log_say(
                    f"Recording episode {recorded + 1}",
                    cfg.play_sounds,
                    voice=cfg.tts_voice,
                    rate=cfg.tts_rate,
                )
                lekiwi_record_loop(
                    **loop_kw,
                    control_time_s=cfg.dataset.episode_time_s,
                    record_data=True,
                )

                if not events["stop_recording"] and (
                    recorded < cfg.dataset.num_episodes - 1 or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds, voice=cfg.tts_voice, rate=cfg.tts_rate)
                    mapper.reset()
                    lekiwi_record_loop(
                        **loop_kw,
                        control_time_s=cfg.dataset.reset_time_s,
                        record_data=False,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds, voice=cfg.tts_voice, rate=cfg.tts_rate)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True, voice=cfg.tts_voice, rate=cfg.tts_rate)
        if dataset:
            dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if quest.is_connected:
            quest.disconnect()
        if keyboard.is_connected:
            keyboard.disconnect()
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
