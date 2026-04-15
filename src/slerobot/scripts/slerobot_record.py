import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from slerobot.configs import parser
from slerobot.teleoperators import Teleoperator,quest3s
from slerobot.teleoperators.quest3s import Quest3sController
from slerobot.robots import Robot,fanuc
from slerobot.robots.fanuc import Fanuc
from slerobot.cameras.utils import make_cameras_from_configs

from slerobot.utils.constants import ACTION, OBS_STR
from slerobot.datasets.slerobot_datasets import sLerobotDataset
from slerobot.datasets.utils import build_dataset_frame,combine_feature_dicts,convert_quest3s_to_fanuc_action
from slerobot.datasets.video_utils import VideoEncodingManager
from slerobot.utils.control_utils import init_keyboard_listener,is_headless
from slerobot.utils.robot_utils import precise_sleep
from slerobot.utils.utils import init_logging, log_say

@dataclass
class DatasetRecordConfig:
    """Configuration for recording a dataset.
    
    Attributes:
        repo_id: Dataset identifier
        single_task: Short description of the task
        root_dir: Root directory where the dataset will be stored
        fps: Limit the frames per second (default: 20)
        episode_time_s: Number of seconds of data recording per episode (default: 60)
        reset_time_s: Number of seconds of resetting the environment after each episode (default: 60)
        num_episodes: Number of episodes to record (default: 50)
        video: Encode frames in the dataset into video (default: True)
        push_to_hub: Upload to hugging face hub (default: True)
        private: Upload on private repo (default: False)
        tags: Add tags to your dataset on hub (default: None)
        num_image_writer_process: Num of subprocesses handling the saving of frames as PNG (default: 0)
        num_image_write_threads_per_camera: Num of threads writing the frames as png images on disk (default: 4)
        video_encoding_batch_size: Num of episodes to record before batch encoding videos (default: 1)
        vcodec: Video codec for encoding (default: "libsvtav1")
        streaming_encoding: Enable streaming video encoding (default: False)
        encoder_queue_maxsize: Maximum number of frames to buffer per camera when using streaming encoding (default: 30)
    """
    
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
class FanucConfig:
    """Configuration for Fanuc robot connection.
    
    Attributes:
        host: IP address of the Fanuc robot controller
        port: Port number for TCP connection (default: 16001)
        group: Robot group number (default: 1)
        utool: User tool number (default: 1)
        uframe: User frame number (default: 0)
        speed: Robot speed percentage (default: 150)
        term_type: Termination type (default: "CNT")
        term_value: Termination value (default: 100)
        gripper_lcb_type: Fanuc linear control block type for gripper output (default: "TA")
        gripper_lcb_value: Fanuc linear control block value for gripper output (default: 10)
        gripper_port_type: Fanuc output port type for gripper control (default: 2)
        gripper_state_port_number: Fanuc DIN input port used to read gripper state (default: None)
        gripper_port_number: Shared Fanuc output port used to control gripper (default: None)
        gripper_open_port_number: Fanuc output port used to open gripper (default: 3)
        gripper_close_port_number: Fanuc output port used to close gripper (default: 4)
        gripper_open_value: Fanuc output value used to open gripper (default: "ON")
        gripper_close_value: Fanuc output value used to close gripper (default: "ON")
        cameras: Dictionary of camera configurations (default: None)
            Example: {"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 20}}
    """
    host: str = "172.30.109.22"
    port: int = 16001
    group: int = 1
    utool: int = 1
    uframe: int = 0
    speed: int = 250
    term_type: str = "CNT"
    term_value: int = 100
    gripper_lcb_type: str | None = "TA"
    gripper_lcb_value: int = 10
    gripper_port_type: int | None = 2
    gripper_state_port_number: int | None = None
    gripper_port_number: int | None = None
    gripper_open_port_number: int | None = 3
    gripper_close_port_number: int | None = 4
    gripper_open_value: str = "ON"
    gripper_close_value: str = "ON"
    cameras: dict[str, Any] | None = None


@dataclass
class Quest3sConfig:
    """Configuration for Quest 3s teleoperation.
    
    Attributes:
        mqtt_broker: MQTT broker IP address (default: "10.22.9.10")
        mqtt_port: MQTT broker port (default: 1883)
        mqtt_topic: MQTT topic to subscribe to (default: "quest/data")
    """
    mqtt_broker: str = "10.22.9.10"
    mqtt_port: int = 1883
    mqtt_topic: str = "quest/data"



@dataclass
class RecordConfig:
    """Configuration for recording robot demonstrations.
    
    Attributes:
        dataset: DatasetRecordConfig - configuration for the dataset
        robot: FanucConfig - configuration for the Fanuc robot
        teleop: Quest3sConfig or None - configuration for teleoperation (default: None)
        policy: PreTrainedConfig or None - pretrained policy configuration
        display_data: Whether to display data on screen (default: False)
        display_ip: IP address of remote rerun server (default: None)
        display_port: Port of remote rerun server (default: None)
        display_compressed_images: Whether to display compressed images in Rerun (default: False)
        play_sounds: Whether to use vocal synthesis to read events (default: True)
        resume: Whether to resume recording on an existing dataset (default: False)
    """
    
    dataset: DatasetRecordConfig
    robot: FanucConfig = field(default_factory=FanucConfig)
    teleop: Quest3sConfig | None = None
    policy: object | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False
    
    def __post_init__(self):
        """Validate configuration if needed.
        
        Note: teleop and policy can both be None initially, but at least one must be
        provided when record() is actually called.
        """
        pass
    
    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Get fields that should be treated as paths for config parsing.
        
        This enables the parser to load config from the policy using `--policy.path=local/dir`
        """
        return ["policy"]



""" This enables the parser to load config from the policy using `--policy.path=local/dir`
        pass

    --------------- record_loop() data flow --------------------------
    # just ignore all the processor right now
    [ Robot ]
        V
    [ robot.get_observation() ] ---> raw_obs
        V
    [ robot_observation_processor ] ---> processed_obs
        V
    .-----( ACTION LOGIC )------------------.
    V                                       V
    [ From Teleoperator ]                   [ From Policy ]
    |                                       |
    |  [teleop.get_action] -> raw_action    |   [predict_action]
    |          |                            |          |
    |          V                            |          V
    | [teleop_action_processor]             |          |
    |          |                            |          |
    '---> processed_teleop_action           '---> processed_policy_action
    |                                       |
    '-------------------------.-------------'
                            V
                [ robot_action_processor ] --> robot_action_to_send
                            V
                    [ robot.send_action() ] -- (Robot Executes)
                            V
                    ( Save to Dataset )
                            V
                ( Rerun Log / Loop Wait ) """


def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    dataset: sLerobotDataset | None = None,
    teleop: Teleoperator | None = None,
    policy: None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    robot_speed: int | None = None,
    robot_term_type: str | None = None,
    robot_term_value: int | None = None,
    gripper_lcb_type: str | None = None,
    gripper_lcb_value: int = 10,
    gripper_port_type: int | None = None,
    gripper_port_number: int | None = None,
    gripper_open_port_number: int | None = None,
    gripper_close_port_number: int | None = None,
    gripper_open_value: str = "ON",
    gripper_close_value: str = "ON",
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    BUFFER_SIZE = 4
    MIN_DIST_MM = 0.5
    INTER_PACKET_DELAY = 0.002

    pending: set = set()
    last_sent_pose = None
    last_sent_grip = None
    observation_grip_value: float | None = None

    no_action_count = 0
    timestamp = 0.0
    start_episode_t = time.perf_counter() if policy is not None else None
    episode_started = policy is not None

    action_sent_count = 0
    action_skipped_buffer_full = 0
    action_skipped_no_action = 0
    action_skipped_spatial_filter = 0
    last_diagnostic_time = time.perf_counter()
    diagnostic_interval_s = 1.0

    sample_period_s = 1.0 / fps
    next_sample_t: float | None = None

    _baseline_printed = False
    _waiting_for_first_action_logged = False

    def _advance_sample_deadline(now_t: float) -> None:
        nonlocal next_sample_t
        if next_sample_t is None:
            return
        next_sample_t += sample_period_s
        while next_sample_t <= now_t:
            next_sample_t += sample_period_s

    while True:
        start_loop_t = time.perf_counter()

        if episode_started and control_time_s is not None and timestamp >= control_time_s:
            break

        if events["exit_early"]:
            events["exit_early"] = False
            break

        if episode_started:
            if next_sample_t is None and start_episode_t is not None:
                next_sample_t = start_episode_t
            if next_sample_t is not None:
                now_t = time.perf_counter()
                if now_t < next_sample_t:
                    precise_sleep(next_sample_t - now_t)

        while True:
            seq_id, err = robot.check_ack()
            if seq_id is None:
                break
            pending.discard(seq_id)

        # Buffer 满时跳帧
        if len(pending) >= BUFFER_SIZE:
            time.sleep(0.001)
            action_skipped_buffer_full += 1
            if episode_started:
                _advance_sample_deadline(time.perf_counter())
            continue

        if policy is not None:
            obs = robot.get_observation()
            action = policy.predict(obs)
            policy_action = action
            action_to_save = policy_action
            observation_frame = {}
            if dataset is not None:
                observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)

        elif teleop is not None:
            action = teleop.get_action()

            if not _baseline_printed:
                _baseline_printed = True

            if action is None:
                time.sleep(0.001)
                if not episode_started:
                    if not _waiting_for_first_action_logged:
                        _waiting_for_first_action_logged = True
                else:
                    action_skipped_no_action += 1
                    # if action_skipped_no_action % 50 == 0:
                    #     logging.warning(
                    #         f"[NO_ACTION] 已连续跳过 {action_skipped_no_action} 帧（无 MQTT 数据），"
                    #         f"累计耗时 {timestamp:.1f}s。请检查 Quest3s 是否在发送数据。"
                    #     )
                    _advance_sample_deadline(time.perf_counter())
                continue

            if not episode_started:
                start_episode_t = time.perf_counter()
                timestamp = 0.0
                episode_started = True
                last_diagnostic_time = start_episode_t
                next_sample_t = start_episode_t
                # logging.info("[START] 收到首个 teleop action，开始记录 episode 时间与诊断统计。")

            target_pose = (
                action["position"]["x"],
                action["position"]["y"],
                action["position"]["z"],
                action["rotation"]["w"],
                action["rotation"]["p"],
                action["rotation"]["r"],
            )
            grip_pressed = int(action.get("buttons", {}).get("grip", 0))
            observation_grip_value = float(grip_pressed)

            if last_sent_pose is not None:
                dx = target_pose[0] - last_sent_pose[0]
                dy = target_pose[1] - last_sent_pose[1]
                dz = target_pose[2] - last_sent_pose[2]
                dist = (dx**2 + dy**2 + dz**2) ** 0.5
                grip_changed = last_sent_grip is None or grip_pressed != last_sent_grip
                if dist < MIN_DIST_MM and not grip_changed:
                    time.sleep(0.001)
                    action_skipped_spatial_filter += 1
                    # if action_skipped_spatial_filter % 100 == 0:
                    #     logging.warning(
                    #         f"[SPATIAL_FILTER] 已跳过 {action_skipped_spatial_filter} 帧（dist={dist:.4f}mm < {MIN_DIST_MM}mm）。"
                    #         f"机器人静止或 MIN_DIST_MM 设置过大？"
                    #     )
                    _advance_sample_deadline(time.perf_counter())
                    continue

            teleop_action = {
                "position": target_pose,
                "speed": robot_speed,
                "term_type": robot_term_type,
                "term_value": robot_term_value,
            }

            selected_gripper_port = None
            if (
                gripper_open_port_number is not None
                and gripper_close_port_number is not None
            ):
                selected_gripper_port = (
                    gripper_close_port_number if grip_pressed else gripper_open_port_number
                )
            elif gripper_port_number is not None:
                selected_gripper_port = gripper_port_number

            if (
                gripper_lcb_type is not None
                and gripper_port_type is not None
                and selected_gripper_port is not None
            ):
                teleop_action.update(
                    {
                        "lcb_type": gripper_lcb_type,
                        "lcb_value": gripper_lcb_value,
                        "port_type": gripper_port_type,
                        "port_number": selected_gripper_port,
                        "port_value": (
                            gripper_close_value if grip_pressed else gripper_open_value
                        ),
                    }
                )

            action_to_save = {
                "j0": target_pose[0], "j1": target_pose[1], "j2": target_pose[2],
                "j3": target_pose[3], "j4": target_pose[4], "j5": target_pose[5],
                "j7": float(grip_pressed),
            }
            last_sent_pose = target_pose
            last_sent_grip = grip_pressed
            observation_frame = {}

        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10000 == 0:
                logging.warning("No policy or teleoperator provided, skipping action generation.")
            time.sleep(0.001)
            continue

        # send_action
        action_to_send = policy_action if policy is not None else teleop_action
        robot.send_action(action_to_send)

        seq_id = robot.seq_id - 1
        pending.add(seq_id)

        action_sent_count += 1

        if len(pending) >= BUFFER_SIZE:
            time.sleep(INTER_PACKET_DELAY)

        _advance_sample_deadline(time.perf_counter())

        # 数据集保存
        if dataset is not None:
            if teleop is not None and not observation_frame:
                obs = robot.get_observation()
                if observation_grip_value is not None:
                    obs["j7"] = observation_grip_value
                observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)

            action_frame = build_dataset_frame(dataset.features, action_to_save, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if start_episode_t is not None:
            timestamp = time.perf_counter() - start_episode_t

        now = time.perf_counter()
        if episode_started and now - last_diagnostic_time >= diagnostic_interval_s:
            actual_fps = action_sent_count / timestamp if timestamp > 0 else 0

            print(f"[FPS] t={timestamp:.1f}s actual={actual_fps:.2f}fps target={fps} sent={action_sent_count}")

            last_diagnostic_time = now

@parser.wrap()
def record(cfg: RecordConfig) -> sLerobotDataset:
        init_logging()
        # logging.info(pformat(asdict(cfg)))
        # logging.info(f"[TELEOP CFG] type={type(cfg.teleop)}, value={cfg.teleop}")

        
        # Initialize Fanuc robot with configuration
        robot = Fanuc(
            host=cfg.robot.host,
            port=cfg.robot.port,
            group=cfg.robot.group,
            utool=cfg.robot.utool,
            uframe=cfg.robot.uframe,
            speed=cfg.robot.speed,
            term_type=cfg.robot.term_type,
            term_value=cfg.robot.term_value,
            gripper_port_number=cfg.robot.gripper_state_port_number,
        )
        # Store cameras config if provided
        if cfg.robot.cameras is not None:
            robot.cameras = make_cameras_from_configs(cfg.robot.cameras)
        
        # Initialize Quest3s teleoperator if configured
        if cfg.teleop is not None:
            # Check if teleop config has specific parameters
            if hasattr(cfg.teleop, "mqtt_broker"):
                teleop = Quest3sController(
                    mqtt_broker=cfg.teleop.mqtt_broker,
                    mqtt_port=getattr(cfg.teleop, "mqtt_port", 1883),
                    mqtt_topic=getattr(cfg.teleop, "mqtt_topic", "quest/data"),
                )
            else:
                # Use default Quest3s settings if config is just True or a simple object
                teleop = Quest3sController()
        else:
            teleop = None

        dataset_features = combine_feature_dicts(
            robot.observation_features,
            robot.action_features,
        ) # [ TODO: implement the feature combination logic in slerobot.datasets.utils]


        dataset = None
        listener = None
        events = None
        
        try:
            if cfg.resume:
                dataset = sLerobotDataset(
                    repo_id=cfg.dataset.repo_id,
                    root_dir=cfg.dataset.root,
                    batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                    vcodec=cfg.dataset.vcodec,
                    streaming_encoding=cfg.dataset.streaming_encoding,
                    encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                    encoder_threads=cfg.dataset.encoder_threads,
                )
                if hasattr(robot,"cameras") and len(robot.cameras) > 0:
                    dataset.start_image_writer(
                        num_processes=cfg.dataset.num_image_writer_process,
                        num_threads_per_camera=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                    )
            else:
                # Get number of cameras for proper thread allocation
                num_cameras = len(robot.cameras) if hasattr(robot, "cameras") and robot.cameras else 1
                
                dataset = sLerobotDataset.create(
                    cfg.dataset.repo_id,
                    cfg.dataset.fps,
                    root=cfg.dataset.root,
                    robot_type=getattr(robot, "name", "fanuc"),
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
            # Load pretrained policy
            policy = None #[# TODO: use teleop rgiht now, will implement the policy loading later]

            robot.connect()
            
            # Connect cameras if configured
            for camera in robot.cameras.values():
                camera.connect()
            
            if teleop is not None:
                teleop.connect()
                time.sleep(2.0)

            listener, events = init_keyboard_listener()

            if not cfg.dataset.streaming_encoding:
                logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

            with VideoEncodingManager(dataset):
                recorded_episodes = 0
                while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                    log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        teleop=teleop,
                        policy=policy,
                        dataset=dataset,
                        events=events,
                        fps=cfg.dataset.fps,
                        control_time_s=cfg.dataset.episode_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_compressed_images=cfg.display_compressed_images,
                        robot_speed=cfg.robot.speed,
                        robot_term_type=cfg.robot.term_type,
                        robot_term_value=cfg.robot.term_value,
                        gripper_lcb_type=cfg.robot.gripper_lcb_type,
                        gripper_lcb_value=cfg.robot.gripper_lcb_value,
                        gripper_port_type=cfg.robot.gripper_port_type,
                        gripper_port_number=cfg.robot.gripper_port_number,
                        gripper_open_port_number=cfg.robot.gripper_open_port_number,
                        gripper_close_port_number=cfg.robot.gripper_close_port_number,
                        gripper_open_value=cfg.robot.gripper_open_value,
                        gripper_close_value=cfg.robot.gripper_close_value,
                    )

                    if not events["stop_recording"] and (
                        (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                    ):
                        log_say("Reset the environment", cfg.play_sounds)

                        record_loop(
                            robot=robot,
                            teleop=teleop,
                            policy=policy,
                            dataset=dataset,
                            events=events,
                            fps=cfg.dataset.fps,
                            control_time_s=cfg.dataset.reset_time_s,
                            single_task=cfg.dataset.single_task,
                            display_data=cfg.display_data,
                            display_compressed_images=cfg.display_compressed_images,
                            robot_speed=cfg.robot.speed,
                            robot_term_type=cfg.robot.term_type,
                            robot_term_value=cfg.robot.term_value,
                            gripper_lcb_type=cfg.robot.gripper_lcb_type,
                            gripper_lcb_value=cfg.robot.gripper_lcb_value,
                            gripper_port_type=cfg.robot.gripper_port_type,
                            gripper_port_number=cfg.robot.gripper_port_number,
                            gripper_open_port_number=cfg.robot.gripper_open_port_number,
                            gripper_close_port_number=cfg.robot.gripper_close_port_number,
                            gripper_open_value=cfg.robot.gripper_open_value,
                            gripper_close_value=cfg.robot.gripper_close_value,
                        )

                    if events["rerecord_episode"]:
                        log_say("Re-record episode", cfg.play_sounds)
                        events["rerecord_episode"] = False
                        events["exit_early"] = False
                        dataset.clear_episode_buffer()
                        continue

                    dataset.save_episode()
                    recorded_episodes += 1
        finally:
            log_say("Stop recording", cfg.play_sounds, blocking=True)

            if dataset:
                dataset.finalize()
            
            # Disconnect cameras
            for camera in robot.cameras.values():
                try:
                    camera.disconnect()
                except Exception as e:
                    logging.warning(f"Failed to disconnect camera: {e}")
            
            if robot.is_connected:
                robot.disconnect()

            if teleop and teleop.is_connected:
                teleop.disconnect()
            
            if listener is not None and not is_headless():
                listener.stop()

            if dataset is not None and cfg.dataset.push_to_hub:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private) 
            log_say("Exiting, goodbye!", cfg.play_sounds)

        return dataset

    

if __name__ == "__main__":
        record()
