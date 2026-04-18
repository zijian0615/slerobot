import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slerobot.configs import parser
from slerobot.configs.policies import PreTrainedConfig
from slerobot.teleoperators import Teleoperator
from slerobot.teleoperators.quest3s import Quest3sController
from slerobot.robots import Robot
from slerobot.robots.fanuc import Fanuc
from slerobot.cameras.utils import make_cameras_from_configs

from slerobot.utils.constants import ACTION, OBS_STR
from slerobot.datasets.slerobot_datasets import sLerobotDataset
from slerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from slerobot.datasets.video_utils import VideoEncodingManager

from slerobot.policies.pretrained import PreTrainedPolicy
from slerobot.policies.utils import make_robot_action

from slerobot.utils.control_utils import init_keyboard_listener, is_headless, predict_action
from slerobot.utils.utils import get_safe_torch_device, init_logging, log_say

from slerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from slerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

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
    policy: PreTrainedConfig | None = None
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
        policy_path = parser.get_path_arg("policy")

        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")
    
    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Get fields that should be treated as paths for config parsing.
        
        This enables the parser to load config from the policy using `--policy.path=local/dir`
        """
        return ["policy"]


def _apply_robot_action_metadata(
    action: RobotAction,
    *,
    robot_speed: int | None,
    robot_term_type: str | None,
    robot_term_value: int | None,
    gripper_lcb_type: str | None,
    gripper_lcb_value: int,
    gripper_port_type: int | None,
    gripper_port_number: int | None,
    gripper_open_port_number: int | None,
    gripper_close_port_number: int | None,
    gripper_open_value: str,
    gripper_close_value: str,
) -> RobotAction:
    robot_action = dict(action)

    if robot_speed is not None:
        robot_action.setdefault("speed", robot_speed)
    if robot_term_type is not None:
        robot_action.setdefault("term_type", robot_term_type)
    if robot_term_value is not None:
        robot_action.setdefault("term_value", robot_term_value)

    if "j7" not in robot_action:
        return robot_action

    discrete_j7 = float(float(robot_action["j7"]) >= 0.5)
    robot_action["j7"] = discrete_j7

    selected_gripper_port = None
    if gripper_open_port_number is not None and gripper_close_port_number is not None:
        selected_gripper_port = gripper_close_port_number if discrete_j7 else gripper_open_port_number
    elif gripper_port_number is not None:
        selected_gripper_port = gripper_port_number

    if (
        gripper_lcb_type is not None
        and gripper_port_type is not None
        and selected_gripper_port is not None
    ):
        robot_action.update(
            {
                "lcb_type": gripper_lcb_type,
                "lcb_value": gripper_lcb_value,
                "port_type": gripper_port_type,
                "port_number": selected_gripper_port,
                "port_value": gripper_close_value if discrete_j7 else gripper_open_value,
            }
        )

    return robot_action


def _log_policy_action_flow(act_processed_policy: RobotAction, robot_action_to_send: RobotAction) -> None:
    debug_counter = getattr(_log_policy_action_flow, "_debug_counter", 0)
    if debug_counter >= 10:
        return

    logging.info("[POLICY_ACTION_DICT] %s", act_processed_policy)
    logging.info("[ROBOT_ACTION_TO_SEND] %s", robot_action_to_send)
    setattr(_log_policy_action_flow, "_debug_counter", debug_counter + 1)


def _log_processor_pipeline(name: str, pipeline: PolicyProcessorPipeline | None) -> None:
    if pipeline is None:
        logging.info("[%s] None", name)
        return

    step_summaries: list[str] = []
    for idx, step in enumerate(pipeline.steps):
        config = getattr(step, "config", None)
        if not config:
            config = {
                key: value
                for key, value in vars(step).items()
                if key not in {"_current_transition", "stats"}
            }

        state_keys: list[str] = []
        if hasattr(step, "state_dict"):
            try:
                state_keys = list(step.state_dict().keys())
            except Exception:
                state_keys = ["<state_dict_error>"]

        step_summaries.append(
            f"step={idx} class={step.__class__.__name__} config={config} state_keys={state_keys}"
        )

    for summary in step_summaries:
        logging.info("[%s] %s", name, summary)


def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction] | None = None,
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction] | None = None,
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation] | None = None,
    dataset: sLerobotDataset | None = None,
    teleop: Teleoperator | list | None = None,
    policy: PreTrainedPolicy | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
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
    record_data: bool = True,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    if policy is not None and dataset is None:
        raise ValueError("A dataset is required when using a policy for recording.")

    if policy is not None:
        policy.reset()

    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        default_teleop_processor, default_robot_action_processor, default_robot_observation_processor = (
            make_default_processors()
        )
        teleop_action_processor = teleop_action_processor or default_teleop_processor
        robot_action_processor = robot_action_processor or default_robot_action_processor
        robot_observation_processor = robot_observation_processor or default_robot_observation_processor

    BUFFER_SIZE = 4
    INTER_PACKET_DELAY = 0.002

    pending: set = set()
    observation_grip_value: float | None = None

    no_action_count = 0
    timestamp = 0.0
    start_episode_t = time.perf_counter()

    action_sent_count = 0
    action_skipped_buffer_full = 0
    last_diagnostic_time = time.perf_counter()
    diagnostic_interval_s = 1.0

    while True:
        if control_time_s is not None and timestamp >= control_time_s:
            break

        if events["exit_early"]:
            events["exit_early"] = False
            break

        while True:
            seq_id, err = robot.check_ack()
            if seq_id is None:
                break
            pending.discard(seq_id)

        if len(pending) >= BUFFER_SIZE:
            time.sleep(0.001)
            action_skipped_buffer_full += 1
            timestamp = time.perf_counter() - start_episode_t
            continue

        obs = robot.get_observation()
        if observation_grip_value is not None:
            obs["j7"] = observation_grip_value
        obs = robot_observation_processor(obs)

        observation_frame = (
            build_dataset_frame(dataset.features, obs, prefix=OBS_STR) if dataset is not None else {}
        )

        act_processed_policy: RobotAction | None = None
        act_processed_teleop: RobotAction | None = None
        action_values: RobotAction | PolicyAction | None = None

        if policy is not None and preprocessor is not None and postprocessor is not None and record_data:
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
            act_processed_policy = make_robot_action(action_values, dataset.features)
        elif policy is None and isinstance(teleop, Teleoperator):
            act = teleop.get_action()
            if act is None:
                time.sleep(0.005)
                timestamp = time.perf_counter() - start_episode_t
                continue
            act_processed_teleop = teleop_action_processor((act, obs))
        elif policy is None and isinstance(teleop, list):
            teleop_arm, teleop_keyboard = teleop
            arm_action = teleop_arm.get_action()
            if arm_action is None:
                time.sleep(0.005)
                timestamp = time.perf_counter() - start_episode_t
                continue
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No policy or teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            time.sleep(0.01)
            timestamp = time.perf_counter() - start_episode_t
            continue

        no_action_count = 0

        if policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        robot_action_to_send = _apply_robot_action_metadata(
            robot_action_to_send,
            robot_speed=robot_speed,
            robot_term_type=robot_term_type,
            robot_term_value=robot_term_value,
            gripper_lcb_type=gripper_lcb_type,
            gripper_lcb_value=gripper_lcb_value,
            gripper_port_type=gripper_port_type,
            gripper_port_number=gripper_port_number,
            gripper_open_port_number=gripper_open_port_number,
            gripper_close_port_number=gripper_close_port_number,
            gripper_open_value=gripper_open_value,
            gripper_close_value=gripper_close_value,
        )
        if policy is not None and act_processed_policy is not None:
            _log_policy_action_flow(act_processed_policy, robot_action_to_send)
        if "j7" in robot_action_to_send:
            observation_grip_value = float(robot_action_to_send["j7"])

        robot.send_action(robot_action_to_send)

        seq_id = robot.seq_id - 1
        pending.add(seq_id)

        action_sent_count += 1

        if len(pending) >= BUFFER_SIZE:
            time.sleep(INTER_PACKET_DELAY)

        if record_data and dataset is not None:
            action_frame = build_dataset_frame(dataset.features, robot_action_to_send, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        timestamp = time.perf_counter() - start_episode_t

        now = time.perf_counter()
        if now - last_diagnostic_time >= diagnostic_interval_s:
            actual_fps = action_sent_count / timestamp if timestamp > 0 else 0

            print(
                f"[FPS] t={timestamp:.1f}s actual={actual_fps:.2f}fps target={fps} "
                f"sent={action_sent_count} pending={len(pending)} skipped_buffer={action_skipped_buffer_full}"
            )

            last_diagnostic_time = now

@parser.wrap()
def record(cfg: RecordConfig) -> sLerobotDataset:
    init_logging()

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
    if cfg.robot.cameras is not None:
        robot.cameras = make_cameras_from_configs(cfg.robot.cameras)

    if cfg.teleop is not None:
        if hasattr(cfg.teleop, "mqtt_broker"):
            teleop = Quest3sController(
                mqtt_broker=cfg.teleop.mqtt_broker,
                mqtt_port=getattr(cfg.teleop, "mqtt_port", 1883),
                mqtt_topic=getattr(cfg.teleop, "mqtt_topic", "quest/data"),
            )
        else:
            teleop = Quest3sController()
    else:
        teleop = None

    policy = None
    preprocessor = None
    postprocessor = None
    if cfg.policy is not None:
        if cfg.policy.pretrained_path is None:
            raise ValueError("Policy config is missing `pretrained_path`. Please pass `--policy.path=...`.")

        if cfg.policy.type == "act":
            from slerobot.policies.act.modeling_act import ACTPolicy

            policy = ACTPolicy.from_pretrained(
                pretrained_name_or_path=cfg.policy.pretrained_path,
                config=cfg.policy,
            )
            preprocessor = PolicyProcessorPipeline.from_pretrained(
                cfg.policy.pretrained_path,
                config_filename="policy_preprocessor.json",
            )
            postprocessor = PolicyProcessorPipeline.from_pretrained(
                cfg.policy.pretrained_path,
                config_filename="policy_postprocessor.json",
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            )
            _log_processor_pipeline("POLICY_PREPROCESSOR", preprocessor)
            _log_processor_pipeline("POLICY_POSTPROCESSOR", postprocessor)
        else:
            raise ValueError(f"Unsupported policy type for recording: {cfg.policy.type}")

    dataset_features = combine_feature_dicts(robot.observation_features, robot.action_features)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

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
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_process,
                    num_threads_per_camera=cfg.dataset.num_image_write_threads_per_camera * len(robot.cameras),
                )
        else:
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

        robot.connect()

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
                log_say(f"Recording episode {recorded_episodes + 1}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    teleop=teleop,
                    policy=policy,
                    dataset=dataset,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=cfg.display_compressed_images,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
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
                    record_data=True,
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
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_compressed_images=cfg.display_compressed_images,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
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
                        record_data=False,
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
