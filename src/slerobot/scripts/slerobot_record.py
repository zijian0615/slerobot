import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# try:
#     import rerun as rr
#     HAS_RERUN = True
# except ImportError:
#     HAS_RERUN = False

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
from slerobot.policies.act.modeling_act import ACTEigenCAMHelper, ACTPolicy, attention_cam_method_display_name
from slerobot.policies.factory import get_policy_class, make_pre_post_processors

from slerobot.utils.control_utils import init_keyboard_listener, is_headless, predict_action
from slerobot.utils.robot_utils import decode_fanuc_pose_dict, encode_fanuc_pose_dict
from slerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from slerobot.utils.live_telemetry import push_live_telemetry
from slerobot.utils.visualization_utils import _init_rerun, log_rerun_data, shutdown_rerun


def _ui_telemetry_enabled() -> bool:
    return os.getenv("SLEROBOT_TELEMETRY_PUSH", "").lower() in ("1", "true", "yes")

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
        tts_voice: TTS voice name for episode/reset prompts (macOS: `say -v '?'` to list)
        tts_rate: TTS speaking rate in words per minute (macOS `say -r`, optional)
        resume: Whether to resume recording on an existing dataset (default: False)
        enable_attention_visualization: Whether to capture and save attention visualizations from ACT policy (default: False)
        realtime_attention_display: Whether to display attention visualizations in real-time using Rerun (default: False)
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
    tts_voice: str | None = None
    tts_rate: int | None = None
    resume: bool = False
    enable_attention_visualization: bool = False
    realtime_attention_display: bool = False
    
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

    if "j7" not in robot_action:
        buttons = robot_action.get("buttons")
        if isinstance(buttons, dict) and "grip" in buttons:
            robot_action["j7"] = float(bool(buttons.get("grip", 0)))

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
    robot_backend: str = "fanuc",
    quest_mapper: Any | None = None,
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
    attention_visualization_enabled: bool = False,
    attention_output_dir: Path | str | None = None,
    realtime_attention_display: bool = False,
    play_sounds: bool = True,
    tts_voice: str | None = None,
    tts_rate: int | None = None,
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

    use_fanuc_backend = robot_backend == "fanuc"
    BUFFER_SIZE = 4
    INTER_PACKET_DELAY = 0.002

    pending: set = set()
    observation_grip_value: float | None = None

    no_action_count = 0
    timestamp = 0.0
    start_episode_t = time.perf_counter()
    target_dt = 1.0 / fps

    action_sent_count = 0
    action_skipped_buffer_full = 0
    rerun_step = 0
    last_diagnostic_time = time.perf_counter()
    diagnostic_interval_s = 1.0

    while True:
        loop_start = time.perf_counter()
        if control_time_s is not None and timestamp >= control_time_s:
            break

        if events["exit_early"]:
            events["exit_early"] = False
            break

        if use_fanuc_backend:
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
        if use_fanuc_backend and observation_grip_value is not None:
            obs["j7"] = observation_grip_value
        obs = robot_observation_processor(obs)

        observation_frame = (
            build_dataset_frame(dataset.features, obs, prefix=OBS_STR) if dataset is not None else {}
        )

        act_processed_policy: RobotAction | None = None
        act_processed_teleop: RobotAction | None = None
        action_values: RobotAction | PolicyAction | None = None
        encoded_action_values: RobotAction | None = None

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
                warning_voice = None
                if policy.config.warning_speech_voice:
                    warning_voice = policy.config.warning_speech_voice
                elif tts_voice:
                    warning_voice = tts_voice
                warning_rate = policy.config.warning_speech_rate
                if warning_rate is None:
                    warning_rate = tts_rate
                policy.announce_attention_warnings(
                    play_sounds=play_sounds,
                    voice=warning_voice,
                    rate=warning_rate,
                )

            if (
                record_data
                and attention_visualization_enabled
                and isinstance(policy, ACTPolicy)
                and policy.last_attention_maps is not None
                and attention_output_dir is not None
            ):
                try:
                    import cv2

                    output_dir = (
                        Path(attention_output_dir)
                        / "attention_maps"
                        / f"frame_{int(timestamp * fps)}"
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)

                    for camera_idx, (image_key, attn_map) in enumerate(
                        policy.last_attention_maps.items()
                    ):
                        camera_key = image_key.split(".")[-1]
                        if camera_key not in obs or attn_map is None:
                            continue
                        overlay_helper = (
                            policy.attention_overlay_helper
                            if isinstance(policy, ACTPolicy)
                            else ACTEigenCAMHelper
                        )
                        vis = overlay_helper.overlay_attention_on_image(
                            obs[camera_key],
                            attn_map,
                            overlay_alpha=0.5,
                            use_rgb=True,
                            edge_warning=policy.last_grad_cam_edge_warnings.get(image_key, False),
                            edge_margin_px=policy.config.grad_cam_edge_margin_px,
                            edge_warning_mean=policy.last_grad_cam_edge_stats.get(image_key),
                            edge_warning_threshold=policy.config.grad_cam_edge_mean_threshold,
                            edge_warning_roi_mode=policy.config.attention_roi_mode,
                        )
                        output_path = output_dir / f"camera_{camera_idx}_attention.png"
                        cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    logging.warning(f"Failed to save attention visualization: {e}")

            act_processed_policy = make_robot_action(action_values, dataset.features)
        elif policy is None and isinstance(teleop, Teleoperator):
            act = teleop.get_action()
            if act is None:
                time.sleep(0.005)
                timestamp = time.perf_counter() - start_episode_t
                continue
            act_processed_teleop = teleop_action_processor((act, obs))
        elif policy is None and isinstance(teleop, list) and robot_backend == "lekiwi":
            quest_ctrl, keyboard_ctrl = teleop
            quest_act = quest_ctrl.get_action()
            if quest_act is None:
                time.sleep(0.005)
                timestamp = time.perf_counter() - start_episode_t
                continue
            if quest_mapper is None:
                raise ValueError("quest_mapper is required when robot_backend='lekiwi'")
            keyboard_pressed = keyboard_ctrl.get_action()
            base_action = (
                robot._from_keyboard_to_base_action(keyboard_pressed)
                if hasattr(robot, "_from_keyboard_to_base_action")
                else {}
            )
            act = quest_mapper.map_action(quest_act, obs, base_action=base_action or None)
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

        if use_fanuc_backend:
            if policy is not None and act_processed_policy is not None:
                encoded_action_values = encode_fanuc_pose_dict(act_processed_policy)
                action_values = encoded_action_values
                robot_action_to_send = robot_action_processor(
                    (decode_fanuc_pose_dict(encoded_action_values), obs)
                )
            else:
                encoded_action_values = encode_fanuc_pose_dict(act_processed_teleop or {})
                action_values = encoded_action_values
                robot_action_to_send = robot_action_processor(
                    (decode_fanuc_pose_dict(encoded_action_values), obs)
                )

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
        else:
            action_values = dict(act_processed_teleop or {})
            encoded_action_values = action_values
            robot_action_to_send = robot_action_processor((action_values, obs))

        robot.send_action(robot_action_to_send)

        if use_fanuc_backend:
            seq_id = robot.seq_id - 1
            pending.add(seq_id)

        action_sent_count += 1

        if use_fanuc_backend and len(pending) >= BUFFER_SIZE:
            time.sleep(INTER_PACKET_DELAY)

        if record_data and dataset is not None:
            action_frame_values = dict(encoded_action_values or {})
            if use_fanuc_backend and "j7" in robot_action_to_send:
                action_frame_values["j7"] = float(robot_action_to_send["j7"])
            action_frame = build_dataset_frame(dataset.features, action_frame_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            attention_maps_for_obs = None
            grad_cam_edge_warnings_for_obs = None
            grad_cam_edge_stats_for_obs = None
            if isinstance(policy, ACTPolicy) and policy.last_attention_maps is not None:
                attention_maps_for_obs = {}
                grad_cam_edge_warnings_for_obs = {}
                grad_cam_edge_stats_for_obs = {}
                for image_key, attn_map in policy.last_attention_maps.items():
                    camera_key = image_key.split(".")[-1]
                    if camera_key in obs:
                        attention_maps_for_obs[camera_key] = attn_map
                        grad_cam_edge_warnings_for_obs[camera_key] = policy.last_grad_cam_edge_warnings.get(
                            image_key, False
                        )
                        if image_key in policy.last_grad_cam_edge_stats:
                            grad_cam_edge_stats_for_obs[camera_key] = policy.last_grad_cam_edge_stats[image_key]
            if _ui_telemetry_enabled():
                push_live_telemetry(
                    obs,
                    robot_action_to_send,
                    step=rerun_step,
                    attention_maps=attention_maps_for_obs,
                    grad_cam_edge_warnings=grad_cam_edge_warnings_for_obs,
                    grad_cam_edge_stats=grad_cam_edge_stats_for_obs,
                    grad_cam_edge_threshold=policy.config.grad_cam_edge_mean_threshold
                    if isinstance(policy, ACTPolicy)
                    else None,
                    grad_cam_edge_margin_px=policy.config.grad_cam_edge_margin_px
                    if isinstance(policy, ACTPolicy)
                    else None,
                    grad_cam_edge_roi_mode=policy.config.attention_roi_mode
                    if isinstance(policy, ACTPolicy)
                    else "mitigation",
                    attention_overlay_helper=policy.attention_overlay_helper
                    if isinstance(policy, ACTPolicy)
                    else ACTEigenCAMHelper,
                )
            else:
                log_rerun_data(
                    obs,
                    robot_action_to_send,
                    attention_maps=attention_maps_for_obs,
                    grad_cam_edge_warnings=grad_cam_edge_warnings_for_obs,
                    grad_cam_edge_stats=grad_cam_edge_stats_for_obs,
                    grad_cam_edge_threshold=policy.config.grad_cam_edge_mean_threshold
                    if isinstance(policy, ACTPolicy)
                    else None,
                    grad_cam_edge_margin_px=policy.config.grad_cam_edge_margin_px
                    if isinstance(policy, ACTPolicy)
                    else None,
                    grad_cam_edge_roi_mode=policy.config.attention_roi_mode
                    if isinstance(policy, ACTPolicy)
                    else "mitigation",
                    attention_overlay_helper=policy.attention_overlay_helper
                    if isinstance(policy, ACTPolicy)
                    else ACTEigenCAMHelper,
                    control_step=rerun_step,
                )
            rerun_step += 1

        timestamp = time.perf_counter() - start_episode_t

        if not use_fanuc_backend:
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(target_dt - elapsed, 0.0))

        now = time.perf_counter()
        if now - last_diagnostic_time >= diagnostic_interval_s:
            actual_fps = action_sent_count / timestamp if timestamp > 0 else 0
            pending_info = len(pending) if use_fanuc_backend else 0
            skipped_info = action_skipped_buffer_full if use_fanuc_backend else 0

            print(
                f"[FPS] t={timestamp:.1f}s actual={actual_fps:.2f}fps target={fps} "
                f"sent={action_sent_count} pending={pending_info} skipped_buffer={skipped_info}"
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
    attention_visualization_enabled = False
    if cfg.policy is not None:
        if cfg.policy.pretrained_path is None:
            raise ValueError("Policy config is missing `pretrained_path`. Please pass `--policy.path=...`.")

        try:
            policy_cls = get_policy_class(cfg.policy.type)
            if cfg.enable_attention_visualization and cfg.policy.type == "act":
                cam_name = attention_cam_method_display_name(cfg.policy.attention_cam_method)
                warning_label = cfg.policy.attention_camera or "all"
                logging.info(
                    "Enabling ACT %s on all cameras; ROI mode=%s on: %s",
                    cam_name,
                    cfg.policy.attention_roi_mode,
                    warning_label,
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
        except Exception as e:
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
        if cfg.display_data and not _ui_telemetry_enabled():
            if os.environ.get("SLEROBOT_RERUN_CONNECT_ONLY") == "1":
                grpc_port = int(os.environ.get("SLEROBOT_RERUN_GRPC_PORT", "9876"))
                _init_rerun(
                    session_name="slerobot_record",
                    connect_ip="127.0.0.1",
                    connect_port=grpc_port,
                )
            else:
                _init_rerun(
                    session_name="slerobot_record",
                    connect_ip=cfg.display_ip,
                    connect_port=cfg.display_port,
                )
        # Initialize Rerun for real-time attention visualization if enabled
        # if cfg.realtime_attention_display and HAS_RERUN and cfg.enable_attention_visualization:
        #     try:
        #         import logging as py_logging
                
        #         # Suppress Rerun's verbose logging
        #         py_logging.getLogger("rerun").setLevel(py_logging.WARNING)
                
        #         rr.init(f"slerobot_record_{cfg.dataset.repo_id}", spawn=True)
                
        #         # Configure Rerun layout for attention visualization
        #         try:
        #             rr.script_setup(
        #                 entitlements=[]
        #             )
        #         except:
        #             pass  # Layout setup is optional
                
        #         logging.info(f"Rerun initialized for real-time visualization on http://localhost:9090")
        #         logging.info(f"You can also view data at: http://127.0.0.1:9090")
                
        #     except Exception as e:
        #         logging.warning(f"Failed to initialize Rerun: {e}")
        #         import traceback
        #         logging.debug(traceback.format_exc())
        #         cfg.realtime_attention_display = False

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        # Prepare attention visualization output directory if enabled
        attention_output_dir = None
        if attention_visualization_enabled:
            import os
            attention_output_dir = Path(cfg.dataset.root or "./") / cfg.dataset.repo_id / "attention_visualizations"
            logging.info(f"Attention visualizations will be saved to: {attention_output_dir}")

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(
                    f"Recording episode {recorded_episodes + 1}",
                    cfg.play_sounds,
                    voice=cfg.tts_voice,
                    rate=cfg.tts_rate,
                )
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
                    attention_visualization_enabled=attention_visualization_enabled,
                    attention_output_dir=attention_output_dir,
                    realtime_attention_display=cfg.realtime_attention_display,
                    play_sounds=cfg.play_sounds,
                    tts_voice=cfg.tts_voice,
                    tts_rate=cfg.tts_rate,
                )

                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say(
                        "Reset the environment",
                        cfg.play_sounds,
                        voice=cfg.tts_voice,
                        rate=cfg.tts_rate,
                    )

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
                        attention_visualization_enabled=attention_visualization_enabled,
                        attention_output_dir=attention_output_dir,
                        realtime_attention_display=cfg.realtime_attention_display,
                        play_sounds=cfg.play_sounds,
                        tts_voice=cfg.tts_voice,
                        tts_rate=cfg.tts_rate,
                    )

                if events["rerecord_episode"]:
                    log_say(
                        "Re-record episode",
                        cfg.play_sounds,
                        voice=cfg.tts_voice,
                        rate=cfg.tts_rate,
                    )
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say(
            "Stop recording",
            cfg.play_sounds,
            blocking=True,
            voice=cfg.tts_voice,
            rate=cfg.tts_rate,
        )

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

        if cfg.display_data and not _ui_telemetry_enabled():
            shutdown_rerun()

        if dataset is not None and cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
        log_say("Exiting, goodbye!", cfg.play_sounds, voice=cfg.tts_voice, rate=cfg.tts_rate)

    return dataset

    

if __name__ == "__main__":
    record()
