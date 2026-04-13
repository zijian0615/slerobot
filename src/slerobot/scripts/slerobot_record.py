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
        fps: Limit the frames per second (default: 30)
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
    fps: int = 30
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
        cameras: Dictionary of camera configurations (default: None)
            Example: {"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}
    """
    host: str = "192.168.1.100"
    port: int = 16001
    group: int = 1
    utool: int = 1
    uframe: int = 0
    speed: int = 150
    term_type: str = "CNT"
    term_value: int = 100
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
    robot:Robot,
    events:dict,
    fps:int,
    dataset:sLerobotDataset | None=None,
    teleop: Teleoperator | None=None,
    policy: None=None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    no_action_count = 0
    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # 1. get observation
        obs = robot.get_observation()
        # get processed observation [pass]
        obs = obs
        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs, prefix = OBS_STR)
        
        # 2. get action from teleop or policy
        if policy is not None:
            action = policy.predict(obs) # TODO:[pass]
            # process policy action [pass]
            policy_action = action
        
        elif teleop is not None:
            act = teleop.get_action()
            # If no new action from teleop, skip this frame
            if act is None:
                continue
            # Convert Quest3s format to Fanuc format
            teleop_action = convert_quest3s_to_fanuc_action(act)
        
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10000 == 0:
                logging.warning(
                    "No policy or teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            continue
        
        # 3. send action, applying a pipeline to the action , default is IdentityProcessor
        action_to_send = policy_action if policy is not None else teleop_action
        _sent_action = robot.send_action(action_to_send)

        # 4. save to dataset
        if dataset is not None: 
            action_frame = build_dataset_frame(dataset.features, action_to_send, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)
        
        # 5. display data [pass]

        dt_s = time.perf_counter() - start_loop_t
        sleep_s = max(0, 1 / fps - dt_s)
        if sleep_s < 0:
            logging.warning(f"Loop is running slower than the target fps ({1/dt_s:.2f} fps), consider reducing the fps or optimizing the processing pipeline.")
        precise_sleep(sleep_s)

        timestamp = time.perf_counter() - start_episode_t

@parser.wrap()
def record(cfg: RecordConfig) -> sLerobotDataset:
        init_logging()
        logging.info(pformat(asdict(cfg)))
        
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
