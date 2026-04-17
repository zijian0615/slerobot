import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from slerobot.configs import parser
from slerobot.datasets.slerobot_datasets import sLerobotDataset
from slerobot.robots.fanuc import Fanuc
from slerobot.utils.control_utils import init_keyboard_listener, is_headless
from slerobot.utils.robot_utils import precise_sleep
from slerobot.utils.utils import init_logging, log_say


@dataclass
class DatasetReplayConfig:
	repo_id: str
	episode: int = 0
	root: str | None = None
	fps: int | None = None
	use_dataset_timestamps: bool = True
	download_videos: bool = False
	max_inflight_actions: int = 4
	ack_timeout_s: float = 20.0
	start_delay_s: float = 2.0


@dataclass
class FanucReplayConfig:
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
	gripper_port_number: int | None = None
	gripper_open_port_number: int | None = 3
	gripper_close_port_number: int | None = 4
	gripper_open_value: str = "ON"
	gripper_close_value: str = "ON"


@dataclass
class ReplayConfig:
	dataset: DatasetReplayConfig
	robot: FanucReplayConfig = field(default_factory=FanucReplayConfig)
	play_sounds: bool = True


def _to_python_value(value: Any) -> Any:
	if hasattr(value, "detach"):
		value = value.detach().cpu()

	if hasattr(value, "tolist"):
		try:
			return value.tolist()
		except Exception:
			pass

	if hasattr(value, "item"):
		try:
			return value.item()
		except Exception:
			pass

	return value


def _build_action_dict(action_names: list[str], action_values: Any) -> dict[str, float]:
	values = _to_python_value(action_values)
	if not isinstance(values, (list, tuple)):
		values = [values]

	return {
		name: float(values[idx])
		for idx, name in enumerate(action_names)
		if idx < len(values)
	}


def _build_robot_action(action_dict: dict[str, float], cfg: FanucReplayConfig) -> dict[str, Any]:
	robot_action = {
		"j0": float(action_dict.get("j0", 0.0)),
		"j1": float(action_dict.get("j1", 0.0)),
		"j2": float(action_dict.get("j2", 0.0)),
		"j3": float(action_dict.get("j3", 0.0)),
		"j4": float(action_dict.get("j4", 0.0)),
		"j5": float(action_dict.get("j5", 0.0)),
		"speed": cfg.speed,
		"term_type": cfg.term_type,
		"term_value": cfg.term_value,
	}

	grip_value = float(action_dict.get("j7", 0.0))
	grip_pressed = int(grip_value >= 0.5)
	robot_action["j7"] = float(grip_pressed)

	selected_gripper_port = None
	if cfg.gripper_open_port_number is not None and cfg.gripper_close_port_number is not None:
		selected_gripper_port = cfg.gripper_close_port_number if grip_pressed else cfg.gripper_open_port_number
	elif cfg.gripper_port_number is not None:
		selected_gripper_port = cfg.gripper_port_number

	if (
		cfg.gripper_lcb_type is not None
		and cfg.gripper_port_type is not None
		and selected_gripper_port is not None
	):
		robot_action.update(
			{
				"lcb_type": cfg.gripper_lcb_type,
				"lcb_value": cfg.gripper_lcb_value,
				"port_type": cfg.gripper_port_type,
				"port_number": selected_gripper_port,
				"port_value": cfg.gripper_close_value if grip_pressed else cfg.gripper_open_value,
			}
		)

	return robot_action


def _wait_oldest_pending(pending: deque[tuple[int, Any]], timeout_s: float) -> None:
	seq_id, future = pending.popleft()
	err_id = future.result(timeout=timeout_s)
	if err_id != 0:
		raise RuntimeError(f"Replay action failed, seq_id={seq_id}, error_id={err_id}")


def replay_episode(
	robot: Fanuc,
	dataset: sLerobotDataset,
	episode_index: int,
	cfg: ReplayConfig,
	events: dict[str, bool],
) -> int:
	if "action" not in dataset.features:
		raise ValueError("Dataset does not contain `action` feature, cannot replay trajectory.")

	action_names = dataset.features["action"].get("names") or []
	if not action_names:
		raise ValueError("Dataset `action` feature does not define action names.")

	num_frames = len(dataset.hf_dataset)
	if num_frames == 0:
		raise ValueError(f"Episode {episode_index} has no frames to replay.")

	pending: deque[tuple[int, Any]] = deque()
	replay_fps = cfg.dataset.fps or dataset.fps
	previous_timestamp: float | None = None
	next_target_t: float | None = None
	replay_started_t = time.perf_counter()
	last_log_t = replay_started_t

	for frame_offset in range(num_frames):
		if events["stop_recording"] or events["exit_early"]:
			break

		row = dataset.hf_dataset[frame_offset]
		current_timestamp = _to_python_value(row.get("timestamp")) if "timestamp" in row else None

		if frame_offset > 0:
			if cfg.dataset.use_dataset_timestamps and current_timestamp is not None and previous_timestamp is not None:
				delta_s = max(0.0, float(current_timestamp) - float(previous_timestamp))
			else:
				delta_s = 1.0 / replay_fps

			if next_target_t is None:
				next_target_t = time.perf_counter() + delta_s
			else:
				next_target_t += delta_s

			precise_sleep(next_target_t - time.perf_counter())
		else:
			next_target_t = time.perf_counter()

		action_dict = _build_action_dict(action_names, row["action"])
		robot_action = _build_robot_action(action_dict, cfg.robot)
		future = robot.send_action(robot_action)
		pending.append((robot.seq_id - 1, future))

		while len(pending) > max(1, cfg.dataset.max_inflight_actions):
			_wait_oldest_pending(pending, cfg.dataset.ack_timeout_s)

		previous_timestamp = float(current_timestamp) if current_timestamp is not None else previous_timestamp

		now = time.perf_counter()
		if now - last_log_t >= 1.0:
			elapsed = now - replay_started_t
			actual_fps = (frame_offset + 1) / elapsed if elapsed > 0 else 0.0
			logging.info(
				"[REPLAY] episode=%s frame=%s/%s actual_fps=%.2f target_fps=%s",
				episode_index,
				frame_offset + 1,
				num_frames,
				actual_fps,
				replay_fps,
			)
			last_log_t = now

	while pending:
		_wait_oldest_pending(pending, cfg.dataset.ack_timeout_s)

	return num_frames


@parser.wrap()
def replay(cfg: ReplayConfig) -> sLerobotDataset:
	init_logging()

	dataset = sLerobotDataset(
		repo_id=cfg.dataset.repo_id,
		root=cfg.dataset.root,
		episodes=[cfg.dataset.episode],
		download_videos=cfg.dataset.download_videos,
	)

	if cfg.dataset.episode < 0 or cfg.dataset.episode >= dataset.meta.total_episodes:
		raise IndexError(
			f"Episode index {cfg.dataset.episode} out of range. "
			f"Available episodes: 0 to {dataset.meta.total_episodes - 1}."
		)

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

	listener = None
	events = None

	try:
		robot.connect()
		listener, events = init_keyboard_listener()

		log_say(
			f"Start replay episode {cfg.dataset.episode} from {cfg.dataset.repo_id}",
			cfg.play_sounds,
		)

		if cfg.dataset.start_delay_s > 0:
			logging.info("Replay will start in %.1f seconds", cfg.dataset.start_delay_s)
			precise_sleep(cfg.dataset.start_delay_s)

		replayed_frames = replay_episode(
			robot=robot,
			dataset=dataset,
			episode_index=cfg.dataset.episode,
			cfg=cfg,
			events=events,
		)

		logging.info(
			"Replay finished: repo_id=%s episode=%s frames=%s",
			cfg.dataset.repo_id,
			cfg.dataset.episode,
			replayed_frames,
		)
		return dataset
	finally:
		log_say("Stop replay", cfg.play_sounds, blocking=True)

		if robot.is_connected:
			robot.disconnect()

		if listener is not None and not is_headless():
			listener.stop()


if __name__ == "__main__":
	replay()
