"""Fixed-rate record loop for LeKiwi + Quest 3s + keyboard base teleop."""

from __future__ import annotations

import logging
import time
from typing import Any

from slerobot.datasets.slerobot_datasets import sLerobotDataset
from slerobot.datasets.utils import build_dataset_frame
from slerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from slerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from slerobot.teleoperators.lekiwi_keyboard import LeKiwiKeyboardTeleop
from slerobot.teleoperators.quest3s import Quest3sController
from slerobot.utils.constants import ACTION, OBS_STR
from slerobot.utils.lekiwi_action_debug import log_lekiwi_action_debug
from slerobot.utils.lekiwi_quest_utils import (
    LeKiwiQuestMapper,
    base_action_is_active,
)
from slerobot.utils.visualization_utils import log_rerun_data

logger = logging.getLogger(__name__)


def lekiwi_record_loop(
    robot: LeKiwiClient,
    quest: Quest3sController,
    keyboard: LeKiwiKeyboardTeleop,
    mapper: LeKiwiQuestMapper,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: sLerobotDataset | None = None,
    control_time_s: float | int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    record_data: bool = True,
) -> None:
    """Run one episode (or reset segment) at ``fps`` with aligned obs, action, and ZMQ."""
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"Dataset fps must match loop fps ({dataset.fps} != {fps}).")

    target_dt = 1.0 / fps
    start_t = time.perf_counter()
    timestamp = 0.0
    rerun_step = 0
    frames_recorded = 0

    while True:
        loop_start = time.perf_counter()
        if control_time_s is not None and timestamp >= control_time_s:
            break
        if events.get("exit_early"):
            events["exit_early"] = False
            break

        obs = robot_observation_processor(robot.get_observation())

        quest_act = quest.get_action()
        keyboard_pressed = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_pressed)
        base_for_mapper = base_action if base_action_is_active(base_action) else None
        mapped = mapper.map_action(quest_act, obs, base_action=base_for_mapper)
        action_processed = teleop_action_processor((mapped, obs))
        robot_action = robot_action_processor((dict(action_processed), obs))

        log_lekiwi_action_debug(
            "mac_after_mapper",
            quest_action=quest_act,
            observation=obs,
            mapped_action=action_processed,
            quest_mapper=mapper,
        )
        robot.send_action(robot_action)
        log_lekiwi_action_debug("mac_before_send_action", mapped_action=robot_action, observation=obs)

        if record_data and dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, robot_action, prefix=ACTION)
            dataset.add_frame({**observation_frame, **action_frame, "task": single_task})
            frames_recorded += 1

        if display_data:
            log_rerun_data(obs, robot_action, control_step=rerun_step)
            rerun_step += 1

        timestamp = time.perf_counter() - start_t
        time.sleep(max(target_dt - (time.perf_counter() - loop_start), 0.0))

    logger.info(
        "LeKiwi segment done: %.1fs, %d frames sent%s",
        timestamp,
        int(timestamp * fps),
        f", {frames_recorded} logged" if record_data and dataset is not None else "",
    )
