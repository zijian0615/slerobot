#!/usr/bin/env python
"""Interactive LeKiwi motor calibration (run on the Raspberry Pi next to the robot).

Example:
```shell
python -m slerobot.scripts.slerobot_lekiwi_calibrate --recalibrate
python -m slerobot.scripts.slerobot_lekiwi_calibrate --robot-port /dev/ttyACM0 --recalibrate
```
"""

from __future__ import annotations

import argparse
import logging

from slerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
from slerobot.robots.lekiwi.lekiwi import LeKiwi
from slerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate LeKiwi arm + base motors.")
    parser.add_argument("--robot-port", type=str, default="/dev/ttyACM0")
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Ignore existing calibration file and run a full new calibration.",
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Skip cameras (recommended on Pi).",
    )
    args = parser.parse_args(argv)
    init_logging()

    robot_config = LeKiwiConfig(port=args.robot_port)
    if args.no_cameras:
        robot_config.cameras = {}

    robot = LeKiwi(robot_config)
    logger.info("Connecting to LeKiwi on %s ...", args.robot_port)
    if robot.is_connected:
        raise RuntimeError(f"{robot} is already connected")
    robot.bus.connect()
    robot.calibrate(force=args.recalibrate)
    robot.configure()
    robot.disconnect()

    print("\n=== Calibration complete ===")
    print(f"Saved to: {robot.calibration_fpath}")
    print(
        "\nFor Quest teleop on Mac, copy this file to:\n"
        "  ~/.cache/huggingface/lerobot/calibration/robots/lekiwi_client/lekiwi_client.json"
    )


if __name__ == "__main__":
    main()
