#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LeKiwi ZMQ host on the robot (e.g. Raspberry Pi).

Runs on the mobile base + follower arm. Quest (or another client) sends actions over ZMQ;
this process executes them on the real hardware. No leader arm is required.

Typical Quest workflow:
  1. On the Pi:  slerobot-lekiwi-host
  2. On the PC:  slerobot-lekiwi-record --robot.remote_ip=<pi-ip> ...
"""

import argparse
import base64
import json
import logging
import time

import cv2
import zmq

from .config_lekiwi import LeKiwiConfig, LeKiwiHostConfig
from .lekiwi import LeKiwi

logger = logging.getLogger(__name__)


class LeKiwiHost:
    def __init__(self, config: LeKiwiHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="LeKiwi ZMQ host for Quest / remote teleop (no leader arm)."
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run interactive leader-style calibration once (not needed for Quest teleop).",
    )
    parser.add_argument(
        "--connection-time-s",
        type=int,
        default=0,
        help="Host loop duration in seconds (0 = until Ctrl+C).",
    )
    parser.add_argument("--port-cmd", type=int, default=5555)
    parser.add_argument("--port-observations", type=int, default=5556)
    parser.add_argument("--robot-port", type=str, default="/dev/ttyACM0")
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Connect motors only (skip cameras if /dev/video* resolution mismatches).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    host_config = LeKiwiHostConfig(
        port_zmq_cmd=args.port_cmd,
        port_zmq_observations=args.port_observations,
        connection_time_s=args.connection_time_s,
        calibrate_on_connect=args.calibrate,
    )
    robot_config = LeKiwiConfig(port=args.robot_port)
    if args.no_cameras:
        robot_config.cameras = {}
        logger.info("Running without cameras (--no-cameras).")

    logger.info("Configuring LeKiwi (Quest / remote teleop host)")
    robot = LeKiwi(robot_config)

    logger.info("Connecting LeKiwi (calibrate_on_connect=%s)", host_config.calibrate_on_connect)
    robot.connect(calibrate=host_config.calibrate_on_connect)

    host = LeKiwiHost(host_config)
    last_cmd_time = time.time()
    watchdog_active = False
    client_seen = False
    logger.info(
        "Host ready on ZMQ cmd=%s obs=%s — start slerobot-lekiwi-record on your Quest machine.",
        host_config.port_zmq_cmd,
        host_config.port_zmq_observations,
    )

    try:
        start = time.perf_counter()
        while host_config.connection_time_s == 0 or (time.perf_counter() - start) < host_config.connection_time_s:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
                client_seen = True
            except zmq.Again:
                pass
            except Exception as e:
                logger.error("Message fetching failed: %s", e)

            now = time.time()
            if (
                client_seen
                and (now - last_cmd_time > host.watchdog_timeout_ms / 1000)
                and not watchdog_active
            ):
                logger.warning(
                    "No command for >%sms — stopping base.",
                    host.watchdog_timeout_ms,
                )
                watchdog_active = True
                robot.stop_base()

            try:
                last_observation = robot.get_observation()
            except ConnectionError as exc:
                logger.error(
                    "Motor bus read failed (check USB /dev/ttyACM0, power, and motor IDs): %s",
                    exc,
                )
                time.sleep(0.05)
                continue

            for cam_key, cam in robot.cameras.items():
                if not cam.is_connected or cam_key not in last_observation:
                    continue
                frame = last_observation[cam_key]
                ret, buffer = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                )
                if ret:
                    last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                else:
                    last_observation[cam_key] = ""

            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            elapsed = time.time() - loop_start_time
            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — exiting.")
    finally:
        robot.disconnect()
        host.disconnect()
        logger.info("LeKiwi host stopped.")


if __name__ == "__main__":
    main()
