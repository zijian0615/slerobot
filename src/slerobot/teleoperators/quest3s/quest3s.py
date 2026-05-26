import json
import logging
import threading
import time
from datetime import datetime

import paho.mqtt.client as mqtt

from ..teleoperator import Teleoperator

logger = logging.getLogger(__name__)


class Quest3sController(Teleoperator):
    """Quest 3s hand tracking via MQTT (Fanuc-style pose + buttons)."""

    def __init__(
        self,
        mqtt_broker: str = "10.22.9.10",
        mqtt_port: int = 1883,
        mqtt_topic: str = "quest/data",
    ):
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        self.client = None
        self.listener_thread = None
        self.is_connected = False
        self._lock = threading.Lock()
        self._latest_action: dict | None = None
        self._last_action_time: float | None = None

    @staticmethod
    def _neutral_action() -> dict:
        return {
            "timestamp": datetime.now(),
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"w": 0.0, "p": 0.0, "r": 0.0},
            "buttons": {"trigger": 0, "grip": 0, "a": 0},
        }

    def connect(self):
        try:
            logger.info(
                "[Quest3s] Connecting to %s:%s topic %s",
                self.mqtt_broker,
                self.mqtt_port,
                self.mqtt_topic,
            )
            self.client = mqtt.Client()
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
            self.client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.listener_thread = threading.Thread(target=self._mqtt_loop, daemon=True)
            self.listener_thread.start()
            self.is_connected = True
            logger.info("[Quest3s] Connected and listening.")
        except Exception as e:
            logger.error("[Quest3s] Connection failed: %s", e)
            raise

    def disconnect(self):
        try:
            if self.client:
                self.client.disconnect()
            self.is_connected = False
            logger.info("[Quest3s] Disconnected.")
        except Exception as e:
            logger.error("[Quest3s] Disconnection failed: %s", e)

    def get_action(self) -> dict:
        """Latest MQTT frame, or neutral pose if none received yet."""
        with self._lock:
            return self._latest_action if self._latest_action is not None else self._neutral_action()

    def _on_connect(self, client, userdata, flags, rc):
        logger.info("[Quest3s] MQTT connected rc=%s", rc)
        result, mid = client.subscribe(self.mqtt_topic)
        logger.info("[Quest3s] subscribe result=%s mid=%s", result, mid)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            if not getattr(self, "_logged_first_payload", False):
                logger.info("[Quest3s] first MQTT payload keys: %s", list(payload.keys()))
                logger.info("[Quest3s] first MQTT payload: %s", payload)
                self._logged_first_payload = True
            action = self._parse_payload(payload, datetime.now())
            with self._lock:
                self._latest_action = action
                self._last_action_time = time.perf_counter()
        except Exception as e:
            logger.error("[Quest3s] Message parsing failed: %s", e)

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logger.warning("[Quest3s] Unexpected disconnection (code %s)", rc)

    def _mqtt_loop(self):
        try:
            self.client.loop_forever()
        except Exception as e:
            logger.error("[Quest3s] MQTT loop error: %s", e)

    @staticmethod
    def _pick_float(payload: dict, *keys: str, default: float = 0.0) -> float:
        for key in keys:
            if key in payload and payload[key] is not None:
                return float(payload[key])
        return default

    @staticmethod
    def _vec_norm3(x: float, y: float, z: float) -> float:
        return abs(x) + abs(y) + abs(z)

    def _parse_position(self, payload: dict) -> tuple[float, float, float]:
        fanuc_present = any(k in payload for k in ("x", "y", "z"))
        quest_present = any(k in payload for k in ("px", "py", "pz"))
        fx = self._pick_float(payload, "x", default=0.0)
        fy = self._pick_float(payload, "y", default=0.0)
        fz = self._pick_float(payload, "z", default=0.0)
        px = self._pick_float(payload, "px", "posX", "positionX", default=0.0)
        py = self._pick_float(payload, "py", "posY", "positionY", default=0.0)
        pz = self._pick_float(payload, "pz", "posZ", "positionZ", default=0.0)
        if quest_present and (
            not fanuc_present
            or (
                self._vec_norm3(fx, fy, fz) < 1e-9
                and self._vec_norm3(px, py, pz) > self._vec_norm3(fx, fy, fz)
            )
        ):
            return px, py, pz
        if fanuc_present:
            return fx, fy, fz
        return (
            self._pick_float(payload, "x", "px", "posX", "positionX"),
            self._pick_float(payload, "y", "py", "posY", "positionY"),
            self._pick_float(payload, "z", "pz", "posZ", "positionZ"),
        )

    def _parse_rotation(self, payload: dict) -> tuple[float, float, float]:
        fanuc_present = any(k in payload for k in ("w", "p", "r"))
        quest_present = any(k in payload for k in ("rw", "rp", "rr"))
        fw = self._pick_float(payload, "w", default=0.0)
        fp = self._pick_float(payload, "p", default=0.0)
        fr = self._pick_float(payload, "r", default=0.0)
        rw = self._pick_float(payload, "rw", "rotW", default=0.0)
        rp = self._pick_float(payload, "rp", "rotP", default=0.0)
        rr = self._pick_float(payload, "rr", "rotR", default=0.0)
        if quest_present and (
            not fanuc_present
            or (
                self._vec_norm3(fw, fp, fr) < 1e-9
                and self._vec_norm3(rw, rp, rr) > self._vec_norm3(fw, fp, fr)
            )
        ):
            return rw, rp, rr
        if fanuc_present:
            return fw, fp, fr
        return (
            self._pick_float(payload, "w", "rw", "rotW"),
            self._pick_float(payload, "p", "rp", "rotP"),
            self._pick_float(payload, "r", "rr", "rotR"),
        )

    def _parse_payload(self, payload: dict, timestamp: datetime) -> dict:
        x, y, z = self._parse_position(payload)
        w, p, r = self._parse_rotation(payload)
        result = {
            "timestamp": timestamp,
            "position": {"x": x, "y": y, "z": z},
            "rotation": {"w": w, "p": p, "r": r},
            "buttons": {
                "trigger": int(payload.get(
                    "triggerButton",
                    payload.get("trigger", payload.get("Trigger", 0)),
                )),
                "grip": int(payload.get(
                    "gripButton",
                    payload.get("grip", payload.get("Grip", 0)),
                )),
                "a": int(payload.get(
                    "a",
                    payload.get(
                        "A",
                        payload.get(
                            "aButton",
                            payload.get(
                                "buttonA",
                                payload.get(
                                    "primaryButton",
                                    payload.get(
                                        "secondaryButton",
                                        payload.get("button_a", 0),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )),
            },
        }
        # Pass through extra button fields for debug visibility
        for key in ("secondaryButton", "primaryButton"):
            if key in payload:
                result[key] = payload[key]
        for key in (
            "joystickX",
            "joystickY",
            "joystick_x",
            "joystick_y",
            "thumbstickX",
            "thumbstickY",
            "leftThumbstickX",
            "leftThumbstickY",
            "rightThumbstickX",
            "rightThumbstickY",
            "axisX",
            "axisY",
            "moveX",
            "moveY",
        ):
            if key in payload:
                result[key] = float(payload[key])
        return result

    def __repr__(self):
        return f"Quest3sController({self.mqtt_broker}:{self.mqtt_port})"
