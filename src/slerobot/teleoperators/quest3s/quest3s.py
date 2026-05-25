
# import json
# import queue
# import threading
# import time
# import logging
# from datetime import datetime

# import paho.mqtt.client as mqtt

# from ..controller import Controller

# logger = logging.getLogger(__name__)


# class Quest3sController(Controller):
#     """Controller for Quest 3s hand tracking via MQTT."""
    
#     def __init__(self, mqtt_broker: str = "10.22.9.10", mqtt_port: int = 1883, mqtt_topic: str = "quest/data"):
#         """
#         Initialize the Quest3s controller.
        
#         Args:
#             mqtt_broker: MQTT broker address
#             mqtt_port: MQTT broker port
#             mqtt_topic: MQTT topic to subscribe to
#         """
#         self.mqtt_broker = mqtt_broker
#         self.mqtt_port = mqtt_port
#         self.mqtt_topic = mqtt_topic
        
#         self.client = None
#         self.data_queue = queue.Queue()
#         self.listener_thread = None
#         self.is_connected = False
#         self.latest_action = None
        
#     def connect(self):
#         """Connect to the Quest 3s MQTT server and subscribe to the topic for receiving key inputs."""
#         try:
#             logger.info(f"[Quest3s] Connecting to {self.mqtt_broker}:{self.mqtt_port} on topic {self.mqtt_topic}")
            
#             self.client = mqtt.Client()
#             self.client.on_connect = self._on_connect
#             self.client.on_message = self._on_message
#             self.client.on_disconnect = self._on_disconnect
            
#             self.client.connect(self.mqtt_broker, self.mqtt_port, 60)
            
#             # Start background listener thread
#             self.listener_thread = threading.Thread(target=self._mqtt_loop, daemon=True)
#             self.listener_thread.start()
            
#             self.is_connected = True
#             logger.info("[Quest3s] Connected and listening.")
#         except Exception as e:
#             logger.error(f"[Quest3s] Connection failed: {e}")
#             raise
    
#     def disconnect(self):
#         """Disconnect from the Quest 3s MQTT server and clean up resources."""
#         try:
#             if self.client:
#                 self.client.disconnect()
#                 self.is_connected = False
#             logger.info("[Quest3s] Disconnected.")
#         except Exception as e:
#             logger.error(f"[Quest3s] Disconnection failed: {e}")
    
#     def get_action(self):
#         """Get the current action based on received data from quest 3s MQTT.
        
#         Returns:
#             The latest action from MQTT this call, or None if no new data received.
#         """
#         start = time.perf_counter()
        
#         new_action = None
#         try:
#             # Drain the queue and keep the latest message
#             while True:
#                 new_action = self.data_queue.get_nowait()
#         except queue.Empty:
#             pass
        
#         # Only update latest_action if we got new data
#         if new_action is not None:
#             self.latest_action = new_action
        
#         dt_ms = (time.perf_counter() - start) * 1e3
#         logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        
#         # Return only new data (None if no new data this call)
#         return new_action
    
#     def _on_connect(self, client, userdata, flags, rc):
#         """MQTT connection callback."""
#         logger.info(f"[Quest3s] MQTT Connected with result code {rc}")
#         client.subscribe(self.mqtt_topic)
    
#     def _on_message(self, client, userdata, msg):
#         """MQTT message callback."""
#         try:
#             payload = json.loads(msg.payload.decode("utf-8"))
#             timestamp = datetime.now()
#             action = self._parse_payload(payload, timestamp)
#             self.data_queue.put(action)
#             logger.debug(f"[Quest3s] Received action: {action}")
#         except Exception as e:
#             logger.error(f"[Quest3s] Message parsing failed: {e}")
    
#     def _on_disconnect(self, client, userdata, rc):
#         """MQTT disconnect callback."""
#         if rc != 0:
#             logger.warning(f"[Quest3s] Unexpected disconnection (code {rc})")
    
#     def _mqtt_loop(self):
#         """Background thread for MQTT event loop."""
#         try:
#             self.client.loop_forever()
#         except Exception as e:
#             logger.error(f"[Quest3s] MQTT loop error: {e}")
    
#     def _parse_payload(self, payload: dict, timestamp: datetime) -> dict:
#         """
#         Parse incoming MQTT payload to action format.
        
#         Args:
#             payload: JSON payload from MQTT
#             timestamp: Timestamp of message
            
#         Returns:
#             Dictionary representing the action
#         """
#         action = {
#             'timestamp': timestamp,
#             'position': {
#                 'x': payload.get('x', 0.0),
#                 'y': payload.get('y', 0.0),
#                 'z': payload.get('z', 0.0),
#             },
#             'rotation': {
#                 'w': payload.get('w', 0.0),
#                 'p': payload.get('p', 0.0),
#                 'r': payload.get('r', 0.0),
#             },
#             'buttons': {
#                 'trigger': payload.get('triggerButton', 0),
#                 'grip': payload.get('gripButton', 0),
#             }
#         }
#         return action
    
#     # def _get_default_action(self) -> dict:
#     #     """Return a default action when no data is available."""
#     #     return {
#     #         'timestamp': datetime.now(),
#     #         'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
#     #         'rotation': {'w': 0.0, 'p': 0.0, 'r': 0.0},
#     #         'buttons': {'trigger': 0, 'grip': 0}
#     #     }
    
#     def __repr__(self):
#         return f"Quest3sController({self.mqtt_broker}:{self.mqtt_port})"
import json
import threading
import time
import logging
from datetime import datetime

import paho.mqtt.client as mqtt

from ..teleoperator import Teleoperator

logger = logging.getLogger(__name__)


class Quest3sController(Teleoperator):

    def __init__(self, mqtt_broker: str = "10.22.9.10", mqtt_port: int = 1883, mqtt_topic: str = "quest/data"):
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic

        self.client = None
        self.listener_thread = None
        self.is_connected = False

        self._lock = threading.Lock()
        self._latest_action = None
        self._last_action_time = None

    def connect(self):
        try:
            logger.info(f"[Quest3s] Connecting to {self.mqtt_broker}:{self.mqtt_port} on topic {self.mqtt_topic}")
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
            logger.error(f"[Quest3s] Connection failed: {e}")
            raise

    def disconnect(self):
        try:
            if self.client:
                self.client.disconnect()
                self.is_connected = False
            logger.info("[Quest3s] Disconnected.")
        except Exception as e:
            logger.error(f"[Quest3s] Disconnection failed: {e}")

    def get_action(self):
        """返回最新一帧；诊断阶段不清空，便于排除 MQTT 稀疏导致的降频。"""
        with self._lock:
            return self._latest_action

    # def _on_connect(self, client, userdata, flags, rc):
    #     logger.info(f"[Quest3s] MQTT Connected with result code {rc}")
    #     client.subscribe(self.mqtt_topic)
    def _on_connect(self, client, userdata, flags, rc):
        logger.info(f"[Quest3s] MQTT Connected with result code {rc}")
        result, mid = client.subscribe(self.mqtt_topic)
        logger.info(f"[Quest3s] Subscribe result={result} mid={mid}")  # 加这行
    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            #print(f"[Quest3s] FULL PAYLOAD KEYS: {list(payload.keys())}", flush=True)  

            action = self._parse_payload(payload, datetime.now())
            with self._lock:
                self._latest_action = action
                self._last_action_time = time.perf_counter()
        except Exception as e:
            logger.error(f"[Quest3s] Message parsing failed: {e}")

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logger.warning(f"[Quest3s] Unexpected disconnection (code {rc})")

    def _mqtt_loop(self):
        try:
            self.client.loop_forever()
        except Exception as e:
            logger.error(f"[Quest3s] MQTT loop error: {e}")

    # def _parse_payload(self, payload: dict, timestamp: datetime) -> dict:
    #     return {
    #         'timestamp': timestamp,
    #         'position': {
    #             'x': payload.get('x', 0.0),
    #             'y': payload.get('y', 0.0),
    #             'z': payload.get('z', 0.0),
    #         },
    #         'rotation': {
    #             'w': payload.get('w', 0.0),
    #             'p': payload.get('p', 0.0),
    #             'r': payload.get('r', 0.0),
    #         },
    #         'buttons': {
    #             'trigger': payload.get('triggerButton', 0),
    #             'grip': payload.get('gripButton', 0),
    #         }
    #     }
    def _parse_payload(self, payload: dict, timestamp: datetime) -> dict:
        result = {
            'timestamp': timestamp,
            'position': {
                'x': payload.get('x', 0.0),
                'y': payload.get('y', 0.0),
                'z': payload.get('z', 0.0),
            },
            'rotation': {
                'w': payload.get('w', 0.0),
                'p': payload.get('p', 0.0),
                'r': payload.get('r', 0.0),
            },
            'buttons': {
                'trigger': payload.get('triggerButton', 0),
                'grip': payload.get('gripButton', 0),
                # Meta Quest A / primary button (field names vary by MQTT publisher)
                'a': int(
                    payload.get(
                        'aButton',
                        payload.get(
                            'buttonA',
                            payload.get('primaryButton', payload.get('A', payload.get('button_a', 0))),
                        ),
                    )
                ),
            },
        }
        # Thumbsticks / joysticks for mobile base (field names vary by Quest app)
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