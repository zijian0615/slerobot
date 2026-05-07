import json
import logging
import queue
import socket
import threading
import time
from concurrent.futures import Future
from typing import Dict, Optional, Tuple

from slerobot.cameras.utils import make_cameras_from_configs
from slerobot.utils.robot_utils import decode_fanuc_pose_dict, encode_fanuc_pose_dict
from ..robot import Robot

logger = logging.getLogger(__name__)


class Fanuc(Robot):
    STATE_POLL_HZ = 15.0

    def __init__(
        self,
        host: str,
        port: int = 16001,
        group: int = 1,
        utool: int = 1,
        uframe: int = 0,
        speed: int = 150,
        term_type: str = "CNT",
        term_value: int = 100,
        gripper_port_number: int | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._group = group
        self._utool = utool
        self._uframe = uframe
        self._speed = speed
        self._term_type = term_type
        self._term_value = term_value
        self._gripper_port_number = gripper_port_number

        self._sock: Optional[socket.socket] = None
        self._buf = b""
        self._connected = False
        self._send_lock = threading.Lock()

        self._latest_pose: Optional[Tuple[float, ...]] = None
        self._latest_t: Optional[float] = None
        self._latest_tick: Optional[int] = None
        self._latest_configuration: Optional[Dict] = None
        self._motion_configuration: Optional[Dict] = None
        self._latest_gripper_state: Optional[int] = None
        self._orientation_debug_count: int = 0

        # seq_id -> Future[int]，接收线程写入，发送方读取
        self._pending_futures: Dict[int, Future] = {}
        self._pending_lock = threading.Lock()
        self._ack_queue: queue.Queue[Tuple[int, int]] = queue.Queue()

        # 后台线程
        self._recv_thread: Optional[threading.Thread] = None
        self._state_thread: Optional[threading.Thread] = None

        # Initialize cameras as empty dict, will be set later if needed
        self.cameras = {}

        self.seq_id = 1

    # ------------------------------------------------------------------ #
    #  连接 / 断开                                                          #
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        if self._connected:
            logger.warning("Already connected - skipping")
            return
        dynamic_port = self._frc_connect()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except Exception:
            pass
        self._sock.settimeout(5.0)
        self._sock.connect((self._host, dynamic_port))

        self._send_json({"Command": "FRC_Initialize", "GroupMask": self._group})
        resp = self._recv_until(lambda r: r.get("Command") == "FRC_Initialize")
        if resp.get("ErrorID", -1) != 0:
            raise RuntimeError(f"FRC_Initialize failed: {resp}")

        self._set_uframe_utool(self._uframe, self._utool)
        self._motion_configuration = {
            "UToolNumber": self._utool,
            "UFrameNumber": self._uframe,
            "Front": 1,
            "Up": 1,
            "Left": 0,
            "Flip": 0,
            "Turn4": 0,
            "Turn5": 0,
            "Turn6": 0,
        }
        self._latest_configuration = dict(self._motion_configuration)
        self._connected = True

        # 启动后台接收线程
        self._recv_thread = threading.Thread(
            target=self._recv_loop, name="fanuc-recv", daemon=True
        )
        self._recv_thread.start()

        self._state_thread = threading.Thread(
            target=self._state_poll_loop, name="fanuc-state-poll", daemon=True
        )
        self._state_thread.start()

        print(f"[Fanuc] recv thread alive: {self._recv_thread.is_alive()}", flush=True)
        deadline = time.time() + 1.0
        while self._latest_pose is None and time.time() < deadline:
            time.sleep(0.01)
        # self.get_observation()
        # logger.info(
        #     "Fanuc connected to %s:%s with UF=%s, UT=%s",
        #     self._host,
        #     dynamic_port,
        #     self._uframe,
        #     self._utool,
        # )

    def disconnect(self) -> None:
        self._connected = False
        # 唤醒所有等待中的 Future，注入连接断开异常
        with self._pending_lock:
            for fut in self._pending_futures.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("Disconnected before ACK"))
            self._pending_futures.clear()

        if self._sock is None:
            return
        try:
            self._send_json({"Command": "FRC_Abort"})
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
        self._sock = None
        self._buf = b""

        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)
            self._recv_thread = None
        if self._state_thread is not None:
            self._state_thread.join(timeout=2.0)
            self._state_thread = None

    # ------------------------------------------------------------------ #
    #  核心接口                                                              #
    # ------------------------------------------------------------------ #

    def send_action(self, action: Dict) -> Future:
        """
        异步发送运动指令，立即返回 Future[int]。

        Future 的结果为 ErrorID（0 表示成功）。
        调用方若需同步等待：
            fut = robot.send_action(action)
            err_id = fut.result(timeout=20.0)

        若需流水线发送，可先堆积多条指令再统一等待：
            futs = [robot.send_action(a) for a in actions]
            results = [f.result(timeout=30.0) for f in futs]

        支持两类末端姿态动作格式：
            1. 直接角度：`j3`/`j4`/`j5`
            2. 连续表示：`j3_sin`/`j3_cos`、`j4_sin`/`j4_cos`、`j5_sin`/`j5_cos`

        当传入连续表示时，会在发送给 FANUC 前自动解码回 `W/P/R` 角度。
        """
        self._require_connected()
        if self._motion_configuration is None:
            self._motion_configuration = {
                "UToolNumber": self._utool,
                "UFrameNumber": self._uframe,
                "Front": 1,
                "Up": 1,
                "Left": 0,
                "Flip": 0,
                "Turn4": 0,
                "Turn5": 0,
                "Turn6": 0,
            }
        if self._latest_configuration is None:
            self._latest_configuration = dict(self._motion_configuration)

        # Support multiple action formats
        # New format: {"j0": x, "j1": y, "j2": z, "j3_sin": ..., "j3_cos": ..., ...} (from dataset)
        # Legacy format: {"state": (x, y, z, w, p, r)} or {"position": (...), "rotation": (...)}
        if "j0" in action and "j1" in action:
            raw_action = dict(action)
            action = decode_fanuc_pose_dict(action)
            if self._orientation_debug_count < 10 and any(key.endswith(("_sin", "_cos")) for key in raw_action):
                logger.info(
                    "[FANUC_DECODED_ORIENTATION] encoded=%s decoded={j3=%.4f, j4=%.4f, j5=%.4f}",
                    {
                        key: raw_action[key]
                        for key in (
                            "j3_sin",
                            "j3_cos",
                            "j4_sin",
                            "j4_cos",
                            "j5_sin",
                            "j5_cos",
                        )
                        if key in raw_action
                    },
                    float(action["j3"]),
                    float(action["j4"]),
                    float(action["j5"]),
                )
                self._orientation_debug_count += 1
            # New format: individual joint values
            x = float(action["j0"])
            y = float(action["j1"])
            z = float(action["j2"])
            w = float(action["j3"])
            p = float(action["j4"])
            r = float(action["j5"])
        elif "state" in action:
            # State format: 6D tuple/array
            state = action["state"]
            if len(state) == 6:
                x, y, z, w, p, r = state
            else:
                raise ValueError(f"Invalid state format. Expected 6 values, got {len(state)}")
        elif "position" in action:
            position = action["position"]
            if isinstance(position, dict):
                rotation = action.get("rotation")
                if not isinstance(rotation, dict):
                    raise ValueError("When 'position' is a dict, 'rotation' must also be a dict")

                x = float(position["x"])
                y = float(position["y"])
                z = float(position["z"])
                w = float(rotation["w"])
                p = float(rotation["p"])
                r = float(rotation["r"])
            elif len(position) == 6:
                # Old format: unpack all 6 values from position
                x, y, z, w, p, r = position
            elif len(position) == 3:
                # Converted format: position has 3 values, rotation is separate
                x, y, z = position
                w, p, r = action["rotation"]
            else:
                raise ValueError(f"Invalid position format. Expected 3 or 6 values, got {len(position)}")
        else:
            raise ValueError("Action must contain 'j0'-'j5', 'state', or 'position' key")

        configuration = dict(self._motion_configuration or {})
        configuration.update(
            {
                "UToolNumber": int(action.get("utool", configuration.get("UToolNumber", self._utool))),
                "UFrameNumber": int(action.get("uframe", configuration.get("UFrameNumber", self._uframe))),
                "Front": int(action.get("front", configuration.get("Front", 1))),
                "Up": int(action.get("up", configuration.get("Up", 1))),
                "Left": int(action.get("left", configuration.get("Left", 0))),
                "Flip": int(action.get("flip", configuration.get("Flip", 0))),
                "Turn4": int(action.get("turn4", configuration.get("Turn4", 0))),
                "Turn5": int(action.get("turn5", configuration.get("Turn5", 0))),
                "Turn6": int(action.get("turn6", configuration.get("Turn6", 0))),
            }
        )
        self._motion_configuration = dict(configuration)

        seq_id = self.seq_id
        self.seq_id += 1

        packet = {
            "Instruction": "FRC_LinearMotion",
            "SequenceID": seq_id,
            "Configuration": configuration,
            "Position": {
                "X": float(x), "Y": float(y), "Z": float(z),
                "W": float(w), "P": float(p), "R": float(r),
                "Ext1": 0.0, "Ext2": 0.0, "Ext3": 0.0,
            },
            "SpeedType": str(action.get("speed_type", "mmSec")),
            "Speed": int(action.get("speed", self._speed)),
            "TermType": str(action.get("term_type", self._term_type)),
            "TermValue": int(action.get("term_value", self._term_value)),
        }

        lcb_type = action.get("lcb_type", action.get("LCBType"))
        lcb_value = action.get("lcb_value", action.get("LCBValue", 0))
        port_type = action.get("port_type", action.get("PortType"))
        port_number = action.get("port_number", action.get("PortNumber"))
        port_value = action.get("port_value", action.get("PortValue"))

        if (
            lcb_type is not None
            and port_type is not None
            and port_number is not None
            and port_value is not None
        ):
            packet.update(
                {
                    "LCBType": str(lcb_type),
                    "LCBValue": int(lcb_value),
                    "PortType": int(port_type),
                    "PortNumber": int(port_number),
                    "PortValue": str(port_value),
                }
            )
            logger.info("Fanuc gripper packet=%s", packet)

        # 先注册 Future，再发送，避免接收线程在发送完成前就收到 ACK 找不到槽位
        fut: Future = Future()
        with self._pending_lock:
            self._pending_futures[seq_id] = fut

        self._send_json(packet)
        return fut

    def get_observation(self) -> Dict:
        """返回当前机器人观测。

        对外部数据流而言，姿态主表示为连续的 `sin/cos` 形式：
        `j3_sin/j3_cos`、`j4_sin/j4_cos`、`j5_sin/j5_cos`。

        夹爪状态仍保留为 `j7`。

        同时保留 `j3/j4/j5` 原始角度字段，仅用于内部调试和兼容逻辑。
        """
        self._require_connected()
        if self._latest_pose is None:
            deadline = time.time() + 1.0
            while self._latest_pose is None and time.time() < deadline:
                time.sleep(0.01)
        if self._latest_pose is None:
            raise RuntimeError("No cartesian observation received yet.")
        if self._gripper_port_number is not None and self._latest_gripper_state is None:
            deadline = time.time() + 1.0
            while self._latest_gripper_state is None and time.time() < deadline:
                time.sleep(0.01)
        
        # Extract individual joint values from position tuple (x, y, z, w, p, r)
        x, y, z, w, p, r = self._latest_pose
        
        obs = {
            "j0": float(x),  # x coordinate
            "j1": float(y),  # y coordinate
            "j2": float(z),  # z coordinate
            "j7": (
                float(self._latest_gripper_state)
                if self._latest_gripper_state is not None
                else 0.0
            ),
            "timestamp": self._latest_t,
            "controller_tick": self._latest_tick,
            "uframe": self._latest_configuration.get("UFrameNumber") if self._latest_configuration else None,
            "utool": self._latest_configuration.get("UToolNumber") if self._latest_configuration else None,
        }
        obs.update(encode_fanuc_pose_dict({"j3": float(w), "j4": float(p), "j5": float(r)}))
        obs["j3"] = float(w)
        obs["j4"] = float(p)
        obs["j5"] = float(r)
        if self._gripper_port_number is not None:
            obs["gripper_state"] = (
                float(self._latest_gripper_state)
                if self._latest_gripper_state is not None
                else 0.0
            )
        
        # Add camera observations if cameras are configured
        if self.cameras:
            for cam_name, camera in self.cameras.items():
                frame = camera.read()
                if frame is not None:
                    obs[cam_name] = frame
        
        return obs

    # ------------------------------------------------------------------ #
    #  后台接收循环                                                          #
    # ------------------------------------------------------------------ #

    def _recv_loop(self) -> None:
        while self._connected and self._sock is not None:
            try:
                self._sock.settimeout(1.0)
                resp = self._read_json()
                #print(f"[recv] {resp}")   # ← 临时加这一行
            except socket.timeout:
                continue
            except (ConnectionError, OSError) as exc:
                if self._connected:
                    logger.error("Recv loop socket error: %s", exc)
                    with self._pending_lock:
                        for fut in self._pending_futures.values():
                            if not fut.done():
                                fut.set_exception(exc)
                        self._pending_futures.clear()
                break
            except Exception as exc:
                logger.exception("Recv loop unexpected error: %s", exc)
                break
            self._dispatch_response(resp)

        logger.debug("Fanuc recv loop exited")

    def _state_poll_loop(self) -> None:
        interval = 1.0 / self.STATE_POLL_HZ
        while self._connected and self._sock is not None:
            start_t = time.perf_counter()
            try:
                self._send_json({"Command": "FRC_ReadCartesianPosition", "Group": self._group})
                if self._gripper_port_number is not None:
                    self._send_json(
                        {
                            "Command": "FRC_ReadDIN",
                            "PortNumber": int(self._gripper_port_number),
                        }
                    )
            except Exception as exc:
                if self._connected:
                    logger.error("State poll send error: %s", exc)
                break
            elapsed = time.perf_counter() - start_t
            time.sleep(max(0.0, interval - elapsed))

    # ------------------------------------------------------------------ #
    #  内部工具                                                              #
    # ------------------------------------------------------------------ #
    def check_ack(self) -> Tuple[Optional[int], Optional[int]]:
        try:
            return self._ack_queue.get_nowait()
        except queue.Empty:
            return None, None

    def _dispatch_response(self, resp: Dict) -> None:
        if resp.get("Command") == "FRC_ReadCartesianPosition":
            self._update_pose_from_response(resp)
            return

        if resp.get("Command") == "FRC_ReadDIN":
            self._update_gripper_state_from_response(resp)
            return

        # Fanuc motion ACK may include `SequenceID` without echoing back
        # `Instruction: FRC_LinearMotion`. Accept any packet carrying
        # a sequence id as a motion completion/ack response.
        if "SequenceID" in resp:
            seq_id = int(resp["SequenceID"])
            err_id = int(resp.get("ErrorID", -1))
            with self._pending_lock:
                fut = self._pending_futures.pop(seq_id, None)
            if fut is not None and not fut.done():
                fut.set_result(err_id)
            self._ack_queue.put((seq_id, err_id))
            return

        if resp.get("Communication") == "FRC_SystemFault":
            logger.error("FRC_SystemFault: %s", resp)

    def _frc_connect(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5.0)
            s.connect((self._host, self._port))
            s.sendall(b'{"Communication": "FRC_Connect"}\r\n')
            data = json.loads(s.recv(4096).decode())
        if data.get("ErrorID", -1) != 0:
            raise RuntimeError(f"FRC_Connect failed: {data}")
        return data["PortNumber"]

    def _set_uframe_utool(self, uframe: int, utool: int) -> None:
        self._send_json(
            {
                "Command": "FRC_SetUFrameUTool",
                "UFrameNumber": int(uframe),
                "UToolNumber": int(utool),
                "Group": int(self._group),
            }
        )
        resp = self._recv_until(lambda r: r.get("Command") == "FRC_SetUFrameUTool")
        if resp.get("ErrorID", -1) != 0:
            raise RuntimeError(f"FRC_SetUFrameUTool failed: {resp}")

    def _send_json(self, payload: Dict) -> None:
        # print(f"[send] {payload}")  
        with self._send_lock:
            self._sock.sendall((json.dumps(payload) + "\r\n").encode("utf-8"))

    def _recv_until(self, predicate) -> Dict:
        """仅在接收线程启动前使用的同步接收。"""
        while True:
            resp = self._read_json()
            self._dispatch_response(resp)
            if predicate(resp):
                return resp

    def _update_pose_from_response(self, resp: Dict) -> None:
        if resp.get("ErrorID", -1) != 0:
            raise RuntimeError(f"FRC_ReadCartesianPosition failed: {resp}")
        position = resp.get("Position") or {}
        config = resp.get("Configuration") or {}
        self._latest_pose = (
            position["X"], position["Y"], position["Z"],
            position["W"], position["P"], position["R"],
        )
        self._latest_t = time.perf_counter()
        self._latest_tick = resp.get("TimeTag")
        self._latest_configuration = dict(config)

    def _update_gripper_state_from_response(self, resp: Dict) -> None:
        if resp.get("ErrorID", -1) != 0:
            logger.error("FRC_ReadDIN failed: %s", resp)
            return

        try:
            self._latest_gripper_state = int(resp["PortValue"])
        except (KeyError, TypeError, ValueError):
            logger.error("Invalid FRC_ReadDIN response: %s", resp)

    def _read_json(self) -> Dict:
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("Connection closed by remote")
            self._buf += chunk
        idx = self._buf.index(b"\n")
        line = self._buf[:idx].rstrip(b"\r")
        self._buf = self._buf[idx + 1:]
        return json.loads(line)

    def _require_connected(self) -> None:
        if not self._connected or self._sock is None:
            raise RuntimeError("Fanuc is not connected. Call connect() first.")

    def __enter__(self) -> "Fanuc":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        """Get observation features including arm state and camera images.
        
        Returns a dict where:
                - Position uses `j0`/`j1`/`j2`
                - Orientation uses continuous representation:
                    `j3_sin`, `j3_cos`, `j4_sin`, `j4_cos`, `j5_sin`, `j5_cos`
                - Gripper state uses `j7`
        - Camera names map to their image shapes (height, width, 3)
        
        These will be merged into "observation.state" by combine_feature_dicts.
        """
        obs_feat = {
            "j0": float,  # x coordinate
            "j1": float,  # y coordinate
            "j2": float,  # z coordinate
            "j3_sin": float,
            "j3_cos": float,
            "j4_sin": float,
            "j4_cos": float,
            "j5_sin": float,
            "j5_cos": float,
            "j7": float,  # gripper state
        }
        
        # Add camera features if cameras are configured
        if self.cameras:
            for cam_name, camera in self.cameras.items():
                # Camera objects have width, height, fps properties
                height = camera.height if hasattr(camera, 'height') else 480
                width = camera.width if hasattr(camera, 'width') else 640
                obs_feat[cam_name] = (height, width, 3)
        
        return obs_feat
    
    @property
    def action_features(self) -> dict[str, type | tuple]:
        """Get action features for the arm.
        
        Returns a dict where:
                - Position uses `j0`/`j1`/`j2`
                - Orientation uses continuous representation:
                    `j3_sin`, `j3_cos`, `j4_sin`, `j4_cos`, `j5_sin`, `j5_cos`
                - Gripper command uses `j7`
        
        These will be merged into "action" by combine_feature_dicts.
        """
        return {
            "j0": float,  # x coordinate
            "j1": float,  # y coordinate
            "j2": float,  # z coordinate
            "j3_sin": float,
            "j3_cos": float,
            "j4_sin": float,
            "j4_cos": float,
            "j5_sin": float,
            "j5_cos": float,
            "j7": float,  # gripper command/state
        }
    
    @property
    def is_connected(self) -> bool:
        """返回机器人是否已连接"""
        return self._connected