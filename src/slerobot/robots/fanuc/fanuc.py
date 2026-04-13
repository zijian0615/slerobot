import json
import logging
import socket
import threading
import time
from concurrent.futures import Future
from typing import Dict, Optional, Tuple

from slerobot.cameras.utils import make_cameras_from_configs
from ..robot import Robot

logger = logging.getLogger(__name__)


class Fanuc(Robot):
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
    ) -> None:
        self._host = host
        self._port = port
        self._group = group
        self._utool = utool
        self._uframe = uframe
        self._speed = speed
        self._term_type = term_type
        self._term_value = term_value

        self._sock: Optional[socket.socket] = None
        self._buf = b""
        self._connected = False

        self._latest_pose: Optional[Tuple[float, ...]] = None
        self._latest_t: Optional[float] = None
        self._latest_tick: Optional[int] = None
        self._latest_configuration: Optional[Dict] = None

        # seq_id -> Future[int]，接收线程写入，发送方读取
        self._pending_futures: Dict[int, Future] = {}
        self._pending_lock = threading.Lock()

        # 接收线程
        self._recv_thread: Optional[threading.Thread] = None

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

        #self._set_uframe_utool(self._uframe, self._utool)
        self._connected = True

        # 启动后台接收线程
        self._recv_thread = threading.Thread(
            target=self._recv_loop, name="fanuc-recv", daemon=True
        )
        self._recv_thread.start()

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
        """
        self._require_connected()
        if self._latest_configuration is None:
            self._latest_configuration = {
            "UToolNumber": self._utool,
            "UFrameNumber": self._uframe,
            "Front": 1, "Up": 1, "Left": 0,
            "Flip": 0, "Turn4": 0, "Turn5": 0, "Turn6": 0,
        }

        # Support both old and new action formats
        # Old format: {"position": (x, y, z, w, p, r), ...}
        # New format: {"position": (x, y, z), "rotation": (w, p, r), ...}
        position = action["position"]
        if len(position) == 6:
            # Old format: unpack all 6 values from position
            x, y, z, w, p, r = position
        elif len(position) == 3:
            # New format: position has 3 values, rotation is separate
            x, y, z = position
            w, p, r = action["rotation"]
        else:
            raise ValueError(f"Invalid position format. Expected 3 or 6 values, got {len(position)}")

        configuration = dict(self._latest_configuration or {})
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
        for key in ("LCBType", "LCBValue", "PortType", "PortNumber", "PortValue"):
            if key in action:
                packet[key] = action[key]

        # 先注册 Future，再发送，避免接收线程在发送完成前就收到 ACK 找不到槽位
        fut: Future = Future()
        with self._pending_lock:
            self._pending_futures[seq_id] = fut

        self._send_json(packet)
        return fut

    def get_observation(self) -> Dict:
        self._require_connected()
        self._send_json({"Command": "FRC_ReadCartesianPosition", "Group": self._group})
        # get_observation 在接收线程启动前可能被调用（connect 内部），
        # 此时走同步路径；启动后接收线程会自动处理响应并更新 _latest_pose
        if self._recv_thread is None or not self._recv_thread.is_alive():
            resp = self._recv_until(lambda r: r.get("Command") == "FRC_ReadCartesianPosition")
            self._update_pose_from_response(resp)
        else:
            # 接收线程负责更新，此处仅等待一次刷新
            deadline = time.time() + 5.0
            old_t = self._latest_t
            while time.time() < deadline:
                if self._latest_t != old_t:
                    break
                time.sleep(0.01)
        
        obs = {
            "position": self._latest_pose,
            "timestamp": self._latest_t,
            "controller_tick": self._latest_tick,
            "uframe": self._latest_configuration.get("UFrameNumber") if self._latest_configuration else None,
            "utool": self._latest_configuration.get("UToolNumber") if self._latest_configuration else None,
        }
        
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
                # print(f"[recv] {resp}")   # ← 临时加这一行
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

    # ------------------------------------------------------------------ #
    #  内部工具                                                              #
    # ------------------------------------------------------------------ #
    def check_ack(self) -> Tuple[Optional[int], Optional[int]]:
        with self._pending_lock:
            for seq_id, fut in list(self._pending_futures.items()):
                if fut.done():
                    self._pending_futures.pop(seq_id)
                    try:
                        return seq_id, fut.result()
                    except Exception:
                        return seq_id, -1
        return None, None
    def _dispatch_response(self, resp: Dict) -> None:
        if resp.get("Instruction") == "FRC_LinearMotion" and "SequenceID" in resp:
            seq_id = int(resp["SequenceID"])
            err_id = int(resp.get("ErrorID", -1))
            with self._pending_lock:
                fut = self._pending_futures.get(seq_id, None)
            if fut is not None and not fut.done():
                fut.set_result(err_id)
            return

        if resp.get("Command") == "FRC_ReadCartesianPosition":
            self._update_pose_from_response(resp)
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
        """Get observation features including arm state and camera images."""
        obs_feat = {
            "x": float,
            "y": float,
            "z": float,
            "w": float,
            "r": float,
            "p": float,  # Fixed: added 'p' instead of duplicate 'y'
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
        """Get action features for the arm."""
        return {
            "x": float,
            "y": float,
            "z": float,
            "w": float,
            "r": float,
            "p": float,  # Fixed: added 'p' instead of duplicate 'y'
        }
    
    @property
    def is_connected(self) -> bool:
        """返回机器人是否已连接"""
        return self._connected