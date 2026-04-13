import json
import socket
import queue
import threading
import time
import logging
from typing import Dict, Optional, Tuple

from ..robot import Robot

logger = logging.getLogger(__name__)


class Fanuc(Robot):

    BUFFER_SIZE = 8
    STATE_POLL_HZ = 15.0  # get_observation() - get_priopception

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
        self._connected = False

        # Socket and framing.
        self._sock: Optional[socket.socket] = None
        self._buf = b""
        self._lock = threading.Lock()
        self._running = False

        # Background threads.
        self._recv_thread: Optional[threading.Thread] = None
        self._state_thread: Optional[threading.Thread] = None

        # Shared state.
        self._ack_queue = queue.Queue()
        self._ack_results: Dict[int, int] = {}
        self._latest_pose: Optional[Tuple[float, ...]] = None
        self._latest_t: Optional[float] = None
        self._latest_uframe: Optional[int] = None
        self._latest_utool: Optional[int] = None
        self._latest_configuration: Optional[Dict[str, int]] = None
        self._state_lock = threading.Lock()
        self._ack_lock = threading.Lock()

        # Motion sequence counter.
        self.seq_id = 1

    def connect(self) -> None:
        if self._connected:
            logger.warning("Already connected - skipping")
            return

        dynamic_port = self._frc_connect()
        logger.info(f"Dynamic port: {dynamic_port}")

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except Exception:
            pass
        self._sock.connect((self._host, dynamic_port))
        self._sock.settimeout(None)

        self._sendall(
            json.dumps({"Command": "FRC_Initialize", "GroupMask": self._group}) + "\r\n"
        )
        resp = self._read_json()
        logger.info(f"FRC_Initialize: {resp}")
        if resp.get("ErrorID", -1) != 0:
            raise RuntimeError(f"FRC_Initialize failed: {resp}")

        # Force controller-side UFRAME/UTOOL before polling starts.
        self._set_uframe_utool(self._uframe, self._utool)
        self._prime_cartesian_state()

        self._running = True
        self._recv_thread = threading.Thread(
            target=self._thread_receiver, daemon=True, name="FRC-Receiver"
        )
        self._recv_thread.start()

        self._state_thread = threading.Thread(
            target=self._thread_state_poller, daemon=True, name="FRC-StatePoller"
        )
        self._state_thread.start()

        self._connected = True
        logger.info(
            f"Fanuc connected to {self._host}:{dynamic_port} with UF={self._uframe}, UT={self._utool}"
        )

    def disconnect(self) -> None:
        self._connected = False
        self._running = False
        try:
            self._sendall(json.dumps({"Command": "FRC_Abort"}) + "\r\n")
            time.sleep(0.1)
        except Exception:
            pass
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        self._sock = None
        logger.info("Fanuc disconnected")

    def send_action(self, action: Dict) -> Dict:
        self._require_connected()

        pos = action["position"]
        x, y, z, w, p, r = pos

        lcb_type = action.get("lcb_type")
        port_type = action.get("port_type")
        port_number = action.get("port_number")
        port_value = action.get("port_value")

        with self._state_lock:
            latest_config = dict(self._latest_configuration or {})

        configuration = {
            "UToolNumber": int(action.get("utool", latest_config.get("UToolNumber", self._utool))),
            "UFrameNumber": int(action.get("uframe", latest_config.get("UFrameNumber", self._uframe))),
            "Front": int(action.get("front", latest_config.get("Front", 1))),
            "Up": int(action.get("up", latest_config.get("Up", 1))),
            "Left": int(action.get("left", latest_config.get("Left", 0))),
            "Flip": int(action.get("flip", latest_config.get("Flip", 0))),
            "Turn4": int(action.get("turn4", latest_config.get("Turn4", 0))),
            "Turn5": int(action.get("turn5", latest_config.get("Turn5", 0))),
            "Turn6": int(action.get("turn6", latest_config.get("Turn6", 0))),
        }

        with self._lock:
            seq_id = self.seq_id
            packet = {
                "Instruction": "FRC_LinearMotion",
                "SequenceID": seq_id,
                "Configuration": configuration,
                "Position": {
                    "X": float(x),
                    "Y": float(y),
                    "Z": float(z),
                    "W": float(w),
                    "P": float(p),
                    "R": float(r),
                    "Ext1": 0.0,
                    "Ext2": 0.0,
                    "Ext3": 0.0,
                },
                "SpeedType": "mmSec",
                "Speed": int(action.get("speed", self._speed)),
                "TermType": str(action.get("term_type", self._term_type)),
                "TermValue": int(action.get("term_value", self._term_value)),
            }

            if (
                lcb_type is not None
                and port_type is not None
                and port_number is not None
                and port_value is not None
            ):
                packet.update(
                    {
                        "LCBType": str(lcb_type),
                        "LCBValue": int(action.get("lcb_value", 0)),
                        "PortType": int(port_type),
                        "PortNumber": int(port_number),
                        "PortValue": str(port_value),
                    }
                )

            self._sock.sendall((json.dumps(packet) + "\r\n").encode("utf-8"))
            self.seq_id += 1
        return action

    def get_observation(self) -> Dict:
        self._require_connected()
        with self._state_lock:
            pose = self._latest_pose
            t = self._latest_t
            uframe = self._latest_uframe
            utool = self._latest_utool
        if pose is None:
            raise RuntimeError("No proprioception observation s_t received yet - try again after ~100ms")
        return {
            "position": pose,
            "timestamp": t,
            "uframe": uframe,
            "utool": utool,
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    def check_ack(self) -> Tuple[Optional[int], Optional[int]]:
        try:
            seq_id, err_id = self._ack_queue.get_nowait()
        except queue.Empty:
            return None, None

        with self._ack_lock:
            self._ack_results.pop(seq_id, None)
        return seq_id, err_id

    def wait_for_ack(
        self,
        seq_id: int,
        timeout: float = 10.0,
        poll_interval: float = 0.005,
    ) -> int:
        self._require_connected()
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._ack_lock:
                eid = self._ack_results.pop(seq_id, None)
            if eid is not None:
                return eid
            time.sleep(poll_interval)
        raise TimeoutError(f"Timed out waiting for ACK seq_id={seq_id}")

    def _thread_state_poller(self) -> None:
        interval = 1.0 / self.STATE_POLL_HZ
        req = (
            json.dumps({"Command": "FRC_ReadCartesianPosition", "Group": self._group})
            + "\r\n"
        ).encode()
        while self._running:
            t0 = time.perf_counter()
            try:
                with self._lock:
                    self._sock.sendall(req)
            except Exception as e:
                if self._running:
                    logger.error(f"StatePoller send error: {e}")
            time.sleep(max(0.0, interval - (time.perf_counter() - t0)))

    def _thread_receiver(self) -> None:
        while self._running:
            try:
                resp = self._read_json()
            except (ConnectionError, OSError) as e:
                if self._running:
                    logger.error(f"Receiver connection lost: {e}")
                break
            except json.JSONDecodeError as e:
                logger.warning(f"Receiver JSON error: {e}")
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Receiver error: {e}")
                break

            if resp.get("Command") == "FRC_ReadCartesianPosition":
                self._handle_cartesian_position(resp)
            elif "SequenceID" in resp:
                seq_id = resp["SequenceID"]
                err_id = resp.get("ErrorID", -1)
                with self._ack_lock:
                    self._ack_results[seq_id] = err_id
                self._ack_queue.put((seq_id, err_id))
            elif resp.get("Communication") == "FRC_SystemFault":
                logger.error(f"FRC_SystemFault: {resp}")

    def _frc_connect(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self._host, self._port))
            s.sendall(b'{"Communication": "FRC_Connect"}\r\n')
            data = json.loads(s.recv(4096).decode())
        logger.info(f"FRC_Connect response: {data}")
        if data.get("ErrorID", -1) != 0:
            raise RuntimeError(f"FRC_Connect failed: {data}")
        return data["PortNumber"]

    def _set_uframe_utool(self, uframe: int, utool: int) -> None:
        packet = {
            "Command": "FRC_SetUFrameUTool",
            "UFrameNumber": int(uframe),
            "UToolNumber": int(utool),
            "Group": int(self._group),
        }
        self._sendall(json.dumps(packet) + "\r\n")
        resp = self._read_json()
        logger.info(f"FRC_SetUFrameUTool: {resp}")
        if resp.get("ErrorID", -1) != 0:
            raise RuntimeError(f"FRC_SetUFrameUTool failed: {resp}")

    def _prime_cartesian_state(self) -> None:
        self._sendall(
            json.dumps({"Command": "FRC_ReadCartesianPosition", "Group": self._group}) + "\r\n"
        )
        resp = self._read_json()
        if resp.get("Command") != "FRC_ReadCartesianPosition":
            raise RuntimeError(f"Unexpected cartesian response: {resp}")
        self._handle_cartesian_position(resp, raise_on_frame_mismatch=True)

    def _handle_cartesian_position(
        self,
        resp: Dict,
        *,
        raise_on_frame_mismatch: bool = False,
    ) -> None:
        if resp.get("ErrorID", -1) != 0:
            msg = f"FRC_ReadCartesianPosition failed: {resp}"
            if raise_on_frame_mismatch:
                raise RuntimeError(msg)
            logger.error(msg)
            return

        config = resp.get("Configuration") or {}
        p = resp.get("Position") or {}
        try:
            pose = (p["X"], p["Y"], p["Z"], p["W"], p["P"], p["R"])
        except KeyError as exc:
            logger.warning(f"Cartesian response missing field {exc}: {resp}")
            return

        actual_uframe = config.get("UFrameNumber")
        actual_utool = config.get("UToolNumber")
        if actual_uframe != self._uframe or actual_utool != self._utool:
            msg = (
                "Controller returned Cartesian pose in unexpected frame/tool: "
                f"expected UF={self._uframe}, UT={self._utool}; "
                f"got UF={actual_uframe}, UT={actual_utool}"
            )
            if raise_on_frame_mismatch:
                raise RuntimeError(msg)
            logger.warning(msg)

        with self._state_lock:
            self._latest_pose = pose
            self._latest_t = time.perf_counter()
            self._latest_uframe = actual_uframe
            self._latest_utool = actual_utool
            self._latest_configuration = dict(config)

    def _sendall(self, msg: str) -> None:
        with self._lock:
            self._sock.sendall(msg.encode("utf-8"))

    def _read_json(self) -> dict:
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("Connection closed by remote")
            self._buf += chunk
        idx = self._buf.index(b"\n")
        line = self._buf[:idx].rstrip(b"\r")
        self._buf = self._buf[idx + 1 :]
        return json.loads(line)

    def _require_connected(self) -> None:
        if not self._connected or self._sock is None:
            raise RuntimeError("Fanuc is not connected. Call connect() first.")

    def __enter__(self) -> "Fanuc":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()
