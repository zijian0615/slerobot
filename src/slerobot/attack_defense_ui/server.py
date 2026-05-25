#!/usr/bin/env python
"""Local web UI: ATTACK / DETECT / MITIGATION recording sessions."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from slerobot.attack_defense_ui.repo_state import MODES, allocate_repo_index, peek_next_repo_ids
from slerobot.attack_defense_ui.telemetry_hub import TELEMETRY

UI_DIR = Path(__file__).resolve().parent
STATIC_DIR = UI_DIR / "static"
CONFIG_PATH = UI_DIR / "config.json"
REPO_ROOT = UI_DIR.parents[3]
UI_PORT = int(os.getenv("ATTACK_DEFENSE_UI_PORT", "8765"))


class RecordingSession:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._mode: str | None = None
        self._last_repo_id: str | None = None
        self._last_repo_index: int | None = None
        self._homeset_running = False
        self._log_lines: list[str] = []
        self._max_log_lines = 400

    @property
    def running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    @property
    def mode(self) -> str | None:
        with self._lock:
            return self._mode

    def get_logs(self, tail: int = 120) -> list[str]:
        with self._lock:
            return list(self._log_lines[-tail:])

    def _append_log(self, line: str) -> None:
        with self._lock:
            self._log_lines.append(line.rstrip())
            if len(self._log_lines) > self._max_log_lines:
                self._log_lines = self._log_lines[-self._max_log_lines :]

    def _load_config(self) -> dict:
        with CONFIG_PATH.open(encoding="utf-8") as f:
            return json.load(f)

    def _run_homeset(self, cfg: dict) -> None:
        script = Path(
            cfg.get(
                "move_linear_script",
                REPO_ROOT / "src" / "slerobot" / "test" / "moveLinear.py",
            )
        )
        if not script.is_absolute():
            script = REPO_ROOT / script
        if not script.is_file():
            raise FileNotFoundError(f"Home set script not found: {script}")

        timeout_s = int(cfg.get("move_linear_timeout_s", 120))
        self._append_log(f"[UI] Home set → {script}")
        self._homeset_running = True
        homeset_env = os.environ.copy()
        homeset_env["ROBOT_HOST"] = str(cfg.get("robot_host", "127.0.0.1"))
        homeset_env["ROBOT_PORT"] = str(cfg.get("robot_port", 16001))
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=homeset_env,
            )
            if result.stdout:
                for line in result.stdout.splitlines():
                    self._append_log(line)
            if result.stderr:
                for line in result.stderr.splitlines():
                    self._append_log(line)
            if result.returncode != 0:
                tail = (result.stderr or result.stdout or "").strip()[-400:]
                raise RuntimeError(f"Home set exited with code {result.returncode}: {tail}")
            self._append_log("[UI] Home set completed")
        finally:
            self._homeset_running = False

    def _build_command(self, mode: str, repo_id: str) -> list[str]:
        cfg = self._load_config()
        attention_on = mode in ("detect", "mitigation")

        cmd = [
            sys.executable,
            "-m",
            "slerobot.scripts.slerobot_record",
            f"--robot.host={cfg['robot_host']}",
            f"--robot.port={cfg['robot_port']}",
            f"--robot.cameras={cfg['cameras_json']}",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.single_task={cfg['single_task']}",
            f"--dataset.num_episodes={cfg['num_episodes']}",
            f"--dataset.episode_time_s={cfg['episode_time_s']}",
            f"--dataset.reset_time_s={cfg['reset_time_s']}",
            f"--policy.path={cfg['policy_path']}",
            "--display_data=true",
            f"--realtime_attention_display={'true' if attention_on else 'false'}",
            f"--enable_attention_visualization={'true' if attention_on else 'false'}",
            f"--tts_voice={cfg['tts_voice']}",
        ]

        if attention_on:
            roi_mode = "detect" if mode == "detect" else "mitigation"
            cmd.extend(
                [
                    f"--policy.attention_cam_method={cfg['attention_cam_method']}",
                    f"--policy.grad_cam_target_action_index={cfg['grad_cam_target_action_index']}",
                    f"--policy.grad_cam_edge_margin_px={cfg['grad_cam_edge_margin_px']}",
                    f"--policy.grad_cam_edge_mean_threshold={cfg['grad_cam_edge_mean_threshold']}",
                    f"--policy.cam_warning_high_activation_threshold={cfg['cam_warning_high_activation_threshold']}",
                    f"--policy.attention_camera={cfg['attention_camera']}",
                    f"--policy.attention_roi_mode={roi_mode}",
                ]
            )
        return cmd

    def _reader_thread(self, stream) -> None:
        for raw in iter(stream.readline, b""):
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                text = str(raw)
            if text:
                self._append_log(text)

    def start(self, mode: str, *, homeset: bool = False) -> dict:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(f"Session already running in {self._mode!r} mode. Stop it first.")
            if self._homeset_running:
                raise RuntimeError("Home set is already running. Please wait.")

        self.stop()
        TELEMETRY.reset()

        cfg = self._load_config()
        if homeset:
            self._run_homeset(cfg)

        repo_index, repo_ids = allocate_repo_index(cfg)
        repo_id = repo_ids[mode]
        self._last_repo_id = repo_id
        self._last_repo_index = repo_index

        cmd = self._build_command(mode, repo_id)
        self._append_log(f"[UI] dataset.repo_id={repo_id} (index={repo_index})")
        self._append_log(f"[UI] Starting {mode.upper()} → {' '.join(cmd)}")

        telemetry_url = f"http://127.0.0.1:{UI_PORT}/api/telemetry/push"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["SLEROBOT_TELEMETRY_PUSH"] = "1"
        env["SLEROBOT_TELEMETRY_URL"] = telemetry_url
        env["SLEROBOT_SESSION_MODE"] = mode

        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

        with self._lock:
            self._process = process
            self._mode = mode

        threading.Thread(target=self._reader_thread, args=(process.stdout,), daemon=True).start()
        threading.Thread(target=self._wait_thread, args=(process,), daemon=True).start()

        next_index, next_repos = peek_next_repo_ids(cfg)
        return {
            "ok": True,
            "mode": mode,
            "pid": process.pid,
            "repo_id": repo_id,
            "repo_index": repo_index,
            "next_repo_index": next_index,
            "next_repo_ids": next_repos,
            "homeset_ran": homeset,
            "attention_visualization": mode in ("detect", "mitigation"),
            "telemetry": True,
            "command": cmd,
        }

    def _wait_thread(self, process: subprocess.Popen) -> None:
        code = process.wait()
        self._append_log(f"[UI] Process exited with code {code}")
        with self._lock:
            if self._process is process:
                self._process = None
                self._mode = None

    def stop(self) -> dict:
        with self._lock:
            process = self._process
            mode = self._mode

        if process is None or process.poll() is not None:
            with self._lock:
                self._process = None
                self._mode = None
            return {"ok": True, "stopped": False, "message": "No active session"}

        self._append_log("[UI] Stopping recording process…")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)

        with self._lock:
            self._process = None
            self._mode = None

        return {"ok": True, "stopped": True, "mode": mode}

    def status(self) -> dict:
        cfg = self._load_config()
        next_index, next_repos = peek_next_repo_ids(cfg)
        return {
            "running": self.running,
            "homeset_running": self._homeset_running,
            "mode": self.mode,
            "pid": self._process.pid if self.running and self._process else None,
            "last_repo_id": self._last_repo_id,
            "last_repo_index": self._last_repo_index,
            "next_repo_index": next_index,
            "next_repo_ids": next_repos,
            "telemetry": True,
            "logs": self.get_logs(),
        }


SESSION = RecordingSession()


class ControlHandler(BaseHTTPRequestHandler):
    server_version = "SLeRobotAttackDefense/2.0"

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path

        if route in ("/", "/index.html"):
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if route.startswith("/static/"):
            rel = route.removeprefix("/static/")
            file_path = STATIC_DIR / rel
            if file_path.suffix == ".css":
                return self._send_file(file_path, "text/css; charset=utf-8")
            if file_path.suffix == ".js":
                return self._send_file(file_path, "application/javascript; charset=utf-8")
            return self._send_file(file_path, "application/octet-stream")
        if route == "/api/status":
            return self._send_json(SESSION.status())
        if route == "/api/config":
            with CONFIG_PATH.open(encoding="utf-8") as f:
                return self._send_json(json.load(f))
        if route in ("/api/telemetry", "/api/telemetry/snapshot"):
            snap = TELEMETRY.snapshot()
            snap["ok"] = True
            return self._send_json(snap)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._send_json({"ok": False, "error": "Invalid JSON"}, HTTPStatus.BAD_REQUEST)

        try:
            if parsed.path == "/api/start":
                mode = body.get("mode", "attack")
                homeset = bool(body.get("homeset", False))
                return self._send_json(SESSION.start(mode, homeset=homeset))
            if parsed.path == "/api/homeset":
                cfg = SESSION._load_config()
                SESSION._run_homeset(cfg)
                return self._send_json({"ok": True, "message": "Home set completed"})
            if parsed.path == "/api/stop":
                return self._send_json(SESSION.stop())
            if parsed.path == "/api/telemetry/push":
                TELEMETRY.push(
                    step=body.get("step"),
                    cameras=body.get("cameras"),
                    attention=body.get("attention"),
                    action=body.get("action"),
                    warnings=body.get("warnings"),
                    mode=body.get("mode") or SESSION.mode,
                )
                return self._send_json({"ok": True})
        except Exception as exc:
            return self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        self.send_error(HTTPStatus.NOT_FOUND)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", UI_PORT), ControlHandler)
    print(f"[ATTACK/DEFENSE UI] http://127.0.0.1:{UI_PORT}")
    print(f"[ATTACK/DEFENSE UI] Config: {CONFIG_PATH}")
    print("[ATTACK/DEFENSE UI] Live feed: cameras + 7 action traces (no Rerun)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        SESSION.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
