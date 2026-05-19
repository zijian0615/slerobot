"""Host Rerun web viewer on :9090 and gRPC proxy on :9876 for browser + SDK."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class RerunWebViewer:
    """Start `rerun --serve-web --web-viewer` once; recording SDK connects via gRPC only."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._grpc_port = 9876
        self._web_port = 9090

    @property
    def web_url(self) -> str:
        return f"http://127.0.0.1:{self._web_port}"

    @property
    def grpc_port(self) -> int:
        return self._grpc_port

    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, grpc_port: int = 9876, web_port: int = 9090) -> str:
        self._grpc_port = grpc_port
        self._web_port = web_port

        if self.running():
            return self.web_url

        rerun_bin = shutil.which("rerun")
        if not rerun_bin:
            raise RuntimeError("Rerun CLI not found. Install with: pip install -U rerun-sdk")

        cmd = [
            rerun_bin,
            "--serve-web",
            "--web-viewer",
            "--port",
            str(grpc_port),
            "--web-viewer-port",
            str(web_port),
        ]
        logger.info("Starting Rerun web viewer: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self._wait_until_ready(web_port, timeout_s=15.0)
        return self.web_url

    def _wait_until_ready(self, web_port: int, timeout_s: float) -> None:
        import urllib.error
        import urllib.request

        url = f"http://127.0.0.1:{web_port}/"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._process and self._process.poll() is not None:
                err = self._process.stderr.read().decode("utf-8", errors="replace") if self._process.stderr else ""
                raise RuntimeError(f"Rerun viewer exited early: {err[:500]}")
            try:
                urllib.request.urlopen(url, timeout=0.5)
                logger.info("Rerun web viewer ready at %s", url)
                return
            except (urllib.error.URLError, TimeoutError, OSError):
                time.sleep(0.35)
        raise RuntimeError(
            f"Rerun web viewer did not become ready on port {web_port} within {timeout_s:.0f}s. "
            f"Try manually: rerun --serve-web --web-viewer --port 9876 --web-viewer-port {web_port}"
        )

    def stop(self) -> None:
        if not self._process:
            return
        try:
            os.killpg(self._process.pid, 15)
            self._process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self._process.pid, 9)
            except ProcessLookupError:
                pass
        self._process = None


RERUN_WEB = RerunWebViewer()
