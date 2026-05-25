"""Keyboard teleop for LeKiwi base velocity (WASD + ZX + RF)."""

from __future__ import annotations

import logging
import threading
from typing import Any

from ..teleoperator import Teleoperator

logger = logging.getLogger(__name__)


class LeKiwiKeyboardTeleop(Teleoperator):
    def __init__(self, teleop_keys: dict[str, str] | None = None):
        self.teleop_keys = teleop_keys or {
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            "speed_up": "r",
            "speed_down": "f",
            "quit": "q",
        }
        self._pressed: set[str] = set()
        self._listener = None
        self.is_connected = False

    def connect(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as exc:
            raise RuntimeError("pynput is required for LeKiwi keyboard teleop") from exc

        def on_press(key):
            try:
                char = key.char
            except AttributeError:
                return
            if char:
                self._pressed.add(char.lower())

        def on_release(key):
            try:
                char = key.char
            except AttributeError:
                return
            if char:
                self._pressed.discard(char.lower())

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        self.is_connected = True
        logger.info("[LeKiwiKeyboard] Listening for base motion keys: %s", self.teleop_keys)

    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._pressed.clear()
        self.is_connected = False

    def get_action(self) -> set[str]:
        return set(self._pressed)
