import select
import sys
import termios
import threading
import time
import tty
import importlib

import numpy as np

_inputs_mod = None
try:
    _inputs_mod = importlib.import_module("inputs")
except ImportError:
    _inputs_mod = None

if _inputs_mod is not None:
    get_gamepad = getattr(_inputs_mod, "get_gamepad", None)
    UnpluggedError = getattr(_inputs_mod, "UnpluggedError", Exception)
else:
    get_gamepad = None

    class UnpluggedError(Exception):
        pass


class HeadlessTeleop:
    """Headless teleop with terminal keyboard and optional gamepad support."""

    def __init__(
        self,
        config_init=(0.0, 0.0, 0.0),
        lin_step=0.2,
        ang_step=0.2,
        max_lin=1.0,
        max_ang=3.0,
        height_init=0.3,
        height_step=0.01,
        min_height=0.2,
        max_height=0.35,
        gamepad_deadzone=0.1,
    ):
        self.cmd_vel = np.array(config_init, dtype=np.float32)
        self.cmd_height = float(height_init)

        self.lin_step = float(lin_step)
        self.ang_step = float(ang_step)
        self.height_step = float(height_step)

        self.max_lin = float(max_lin)
        self.max_ang = float(max_ang)
        self.min_height = float(min_height)
        self.max_height = float(max_height)
        self.gamepad_deadzone = float(gamepad_deadzone)

        self.lock = threading.Lock()
        self.running = True

        self._keyboard_fd = None
        self._keyboard_old_settings = None
        self._keyboard_thread = None
        self._gamepad_thread = None

        self._setup_keyboard()
        self._start_gamepad_thread_if_available()

        print("Headless teleop active.")
        print("Keyboard: W/S linear, A/D yaw, R/F height, SPACE stop.")
        if self._gamepad_thread is not None:
            print("Gamepad: left stick Y linear, right/left stick X yaw, d-pad up/down height, A stop.")
        else:
            print("Gamepad: python package 'inputs' not found or no gamepad events available.")

    def get_command(self):
        with self.lock:
            return self.cmd_vel.copy()

    def get_height_command(self):
        with self.lock:
            return self.cmd_height

    def close(self):
        self.running = False

        if self._keyboard_old_settings is not None and self._keyboard_fd is not None:
            try:
                termios.tcsetattr(self._keyboard_fd, termios.TCSADRAIN, self._keyboard_old_settings)
            except termios.error:
                pass

        if self._keyboard_thread is not None and self._keyboard_thread.is_alive():
            self._keyboard_thread.join(timeout=0.2)

    def _setup_keyboard(self):
        if not sys.stdin.isatty():
            print("Keyboard control disabled: stdin is not a TTY.")
            return

        self._keyboard_fd = sys.stdin.fileno()
        self._keyboard_old_settings = termios.tcgetattr(self._keyboard_fd)
        tty.setcbreak(self._keyboard_fd)

        self._keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._keyboard_thread.start()

    def _start_gamepad_thread_if_available(self):
        if get_gamepad is None:
            return

        self._gamepad_thread = threading.Thread(target=self._gamepad_loop, daemon=True)
        self._gamepad_thread.start()

    def _keyboard_loop(self):
        while self.running:
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                continue

            ch = sys.stdin.read(1)
            if not ch:
                continue

            key = ch.lower()
            with self.lock:
                if key == "w":
                    self.cmd_vel[0] = np.clip(self.cmd_vel[0] + self.lin_step, -self.max_lin, self.max_lin)
                elif key == "s":
                    self.cmd_vel[0] = np.clip(self.cmd_vel[0] - self.lin_step, -self.max_lin, self.max_lin)
                elif key == "a":
                    self.cmd_vel[2] = np.clip(self.cmd_vel[2] + self.ang_step, -self.max_ang, self.max_ang)
                elif key == "d":
                    self.cmd_vel[2] = np.clip(self.cmd_vel[2] - self.ang_step, -self.max_ang, self.max_ang)
                elif key == "r":
                    self.cmd_height = np.clip(self.cmd_height + self.height_step, self.min_height, self.max_height)
                elif key == "f":
                    self.cmd_height = np.clip(self.cmd_height - self.height_step, self.min_height, self.max_height)
                elif key == " ":
                    self.cmd_vel[:] = 0.0

    def _normalize_axis(self, raw, max_raw=32767.0):
        value = float(raw) / float(max_raw)
        value = np.clip(value, -1.0, 1.0)
        if abs(value) < self.gamepad_deadzone:
            return 0.0
        return value

    def _gamepad_loop(self):
        while self.running:
            try:
                events = get_gamepad()
            except UnpluggedError:
                time.sleep(0.5)
                continue
            except Exception:
                time.sleep(0.1)
                continue

            with self.lock:
                for ev in events:
                    print(ev.code, ev.state)
                    if ev.code == "ABS_Y":
                        self.cmd_vel[0] = np.clip(-self._normalize_axis(ev.state-127) * self.max_lin, -self.max_lin, self.max_lin)
                    elif ev.code in ("ABS_RX", "ABS_X"):
                        self.cmd_vel[2] = np.clip(self._normalize_axis(-(ev.state-127)) * self.max_ang, -self.max_ang, self.max_ang)
                    elif ev.code == "ABS_HAT0Y":
                        if ev.state == -1:
                            self.cmd_height = np.clip(self.cmd_height + self.height_step, self.min_height, self.max_height)
                        elif ev.state == 1:
                            self.cmd_height = np.clip(self.cmd_height - self.height_step, self.min_height, self.max_height)
                    elif ev.code == "BTN_SOUTH" and int(ev.state) == 1:
                        self.cmd_vel[:] = 0.0
