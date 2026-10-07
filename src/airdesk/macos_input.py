"""macOS input fixes for the installed PyAutoGUI backend."""

from __future__ import annotations

import subprocess
import time


class MacOSBackend:
    """Use Quartz for text and double clicks; delegate other input."""

    def __init__(self, backend, quartz=None, sleep=time.sleep, launch=None):
        if quartz is None:
            import Quartz as quartz
        self._backend = backend
        self._quartz = quartz
        self._sleep = sleep
        self._launch = launch or subprocess.Popen

    def __getattr__(self, name):
        return getattr(self._backend, name)

    @property
    def FAILSAFE(self):
        return self._backend.FAILSAFE

    @FAILSAFE.setter
    def FAILSAFE(self, value):
        self._backend.FAILSAFE = value

    @property
    def PAUSE(self):
        return self._backend.PAUSE

    @PAUSE.setter
    def PAUSE(self, value):
        self._backend.PAUSE = value

    def size(self):
        bounds = self._quartz.CGDisplayBounds(self._quartz.CGMainDisplayID())
        return round(bounds.size.width), round(bounds.size.height)

    def hotkey(self, *keys, _pause=False):
        """Attach modifier flags explicitly; key-down order alone is unreliable."""
        q = self._quartz
        modifiers = {
            "command": q.kCGEventFlagMaskCommand,
            "shift": q.kCGEventFlagMaskShift,
            "shiftleft": q.kCGEventFlagMaskShift,
            "shiftright": q.kCGEventFlagMaskShift,
            "ctrl": q.kCGEventFlagMaskControl,
            "ctrlleft": q.kCGEventFlagMaskControl,
            "ctrlright": q.kCGEventFlagMaskControl,
            "alt": q.kCGEventFlagMaskAlternate,
            "option": q.kCGEventFlagMaskAlternate,
        }
        keymap = self._backend.platformModule.keyboardMapping
        names = [key.lower() for key in keys]
        if any(keymap.get(key) is None for key in names):
            raise ValueError("unsupported macOS shortcut key")
        self._backend.failSafeCheck()
        held = []
        flags = 0

        def post(key, down, event_flags):
            event = q.CGEventCreateKeyboardEvent(None, keymap[key], down)
            if event is None:
                raise RuntimeError("macOS could not create a shortcut event")
            q.CGEventSetFlags(event, event_flags)
            q.CGEventPost(q.kCGHIDEventTap, event)
            self._sleep(0.01)

        try:
            for key in names:
                flags |= modifiers.get(key, 0)
                held.append(key)
                post(key, True, flags)
        finally:
            release_error = None
            for key in reversed(held):
                flags &= ~modifiers.get(key, 0)
                try:
                    post(key, False, flags)
                except Exception as error:
                    release_error = error
            if release_error is not None:
                raise release_error

    def showMissionControl(self, _pause=False):
        """Open Mission Control without depending on a remappable hotkey."""
        self._backend.failSafeCheck()
        self._launch(
            ["/usr/bin/open", "-a", "Mission Control"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def doubleClick(self, button="left", interval=0.12, _pause=False):
        q = self._quartz
        buttons = {
            "left": (q.kCGMouseButtonLeft, q.kCGEventLeftMouseDown, q.kCGEventLeftMouseUp),
            "right": (q.kCGMouseButtonRight, q.kCGEventRightMouseDown, q.kCGEventRightMouseUp),
        }
        mouse_button, down, up = buttons[button]
        self._backend.failSafeCheck()
        point = tuple(self._backend.position())
        for count in (1, 2):
            if count == 2:
                self._sleep(interval)
                self._backend.failSafeCheck()
            for event_type in (down, up):
                event = q.CGEventCreateMouseEvent(None, event_type, point, mouse_button)
                if event is None:
                    raise RuntimeError("macOS could not create a click event")
                q.CGEventSetIntegerValueField(event, q.kCGMouseEventClickState, count)
                q.CGEventPost(q.kCGHIDEventTap, event)

    def secondClick(self, button="left", _pause=False):
        """Complete a native double-click whose first click was already sent."""
        q = self._quartz
        buttons = {
            "left": (q.kCGMouseButtonLeft, q.kCGEventLeftMouseDown, q.kCGEventLeftMouseUp),
            "right": (q.kCGMouseButtonRight, q.kCGEventRightMouseDown, q.kCGEventRightMouseUp),
        }
        mouse_button, down, up = buttons[button]
        self._backend.failSafeCheck()
        point = tuple(self._backend.position())
        for event_type in (down, up):
            event = q.CGEventCreateMouseEvent(None, event_type, point, mouse_button)
            if event is None:
                raise RuntimeError("macOS could not create a click event")
            q.CGEventSetIntegerValueField(event, q.kCGMouseEventClickState, 2)
            q.CGEventPost(q.kCGHIDEventTap, event)

    def write(self, text, interval=0.0, _pause=False):
        """Insert exact Unicode text without keyboard-layout/Shift translation."""
        q = self._quartz
        chunk_size = 1 if interval else 20
        for start in range(0, len(text), chunk_size):
            self._backend.failSafeCheck()
            chunk = text[start:start + chunk_size]
            # Quartz's length counts UTF-16 code units, not Python characters.
            length = len(chunk.encode("utf-16-le")) // 2
            for is_down in (True, False):
                event = q.CGEventCreateKeyboardEvent(None, 0, is_down)
                if event is None:
                    raise RuntimeError("macOS could not create a text event")
                q.CGEventSetFlags(event, 0)
                q.CGEventKeyboardSetUnicodeString(event, length, chunk)
                q.CGEventPost(q.kCGHIDEventTap, event)
            if interval:
                self._sleep(interval)
