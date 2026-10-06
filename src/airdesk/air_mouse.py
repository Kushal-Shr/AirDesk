"""Stage 5: opt-in, fail-safe macOS mouse movement through PyAutoGUI."""

from __future__ import annotations

from dataclasses import dataclass

from .virtual_cursor import run_virtual_cursor


WINDOW_NAME = "AirDesk - Safe Real Air Mouse"


def map_preview_to_screen(
    point: tuple[int, int],
    preview_size: tuple[int, int],
    screen_size: tuple[int, int],
) -> tuple[int, int]:
    """Scale a clamped preview coordinate to the macOS screen."""
    point_x, point_y = point
    preview_width, preview_height = preview_size
    screen_width, screen_height = screen_size

    preview_x_limit = max(preview_width - 1, 1)
    preview_y_limit = max(preview_height - 1, 1)
    screen_x_limit = max(screen_width - 1, 0)
    screen_y_limit = max(screen_height - 1, 0)

    clamped_x = min(max(point_x, 0), preview_x_limit)
    clamped_y = min(max(point_y, 0), preview_y_limit)
    screen_x = round(clamped_x / preview_x_limit * screen_x_limit)
    screen_y = round(clamped_y / preview_y_limit * screen_y_limit)
    return screen_x, screen_y


@dataclass
class SystemMouseController:
    """Small safety adapter around PyAutoGUI's real mouse output."""

    backend: object
    enabled: bool = False

    def __post_init__(self) -> None:
        self.backend.FAILSAFE = True
        self.backend.PAUSE = 0.0
        size = self.backend.size()
        self.screen_size = (int(size[0]), int(size[1]))
        if self.screen_size[0] <= 0 or self.screen_size[1] <= 0:
            raise RuntimeError(
                "macOS did not provide a usable screen size; mouse control was not started"
            )
        self._held_buttons: set[str] = set()

    def toggle(self) -> None:
        if self.enabled:
            self.release_all_buttons()
        self.reset_actions(require_release=not self.enabled)
        self.enabled = not self.enabled
        state = "ACTIVE" if self.enabled else "SAFE PREVIEW"
        print(f"Real mouse control: {state}")

    def disable(self, reason: str) -> None:
        self.release_all_buttons()
        self.reset_actions(require_release=True)
        if self.enabled:
            self.enabled = False
            print(f"Real mouse control: SAFE PREVIEW ({reason})")

    def reset_actions(self, require_release: bool = False) -> None:
        """Hook for later stages that have stateful click/scroll gestures."""

    def close(self) -> None:
        self.disable("AirDesk closed")

    def _handle_output_error(self, action) -> bool:
        try:
            action()
        except self.backend.FailSafeException:
            self.disable("PyAutoGUI corner fail-safe triggered")
            return False
        except Exception as error:
            self.disable(f"system output failed: {error}")
            return False
        return True

    def move_from_preview(
        self,
        point: tuple[int, int],
        preview_size: tuple[int, int],
    ) -> bool:
        if not self.enabled:
            return False

        screen_point = map_preview_to_screen(point, preview_size, self.screen_size)
        return self._handle_output_error(
            lambda: self.backend.moveTo(*screen_point, _pause=False)
        )

    def click(self, button: str) -> bool:
        if not self.enabled:
            return False
        return self._handle_output_error(
            lambda: self.backend.click(button=button, _pause=False)
        )

    def double_click(self, button: str = "left") -> bool:
        if not self.enabled:
            return False
        return self._handle_output_error(
            lambda: self.backend.doubleClick(
                button=button,
                interval=0.12,
                _pause=False,
            )
        )

    def mouse_down(self, button: str = "left") -> bool:
        if not self.enabled or button in self._held_buttons:
            return False
        succeeded = self._handle_output_error(
            lambda: self.backend.mouseDown(button=button, _pause=False)
        )
        if succeeded:
            self._held_buttons.add(button)
        return succeeded

    def mouse_up(self, button: str = "left") -> bool:
        if button not in self._held_buttons:
            return False
        try:
            self.backend.mouseUp(button=button, _pause=False)
        except Exception as error:
            self._held_buttons.discard(button)
            self.enabled = False
            print(f"Real mouse control: SAFE PREVIEW (button release failed: {error})")
            return False
        self._held_buttons.discard(button)
        return True

    def release_all_buttons(self) -> None:
        for button in tuple(self._held_buttons):
            self.mouse_up(button)

    def scroll(self, steps: int) -> bool:
        if not self.enabled:
            return False
        return self._handle_output_error(
            lambda: self.backend.scroll(steps, _pause=False)
        )

    def hotkey(self, *keys: str) -> bool:
        if not self.enabled:
            return False
        return self._handle_output_error(
            lambda: self.backend.hotkey(*keys, _pause=False)
        )

    def commit_text(self, text: str) -> bool:
        """Type text only after an explicit Air Write confirmation gesture."""
        if not text:
            return False
        return self._handle_output_error(
            lambda: self.backend.write(text, interval=0.0, _pause=False)
        )


def main() -> int:
    try:
        import pyautogui

        system_mouse = SystemMouseController(pyautogui)
    except Exception as error:
        print(f"AirDesk could not initialize PyAutoGUI: {error}")
        print(
            "Check the virtual environment and macOS Accessibility permission, "
            "then try again."
        )
        return 1

    print("Real mouse control starts OFF.")
    print("M toggles real movement. Esc immediately returns to SAFE PREVIEW.")
    print("Moving the macOS pointer into a screen corner triggers PyAutoGUI's fail-safe.")
    return run_virtual_cursor(system_mouse=system_mouse, window_name=WINDOW_NAME)


if __name__ == "__main__":
    raise SystemExit(main())
