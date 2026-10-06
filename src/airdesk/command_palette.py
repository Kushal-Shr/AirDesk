"""Native, non-activating Air Command Palette for desktop shortcuts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaletteCommand:
    key: str
    label: str
    shortcut: tuple[str, ...]


PALETTE_COMMANDS = (
    PaletteCommand("spotlight_search", "Spotlight Search", ("command", "space")),
    PaletteCommand("copy", "Copy", ("command", "c")),
    PaletteCommand("paste", "Paste", ("command", "v")),
    PaletteCommand("cut", "Cut", ("command", "x")),
    PaletteCommand("undo", "Undo", ("command", "z")),
    PaletteCommand("redo", "Redo", ("command", "shift", "z")),
    PaletteCommand("select_all", "Select All", ("command", "a")),
    PaletteCommand("screenshot", "Screenshot", ("command", "shift", "4")),
    PaletteCommand("close_window", "Close Window", ("command", "w")),
)


class MemoryCommandPalette:
    """Deterministic palette used by gesture-controller tests."""

    def __init__(
        self,
        bounds: tuple[int, int, int, int] = (780, 300, 1140, 696),
    ) -> None:
        self.bounds = bounds
        self.visible = False
        self.highlighted_index: int | None = None
        self.enabled_keys: set[str] = set()

    def show(self, enabled_keys: set[str]) -> None:
        self.enabled_keys = set(enabled_keys)
        self.visible = True
        self.highlighted_index = None

    def hide(self) -> None:
        self.visible = False
        self.highlighted_index = None

    def update_enabled(self, enabled_keys: set[str]) -> None:
        new_keys = set(enabled_keys)
        if new_keys == self.enabled_keys:
            return
        self.enabled_keys = new_keys

    def hit_test(self, screen_point: tuple[int, int] | None) -> int | None:
        if not self.visible or screen_point is None:
            return None
        left, top, right, bottom = self.bounds
        x, y = screen_point
        header = 54
        row_height = 36
        if x < left or x > right or y < top + header or y >= bottom:
            return None
        index = int((y - top - header) // row_height)
        return index if 0 <= index < len(PALETTE_COMMANDS) else None

    def highlight(self, index: int | None) -> None:
        self.highlighted_index = index

    def close(self) -> None:
        self.hide()


def _load_appkit():
    import objc
    from AppKit import (
        NSApplication,
        NSBackingStoreBuffered,
        NSColor,
        NSFont,
        NSPanel,
        NSScreen,
        NSTextField,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowStyleMaskBorderless,
        NSWindowStyleMaskNonactivatingPanel,
        NSStatusWindowLevel,
    )
    from Foundation import NSMakeRect

    return locals()


class NativeCommandPalette:
    """Centered palette controlled by hand position, never by keyboard focus."""

    WIDTH = 360
    HEADER = 54
    ROW_HEIGHT = 36
    FOOTER = 20
    HEIGHT = HEADER + ROW_HEIGHT * len(PALETTE_COMMANDS) + FOOTER

    def __init__(self) -> None:
        kit = _load_appkit()
        NSPanel = kit["NSPanel"]

        # PyObjC registers class names process-wide. Keep this name unique from
        # NativeInkOverlay's panel subclass so both windows can coexist.
        class AirDeskCommandPalettePanel(NSPanel):
            def canBecomeKeyWindow(panel_self):
                return False

            def canBecomeMainWindow(panel_self):
                return False

        self._kit = kit
        self._app = kit["NSApplication"].sharedApplication()
        screen = kit["NSScreen"].mainScreen().frame()
        left = round(screen.origin.x + (screen.size.width - self.WIDTH) / 2)
        bottom = round(screen.origin.y + (screen.size.height - self.HEIGHT) / 2)
        frame = kit["NSMakeRect"](left, bottom, self.WIDTH, self.HEIGHT)
        style = (
            kit["NSWindowStyleMaskBorderless"]
            | kit["NSWindowStyleMaskNonactivatingPanel"]
        )
        self._panel = AirDeskCommandPalettePanel.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, kit["NSBackingStoreBuffered"], False
        )
        self._panel.setOpaque_(True)
        self._panel.setBackgroundColor_(
            kit["NSColor"].colorWithCalibratedWhite_alpha_(0.08, 0.96)
        )
        self._panel.setHasShadow_(True)
        self._panel.setIgnoresMouseEvents_(True)
        self._panel.setLevel_(kit["NSStatusWindowLevel"])
        self._panel.setHidesOnDeactivate_(False)
        self._panel.setCollectionBehavior_(
            kit["NSWindowCollectionBehaviorCanJoinAllSpaces"]
            | kit["NSWindowCollectionBehaviorFullScreenAuxiliary"]
        )

        content = self._panel.contentView()
        title = self._make_label(
            "AIR COMMAND PALETTE",
            18,
            self.HEIGHT - 40,
            self.WIDTH - 36,
            25,
            bold=True,
        )
        content.addSubview_(title)
        self._rows = []
        for index, command in enumerate(PALETTE_COMMANDS):
            y = self.HEIGHT - self.HEADER - (index + 1) * self.ROW_HEIGHT + 5
            row = self._make_label("", 12, y, self.WIDTH - 24, 28)
            content.addSubview_(row)
            self._rows.append(row)

        self.bounds = (
            left,
            round(screen.size.height - bottom - self.HEIGHT),
            left + self.WIDTH,
            round(screen.size.height - bottom),
        )
        self.visible = False
        self.highlighted_index = None
        self.enabled_keys: set[str] = set()

    def _make_label(self, text, x, y, width, height, *, bold=False):
        label = self._kit["NSTextField"].alloc().initWithFrame_(
            self._kit["NSMakeRect"](x, y, width, height)
        )
        label.setStringValue_(text)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setTextColor_(self._kit["NSColor"].whiteColor())
        label.setFont_(
            self._kit["NSFont"].boldSystemFontOfSize_(15.0)
            if bold
            else self._kit["NSFont"].systemFontOfSize_(14.0)
        )
        return label

    def _redraw_rows(self) -> None:
        for index, (row, command) in enumerate(zip(self._rows, PALETTE_COMMANDS)):
            enabled = command.key in self.enabled_keys
            shortcut = " + ".join(part.title() for part in command.shortcut)
            marker = "✓" if enabled else "○"
            row.setStringValue_(f"  {marker}  {command.label}     {shortcut}")
            row.setTextColor_(
                self._kit["NSColor"].whiteColor()
                if enabled
                else self._kit["NSColor"].disabledControlTextColor()
            )
            highlighted = index == self.highlighted_index
            row.setDrawsBackground_(highlighted)
            if highlighted:
                row.setBackgroundColor_(
                    self._kit["NSColor"].colorWithCalibratedRed_green_blue_alpha_(
                        0.10, 0.42, 0.95, 0.88
                    )
                )

    def show(self, enabled_keys: set[str]) -> None:
        self.enabled_keys = set(enabled_keys)
        self.highlighted_index = None
        self.visible = True
        self._redraw_rows()
        self._panel.orderFrontRegardless()

    def hide(self) -> None:
        self.visible = False
        self.highlighted_index = None
        self._panel.orderOut_(None)

    def update_enabled(self, enabled_keys: set[str]) -> None:
        new_keys = set(enabled_keys)
        if new_keys == self.enabled_keys:
            return
        self.enabled_keys = new_keys
        self._redraw_rows()

    def hit_test(self, screen_point: tuple[int, int] | None) -> int | None:
        if not self.visible or screen_point is None:
            return None
        left, top, right, bottom = self.bounds
        x, y = screen_point
        if x < left or x > right or y < top + self.HEADER or y >= bottom:
            return None
        index = int((y - top - self.HEADER) // self.ROW_HEIGHT)
        return index if 0 <= index < len(PALETTE_COMMANDS) else None

    def highlight(self, index: int | None) -> None:
        if index == self.highlighted_index:
            return
        self.highlighted_index = index
        self._redraw_rows()

    def close(self) -> None:
        self.hide()
        self._panel.close()
