"""Native, non-activating Air Command Palette for desktop shortcuts."""

from __future__ import annotations

from dataclasses import dataclass

from .apple_theme import configure_floating_panel, make_glass_container, make_tinted_card


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
    PaletteCommand("save_document", "Save Document", ("command", "s")),
    PaletteCommand("screenshot", "Screenshot", ("command", "shift", "4")),
    PaletteCommand("close_window", "Close Window", ("command", "w")),
)


class MemoryCommandPalette:
    """Deterministic palette used by gesture-controller tests."""

    def __init__(
        self,
        bounds: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.bounds = bounds or (
            780,
            300,
            1140,
            300 + 54 + 36 * len(PALETTE_COMMANDS) + 18,
        )
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
        NSTextAlignmentRight,
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

    WIDTH = 390
    HEADER = 76
    ROW_HEIGHT = 42
    FOOTER = 18
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
        screen = kit["NSScreen"].screens()[0].frame()
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
        configure_floating_panel(self._panel)
        self._panel.setIgnoresMouseEvents_(True)
        self._panel.setLevel_(kit["NSStatusWindowLevel"])
        self._panel.setHidesOnDeactivate_(False)
        self._panel.setCollectionBehavior_(
            kit["NSWindowCollectionBehaviorCanJoinAllSpaces"]
            | kit["NSWindowCollectionBehaviorFullScreenAuxiliary"]
        )

        glass, content = make_glass_container(
            kit["NSMakeRect"](0, 0, self.WIDTH, self.HEIGHT),
            corner_radius=26.0,
        )
        self._panel.setContentView_(glass)
        title = self._make_label(
            "Air Commands",
            22,
            self.HEIGHT - 36,
            self.WIDTH - 36,
            26,
            bold=True,
        )
        content.addSubview_(title)
        subtitle = self._make_label(
            "Point with your left hand  •  pinch to choose",
            22,
            self.HEIGHT - 58,
            self.WIDTH - 44,
            18,
            secondary=True,
        )
        content.addSubview_(subtitle)
        self._rows = []
        self._row_views = []
        self._shortcut_labels = []
        for index, command in enumerate(PALETTE_COMMANDS):
            y = self.HEIGHT - self.HEADER - (index + 1) * self.ROW_HEIGHT + 4
            row_view = make_tinted_card(
                kit["NSMakeRect"](12, y, self.WIDTH - 24, self.ROW_HEIGHT - 6),
                corner_radius=12.0,
                alpha=0.055,
            )
            row = self._make_label("", 14, 7, 230, 23)
            shortcut = self._make_label(
                "",
                238,
                7,
                self.WIDTH - 24 - 252,
                23,
                secondary=True,
            )
            shortcut.setAlignment_(kit["NSTextAlignmentRight"])
            row_view.addSubview_(row)
            row_view.addSubview_(shortcut)
            content.addSubview_(row_view)
            self._rows.append(row)
            self._row_views.append(row_view)
            self._shortcut_labels.append(shortcut)

        self.bounds = (
            left,
            round(screen.size.height - bottom - self.HEIGHT),
            left + self.WIDTH,
            round(screen.size.height - bottom),
        )
        self.visible = False
        self.highlighted_index = None
        self.enabled_keys: set[str] = set()

    def _make_label(
        self, text, x, y, width, height, *, bold=False, secondary=False
    ):
        label = self._kit["NSTextField"].alloc().initWithFrame_(
            self._kit["NSMakeRect"](x, y, width, height)
        )
        label.setStringValue_(text)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setTextColor_(
            self._kit["NSColor"].secondaryLabelColor()
            if secondary
            else self._kit["NSColor"].labelColor()
        )
        label.setFont_(
            self._kit["NSFont"].boldSystemFontOfSize_(19.0)
            if bold
            else self._kit["NSFont"].systemFontOfSize_(13.0)
        )
        return label

    def _redraw_rows(self) -> None:
        for index, (row, row_view, shortcut_label, command) in enumerate(
            zip(
                self._rows,
                self._row_views,
                self._shortcut_labels,
                PALETTE_COMMANDS,
            )
        ):
            enabled = command.key in self.enabled_keys
            shortcut = " + ".join(part.title() for part in command.shortcut)
            marker = "●" if enabled else "○"
            row.setStringValue_(f"{marker}   {command.label}")
            shortcut_label.setStringValue_(shortcut)
            row.setTextColor_(
                self._kit["NSColor"].labelColor()
                if enabled
                else self._kit["NSColor"].disabledControlTextColor()
            )
            shortcut_label.setTextColor_(
                self._kit["NSColor"].secondaryLabelColor()
                if enabled
                else self._kit["NSColor"].disabledControlTextColor()
            )
            highlighted = index == self.highlighted_index
            if highlighted:
                row_view.layer().setBackgroundColor_(
                    self._kit["NSColor"].systemBlueColor()
                    .colorWithAlphaComponent_(0.78)
                    .CGColor()
                )
                row.setTextColor_(self._kit["NSColor"].whiteColor())
                shortcut_label.setTextColor_(self._kit["NSColor"].whiteColor())
            else:
                row_view.layer().setBackgroundColor_(
                    self._kit["NSColor"].labelColor()
                    .colorWithAlphaComponent_(0.055)
                    .CGColor()
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
