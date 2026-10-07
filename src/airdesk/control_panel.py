"""Native macOS AirDesk control panel and platform-neutral panel state."""

from __future__ import annotations

from dataclasses import dataclass

from .apple_theme import configure_floating_panel, make_glass_container, make_tinted_card


CONTROL_INSTRUCTIONS = (
    ("Desktop gestures", "Pointer — left index only"),
    ("Desktop gestures", "Scroll — left index + middle; move vertically"),
    ("Desktop gestures", "Left click — quick right thumb–index pinch"),
    ("Desktop gestures", "Double click — two quick right thumb–index pinches"),
    ("Desktop gestures", "Right click — right thumb + index + middle pinch"),
    ("Desktop gestures", "Drag / select — hold right pinch ~0.3s; move left pointer"),
    ("Desktop gestures", "Mission Control — right 3-finger swipe up"),
    ("Desktop gestures", "Switch apps — right 3-finger swipe left / right"),
    ("Desktop gestures", "Enter — right index only; hold briefly"),
    ("Desktop gestures", "Lock a hand — close that hand into a fist"),
    ("Air Command Palette", "Open — right index + middle up for ~0.3s"),
    ("Air Command Palette", "Highlight — move the left-hand pointer"),
    ("Air Command Palette", "Choose — right thumb–index pinch"),
    ("Air Command Palette", "Cancel — right fist or move hand away"),
    ("Air Command Palette", "Spotlight — Command + Space"),
    ("Air Command Palette", "Copy / Paste / Cut — Command + C / V / X"),
    ("Air Command Palette", "Undo / Redo — Command + Z / Shift + Z"),
    ("Air Command Palette", "Save Document — Command + S"),
    ("Air Command Palette", "Screenshot — Command + Shift + 4"),
    ("Air Command Palette", "Close Window — Command + W"),
)


@dataclass(frozen=True)
class ControlPanelStatus:
    """Small view model that keeps camera code independent from AppKit."""

    mode: str = "STOPPED"
    left_hand: str = "LOCKED"
    right_hand: str = "LOCKED"
    fps: float = 0.0
    processing_ms: float = 0.0


def build_control_panel_status(
    *,
    system_enabled: bool,
    paused: bool,
    left_locked: bool,
    right_locked: bool,
    cursor_visible: bool,
    left_action: str | None,
    right_action: str | None,
    fps: float,
    processing_ms: float,
    writing_active: bool = False,
) -> ControlPanelStatus:
    """Translate current runtime values into the panel's stable vocabulary."""
    mode = "PAUSED" if paused else ("LIVE" if system_enabled else "STOPPED")
    if writing_active:
        mode = f"{mode} · AIR WRITE"

    if left_locked:
        left_hand = "LOCKED"
    elif left_action and left_action.upper() in {"SCROLL", "DRAG", "COMMAND"}:
        left_hand = left_action.upper()
    elif cursor_visible:
        left_hand = "POINTER"
    else:
        left_hand = "READY"

    if right_locked:
        right_hand = "LOCKED"
    elif right_action and right_action.upper() == "DRAG":
        right_hand = "DRAG"
    elif right_action and right_action.upper() == "ENTER":
        right_hand = "ENTER"
    elif right_action and right_action.upper() in {"COMMAND", "SWIPE", "CLICK"}:
        right_hand = "COMMAND"
    else:
        right_hand = "READY"

    return ControlPanelStatus(
        mode=mode,
        left_hand=left_hand,
        right_hand=right_hand,
        fps=max(0.0, fps),
        processing_ms=max(0.0, processing_ms),
    )


class MemoryControlPanel:
    """In-memory implementation used by tests and non-native integrations."""

    def __init__(self, config=None) -> None:
        self.config = config
        self.status = ControlPanelStatus()
        self.visible = True

    def update_status(self, status: ControlPanelStatus) -> None:
        self.status = status

    def pump(self) -> None:
        pass

    def show(self) -> None:
        self.visible = True

    def hide(self) -> None:
        self.visible = False

    def toggle_visibility(self) -> None:
        if self.visible:
            self.hide()
        else:
            self.show()

    def close(self) -> None:
        self.hide()


def _load_appkit():
    """Import AppKit lazily so unit tests do not require a macOS UI session."""
    import objc
    from AppKit import (
        NSApplication,
        NSApplicationActivationPolicyAccessory,
        NSBackingStoreBuffered,
        NSButton,
        NSColor,
        NSDefaultRunLoopMode,
        NSEventMaskAny,
        NSFloatingWindowLevel,
        NSFont,
        NSPanel,
        NSScreen,
        NSTextField,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowStyleMaskBorderless,
        NSWindowStyleMaskUtilityWindow,
    )
    from Foundation import NSDate, NSMakeRect, NSObject

    return locals()


class NativeControlPanel:
    """Read-only floating gesture guide with live state and a hide control."""

    WIDTH = 420
    HEIGHT = 760

    def __init__(self, config=None) -> None:
        kit = _load_appkit()
        NSObject = kit["NSObject"]

        class AirDeskControlPanelTarget(NSObject):
            def initWithOwner_(target_self, owner):
                target_self = kit["objc"].super(
                    AirDeskControlPanelTarget,
                    target_self,
                ).init()
                if target_self is None:
                    return None
                target_self.owner = owner
                return target_self

            def hidePanel_(target_self, _sender):
                target_self.owner.hide()

        self._kit = kit
        self.config = config
        self.status = ControlPanelStatus()
        self._rendered_status = None
        self.visible = False
        self._app = kit["NSApplication"].sharedApplication()
        self._app.setActivationPolicy_(kit["NSApplicationActivationPolicyAccessory"])

        screen = kit["NSScreen"].mainScreen().visibleFrame()
        origin_x = max(screen.origin.x, screen.origin.x + screen.size.width - self.WIDTH - 18)
        origin_y = max(screen.origin.y, screen.origin.y + screen.size.height - self.HEIGHT - 18)
        frame = kit["NSMakeRect"](origin_x, origin_y, self.WIDTH, self.HEIGHT)
        style = kit["NSWindowStyleMaskBorderless"] | kit["NSWindowStyleMaskUtilityWindow"]
        self._panel = kit["NSPanel"].alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            style,
            kit["NSBackingStoreBuffered"],
            False,
        )
        configure_floating_panel(self._panel)
        self._panel.setLevel_(kit["NSFloatingWindowLevel"])
        self._panel.setCollectionBehavior_(
            kit["NSWindowCollectionBehaviorCanJoinAllSpaces"]
            | kit["NSWindowCollectionBehaviorFullScreenAuxiliary"]
        )
        self._target = AirDeskControlPanelTarget.alloc().initWithOwner_(self)

        glass, content = make_glass_container(
            kit["NSMakeRect"](0, 0, self.WIDTH, self.HEIGHT),
            corner_radius=28.0,
        )
        self._panel.setContentView_(glass)
        y = self.HEIGHT - 45
        title = self._make_label("AirDesk", 22, y, 250, 28, bold=True, font_size=21.0)
        content.addSubview_(title)
        subtitle = self._make_label(
            "Gesture Guide",
            22,
            y - 22,
            250,
            20,
            secondary=True,
        )
        content.addSubview_(subtitle)
        hide_button = kit["NSButton"].alloc().initWithFrame_(
            kit["NSMakeRect"](338, y - 9, 62, 30)
        )
        hide_button.setTitle_("Hide")
        hide_button.setBezelStyle_(1)
        hide_button.setTarget_(self._target)
        hide_button.setAction_("hidePanel:")
        content.addSubview_(hide_button)
        y -= 88
        status_card = make_tinted_card(
            kit["NSMakeRect"](16, y, self.WIDTH - 32, 62),
            corner_radius=16.0,
            alpha=0.09,
        )
        content.addSubview_(status_card)
        self._mode_label = self._make_label(
            "●  STOPPED", 14, 34, 180, 21, bold=True, font_size=13.0
        )
        status_card.addSubview_(self._mode_label)
        self._performance_label = self._make_label(
            "0.0 FPS  •  0.0 ms", 205, 34, 166, 21, secondary=True
        )
        self._performance_label.setAlignment_(2)
        status_card.addSubview_(self._performance_label)
        self._hands_label = self._make_label(
            "LEFT  LOCKED     •     RIGHT  LOCKED", 14, 10, 360, 20
        )
        status_card.addSubview_(self._hands_label)
        y -= 28
        notice = self._make_label(
            "Controls are always available  •  drag this panel to move it",
            22,
            y,
            self.WIDTH - 36,
            22,
            secondary=True,
            font_size=11.0,
        )
        content.addSubview_(notice)
        y -= 32

        current_section = None
        for section, instruction in CONTROL_INSTRUCTIONS:
            if section != current_section:
                current_section = section
                heading = self._make_label(
                    current_section.upper(),
                    22,
                    y,
                    self.WIDTH - 44,
                    21,
                    bold=True,
                    secondary=True,
                    font_size=11.0,
                )
                content.addSubview_(heading)
                y -= 24
            row = self._make_label(
                instruction,
                24,
                y,
                self.WIDTH - 48,
                22,
                font_size=11.8,
            )
            content.addSubview_(row)
            y -= 24

        footer = self._make_label(
            "Click Hide; press P in the camera preview to show this guide again.",
            22,
            13,
            self.WIDTH - 36,
            20,
            secondary=True,
            font_size=10.5,
        )
        content.addSubview_(footer)
        self.show()

    def _make_label(
        self,
        text,
        x,
        y,
        width,
        height,
        *,
        bold=False,
        secondary=False,
        font_size=12.0,
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
            self._kit["NSFont"].boldSystemFontOfSize_(font_size)
            if bold
            else self._kit["NSFont"].systemFontOfSize_(font_size)
        )
        return label

    def show(self) -> None:
        self.visible = True
        self._rendered_status = None
        self.update_status(self.status)
        self._panel.orderFrontRegardless()

    def hide(self) -> None:
        self.visible = False
        self._panel.orderOut_(None)

    def toggle_visibility(self) -> None:
        if self.visible:
            self.hide()
        else:
            self.show()

    def update_status(self, status: ControlPanelStatus) -> None:
        self.status = status
        if not self.visible:
            return
        previous = self._rendered_status
        if previous is None or previous.mode != status.mode:
            self._mode_label.setStringValue_(f"●  {status.mode}")
            if status.mode.startswith("LIVE"):
                mode_color = self._kit["NSColor"].systemGreenColor()
            elif status.mode.startswith("PAUSED"):
                mode_color = self._kit["NSColor"].systemOrangeColor()
            else:
                mode_color = self._kit["NSColor"].secondaryLabelColor()
            self._mode_label.setTextColor_(mode_color)
        if (
            previous is None
            or previous.left_hand != status.left_hand
            or previous.right_hand != status.right_hand
        ):
            self._hands_label.setStringValue_(
                f"LEFT  {status.left_hand}     •     RIGHT  {status.right_hand}"
            )
        if (
            previous is None
            or previous.fps != status.fps
            or previous.processing_ms != status.processing_ms
        ):
            self._performance_label.setStringValue_(
                f"{status.fps:.1f} FPS  •  {status.processing_ms:.1f} ms"
            )
        self._rendered_status = status

    def pump(self) -> None:
        kit = self._kit
        while True:
            event = self._app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                kit["NSEventMaskAny"],
                kit["NSDate"].dateWithTimeIntervalSinceNow_(0),
                kit["NSDefaultRunLoopMode"],
                True,
            )
            if event is None:
                break
            self._app.sendEvent_(event)
        self._app.updateWindows()

    def close(self) -> None:
        self.hide()
        self._panel.close()
