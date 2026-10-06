"""Transparent, click-through macOS ink overlay for AirDesk."""

from __future__ import annotations


class MemoryInkOverlay:
    """Small in-memory overlay used by controller tests."""

    def __init__(self, size: tuple[int, int] = (1440, 900)) -> None:
        self.size = size
        self.strokes: list[list[tuple[int, int]]] = []
        self.visible = False
        self.status = ""

    def show(self) -> None:
        self.visible = True

    def hide(self) -> None:
        self.visible = False
        self.end_stroke()

    def add_point(self, point: tuple[int, int]) -> None:
        if not self.strokes or not self.strokes[-1]:
            self.strokes.append([])
        self.strokes[-1].append(point)

    def end_stroke(self) -> None:
        if self.strokes and not self.strokes[-1]:
            self.strokes.pop()
        if self.strokes:
            self.strokes.append([])

    def clear(self) -> None:
        self.strokes.clear()

    def set_status(self, text: str) -> None:
        self.status = text

    def pump(self) -> None:
        pass

    def close(self) -> None:
        self.hide()


def _load_appkit():
    """Import native UI modules lazily so controller tests stay platform-safe."""
    import objc
    from AppKit import (
        NSApplication,
        NSApplicationActivationPolicyAccessory,
        NSBackingStoreBuffered,
        NSBezierPath,
        NSColor,
        NSDefaultRunLoopMode,
        NSEventMaskAny,
        NSFont,
        NSFontAttributeName,
        NSForegroundColorAttributeName,
        NSLineCapStyleRound,
        NSLineJoinStyleRound,
        NSPanel,
        NSScreen,
        NSStatusWindowLevel,
        NSView,
        NSWindowCollectionBehaviorCanJoinAllSpaces,
        NSWindowCollectionBehaviorFullScreenAuxiliary,
        NSWindowCollectionBehaviorIgnoresCycle,
        NSWindowCollectionBehaviorStationary,
        NSWindowStyleMaskBorderless,
        NSWindowStyleMaskNonactivatingPanel,
    )
    from Foundation import NSDate, NSMakePoint, NSMakeRect, NSString

    return locals()


class NativeInkOverlay:
    """A non-activating AppKit panel that displays ink above the desktop."""

    def __init__(self) -> None:
        kit = _load_appkit()
        objc = kit["objc"]
        NSView = kit["NSView"]
        NSPanel = kit["NSPanel"]

        class InkView(NSView):
            def initWithFrame_(view_self, frame):
                view_self = objc.super(InkView, view_self).initWithFrame_(frame)
                if view_self is None:
                    return None
                view_self.ink_strokes = []
                view_self.status_text = ""
                return view_self

            def isOpaque(view_self):
                return False

            def drawRect_(view_self, _dirty_rect):
                height = view_self.bounds().size.height
                width = view_self.bounds().size.width
                kit["NSColor"].colorWithCalibratedRed_green_blue_alpha_(
                    0.16, 0.47, 1.0, 0.96
                ).setStroke()
                for stroke in view_self.ink_strokes:
                    if len(stroke) < 2:
                        continue
                    path = kit["NSBezierPath"].bezierPath()
                    path.setLineWidth_(6.0)
                    path.setLineCapStyle_(kit["NSLineCapStyleRound"])
                    path.setLineJoinStyle_(kit["NSLineJoinStyleRound"])
                    first_x, first_y = stroke[0]
                    path.moveToPoint_(kit["NSMakePoint"](first_x, height - first_y))
                    for point_x, point_y in stroke[1:]:
                        path.lineToPoint_(kit["NSMakePoint"](point_x, height - point_y))
                    path.stroke()

                panel = kit["NSBezierPath"].bezierPathWithRoundedRect_xRadius_yRadius_(
                    kit["NSMakeRect"](16, height - 66, min(width - 32, 900), 48),
                    12,
                    12,
                )
                kit["NSColor"].colorWithCalibratedWhite_alpha_(0.08, 0.72).setFill()
                panel.fill()
                attributes = {
                    kit["NSFontAttributeName"]: kit["NSFont"].systemFontOfSize_weight_(15.0, 0.35),
                    kit["NSForegroundColorAttributeName"]: kit["NSColor"].whiteColor(),
                }
                kit["NSString"].stringWithString_(view_self.status_text).drawAtPoint_withAttributes_(
                    kit["NSMakePoint"](32, height - 50), attributes
                )

        class NonActivatingPanel(NSPanel):
            def canBecomeKeyWindow(panel_self):
                return False

            def canBecomeMainWindow(panel_self):
                return False

        self._kit = kit
        self._app = kit["NSApplication"].sharedApplication()
        self._app.setActivationPolicy_(kit["NSApplicationActivationPolicyAccessory"])
        screen_frame = kit["NSScreen"].mainScreen().frame()
        self.size = (round(screen_frame.size.width), round(screen_frame.size.height))
        style = kit["NSWindowStyleMaskBorderless"] | kit["NSWindowStyleMaskNonactivatingPanel"]
        self._panel = NonActivatingPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            screen_frame, style, kit["NSBackingStoreBuffered"], False
        )
        self._panel.setOpaque_(False)
        self._panel.setBackgroundColor_(kit["NSColor"].clearColor())
        self._panel.setHasShadow_(False)
        self._panel.setIgnoresMouseEvents_(True)
        self._panel.setLevel_(kit["NSStatusWindowLevel"])
        self._panel.setHidesOnDeactivate_(False)
        self._panel.setCollectionBehavior_(
            kit["NSWindowCollectionBehaviorCanJoinAllSpaces"]
            | kit["NSWindowCollectionBehaviorFullScreenAuxiliary"]
            | kit["NSWindowCollectionBehaviorStationary"]
            | kit["NSWindowCollectionBehaviorIgnoresCycle"]
        )
        self._view = InkView.alloc().initWithFrame_(screen_frame)
        self._panel.setContentView_(self._view)
        self.visible = False

    @property
    def strokes(self):
        return self._view.ink_strokes

    def show(self) -> None:
        self.visible = True
        self._panel.orderFrontRegardless()
        self._view.setNeedsDisplay_(True)

    def hide(self) -> None:
        self.visible = False
        self.end_stroke()
        self._panel.orderOut_(None)

    def add_point(self, point: tuple[int, int]) -> None:
        if not self.strokes or not self.strokes[-1]:
            self.strokes.append([])
        self.strokes[-1].append(point)
        self._view.setNeedsDisplay_(True)

    def end_stroke(self) -> None:
        if self.strokes and not self.strokes[-1]:
            self.strokes.pop()
        if self.strokes:
            self.strokes.append([])

    def clear(self) -> None:
        self.strokes.clear()
        self._view.setNeedsDisplay_(True)

    def set_status(self, text: str) -> None:
        self._view.status_text = text
        self._view.setNeedsDisplay_(True)

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
