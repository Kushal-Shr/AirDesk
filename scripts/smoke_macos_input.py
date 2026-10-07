"""Exercise real input only in a temporary AirDesk-owned test window."""

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    import AppKit as A
    import Quartz
    import pyautogui
    from Foundation import NSDate, NSMakeRect
    from airdesk.air_mouse import SystemMouseController
    from airdesk.desktop_controls import MacEscapeMonitor

    if not Quartz.CGPreflightPostEventAccess():
        raise RuntimeError("Accessibility permission is required for this input test")

    clicks = []
    drags = []

    class AirDeskDemoTestTextView(A.NSTextView):
        def mouseDown_(self, event):
            # Do not enter NSTextView's blocking mouse tracking loop: this
            # diagnostic pumps events itself so it can post the matching up.
            clicks.append(event.clickCount())

        def mouseDragged_(self, event):
            drags.append(event.locationInWindow())

    app = A.NSApplication.sharedApplication()
    # Cocoa resolves standard Command shortcuts through the app menu. Supply
    # the Edit item that an ordinary text editor has, then test real dispatch.
    menu = A.NSMenu.alloc().initWithTitle_("AirDesk Input Test")
    edit_item = A.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Edit", None, "")
    edit_menu = A.NSMenu.alloc().initWithTitle_("Edit")
    select_all = A.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
        "Select All", "selectAll:", "a"
    )
    select_all.setKeyEquivalentModifierMask_(A.NSEventModifierFlagCommand)
    edit_menu.addItem_(select_all)
    edit_item.setSubmenu_(edit_menu)
    menu.addItem_(edit_item)
    app.setMainMenu_(menu)
    previous_app = A.NSWorkspace.sharedWorkspace().frontmostApplication()
    previous_position = tuple(pyautogui.position())
    window = A.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(150, 200, 600, 240), A.NSWindowStyleMaskTitled,
        A.NSBackingStoreBuffered, False,
    )
    window.setReleasedWhenClosed_(False)
    window.setTitle_("AirDesk input verification — temporary test field")
    view = AirDeskDemoTestTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 600, 240))
    view.setAutomaticDashSubstitutionEnabled_(False)
    view.setAutomaticQuoteSubstitutionEnabled_(False)
    view.setAutomaticTextReplacementEnabled_(False)
    view.setAutomaticSpellingCorrectionEnabled_(False)
    window.setContentView_(view)
    controller = SystemMouseController(pyautogui)
    monitor = MacEscapeMonitor()

    def pump(seconds=0.15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                A.NSEventMaskAny, NSDate.dateWithTimeIntervalSinceNow_(0.01),
                A.NSDefaultRunLoopMode, True,
            )
            if event is not None:
                app.sendEvent_(event)
            app.updateWindows()

    def require_our_focus():
        if app.keyWindow() != window or window.firstResponder() != view:
            raise RuntimeError("Test window lost focus; input test aborted")
        if not A.NSRunningApplication.currentApplication().isActive():
            raise RuntimeError("Another application is active; input test aborted")

    try:
        window.makeKeyAndOrderFront_(None)
        app.activateIgnoringOtherApps_(True)
        window.makeFirstResponder_(view)
        pump()
        require_our_focus()
        assert controller.commit_text("Hi AbZ 2! @#$%&+-_=()[]{}/: ")
        pump()
        expected = "Hi AbZ 2! @#$%&+-_=()[]{}/: "
        if view.string() != expected:
            raise RuntimeError(f"Text mismatch: {view.string()!r}")
        print("PASS real mixed-case, digits, symbols, trailing-space insertion", flush=True)

        require_our_focus()
        controller.toggle()
        controller.hotkey("command", "a")
        pump()
        if view.selectedRange().length != len(expected):
            raise RuntimeError("Command+A did not select the test field's text")
        print("PASS real command shortcut (Select All)", flush=True)

        monitor.start()
        require_our_focus()
        pyautogui.press("esc", _pause=False)
        pump()
        if not monitor.consume_escape():
            raise RuntimeError("Global Escape monitor did not receive Escape")
        print("PASS global Escape detection", flush=True)

        require_our_focus()
        # Quartz coordinates use the primary display's top-left origin.
        screen_height = A.NSScreen.screens()[0].frame().size.height
        point = (350, round(screen_height - 300))
        controller.move_from_preview(point, controller.screen_size)
        controller.click("left")
        controller.second_click()
        pump()
        if clicks[-2:] != [1, 2]:
            raise RuntimeError(f"Native click counts were {clicks!r}")
        print("PASS native double click (click counts 1, 2)", flush=True)

        require_our_focus()
        controller.mouse_down()
        pump(0.05)
        controller.move_from_preview((point[0] + 40, point[1] + 20), controller.screen_size)
        pump(0.05)
        controller.mouse_up()
        pump()
        if not drags:
            raise RuntimeError("macOS received no mouseDragged event")
        print("PASS native drag and mouse release", flush=True)
    finally:
        controller.close()
        monitor.close()
        window.close()
        # Restore the user's app and pointer after the isolated test.
        if previous_app is not None:
            previous_app.activateWithOptions_(A.NSApplicationActivateIgnoringOtherApps)
        pyautogui.moveTo(*previous_position, _pause=False)


if __name__ == "__main__":
    main()
