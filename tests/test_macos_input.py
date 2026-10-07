from types import SimpleNamespace
from unittest.mock import Mock

from airdesk.macos_input import MacOSBackend


def test_double_click_posts_matching_down_up_counts_at_the_same_position():
    events = []
    quartz = SimpleNamespace(
        kCGMouseButtonLeft=0, kCGMouseButtonRight=1,
        kCGEventLeftMouseDown=1, kCGEventLeftMouseUp=2,
        kCGEventRightMouseDown=3, kCGEventRightMouseUp=4,
        kCGMouseEventClickState=7, kCGHIDEventTap=0,
        CGEventCreateMouseEvent=lambda source, kind, point, button: {
            "kind": kind, "point": point, "button": button
        },
        CGEventSetIntegerValueField=lambda event, field, count: event.update(count=count),
        CGEventPost=lambda tap, event: events.append(event),
    )
    backend = Mock()
    backend.position.return_value = (200, 300)
    sleep = Mock()
    MacOSBackend(backend, quartz, sleep).doubleClick()
    assert [(e["kind"], e["count"]) for e in events] == [(1, 1), (2, 1), (1, 2), (2, 2)]
    assert all(e["point"] == (200, 300) for e in events)
    sleep.assert_called_once_with(0.12)
    assert backend.failSafeCheck.call_count == 2


def test_followup_click_uses_native_click_count_two():
    events = []
    quartz = SimpleNamespace(
        kCGMouseButtonLeft=0, kCGMouseButtonRight=1,
        kCGEventLeftMouseDown=1, kCGEventLeftMouseUp=2,
        kCGEventRightMouseDown=3, kCGEventRightMouseUp=4,
        kCGMouseEventClickState=7, kCGHIDEventTap=0,
        CGEventCreateMouseEvent=lambda source, kind, point, button: {
            "kind": kind, "point": point, "button": button
        },
        CGEventSetIntegerValueField=lambda event, field, count: event.update(count=count),
        CGEventPost=lambda tap, event: events.append(event),
    )
    backend = Mock()
    backend.position.return_value = (200, 300)
    MacOSBackend(backend, quartz).secondClick()
    assert [(e["kind"], e["count"]) for e in events] == [(1, 2), (2, 2)]


def test_text_uses_unicode_with_exact_case_symbols_and_utf16_length():
    events = []
    quartz = SimpleNamespace(
        kCGHIDEventTap=0,
        CGEventCreateKeyboardEvent=lambda source, code, down: {"down": down},
        CGEventSetFlags=lambda event, flags: event.update(flags=flags),
        CGEventKeyboardSetUnicodeString=lambda event, length, text: event.update(
            length=length, text=text
        ),
        CGEventPost=lambda tap, event: events.append(event),
    )
    text = "Hi 2! @#$%& 🙂 "
    MacOSBackend(Mock(), quartz).write(text)
    assert len(events) == 2
    assert events[0]["text"] == text
    assert events[0]["length"] == len(text.encode("utf-16-le")) // 2
    assert events[0]["flags"] == 0
    assert events[0]["down"] and not events[1]["down"]


def test_shortcut_events_carry_modifiers_and_release_them_in_reverse_order():
    events = []
    quartz = SimpleNamespace(
        kCGHIDEventTap=0,
        kCGEventFlagMaskCommand=8, kCGEventFlagMaskShift=4,
        kCGEventFlagMaskControl=2, kCGEventFlagMaskAlternate=1,
        CGEventCreateKeyboardEvent=lambda source, code, down: {"code": code, "down": down},
        CGEventSetFlags=lambda event, flags: event.update(flags=flags),
        CGEventPost=lambda tap, event: events.append(event),
    )
    backend = Mock()
    backend.platformModule.keyboardMapping = {"command": 55, "shift": 56, "z": 6}
    MacOSBackend(backend, quartz, sleep=Mock()).hotkey("command", "shift", "z")
    assert [(e["code"], e["down"], e["flags"]) for e in events] == [
        (55, True, 8), (56, True, 12), (6, True, 12),
        (6, False, 12), (56, False, 8), (55, False, 0),
    ]


def test_mission_control_launches_native_macos_application():
    backend = Mock()
    launch = Mock()

    MacOSBackend(backend, Mock(), launch=launch).showMissionControl()

    backend.failSafeCheck.assert_called_once_with()
    args, kwargs = launch.call_args
    assert args[0] == ["/usr/bin/open", "-a", "Mission Control"]
    assert kwargs["start_new_session"] is True
