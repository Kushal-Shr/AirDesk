"""Small deterministic tests for Stage 3's safety logic."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from airdesk.hand_lock import HandLockState, classify_fist


def make_landmarks(closed: bool):
    points = [SimpleNamespace(x=0.5, y=0.7, z=0.0) for _ in range(21)]
    points[0] = SimpleNamespace(x=0.5, y=0.9, z=0.0)

    for finger_index, (mcp_id, pip_id, dip_id, tip_id) in enumerate(
        ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    ):
        x = 0.38 + finger_index * 0.08
        points[mcp_id] = SimpleNamespace(x=x, y=0.65, z=0.0)
        points[pip_id] = SimpleNamespace(x=x, y=0.50, z=0.0)
        points[dip_id] = SimpleNamespace(x=x, y=0.38 if not closed else 0.60, z=0.0)
        points[tip_id] = SimpleNamespace(x=x, y=0.20 if not closed else 0.72, z=0.0)

    return points


class FistClassifierTests(unittest.TestCase):
    def test_open_hand_is_not_a_fist(self):
        self.assertFalse(classify_fist(make_landmarks(closed=False)).is_closed)

    def test_closed_hand_is_a_fist(self):
        self.assertTrue(classify_fist(make_landmarks(closed=True)).is_closed)


class HandLockStateTests(unittest.TestCase):
    def test_open_hand_waits_before_unlocking(self):
        state = HandLockState()
        state.update(fist_detected=False, hand_seen=True, now=1.0)
        state.update(fist_detected=False, hand_seen=True, now=1.24)
        self.assertTrue(state.locked)

        state.update(fist_detected=False, hand_seen=True, now=1.25)
        self.assertFalse(state.locked)

    def test_fist_locks_immediately(self):
        state = HandLockState(locked=False)
        state.update(fist_detected=True, hand_seen=True, now=2.0)
        self.assertTrue(state.locked)

    def test_missing_hand_is_locked(self):
        state = HandLockState(locked=False)
        state.update(fist_detected=False, hand_seen=False, now=3.0)
        self.assertTrue(state.locked)


if __name__ == "__main__":
    unittest.main()

