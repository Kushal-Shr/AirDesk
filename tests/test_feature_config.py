import json
import tempfile
import unittest
from pathlib import Path

from airdesk.feature_config import DEFAULT_FEATURES, FeatureConfigStore


class FeatureConfigStoreTests(unittest.TestCase):
    def test_new_configuration_starts_with_every_feature_off(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeatureConfigStore(Path(directory) / "features.json")

            self.assertEqual(store.as_dict(), DEFAULT_FEATURES)
            self.assertFalse(any(store.as_dict().values()))

    def test_toggle_is_saved_and_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.json"
            store = FeatureConfigStore(path)

            store.set("pointer", True)
            reloaded = FeatureConfigStore(path)

            self.assertTrue(reloaded.get("pointer"))
            self.assertFalse(reloaded.get("close_window"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 1)

    def test_invalid_json_falls_back_to_safe_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.json"
            path.write_text("not json", encoding="utf-8")

            store = FeatureConfigStore(path)

            self.assertFalse(any(store.as_dict().values()))
            self.assertIsNotNone(store.last_error)

    def test_unknown_and_non_boolean_values_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 99,
                        "features": {
                            "pointer": "yes",
                            "copy": True,
                            "unknown_feature": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            store = FeatureConfigStore(path)

            self.assertFalse(store.get("pointer"))
            self.assertFalse(store.get("copy"))
            self.assertNotIn("unknown_feature", store.as_dict())

    def test_disruptive_command_must_be_reenabled_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.json"
            store = FeatureConfigStore(path)
            store.set("close_window", True)

            self.assertTrue(store.get("close_window"))
            self.assertFalse(FeatureConfigStore(path).get("close_window"))


if __name__ == "__main__":
    unittest.main()
