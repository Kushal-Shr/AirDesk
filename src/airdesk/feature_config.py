"""Persist AirDesk feature choices without changing gesture behavior yet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


CONFIG_VERSION = 1
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "airdesk_features.json"


@dataclass(frozen=True)
class FeatureDefinition:
    """Metadata used by both the JSON store and the native control panel."""

    key: str
    label: str
    section: str
    disruptive: bool = False


FEATURE_DEFINITIONS = (
    FeatureDefinition("pointer", "Pointer", "Desktop controls"),
    FeatureDefinition("left_click", "Left Click", "Desktop controls"),
    FeatureDefinition("double_click", "Double Click", "Desktop controls"),
    FeatureDefinition("right_click", "Right Click", "Desktop controls"),
    FeatureDefinition("drag_selection", "Drag / Text Selection", "Desktop controls"),
    FeatureDefinition("scroll", "Scroll", "Desktop controls"),
    FeatureDefinition("mission_control", "Mission Control", "Desktop controls"),
    FeatureDefinition("app_switching", "App Switching", "Desktop controls"),
    FeatureDefinition("air_command_palette", "Air Command Palette", "Commands"),
    FeatureDefinition("spotlight_search", "Spotlight Search", "Commands", True),
    FeatureDefinition("copy", "Copy", "Commands", True),
    FeatureDefinition("paste", "Paste", "Commands", True),
    FeatureDefinition("cut", "Cut", "Commands", True),
    FeatureDefinition("undo", "Undo", "Commands", True),
    FeatureDefinition("redo", "Redo", "Commands", True),
    FeatureDefinition("select_all", "Select All", "Commands", True),
    FeatureDefinition("save_document", "Save Document", "Commands", True),
    FeatureDefinition("screenshot", "Screenshot", "Commands", True),
    FeatureDefinition("close_window", "Close Window", "Commands", True),
)
FEATURE_KEYS = tuple(feature.key for feature in FEATURE_DEFINITIONS)
DEFAULT_FEATURES = {key: False for key in FEATURE_KEYS}
DISRUPTIVE_FEATURES = {
    feature.key for feature in FEATURE_DEFINITIONS if feature.disruptive
}


class FeatureConfigStore:
    """Load and atomically save a small, validated feature configuration."""

    def __init__(self, path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.path = Path(path)
        self.last_error: str | None = None
        self._features = dict(DEFAULT_FEATURES)
        self.load()

    def load(self) -> dict[str, bool]:
        """Load known Boolean values, falling back safely for invalid files."""
        self._features = dict(DEFAULT_FEATURES)
        self.last_error = None
        if not self.path.exists():
            return self.as_dict()

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            stored_features = payload.get("features", {})
            if not isinstance(stored_features, dict):
                raise ValueError("'features' must be a JSON object")
            for key in FEATURE_KEYS:
                value = stored_features.get(key)
                # Command output must be deliberately re-enabled each launch.
                if key not in DISRUPTIVE_FEATURES and isinstance(value, bool):
                    self._features[key] = value
        except (OSError, json.JSONDecodeError, ValueError, AttributeError) as error:
            self.last_error = str(error)
        return self.as_dict()

    def save(self) -> None:
        """Replace the file atomically so an interrupted write cannot corrupt it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(f"{self.path.name}.tmp")
        payload = {
            "version": CONFIG_VERSION,
            "features": self.as_dict(),
        }
        try:
            temporary_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(self.path)
            self.last_error = None
        except OSError as error:
            self.last_error = str(error)
            raise

    def get(self, key: str) -> bool:
        if key not in self._features:
            raise KeyError(f"Unknown AirDesk feature: {key}")
        return self._features[key]

    def set(self, key: str, enabled: bool) -> None:
        if key not in self._features:
            raise KeyError(f"Unknown AirDesk feature: {key}")
        if not isinstance(enabled, bool):
            raise TypeError("Feature state must be a Boolean")
        self._features[key] = enabled
        self.save()

    def as_dict(self) -> dict[str, bool]:
        return dict(self._features)
