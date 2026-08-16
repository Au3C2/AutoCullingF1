"""settings.py — GUI settings persistence (JSON, zero dependencies)."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path

APP_DIR_NAME = "AutoCullingF1"
SETTINGS_FILE = "settings.json"


@dataclass
class AppSettings:
    """Persisted GUI preferences, mirroring the EngineConfig parameters."""

    last_dir: str = ""
    recursive: bool = False
    top_n: int = 11
    p4_policy: str = "auto"
    scale_width: int = 1280
    workers: int = 4
    force: bool = False
    sharp_thresh: float = 0.05
    w_sharp: float = 1.5
    w_comp: float = 2.5
    min_raw: float = 3.1
    conf: float = 0.25
    autocrop: bool = True
    rename: bool = False
    dry_run: bool = False
    rf_api_key: str = ""


def settings_dir() -> Path:
    """Per-user config directory for the current platform."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_DIR_NAME


def _coerce(default, value):
    """Coerce a raw JSON value to the dataclass field type; fall back on failure."""
    if isinstance(value, type(default)):
        return value
    try:
        if isinstance(default, bool):
            return bool(value)
        return type(default)(value)
    except (TypeError, ValueError):
        return default


def load_settings(path: Path | None = None) -> AppSettings:
    """Load settings from JSON; a missing or corrupt file yields defaults."""
    path = path or (settings_dir() / SETTINGS_FILE)
    defaults = AppSettings()
    if not path.exists():
        return defaults
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return defaults
    if not isinstance(raw, dict):
        return defaults
    values = {}
    for f in fields(AppSettings):
        if f.name in raw:
            values[f.name] = _coerce(getattr(defaults, f.name), raw[f.name])
    return AppSettings(**values)


def save_settings(settings: AppSettings, path: Path | None = None) -> None:
    """Persist settings as JSON, creating the config directory if needed."""
    path = path or (settings_dir() / SETTINGS_FILE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        # The GUI must never crash because the config dir is not writable.
        pass
