"""
tests/test_gui_settings.py — settings JSON round-trip and resilience.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.append(os.getcwd())

from cull.gui.settings import AppSettings, load_settings, save_settings, settings_dir


def test_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings(last_dir="C:/photos", top_n=7, workers=2, conf=0.4,
                           p4_policy="never", dry_run=True)
    save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded == settings


def test_missing_file_returns_defaults(tmp_path):
    loaded = load_settings(tmp_path / "nope.json")
    assert loaded == AppSettings()


def test_corrupt_json_returns_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{ not json", encoding="utf-8")
    assert load_settings(path) == AppSettings()


def test_partial_json_keeps_valid_fields(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"top_n": 5, "bogus_field": 1}), encoding="utf-8")
    loaded = load_settings(path)
    assert loaded.top_n == 5
    assert loaded.workers == AppSettings().workers  # missing -> default


def test_type_coercion_from_strings(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"top_n": "9", "recursive": "true", "conf": "0.5"}),
                    encoding="utf-8")
    loaded = load_settings(path)
    assert loaded.top_n == 9
    assert loaded.recursive is True
    assert loaded.conf == 0.5


def test_bad_values_fall_back_to_default(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"top_n": "abc", "min_raw": {"x": 1}}), encoding="utf-8")
    loaded = load_settings(path)
    assert loaded.top_n == AppSettings().top_n
    assert loaded.min_raw == AppSettings().min_raw


def test_settings_dir_is_absolute_path():
    d = settings_dir()
    assert isinstance(d, Path)
    assert d.is_absolute()
    assert d.name == "AutoCullingF1"
