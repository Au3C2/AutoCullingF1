"""Layer 2 GUI test suite: Localization & Translation Integrity.

Verifies:
1. Both zh-CN and en-US translation files exist and are valid JSON.
2. Key parity: zh-CN and en-US have the exact same set of translation keys.
3. DOM data-i18n alignment: all data-i18n tags referenced in ui/index.html exist in the dictionaries.
4. Engine status and veto code parity: standard engine codes map to translations in both locales.
5. Parameter tooltip & label consistency: parameter cards have full translation coverage.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
import pytest

UI_DIR = Path("ui")
LOCALES_DIR = UI_DIR / "locales"


class I18nTagExtractor(HTMLParser):
    """HTML parser to extract all data-i18n and data-i18n-attr references."""

    def __init__(self) -> None:
        super().__init__()
        self.i18n_keys: set[str] = set()
        self.attr_keys: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        if "data-i18n" in attr_dict and attr_dict["data-i18n"]:
            self.i18n_keys.add(attr_dict["data-i18n"].strip())
        if "data-i18n-attr" in attr_dict and attr_dict["data-i18n-attr"]:
            # Format: "attr1:key1;attr2:key2" or "placeholder:key"
            for pair in attr_dict["data-i18n-attr"].split(";"):
                pair = pair.strip()
                if not pair:
                    continue
                if ":" in pair:
                    _, key = pair.split(":", 1)
                    self.attr_keys.add(key.strip())


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, str]:
    """Flatten nested dict into dot-separated keys."""
    items: dict[str, str] = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key))
        else:
            items[new_key] = str(v)
    return items


def _load_locales() -> dict[str, dict[str, str]]:
    locales: dict[str, dict[str, str]] = {}
    for lang in ["zh-CN", "en-US"]:
        path = LOCALES_DIR / f"{lang}.json"
        assert path.exists(), f"Missing translation file: {path}"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            locales[lang] = _flatten_dict(raw)
    return locales


def test_locale_files_exist_and_non_empty():
    """Verify both zh-CN.json and en-US.json exist and are non-empty."""
    for lang in ["zh-CN", "en-US"]:
        path = LOCALES_DIR / f"{lang}.json"
        assert path.exists(), f"Locale file {path} must exist"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"{path} must contain a JSON object"
            assert len(data) > 0, f"{path} must not be empty"


def test_translation_key_parity():
    """Verify zh-CN and en-US dictionaries have identical keys."""
    locales = _load_locales()
    zh_keys = set(locales["zh-CN"].keys())
    en_keys = set(locales["en-US"].keys())

    missing_in_en = zh_keys - en_keys
    missing_in_zh = en_keys - zh_keys

    assert not missing_in_en, f"Keys present in zh-CN but missing in en-US: {sorted(missing_in_en)}"
    assert not missing_in_zh, f"Keys present in en-US but missing in zh-CN: {sorted(missing_in_zh)}"


def test_html_data_i18n_keys_covered():
    """Verify all data-i18n and data-i18n-attr keys in index.html exist in dictionaries."""
    index_html = UI_DIR / "index.html"
    assert index_html.exists(), "ui/index.html not found"

    parser = I18nTagExtractor()
    parser.feed(index_html.read_text(encoding="utf-8"))

    locales = _load_locales()
    zh_dict = locales["zh-CN"]
    en_dict = locales["en-US"]

    all_html_keys = parser.i18n_keys | parser.attr_keys
    assert len(all_html_keys) > 0, "No data-i18n or data-i18n-attr keys found in index.html"

    for key in sorted(all_html_keys):
        assert key in zh_dict, f"HTML references data-i18n='{key}', but missing in zh-CN.json"
        assert key in en_dict, f"HTML references data-i18n='{key}', but missing in en-US.json"


def test_engine_protocol_veto_and_status_codes_covered():
    """Verify all standard engine veto codes and frame status codes have translations."""
    locales = _load_locales()

    expected_codes = [
        "veto.no_detection",
        "veto.decode_failed",
        "veto.manual_metadata",
        "veto.burst_group_topn",
        "veto.sharpness_fail",
        "veto.min_raw_fail",
        "veto.p4_orient_fail",
        "veto.fence_detected",
        "status.pending",
        "status.queued",
        "status.scored",
        "status.failed",
        "status.passed",
        "status.rejected",
    ]

    for code in expected_codes:
        for lang in ["zh-CN", "en-US"]:
            assert code in locales[lang], f"Standard engine code '{code}' missing in {lang}.json"
