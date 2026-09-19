"""Layer 2 GUI Feature Suite test: Runs Node.js test scripts for features 1-7.

Tests:
1. Keyboard navigation & culling hotkeys (Feature 1)
2. Collapsible & compact config panel (Feature 2)
3. Preview Pan & Zoom interactions (Feature 3)
4. Virtual / chunked table performance & anti-race (Feature 4)
5. Burst grouping & single shot interleaving (Feature 6)
6. Context menu boundary clamping & dispatch (Feature 7)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
import pytest

TEST_SCRIPTS = [
    "test_keyboard_shortcuts.js",
    "test_config_panel.js",
    "test_preview_zoom.js",
    "test_table_performance.js",
    "test_grouping_logic.js",
    "test_format_tag.js",
    "test_context_menu.js",
    "test_dom_elements_sync.js",
    "test_app_js_syntax.js",
    "test_app_js_i18n_calls.js",
]


@pytest.mark.parametrize("script_name", TEST_SCRIPTS)
def test_gui_feature_script(script_name: str) -> None:
    script_path = Path(__file__).parent / script_name
    assert script_path.exists(), f"Test script {script_name} not found"

    res = subprocess.run(
        ["node", str(script_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, (
        f"Node test {script_name} failed with code {res.returncode}:\n"
        f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
