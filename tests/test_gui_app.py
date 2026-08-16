"""
tests/test_gui_app.py — Tkinter view-layer tests, driven by pumping the event
loop (no mainloop). Requires a display; skipped automatically without one.
"""

import os
import shutil
import sys
import time
from pathlib import Path

import pytest

sys.path.append(os.getcwd())

TEST_IMG = Path("tests/test_img")


@pytest.fixture
def app_ui(tmp_path):
    """Fresh hidden Tk root with a CullApp; skip when no display exists."""
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as exc:
        print(f"\n[FIXTURE] Tk() failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        pytest.skip(f"no display server available: {exc}")
    root.withdraw()
    from cull.gui.app import CullApp
    app = CullApp(root, settings_path=tmp_path / "settings.json")
    root.update()
    yield root, app
    # Stop and join any still-running worker: CullWorker hooks the ROOT logger,
    # so a leftover engine would contaminate later tests' message queues.
    if app.worker and app.worker.is_running():
        app.worker.stop()
        app.worker.wait(timeout=60)
    try:
        app.close()  # cancels the poll loop before destroying the root
    except Exception:
        pass


def btn_state(app, name: str) -> str:
    """Return the widget state as a plain string (CustomTkinter API)."""
    return str(getattr(app, name).cget("state"))


def pump_until(root, app, condition, timeout: float = 240.0) -> bool:
    """Drive the event loop until *condition* holds or *timeout* seconds pass."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root.update()
        if condition():
            return True
        time.sleep(0.02)
    return False


def drain_queue(root, app, rounds: int = 20):
    for _ in range(rounds):
        root.update()
        time.sleep(0.02)


def copy_photos(dst: Path, copies: int = 1) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for i in range(copies):
        for src in sorted(TEST_IMG.glob("*.jpg")):
            shutil.copy(src, dst / f"{i:02d}_{src.name}")


def configure(app, input_dir: Path) -> None:
    app.dir_var.set(str(input_dir))
    app.var_workers.set("1")
    app.var_scale_width.set("512")
    app.var_force.set(True)
    app.var_dry_run.set(True)


def test_app_builds(app_ui):
    root, app = app_ui
    assert app.tree is not None
    assert len(app.tree["columns"]) == 8
    assert btn_state(app, "btn_start") == "normal"
    assert btn_state(app, "btn_stop") == "disabled"
    assert btn_state(app, "btn_export") == "disabled"
    assert app.progress.get() == 0.0
    assert app.stage_var.get() == "就绪"


def test_app_full_run(app_ui):
    root, app = app_ui
    configure(app, TEST_IMG.resolve())
    app.btn_start.invoke()

    assert app.worker is not None
    assert app.worker.is_running()
    assert btn_state(app, "btn_start") == "disabled"  # locked while running

    assert pump_until(root, app, lambda: not app.worker.is_running()), "run did not finish"
    drain_queue(root, app)

    assert app.progress.get() == 1.0
    assert app.stage_var.get() == "100%  完成"
    assert "完成" in app.status_var.get()
    assert len(app.tree.get_children()) == 6  # one row per test image
    assert app._log_text.get("1.0", "end").strip(), "log panel should have content"
    assert btn_state(app, "btn_export") == "normal"


def test_app_table_filter_and_sort(app_ui):
    """Filter (keep-only) and column sorting reorder the tree without breaking rows."""
    root, app = app_ui
    configure(app, TEST_IMG.resolve())
    app.btn_start.invoke()
    assert pump_until(root, app, lambda: not app.worker.is_running()), "run did not finish"
    drain_queue(root, app)

    all_rows = len(app.tree.get_children())
    assert all_rows == 6

    app.filter_var.set("仅保留")
    app._rebuild_rows()
    keep_rows = len(app.tree.get_children())
    assert 0 < keep_rows <= all_rows

    app.filter_var.set("全部")
    app._sort_by("raw")
    values = [float(app.tree.item(i)["values"][2]) for i in app.tree.get_children()]
    assert values == sorted(values, reverse=False)  # first click sorts ascending

    # Clicking the same column again flips the order.
    app._sort_by("raw")
    values_rev = [float(app.tree.item(i)["values"][2]) for i in app.tree.get_children()]
    assert values_rev == sorted(values_rev, reverse=True)


def test_app_cancel(app_ui, tmp_path):
    root, app = app_ui
    dir_many = tmp_path / "many"
    copy_photos(dir_many, copies=6)  # 36 images so the run outlasts the cancel
    configure(app, dir_many)
    app.btn_start.invoke()

    assert app.worker.is_running()
    time.sleep(0.5)  # let scan/models start, then interrupt
    app.btn_stop.invoke()

    assert pump_until(root, app, lambda: not app.worker.is_running(), timeout=120), "cancel did not finish"
    drain_queue(root, app)

    assert app.stage_var.get() == "已取消"
    assert "已取消" in app.status_var.get()
    assert btn_state(app, "btn_start") == "normal"
    assert btn_state(app, "btn_stop") == "disabled"


def test_app_preview_during_run(app_ui, tmp_path):
    """A row selected while scoring is still active must render its preview.

    Regression: streamed rows used auto-generated iids ("I001") that crashed
    the int() conversion in _selected_score, so previews only worked after
    the run finished.
    """
    root, app = app_ui
    dir_many = tmp_path / "preview"
    copy_photos(dir_many, copies=8)  # 48 images -> the run outlasts the preview
    configure(app, dir_many)
    app.btn_start.invoke()

    assert pump_until(root, app, lambda: len(app.tree.get_children()) > 0), "no streamed rows"
    app.tree.selection_set("0")
    assert pump_until(root, app, lambda: app._preview_ctk is not None, timeout=30), \
        "preview never rendered"
    assert app.worker.is_running(), "run ended before the preview rendered"
    assert app.preview_label.cget("text") == ""  # placeholder cleared, photo shown


def test_app_close_saves_settings(app_ui, tmp_path):
    root, app = app_ui
    app.dir_var.set(str(Path("C:/some/photos")))
    app.var_top_n.set("7")
    app.var_dry_run.set(True)
    app._on_close()  # no worker running -> saves settings and destroys root

    from cull.gui.settings import load_settings
    saved = load_settings(tmp_path / "settings.json")
    assert saved.last_dir == str(Path("C:/some/photos"))
    assert saved.top_n == 7
    assert saved.dry_run is True
