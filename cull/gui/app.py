"""app.py — CustomTkinter main window for Auto-Culling."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk

from cull.engine import CullingEngine, EngineConfig
from cull.gui.preview import render_pil
from cull.gui.settings import AppSettings, load_settings, save_settings
from cull.gui.worker import (
    CANCELLED, DONE, ERROR, FRAME, LOG_LINE, PATHS, STAGE, TOTAL, CullWorker,
)
from cull.scorer import ImageScore

log = logging.getLogger(__name__)

POLL_MS = 100          # main-thread queue polling interval
MAX_LOG_LINES = 2000   # log panel cap; oldest lines are dropped
FILTER_ALL = "全部"
FILTER_KEEP = "仅保留"
FILTER_DISCARD = "仅丢弃"

# Bounded work per poll tick: draining every queued message in one go stalls
# the main thread during heavy scoring (each frame pushes FRAME + LOG_LINE),
# which in turn starves preview updates until the run ends.
_MAX_MSGS_PER_POLL = 200

# Fixed preview pane width (px). The pane never resizes with its content, so
# switching between the placeholder text and a photo does not reflow the layout.
PREVIEW_W = 520

# Engine progress callbacks are not proportional to wall time (scoring takes
# ~80% of the run but spans 0.90->0.95), so we remap the phases onto a scale
# where the parallel scoring phase spreads across most of the bar.
_STAGE_SCALE = {
    "Collecting images...": 0.05,
    "Renaming images...": 0.10,
    "Reading EXIF metadata...": 0.15,
    "Grouping burst sequences...": 0.20,
    "Loading models...": 0.30,
    "Analyzing images...": 0.35,
    "Saving metadata...": 0.96,
    "Done!": 1.0,
    "Cancelled": 0.0,
}
_SCORE_START = 0.35
_SCORE_END = 0.96

_COLUMNS = (
    ("rating", "星级", 60, "center"),
    ("name", "文件", 240, "w"),
    ("raw", "评分", 70, "e"),
    ("sharp", "锐度", 70, "e"),
    ("comp", "构图", 70, "e"),
    ("veto", "否决原因", 160, "w"),
    ("group", "连拍组", 70, "center"),
    ("manual", "人工", 60, "center"),
)

_SORT_KEYS = {
    "rating": lambda s: s.rating,
    "name": lambda s: s.path.name.lower(),
    "raw": lambda s: s.raw_score,
    "sharp": lambda s: s.s_sharp,
    "comp": lambda s: s.s_comp,
    "veto": lambda s: s.veto_reason,
    "group": lambda s: s.burst_group,
    "manual": lambda s: s.is_manual,
}


class CullApp:
    """Main application window."""

    def __init__(self, root: ctk.CTk, settings_path: Path | None = None):
        self.root = root
        self.settings_path = settings_path
        self.settings = load_settings(settings_path)
        self.msg_queue: "queue.Queue[tuple]" = queue.Queue()
        self.preview_queue: "queue.Queue[tuple]" = queue.Queue()
        self.worker: CullWorker | None = None
        self._scores: list[ImageScore] = []
        self._stream_scores: list[ImageScore] = []  # rows shown during scoring
        self._row_order: list[int] = []
        self._sort_col: str | None = None
        self._sort_rev = False
        self._frames_done = 0
        self._total_frames = 0
        self._keep_count = 0
        self._reject_count = 0
        self._name_to_path: dict[str, str] = {}
        self._preview_ctk = None        # keep a reference so Tk does not GC it
        self._preview_target: ImageScore | None = None
        self._preview_busy = False
        self._preview_pending: ImageScore | None = None

        self._build_ui()
        self._apply_settings()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_id = self.root.after(POLL_MS, self._poll_queue)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        self.root.title("Auto-Culling — F1 连拍选片")
        self.root.geometry("1420x900")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(4, weight=1)

        outer = ctk.CTkFrame(self.root, corner_radius=0)
        outer.grid(row=0, column=0, sticky="nsew", padx=10, pady=8)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(4, weight=1)

        # --- directory row -------------------------------------------------
        dir_row = ctk.CTkFrame(outer, corner_radius=8)
        dir_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        dir_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(dir_row, text="照片目录:").grid(row=0, column=0, padx=(12, 6), pady=8)
        self.dir_var = ctk.StringVar()
        ctk.CTkEntry(dir_row, textvariable=self.dir_var).grid(
            row=0, column=1, sticky="ew", padx=6, pady=8)
        ctk.CTkButton(dir_row, text="选择目录…", width=100,
                      command=self._choose_dir).grid(row=0, column=2, padx=(6, 12), pady=8)

        # --- parameter panels ---------------------------------------------
        params = ctk.CTkFrame(outer, corner_radius=8)
        params.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        params.grid_columnconfigure(0, weight=1)
        params.grid_columnconfigure(1, weight=1)
        basic_frame = ctk.CTkFrame(params, corner_radius=8, fg_color="transparent")
        basic_frame.grid(row=0, column=0, sticky="ew", padx=(12, 6), pady=8)
        self._build_basic_params(basic_frame)

        adv_frame = ctk.CTkFrame(params, corner_radius=8, fg_color="transparent")
        adv_frame.grid(row=0, column=1, sticky="ew", padx=(6, 12), pady=8)
        self._build_advanced_params(adv_frame)

        # --- control row ---------------------------------------------------
        controls = ctk.CTkFrame(outer, corner_radius=8)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.btn_start = ctk.CTkButton(controls, text="开始选片", width=120, command=self._start)
        self.btn_start.pack(side="left", padx=(12, 6), pady=6)
        self.btn_stop = ctk.CTkButton(controls, text="停止", width=90, command=self._stop,
                                      state="disabled")
        self.btn_stop.pack(side="left", padx=6, pady=6)
        self.btn_export = ctk.CTkButton(controls, text="导出 CSV…", width=110,
                                        command=self._export_csv, state="disabled")
        self.btn_export.pack(side="left", padx=6, pady=6)

        # --- progress panel ------------------------------------------------
        prog = ctk.CTkFrame(outer, corner_radius=8)
        prog.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        prog.grid_columnconfigure(0, weight=1)
        self.progress = ctk.CTkProgressBar(prog, height=18)
        self.progress.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 2))
        self.progress.set(0)
        self.stage_var = ctk.StringVar(value="就绪")
        ctk.CTkLabel(prog, textvariable=self.stage_var, anchor="w").grid(
            row=1, column=0, sticky="ew", padx=12, pady=(0, 2))
        self.frame_stat_var = ctk.StringVar(value="")
        ctk.CTkLabel(prog, textvariable=self.frame_stat_var, anchor="e").grid(
            row=1, column=1, sticky="ew", padx=12, pady=(0, 2))

        # --- results + preview ---------------------------------------------
        body = ctk.CTkFrame(outer, corner_radius=8)
        body.grid(row=4, column=0, sticky="nsew", pady=(0, 8))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=0)   # preview pane: fixed width
        body.grid_rowconfigure(1, weight=1)

        filter_row = ctk.CTkFrame(body, fg_color="transparent")
        filter_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 4))
        ctk.CTkLabel(filter_row, text="筛选:").pack(side="left")
        self.filter_var = ctk.StringVar(value=FILTER_ALL)
        filter_box = ctk.CTkComboBox(filter_row, variable=self.filter_var, width=110,
                                     state="readonly",
                                     values=[FILTER_ALL, FILTER_KEEP, FILTER_DISCARD])
        filter_box.pack(side="left", padx=8)
        filter_box.configure(command=lambda _c: self._rebuild_rows())

        self.tree = ttk.Treeview(body, columns=[c[0] for c in _COLUMNS], show="headings",
                                 selectmode="browse")
        for col_id, title, width, anchor in _COLUMNS:
            self.tree.heading(col_id, text=title,
                              command=lambda c=col_id: self._sort_by(c))
            self.tree.column(col_id, width=width, anchor=anchor)
        vsb = ttk.Scrollbar(body, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=1, column=0, sticky="nsew", padx=(12, 4), pady=(0, 12))
        vsb.grid(row=1, column=0, sticky="nse", padx=(0, 8), pady=(0, 12))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Fixed-size preview pane: grid_propagate(False) pins its geometry, so
        # the "loading" text and the photo render inside the same rectangle and
        # the surrounding layout never shifts.
        self.preview_pane = ctk.CTkFrame(body, corner_radius=8, width=PREVIEW_W,
                                         fg_color=("gray90", "gray17"))
        self.preview_pane.grid(row=0, column=1, rowspan=2, sticky="ns", padx=(8, 12), pady=8)
        self.preview_pane.grid_propagate(False)
        self.preview_label = ctk.CTkLabel(self.preview_pane, text="选中结果行查看预览",
                                          anchor="center", corner_radius=8,
                                          fg_color="transparent")
        self.preview_label.pack(fill="both", expand=True, padx=4, pady=4)

        # ttk.Treeview default row height (20px) is too short under Windows DPI
        # scaling (125%/150%), which clips text and makes rows look overlapped.
        style = ttk.Style(self.root)
        try:
            scale = float(self.root.tk.call("tk", "scaling"))
        except Exception:
            scale = 1.0
        row_h = max(26, int(round(22 * scale)))
        style.configure("Treeview", rowheight=row_h, font=("Microsoft YaHei UI", 10))
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 10, "bold"))

        # --- log panel -----------------------------------------------------
        log_frame = ctk.CTkFrame(outer, corner_radius=8)
        log_frame.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        log_frame.grid_columnconfigure(0, weight=1)
        self._log_text = ctk.CTkTextbox(log_frame, height=140, wrap="none")
        self._log_text.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 8))
        self._log_text.configure(state="disabled")

        # --- status bar ----------------------------------------------------
        self.status_var = ctk.StringVar(value="就绪")
        ctk.CTkLabel(outer, textvariable=self.status_var, anchor="w", corner_radius=6,
                     fg_color=("gray88", "gray22")).grid(row=6, column=0, sticky="ew", pady=(0, 4))

    def _build_basic_params(self, frame: ctk.CTkFrame) -> None:
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text="基本参数", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self.var_recursive = ctk.BooleanVar()
        self.var_force = ctk.BooleanVar()
        self.var_top_n = ctk.StringVar()
        self.var_p4 = ctk.StringVar()
        self.var_scale_width = ctk.StringVar()
        self.var_workers = ctk.StringVar()

        ctk.CTkCheckBox(frame, text="递归扫描子目录", variable=self.var_recursive).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=1)
        ctk.CTkCheckBox(frame, text="强制重新分析（忽略已有评分）", variable=self.var_force).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=1)
        ctk.CTkLabel(frame, text="每连拍保留数 (Top-N):").grid(row=3, column=0, sticky="w", pady=1)
        ctk.CTkEntry(frame, textvariable=self.var_top_n, width=70).grid(
            row=3, column=1, sticky="w", pady=1)
        ctk.CTkLabel(frame, text="P4 朝向策略:").grid(row=4, column=0, sticky="w", pady=1)
        ctk.CTkComboBox(frame, variable=self.var_p4, state="readonly", width=90,
                        values=["always", "never", "auto"]).grid(row=4, column=1, sticky="w", pady=1)
        ctk.CTkLabel(frame, text="解码宽度:").grid(row=5, column=0, sticky="w", pady=1)
        ctk.CTkEntry(frame, textvariable=self.var_scale_width, width=70).grid(
            row=5, column=1, sticky="w", pady=1)
        ctk.CTkLabel(frame, text="并发线程:").grid(row=6, column=0, sticky="w", pady=1)
        ctk.CTkEntry(frame, textvariable=self.var_workers, width=70).grid(
            row=6, column=1, sticky="w", pady=1)

    def _build_advanced_params(self, frame: ctk.CTkFrame) -> None:
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text="高级参数", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self.var_sharp = ctk.StringVar()
        self.var_w_sharp = ctk.StringVar()
        self.var_w_comp = ctk.StringVar()
        self.var_min_raw = ctk.StringVar()
        self.var_conf = ctk.StringVar()
        self.var_autocrop = ctk.BooleanVar()
        self.var_rename = ctk.BooleanVar()
        self.var_dry_run = ctk.BooleanVar()
        self.var_rf_key = ctk.StringVar()

        self._float_entry(frame, 1, "锐度阈值:", self.var_sharp)
        self._float_entry(frame, 2, "锐度权重:", self.var_w_sharp)
        self._float_entry(frame, 3, "构图权重:", self.var_w_comp)
        self._float_entry(frame, 4, "最低总分:", self.var_min_raw)
        self._float_entry(frame, 5, "检测置信度:", self.var_conf)
        ctk.CTkCheckBox(frame, text="自动裁剪", variable=self.var_autocrop).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=1)
        ctk.CTkCheckBox(frame, text="重命名文件", variable=self.var_rename).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=1)
        ctk.CTkCheckBox(frame, text="试运行（不写任何文件）", variable=self.var_dry_run).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=1)
        ctk.CTkLabel(frame, text="Roboflow API Key:").grid(row=9, column=0, sticky="w", pady=1)
        ctk.CTkEntry(frame, textvariable=self.var_rf_key, width=140).grid(
            row=9, column=1, sticky="w", pady=1)

    def _float_entry(self, frame: ctk.CTkFrame, row: int, label: str, var: ctk.StringVar) -> None:
        ctk.CTkLabel(frame, text=label).grid(row=row, column=0, sticky="w", pady=1)
        ctk.CTkEntry(frame, textvariable=var, width=70).grid(row=row, column=1, sticky="w", pady=1)

    # ------------------------------------------------------------ settings

    def _apply_settings(self) -> None:
        s = self.settings
        self.dir_var.set(s.last_dir)
        self.var_recursive.set(s.recursive)
        self.var_force.set(s.force)
        self.var_top_n.set(str(s.top_n))
        self.var_p4.set(s.p4_policy)
        self.var_scale_width.set(str(s.scale_width))
        self.var_workers.set(str(s.workers))
        self.var_sharp.set(f"{s.sharp_thresh:g}")
        self.var_w_sharp.set(f"{s.w_sharp:g}")
        self.var_w_comp.set(f"{s.w_comp:g}")
        self.var_min_raw.set(f"{s.min_raw:g}")
        self.var_conf.set(f"{s.conf:g}")
        self.var_autocrop.set(s.autocrop)
        self.var_rename.set(s.rename)
        self.var_dry_run.set(s.dry_run)
        self.var_rf_key.set(s.rf_api_key)

    def _collect_settings(self) -> AppSettings:
        s = self.settings
        s.last_dir = self.dir_var.get().strip()
        try:
            s.recursive = self.var_recursive.get()
            s.force = self.var_force.get()
            s.top_n = int(self.var_top_n.get())
            s.p4_policy = self.var_p4.get()
            s.scale_width = int(self.var_scale_width.get())
            s.workers = int(self.var_workers.get())
            s.sharp_thresh = float(self.var_sharp.get())
            s.w_sharp = float(self.var_w_sharp.get())
            s.w_comp = float(self.var_w_comp.get())
            s.min_raw = float(self.var_min_raw.get())
            s.conf = float(self.var_conf.get())
            s.autocrop = self.var_autocrop.get()
            s.rename = self.var_rename.get()
            s.dry_run = self.var_dry_run.get()
            s.rf_api_key = self.var_rf_key.get().strip()
        except Exception:
            messagebox.showerror("参数错误", "数值参数格式不正确，请检查后重试。")
            raise
        return s

    # ------------------------------------------------------------- actions

    def _choose_dir(self) -> None:
        initial = self.dir_var.get().strip() or self.settings.last_dir
        chosen = filedialog.askdirectory(initialdir=initial or None, title="选择照片目录")
        if chosen:
            self.dir_var.set(chosen)

    def _start(self) -> None:
        try:
            settings = self._collect_settings()
        except Exception:
            return
        input_dir = Path(settings.last_dir)
        if not input_dir.is_dir():
            messagebox.showerror("错误", "请选择有效的照片目录。")
            return

        config = EngineConfig(
            input_dir=input_dir,
            recursive=settings.recursive,
            top_n=settings.top_n,
            p4_policy=settings.p4_policy,
            scale_width=settings.scale_width,
            workers=settings.workers,
            force=settings.force,
            sharp_thresh=settings.sharp_thresh,
            w_sharp=settings.w_sharp,
            w_comp=settings.w_comp,
            min_raw=settings.min_raw,
            conf=settings.conf,
            autocrop=settings.autocrop,
            rename=settings.rename,
            dry_run=settings.dry_run,
            rf_api_key=settings.rf_api_key or None,
        )

        # Drain any stale messages from a previous run.
        for q in (self.msg_queue, self.preview_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        self._reset_run_state()
        self.worker = CullWorker(config, self.msg_queue)
        self._set_running(True)
        self.stage_var.set("启动中…")
        self.worker.start()

    def _stop(self) -> None:
        if self.worker and self.worker.is_running():
            self.worker.stop()
            self.stage_var.set("正在停止…")

    def _export_csv(self) -> None:
        engine = self.worker.engine if self.worker else None
        if engine is None or not getattr(engine, "all_scores", None):
            messagebox.showinfo("提示", "请先完成一次选片任务。")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV 文件", "*.csv")],
            initialfile="cull_scores.csv")
        if not path:
            return
        try:
            engine.export_scores_csv(Path(path))
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        messagebox.showinfo("完成", f"已导出到:\n{path}")

    # ------------------------------------------------------- worker events

    def _poll_queue(self) -> None:
        # Bounded drain: cap the messages handled per tick so preview updates
        # still get their slot during heavy scoring (see _MAX_MSGS_PER_POLL).
        processed = 0
        while processed < _MAX_MSGS_PER_POLL:
            try:
                msg = self.msg_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1
            kind = msg[0]
            if kind == STAGE:
                self._on_stage(msg[1], msg[2])
            elif kind == LOG_LINE:
                self._append_log(msg[1])
            elif kind == TOTAL:
                self._total_frames = msg[1]
                self._update_frame_stat()
            elif kind == PATHS:
                self._name_to_path = dict(msg[1])
            elif kind == FRAME:
                self._on_frame(*msg[1:])
            elif kind == DONE:
                self._on_done(msg[1])
            elif kind == CANCELLED:
                self._on_cancelled(msg[1])
            elif kind == ERROR:
                self._on_error(msg[1])
        self._poll_previews()
        self._poll_id = self.root.after(POLL_MS, self._poll_queue)

    def _on_stage(self, msg: str, _progress: float) -> None:
        self.stage_var.set(msg)
        self.progress.set(_STAGE_SCALE.get(msg, 0.0))
        if msg == "Saving metadata...":
            self._update_frame_stat()

    def _on_frame(self, name: str, rating: int, sharp: float, comp: float,
                  raw: float, veto: str) -> None:
        self._frames_done += 1
        if rating > 0:
            self._keep_count += 1
        else:
            self._reject_count += 1
        self._append_stream_row(name, rating, sharp, comp, raw, veto)
        self._update_frame_stat()

    def _on_done(self, result: dict) -> None:
        self._scores = result["scores"]
        self._stream_scores = []
        self._rebuild_rows()
        elapsed = result["elapsed"]
        total = result["total"]
        speed = total / elapsed if elapsed > 0 and total else 0.0
        stars = result.get("stars", {})
        dist = "  ".join(f"{n}★×{stars.get(n, 0)}" for n in range(1, 6))
        self.status_var.set(
            f"完成: 共 {total} 张 | 保留 {result.get('keep', 0)} | "
            f"丢弃 {result.get('reject', 0)} | 耗时 {elapsed:.1f}s ({speed:.1f} 张/秒) | {dist}")
        self.progress.set(1.0)
        self.stage_var.set("100%  完成")
        self._set_running(False)
        self.btn_export.configure(state="normal")

    def _on_cancelled(self, scores: list) -> None:
        self._scores = scores
        self._stream_scores = []
        self._rebuild_rows()
        self.status_var.set(f"已取消 — 保留已打分 {len(scores)} 张的结果（未写入任何文件）")
        self.stage_var.set("已取消")
        self._set_running(False)

    def _on_error(self, message: str) -> None:
        self.stage_var.set("出错")
        self._set_running(False)
        messagebox.showerror("选片失败", message)

    def _reset_run_state(self) -> None:
        self._scores = []
        self._stream_scores = []
        self._row_order = []
        self._frames_done = 0
        self._total_frames = 0
        self._keep_count = 0
        self._reject_count = 0
        self._name_to_path = {}
        self._preview_target = None
        self._preview_ctk = None
        self._preview_pending = None
        self._preview_busy = False
        self.preview_label.configure(image="", text="选中结果行查看预览")
        self.tree.delete(*self.tree.get_children())
        self.btn_export.configure(state="disabled")
        self.progress.set(0)

    def _set_running(self, running: bool) -> None:
        self.btn_start.configure(state="disabled" if running else "normal")
        self.btn_stop.configure(state="normal" if running else "disabled")

    # ------------------------------------------------------------ results

    def _append_stream_row(self, name: str, rating: int, sharp: float, comp: float,
                           raw: float, veto: str) -> None:
        """Insert a row as soon as the frame is scored (no wait for DONE)."""
        if self.filter_var.get() != FILTER_ALL or self._sort_col is not None:
            return  # ordering would be wrong; rebuilt wholesale at DONE
        path = Path(self._name_to_path.get(name, "") or name)
        score = ImageScore(path=path, s_sharp=sharp, s_comp=comp, raw_score=raw,
                           rating=rating, veto_reason=veto)
        self._stream_scores.append(score)
        # Numeric iid ("0", "1", ...) — same scheme as _rebuild_rows — so that
        # _selected_score() can int() it during the streaming phase. Auto iids
        # ("I001") crash row selection while a run is still active.
        self.tree.insert("", "end", iid=str(len(self._stream_scores) - 1), values=(
            str(rating), name, f"{raw:.2f}", f"{sharp:.3f}", f"{comp:.3f}",
            veto, "", "",
        ))

    def _rebuild_rows(self) -> None:
        """Re-apply filter + sort and repopulate the tree."""
        self.tree.delete(*self.tree.get_children())
        order = list(range(len(self._scores)))
        f = self.filter_var.get()
        if f == FILTER_KEEP:
            order = [i for i in order if self._scores[i].rating > 0]
        elif f == FILTER_DISCARD:
            order = [i for i in order if self._scores[i].rating <= 0]
        if self._sort_col in _SORT_KEYS:
            key = _SORT_KEYS[self._sort_col]
            order.sort(key=lambda i: key(self._scores[i]), reverse=self._sort_rev)
        self._row_order = order
        for pos, idx in enumerate(order):
            s = self._scores[idx]
            self.tree.insert("", "end", iid=str(pos), values=(
                str(s.rating),
                s.path.name,
                f"{s.raw_score:.2f}",
                f"{s.s_sharp:.3f}",
                f"{s.s_comp:.3f}",
                s.veto_reason or "",
                str(s.burst_group),
                "是" if s.is_manual else "",
            ))

    def _sort_by(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col, self._sort_rev = col, False
        if self._scores:
            self._rebuild_rows()

    def _on_select(self, _event=None) -> None:
        score = self._selected_score()
        if score is None:
            return
        self._preview_target = score
        self.preview_label.configure(image="", text=f"加载中: {score.path.name} …")
        if not self._preview_busy:
            self._launch_preview(score)
        else:
            # A decode is already running; queue this selection so it loads
            # as soon as the current one finishes.
            self._preview_pending = score

    def _launch_preview(self, score: ImageScore) -> None:
        self._preview_busy = True
        target = score
        max_size = self._preview_max_size()

        def work():
            try:
                pil = render_pil(target, max_size=max_size)
            except Exception:
                pil = None
            self.preview_queue.put((target.path, pil))

        threading.Thread(target=work, daemon=True, name="preview").start()

    def _preview_max_size(self) -> int:
        """Longest-side limit so the rendered photo fits the fixed pane."""
        w = self.preview_pane.winfo_width()
        h = self.preview_pane.winfo_height()
        if w <= 1 or h <= 1:
            return 480
        return max(256, min(w, h) - 12)

    def _selected_score(self) -> ImageScore | None:
        selection = self.tree.selection()
        if not selection:
            return None
        iid = int(selection[0])
        if self._row_order:
            return self._scores[self._row_order[iid]]
        if iid < len(self._stream_scores):     # streaming phase: no full scores yet
            return self._stream_scores[iid]
        return None

    def _poll_previews(self) -> None:
        try:
            while True:
                path, pil = self.preview_queue.get_nowait()
                target = self._preview_target
                if pil is None:
                    if target is not None and path == target.path:
                        self.preview_label.configure(image="", text=f"无法预览: {path.name}")
                elif target is not None and path == target.path:
                    # CTkImage (not raw PhotoImage) so the preview scales
                    # correctly on HighDPI displays.
                    self._preview_ctk = ctk.CTkImage(light_image=pil, dark_image=pil,
                                                     size=pil.size)
                    self.preview_label.configure(image=self._preview_ctk, text="")
                self._preview_busy = False
        except queue.Empty:
            pass
        # Load the most recent selection that arrived while a decode was busy.
        if not self._preview_busy and self._preview_pending is not None:
            pending, self._preview_pending = self._preview_pending, None
            if pending is self._preview_target:
                self._launch_preview(pending)

    # --------------------------------------------------------------- misc

    def _update_frame_stat(self) -> None:
        if self._total_frames:
            self.frame_stat_var.set(
                f"已打分 {self._frames_done}/{self._total_frames} · "
                f"保留 {self._keep_count} · 丢弃 {self._reject_count}")
            frac = min(1.0, self._frames_done / self._total_frames)
            self.progress.set(_SCORE_START + (_SCORE_END - _SCORE_START) * frac)
        else:
            self.frame_stat_var.set(
                f"已打分 {self._frames_done} · 保留 {self._keep_count} · 丢弃 {self._reject_count}")

    def _append_log(self, line: str) -> None:
        self._log_text.configure(state="normal")
        self._log_text.insert("end", line + "\n")
        line_count = int(self._log_text.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            self._log_text.delete("1.0", f"{line_count - MAX_LOG_LINES + 100}.0")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def close(self) -> None:
        """Cancel the poll loop and destroy the window (no confirmation)."""
        try:
            self.root.after_cancel(self._poll_id)
        except Exception:
            pass
        self.root.destroy()

    def _on_close(self) -> None:
        if self.worker and self.worker.is_running():
            if not messagebox.askyesno("退出", "任务正在运行，取消并退出？"):
                return
            self.worker.stop()
            self.worker.wait(timeout=5)
        try:
            self._collect_settings()
        except Exception:
            pass
        save_settings(self.settings, self.settings_path)
        self.close()