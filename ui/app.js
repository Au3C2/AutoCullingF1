/* Auto-Culling Tauri UI — frontend logic (vanilla JS, window.__TAURI__ core API) */

"use strict";

const T = window.__TAURI__;
const invoke = T.core.invoke;
// NOTE: with the global bundle, `event` is a TOP-LEVEL module
// (window.__TAURI__.event.listen), not under core.
const listen = T.event.listen;

/* ---------- state ---------- */

const state = {
  running: false,
  scannedDir: null,
  paths: {},                 // name -> abs path
  rows: [],                  // {name, rating, sharp, comp, raw, veto, status, el}
  scored: 0,
  totalFrames: 0,
  keep: 0,
  reject: 0,
  filter: "all",
  sortKey: null,
  sortRev: false,
  selectedPath: null,
  previewBusy: false,
  previewPending: null,
};

const $ = (id) => document.getElementById(id);

const STAGE_SCALE = {
  "Collecting images...": 0.05,
  "Renaming images...": 0.10,
  "Reading EXIF metadata...": 0.15,
  "Grouping burst sequences...": 0.20,
  "Loading models...": 0.30,
  "Analyzing images...": 0.35,
  "Saving metadata...": 0.96,
  "Done!": 1.0,
  "Cancelled": 0.0,
};
const SCORE_START = 0.35;
const SCORE_END = 0.96;
const MAX_LOG_LINES = 2000;

/* ---------- param persistence (localStorage) ---------- */

const PARAM_IDS = ["dir", "pRecursive", "pForce", "pTopN", "pP4", "pScale",
                   "pWorkers", "pSharp", "pWSharp", "pWComp", "pMinRaw",
                   "pConf", "pAutocrop", "pRename", "pDryRun", "pRfKey"];

function loadSettings() {
  try {
    const raw = localStorage.getItem("ac-settings");
    if (!raw) return;
    const s = JSON.parse(raw);
    for (const id of PARAM_IDS) {
      const el = $(id);
      if (el && id in s) {
        if (el.type === "checkbox") el.checked = !!s[id];
        else el.value = String(s[id]);
      }
    }
  } catch (_) { /* corrupt settings: ignore */ }
}

function saveSettings() {
  const s = {};
  for (const id of PARAM_IDS) {
    const el = $(id);
    if (el) s[id] = el.type === "checkbox" ? el.checked : el.value;
  }
  try { localStorage.setItem("ac-settings", JSON.stringify(s)); } catch (_) {}
}

/* ---------- run config ---------- */

function collectConfig() {
  const num = (id) => Number($(id).value);
  return {
    inputDir: $("dir").value.trim(),
    recursive: $("pRecursive").checked,
    topN: num("pTopN"),
    p4Policy: $("pP4").value,
    scaleWidth: num("pScale"),
    workers: num("pWorkers"),
    force: $("pForce").checked,
    sharpThresh: num("pSharp"),
    wSharp: num("pWSharp"),
    wComp: num("pWComp"),
    minRaw: num("pMinRaw"),
    conf: num("pConf"),
    autocrop: $("pAutocrop").checked,
    rename: $("pRename").checked,
    dryRun: $("pDryRun").checked,
    rfApiKey: $("pRfKey").value.trim() || null,
  };
}

/* ---------- controls ---------- */

function setRunning(running) {
  state.running = running;
  $("start").disabled = running;
  $("stop").disabled = !running;
  if (running) $("export").disabled = true;
}

function toggleParams(force) {
  const sec = $("params");
  const show = force !== undefined ? force : sec.classList.contains("collapsed");
  sec.classList.toggle("collapsed", !show);
  $("toggleParamsBtn").textContent = show ? "⚙ 收起设置" : "⚙ 筛片设置";
  $("toggleParams").textContent = show ? "收起设置 ⌃" : "展开设置 ⌄";
}

async function scanDir(dir) {
  $("stage").textContent = "扫描目录…";
  try {
    await invoke("scan_directory", {
      dir,
      recursive: $("pRecursive").checked,
    });
  } catch (err) {
    setStatus("扫描失败: " + err);
  }
}

async function startRun() {
  const cfg = collectConfig();
  if (!cfg.inputDir) {
    setStatus("请先选择照片目录");
    return;
  }
  saveSettings();
  resetRun();
  setRunning(true);
  $("stage").textContent = "启动中…";
  try {
    await invoke("start_run", { config: cfg });
  } catch (err) {
    setRunning(false);
    setStatus("启动失败: " + err);
  }
}

async function stopRun() {
  try { await invoke("stop_run"); } catch (_) {}
  $("stage").textContent = "正在停止…";
}

async function exportCsv() {
  try {
    const path = await invoke("export_csv");
    setStatus("已导出: " + path);
  } catch (err) {
    setStatus("导出失败: " + err);
  }
}

async function pickDir() {
  try {
    const dir = await invoke("pick_directory");
    if (dir) {
      $("dir").value = dir;
      saveSettings();
      await scanDir(dir);
    }
  } catch (err) {
    setStatus("选择目录失败: " + err);
  }
}

/* ---------- reset / rows ---------- */

function resetRun() {
  for (const r of state.rows) {
    r.rating = 0; r.sharp = 0; r.comp = 0; r.raw = 0;
    r.veto = ""; r.status = "pending";
  }
  state.scored = 0;
  state.keep = 0;
  state.reject = 0;
  state.selectedPath = null;
  state.previewPending = null;
  clearPreview();
  rebuildRows();
  $("bar").style.width = "0%";
  $("frameStat").textContent = "";
  $("export").disabled = true;
  $("status").textContent = "就绪";
}

function starsHtml(rating) {
  if (rating <= 0) return `<span class="stars zero">✕</span>`;
  return `<span class="stars">${"★".repeat(rating)}</span>`;
}

function rowHtml(r, idx) {
  const cls = r.status === "pending" ? "pending" : (r.rating > 0 ? "keep" : "reject");
  const score = r.status === "pending" ? "—" : r.raw.toFixed(2);
  const sharp = r.status === "pending" ? "—" : r.sharp.toFixed(3);
  const comp = r.status === "pending" ? "—" : r.comp.toFixed(3);
  const stars = r.status === "pending" ? `<span class="stars zero">…</span>` : starsHtml(r.rating);
  return `<tr data-idx="${idx}" class="${r.status === "pending" ? "pending" : ""}${r.flash ? " flash" : ""}">
    <td>${stars}</td>
    <td class="name" title="${escapeHtml(r.name)}">${escapeHtml(r.name)}</td>
    <td class="num ${cls}">${score}</td>
    <td class="num">${sharp}</td>
    <td class="num">${comp}</td>
    <td class="veto">${escapeHtml(r.veto || "")}</td>
  </tr>`;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function visibleRows() {
  let list = state.rows;
  if (state.filter === "keep") list = list.filter((r) => r.status !== "pending" && r.rating > 0);
  if (state.filter === "discard") list = list.filter((r) => r.status !== "pending" && r.rating <= 0);
  if (state.filter === "pending") list = list.filter((r) => r.status === "pending");
  if (state.sortKey) {
    const k = state.sortKey;
    // Pending rows always sort last so scored results stay visible.
    list = [...list].sort((a, b) => {
      if ((a.status === "pending") !== (b.status === "pending")) {
        return a.status === "pending" ? 1 : -1;
      }
      const va = a[k], vb = b[k];
      let cmp;
      if (typeof va === "string") cmp = va.localeCompare(vb);
      else cmp = va - vb;
      return state.sortRev ? -cmp : cmp;
    });
  }
  return list;
}

function rebuildRows() {
  const list = visibleRows();
  $("rows").innerHTML = list.map((r) => rowHtml(r, state.rows.indexOf(r))).join("");
  const scored = state.rows.filter((r) => r.status !== "pending").length;
  $("count").textContent = state.rows.length
    ? `${scored} / ${state.rows.length} 张`
    : "";
}

function fillRow(r) {
  // Update the row's DOM in place (works with any filter/sort because the
  // element identity comes from the row object itself).
  if (!r.el || !r.el.isConnected) {
    rebuildRows();
    return;
  }
  const tmp = document.createElement("tbody");
  tmp.innerHTML = rowHtml(r, 0);
  const fresh = tmp.firstElementChild;
  r.el.replaceWith(fresh);
  r.el = fresh;
  fresh.classList.add("flash");
  fresh.addEventListener("animationend", () => fresh.classList.remove("flash"), { once: true });
  const scored = state.rows.filter((x) => x.status !== "pending").length;
  $("count").textContent = `${scored} / ${state.rows.length} 张`;
}

function applyFrame(e) {
  const r = state.rows.find((x) => x.name === e.name);
  if (!r) return;
  r.rating = e.rating;
  r.sharp = e.sharp;
  r.comp = e.comp;
  r.raw = e.raw;
  r.veto = e.veto || "";
  r.status = e.status || "scored";
  if (r.rating > 0) state.keep++; else state.reject++;
  state.scored++;
  fillRow(r);
}

function sortBy(key) {
  if (state.sortKey === key) state.sortRev = !state.sortRev;
  else { state.sortKey = key; state.sortRev = false; }
  document.querySelectorAll("#grid th").forEach((th) =>
    th.classList.toggle("sorted", th.dataset.k === key));
  if (state.rows.length) rebuildRows();
}

/* ---------- progress ---------- */

function setProgress(frac) {
  $("bar").style.width = `${Math.round(frac * 100)}%`;
}

function updateFrameStat() {
  const t = state.totalFrames || state.rows.length || 0;
  $("frameStat").textContent =
    `已打分 ${state.scored}/${t} · 保留 ${state.keep} · 丢弃 ${state.reject}`;
  if (t > 0) {
    const frac = Math.min(1, state.scored / t);
    setProgress(SCORE_START + (SCORE_END - SCORE_START) * frac);
  }
}

/* ---------- preview ---------- */

function clearPreview() {
  state.selectedPath = null;
  $("previewImg").style.display = "none";
  $("previewImg").removeAttribute("src");
  const ov = $("previewOverlay");
  ov.textContent = "选中结果行查看预览";
  ov.style.display = "flex";
  $("previewPane").classList.remove("loading");
}

async function requestPreview(path) {
  if (state.previewBusy) { state.previewPending = path; return; }
  state.previewBusy = true;
  const pane = $("previewPane");
  const size = Math.max(256, Math.min(pane.clientWidth, pane.clientHeight) - 16);
  try {
    // The Rust command returns the base64 PNG as a bare string.
    const res = await invoke("preview", { path, size });
    if (path !== state.selectedPath) return; // selection changed meanwhile
    if (res) {
      $("previewImg").src = "data:image/png;base64," + res;
      $("previewImg").style.display = "block";
      $("previewOverlay").style.display = "none";
    } else {
      const ov = $("previewOverlay");
      ov.textContent = "无法预览: " + path.split(/[\\/]/).pop();
      ov.style.display = "flex";
    }
  } catch (err) {
    if (path === state.selectedPath) {
      $("previewOverlay").textContent = "无法预览: " + path.split(/[\\/]/).pop();
      $("previewOverlay").style.display = "flex";
    }
  } finally {
    state.previewBusy = false;
    if (state.previewPending) {
      const next = state.previewPending;
      state.previewPending = null;
      if (next === state.selectedPath) requestPreview(next);
    }
  }
}

function selectRow(name) {
  if (!name) return;
  // Track the ABSOLUTE path: requestPreview compares against it, and the
  // paths map resolves the name -> abs path for the sidecar request.
  const abs = state.paths[name];
  if (!abs) return;
  state.selectedPath = abs;
  document.querySelectorAll("#rows tr.selected").forEach((tr) => tr.classList.remove("selected"));
  const row = state.rows.find((r) => r.name === name);
  if (row && row.el) row.el.classList.add("selected");
  const ov = $("previewOverlay");
  ov.textContent = "加载中: " + name + " …";
  ov.style.display = "flex";
  $("previewImg").style.display = "none";
  $("previewPane").classList.add("loading");
  requestPreview(abs);
}

/* ---------- events from the Rust shell ---------- */

async function onEvent(evt) {
  const e = evt.payload;
  switch (e.kind) {
    case "scanned": {
      state.scannedDir = e.dir;
      state.paths = e.paths || {};
      state.rows = Object.keys(state.paths).map((name) => ({
        name, rating: 0, sharp: 0, comp: 0, raw: 0, veto: "",
        status: "pending", el: null,
      }));
      state.totalFrames = state.rows.length;
      state.scored = 0;
      state.keep = 0;
      state.reject = 0;
      state.selectedPath = null;
      clearPreview();
      rebuildRows();
      $("stage").textContent = `已扫描 ${state.totalFrames} 张待筛`;
      $("frameStat").textContent = "";
      setStatus(`目录就绪：${state.totalFrames} 张待筛 — 可调整参数后开始选片`);
      break;
    }
    case "scan_error": {
      $("stage").textContent = "扫描失败";
      setStatus("扫描失败: " + e.message);
      break;
    }
    case "stage": {
      $("stage").textContent = e.msg;
      setProgress(STAGE_SCALE[e.msg] ?? 0);
      break;
    }
    case "group": {
      updateFrameStat();
      break;
    }
    case "frame": {
      applyFrame(e);
      updateFrameStat();
      break;
    }
    case "done": {
      setProgress(1);
      $("stage").textContent = "100%  完成";
      setRunning(false);
      $("export").disabled = false;
      const stars = e.stars || {};
      const dist = Object.keys(stars).sort()
        .map((n) => `${n}★×${stars[n]}`).join("  ");
      const ips = e.total > 0 && e.elapsed > 0 ? (e.total / e.elapsed).toFixed(1) : "0";
      setStatus(`完成: 共 ${e.total} 张 | 保留 ${e.keep} | 丢弃 ${e.reject} | 耗时 ${e.elapsed.toFixed(1)}s (${ips} 张/秒) | ${dist}`);
      break;
    }
    case "cancelled": {
      $("stage").textContent = "已取消";
      setRunning(false);
      setStatus(`已取消 — 保留已打分 ${e.count} 张的结果（未写入任何文件）`);
      break;
    }
    case "error": {
      $("stage").textContent = "出错";
      setRunning(false);
      setStatus("选片失败: " + e.message);
      break;
    }
    case "log": {
      appendLog(e.line);
      break;
    }
  }
}

/* ---------- log ---------- */

function appendLog(line) {
  const el = $("log");
  el.textContent += line + "\n";
  const lines = el.textContent.split("\n");
  if (lines.length > MAX_LOG_LINES) {
    el.textContent = lines.slice(lines.length - MAX_LOG_LINES).join("\n");
  }
  el.scrollTop = el.scrollHeight;
}

/* ---------- status ---------- */

function setStatus(msg) {
  $("status").textContent = msg;
}

/* ---------- resizable workspace splitter ---------- */

function initSplitter() {
  const resizer = $("resizer");
  const workspace = document.querySelector(".workspace");
  const tablePane = document.querySelector(".table-pane");
  if (!resizer || !workspace || !tablePane) return;

  // Restore saved ratio or default to 38%
  const savedRatio = localStorage.getItem("ac-table-ratio");
  if (savedRatio) {
    const r = parseFloat(savedRatio);
    if (!isNaN(r) && r >= 0.15 && r <= 0.85) {
      tablePane.style.setProperty("--table-width", (r * 100).toFixed(1) + "%");
    }
  }

  let dragging = false;
  let previewTimer = null;

  resizer.addEventListener("mousedown", (e) => {
    e.preventDefault();
    dragging = true;
    resizer.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const rect = workspace.getBoundingClientRect();
    const offsetX = e.clientX - rect.left;
    const totalW = rect.width;
    // Enforce min widths (260px table, 260px preview)
    const minTableW = 260;
    const minPreviewW = 260;
    const clampedX = Math.max(minTableW, Math.min(totalW - minPreviewW, offsetX));
    const ratio = clampedX / totalW;
    tablePane.style.setProperty("--table-width", (ratio * 100).toFixed(1) + "%");
  });

  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";

    // Save ratio
    const rect = workspace.getBoundingClientRect();
    const tableRect = tablePane.getBoundingClientRect();
    if (rect.width > 0) {
      const ratio = tableRect.width / rect.width;
      try { localStorage.setItem("ac-table-ratio", ratio.toFixed(4)); } catch (_) {}
    }

    // Refresh preview if we currently have an active selected photo
    if (state.selectedPath) {
      clearTimeout(previewTimer);
      previewTimer = setTimeout(() => {
        if (state.selectedPath) {
          state.previewBusy = false;
          requestPreview(state.selectedPath);
        }
      }, 150);
    }
  });
}

/* ---------- wiring ---------- */

function wire() {
  initSplitter();
  $("pickDir").addEventListener("click", pickDir);
  $("start").addEventListener("click", startRun);
  $("stop").addEventListener("click", stopRun);
  $("export").addEventListener("click", exportCsv);
  $("toggleParamsBtn").addEventListener("click", () => toggleParams());
  $("toggleParams").addEventListener("click", () => toggleParams());
  $("filter").addEventListener("change", (ev) => {
    state.filter = ev.target.value;
    rebuildRows();
  });
  document.querySelectorAll("#grid th").forEach((th) =>
    th.addEventListener("click", () => sortBy(th.dataset.k)));
  $("rows").addEventListener("click", (ev) => {
    const tr = ev.target.closest("tr");
    if (!tr) return;
    const r = state.rows[Number(tr.dataset.idx)];
    if (r) selectRow(r.name);
  });
  $("toggleLog").addEventListener("click", () => {
    const el = $("log");
    el.classList.toggle("hidden");
    $("toggleLog").textContent = el.classList.contains("hidden") ? "展开" : "收起";
  });
  $("dir").addEventListener("change", (ev) => {
    saveSettings();
    const dir = ev.target.value.trim();
    if (dir) scanDir(dir);
  });
  $("pRecursive").addEventListener("change", () => {
    saveSettings();
    if (state.scannedDir) scanDir(state.scannedDir);
  });
  window.addEventListener("beforeunload", saveSettings);
}

(async function main() {
  wire();
  loadSettings();
  toggleParams(false);
  await listen("evt", onEvent);
  await listen("run-status", (ev) => setRunning(ev.payload.running));
  const initDir = $("dir").value.trim();
  if (initDir) {
    scanDir(initDir);
  }
})();
