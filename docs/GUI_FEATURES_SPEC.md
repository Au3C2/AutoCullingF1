# GUI Features Technical Specification (Features 1 - 7)

This specification defines the contract, interfaces, user interactions, boundary cases, and testing criteria for features 1 to 7 in the Auto-Culling Tauri Desktop GUI.

---

## 1. Feature 1: Keyboard Navigation Flow & Culling Hotkeys

### 1.1 Specification & Keymap
- When focus is NOT inside an editable element (`<input>`, `<select>`, `<textarea>`):
  - `ArrowDown` / `j`: Move selection to next visible photo.
  - `ArrowUp` / `k`: Move selection to previous visible photo.
  - `Home`: Select first visible photo.
  - `End`: Select last visible photo.
  - `Space`: Toggle full-screen / expanded preview dialog or focus switch.
  - `1`, `2`, `3`, `4`, `5`: Set manual rating of currently selected photo to 1~5 stars.
  - `0` / `x`: Mark current photo as rejected (`rating = -1`, `veto = 'manual_reject'`).
- Hotkey Protection:
  - Any keypress originating from or bubbling through `input`, `textarea`, or `select` MUST be ignored.
  - Modifier combinations with `Ctrl`, `Meta`, `Alt` (except `Cmd/Ctrl+O` which is browse) must NOT trigger single-key culling.

### 1.2 Boundary Cases
- Empty list: Pressing arrows or number keys does nothing, does not throw error.
- Grouped view: Navigating down/up smoothly skips collapsed items and traverses only visible selectable rows.
- Rating change updates: Immediate UI reflection on row, preview badge, and `KEEP`/`REJECT` telemetry counts.

---

## 2. Feature 2: Collapsible & Compact Config Panel with Defaults Reset

### 2.1 UI & Behavior
- Toggle button `#btnToggleConfig` in `#app` top/panel header to collapse/expand `.tau-config-panel`.
- Collapsed state: `.tau-config-panel.collapsed`.
  - Body grid is hidden via CSS (`display: none` or transition).
  - A compact summary pill bar `#configSummaryBar` appears showing active highlights:
    `Top-N: {top_n} · Workers: {workers} · Sharp: {sharp} · YOLO: {conf}`.
- Reset button `#btnResetDefaults`:
  - Resets all form fields to their declared default values.
  - Clears `ac-param-*` keys from `localStorage` and restores secrets.
- Persistence:
  - Collapsed state saved to `localStorage.getItem('ac-config-collapsed') === 'true'`.

### 2.2 Boundary Cases
- Screen resizing: Compact summary must gracefully wrap or truncate without overflowing topbar.
- Resetting while running: Reset button disabled when `state.isRunning === true`.

---

## 3. Feature 3: Image Preview Pan & Zoom (Interactive Inspection)

### 3.1 Interaction Model
- Container `#previewContainer` with image `#previewImg`.
- State: `zoomLevel` (1.0 to 5.0, default 1.0), `panX`, `panY`, `isPanning`.
- Wheel event `wheel` (when pointer inside `#previewContainer`):
  - `deltaY < 0`: Zoom in (+0.25).
  - `deltaY > 0`: Zoom out (-0.25), minimum 1.0.
  - At 1.0 zoom: `panX = 0`, `panY = 0`.
- Mouse Drag (Pan):
  - `mousedown`: Start drag if `zoomLevel > 1.0`.
  - `mousemove`: Update `panX`, `panY` with clamped boundaries.
  - `mouseup` / `mouseleave`: End drag.
- Double Click:
  - If `zoomLevel > 1.0`: Reset to 1.0 (Fit to view).
  - If `zoomLevel === 1.0`: Zoom to 2.5x centered on click position.
- Selection Change:
  - Reset `zoomLevel = 1.0, panX = 0, panY = 0` automatically upon switching images.

### 3.2 Boundary Cases
- Image loading / empty state: Wheel and pan events ignored.
- Fast switching: Cancel in-flight gesture transforms.

---

## 4. Feature 4: High-Performance Virtual / Chunked Table Rendering & Anti-Race

### 4.1 Specification
- Table rendering strategy:
  - Instead of re-creating 5000 DOM nodes simultaneously, provide chunked / incremental rendering or virtual window rendering.
  - Initial slice: Render first 100 visible items immediately, and dynamically append/window on scroll (`requestAnimationFrame` or `scroll` throttling).
- Anti-Race in Preview & IPC:
  - Monotonically increasing `previewSeq` counter in frontend state.
  - Any preview result whose `seq < state.currentPreviewSeq` is silently discarded.

### 4.2 Boundary Cases
- Rapid clicking on 10 photos in 1 second: Only the latest preview is applied, no old previews overwrite the active selection.
- 10,000 photos imported: Initial render < 50ms, scrolling remains 60fps.

---

## 5. Feature 5: Drag & Drop Folder Import (HTML5)

### 5.1 UI & IPC
- Drag events on `#app`:
  - `dragover` / `dragenter`: Prevent default, add visual drag-hover class `.tau-drag-active`.
  - `dragleave` / `drop`: Remove `.tau-drag-active`.
- Tauri File Drop or Web DragDrop:
  - Tauri 2 provides `tauri://drag-drop` event or HTML5 `dataTransfer.files`.
  - Listen for `tauri://drag-drop` via `listenTauri('tauri://drag-drop')` and fallback to HTML5 drop.
  - Retrieve folder path, populate `#inputDir`, save `ac-last-dir`, enable `#btnRun`, and trigger `scan`.

### 5.2 Boundary Cases
- Dropping a single image file instead of folder: Detect parent directory or prompt user.
- Dropping invalid/non-existent paths: Handled by scan error telemetry without UI crash.

---

## 6. Feature 6: Burst Group Aggregated View & Single Shot Interleaving

### 6.1 Data Model & Grouping Algorithm
- Group detection:
  - Given sorted photo list by timestamp/filename:
  - If consecutive photos are within $\Delta t \le 1.0\text{s}$ (or share burst sequence ID), assign same `burstGroupId`.
  - If a group has $N \ge 2$: Classified as `burst_group`.
  - If a group has $N = 1$: Classified as `single_shot`.
- Render Item Structure:
  ```typescript
  interface BurstGroup {
    id: string;
    items: PhotoItem[];
    isExpanded: boolean;
    startTime: string;
    endTime: string;
    count: number;
    keepCount: number;
    rejectCount: number;
    winnerPath: string; // Highest raw score
  }
  ```
- View Mode:
  - `viewMode`: `'grouped'` (default) vs `'flat'`.
  - Switched via button `#btnViewMode` with icon/text.
- In Grouped View:
  - Burst Group Header: Shows `#01 BURST [N photos] (X Keep, Y Reject) [Mini Bar]`, clickable to toggle expand/collapse.
  - Single Shot: Displayed as normal row with badge `[SINGLE]`.
  - Winner highlight: Crown/gold badge on the top-ranked photo in the burst.
  - Top-N Veto indicator: Distinct label `[TOP-N]` for shots discarded due to group quota.

### 6.2 Boundary Cases
- 100% single shots (no bursts): Displays cleanly without redundant group headers.
- 100% bursts: All headers displayed; groups with 0 keeps default to collapsed; groups with keeps default to expanded.
- Filter (Keep / Reject): Groups dynamically show only matching photos; if 0 photos match filter, group header is hidden.

---

## 7. Feature 7: Context Menu & Local File Integration

### 7.1 UI & Commands
- Right-click (`contextmenu`) on any photo row:
  - Prevent default browser menu.
  - Show floating Cyber dark context menu `#contextMenu` at `(e.clientX, e.clientY)`.
  - Items:
    1. **在文件管理器中显示 (Show in Folder / Finder)**: Invokes Tauri command `show_in_folder` (or `open_path`).
    2. **复制文件路径 (Copy File Path)**: Copies `item.path` to clipboard.
    3. **复制文件名 (Copy File Name)**: Copies `item.name` to clipboard.
    4. **设置为保留 (Mark Keep ★)**: Quick rating toggle.
    5. **设置为淘汰 (Mark Reject ✕)**: Quick reject toggle.
- Click anywhere outside closes menu. `Esc` closes menu.

### 7.2 Boundary Cases
- Right-click near window edge: Menu automatically adjusts position so it does not overflow screen.
- In-flight operations: Safe clipboard access fallback.

---

## 8. Test Verification Criteria
All features must have automated unit/integration tests running under Node.js / Jest or Python harness in `tests/test_gui/`:
1. `test_gui_shortcuts.js` - Key navigation, filtering, hotkey ignores.
2. `test_gui_grouping.js` - Burst grouping algorithm, single shot interleaving, winner detection, filter interaction.
3. `test_gui_config.js` - Collapsible panel state, parameter reset defaults, persistence.
4. `test_gui_preview_zoom.js` - Pan & Zoom math, boundary clamping, image switch reset.
