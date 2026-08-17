//! auto-culling Tauri shell: spawns the Python CLI as a RESIDENT sidecar
//! speaking the JSON Lines protocol (cull/protocol.py). The sidecar answers
//! scan/run/cancel/preview commands until quit; the shell forwards its
//! events to the frontend and answers preview requests through it.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_shell::process::{Command, CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunConfig {
    input_dir: String,
    recursive: bool,
    top_n: i32,
    p4_policy: String,
    scale_width: i32,
    workers: i32,
    force: bool,
    sharp_thresh: f64,
    w_sharp: f64,
    w_comp: f64,
    min_raw: f64,
    conf: f64,
    autocrop: bool,
    rename: bool,
    dry_run: bool,
    rf_api_key: Option<String>,
}

impl RunConfig {
    /// Serialize into the sidecar `run` command's config override object.
    fn to_sidecar_config(&self) -> serde_json::Value {
        serde_json::json!({
            "recursive": self.recursive,
            "top_n": self.top_n,
            "p4_policy": self.p4_policy,
            "scale_width": self.scale_width,
            "workers": self.workers,
            "force": self.force,
            "sharp_thresh": self.sharp_thresh,
            "w_sharp": self.w_sharp,
            "w_comp": self.w_comp,
            "min_raw": self.min_raw,
            "conf": self.conf,
            "autocrop": self.autocrop,
            "rename": self.rename,
            "dry_run": self.dry_run,
            "rf_api_key": self.rf_api_key,
        })
    }
}

#[derive(Serialize, Clone, Default)]
struct ScoreRow {
    name: String,
    rating: i32,
    sharp: f64,
    comp: f64,
    raw: f64,
    veto: String,
}

/// The resident sidecar. `child` doubles as stdin: JSON commands
/// (run/cancel/preview/quit) are written through `CommandChild::write`.
struct Sidecar {
    gen: u64,
    child: Mutex<CommandChild>,
}

#[derive(Default)]
struct AppState {
    sidecar: Mutex<Option<Sidecar>>,
    scores: Mutex<Vec<ScoreRow>>,
    /// Pending preview requests keyed by path (sidecar responds asynchronously).
    preview_waiters: Mutex<Vec<(String, tokio::sync::oneshot::Sender<Option<String>>)>>,
    next_gen: std::sync::atomic::AtomicU64,
}

/// Append a timestamped line to gui.log next to the executable (debug aid —
/// the windowed app has no console).
fn log_line(msg: &str) {
    use std::io::Write;
    let path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("gui.log")));
    if let Some(p) = path {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let _ = writeln!(f, "[{secs}] {msg}");
        }
    }
}

/// Repo root for dev fallbacks: `tauri dev` runs the binary with CWD set to
/// `src-tauri/`, so step up one level when we detect the tauri config there.
fn repo_root() -> std::path::PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    if cwd.join("tauri.conf.json").exists() {
        cwd.parent().map(|p| p.to_path_buf()).unwrap_or(cwd)
    } else {
        cwd
    }
}

/// Command string for the sidecar. In packaged builds this is the bundled
/// `cull-sidecar` external binary; in debug builds (tauri dev) prefer the repo
/// venv's Python running cull_photos.py directly (the external binary is a
/// placeholder at best, and the venv script iterates without repackaging).
fn sidecar_command(app: &AppHandle) -> Result<Command, String> {
    #[cfg(debug_assertions)]
    {
        let root = repo_root();
        let script = root.join("cull_photos.py");
        // The venv layout differs by platform: .venv/Scripts/python.exe on
        // Windows, .venv/bin/python on macOS/Linux.
        let py = if cfg!(windows) {
            root.join(".venv/Scripts/python.exe")
        } else {
            root.join(".venv/bin/python")
        };
        if script.exists() && py.exists() {
            log_line("dev mode: using venv python sidecar");
            return Ok(app
                .shell()
                .command(py.to_string_lossy().as_ref())
                .args([script.to_string_lossy().as_ref(), "--json-lines"]));
        }
    }
    match app.shell().sidecar("cull-sidecar") {
        Ok(cmd) => Ok(cmd),
        Err(err) => {
            log_line(&format!("bundled sidecar unavailable ({err}); trying venv python"));
            let root = repo_root();
            let script = root.join("cull_photos.py");
            let py = if cfg!(windows) {
                root.join(".venv/Scripts/python.exe")
            } else {
                root.join(".venv/bin/python")
            };
            if script.exists() && py.exists() {
                Ok(app
                    .shell()
                    .command(py.to_string_lossy().as_ref())
                    .args([script.to_string_lossy().as_ref(), "--json-lines"]))
            } else {
                Err(format!("sidecar binary not found: {err}"))
            }
        }
    }
}

/// Spawn (or reuse) the resident sidecar and return its generation id.
/// The sidecar is started WITHOUT run arguments; a run is a `run` command.
fn ensure_sidecar(app: &AppHandle, state: &State<'_, AppState>) -> Result<u64, String> {
    let mut guard = state.sidecar.lock().unwrap();
    if let Some(existing) = guard.as_ref() {
        return Ok(existing.gen);
    }

    let sidecar = sidecar_command(app)?;
    // NOTE: Command::spawn already pipes stdin (os_pipe) — CommandChild::write
    // reaches the child. Keeping the CommandChild alive is what keeps the
    // sidecar's stdin open, which the preview command loop depends on.
    let (rx, child) = sidecar.spawn().map_err(|e| {
        log_line(&format!("spawn() error: {e}"));
        e.to_string()
    })?;
    let gen = state.next_gen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    log_line(&format!("sidecar spawned (gen {gen})"));
    *guard = Some(Sidecar {
        gen,
        child: Mutex::new(child),
    });
    drop(guard);

    let app2 = app.clone();
    std::thread::spawn(move || read_loop(app2, gen, rx));
    Ok(gen)
}

/// Write one JSON command line to the sidecar's stdin.
fn send_command(state: &State<'_, AppState>, payload: serde_json::Value) -> Result<(), String> {
    let guard = state.sidecar.lock().unwrap();
    let Some(sidecar) = guard.as_ref() else {
        return Err("engine is not running".into());
    };
    let mut line = payload.to_string();
    line.push('\n'); // the sidecar reads its stdin line by line
    let mut child = sidecar.child.lock().unwrap();
    child
        .write(line.as_bytes())
        .map_err(|e| format!("failed to write to sidecar: {e}"))
}

fn read_loop(app: AppHandle, gen: u64, mut rx: tokio::sync::mpsc::Receiver<CommandEvent>) {
    log_line(&format!("read_loop started (gen {gen})"));
    while let Some(event) = rx.blocking_recv() {
        match event {
            CommandEvent::Stdout(bytes) => {
                handle_stdout(&app, &bytes);
            }
            CommandEvent::Stderr(bytes) => {
                log_line(&format!(
                    "sidecar stderr: {}",
                    String::from_utf8_lossy(&bytes).trim()
                ));
            }
            CommandEvent::Terminated(_) => {
                log_line(&format!("sidecar terminated (gen {gen})"));
                // Only clear the state if the terminating process is still
                // the current one — a retired sidecar must not drop a newer
                // spawn.
                let state_ref = app.state::<AppState>();
                let mut guard = state_ref.sidecar.lock().unwrap();
                if let Some(s) = guard.as_ref() {
                    if s.gen == gen {
                        *guard = None;
                    }
                }
                drop(guard);
                let _ = app.emit("run-status", serde_json::json!({ "running": false }));
                break;
            }
            _ => {}
        }
    }
    log_line("read_loop ended");
}

/// Parse one stdout chunk from the sidecar into events and forward them.
fn handle_stdout(app: &AppHandle, bytes: &[u8]) {
    let text = String::from_utf8_lossy(bytes);
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            log_line(&format!("parse failed ({} bytes): {}", line.len(), line));
            continue;
        };
        let Some(kind) = value.get("type").and_then(|v| v.as_str()).map(str::to_string) else {
            continue;
        };

        if kind == "preview" {
            // Deliver to the awaiting command, do not forward.
            let path = value
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let png = value.get("png").and_then(|v| v.as_str()).map(str::to_string);
            let state_ref = app.state::<AppState>();
            let mut waiters = state_ref.preview_waiters.lock().unwrap();
            if let Some(pos) = waiters.iter().position(|(p, _)| *p == path) {
                let (_, tx) = waiters.remove(pos);
                let _ = tx.send(png);
            } else {
                log_line("preview waiter NOT found");
            }
            continue;
        }

        if kind == "frame" {
            let row = ScoreRow {
                name: value
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                rating: value.get("rating").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
                sharp: value.get("sharp").and_then(|v| v.as_f64()).unwrap_or(0.0),
                comp: value.get("comp").and_then(|v| v.as_f64()).unwrap_or(0.0),
                raw: value.get("raw").and_then(|v| v.as_f64()).unwrap_or(0.0),
                veto: value
                    .get("veto")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
            };
            app.state::<AppState>().scores.lock().unwrap().push(row);
        }

        // Lifecycle events drive the running flag.
        if kind == "done" || kind == "cancelled" || kind == "error" {
            let _ = app.emit("run-status", serde_json::json!({ "running": false }));
        }

        // Rename "type" -> "kind" for the frontend dispatch.
        let mut payload = value;
        if let Some(obj) = payload.as_object_mut() {
            obj.remove("type");
        }
        payload["kind"] = serde_json::Value::String(kind);
        let _ = app.emit("evt", payload);
    }
}

/// Kill the sidecar and clear the slot; a later command respawns it.
fn retire_sidecar(state: &State<'_, AppState>) {
    let mut guard = state.sidecar.lock().unwrap();
    if let Some(old) = guard.take() {
        log_line("retiring sidecar");
        {
            let mut old_child = old.child.lock().unwrap();
            let _ = old_child.write(b"{\"cmd\":\"quit\"}\n");
        }
        let child = old.child.into_inner().unwrap();
        let _ = child.kill();
    }
}

#[tauri::command]
fn start_run(app: AppHandle, state: State<'_, AppState>, config: RunConfig) -> Result<(), String> {
    log_line(&format!("start_run: dir={}", config.input_dir));
    state.scores.lock().unwrap().clear();

    ensure_sidecar(&app, &state)?;
    let payload = serde_json::json!({
        "cmd": "run",
        "dir": config.input_dir,
        "config": config.to_sidecar_config(),
    });
    send_command(&state, payload)?;
    let _ = app.emit("run-status", serde_json::json!({ "running": true }));
    log_line("run command sent");
    Ok(())
}

#[tauri::command]
fn stop_run(state: State<'_, AppState>) -> Result<(), String> {
    send_command(&state, serde_json::json!({"cmd": "cancel"}))
}

/// List the cullable shots of a directory without running the engine.
#[tauri::command]
fn scan_directory(
    app: AppHandle,
    state: State<'_, AppState>,
    dir: String,
    recursive: bool,
) -> Result<(), String> {
    log_line(&format!("scan_directory: dir={dir} recursive={recursive}"));
    ensure_sidecar(&app, &state)?;
    let payload = serde_json::json!({"cmd": "scan", "dir": dir, "recursive": recursive});
    send_command(&state, payload)
}

/// Request a thumbnail from the persistent sidecar and wait for the reply.
// NOTE: async command — a sync command blocks the main thread while awaiting
// the reply, freezing the whole window (observed as "未响应" during a run).
#[tauri::command]
async fn preview(state: State<'_, AppState>, path: String, size: u32) -> Result<Option<String>, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    {
        // Release the lock immediately: the read_loop thread needs it to
        // deliver the sidecar's reply while this command awaits recv.
        let mut waiters = state.preview_waiters.lock().unwrap();
        waiters.push((path.clone(), tx));
    }

    let payload = serde_json::json!({"cmd": "preview", "path": path, "size": size});
    if let Err(err) = send_command(&state, payload) {
        state
            .preview_waiters
            .lock()
            .unwrap()
            .retain(|(p, _)| *p != path);
        return Err(err);
    }

    rx.await.map_err(|_| "preview failed".to_string())
}

#[tauri::command]
async fn pick_directory(app: AppHandle) -> Result<Option<String>, String> {
    // Run the blocking dialog off the async runtime: its internal block_on
    // panics on tokio worker threads, and spawn_blocking threads satisfy the
    // plugin's "not on the main thread" requirement while the blocking API
    // initializes COM itself.
    let dialog = app.dialog().file().set_title("选择照片目录");
    let dir = tauri::async_runtime::spawn_blocking(move || dialog.blocking_pick_folder())
        .await
        .map_err(|e| e.to_string())?;
    Ok(dir.map(|d| d.to_string()))
}

#[tauri::command]
async fn export_csv(app: AppHandle, state: State<'_, AppState>) -> Result<String, String> {
    let rows = state.scores.lock().unwrap().clone();
    if rows.is_empty() {
        return Err("no results to export — run a culling job first".into());
    }
    let dialog = app
        .dialog()
        .file()
        .set_title("导出 CSV")
        .add_filter("CSV", &["csv"])
        .set_file_name("cull_scores.csv");
    let picked = tauri::async_runtime::spawn_blocking(move || dialog.blocking_save_file())
        .await
        .map_err(|e| e.to_string())?;
    let Some(file_path) = picked else {
        return Err("cancelled".into());
    };
    let path = file_path.into_path().map_err(|e| e.to_string())?;

    let mut out = String::from(
        "filename,s_sharp,s_comp,raw_score,rating,vetoed,veto_reason,n_detections,burst_group,has_arw\n",
    );
    for r in &rows {
        out.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{},,{},{},,",
            r.name,
            r.sharp,
            r.comp,
            r.raw,
            r.rating,
            if r.rating > 0 { "" } else { "True" },
            r.veto.replace(',', " "),
        ));
        out.push('\n');
    }
    std::fs::write(&path, out).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().to_string())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            start_run,
            stop_run,
            scan_directory,
            preview,
            pick_directory,
            export_csv
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_app, event| {
            // Kill the sidecar when the app exits so no orphan process remains.
            if let tauri::RunEvent::Exit = event {
                retire_sidecar(&_app.state::<AppState>());
            }
        });
}
