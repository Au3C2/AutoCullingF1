//! auto-culling Tauri shell: spawns the Python CLI as a sidecar speaking the
//! JSON Lines protocol (cull/protocol.py), forwards its events to the
//! frontend, and answers preview requests through the persistent sidecar.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::mpsc::{channel, Sender};
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
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

#[derive(Serialize, Clone, Default)]
struct ScoreRow {
    name: String,
    rating: i32,
    sharp: f64,
    comp: f64,
    raw: f64,
    veto: String,
}

/// The running (or idle) sidecar process. `child` doubles as stdin: writing
/// "cancel\n" / preview commands goes through `CommandChild::write`.
struct Sidecar {
    gen: u64,
    child: Mutex<CommandChild>,
}

struct AppState {
    sidecar: Mutex<Option<Sidecar>>,
    scores: Mutex<Vec<ScoreRow>>,
    /// Pending preview requests keyed by path (sidecar responds asynchronously).
    preview_waiters: Mutex<Vec<(String, Sender<Option<String>>)>>,
    next_gen: std::sync::atomic::AtomicU64,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            sidecar: Mutex::new(None),
            scores: Mutex::new(Vec::new()),
            preview_waiters: Mutex::new(Vec::new()),
            next_gen: std::sync::atomic::AtomicU64::new(1),
        }
    }
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

fn push_flag(args: &mut Vec<String>, cond: bool, flag: &str) {
    if cond {
        args.push(flag.to_string());
    }
}

/// Build the CLI argument vector for a run (json-lines mode is implied).
fn build_args(config: &RunConfig) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "--input-dir".into(),
        config.input_dir.clone(),
        "--json-lines".into(),
        "--top-n".into(),
        config.top_n.to_string(),
        "--p4-policy".into(),
        config.p4_policy.clone(),
        "--scale-width".into(),
        config.scale_width.to_string(),
        "--workers".into(),
        config.workers.to_string(),
        "--sharp-thresh".into(),
        config.sharp_thresh.to_string(),
        "--w-sharp".into(),
        config.w_sharp.to_string(),
        "--w-comp".into(),
        config.w_comp.to_string(),
        "--min-raw".into(),
        config.min_raw.to_string(),
        "--conf".into(),
        config.conf.to_string(),
    ];
    push_flag(&mut args, config.recursive, "--recursive");
    push_flag(&mut args, config.force, "-f");
    push_flag(&mut args, !config.autocrop, "--crop-off");
    push_flag(&mut args, config.rename, "--rename");
    push_flag(&mut args, config.dry_run, "--dry-run");
    if let Some(key) = &config.rf_api_key {
        if !key.is_empty() {
            args.push("--rf-api-key".into());
            args.push(key.clone());
        }
    }
    args
}

fn read_loop(app: AppHandle, gen: u64, mut rx: tokio::sync::mpsc::Receiver<CommandEvent>) {
    log_line(&format!("read_loop started (gen {gen})"));
    while let Some(event) = rx.blocking_recv() {
        match event {
            CommandEvent::Stdout(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                for line in text.lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                        log_line(&format!("parse failed ({} bytes): {}", line.len(), line));
                        continue;
                    };
                    let Some(kind) = value.get("type").and_then(|v| v.as_str()).map(str::to_string)
                    else {
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
                        log_line(&format!(
                            "preview evt received: path={path} png={}",
                            png.as_ref().map(|s| s.len()).unwrap_or(0)
                        ));
                        let state_ref = app.state::<AppState>();
                        let mut waiters = state_ref.preview_waiters.lock().unwrap();
                        if let Some(pos) = waiters.iter().position(|(p, _)| *p == path) {
                            log_line("preview waiter matched");
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

                    // Rename "type" -> "kind" for the frontend dispatch.
                    let mut payload = value;
                    if let Some(obj) = payload.as_object_mut() {
                        obj.remove("type");
                    }
                    payload["kind"] = serde_json::Value::String(kind);
                    let _ = app.emit("evt", payload);
                }
            }
            CommandEvent::Terminated(_) => {
                log_line(&format!("sidecar terminated (gen {gen})"));
                // Only clear the state if the terminating process is still the
                // current one — a retired sidecar must not drop a newer spawn
                // (dropping its CommandChild closes the stdin pipe and kills
                // the new sidecar's command loop).
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
            CommandEvent::Stderr(bytes) => {
                log_line(&format!(
                    "sidecar stderr: {}",
                    String::from_utf8_lossy(&bytes).trim()
                ));
            }
            _ => {}
        }
    }
    log_line("read_loop ended");
}

#[tauri::command]
fn start_run(app: AppHandle, state: State<'_, AppState>, config: RunConfig) -> Result<(), String> {
    log_line(&format!("start_run: dir={}", config.input_dir));
    // Retire any previous sidecar (one run per process).
    let mut guard = state.sidecar.lock().unwrap();
    if let Some(old) = guard.take() {
        log_line("retiring previous sidecar");
        {
            let mut old_child = old.child.lock().unwrap();
            let _ = old_child.write(b"{\"cmd\":\"quit\"}\n");
        }
        let child = old.child.into_inner().unwrap();
        let _ = child.kill();
    }
    state.scores.lock().unwrap().clear();

    let args = build_args(&config);
    let sidecar = match app.shell().sidecar("cull-sidecar") {
        Ok(s) => s,
        Err(e) => {
            log_line(&format!("sidecar() error: {e}"));
            return Err(e.to_string());
        }
    };
    log_line("spawning sidecar...");
    // NOTE: Command::spawn already pipes stdin (os_pipe) — CommandChild::write
    // reaches the child. Keeping the CommandChild alive (i.e. not dropping it
    // via a stale Terminated handler) is what keeps the sidecar's stdin open
    // after the run, which the preview command loop depends on.
    let (rx, child) = match sidecar.args(&args).spawn() {
        Ok(pair) => pair,
        Err(e) => {
            log_line(&format!("spawn() error: {e}"));
            return Err(e.to_string());
        }
    };
    let gen = state.next_gen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    log_line(&format!("sidecar spawned (gen {gen})"));
    *guard = Some(Sidecar {
        gen,
        child: Mutex::new(child),
    });
    drop(guard);

    let app2 = app.clone();
    std::thread::spawn(move || read_loop(app2, gen, rx));
    let _ = app.emit("run-status", serde_json::json!({ "running": true }));
    log_line("start_run done");
    Ok(())
}

#[tauri::command]
fn stop_run(state: State<'_, AppState>) -> Result<(), String> {
    let guard = state.sidecar.lock().unwrap();
    if let Some(sidecar) = guard.as_ref() {
        let mut child = sidecar.child.lock().unwrap();
        child
            .write(b"cancel\n")
            .map_err(|e| format!("failed to write to sidecar: {e}"))?;
    }
    Ok(())
}

/// Request a thumbnail from the persistent sidecar and wait for the reply.
#[tauri::command]
fn preview(state: State<'_, AppState>, path: String, size: u32) -> Result<Option<String>, String> {
    log_line(&format!("preview cmd: size={size} path={path}"));
    let (tx, rx) = channel::<Option<String>>();
    {
        // Release the lock immediately: the read_loop thread needs it to
        // deliver the sidecar's reply while this command awaits recv.
        let mut waiters = state.preview_waiters.lock().unwrap();
        waiters.push((path.clone(), tx));
    }

    {
        let guard = state.sidecar.lock().unwrap();
        let Some(sidecar) = guard.as_ref() else {
            log_line("preview: engine not running");
            return Err("engine is not running".into());
        };
        let mut child = sidecar.child.lock().unwrap();
        // Newline-terminated: the sidecar reads its stdin line by line.
        let req = serde_json::json!({"cmd": "preview", "path": path, "size": size}).to_string()
            + "\n";
        match child.write(req.as_bytes()) {
            Ok(_) => log_line("preview: request written to sidecar"),
            Err(e) => {
                log_line(&format!("preview: write failed: {e}"));
                return Err(format!("failed to write to sidecar: {e}"));
            }
        }
    }

    let reply = rx.recv_timeout(Duration::from_secs(60));
    log_line(&format!("preview: reply {:?}", reply.is_ok()));
    reply.map_err(|_| "preview timed out".to_string())
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
            preview,
            pick_directory,
            export_csv
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_app, event| {
            // Kill the sidecar when the app exits so no orphan process remains.
            if let tauri::RunEvent::Exit = event {
                if let Some(sidecar) = _app.state::<AppState>().sidecar.lock().unwrap().take() {
                    {
                        let mut child = sidecar.child.lock().unwrap();
                        let _ = child.write(b"{\"cmd\":\"quit\"}\n");
                    }
                    let child = sidecar.child.into_inner().unwrap();
                    let _ = child.kill();
                }
            }
        });
}
