// Tauri 2 Rust Backend for Auto-Culling Desktop GUI

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::sync::oneshot;

struct SidecarState {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    preview_waiters: Arc<Mutex<HashMap<String, Vec<oneshot::Sender<serde_json::Value>>>>>,
}

#[tauri::command]
async fn select_folder(app: AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog().file().pick_folder(move |folder_path| {
        let path_str = folder_path.map(|p| p.to_string());
        let _ = tx.send(path_str);
    });
    rx.await.map_err(|e| e.to_string())
}

/// Helper to find repo root directory
fn repo_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if cwd.join("cull_photos.py").exists() {
        return cwd;
    }
    if cwd.join("tauri.conf.json").exists() {
        if let Some(parent) = cwd.parent() {
            if parent.join("cull_photos.py").exists() {
                return parent.to_path_buf();
            }
        }
    }
    if let Ok(mut exe) = std::env::current_exe() {
        while exe.pop() {
            if exe.join("cull_photos.py").exists() {
                return exe;
            }
        }
    }
    cwd
}

fn ensure_sidecar(app: &AppHandle, state: &mut SidecarState) -> Result<(), String> {
    if state.child.is_some() {
        return Ok(());
    }

    let root = repo_root();
    let resource_dir = app
        .path()
        .resource_dir()
        .unwrap_or_else(|_| PathBuf::from("."));
    
    let mut cmd: Command;
    let venv_python = if cfg!(windows) {
        root.join(".venv/Scripts/python.exe")
    } else {
        root.join(".venv/bin/python")
    };
    let script_path = root.join("cull_photos.py");

    if venv_python.exists() && script_path.exists() {
        cmd = Command::new(&venv_python);
        cmd.current_dir(&root);
        cmd.arg(&script_path).arg("--json-lines");
    } else {
        // Production bundle fallback
        let binary_name = if cfg!(windows) {
            "cull-sidecar.exe"
        } else {
            "cull-sidecar"
        };
        let sidecar_bin = resource_dir.join(binary_name);
        if sidecar_bin.exists() {
            cmd = Command::new(sidecar_bin);
            cmd.arg("--json-lines");
        } else {
            cmd = Command::new("python3");
            cmd.current_dir(&root);
            cmd.arg(&script_path).arg("--json-lines");
        }
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn sidecar: {}", e))?;
    let stdin = child.stdin.take().ok_or("Failed to open sidecar stdin")?;
    let stdout = child.stdout.take().ok_or("Failed to open sidecar stdout")?;
    if let Some(err_pipe) = child.stderr.take() {
        std::thread::spawn(move || {
            let reader = BufReader::new(err_pipe);
            for line in reader.lines() {
                if let Ok(line_str) = line {
                    eprintln!("[sidecar stderr] {}", line_str);
                }
            }
        });
    }

    let waiters = Arc::clone(&state.preview_waiters);
    let app_clone = app.clone();

    // Spawn reader thread for line-delimited JSON events
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(line_str) = line {
                let line_trimmed = line_str.trim();
                if line_trimmed.is_empty() {
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(line_trimmed) {
                    if let Some(event_type) = val.get("type").and_then(|t| t.as_str()).map(|s| s.to_string()) {
                        if event_type == "preview" {
                            if let Some(path) = val.get("path").and_then(|p| p.as_str()) {
                                let mut map = waiters.lock().unwrap();
                                if let Some(senders) = map.remove(path) {
                                    for sender in senders {
                                        let _ = sender.send(val.clone());
                                    }
                                }
                            }
                        }
                        // Emit event to webview
                        let _ = app_clone.emit(&event_type, val);
                    }
                }
            }
        }
    });

    state.child = Some(child);
    state.stdin = Some(stdin);
    Ok(())
}

fn send_sidecar_command(
    app: &AppHandle,
    state: &mut SidecarState,
    payload: serde_json::Value,
) -> Result<(), String> {
    ensure_sidecar(app, state)?;
    if let Some(ref mut stdin) = state.stdin {
        let mut msg = serde_json::to_string(&payload).map_err(|e| e.to_string())?;
        msg.push('\n');
        stdin.write_all(msg.as_bytes()).map_err(|e| e.to_string())?;
        stdin.flush().map_err(|e| e.to_string())?;
        Ok(())
    } else {
        Err("Sidecar stdin not available".into())
    }
}

#[tauri::command]
async fn scan(
    app: AppHandle,
    state: State<'_, Arc<Mutex<SidecarState>>>,
    dir: String,
    recursive: bool,
) -> Result<(), String> {
    let payload = serde_json::json!({
        "cmd": "scan",
        "dir": dir,
        "recursive": recursive
    });
    let mut guard = state.lock().unwrap();
    send_sidecar_command(&app, &mut guard, payload)
}

#[tauri::command]
async fn run(
    app: AppHandle,
    state: State<'_, Arc<Mutex<SidecarState>>>,
    dir: String,
    config: serde_json::Value,
) -> Result<(), String> {
    let payload = serde_json::json!({
        "cmd": "run",
        "dir": dir,
        "config": config
    });
    let mut guard = state.lock().unwrap();
    send_sidecar_command(&app, &mut guard, payload)
}

#[tauri::command]
async fn cancel(
    app: AppHandle,
    state: State<'_, Arc<Mutex<SidecarState>>>,
) -> Result<(), String> {
    let payload = serde_json::json!({
        "cmd": "cancel"
    });
    let mut guard = state.lock().unwrap();
    send_sidecar_command(&app, &mut guard, payload)
}

#[tauri::command]
async fn preview(
    app: AppHandle,
    state: State<'_, Arc<Mutex<SidecarState>>>,
    path: String,
    size: Option<u32>,
) -> Result<serde_json::Value, String> {
    let (tx, rx) = oneshot::channel();
    {
        let waiters = {
            let guard = state.lock().unwrap();
            Arc::clone(&guard.preview_waiters)
        };
        waiters.lock().unwrap().entry(path.clone()).or_insert_with(Vec::new).push(tx);

        let payload = serde_json::json!({
            "cmd": "preview",
            "path": path,
            "size": size.unwrap_or(640)
        });
        let mut guard = state.lock().unwrap();
        send_sidecar_command(&app, &mut guard, payload)?;
    }

    match tokio::time::timeout(std::time::Duration::from_secs(10), rx).await {
        Ok(Ok(val)) => Ok(val),
        Ok(Err(_)) => Err("Preview channel dropped".into()),
        Err(_) => Err("Preview request timed out".into()),
    }
}

#[tauri::command]
async fn export_csv(
    _app: AppHandle,
    _state: State<'_, Arc<Mutex<SidecarState>>>,
    dir: String,
) -> Result<String, String> {
    let path = PathBuf::from(dir).join("scores.csv");
    Ok(path.to_string_lossy().to_string())
}

fn main() {
    let sidecar_state = Arc::new(Mutex::new(SidecarState {
        child: None,
        stdin: None,
        preview_waiters: Arc::new(Mutex::new(HashMap::new())),
    }));

    let sidecar_clone = Arc::clone(&sidecar_state);

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(sidecar_state)
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                if let Ok(icon) = tauri::image::Image::from_bytes(include_bytes!("../icons/128x128@2x.png")) {
                    let _ = window.set_icon(icon);
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            select_folder,
            scan,
            run,
            cancel,
            preview,
            export_csv
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(move |_app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                let mut guard = sidecar_clone.lock().unwrap();
                if let Some(ref mut stdin) = guard.stdin {
                    let _ = stdin.write_all(b"{\"cmd\":\"quit\"}\n");
                    let _ = stdin.flush();
                }
                if let Some(ref mut child) = guard.child {
                    let _ = child.kill();
                }
            }
        });
}
