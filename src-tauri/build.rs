fn main() {
    // tauri.conf.json declares binaries/cull-sidecar as externalBin, and the
    // tauri build script hard-fails when the per-target file is missing.
    // Dev builds (tauri dev / cargo check) do not need the packaged sidecar —
    // main.rs falls back to the repo venv python — so create an empty
    // placeholder that gets overwritten by the real PyInstaller artifact
    // before `tauri build`. The triple is taken from cargo's HOST/TARGET
    // env vars (TAURI_ENV_TARGET_TRIPLE is only set inside tauri_build's
    // re-run, not for the first invocation).
    let triple = std::env::var("TARGET")
        .or_else(|_| std::env::var("HOST"))
        .map(|t| t.to_string())
        .unwrap_or_default();
    if !triple.is_empty() {
        // tauri resolves externalBin as `<program>-<triple><EXE_SUFFIX>`, so the
        // placeholder must carry the platform executable suffix or tauri_build
        // still hard-fails on the missing file.
        let suffix = if cfg!(windows) { ".exe" } else { "" };
        let placeholder = std::path::Path::new("binaries")
            .join(format!("cull-sidecar-{triple}{suffix}"));
        if !placeholder.exists() {
            let _ = std::fs::create_dir_all("binaries");
            let _ = std::fs::write(&placeholder, b"");
            println!("cargo:warning=created placeholder {}", placeholder.display());
        }
    }
    tauri_build::build()
}
