// Verifies Tauri asset-protocol scope semantics for the highres cache paths.
// Mirrors tauri-2.11.5/src/scope/fs.rs: require_literal_separator: true,
// require_literal_leading_dot: true (unix default).

use glob::MatchOptions;
use glob::Pattern;

fn matches(pattern: &str, path: &str) -> bool {
    let pat = Pattern::new(pattern).unwrap();
    let path: std::path::PathBuf = std::path::Path::new(path).components().collect();
    pat.matches_path_with(
        &path,
        MatchOptions {
            require_literal_separator: true,
            require_literal_leading_dot: true,
            case_sensitive: true,
            ..Default::default()
        },
    )
}

#[test]
fn dotfile_cache_dir_is_not_matched_by_doublestar_wildcard() {
    // Tauri unix default: require_literal_leading_dot = true.
    // "~/.cache/..." contains a dot-directory component -> "**" must NOT match.
    assert!(!matches(
        "**",
        "/Users/joeylin/.cache/auto_culling/highres/IMG_1_hash.jpg"
    ));
}

#[test]
fn os_cache_dir_is_matched_by_doublestar_wildcard() {
    // After moving the cache to the OS cache dir (~/Library/Caches on macOS,
    // a non-dotfile path), the "**" wildcard matches again.
    assert!(matches(
        "**",
        "/Users/joeylin/Library/Caches/AutoCulling/highres/IMG_1_hash.jpg"
    ));
    // Resolved "$CACHE/AutoCulling/highres/**" scope entry (Tauri resolves
    // $CACHE -> ~/Library/Caches on macOS before compiling the pattern).
    assert!(matches(
        "/Users/joeylin/Library/Caches/AutoCulling/highres/**",
        "/Users/joeylin/Library/Caches/AutoCulling/highres/IMG_1_hash.jpg"
    ));
}
