//! Self-update from GitHub Releases: the latest release's `FH1Rewrite-<ver>-win64.zip` replaces the program files
//! (the launcher, `bin\`, `licenses\`, README.txt) in place. `data\` is never touched; when a release changes the
//! converted data, the new launcher's data revision asks for the usual Update run (main.rs `RELEASE`).
//!
//! HTTP goes through Windows' own `curl.exe` (Windows 10 1803+), like fh1setup's downloads. The zip is checked against
//! the SHA-256 digest GitHub publishes for the asset. The running launcher renames itself to `*.old` (Windows allows
//! renaming a running exe), writes the new one and restarts; leftovers are removed on the next start.
//!
//! Linux takes `-linux-x64.tar.gz` (unpacked with `tar`, keeping the executable bits). macOS takes `-macos-arm64.zip`,
//! whose `FH1 Rewrite.app` replaces the running app bundle's files in place ([`install_root`]); the new launcher is
//! started through LaunchServices (`open -n`).
//!
//! `FH1_UPDATE=off` disables the check; `FH1_UPDATE_REPO=owner/name` points at another repository (testing).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;

use sha2::{Digest, Sha256};

const REPO: &str = "Sm1jjj/fh1_rewrite";
/// The release asset for this platform (tools/release.ps1 / the Linux and macOS workflows).
const ASSET_SUFFIX: &str = if cfg!(windows) {
    "-win64.zip"
} else if cfg!(target_os = "macos") {
    "-macos-arm64.zip"
} else {
    "-linux-x64.tar.gz"
};
/// The app bundle's name in the macOS release archive.
const APP: &str = "FH1 Rewrite.app";

#[derive(Clone, Debug)]
pub struct Release {
    pub version: String,
    pub url: String,
    pub size: u64,
    /// Hex SHA-256 from the asset's `digest` (`sha256:<hex>`), when GitHub provides it.
    pub sha256: Option<String>,
    pub page: String,
}

pub enum UpdateMsg {
    /// Newer release found (None = up to date or check failed quietly).
    Checked(Option<Release>),
    Progress(f32),
    /// Files replaced: restart into the new launcher.
    Installed,
    Failed(String),
}

fn repo() -> String {
    std::env::var("FH1_UPDATE_REPO").ok().filter(|r| !r.is_empty()).unwrap_or_else(|| REPO.to_owned())
}

pub fn enabled() -> bool {
    !std::env::var("FH1_UPDATE").is_ok_and(|v| v.eq_ignore_ascii_case("off") || v == "0")
}

fn curl() -> Command {
    let mut c = Command::new(if cfg!(windows) { "curl.exe" } else { "curl" });
    crate::hide_console(&mut c).stdin(Stdio::null());
    c
}

/// `1.2.3` (a leading `v` ignored) as numbers, for comparing versions.
fn parse(v: &str) -> Vec<u64> {
    v.trim().trim_start_matches(['v', 'V']).split(['.', '-']).map_while(|p| p.parse().ok()).collect()
}

pub fn newer(remote: &str, local: &str) -> bool {
    parse(remote) > parse(local)
}

/// The latest release, if it is newer than `current` and has a Windows zip. Errors are swallowed (offline = no update).
pub fn check(current: &str) -> Option<Release> {
    let out = curl()
        .args(["-sSfL", "--max-time", "15", "-H", "Accept: application/vnd.github+json", "-H"])
        .arg(format!("User-Agent: fh1-launcher/{current}"))
        .arg(format!("https://api.github.com/repos/{}/releases/latest", repo()))
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    let tag = v["tag_name"].as_str()?;
    let force = std::env::var("FH1_UPDATE_FORCE").is_ok_and(|v| v == "1");
    if !force && !newer(tag, current) {
        return None;
    }
    let asset = v["assets"].as_array()?.iter().find(|a| a["name"].as_str().is_some_and(|n| n.ends_with(ASSET_SUFFIX)))?;
    Some(Release {
        version: tag.trim_start_matches('v').to_owned(),
        url: asset["browser_download_url"].as_str()?.to_owned(),
        size: asset["size"].as_u64().unwrap_or(0),
        sha256: asset["digest"].as_str().and_then(|d| d.strip_prefix("sha256:")).map(str::to_ascii_lowercase),
        page: v["html_url"].as_str().unwrap_or("").to_owned(),
    })
}

/// Download, verify and unpack `rel` over `root`, reporting through `tx`. Run on a worker thread.
pub fn install(rel: &Release, root: &Path, data: &Path, tx: &Sender<UpdateMsg>, repaint: &dyn Fn()) {
    let send = |m: UpdateMsg| {
        let _ = tx.send(m);
        repaint();
    };
    match install_inner(rel, root, data, &|p| send(UpdateMsg::Progress(p))) {
        Ok(()) => send(UpdateMsg::Installed),
        Err(e) => send(UpdateMsg::Failed(e)),
    }
}

fn install_inner(rel: &Release, root: &Path, data: &Path, progress: &dyn Fn(f32)) -> Result<(), String> {
    let dir = data.join("updates");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let zip_path = dir.join(format!("FH1Rewrite-{}{ASSET_SUFFIX}", rel.version));
    let _ = std::fs::remove_file(&zip_path);

    // Download (progress = bytes on disk / asset size, polled while curl runs).
    let mut child = curl()
        .args(["-sSfL", "--retry", "2", "-o"])
        .arg(&zip_path)
        .arg(&rel.url)
        .spawn()
        .map_err(|e| format!("could not start curl.exe: {e}"))?;
    let status = loop {
        if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
            break s;
        }
        if rel.size > 0 {
            let got = zip_path.metadata().map(|m| m.len()).unwrap_or(0);
            progress(0.9 * got as f32 / rel.size as f32);
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    };
    if !status.success() {
        return Err("download failed (check your internet connection)".into());
    }

    // Verify.
    let mut f = std::fs::File::open(&zip_path).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if let Some(want) = &rel.sha256 {
        if &got != want {
            let _ = std::fs::remove_file(&zip_path);
            return Err("the download is damaged (checksum mismatch); try again".into());
        }
    }
    progress(0.92);

    // Unpack to a staging folder first, so a broken archive never leaves half an install.
    let stage = dir.join(format!("stage-{}", rel.version));
    let _ = std::fs::remove_dir_all(&stage);
    std::fs::create_dir_all(&stage).map_err(|e| e.to_string())?;
    unpack(&zip_path, &stage)?;
    let mut files = Vec::new();
    walk(&stage, &stage, &mut files);
    let files: Vec<(String, String)> = files.into_iter().filter_map(|f| installed_path(&f).map(|to| (f, to))).collect();
    if !files.iter().any(|(_, f)| f == LAUNCHER) || !files.iter().any(|(_, f)| f == ENGINE) {
        return Err("the update zip doesn't look like an FH1 Rewrite release".into());
    }
    progress(0.96);

    // Swap: running exes are renamed aside (allowed on Windows), then the new files move in.
    for (f, to) in &files {
        let (src, dst) = (stage.join(f), root.join(to));
        if let Some(p) = dst.parent() {
            std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        if dst.exists() {
            let old = old_name(&dst);
            let _ = std::fs::remove_file(&old);
            if std::fs::rename(&dst, &old).is_err() {
                // Not running and not renameable: try a plain overwrite.
                std::fs::copy(&src, &dst).map_err(|e| format!("{} is in use: {e}", dst.display()))?;
                continue;
            }
        }
        if std::fs::rename(&src, &dst).is_err() {
            std::fs::copy(&src, &dst).map_err(|e| format!("writing {}: {e}", dst.display()))?;
        }
    }
    let _ = std::fs::remove_dir_all(&stage);
    let _ = std::fs::remove_file(&zip_path);
    progress(1.0);
    Ok(())
}

/// The launcher, relative to [`install_root`].
pub const LAUNCHER: &str = if cfg!(windows) {
    "FH1 Rewrite.exe"
} else if cfg!(target_os = "macos") {
    "Contents/MacOS/fh1-rewrite"
} else {
    "fh1-rewrite"
};
/// The game, relative to [`install_root`].
const ENGINE: &str = if cfg!(windows) {
    "bin/fh1-engine.exe"
} else if cfg!(target_os = "macos") {
    "Contents/MacOS/fh1-engine"
} else {
    "bin/fh1-engine"
};

/// What an update replaces, given the launcher's folder: that folder, or on macOS the app bundle around it
/// (`<X>.app/Contents/MacOS`).
pub fn install_root(launcher_dir: &Path) -> PathBuf {
    let bundle = launcher_dir.parent().filter(|c| c.ends_with("Contents")).and_then(Path::parent);
    match bundle {
        Some(b) if cfg!(target_os = "macos") && launcher_dir.ends_with("MacOS") && b.extension().is_some_and(|e| e == "app") => b.to_path_buf(),
        _ => launcher_dir.to_path_buf(),
    }
}

/// Where a file of the unpacked release goes, relative to [`install_root`]; None = not installed (the data folder,
/// and on macOS everything outside the app bundle, e.g. README.txt).
fn installed_path(staged: &str) -> Option<String> {
    if cfg!(target_os = "macos") {
        staged.strip_prefix(APP)?.strip_prefix('/').map(str::to_owned)
    } else {
        (!staged.starts_with("data/")).then(|| staged.to_owned())
    }
}

/// Unpack the release archive into `stage`, dropping its top folder (`FH1Rewrite-<ver>/`).
#[cfg(not(target_os = "linux"))]
fn unpack(archive: &Path, stage: &Path) -> Result<(), String> {
    let file = std::fs::File::open(archive).map_err(|e| e.to_string())?;
    let mut ar = zip::ZipArchive::new(std::io::BufReader::new(file)).map_err(|e| format!("bad update zip: {e}"))?;
    for i in 0..ar.len() {
        let mut e = ar.by_index(i).map_err(|e| e.to_string())?;
        let name = e.name().replace('\\', "/");
        let Some((_, rel_path)) = name.split_once('/') else { continue };
        // Skip directories and anything that would escape the install.
        if rel_path.is_empty() || e.is_dir() || rel_path.split('/').any(|c| c == ".." || c.contains(':')) {
            continue;
        }
        let dst = stage.join(rel_path);
        std::fs::create_dir_all(dst.parent().unwrap()).map_err(|e| e.to_string())?;
        let mut out = std::fs::File::create(&dst).map_err(|e| e.to_string())?;
        std::io::copy(&mut e, &mut out).map_err(|e| format!("unpacking {rel_path}: {e}"))?;
        // The executables' mode bits (macOS zips record them).
        #[cfg(unix)]
        if let Some(mode) = e.unix_mode() {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dst, std::fs::Permissions::from_mode(mode & 0o777)).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Unpack the release archive into `stage` with the system `tar` (keeps the executable bits).
#[cfg(target_os = "linux")]
fn unpack(archive: &Path, stage: &Path) -> Result<(), String> {
    let ok = Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(stage)
        .arg("--strip-components=1")
        .arg("--no-same-owner")
        .status()
        .map_err(|e| format!("could not run tar: {e}"))?
        .success();
    if ok { Ok(()) } else { Err("bad update archive".into()) }
}

/// Every file under `dir`, as `/`-separated paths relative to `base`.
fn walk(base: &Path, dir: &Path, out: &mut Vec<String>) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(base, &p, out);
        } else if let Ok(rel) = p.strip_prefix(base) {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

fn old_name(p: &Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".old");
    PathBuf::from(s)
}

/// Removes `*.old` files a previous update left behind (the old launcher / engine).
pub fn cleanup(root: &Path) {
    let dirs: &[&str] = if cfg!(target_os = "macos") {
        &["Contents", "Contents/MacOS", "Contents/Resources", "Contents/Resources/licenses", "Contents/_CodeSignature"]
    } else {
        &["", "bin", "licenses"]
    };
    for dir in dirs.iter().map(|d| root.join(d)) {
        for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            if e.path().extension().is_some_and(|x| x == "old") {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
}

/// Start the (new) launcher and let this one close.
pub fn restart(root: &Path) {
    // macOS: a new instance of the bundle through LaunchServices (plain `open` would just bring this one forward).
    if cfg!(target_os = "macos") && root.extension().is_some_and(|e| e == "app") && Command::new("open").arg("-n").arg(root).spawn().is_ok() {
        return;
    }
    let exe = root.join(LAUNCHER);
    let target = if exe.is_file() { exe } else { std::env::current_exe().unwrap_or(exe) };
    let _ = Command::new(target).current_dir(root).spawn();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_paths() {
        let dir = Path::new("/Applications/FH1 Rewrite.app/Contents/MacOS");
        if cfg!(target_os = "macos") {
            assert_eq!(install_root(dir), Path::new("/Applications/FH1 Rewrite.app"));
            assert_eq!(installed_path("FH1 Rewrite.app/Contents/MacOS/fh1-engine").as_deref(), Some(ENGINE));
            assert_eq!(installed_path("README.txt"), None);
        } else {
            assert_eq!(install_root(dir), dir);
            assert_eq!(installed_path(ENGINE).as_deref(), Some(ENGINE));
            assert_eq!(installed_path("data/launcher.json"), None);
        }
        assert_eq!(install_root(Path::new("/opt/fh1")), Path::new("/opt/fh1"));
    }

    #[test]
    fn versions() {
        assert!(newer("v0.1.2", "0.1.1"));
        assert!(newer("0.2.0", "0.1.9"));
        assert!(newer("v1.0.0", "0.9.12"));
        assert!(!newer("v0.1.1", "0.1.1"));
        assert!(!newer("v0.1.0", "0.1.1"));
    }

    /// Downloads the latest real release into a temp "install" with fake old files and checks the swap.
    /// `cargo test --release -p fh1-launcher -- --ignored installs_latest_release` (network, ~75 MB).
    #[test]
    #[ignore]
    fn installs_latest_release() {
        std::env::set_var("FH1_UPDATE_FORCE", "1");
        installs(&check("0.0.0").expect("latest release"), "latest");
    }

    /// The same with a release archive built locally (the macOS workflow's smoke test):
    /// `FH1_UPDATE_TEST_ARCHIVE=<archive> cargo test --release -p fh1-launcher -- --ignored installs_local_archive`.
    #[test]
    #[ignore]
    fn installs_local_archive() {
        let Some(archive) = std::env::var_os("FH1_UPDATE_TEST_ARCHIVE") else { return };
        let archive = std::path::absolute(archive).unwrap();
        let bytes = std::fs::read(&archive).unwrap();
        let sha256 = Some(Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect());
        let path = archive.display().to_string().replace('\\', "/").replace(' ', "%20");
        let url = format!("file://{}{path}", if path.starts_with('/') { "" } else { "/" });
        installs(&Release { version: "0.0.0-local".into(), url, size: bytes.len() as u64, sha256, page: String::new() }, "local");
    }

    /// Installs `rel` over a temp "install" holding fake old files and checks the swap and the clean-up.
    fn installs(rel: &Release, tag: &str) {
        let root = std::env::temp_dir().join(format!("fh1-update-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("data")).unwrap();
        std::fs::create_dir_all(root.join(ENGINE).parent().unwrap()).unwrap();
        std::fs::create_dir_all(root.join(LAUNCHER).parent().unwrap()).unwrap();
        std::fs::write(root.join(LAUNCHER), b"old launcher").unwrap();
        std::fs::write(root.join(ENGINE), b"old engine").unwrap();
        std::fs::write(root.join("data/keep.txt"), b"user data").unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        install(rel, &root, &root.join("data"), &tx, &|| {});
        let msgs: Vec<_> = rx.try_iter().collect();
        assert!(matches!(msgs.last(), Some(UpdateMsg::Installed)), "{:?}", msgs.iter().filter_map(|m| if let UpdateMsg::Failed(e) = m { Some(e.clone()) } else { None }).collect::<Vec<_>>());
        assert!(std::fs::metadata(root.join(ENGINE)).unwrap().len() > 1_000_000);
        assert!(std::fs::metadata(root.join(LAUNCHER)).unwrap().len() > 1_000_000);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for f in [ENGINE, LAUNCHER] {
                assert!(std::fs::metadata(root.join(f)).unwrap().permissions().mode() & 0o111 != 0, "{f} is not executable");
            }
        }
        assert!(old_name(&root.join(ENGINE)).is_file());
        assert_eq!(std::fs::read(root.join("data/keep.txt")).unwrap(), b"user data");
        cleanup(&root);
        assert!(!old_name(&root.join(ENGINE)).exists() && !old_name(&root.join(LAUNCHER)).exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
