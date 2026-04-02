//! Configures the `tracing` subscriber for hipflex.
//!
//! An LD_PRELOAD library must not write to stderr by default — frameworks like PyTorch
//! capture subprocess stderr and assert on the output. Logs go to a daily-rotated file
//! under `/tmp/hipflex/` unless overridden.

use std::env;
use std::path::Path;

use tracing::level_filters::LevelFilter;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::prelude::*;
use tracing_subscriber::{fmt, registry, EnvFilter, Layer, Registry};

const DEFAULT_DIR: &str = "/tmp/hipflex";
const DEFAULT_PREFIX: &str = "hipflex.log";
const ENV_ENABLE: &str = "FH_ENABLE_LOG";
pub const LOG_PATH_ENV_VAR: &str = "FH_LOG_PATH";
const ENV_LEVEL: &str = "FH_LOG_LEVEL";
const STDERR_SENTINEL: &str = "stderr";

fn to_stderr() -> Box<dyn Layer<Registry> + Send + Sync> {
    fmt::layer()
        .with_writer(std::io::stderr)
        .with_target(true)
        .boxed()
}

fn to_file(raw: &str) -> Box<dyn Layer<Registry> + Send + Sync> {
    let path = Path::new(raw);

    let (dir, prefix) = if path.is_dir() {
        (path, DEFAULT_PREFIX)
    } else {
        let dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let prefix = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or(DEFAULT_PREFIX);
        (dir, prefix)
    };

    if let Err(err) = std::fs::create_dir_all(dir) {
        eprintln!("hipflex: cannot create log dir {}: {err}", dir.display());
        return to_stderr();
    }

    match RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix(prefix)
        .build(dir)
    {
        Ok(appender) => fmt::layer()
            .with_writer(appender)
            .with_target(true)
            .with_ansi(false)
            .boxed(),
        Err(err) => {
            eprintln!(
                "hipflex: failed to create log appender at {}: {err}",
                dir.display()
            );
            to_stderr()
        }
    }
}

fn is_disabled() -> bool {
    matches!(
        env::var(ENV_ENABLE).as_deref(),
        Ok("off") | Ok("0") | Ok("false")
    )
}

fn build_filter() -> EnvFilter {
    if is_disabled() {
        return EnvFilter::new("off");
    }
    EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .with_env_var(ENV_LEVEL)
        .from_env_lossy()
}

/// Build the format layer for the given log path.
///
/// Destination priority:
///   1. `FH_ENABLE_LOG=off|0|false` → disabled
///   2. `FH_LOG_PATH=stderr`        → stderr
///   3. `FH_LOG_PATH=/some/path`    → file/directory
///   4. Default                     → `/tmp/hipflex/hipflex.log.*`
pub fn get_fmt_layer(log_path: Option<String>) -> Box<dyn Layer<Registry> + Send + Sync> {
    let output = match log_path.as_deref() {
        Some(STDERR_SENTINEL) => to_stderr(),
        Some(path) => to_file(path),
        None => {
            let _ = std::fs::create_dir_all(DEFAULT_DIR);
            to_file(DEFAULT_DIR)
        }
    };
    output.with_filter(build_filter()).boxed()
}

/// Install the global tracing subscriber using `FH_LOG_PATH` from the environment.
pub fn init() {
    let path = env::var(LOG_PATH_ENV_VAR).ok().filter(|s| !s.is_empty());
    registry().with(get_fmt_layer(path)).init();
}

/// Install the global tracing subscriber with an explicit log path.
pub fn init_with_log_path(log_path: String) {
    registry().with(get_fmt_layer(Some(log_path))).init();
}
