//! Frida GUM-based function interception for `libamdhip64.so` and SMI libraries.
//!
//! Provides [`HookManager`] which wraps Frida's interceptor in a transaction-scoped
//! API, and [`HookFn`] which stores the original function pointer after replacement.

use std::ffi::c_void;
use std::ops::Deref;
use std::sync::{LazyLock, OnceLock};

use frida_gum::interceptor::Interceptor;
pub use frida_gum::interceptor::{InvocationContext, InvocationListener, Listener};
use frida_gum::{Gum, Module, NativePointer};

use crate::HookError;

static GUM: LazyLock<Gum> = LazyLock::new(Gum::obtain);

/// Resolve a loaded shared library by filename prefix, scanning `/proc/self/maps` on Linux.
///
/// Returns the full filesystem path if found. Uses `procfs` on Linux and falls back to
/// Frida's module enumeration on other platforms.
fn resolve_library(prefix: &str) -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let proc = procfs::process::Process::myself().ok()?;
        let maps = proc.maps().ok()?;

        for entry in maps {
            let procfs::process::MMapPath::Path(ref path) = entry.pathname else {
                continue;
            };
            let filename = path.file_name()?.to_str()?;
            if filename.starts_with(prefix) {
                tracing::debug!("resolved library prefix '{prefix}' → {}", path.display());
                return path.to_str().map(String::from);
            }
        }
        tracing::debug!("library prefix '{prefix}' not found in /proc/self/maps");
        None
    }

    #[cfg(not(target_os = "linux"))]
    {
        use frida_gum::ModuleMap;
        let mut map = ModuleMap::new();
        map.update();
        for module in map.values() {
            let name = module.name();
            if name.starts_with(prefix) {
                tracing::debug!("resolved library prefix '{prefix}' via Frida: {name}");
                return Some(name.to_string());
            }
        }
        tracing::debug!("library prefix '{prefix}' not found via Frida module map");
        None
    }
}

/// Check whether a library whose filename starts with `prefix` is currently loaded.
pub fn is_module_loaded(prefix: &str) -> bool {
    resolve_library(prefix).is_some()
}

/// Scoped helper for hooking symbols within a single module (or globally).
pub struct Hooker<'a> {
    interceptor: &'a mut Interceptor,
    module_path: Option<&'a str>,
}

impl Hooker<'_> {
    /// Replace `symbol` with `detour`, returning a pointer to the original implementation.
    pub fn hook_export(
        &mut self,
        symbol: &str,
        detour: *mut c_void,
    ) -> Result<NativePointer, HookError> {
        let target = match self.module_path {
            Some(path) => Module::load(&GUM, path).find_export_by_name(symbol),
            None => Module::find_global_export_by_name(symbol),
        }
        .ok_or_else(|| HookError::NoSymbolName(symbol.to_owned().into()))?;

        tracing::debug!("replacing {symbol} at {:p}", target.0);

        let original = self
            .interceptor
            .replace(
                target,
                NativePointer(detour),
                NativePointer(std::ptr::null_mut()),
            )
            .map_err(HookError::from)?;

        tracing::debug!("hooked {symbol}, original at {:p}", original.0);
        Ok(original)
    }
}

/// Manages Frida GUM interceptor transactions and resolved module paths.
///
/// A transaction is started on creation (`Default::default()`) and committed on drop.
/// All replacements within the transaction are applied atomically.
pub struct HookManager {
    interceptor: Interceptor,
    /// Resolved full paths for libraries we've hooked into.
    pub module_names: Vec<String>,
}

impl Default for HookManager {
    fn default() -> Self {
        let mut interceptor = Interceptor::obtain(&GUM);
        interceptor.begin_transaction();
        Self {
            interceptor,
            module_names: Vec::new(),
        }
    }
}

impl Drop for HookManager {
    fn drop(&mut self) {
        self.interceptor.end_transaction();
    }
}

impl HookManager {
    /// Create a [`Hooker`] scoped to the given module prefix (e.g. `"libamdhip64."`).
    ///
    /// Resolves the prefix to a full path via `/proc/self/maps` and caches it. Pass
    /// `None` to hook global symbols.
    pub fn hooker<'a>(
        &'a mut self,
        module_prefix: Option<&'a str>,
    ) -> Result<Hooker<'a>, HookError> {
        let resolved = match module_prefix {
            Some(prefix) => {
                let full_path = resolve_library(prefix)
                    .ok_or_else(|| HookError::NoModuleName(prefix.to_owned().into()))?;

                // Cache the resolved path so we don't re-scan /proc/self/maps
                let idx = self
                    .module_names
                    .iter()
                    .position(|existing| *existing == full_path);
                let stored = match idx {
                    Some(i) => &self.module_names[i],
                    None => {
                        self.module_names.push(full_path);
                        self.module_names.last().expect("just pushed")
                    }
                };
                Some(stored.as_str())
            }
            None => None,
        };

        tracing::debug!("hooking module: {resolved:?}");

        Ok(Hooker {
            interceptor: &mut self.interceptor,
            module_path: resolved,
        })
    }

    /// Convenience: resolve module + hook a single symbol in one call.
    pub fn hook_export(
        &mut self,
        module_prefix: Option<&str>,
        symbol: &str,
        detour: *mut c_void,
    ) -> Result<NativePointer, HookError> {
        self.hooker(module_prefix)?.hook_export(symbol, detour)
    }

    /// Attach an [`InvocationListener`] to an already-resolved function pointer.
    pub fn attach<I: InvocationListener>(
        &mut self,
        function: NativePointer,
        listener: &mut I,
    ) -> Result<Listener, HookError> {
        self.interceptor
            .attach(function, listener)
            .map_err(HookError::from)
    }

    /// Detach a previously attached listener.
    pub fn detach(&mut self, listener: Listener) {
        self.interceptor.detach(listener);
    }
}

/// Thread-safe container for a lazily-initialized function pointer.
///
/// Wraps `OnceLock<T>` with `Deref` so the stored function can be called directly
/// after initialization. Panics on deref if not yet initialized.
#[derive(Debug)]
pub struct HookFn<T>(OnceLock<T>);

impl<T> HookFn<T> {
    /// Create an empty, uninitialized container.
    pub const fn default_const() -> Self {
        Self(OnceLock::new())
    }

    /// Store the function pointer. Returns `Err(value)` if already set.
    pub fn set(&self, value: T) -> Result<(), T> {
        self.0.set(value)
    }

    /// Get a reference if initialized.
    pub fn get(&self) -> Option<&T> {
        self.0.get()
    }

    /// Returns `true` if the function pointer has not been set yet.
    pub fn is_none(&self) -> bool {
        self.0.get().is_none()
    }
}

impl<T> Deref for HookFn<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0
            .get()
            .expect("HookFn dereferenced before initialization")
    }
}
