/// Intercept a shared library symbol via Frida GUM, storing the original function pointer.
///
/// Replaces `symbol` in `module` with `detour`, transmutes the original into `fn_type`,
/// and stores it in `storage` (a `HookFn<fn_type>`). Errors if the symbol is missing
/// or `storage` was already initialized.
#[macro_export]
macro_rules! replace_symbol {
    ($manager:expr, $module:expr, $symbol:expr, $detour:expr, $fn_type:ty, $storage:expr) => {{
        (|| -> Result<(), $crate::HookError> {
            let original_ptr = $manager
                .hook_export($module, $symbol, $detour as *mut std::ffi::c_void)?
                .0;
            let original: $fn_type = unsafe { std::mem::transmute(original_ptr) };
            tracing::trace!(concat!("hooked ", stringify!($symbol)));
            $storage.set(original).map_err(|_| {
                $crate::HookError::HookAlreadyInitialized(std::borrow::Cow::Borrowed($symbol))
            })
        })()
    }};
}
