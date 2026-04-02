#![warn(clippy::indexing_slicing)]

use proc_macro::TokenStream;

use proc_macro2::Span;
use quote::quote;
use syn::{Ident, ItemFn, Type};

/// Generates boilerplate for a Frida GUM hook detour function.
///
/// Given a function named `foo_detour`, this attribute produces:
///
/// 1. A type alias: `type FnFoo = unsafe extern "C" fn(...) -> ...;`
/// 2. A static slot: `static FN_FOO: HookFn<FnFoo> = HookFn::default_const();`
/// 3. The original function body, wrapped with a `tracing::trace!` entry log.
///
/// The naming convention strips `_detour` from the function name, converts the
/// remainder to `FnPascalCase` for the type alias and `FN_UPPER_CASE` for the static.
#[proc_macro_attribute]
pub fn hook_fn(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = syn::parse_macro_input!(item as ItemFn);
    let sig = &func.sig;
    let vis = &func.vis;

    let raw_name = sig.ident.to_string();
    let base = raw_name.strip_suffix("_detour").unwrap_or(&raw_name);

    let type_ident = {
        let mut chars = base.chars();
        let capitalized: String = match chars.next() {
            Some(c) => c.to_uppercase().chain(chars).collect(),
            None => String::new(),
        };
        Ident::new(&format!("Fn{capitalized}"), Span::call_site())
    };

    let static_ident = Ident::new(&format!("FN_{}", base.to_uppercase()), Span::call_site());

    let unsafety = sig.unsafety;
    let abi = &sig.abi;
    let ret = &sig.output;

    let mut param_types: Vec<Box<Type>> = sig
        .inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Receiver(_) => panic!("hook functions must not take `self`"),
            syn::FnArg::Typed(pat) => pat.ty.clone(),
        })
        .collect();

    if sig.variadic.is_some() {
        param_types.push(Box::new(Type::Verbatim(quote! { ... })));
    }

    let bare_fn = quote! { #unsafety #abi fn(#(#param_types),*) #ret };
    let fn_name = &sig.ident;
    let body = &func.block;

    let mut traced_func = func.clone();
    let wrapped = quote! {{
        tracing::trace!(target: "hook_fn", "Called {}", stringify!(#fn_name));
        #body
    }};
    if let Ok(block) = syn::parse2::<syn::Block>(wrapped) {
        *traced_func.block = block;
    }

    let expanded = quote! {
        #[allow(non_camel_case_types)]
        #vis type #type_ident = #bare_fn;

        #[allow(non_upper_case_globals)]
        #vis static #static_ident: hipflex_internal::hooks::HookFn<#type_ident> =
            hipflex_internal::hooks::HookFn::default_const();

        #[allow(non_upper_case_globals)]
        #traced_func
    };

    TokenStream::from(expanded)
}
