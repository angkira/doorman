//! Build script for `doormand`.
//!
//! Almost entirely a no-op: the daemon's default/ORT builds need no native
//! compilation here. The ONE thing it does is compile the ncnn C++ FFI shim
//! and emit the libncnn (+ Vulkan/glslang) link flags — but ONLY when the
//! `backend-ncnn` feature is enabled. Every other feature build (backend-ort,
//! backend-ort-rocm, etc.) skips this code path entirely, so the shim cannot
//! perturb them.
//!
//! libncnn linking source (PHASE 1): a locally-built static ncnn with
//! `-DNCNN_VULKAN=ON`. The prefix is configurable via the `DOORMAN_NCNN_DIR`
//! env var; it defaults to `/tmp/ncnn_install` (where this phase's build was
//! installed). Distribution/vendoring of libncnn is PHASE 3.

fn main() {
    // Re-run if the feature toggles or the shim sources change.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DOORMAN_NCNN_DIR");

    #[cfg(feature = "backend-ncnn")]
    build_ncnn_shim();
}

#[cfg(feature = "backend-ncnn")]
fn build_ncnn_shim() {
    use std::path::PathBuf;

    println!("cargo:rerun-if-changed=ncnn_shim/ncnn_shim.cpp");
    println!("cargo:rerun-if-changed=ncnn_shim/ncnn_shim.h");

    // ncnn install prefix (static lib + headers), built with NCNN_VULKAN=ON.
    let ncnn_dir = std::env::var("DOORMAN_NCNN_DIR")
        .unwrap_or_else(|_| "/tmp/ncnn_install".to_string());
    let ncnn_dir = PathBuf::from(ncnn_dir);
    let ncnn_inc = ncnn_dir.join("include").join("ncnn");
    let ncnn_lib = ncnn_dir.join("lib");

    if !ncnn_inc.join("net.h").exists() {
        panic!(
            "backend-ncnn: ncnn headers not found at {} (set DOORMAN_NCNN_DIR to a \
             static ncnn install built with -DNCNN_VULKAN=ON)",
            ncnn_inc.display()
        );
    }

    // Compile the C++ shim against the ncnn headers.
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("ncnn_shim/ncnn_shim.cpp")
        .include("ncnn_shim")
        .include(&ncnn_inc)
        .flag_if_supported("-fopenmp")
        .warnings(false)
        .compile("doorman_ncnn_shim");

    // Link search path for the static ncnn + glslang archives.
    println!("cargo:rustc-link-search=native={}", ncnn_lib.display());

    // ncnn (static) and its glslang dependencies (modern unified glslang build
    // keeps everything in libglslang.a; the others are harmless stubs).
    println!("cargo:rustc-link-lib=static=ncnn");
    println!("cargo:rustc-link-lib=static=glslang");
    println!("cargo:rustc-link-lib=static=SPIRV");
    println!("cargo:rustc-link-lib=static=MachineIndependent");
    println!("cargo:rustc-link-lib=static=GenericCodeGen");
    println!("cargo:rustc-link-lib=static=OSDependent");
    println!("cargo:rustc-link-lib=static=glslang-default-resource-limits");

    // Vulkan is dlopen'd by ncnn at runtime, but link it for symbol resolution
    // robustness, plus OpenMP/pthread/dl/stdc++ that ncnn needs.
    println!("cargo:rustc-link-lib=dylib=vulkan");
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
}
