use std::env;
use std::path::PathBuf;

fn main() {
    // Get Python from PYO3_PYTHON or VIRTUAL_ENV
    let python = env::var("PYO3_PYTHON")
        .or_else(|_| {
            env::var("VIRTUAL_ENV").map(|venv| format!("{}/bin/python3", venv))
        })
        .unwrap_or_else(|_| "python3".to_string());
    
    // Get onnxruntime library path from Python
    let output = std::process::Command::new(&python)
        .args(&["-c", "import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))"])
        .output()
        .expect("Failed to run Python to find onnxruntime");
    
    if output.status.success() {
        let ort_dir = String::from_utf8(output.stdout)
            .expect("Invalid UTF-8 from Python")
            .trim()
            .to_string();
        
        let ort_lib_dir = format!("{}/capi", ort_dir);
        
        // Tell cargo to look for libraries in this path
        println!("cargo:rustc-link-search=native={}", ort_lib_dir);
        
        // Set rpath so the library can be found at runtime
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", ort_lib_dir);
        
        // Tell ort crate where to find libonnxruntime.so
        println!("cargo:rustc-env=ORT_DYLIB_PATH={}/libonnxruntime.so", ort_lib_dir);
        
        eprintln!("Build: Found onnxruntime at {}", ort_lib_dir);
    } else {
        eprintln!("Warning: Could not find onnxruntime library path");
        eprintln!("Build may fail or runtime may not find libonnxruntime.so");
    }
    
    // Rerun if environment changes
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
}
