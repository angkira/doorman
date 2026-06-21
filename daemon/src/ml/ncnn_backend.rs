//! Safe Rust wrapper over the ncnn C-FFI shim (`ncnn_shim/ncnn_shim.{h,cpp}`).
//!
//! PHASE 1 (foundation): prove we can load an ncnn `.param`/`.bin` model and run
//! inference on the integrated Radeon 780M (RADV PHOENIX) via Vulkan — never the
//! discrete dGPU — and that outputs match onnxruntime-CPU.
//!
//! This module is gated entirely on `feature = "backend-ncnn"`. Wiring it into
//! the `MLBackend` trait is PHASE 2; here we only expose `NcnnNet` + device
//! enumeration/selection so an example/test can exercise the FFI directly.
//!
//! Teardown order (enforced by `Drop`): each `NcnnNet` is destroyed first, and
//! the process-global Vulkan gpu instance is released by a refcounted
//! `GpuInstance` guard that every `NcnnNet` holds — so the gpu instance is only
//! torn down after the last net is dropped.

#![allow(dead_code)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};

// ---- raw FFI -----------------------------------------------------------------

#[allow(non_camel_case_types)]
type c_size = usize;

#[repr(C)]
struct DoormanNcnnNet {
    _private: [u8; 0],
}

extern "C" {
    fn doorman_ncnn_gpu_instance_create() -> c_int;
    fn doorman_ncnn_gpu_instance_destroy();
    fn doorman_ncnn_gpu_count() -> c_int;
    fn doorman_ncnn_gpu_info(
        device_index: c_int,
        name_buf: *mut c_char,
        name_buf_len: c_size,
        out_type: *mut c_int,
    ) -> c_int;
    fn doorman_ncnn_select_device(
        name_substr: *const c_char,
        index: c_int,
        out_index: *mut c_int,
    ) -> c_int;
    fn doorman_ncnn_net_create(
        param_path: *const c_char,
        bin_path: *const c_char,
        out_net: *mut *mut DoormanNcnnNet,
    ) -> c_int;
    fn doorman_ncnn_net_infer(
        net: *mut DoormanNcnnNet,
        input_name: *const c_char,
        data: *const f32,
        c: c_int,
        h: c_int,
        w: c_int,
        output_name: *const c_char,
        out_data: *mut f32,
        out_cap: c_size,
        out_c: *mut c_int,
        out_h: *mut c_int,
        out_w: *mut c_int,
        out_len: *mut c_size,
    ) -> c_int;
    fn doorman_ncnn_net_destroy(net: *mut DoormanNcnnNet);
}

// Mirror of the shim error codes / device kinds.
const OK: c_int = 0;

/// Vulkan device kind (mirror of ncnn `GpuInfo::type()`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuKind {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Unknown(i32),
}

impl From<c_int> for GpuKind {
    fn from(t: c_int) -> Self {
        match t {
            0 => GpuKind::Discrete,
            1 => GpuKind::Integrated,
            2 => GpuKind::Virtual,
            3 => GpuKind::Cpu,
            other => GpuKind::Unknown(other),
        }
    }
}

/// A described Vulkan device.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub index: i32,
    pub name: String,
    pub kind: GpuKind,
}

fn err_str(code: c_int) -> &'static str {
    match code {
        -1 => "no Vulkan device / gpu instance failed",
        -2 => "requested device not found",
        -3 => "refused: only the discrete GPU matched (use an explicit index to override)",
        -4 => "load_param failed",
        -5 => "load_model (.bin) failed",
        -6 => "extractor.input failed",
        -7 => "extractor.extract failed",
        -8 => "bad argument",
        -9 => "allocation/capacity error",
        _ => "unknown ncnn shim error",
    }
}

// ---- gpu instance (refcounted RAII guard) ------------------------------------

/// RAII guard over the process-global ncnn Vulkan gpu instance. Cloning bumps
/// the shim's internal refcount; dropping releases it. The instance is only
/// destroyed when the last guard (and thus the last `NcnnNet`) is dropped,
/// guaranteeing the teardown order Net -> gpu instance.
#[derive(Clone)]
pub struct GpuInstance {
    inner: Arc<GpuInstanceInner>,
}

struct GpuInstanceInner;

impl GpuInstance {
    /// Create (or attach to) the global Vulkan gpu instance.
    pub fn create() -> Result<Self> {
        let rc = unsafe { doorman_ncnn_gpu_instance_create() };
        if rc != OK {
            bail!("ncnn gpu instance: {}", err_str(rc));
        }
        Ok(GpuInstance {
            inner: Arc::new(GpuInstanceInner),
        })
    }

    /// Number of Vulkan devices ncnn can see.
    pub fn device_count(&self) -> i32 {
        unsafe { doorman_ncnn_gpu_count() }
    }

    /// Enumerate all Vulkan devices.
    pub fn devices(&self) -> Result<Vec<GpuDevice>> {
        let n = self.device_count();
        let mut out = Vec::with_capacity(n.max(0) as usize);
        for i in 0..n {
            let mut buf = [0i8; 256];
            let mut kind: c_int = -1;
            let rc = unsafe {
                doorman_ncnn_gpu_info(i, buf.as_mut_ptr() as *mut c_char, buf.len(), &mut kind)
            };
            if rc != OK {
                bail!("ncnn gpu_info({i}): {}", err_str(rc));
            }
            let name = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) }
                .to_string_lossy()
                .into_owned();
            out.push(GpuDevice {
                index: i,
                name,
                kind: GpuKind::from(kind),
            });
        }
        Ok(out)
    }
}

impl Drop for GpuInstanceInner {
    fn drop(&mut self) {
        unsafe { doorman_ncnn_gpu_instance_destroy() };
    }
}

// ---- device selector ---------------------------------------------------------

/// How to pick the Vulkan device. The default is the first INTEGRATED device,
/// and the shim REFUSES to fall back to the discrete GPU unless an explicit
/// index is given.
#[derive(Debug, Clone)]
pub enum DeviceSelector {
    /// First device whose name contains this substring (case-insensitive),
    /// e.g. "RADV PHOENIX".
    NameSubstring(String),
    /// Exactly this device index (caller accepts responsibility, e.g. dGPU).
    Index(i32),
    /// First integrated device; never the discrete one.
    DefaultIntegrated,
}

impl Default for DeviceSelector {
    fn default() -> Self {
        DeviceSelector::DefaultIntegrated
    }
}

/// Resolve a selector to a concrete device index via the shim (which also
/// records the selection for the next `NcnnNet::new`). Returns the chosen index.
fn select_device(inst: &GpuInstance, sel: &DeviceSelector) -> Result<i32> {
    let _ = inst; // selection runs against the global instance the guard holds.
    let mut out_index: c_int = -1;
    let rc = match sel {
        DeviceSelector::NameSubstring(s) => {
            let cs = CString::new(s.as_str()).context("device name has interior NUL")?;
            unsafe { doorman_ncnn_select_device(cs.as_ptr(), -1, &mut out_index) }
        }
        DeviceSelector::Index(i) => unsafe {
            doorman_ncnn_select_device(std::ptr::null(), *i, &mut out_index)
        },
        DeviceSelector::DefaultIntegrated => unsafe {
            doorman_ncnn_select_device(std::ptr::null(), -1, &mut out_index)
        },
    };
    if rc != OK {
        bail!("ncnn device selection: {}", err_str(rc));
    }
    Ok(out_index)
}

// ---- net ---------------------------------------------------------------------

/// Output of one inference: a map of output-blob name to (data, [c, h, w]).
pub type InferOutput = std::collections::HashMap<String, (Vec<f32>, [i32; 3])>;

/// A safe handle to an ncnn `Net` bound to a selected Vulkan device.
pub struct NcnnNet {
    handle: *mut DoormanNcnnNet,
    /// Keeps the gpu instance alive until this net is dropped (teardown order).
    _instance: GpuInstance,
    device_index: i32,
    device_name: String,
}

// The ncnn Net + extractor calls are not internally synchronized; we expose a
// single-threaded wrapper. `NcnnNet` owns its handle exclusively. It is `Send`
// (can move to another thread) but not `Sync`.
unsafe impl Send for NcnnNet {}

impl NcnnNet {
    /// Load a `.param`/`.bin` model and bind it to the device chosen by `sel`.
    ///
    /// Creates/attaches the global Vulkan gpu instance, resolves the selector
    /// (refusing the dGPU unless an explicit index was given), then loads.
    pub fn new(
        param_path: impl AsRef<Path>,
        bin_path: impl AsRef<Path>,
        sel: DeviceSelector,
    ) -> Result<Self> {
        let instance = GpuInstance::create()?;
        let device_index = select_device(&instance, &sel)?;

        // Record the chosen device name for diagnostics / proof.
        let device_name = instance
            .devices()?
            .into_iter()
            .find(|d| d.index == device_index)
            .map(|d| d.name)
            .unwrap_or_else(|| format!("vulkan{device_index}"));

        let param = path_cstring(param_path.as_ref())?;
        let bin = path_cstring(bin_path.as_ref())?;

        let mut handle: *mut DoormanNcnnNet = std::ptr::null_mut();
        let rc = unsafe { doorman_ncnn_net_create(param.as_ptr(), bin.as_ptr(), &mut handle) };
        if rc != OK || handle.is_null() {
            bail!("ncnn net_create: {}", err_str(rc));
        }

        Ok(NcnnNet {
            handle,
            _instance: instance,
            device_index,
            device_name,
        })
    }

    /// The Vulkan device index this net runs on.
    pub fn device_index(&self) -> i32 {
        self.device_index
    }

    /// The Vulkan device name this net runs on (e.g. "AMD Radeon Graphics (RADV PHOENIX)").
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Run one forward pass: feed `data` (CHW f32, shape `[c,h,w]`) into the
    /// `input_name` blob and extract `output_name`. Returns the output as a
    /// single-entry map keyed by `output_name`.
    pub fn infer(
        &self,
        input_name: &str,
        data: &[f32],
        shape: [i32; 3],
        output_name: &str,
    ) -> Result<InferOutput> {
        let [c, h, w] = shape;
        let expected = (c as i64 * h as i64 * w as i64).max(0) as usize;
        if data.len() != expected {
            bail!(
                "input buffer len {} != c*h*w = {} for shape {:?}",
                data.len(),
                expected,
                shape
            );
        }

        let in_name = CString::new(input_name).context("input_name has NUL")?;
        let out_name = CString::new(output_name).context("output_name has NUL")?;

        // First call with a null buffer to learn the output shape.
        let mut oc: c_int = 0;
        let mut oh: c_int = 0;
        let mut ow: c_int = 0;
        let mut olen: c_size = 0;
        let rc = unsafe {
            doorman_ncnn_net_infer(
                self.handle,
                in_name.as_ptr(),
                data.as_ptr(),
                c,
                h,
                w,
                out_name.as_ptr(),
                std::ptr::null_mut(),
                0,
                &mut oc,
                &mut oh,
                &mut ow,
                &mut olen,
            )
        };
        // -9 (ALLOC) is expected here because we passed a null/zero buffer; the
        // shape out-params are still valid. Any other non-OK is a real failure.
        if rc != OK && rc != -9 {
            bail!("ncnn infer (shape probe): {}", err_str(rc));
        }
        if olen == 0 {
            bail!("ncnn infer produced empty output for blob '{output_name}'");
        }

        // Second call with a correctly-sized buffer.
        let mut out_data = vec![0.0f32; olen];
        let mut oc2: c_int = 0;
        let mut oh2: c_int = 0;
        let mut ow2: c_int = 0;
        let mut olen2: c_size = 0;
        let rc = unsafe {
            doorman_ncnn_net_infer(
                self.handle,
                in_name.as_ptr(),
                data.as_ptr(),
                c,
                h,
                w,
                out_name.as_ptr(),
                out_data.as_mut_ptr(),
                out_data.len(),
                &mut oc2,
                &mut oh2,
                &mut ow2,
                &mut olen2,
            )
        };
        if rc != OK {
            bail!("ncnn infer: {}", err_str(rc));
        }
        out_data.truncate(olen2);

        let mut map = InferOutput::new();
        map.insert(output_name.to_string(), (out_data, [oc2, oh2, ow2]));
        Ok(map)
    }
}

impl Drop for NcnnNet {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { doorman_ncnn_net_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
        // `_instance` drops AFTER this body -> gpu instance released last.
    }
}

fn path_cstring(p: &Path) -> Result<CString> {
    let s = p
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF8 path: {}", p.display()))?;
    CString::new(s).map_err(|_| anyhow!("path has interior NUL: {}", p.display()))
}
