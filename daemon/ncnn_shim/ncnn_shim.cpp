/*
 * ncnn_shim.cpp — see ncnn_shim.h.
 *
 * Thin C++ shim over ncnn's C++ API exposing a C ABI for Rust. PHASE 1.
 *
 * Device-selection invariant: we NEVER silently use the discrete GPU. ncnn's
 * default GPU index on this box is the dGPU, so we always enumerate and pick
 * the integrated RADV PHOENIX device explicitly. If the only candidate is the
 * discrete GPU and the caller did not pass an explicit index, we refuse.
 */
#include "ncnn_shim.h"

#include "net.h"
#include "gpu.h"
#include "mat.h"

#include <string>
#include <cctype>
#include <cstring>
#include <mutex>

namespace {

std::mutex g_lock;
int g_instance_refs = 0;
// Index chosen by doorman_ncnn_select_device, consumed by net_create.
int g_selected_index = -1;

std::string to_upper(const char* s) {
    std::string out;
    if (!s) return out;
    for (const char* p = s; *p; ++p) out.push_back((char)std::toupper((unsigned char)*p));
    return out;
}

} // namespace

struct DoormanNcnnNet {
    ncnn::Net net;
    int device_index = -1;
};

extern "C" {

int doorman_ncnn_gpu_instance_create(void) {
    std::lock_guard<std::mutex> guard(g_lock);
    if (g_instance_refs == 0) {
        if (ncnn::create_gpu_instance() != 0) {
            return DOORMAN_NCNN_ERR_NO_VULKAN;
        }
    }
    if (ncnn::get_gpu_count() <= 0) {
        // Roll back the instance we may have just created.
        if (g_instance_refs == 0) ncnn::destroy_gpu_instance();
        return DOORMAN_NCNN_ERR_NO_VULKAN;
    }
    g_instance_refs += 1;
    return DOORMAN_NCNN_OK;
}

void doorman_ncnn_gpu_instance_destroy(void) {
    std::lock_guard<std::mutex> guard(g_lock);
    if (g_instance_refs > 0) {
        g_instance_refs -= 1;
        if (g_instance_refs == 0) {
            ncnn::destroy_gpu_instance();
            g_selected_index = -1;
        }
    }
}

int doorman_ncnn_gpu_count(void) {
    return ncnn::get_gpu_count();
}

int doorman_ncnn_gpu_info(int device_index, char* name_buf, size_t name_buf_len, int* out_type) {
    int cnt = ncnn::get_gpu_count();
    if (device_index < 0 || device_index >= cnt) return DOORMAN_NCNN_ERR_NO_DEVICE;
    const ncnn::GpuInfo& info = ncnn::get_gpu_info(device_index);
    if (name_buf && name_buf_len > 0) {
        const char* name = info.device_name();
        std::strncpy(name_buf, name ? name : "", name_buf_len - 1);
        name_buf[name_buf_len - 1] = '\0';
    }
    if (out_type) *out_type = info.type();
    return DOORMAN_NCNN_OK;
}

int doorman_ncnn_select_device(const char* name_substr, int index, int* out_index) {
    std::lock_guard<std::mutex> guard(g_lock);
    int cnt = ncnn::get_gpu_count();
    if (cnt <= 0) return DOORMAN_NCNN_ERR_NO_VULKAN;

    bool has_substr = (name_substr != nullptr && name_substr[0] != '\0');

    // 1) explicit name substring (case-insensitive) wins.
    if (has_substr) {
        std::string needle = to_upper(name_substr);
        for (int i = 0; i < cnt; ++i) {
            const ncnn::GpuInfo& info = ncnn::get_gpu_info(i);
            std::string hay = to_upper(info.device_name());
            if (hay.find(needle) != std::string::npos) {
                g_selected_index = i;
                if (out_index) *out_index = i;
                return DOORMAN_NCNN_OK;
            }
        }
        return DOORMAN_NCNN_ERR_NO_DEVICE;
    }

    // 2) explicit index — caller takes responsibility (allowed to be the dGPU).
    if (index >= 0) {
        if (index >= cnt) return DOORMAN_NCNN_ERR_NO_DEVICE;
        g_selected_index = index;
        if (out_index) *out_index = index;
        return DOORMAN_NCNN_OK;
    }

    // 3) default: first INTEGRATED device, NEVER the discrete one.
    int integrated = -1;
    bool saw_discrete = false;
    for (int i = 0; i < cnt; ++i) {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(i);
        int t = info.type();
        if (t == DOORMAN_NCNN_GPU_INTEGRATED && integrated < 0) integrated = i;
        if (t == DOORMAN_NCNN_GPU_DISCRETE) saw_discrete = true;
    }
    if (integrated >= 0) {
        g_selected_index = integrated;
        if (out_index) *out_index = integrated;
        return DOORMAN_NCNN_OK;
    }
    // No integrated device. Refuse to silently use the discrete GPU.
    if (saw_discrete) return DOORMAN_NCNN_ERR_REFUSE_DGPU;
    return DOORMAN_NCNN_ERR_NO_DEVICE;
}

int doorman_ncnn_net_create(const char* param_path, const char* bin_path, DoormanNcnnNet** out_net) {
    if (!param_path || !bin_path || !out_net) return DOORMAN_NCNN_ERR_ARG;
    int sel;
    {
        std::lock_guard<std::mutex> guard(g_lock);
        sel = g_selected_index;
    }
    if (sel < 0) return DOORMAN_NCNN_ERR_NO_DEVICE;

    DoormanNcnnNet* n = new DoormanNcnnNet();
    n->device_index = sel;
    n->net.opt.use_vulkan_compute = true;
    n->net.set_vulkan_device(sel);

    if (n->net.load_param(param_path) != 0) {
        delete n;
        return DOORMAN_NCNN_ERR_LOAD_PARAM;
    }
    if (n->net.load_model(bin_path) != 0) {
        delete n;
        return DOORMAN_NCNN_ERR_LOAD_BIN;
    }
    *out_net = n;
    return DOORMAN_NCNN_OK;
}

int doorman_ncnn_net_infer(DoormanNcnnNet* net,
                           const char* input_name,
                           const float* data, int c, int h, int w,
                           const char* output_name,
                           float* out_data, size_t out_cap,
                           int* out_c, int* out_h, int* out_w, size_t* out_len) {
    if (!net || !input_name || !output_name || !data) return DOORMAN_NCNN_ERR_ARG;
    if (c <= 0 || h <= 0 || w <= 0) return DOORMAN_NCNN_ERR_ARG;

    // Build a CHW ncnn::Mat from the contiguous caller buffer (respect cstep
    // padding by copying channel-by-channel).
    ncnn::Mat in(w, h, c);
    if (in.empty()) return DOORMAN_NCNN_ERR_ALLOC;
    const size_t plane = (size_t)w * (size_t)h;
    for (int ch = 0; ch < c; ++ch) {
        float* dst = in.channel(ch);
        const float* src = data + (size_t)ch * plane;
        std::memcpy(dst, src, plane * sizeof(float));
    }

    ncnn::Extractor ex = net->net.create_extractor();
    if (ex.input(input_name, in) != 0) return DOORMAN_NCNN_ERR_INPUT;

    ncnn::Mat out;
    if (ex.extract(output_name, out) != 0) return DOORMAN_NCNN_ERR_EXTRACT;

    int oc = out.c;
    int oh = out.h;
    int ow = out.w;
    // Normalize degenerate dims (ncnn collapses to dims<3).
    if (out.dims == 1) { oc = 1; oh = 1; ow = out.w; }
    else if (out.dims == 2) { oc = 1; oh = out.h; ow = out.w; }

    if (out_c) *out_c = oc;
    if (out_h) *out_h = oh;
    if (out_w) *out_w = ow;

    size_t total = (size_t)oc * (size_t)oh * (size_t)ow;
    if (out_len) *out_len = total;

    if (!out_data || out_cap < total) {
        return DOORMAN_NCNN_ERR_ALLOC;
    }

    // Copy out channel-by-channel into contiguous CHW (drop cstep padding).
    const size_t oplane = (size_t)oh * (size_t)ow;
    if (out.dims >= 3) {
        for (int ch = 0; ch < oc; ++ch) {
            const float* src = out.channel(ch);
            std::memcpy(out_data + (size_t)ch * oplane, src, oplane * sizeof(float));
        }
    } else {
        std::memcpy(out_data, (const float*)out.data, total * sizeof(float));
    }
    return DOORMAN_NCNN_OK;
}

void doorman_ncnn_net_destroy(DoormanNcnnNet* net) {
    if (!net) return;
    // Extractors are stack-scoped in infer(); destroying the Net here releases
    // its Vulkan resources. The gpu instance is torn down separately by the
    // Rust wrapper AFTER all nets are dropped (teardown order: extractor ->
    // Net -> gpu instance).
    net->net.clear();
    delete net;
}

} // extern "C"
