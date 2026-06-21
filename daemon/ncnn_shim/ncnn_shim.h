/*
 * ncnn_shim.h — minimal C ABI over ncnn + Vulkan for doorman.
 *
 * PHASE 1 (foundation): load an ncnn .param/.bin model and run inference on a
 * caller-selected Vulkan device — specifically the integrated Radeon 780M
 * (RADV PHOENIX), NEVER the discrete dGPU. The shim enumerates the Vulkan
 * devices, refuses to silently fall back to the discrete GPU, and enforces the
 * ncnn teardown order (extractor -> Net -> gpu instance).
 *
 * All functions are C-linkage so Rust can bind them via a thin `extern "C"`
 * block. Strings are NUL-terminated UTF-8. Return codes: 0 = OK, negative =
 * error (see DOORMAN_NCNN_ERR_*). Out-params are only written on success.
 *
 * This file is additive and only compiled under the `backend-ncnn` cargo
 * feature; it does not touch the ORT backends.
 */
#ifndef DOORMAN_NCNN_SHIM_H
#define DOORMAN_NCNN_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- error codes ------------------------------------------------------- */
#define DOORMAN_NCNN_OK 0
#define DOORMAN_NCNN_ERR_NO_VULKAN -1     /* gpu instance / no Vulkan device   */
#define DOORMAN_NCNN_ERR_NO_DEVICE -2     /* requested device not found        */
#define DOORMAN_NCNN_ERR_REFUSE_DGPU -3   /* only the discrete GPU matched     */
#define DOORMAN_NCNN_ERR_LOAD_PARAM -4    /* load_param failed                 */
#define DOORMAN_NCNN_ERR_LOAD_BIN -5      /* load_model (.bin) failed          */
#define DOORMAN_NCNN_ERR_INPUT -6         /* extractor.input failed            */
#define DOORMAN_NCNN_ERR_EXTRACT -7       /* extractor.extract failed          */
#define DOORMAN_NCNN_ERR_ARG -8           /* bad argument                      */
#define DOORMAN_NCNN_ERR_ALLOC -9         /* allocation / capacity             */

/* Opaque handles. */
typedef struct DoormanNcnnNet DoormanNcnnNet;

/* Vulkan device kinds (mirror of ncnn GpuInfo::type()). */
#define DOORMAN_NCNN_GPU_DISCRETE 0
#define DOORMAN_NCNN_GPU_INTEGRATED 1
#define DOORMAN_NCNN_GPU_VIRTUAL 2
#define DOORMAN_NCNN_GPU_CPU 3

/*
 * Initialize the global Vulkan gpu instance (idempotent / refcounted).
 * Returns 0 on success, DOORMAN_NCNN_ERR_NO_VULKAN if no Vulkan device exists.
 */
int doorman_ncnn_gpu_instance_create(void);

/* Tear down the global gpu instance. Refcounted against create(). */
void doorman_ncnn_gpu_instance_destroy(void);

/* Number of Vulkan devices ncnn can see (after gpu_instance_create). */
int doorman_ncnn_gpu_count(void);

/*
 * Describe a Vulkan device by index. `name_buf` receives the NUL-terminated
 * device name (truncated to name_buf_len). `out_type` receives the
 * DOORMAN_NCNN_GPU_* kind. Returns 0 on success.
 */
int doorman_ncnn_gpu_info(int device_index, char* name_buf, size_t name_buf_len, int* out_type);

/*
 * Select the target Vulkan device for inference.
 *
 *   name_substr : if non-NULL/non-empty, pick the first device whose name
 *                 contains this substring (case-insensitive), e.g. "RADV PHOENIX".
 *   index       : if >= 0 AND name_substr is NULL/empty, pick exactly this index.
 *   both unset  : pick the first INTEGRATED device.
 *
 * Guarantees: if the only match is a DISCRETE gpu and the caller did NOT pass
 * an explicit index, returns DOORMAN_NCNN_ERR_REFUSE_DGPU and selects nothing.
 * On success writes the chosen index to *out_index.
 */
int doorman_ncnn_select_device(const char* name_substr, int index, int* out_index);

/*
 * Create an ncnn Net bound to the device chosen by doorman_ncnn_select_device,
 * with use_vulkan_compute = true, and load param + bin from the given paths.
 * On success *out_net receives the handle (caller frees with destroy).
 */
int doorman_ncnn_net_create(const char* param_path, const char* bin_path, DoormanNcnnNet** out_net);

/*
 * Run one forward pass.
 *   input_name  : input blob name (e.g. "in0")
 *   data        : CHW f32 buffer of length c*h*w
 *   c,h,w       : input shape
 *   output_name : output blob to extract (e.g. "out0")
 *   out_data    : caller buffer for the f32 result (CHW, channel-contiguous)
 *   out_cap     : capacity of out_data in float elements
 *   out_c,h,w   : receive the output shape (any may be NULL)
 *   out_len     : receives number of f32 written (= oc*oh*ow)
 * Returns 0 on success. If out_data is NULL or out_cap is too small, the
 * shape out-params are still filled and DOORMAN_NCNN_ERR_ALLOC is returned so
 * the caller can size and retry.
 */
int doorman_ncnn_net_infer(DoormanNcnnNet* net,
                           const char* input_name,
                           const float* data, int c, int h, int w,
                           const char* output_name,
                           float* out_data, size_t out_cap,
                           int* out_c, int* out_h, int* out_w, size_t* out_len);

/* Destroy a Net (in correct order: any internal extractor state, then Net). */
void doorman_ncnn_net_destroy(DoormanNcnnNet* net);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DOORMAN_NCNN_SHIM_H */
