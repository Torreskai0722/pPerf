// my_cudnn_profile.cpp

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cudnn/Handle.h>          // at::native::getCudnnHandle()
#include <cudnn.h>
#include <cudnn_backend.h>
#include <nvToolsExt.h>

#include <stdexcept>
#include <vector>
#include <chrono>
#include <limits>                       // std::numeric_limits
#include <algorithm>                    // std::min_element, std::max_element
#include <string>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

static void checkCudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " + cudnnGetErrorString(status));
    }
}

// Add this helper to see exactly where it dies
static void checkCudnnBackend(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "CUDNN ERROR: %s (Status %d)\n", msg, (int)status);
        throw std::runtime_error(std::string("[backend] ") + msg + ": " + cudnnGetErrorString(status));
    }
}

struct BackendDesc {
    cudnnBackendDescriptor_t desc{nullptr};

    BackendDesc() = default;

    explicit BackendDesc(cudnnBackendDescriptorType_t type) {
        checkCudnnBackend(cudnnBackendCreateDescriptor(type, &desc), "create backend desc");
    }

    ~BackendDesc() {
        if (desc) cudnnBackendDestroyDescriptor(desc);
    }

    BackendDesc(const BackendDesc&) = delete;
    BackendDesc& operator=(const BackendDesc&) = delete;

    BackendDesc(BackendDesc&& other) noexcept : desc(other.desc) {
        other.desc = nullptr;
    }
    BackendDesc& operator=(BackendDesc&& other) noexcept {
        if (this != &other) {
            if (desc) cudnnBackendDestroyDescriptor(desc);
            desc = other.desc;
            other.desc = nullptr;
        }
        return *this;
    }

    operator cudnnBackendDescriptor_t() const { return desc; }
};

///////////////////////////////////////////////////////////////
// BACKEND HELPERS
///////////////////////////////////////////////////////////////

static BackendDesc create_backend_tensor_4d(int64_t n, int64_t c, int64_t h, int64_t w, int64_t uid) {
    BackendDesc t(CUDNN_BACKEND_TENSOR_DESCRIPTOR);

    int64_t dims[4]    = {n, c, h, w};
    int64_t strides[4] = {c * h * w, h * w, w, 1};
    int64_t alignment  = 16; // Mandatory for many Backend Engines
    cudnnDataType_t dt = CUDNN_DATA_FLOAT;

    checkCudnnBackend(cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt), "set dt");
    checkCudnnBackend(cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims), "set dims");
    checkCudnnBackend(cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides), "set strides");
    checkCudnnBackend(cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid), "set uid");
    checkCudnnBackend(cudnnBackendSetAttribute(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment), "set align");

    checkCudnnBackend(cudnnBackendFinalize(t), "finalize tensor");
    return t;
}

// Build a backend operation graph for a single conv forward (no bias)
static BackendDesc build_conv_op_graph(
    cudnnHandle_t handle,
    int64_t n, int64_t c_in, int64_t h_in, int64_t w_in,
    int64_t c_out, int64_t kH, int64_t kW,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dil_h, int dil_w,
    int groups
) {

    // Tensors
    auto xTensor = create_backend_tensor_4d(n, c_in, h_in, w_in, 0);
    auto yTensor = create_backend_tensor_4d(
        n,
        c_out,
        (h_in + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1,
        (w_in + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1,
        1);

    auto wTensor = create_backend_tensor_4d(c_out, c_in / groups, kH, kW, 2);

    // Convolution forward op
    BackendDesc convOp(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);

    {
        // Conv descriptor (backend)
        BackendDesc convDesc(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);

        cudnnDataType_t compType = CUDNN_DATA_FLOAT;
        checkCudnnBackend(
            cudnnBackendSetAttribute(
                convDesc,
                CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                CUDNN_TYPE_DATA_TYPE,
                1,
                &compType),
            "set conv comp type");

        int64_t pads[2]      = {pad_h, pad_w};
        int64_t strides[2]   = {stride_h, stride_w};
        int64_t dilations[2] = {dil_h, dil_w};

        int64_t spatial_dims = 2;
        checkCudnnBackend(
            cudnnBackendSetAttribute(
                convDesc,
                CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
                CUDNN_TYPE_INT64,
                1,
                &spatial_dims),
            "set conv spatial dims");

        checkCudnnBackend(
            cudnnBackendSetAttribute(
                convDesc,
                CUDNN_ATTR_CONVOLUTION_DILATIONS,
                CUDNN_TYPE_INT64,
                2,
                dilations),
            "set conv dilation");

        checkCudnnBackend(
            cudnnBackendSetAttribute(
                convDesc,
                CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                CUDNN_TYPE_INT64,
                2,
                strides),
            "set conv strides");

        checkCudnnBackend(
            cudnnBackendSetAttribute(
                convDesc,
                CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                CUDNN_TYPE_INT64,
                2,
                pads),
            "set conv pre-padding");

        checkCudnnBackend(
            cudnnBackendSetAttribute(
                convDesc,
                CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                CUDNN_TYPE_INT64,
                2,
                pads),
            "set conv post-padding");

        cudnnConvolutionMode_t conv_mode = CUDNN_CROSS_CORRELATION;
        checkCudnnBackend(
            cudnnBackendSetAttribute(
                convDesc,
                CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                CUDNN_TYPE_CONVOLUTION_MODE,
                1,
                &conv_mode),
            "set conv mode");

        checkCudnnBackend(cudnnBackendFinalize(convDesc), "finalize convDesc");

        // Plug tensors + convDesc into the convOp
        cudnnBackendDescriptor_t xDesc = xTensor;
        cudnnBackendDescriptor_t wDesc = wTensor;
        cudnnBackendDescriptor_t yDesc = yTensor;
        cudnnBackendDescriptor_t cDesc = convDesc;

        checkCudnnBackend(
            cudnnBackendSetAttribute(
                convOp,
                CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &xDesc),
            "set convOp X");

        checkCudnnBackend(
            cudnnBackendSetAttribute(
                convOp,
                CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &wDesc),
            "set convOp W");

        checkCudnnBackend(
            cudnnBackendSetAttribute(
                convOp,
                CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &yDesc),
            "set convOp Y");

        checkCudnnBackend(
            cudnnBackendSetAttribute(
                convOp,
                CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &cDesc),
            "set convOp convDesc");

        // Set ALPHA (critical for output scaling!)
        float alpha = 1.0f;
        checkCudnnBackend(
            cudnnBackendSetAttribute(
                convOp,
                CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                CUDNN_TYPE_FLOAT,
                1,
                &alpha),
            "set convOp ALPHA");

        // Set BETA (0.0 means don't accumulate into existing output)
        float beta = 0.0f;
        checkCudnnBackend(
            cudnnBackendSetAttribute(
                convOp,
                CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                CUDNN_TYPE_FLOAT,
                1,
                &beta),
            "set convOp BETA");

        checkCudnnBackend(cudnnBackendFinalize(convOp), "finalize convOp");
    }

    // Operation graph with single op
    BackendDesc opGraph(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    {
        cudnnBackendDescriptor_t ops[1] = { convOp };

        // Your header supports OPERATIONGRAPH_HANDLE; set it.
        checkCudnnBackend(
            cudnnBackendSetAttribute(
                opGraph,
                CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                CUDNN_TYPE_HANDLE,
                1,
                &handle),
            "set opGraph handle");

        checkCudnnBackend(
            cudnnBackendSetAttribute(
                opGraph,
                CUDNN_ATTR_OPERATIONGRAPH_OPS,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                ops),
            "set opGraph ops");

        // DO NOT set OP_COUNT (not present in your cuDNN header)

        checkCudnnBackend(cudnnBackendFinalize(opGraph), "finalize opGraph");
    }

    return opGraph;
}

///////////////////////////////////////////////////////////////
// MAIN PROFILING FUNCTION
///////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor>
profile_conv2d_variants(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups,
    int64_t max_engine_variants,
    int64_t iters,
    const std::string& module_name
) {
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    CHECK_FLOAT(input);
    CHECK_FLOAT(weight);
    TORCH_CHECK(stride.size() == 2, "stride must be [h,w]");
    TORCH_CHECK(padding.size() == 2, "padding must be [h,w]");
    TORCH_CHECK(dilation.size() == 2, "dilation must be [h,w]");

    auto bias = bias_opt.has_value() ? bias_opt.value() : torch::Tensor();
    if (bias_opt.has_value()) {
        CHECK_CUDA(bias);
        CHECK_CONTIGUOUS(bias);
        CHECK_FLOAT(bias);
    }

    auto opts = input.options();

    int64_t N     = input.size(0);
    int64_t C_in  = input.size(1);
    int64_t H_in  = input.size(2);
    int64_t W_in  = input.size(3);
    int64_t C_out = weight.size(0);
    int64_t kH    = weight.size(2);
    int64_t kW    = weight.size(3);

    int64_t stride_h = stride[0];
    int64_t stride_w = stride[1];
    int64_t pad_h    = padding[0];
    int64_t pad_w    = padding[1];
    int64_t dil_h    = dilation[0];
    int64_t dil_w    = dilation[1];

    int64_t H_out = (H_in + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    int64_t W_out = (W_in + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;

    auto output = torch::empty({N, C_out, H_out, W_out}, opts);

    // cuDNN handle / stream from PyTorch
    cudnnHandle_t handle = at::native::getCudnnHandle();
    auto stream = at::cuda::getCurrentCUDAStream();
    checkCudnn(cudnnSetStream(handle, stream.stream()), "set cudnn stream");

    ///////////////////////////////////////////////////
    // ENGINE-CONFIG-LEVEL PROFILING (backend API)
    ///////////////////////////////////////////////////

    auto opGraph = build_conv_op_graph(
        handle,
        N, C_in, H_in, W_in,
        C_out, kH, kW,
        (int)pad_h, (int)pad_w,
        (int)stride_h, (int)stride_w,
        (int)dil_h, (int)dil_w,
        (int)groups);

    BackendDesc engineHeur(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);

    {
        cudnnBackendDescriptor_t gDesc = opGraph;
        checkCudnnBackend(
            cudnnBackendSetAttribute(
                engineHeur,
                CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &gDesc),
            "set ENGINEHEUR op graph");

        cudnnBackendHeurMode_t mode = CUDNN_HEUR_MODE_INSTANT;
        checkCudnnBackend(
            cudnnBackendSetAttribute(
                engineHeur,
                CUDNN_ATTR_ENGINEHEUR_MODE,
                CUDNN_TYPE_HEUR_MODE,
                1,
                &mode),
            "set ENGINEHEUR mode");

        checkCudnnBackend(cudnnBackendFinalize(engineHeur), "finalize ENGINEHEUR");
    }

    // 1. Get the count
    int64_t numCfgs = 0;
    checkCudnnBackend(cudnnBackendGetAttribute(engineHeur, CUDNN_ATTR_ENGINEHEUR_RESULTS, 
                    CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &numCfgs, nullptr), "get count");

    // 2. Create the managed objects
    // This allocates the descriptors and ensures they are destroyed when this vector goes out of scope
    std::vector<BackendDesc> engineCfgs_managed(numCfgs);
    for (int i = 0; i < numCfgs; ++i) {
        engineCfgs_managed[i] = BackendDesc(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    }

    // 3. Create a raw pointer array to pass to the API
    std::vector<cudnnBackendDescriptor_t> engineCfgs_raw(numCfgs);
    for (int i = 0; i < numCfgs; ++i) {
        engineCfgs_raw[i] = engineCfgs_managed[i].desc;
    }

    // 4. Fill the descriptors
    int64_t actualNumCfgs = 0;
    checkCudnnBackend(
        cudnnBackendGetAttribute(
            engineHeur,
            CUDNN_ATTR_ENGINEHEUR_RESULTS,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            numCfgs,
            &actualNumCfgs,
            engineCfgs_raw.data()), // This is what the API needs
        "get ENGINEHEUR results");

    int64_t useCfgs = std::min<int64_t>(numCfgs, max_engine_variants);

    auto engineTable = torch::zeros({useCfgs, 2}, torch::kFloat32);
    float* raw_data_ptr = engineTable.data_ptr<float>(); // Get the direct address

    for (int64_t i = 0; i < useCfgs; ++i) {
        cudnnBackendDescriptor_t ec = engineCfgs_raw[i];
    
        BackendDesc plan(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        
        // 1. Link the Engine Config
        checkCudnnBackend(
            cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ec),
            "set plan engine cfg");
    
        // 2. Link the Handle (Required for plan compilation/finalization)
        checkCudnnBackend(
            cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle),
            "set plan handle");
    
        checkCudnnBackend(cudnnBackendFinalize(plan), "finalize exec plan");

        // 3. Query required workspace
        int64_t workspaceSize = 0;
        checkCudnnBackend(
            cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, nullptr, &workspaceSize),
            "get plan workspace size");

        // Safely allocate workspace (only if > 0)
        at::Tensor wsTensor;
        void* wsPtr = nullptr;
        if (workspaceSize > 0) {
            wsTensor = at::empty({workspaceSize}, input.options().dtype(at::kByte));
            wsPtr = wsTensor.data_ptr();
        }

        // 4. Set up Variant Pack
        // IMPORTANT: Keep these arrays in the same scope as the Execute call!
        void* data_ptrs[3] = {input.data_ptr(), weight.data_ptr(), output.data_ptr()};
        int64_t uids[3] = {0, 2, 1}; // Ensure these match the UIDs in build_conv_op_graph

        BackendDesc variantPack(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        
        checkCudnnBackend(
            cudnnBackendSetAttribute(variantPack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, data_ptrs),
            "set variantPack data ptrs");

        checkCudnnBackend(
            cudnnBackendSetAttribute(variantPack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids),
            "set variantPack uids");

        checkCudnnBackend(
            cudnnBackendSetAttribute(variantPack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &wsPtr),
            "set variantPack workspace");

        cudnnStatus_t status = cudnnBackendFinalize(variantPack);
        
        if (status == CUDNN_STATUS_NOT_SUPPORTED) {
            // Log the skip and move to the next config
            fprintf(stderr, "Skipping engine config %ld: Not supported on this hardware/layout.\n", i);
            continue; 
        } else if (status != CUDNN_STATUS_SUCCESS) {
            // Real error
            checkCudnnBackend(status, "finalize variantPack");
        }

        // 4. Single Execution Profiling
        cudaDeviceSynchronize(); // Clear previous work

        // Wrap each engine config with NVTX range using module_name
        std::string nvtx_label = module_name + " | engine_config_" + std::to_string(i);
        nvtxRangePushA(nvtx_label.c_str());

        auto t0 = std::chrono::high_resolution_clock::now();

        checkCudnnBackend(cudnnBackendExecute(handle, plan, variantPack), "backend execute");

        // FORCE THE CPU TO WAIT AND CHECK FOR ERRORS HERE
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            exit(1);
        }

        cudaDeviceSynchronize(); // Wait for kernel to finish
        nvtxRangePop();

        auto t1 = std::chrono::high_resolution_clock::now();

        float duration_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // 5. Store results
        raw_data_ptr[i * 2 + 0] = duration_ms;
        raw_data_ptr[i * 2 + 1] = static_cast<float>(workspaceSize);
    }

    return std::make_tuple(output, engineTable);
}

///////////////////////////////////////////////////////////////
// ENGINE SELECTION FUNCTION
///////////////////////////////////////////////////////////////

torch::Tensor
execute_conv2d_with_engine(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    c10::optional<torch::Tensor> bias_opt,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups,
    int64_t engine_index
) {
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    CHECK_FLOAT(input);
    CHECK_FLOAT(weight);
    TORCH_CHECK(stride.size() == 2, "stride must be [h,w]");
    TORCH_CHECK(padding.size() == 2, "padding must be [h,w]");
    TORCH_CHECK(dilation.size() == 2, "dilation must be [h,w]");

    auto bias = bias_opt.has_value() ? bias_opt.value() : torch::Tensor();
    if (bias_opt.has_value()) {
        CHECK_CUDA(bias);
        CHECK_CONTIGUOUS(bias);
        CHECK_FLOAT(bias);
    }

    auto opts = input.options();

    int64_t N     = input.size(0);
    int64_t C_in  = input.size(1);
    int64_t H_in  = input.size(2);
    int64_t W_in  = input.size(3);
    int64_t C_out = weight.size(0);
    int64_t kH    = weight.size(2);
    int64_t kW    = weight.size(3);

    int64_t stride_h = stride[0];
    int64_t stride_w = stride[1];
    int64_t pad_h    = padding[0];
    int64_t pad_w    = padding[1];
    int64_t dil_h    = dilation[0];
    int64_t dil_w    = dilation[1];

    int64_t H_out = (H_in + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
    int64_t W_out = (W_in + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;

    auto output = torch::empty({N, C_out, H_out, W_out}, opts);

    // cuDNN handle / stream from PyTorch
    cudnnHandle_t handle = at::native::getCudnnHandle();
    auto stream = at::cuda::getCurrentCUDAStream();
    checkCudnn(cudnnSetStream(handle, stream.stream()), "set cudnn stream");

    ///////////////////////////////////////////////////
    // BUILD OPERATION GRAPH
    ///////////////////////////////////////////////////

    auto opGraph = build_conv_op_graph(
        handle,
        N, C_in, H_in, W_in,
        C_out, kH, kW,
        (int)pad_h, (int)pad_w,
        (int)stride_h, (int)stride_w,
        (int)dil_h, (int)dil_w,
        (int)groups);

    BackendDesc engineHeur(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);

    {
        cudnnBackendDescriptor_t gDesc = opGraph;
        checkCudnnBackend(
            cudnnBackendSetAttribute(
                engineHeur,
                CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                CUDNN_TYPE_BACKEND_DESCRIPTOR,
                1,
                &gDesc),
            "set ENGINEHEUR op graph");

        cudnnBackendHeurMode_t mode = CUDNN_HEUR_MODE_INSTANT;
        checkCudnnBackend(
            cudnnBackendSetAttribute(
                engineHeur,
                CUDNN_ATTR_ENGINEHEUR_MODE,
                CUDNN_TYPE_HEUR_MODE,
                1,
                &mode),
            "set ENGINEHEUR mode");

        checkCudnnBackend(cudnnBackendFinalize(engineHeur), "finalize ENGINEHEUR");
    }

    // Get the count of available engine configs
    int64_t numCfgs = 0;
    checkCudnnBackend(cudnnBackendGetAttribute(engineHeur, CUDNN_ATTR_ENGINEHEUR_RESULTS, 
                    CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &numCfgs, nullptr), "get count");

    TORCH_CHECK(engine_index >= 0 && engine_index < numCfgs, 
                "engine_index ", engine_index, " out of range [0, ", numCfgs, ")");

    // Only create descriptors up to the one we need
    int64_t numCfgsToFetch = engine_index + 1;
    
    // Create managed engine config descriptors (only what we need)
    std::vector<BackendDesc> engineCfgs_managed(numCfgsToFetch);
    for (int i = 0; i < numCfgsToFetch; ++i) {
        engineCfgs_managed[i] = BackendDesc(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    }

    // Create raw pointer array for the API
    std::vector<cudnnBackendDescriptor_t> engineCfgs_raw(numCfgsToFetch);
    for (int i = 0; i < numCfgsToFetch; ++i) {
        engineCfgs_raw[i] = engineCfgs_managed[i].desc;
    }

    // Fill only the descriptors we need
    int64_t actualNumCfgs = 0;
    checkCudnnBackend(
        cudnnBackendGetAttribute(
            engineHeur,
            CUDNN_ATTR_ENGINEHEUR_RESULTS,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            numCfgsToFetch,
            &actualNumCfgs,
            engineCfgs_raw.data()),
        "get ENGINEHEUR results");

    ///////////////////////////////////////////////////
    // SELECT AND EXECUTE SPECIFIC ENGINE CONFIG
    ///////////////////////////////////////////////////

    cudnnBackendDescriptor_t ec = engineCfgs_raw[engine_index];

    BackendDesc plan(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    
    // Link the Engine Config
    checkCudnnBackend(
        cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ec),
        "set plan engine cfg");

    // Link the Handle
    checkCudnnBackend(
        cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle),
        "set plan handle");

    checkCudnnBackend(cudnnBackendFinalize(plan), "finalize exec plan");

    // Query required workspace
    int64_t workspaceSize = 0;
    checkCudnnBackend(
        cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, nullptr, &workspaceSize),
        "get plan workspace size");

    // Allocate workspace if needed
    at::Tensor wsTensor;
    void* wsPtr = nullptr;
    if (workspaceSize > 0) {
        wsTensor = at::empty({workspaceSize}, input.options().dtype(at::kByte));
        wsPtr = wsTensor.data_ptr();
    }

    // Set up Variant Pack
    void* data_ptrs[3] = {input.data_ptr(), weight.data_ptr(), output.data_ptr()};
    int64_t uids[3] = {0, 2, 1}; // Match UIDs in build_conv_op_graph

    BackendDesc variantPack(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    
    checkCudnnBackend(
        cudnnBackendSetAttribute(variantPack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, data_ptrs),
        "set variantPack data ptrs");

    checkCudnnBackend(
        cudnnBackendSetAttribute(variantPack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids),
        "set variantPack uids");

    checkCudnnBackend(
        cudnnBackendSetAttribute(variantPack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &wsPtr),
        "set variantPack workspace");

    cudnnStatus_t status = cudnnBackendFinalize(variantPack);
    
    if (status == CUDNN_STATUS_NOT_SUPPORTED) {
        throw std::runtime_error("Selected engine config " + std::to_string(engine_index) + " is not supported on this hardware/layout.");
    } else if (status != CUDNN_STATUS_SUCCESS) {
        checkCudnnBackend(status, "finalize variantPack");
    }

    // Execute the selected engine config
    checkCudnnBackend(cudnnBackendExecute(handle, plan, variantPack), "backend execute");

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    return output;
}

///////////////////////////////////////////////////////////////
// PYTORCH BINDINGS
///////////////////////////////////////////////////////////////

TORCH_LIBRARY(LUT_perf, m) {
    m.def("profile_conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, int max_engine_variants, int iters, str module_name) -> (Tensor, Tensor)");
    m.def("execute_conv2d_with_engine(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, int engine_index) -> Tensor");
}

TORCH_LIBRARY_IMPL(LUT_perf, CUDA, m) {
    m.impl("profile_conv2d", &profile_conv2d_variants);
    m.impl("execute_conv2d_with_engine", &execute_conv2d_with_engine);
}
