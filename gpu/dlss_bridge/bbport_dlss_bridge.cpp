// SPDX-FileCopyrightText: Copyright 2026 IFreemz (shadps4_dlss bridge), bbport contributors
// SPDX-License-Identifier: MIT
//
// libbbport_dlss.so / bbport_dlss.dll: the only part of the project that uses the NVIDIA DLSS
// (NGX) SDK; the port loads it at run time. Linux: build.sh with DLSS_SDK_ROOT; Windows: MSVC
// (the SDK's Windows libraries are MSVC only), see CMakeLists.txt.
// Adapted from IFreemz/shadPS4-Bloodborne-DLSS-FSR (dlss_bridge).

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "bbport_dlss_bridge.h" // Vulkan first: the NGX headers expect its types

#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers_dlssg.h>
#include <nvsdk_ngx_helpers_dlssg_vk.h>

namespace {

constexpr char ProjectId[] = "42359704-c9f3-4806-9fb5-d469000fdb8a";
#ifdef _WIN32
constexpr char EngineVersion[] = "bbport-windows";
#else
constexpr char EngineVersion[] = "bbport-linux";
#endif

struct State {
    std::mutex ngx_mutex;
    BbDlssLogFn log{};
    std::vector<std::wstring> dll_directories;
    std::vector<const wchar_t*> dll_path_ptrs;
    std::wstring data_directory;
    NVSDK_NGX_FeatureCommonInfo common{};
    NVSDK_NGX_FeatureDiscoveryInfo discovery{};
    NVSDK_NGX_FeatureDiscoveryInfo fg_discovery{};
    std::vector<VkExtensionProperties> instance_extensions;
    std::vector<VkExtensionProperties> device_extensions;
    NVSDK_NGX_Parameter* capabilities{};
    NVSDK_NGX_Parameter* parameters{};
    NVSDK_NGX_Handle* feature{};
    BbDlssFeature feature_desc{};
    NVSDK_NGX_Parameter* fg_parameters{};
    NVSDK_NGX_Handle* fg_feature{};
    BbDlssFgFeature fg_feature_desc{};
    VkDevice device{};
    bool configured{};
    bool initialized{};
    bool fg_device_eligible{};
    bool fg_available{};
    uint32_t fg_max_multi_frame{};
} state;

void AppendUniqueExtensions(std::vector<VkExtensionProperties>& out, const VkExtensionProperties* list,
                            uint32_t count) {
    if (!list || count == 0)
        return;
    for (uint32_t i = 0; i < count; ++i) {
        const std::string_view name{list[i].extensionName};
        const bool exists = std::any_of(out.begin(), out.end(), [&](const VkExtensionProperties& e) {
            return std::string_view{e.extensionName} == name;
        });
        if (!exists) {
            out.push_back(list[i]);
        }
    }
}

template <typename... Args>
void Log(int warning, const char* format, Args... args) {
    if (!state.log)
        return;
    if constexpr (sizeof...(Args) == 0) {
        state.log(warning, format); // a plain message (no format string without arguments)
    } else {
        std::array<char, 1024> text{};
        std::snprintf(text.data(), text.size(), format, args...);
        state.log(warning, text.data());
    }
}

bool Check(const char* operation, NVSDK_NGX_Result result) {
    if (NVSDK_NGX_FAILED(result)) {
        Log(1, "%s failed: 0x%08x", operation, static_cast<unsigned>(result));
        return false;
    }
    return true;
}

void NVSDK_CONV NgxLog(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    Log(0, "NGX: %s", message ? message : "");
}

NVSDK_NGX_FeatureDiscoveryInfo MakeDiscovery(NVSDK_NGX_Feature feature_id) {
    NVSDK_NGX_FeatureDiscoveryInfo info{};
    info.SDKVersion = NVSDK_NGX_Version_API;
    info.FeatureID = feature_id;
    info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    info.Identifier.v.ProjectDesc = {ProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, EngineVersion};
    info.ApplicationDataPath = state.data_directory.c_str();
    info.FeatureInfo = &state.common;
    return info;
}

int32_t ConfigureEx(const wchar_t* const* dll_directories, uint32_t dll_directory_count,
                    const wchar_t* data_directory, BbDlssLogFn log) {
    std::lock_guard lock(state.ngx_mutex);
    state.log = log;
    state.dll_directories.clear();
    state.dll_path_ptrs.clear();
    if (dll_directories && dll_directory_count > 0) {
        for (uint32_t i = 0; i < dll_directory_count; ++i) {
            if (dll_directories[i] && dll_directories[i][0] != L'\0') {
                state.dll_directories.emplace_back(dll_directories[i]);
            }
        }
    }
    if (state.dll_directories.empty()) {
        state.dll_directories.emplace_back(L".");
    }
    for (const auto& dir : state.dll_directories) {
        state.dll_path_ptrs.push_back(dir.c_str());
    }
    state.data_directory = data_directory ? data_directory : L".";
    state.common.PathListInfo = {state.dll_path_ptrs.data(),
                                 static_cast<unsigned int>(state.dll_path_ptrs.size())};
    state.common.LoggingInfo = {NgxLog, NVSDK_NGX_LOGGING_LEVEL_OFF, false};
    state.discovery = MakeDiscovery(NVSDK_NGX_Feature_SuperSampling);
    state.fg_discovery = MakeDiscovery(NVSDK_NGX_Feature_FrameGeneration);
    state.instance_extensions.clear();
    state.device_extensions.clear();
    state.fg_device_eligible = false;
    state.fg_available = false;
    state.fg_max_multi_frame = 0;
    state.configured = true;
    return 1;
}

int32_t Configure(const wchar_t* dll_directory, const wchar_t* data_directory, BbDlssLogFn log) {
    const wchar_t* dir = dll_directory ? dll_directory : L".";
    return ConfigureEx(&dir, 1, data_directory, log);
}

int32_t InstanceExtensions(uint32_t* count, const VkExtensionProperties** extensions) {
    std::lock_guard lock(state.ngx_mutex);
    if (!state.configured || !count || !extensions)
        return 0;
    uint32_t sr_count{};
    VkExtensionProperties* sr_required{};
    if (!Check("Instance extension query",
               NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&state.discovery, &sr_count,
                                                                       &sr_required)))
        return 0;
    state.instance_extensions.clear();
    AppendUniqueExtensions(state.instance_extensions, sr_required, sr_count);

    // Non-fatal DLSS-G instance extension query so DLSS-SR remains active if DLSS-G is missing.
    uint32_t fg_count{};
    VkExtensionProperties* fg_required{};
    const auto fg_res = NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
        &state.fg_discovery, &fg_count, &fg_required);
    if (NVSDK_NGX_SUCCEED(fg_res)) {
        AppendUniqueExtensions(state.instance_extensions, fg_required, fg_count);
    } else {
        Log(0, "DLSS-G instance extension query skipped: 0x%08x", static_cast<unsigned>(fg_res));
    }

    *count = static_cast<uint32_t>(state.instance_extensions.size());
    *extensions = state.instance_extensions.data();
    return 1;
}

int32_t DeviceExtensions(VkInstance instance, VkPhysicalDevice physical, uint32_t* count,
                         const VkExtensionProperties** extensions) {
    std::lock_guard lock(state.ngx_mutex);
    if (!state.configured || !count || !extensions)
        return 0;
    NVSDK_NGX_FeatureRequirement support{};
    if (!Check("Feature requirement query",
               NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, physical, &state.discovery,
                                                       &support)))
        return 0;
    if (support.FeatureSupported != NVSDK_NGX_FeatureSupportResult_Supported) {
        Log(1, "DLSS is not supported on this GPU or driver (code 0x%x)",
            static_cast<unsigned>(support.FeatureSupported));
        return 0;
    }
    uint32_t sr_count{};
    VkExtensionProperties* sr_required{};
    if (!Check("Device extension query",
               NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
                   instance, physical, &state.discovery, &sr_count, &sr_required)))
        return 0;

    state.device_extensions.clear();
    AppendUniqueExtensions(state.device_extensions, sr_required, sr_count);

    // Query DLSS-G device requirements non-fatally so RTX 20/30 or non-FG setups keep DLSS-SR.
    state.fg_device_eligible = false;
    NVSDK_NGX_FeatureRequirement fg_support{};
    const auto fg_req_res =
        NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, physical, &state.fg_discovery,
                                                &fg_support);
    if (NVSDK_NGX_SUCCEED(fg_req_res) &&
        fg_support.FeatureSupported == NVSDK_NGX_FeatureSupportResult_Supported) {
        uint32_t fg_count{};
        VkExtensionProperties* fg_required{};
        const auto fg_ext_res = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
            instance, physical, &state.fg_discovery, &fg_count, &fg_required);
        if (NVSDK_NGX_SUCCEED(fg_ext_res)) {
            AppendUniqueExtensions(state.device_extensions, fg_required, fg_count);
            state.fg_device_eligible = true;
        } else {
            Log(1, "DLSS-G device extension query failed: 0x%08x",
                static_cast<unsigned>(fg_ext_res));
        }
    } else {
        Log(0, "DLSS-G unavailable on this GPU/driver (req 0x%08x, support 0x%x)",
            static_cast<unsigned>(fg_req_res), static_cast<unsigned>(fg_support.FeatureSupported));
    }

    *count = static_cast<uint32_t>(state.device_extensions.size());
    *extensions = state.device_extensions.data();
    return 1;
}

int32_t Initialize(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                   PFN_vkGetInstanceProcAddr get_instance_proc,
                   PFN_vkGetDeviceProcAddr get_device_proc) {
    std::lock_guard lock(state.ngx_mutex);
    if (!state.configured || state.initialized)
        return 0;
    if (!Check("NGX initialization",
               NVSDK_NGX_VULKAN_Init_with_ProjectID(
                   ProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, EngineVersion,
                   state.data_directory.c_str(), instance, physical, device, get_instance_proc,
                   get_device_proc, &state.common)))
        return 0;
    state.initialized = true;
    state.device = device;
    if (!Check("Capability query", NVSDK_NGX_VULKAN_GetCapabilityParameters(&state.capabilities)))
        return 0;
    int available{}, needs_driver{};
    NVSDK_NGX_Parameter_GetI(state.capabilities, NVSDK_NGX_Parameter_SuperSampling_Available,
                             &available);
    NVSDK_NGX_Parameter_GetI(state.capabilities,
                             NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needs_driver);
    if (needs_driver) {
        Log(1, "DLSS needs a newer NVIDIA driver");
        return 0;
    }
    if (!available) {
        Log(1, "DLSS Super Resolution is not available on this system");
        return 0;
    }

    state.fg_available = false;
    state.fg_max_multi_frame = 0;
    if (state.fg_device_eligible) {
        int fg_avail{}, fg_needs_driver{}, fg_init_result{};
        unsigned int fg_max_multi{};
        NVSDK_NGX_Parameter_GetI(state.capabilities, NVSDK_NGX_Parameter_FrameGeneration_Available,
                                 &fg_avail);
        NVSDK_NGX_Parameter_GetI(state.capabilities,
                                 NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver,
                                 &fg_needs_driver);
        NVSDK_NGX_Parameter_GetI(state.capabilities,
                                 NVSDK_NGX_Parameter_FrameGeneration_FeatureInitResult,
                                 &fg_init_result);
        NVSDK_NGX_Parameter_GetUI(state.capabilities, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax,
                                  &fg_max_multi);
        if (fg_needs_driver) {
            Log(1, "DLSS-G needs a newer NVIDIA driver");
        } else if (fg_avail) {
            state.fg_available = true;
            state.fg_max_multi_frame = std::max(1u, static_cast<uint32_t>(fg_max_multi));
        } else {
            Log(0, "DLSS-G capability disabled (available=%d, init=0x%08x)", fg_avail,
                static_cast<unsigned>(fg_init_result));
        }
    }
    return 1;
}

void ReleaseFeatureLocked() {
    if (state.feature) {
        NVSDK_NGX_VULKAN_ReleaseFeature(state.feature);
        state.feature = nullptr;
    }
    if (state.parameters) {
        NVSDK_NGX_VULKAN_DestroyParameters(state.parameters);
        state.parameters = nullptr;
    }
}

void ReleaseFeature() {
    std::lock_guard lock(state.ngx_mutex);
    ReleaseFeatureLocked();
}

void FgReleaseLocked() {
    if (state.fg_feature) {
        NVSDK_NGX_VULKAN_ReleaseFeature(state.fg_feature);
        state.fg_feature = nullptr;
    }
    if (state.fg_parameters) {
        NVSDK_NGX_VULKAN_DestroyParameters(state.fg_parameters);
        state.fg_parameters = nullptr;
    }
}

void FgRelease() {
    std::lock_guard lock(state.ngx_mutex);
    FgReleaseLocked();
}

int32_t CreateFeature(VkCommandBuffer command, const BbDlssFeature* desc) {
    std::lock_guard lock(state.ngx_mutex);
    if (!state.initialized || !state.capabilities || !desc)
        return 0;
    ReleaseFeatureLocked();
    if (!Check("Parameter allocation", NVSDK_NGX_VULKAN_AllocateParameters(&state.parameters)))
        return 0;
    for (const char* key : {NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality})
        NVSDK_NGX_Parameter_SetUI(state.parameters, key, desc->preset);
    static constexpr std::array qualities{
        NVSDK_NGX_PerfQuality_Value_DLAA, NVSDK_NGX_PerfQuality_Value_MaxQuality,
        NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf,
        NVSDK_NGX_PerfQuality_Value_UltraPerformance};
    NVSDK_NGX_DLSS_Create_Params settings{};
    settings.Feature.InWidth = desc->input_width;
    settings.Feature.InHeight = desc->input_height;
    settings.Feature.InTargetWidth = desc->output_width;
    settings.Feature.InTargetHeight = desc->output_height;
    settings.Feature.InPerfQualityValue = qualities[std::clamp(desc->quality, 0, 4)];
    settings.InFeatureCreateFlags =
        NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
        (desc->depth_inverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0) |
        (desc->hdr ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure
                   : 0);
    const auto result = NGX_VULKAN_CREATE_DLSS_EXT1(state.device, command, 1, 1, &state.feature,
                                                   state.parameters, &settings);
    if (!Check("DLSS feature creation", result) || !state.feature) {
        state.feature = nullptr;
        return 0;
    }
    state.feature_desc = *desc;
    Log(0, "DLSS %ux%u -> %ux%u, quality %d, preset %u%s", desc->input_width, desc->input_height,
        desc->output_width, desc->output_height, desc->quality, desc->preset,
        desc->hdr ? ", HDR input" : "");
    return 1;
}

NVSDK_NGX_Resource_VK Wrap(const BbDlssImage& image, bool writable) {
    return NVSDK_NGX_Create_ImageView_Resource_VK(image.view, image.image, image.range,
                                                  image.format, image.width, image.height,
                                                  writable);
}

int32_t Evaluate(VkCommandBuffer command, const BbDlssEvaluate* evaluate) {
    std::lock_guard lock(state.ngx_mutex);
    if (!state.feature || !state.parameters || !evaluate)
        return 0;
    auto color = Wrap(evaluate->color, false);
    auto depth = Wrap(evaluate->depth, false);
    auto motion = Wrap(evaluate->motion, false);
    auto output = Wrap(evaluate->output, true);
    NVSDK_NGX_VK_DLSS_Eval_Params parameters{};
    parameters.Feature.pInColor = &color;
    parameters.Feature.pInOutput = &output;
    parameters.Feature.InSharpness = evaluate->sharpness;
    parameters.pInDepth = &depth;
    parameters.pInMotionVectors = &motion;
    parameters.InRenderSubrectDimensions = {state.feature_desc.input_width,
                                            state.feature_desc.input_height};
    parameters.InJitterOffsetX = evaluate->jitter_x;
    parameters.InJitterOffsetY = evaluate->jitter_y;
    parameters.InReset = evaluate->reset;
    parameters.InMVScaleX = parameters.InMVScaleY = 1.0f;
    parameters.InPreExposure = parameters.InExposureScale = 1.0f;
    parameters.InFrameTimeDeltaInMsec = evaluate->frame_ms;
    return Check("DLSS evaluation", NGX_VULKAN_EVALUATE_DLSS_EXT(command, state.feature,
                                                                state.parameters, &parameters))
               ? 1
               : 0;
}

int32_t FgAvailable(uint32_t* out_max_multi_frame) {
    std::lock_guard lock(state.ngx_mutex);
    if (out_max_multi_frame) {
        *out_max_multi_frame = state.fg_available ? state.fg_max_multi_frame : 0u;
    }
    return state.fg_available ? 1 : 0;
}

int32_t FgCreate(VkCommandBuffer command, const BbDlssFgFeature* desc) {
    std::lock_guard lock(state.ngx_mutex);
    if (!state.initialized || !state.capabilities || !state.fg_available || !desc)
        return 0;
    FgReleaseLocked();
    if (!Check("DLSS-G parameter allocation",
               NVSDK_NGX_VULKAN_AllocateParameters(&state.fg_parameters)))
        return 0;

    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_Parameter_CreationNodeMask, 1);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_Parameter_VisibilityNodeMask, 1);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_Parameter_Width,
                              desc->backbuffer_width);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_Parameter_Height,
                              desc->backbuffer_height);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_Width,
                              desc->backbuffer_width);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_Height,
                              desc->backbuffer_height);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_BackbufferFormat,
                              static_cast<unsigned int>(desc->backbuffer_format));
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_InternalWidth,
                              desc->input_width);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_InternalHeight,
                              desc->input_height);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_DynamicResolution,
                              desc->dynamic_resolution ? 1u : 0u);

    const unsigned int always_flags =
        NVSDK_NGX_DLSSG_ResourceFlags_Backbuffer | NVSDK_NGX_DLSSG_ResourceFlags_MVecs |
        NVSDK_NGX_DLSSG_ResourceFlags_Depth | NVSDK_NGX_DLSSG_ResourceFlags_OutputInterpolated;
    const unsigned int never_flags =
        NVSDK_NGX_DLSSG_ResourceFlags_UI | NVSDK_NGX_DLSSG_ResourceFlags_UIAlpha |
        NVSDK_NGX_DLSSG_ResourceFlags_BidirectionalDistortionField |
        NVSDK_NGX_DLSSG_ResourceFlags_OutputReal |
        NVSDK_NGX_DLSSG_ResourceFlags_OutputDisableInterpolation;
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters,
                              NVSDK_NGX_DLSSG_Parameter_ResourceAlwaysProvided_Flags, always_flags);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters,
                              NVSDK_NGX_DLSSG_Parameter_ResourceNeverProvided_Flags, never_flags);

    const auto result = NVSDK_NGX_VULKAN_CreateFeature1(
        state.device, command, NVSDK_NGX_Feature_FrameGeneration, state.fg_parameters,
        &state.fg_feature);
    if (!Check("DLSS-G feature creation", result) || !state.fg_feature) {
        state.fg_feature = nullptr;
        return 0;
    }
    state.fg_feature_desc = *desc;
    Log(0, "DLSS-G %ux%u (internal %ux%u), fmt %u, max multi-frame %u%s", desc->backbuffer_width,
        desc->backbuffer_height, desc->input_width, desc->input_height,
        static_cast<unsigned>(desc->backbuffer_format), state.fg_max_multi_frame,
        desc->hdr ? ", HDR" : "");
    return 1;
}

void CopyMat4(float dst[4][4], const float src[16], bool identity_if_zero) {
    bool non_zero = false;
    for (int i = 0; i < 16; ++i) {
        if (src[i] != 0.0f) {
            non_zero = true;
            break;
        }
    }
    if (!non_zero && identity_if_zero) {
        std::memset(dst, 0, sizeof(float) * 16);
        dst[0][0] = dst[1][1] = dst[2][2] = dst[3][3] = 1.0f;
        return;
    }
    std::memcpy(dst, src, sizeof(float) * 16);
}

int32_t FgEvaluate(VkCommandBuffer command, const BbDlssFgEvaluate* evaluate) {
    std::lock_guard lock(state.ngx_mutex);
    if (!state.fg_feature || !state.fg_parameters || !evaluate)
        return 0;

    auto backbuffer = Wrap(evaluate->backbuffer, false);
    auto depth = Wrap(evaluate->depth, false);
    auto motion = Wrap(evaluate->motion, false);
    auto output = Wrap(evaluate->output, true);
    NVSDK_NGX_Resource_VK hudless{};
    const bool has_hudless = evaluate->hudless.image != VK_NULL_HANDLE;
    if (has_hudless) {
        hudless = Wrap(evaluate->hudless, false);
    }

    NVSDK_NGX_VK_DLSSG_Eval_Params eval{};
    eval.pBackbuffer = &backbuffer;
    eval.pDepth = &depth;
    eval.pMVecs = &motion;
    eval.pHudless = has_hudless ? &hudless : nullptr;
    eval.pOutputInterpFrame = &output;

    NVSDK_NGX_DLSSG_Opt_Eval_Params opt{};
    opt.multiFrameCount = std::max(1u, evaluate->multi_frame_count);
    opt.multiFrameIndex = std::clamp(evaluate->multi_frame_index, 1u, opt.multiFrameCount);
    CopyMat4(opt.cameraViewToClip, evaluate->camera_view_to_clip, true);
    CopyMat4(opt.clipToCameraView, evaluate->clip_to_camera_view, true);
    CopyMat4(opt.clipToLensClip, evaluate->clip_to_lens_clip, true);
    CopyMat4(opt.clipToPrevClip, evaluate->clip_to_prev_clip, true);
    CopyMat4(opt.prevClipToClip, evaluate->prev_clip_to_clip, true);

    opt.jitterOffset[0] = evaluate->jitter_x;
    opt.jitterOffset[1] = evaluate->jitter_y;
    opt.mvecScale[0] = evaluate->mv_scale_x;
    opt.mvecScale[1] = evaluate->mv_scale_y;
    for (int i = 0; i < 3; ++i) {
        opt.cameraPos[i] = evaluate->camera_pos[i];
        opt.cameraUp[i] = evaluate->camera_up[i];
        opt.cameraRight[i] = evaluate->camera_right[i];
        opt.cameraFwd[i] = evaluate->camera_fwd[i];
    }
    opt.cameraNear = evaluate->camera_near;
    opt.cameraFar = evaluate->camera_far;
    opt.cameraFOV = evaluate->camera_fov;
    opt.cameraAspectRatio = evaluate->camera_aspect;
    opt.colorBuffersHDR = (evaluate->hdr != 0);
    opt.depthInverted = (evaluate->depth_inverted != 0);
    opt.cameraMotionIncluded = (evaluate->camera_motion_included != 0);
    opt.reset = (evaluate->reset != 0);
    opt.menuDetectionEnabled = false;

    const uint32_t render_w = evaluate->render_subrect_width ? evaluate->render_subrect_width
                                                             : evaluate->motion.width;
    const uint32_t render_h = evaluate->render_subrect_height ? evaluate->render_subrect_height
                                                              : evaluate->motion.height;
    opt.mvecsSubrectSize = {render_w, render_h};
    opt.depthSubrectSize = {evaluate->depth.width, evaluate->depth.height};
    opt.backbufferSubrectSize = {evaluate->backbuffer.width, evaluate->backbuffer.height};
    opt.outputInterpSubrectSize = {evaluate->output.width, evaluate->output.height};
    if (has_hudless) {
        opt.hudLessSubrectSize = {evaluate->hudless.width, evaluate->hudless.height};
    }

    NVSDK_NGX_Parameter_SetULL(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_BackbufferFrameID,
                               static_cast<unsigned long long>(evaluate->frame_id));
    return Check("DLSS-G evaluation",
                 NGX_VK_EVALUATE_DLSSG(command, state.fg_feature, state.fg_parameters, &eval, &opt))
               ? 1
               : 0;
}

void Shutdown() {
    std::lock_guard lock(state.ngx_mutex);
    FgReleaseLocked();
    ReleaseFeatureLocked();
    if (state.capabilities) {
        NVSDK_NGX_VULKAN_DestroyParameters(state.capabilities);
        state.capabilities = nullptr;
    }
    if (state.initialized) {
        NVSDK_NGX_VULKAN_Shutdown1(state.device);
        state.initialized = false;
        state.device = VK_NULL_HANDLE;
    }
    state.fg_available = false;
    state.fg_device_eligible = false;
    state.fg_max_multi_frame = 0;
}

const BbDlssApi api{BBPORT_DLSS_BRIDGE_ABI,
                    Configure,
                    InstanceExtensions,
                    DeviceExtensions,
                    Initialize,
                    CreateFeature,
                    Evaluate,
                    ReleaseFeature,
                    Shutdown,
                    ConfigureEx,
                    FgAvailable,
                    FgCreate,
                    FgEvaluate,
                    FgRelease};

} // namespace

#ifdef _WIN32
#define BBPORT_DLSS_EXPORT __declspec(dllexport)
#else
#define BBPORT_DLSS_EXPORT __attribute__((visibility("default")))
#endif

extern "C" BBPORT_DLSS_EXPORT const BbDlssApi* BbDlssGetApi() {
    return &api;
}
