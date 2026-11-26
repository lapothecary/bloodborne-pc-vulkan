// SPDX-FileCopyrightText: Copyright 2026 IFreemz, bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>

#include "bbport_settings.h"
#include "bbport_system_info.h"
#include "common/thread.h"
#include "upscale/vk_dlss.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "../../../dlss_bridge/bbport_dlss_bridge.h"
#elif defined(__linux__)
#include <dlfcn.h>
#include "../../../dlss_bridge/bbport_dlss_bridge.h"
#endif

namespace Vulkan {

int Dlss::QualityForScale(float scale) {
    // DLAA only for equal sizes: any real upscale needs at least the Quality mode.
    return scale >= 2.9f ? 4 : scale >= 1.95f ? 3 : scale >= 1.65f ? 2 : scale > 1.01f ? 1 : 0;
}

#if defined(_WIN32) || defined(__linux__)

namespace {
#ifdef _WIN32
constexpr wchar_t BridgeName[] = L"bbport_dlss.dll";
constexpr wchar_t NgxName[] = L"nvngx_dlss.dll";
using Module = HMODULE;

std::filesystem::path ExecutableDirectory() {
    std::wstring path(MAX_PATH, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    path.resize(length);
    return std::filesystem::path{path}.parent_path();
}

bool HasNgx(const std::filesystem::path& directory) {
    return std::filesystem::is_regular_file(directory / NgxName);
}

Module LoadModule(const std::filesystem::path& path) {
    return LoadLibraryW(path.c_str());
}

void* FindSymbol(Module module, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(module, name));
}
#else
// bbport (Linux): libbbport_dlss.so (gpu/dlss_bridge) and NVIDIA's libnvidia-ngx-dlss.so.<version>
// next to bb-probe; NGX finds the latter through the bridge's path list.
constexpr char BridgeName[] = "libbbport_dlss.so";
using Module = void*;

std::filesystem::path ExecutableDirectory() {
    std::error_code error;
    return std::filesystem::read_symlink("/proc/self/exe", error).parent_path();
}

bool HasNgx(const std::filesystem::path& directory) {
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
        if (entry.path().filename().string().starts_with("libnvidia-ngx-dlss.so.")) {
            return true;
        }
    }
    return false;
}

Module LoadModule(const std::filesystem::path& path) {
    Module module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module) {
        std::printf("DLSS: %s\n", dlerror());
    }
    return module;
}

void* FindSymbol(Module module, const char* name) {
    return dlsym(module, name);
}
#endif

/// The player's DLSS folder (<user>/dlss, BB_GPU_USER_DIR; beside bb-probe without it): NGX's
/// logs, and NVIDIA's library when the player chose one in the launcher (newer than the package's,
/// or the only one: the package may come without it).
std::filesystem::path UserDlssDirectory() {
    if (const char* user = std::getenv("BB_GPU_USER_DIR"); user && user[0]) {
        std::error_code error;
        const auto absolute = std::filesystem::absolute(std::filesystem::path{user} / "dlss", error);
        return error ? std::filesystem::path{user} / "dlss" : absolute;
    }
    return ExecutableDirectory() / "dlss";
}

/// Where NVIDIA's DLSS library is: the player's own first, then beside bb-probe; empty: nowhere.
std::filesystem::path NgxDirectory() {
    for (const auto& directory : {UserDlssDirectory(), ExecutableDirectory()}) {
        if (HasNgx(directory)) {
            return directory;
        }
    }
    return {};
}

/// Where the bridge is: beside bb-probe (the package, a build with DLSS_SDK_ROOT), else the
/// player's DLSS folder; empty: nowhere.
std::filesystem::path BridgeDirectory() {
    for (const auto& directory : {ExecutableDirectory(), UserDlssDirectory()}) {
        std::error_code error;
        if (std::filesystem::is_regular_file(directory / BridgeName, error)) {
            return directory;
        }
    }
    return {};
}

void BridgeLog(int warning, const char* message) {
    std::printf("DLSS: %s%s\n", warning ? "warning: " : "", message);
}

bool CollectExtensions(const VkExtensionProperties* required, u32 count,
                       const std::vector<vk::ExtensionProperties>& available,
                       std::vector<std::string>& storage, std::vector<const char*>& enabled) {
    storage.clear();
    for (u32 i = 0; i < count; ++i) {
        const auto& request = required[i];
        const auto match = std::ranges::find_if(available, [&](const auto& extension) {
            return std::strcmp(extension.extensionName.data(), request.extensionName) == 0 &&
                   extension.specVersion >= request.specVersion;
        });
        if (match == available.end()) {
            std::printf("DLSS: missing Vulkan extension %s\n", request.extensionName);
            return false;
        }
        storage.emplace_back(request.extensionName);
    }
    for (const auto& name : storage) {
        if (std::ranges::none_of(enabled, [&](const char* current) { return name == current; })) {
            enabled.push_back(name.c_str());
        }
    }
    return true;
}

BbDlssImage ToDlssImage(const Dlss::Resource& r) {
    if (!r.image) {
        return {};
    }
    const vk::ImageSubresourceRange range{r.aspect, 0, 1, 0, 1};
    return BbDlssImage{r.image,
                       r.view,
                       static_cast<VkImageSubresourceRange>(range),
                       static_cast<VkFormat>(r.format),
                       r.width,
                       r.height};
}

} // namespace

struct Dlss::Impl {
    Module module{};
    const BbDlssApi* api{};
    std::vector<std::string> instance_extensions, device_extensions;
    std::optional<FeatureDesc> feature;
    std::optional<FgFeatureDesc> fg_feature;
    std::string problem, fg_problem;
    bool eligible{true}, instance_ready{}, device_ready{}, initialized{}, available{};
    bool fg_available{};
    u32 fg_max_multi_frame{};
    mutable std::mutex init_mutex;
    std::atomic<bool> init_done{false};
    std::jthread init_thread;

    void WaitInit() const {
        if (init_done.load(std::memory_order_acquire)) [[likely]] {
            return;
        }
        std::scoped_lock lk{init_mutex};
        if (init_thread.joinable()) {
            const_cast<std::jthread&>(init_thread).join();
        }
    }

    void Disable(std::string_view reason) {
        eligible = available = fg_available = false;
        fg_max_multi_frame = 0;
        problem = reason;
        fg_problem = reason;
        std::printf("DLSS: unavailable (%.*s)\n", int(reason.size()), reason.data());
    }
};

Dlss* Dlss::Get() {
    static Dlss* const dlss = []() -> Dlss* {
        const char* setting = std::getenv("BB_DLSS");
        if (setting && setting[0] == '0') {
            return nullptr;
        }
        if (BridgeDirectory().empty()) {
            return nullptr;
        }
        auto* created = new Dlss;
        // The process may end through quick_exit/_exit paths: release NGX first when it can.
        std::atexit([] {
            if (Dlss* live = Get()) {
                live->Shutdown();
            }
        });
        return created;
    }();
    return dlss;
}

Dlss::Dlss() : impl{std::make_unique<Impl>()} {
    impl->module = LoadModule(BridgeDirectory() / BridgeName);
    const auto get_api =
        impl->module ? reinterpret_cast<BbDlssGetApiFn>(FindSymbol(impl->module, "BbDlssGetApi"))
                     : nullptr;
    impl->api = get_api ? get_api() : nullptr;
    if (!impl->api || impl->api->abi < 1 || impl->api->abi > BBPORT_DLSS_BRIDGE_ABI) {
        impl->Disable("the DLSS bridge library is missing or from another version");
        return;
    }
    const auto ngx = NgxDirectory();
    if (ngx.empty()) {
        impl->Disable("NVIDIA's DLSS library is missing: choose it in the launcher (Upscaler)");
        return;
    }
    // NGX writes its logs and model updates here: beside the saves and shader caches.
    const auto data = UserDlssDirectory();
    std::error_code error;
    std::filesystem::create_directories(data, error);
    std::printf("DLSS: NVIDIA's library from %s\n", BbSysInfo::RedactHomePath(ngx.string()).c_str());
    int32_t configured = 0;
    if (impl->api->abi >= 2 && impl->api->ConfigureEx) {
        std::vector<std::wstring> search_dirs;
        for (const auto& candidate : {UserDlssDirectory(), ExecutableDirectory()}) {
            std::error_code ec;
            if (std::filesystem::is_directory(candidate, ec)) {
                const auto w = candidate.wstring();
                if (std::ranges::find(search_dirs, w) == search_dirs.end()) {
                    search_dirs.push_back(w);
                }
            }
        }
        std::vector<const wchar_t*> ptrs;
        ptrs.reserve(search_dirs.size());
        for (const auto& d : search_dirs) {
            ptrs.push_back(d.c_str());
        }
        configured = impl->api->ConfigureEx(ptrs.data(), static_cast<uint32_t>(ptrs.size()),
                                            data.wstring().c_str(), BridgeLog);
    } else {
        configured = impl->api->Configure(ngx.wstring().c_str(), data.wstring().c_str(), BridgeLog);
    }
    if (!configured) {
        impl->Disable("bridge configuration failed");
    }
}

Dlss::~Dlss() {
    Shutdown();
}

void Dlss::AppendInstanceExtensions(std::vector<const char*>& enabled) {
    if (!impl->eligible) {
        return;
    }
    u32 count{};
    const VkExtensionProperties* required{};
    if (!impl->api->InstanceExtensions(&count, &required)) {
        return impl->Disable("instance extension query failed");
    }
    const auto [result, extensions] = vk::enumerateInstanceExtensionProperties();
    if (result != vk::Result::eSuccess ||
        !CollectExtensions(required, count, extensions, impl->instance_extensions, enabled)) {
        return impl->Disable("required Vulkan instance extensions are missing");
    }
    impl->instance_ready = true;
}

void Dlss::AppendDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                                  std::vector<const char*>& enabled) {
    if (!impl->eligible || !impl->instance_ready) {
        return;
    }
    const auto properties = physical.getProperties();
    if (properties.vendorID != 0x10de) {
        return impl->Disable("not an NVIDIA GPU");
    }
    u32 count{};
    const VkExtensionProperties* required{};
    if (!impl->api->DeviceExtensions(instance, physical, &count, &required)) {
        return impl->Disable("this GPU or driver does not support DLSS (GeForce RTX needed)");
    }
    const auto [result, extensions] = physical.enumerateDeviceExtensionProperties();
    if (result != vk::Result::eSuccess ||
        !CollectExtensions(required, count, extensions, impl->device_extensions, enabled)) {
        return impl->Disable("required Vulkan device extensions are missing");
    }
    impl->device_ready = true;
}

void Dlss::Disable(std::string_view reason) {
    impl->WaitInit();
    impl->Disable(reason);
}

void Dlss::Initialize(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device) {
    if (!impl->eligible || !impl->device_ready || impl->initialized) {
        impl->init_done.store(true, std::memory_order_release);
        if (!impl->eligible || !impl->device_ready) {
            BbSettings::ConfigureDlssgSupport(
                false, 0,
                impl->problem.empty() ? "DLSS-G requires NVIDIA RTX 40+ GPU" : impl->problem.c_str());
        }
        return;
    }
    impl->initialized = true;
    const auto run_init = [p = impl.get(), instance, physical, device]() {
        Common::SetCurrentThreadName("bb:DlssInit");
        if (!p->api->Initialize(instance, physical, device,
                                VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
                                VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr)) {
            p->Disable("NVIDIA NGX initialization failed (update the GPU driver)");
            BbSettings::ConfigureDlssSupport(false, p->problem.c_str());
            BbSettings::ConfigureDlssgSupport(false, 0, p->problem.c_str());
        } else {
            p->available = true;
            std::printf("DLSS: ready\n");
            BbSettings::ConfigureDlssSupport(true, "");
            if (p->api->abi >= 2 && p->api->FgAvailable) {
                uint32_t max_multi = 0;
                if (p->api->FgAvailable(&max_multi)) {
                    p->fg_available = true;
                    p->fg_max_multi_frame = std::max(1u, max_multi);
                    p->fg_problem.clear();
                    std::printf("DLSS-G: ready (max %ux)\n", p->fg_max_multi_frame + 1u);
                    BbSettings::ConfigureDlssgSupport(true, static_cast<int>(p->fg_max_multi_frame),
                                                      "");
                } else {
                    p->fg_problem =
                        "DLSS-G requires NVIDIA RTX 40+ GPU, driver support, and DLSS-G runtime";
                    BbSettings::ConfigureDlssgSupport(false, 0, p->fg_problem.c_str());
                }
            } else {
                p->fg_problem = "DLSS bridge ABI 1 does not support DLSS-G";
                BbSettings::ConfigureDlssgSupport(false, 0, p->fg_problem.c_str());
            }
        }
        p->init_done.store(true, std::memory_order_release);
    };
    const char* sync_env = std::getenv("BB_DLSS_SYNC");
    if (sync_env && sync_env[0] == '1') {
        run_init();
        return;
    }
    impl->init_thread = std::jthread(run_init);
}

bool Dlss::EligibleForBoot() const noexcept {
    return impl->eligible && impl->device_ready;
}

bool Dlss::Available() const {
    impl->WaitInit();
    return impl->available;
}

bool Dlss::FgAvailable() const {
    impl->WaitInit();
    return impl->fg_available;
}

u32 Dlss::FgMaxMultiFrame() const {
    impl->WaitInit();
    return impl->fg_available ? impl->fg_max_multi_frame : 0u;
}

std::string Dlss::Problem() const {
    impl->WaitInit();
    return impl->problem;
}

std::string Dlss::FgProblem() const {
    impl->WaitInit();
    return impl->fg_problem;
}

bool Dlss::HasFeature(const FeatureDesc& desc) const {
    return impl->feature && *impl->feature == desc;
}

bool Dlss::CreateFeature(vk::CommandBuffer command, const FeatureDesc& desc) {
    impl->WaitInit();
    if (!impl->available) {
        return false;
    }
    impl->feature.reset();
    BbDlssFeature feature{desc.input_width,
                          desc.input_height,
                          desc.output_width,
                          desc.output_height,
                          desc.quality,
                          0,
                          static_cast<uint32_t>(std::max(0, desc.preset)),
                          desc.hdr ? 1 : 0};
    if (!impl->api->CreateFeature(command, &feature)) {
        if (desc.preset != 0) {
            feature.preset = 0;
            if (!impl->api->CreateFeature(command, &feature)) {
                return false;
            }
        } else {
            return false;
        }
    }
    impl->feature = desc;
    std::printf("Upscaler: DLSS feature %ux%u -> %ux%u (quality %d, preset %d %s)\n",
                desc.input_width, desc.input_height, desc.output_width, desc.output_height,
                desc.quality, feature.preset,
                feature.preset >= 10 ? "DLSS 4 Transformer" : "DLSS 3 CNN");
    return true;
}

bool Dlss::Evaluate(vk::CommandBuffer command, const Frame& frame) {
    if (!impl->available || !impl->feature) {
        return false;
    }
    const BbDlssEvaluate parameters{
        ToDlssImage(frame.color),  ToDlssImage(frame.depth), ToDlssImage(frame.motion),
        ToDlssImage(frame.output), frame.jitter_x,           frame.jitter_y,
        frame.reset ? 1 : 0,       frame.frame_ms,           frame.sharpness};
    return impl->api->Evaluate(command, &parameters) != 0;
}

void Dlss::ReleaseFeature() {
    impl->feature.reset();
    if (impl->api) {
        impl->api->ReleaseFeature();
    }
}

bool Dlss::HasFgFeature(const FgFeatureDesc& desc) const {
    return impl->fg_feature && *impl->fg_feature == desc;
}

bool Dlss::CreateFgFeature(vk::CommandBuffer command, const FgFeatureDesc& desc) {
    impl->WaitInit();
    if (!impl->fg_available || !impl->api || impl->api->abi < 2 || !impl->api->FgCreate) {
        return false;
    }
    impl->fg_feature.reset();
    const BbDlssFgFeature feature{desc.input_width,
                                  desc.input_height,
                                  desc.backbuffer_width,
                                  desc.backbuffer_height,
                                  static_cast<VkFormat>(desc.backbuffer_format),
                                  desc.dynamic_resolution ? 1 : 0,
                                  desc.depth_inverted ? 1 : 0,
                                  desc.hdr ? 1 : 0};
    if (!impl->api->FgCreate(command, &feature)) {
        return false;
    }
    impl->fg_feature = desc;
    std::printf("FrameGen: DLSS-G feature %ux%u (internal %ux%u, max %ux)\n", desc.backbuffer_width,
                desc.backbuffer_height, desc.input_width, desc.input_height,
                impl->fg_max_multi_frame + 1u);
    return true;
}

bool Dlss::EvaluateFg(vk::CommandBuffer command, const FgFrame& frame) {
    if (!impl->fg_available || !impl->fg_feature || !impl->api || impl->api->abi < 2 ||
        !impl->api->FgEvaluate) {
        return false;
    }
    BbDlssFgEvaluate parameters{};
    parameters.backbuffer = ToDlssImage(frame.backbuffer);
    parameters.depth = ToDlssImage(frame.depth);
    parameters.motion = ToDlssImage(frame.motion);
    if (frame.hudless.image) {
        parameters.hudless = ToDlssImage(frame.hudless);
    }
    parameters.output = ToDlssImage(frame.output);
    parameters.frame_id = frame.frame_id;
    parameters.multi_frame_count = frame.multi_frame_count;
    parameters.multi_frame_index = frame.multi_frame_index;
    parameters.reset = frame.reset ? 1 : 0;
    parameters.depth_inverted = frame.depth_inverted ? 1 : 0;
    parameters.hdr = frame.hdr ? 1 : 0;
    parameters.camera_motion_included = frame.camera_motion_included ? 1 : 0;
    parameters.render_subrect_width = frame.render_subrect_width;
    parameters.render_subrect_height = frame.render_subrect_height;
    parameters.jitter_x = frame.jitter_x;
    parameters.jitter_y = frame.jitter_y;
    parameters.mv_scale_x = frame.mv_scale_x;
    parameters.mv_scale_y = frame.mv_scale_y;
    for (int i = 0; i < 3; ++i) {
        parameters.camera_pos[i] = frame.camera_pos[i];
        parameters.camera_up[i] = frame.camera_up[i];
        parameters.camera_right[i] = frame.camera_right[i];
        parameters.camera_fwd[i] = frame.camera_fwd[i];
    }
    parameters.camera_near = frame.camera_near;
    parameters.camera_far = frame.camera_far;
    parameters.camera_fov = frame.camera_fov;
    parameters.camera_aspect = frame.camera_aspect;
    std::memcpy(parameters.camera_view_to_clip, frame.camera_view_to_clip.data(),
                sizeof(parameters.camera_view_to_clip));
    std::memcpy(parameters.clip_to_camera_view, frame.clip_to_camera_view.data(),
                sizeof(parameters.clip_to_camera_view));
    std::memcpy(parameters.clip_to_prev_clip, frame.clip_to_prev_clip.data(),
                sizeof(parameters.clip_to_prev_clip));
    std::memcpy(parameters.prev_clip_to_clip, frame.prev_clip_to_clip.data(),
                sizeof(parameters.prev_clip_to_clip));
    std::memcpy(parameters.clip_to_lens_clip, frame.clip_to_lens_clip.data(),
                sizeof(parameters.clip_to_lens_clip));
    return impl->api->FgEvaluate(command, &parameters) != 0;
}

void Dlss::ReleaseFgFeature() {
    impl->fg_feature.reset();
    if (impl->api && impl->api->abi >= 2 && impl->api->FgRelease) {
        impl->api->FgRelease();
    }
}

void Dlss::Shutdown() {
    {
        std::scoped_lock lk{impl->init_mutex};
        if (impl->init_thread.joinable()) {
            impl->init_thread.join();
        }
    }
    impl->available = false;
    impl->fg_available = false;
    impl->fg_max_multi_frame = 0;
    impl->feature.reset();
    impl->fg_feature.reset();
    if (impl->api && impl->initialized) {
        impl->api->Shutdown();
        impl->initialized = false;
    }
}

#else

struct Dlss::Impl {};
Dlss::Dlss() = default;
Dlss::~Dlss() = default;
Dlss* Dlss::Get() {
    return nullptr;
}
void Dlss::AppendInstanceExtensions(std::vector<const char*>&) {}
void Dlss::AppendDeviceExtensions(vk::Instance, vk::PhysicalDevice, std::vector<const char*>&) {}
void Dlss::Initialize(vk::Instance, vk::PhysicalDevice, vk::Device) {}
void Dlss::Disable(std::string_view) {}
void Dlss::Shutdown() {}
bool Dlss::EligibleForBoot() const noexcept {
    return false;
}
bool Dlss::Available() const {
    return false;
}
bool Dlss::FgAvailable() const {
    return false;
}
u32 Dlss::FgMaxMultiFrame() const {
    return 0;
}
std::string Dlss::Problem() const {
    return {};
}
std::string Dlss::FgProblem() const {
    return {};
}
bool Dlss::HasFeature(const FeatureDesc&) const {
    return false;
}
bool Dlss::CreateFeature(vk::CommandBuffer, const FeatureDesc&) {
    return false;
}
bool Dlss::Evaluate(vk::CommandBuffer, const Frame&) {
    return false;
}
void Dlss::ReleaseFeature() {}
bool Dlss::HasFgFeature(const FgFeatureDesc&) const {
    return false;
}
bool Dlss::CreateFgFeature(vk::CommandBuffer, const FgFeatureDesc&) {
    return false;
}
bool Dlss::EvaluateFg(vk::CommandBuffer, const FgFrame&) {
    return false;
}
void Dlss::ReleaseFgFeature() {}

#endif

} // namespace Vulkan
