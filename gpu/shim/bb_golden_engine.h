// SPDX-License-Identifier: GPL-2.0-or-later
// bb_vulkan: Golden Engine C++20 Native GX Bridge
#pragma once

#include <cstdint>

namespace BbGoldenEngine {

struct Telemetry {
    std::uint64_t cpp_draws;
    std::uint64_t cpp_dispatches;
    std::uint64_t cpp_gpu_writes;
    std::uint64_t cpp_gpu_write_bytes;
    std::uint64_t cpp_manifest_hits;
    std::uint64_t cpp_camera_updates;
};

struct CameraBridgeState {
    float view[16];
    float inv_view[16];
    float prev_view[16];
    float prev_inv_view[16];
    float near_z;
    float far_z;
    float fov_y;
    float jitter[2];
    std::uint64_t update_epoch;
    bool has_view;
    bool has_inv_view;
    bool has_prev_view;
    bool has_params;
};

struct ShaderManifestInfo {
    const char* bundle;
    const char* name;
    const char* stage;
    std::uint64_t shader_hash;
    std::uint32_t crc32;
    std::uint32_t code_bytes;
    std::uint32_t instructions;
    std::uint8_t image_count;
    std::uint8_t sampler_count;
    std::uint8_t buffer_count;
    std::uint32_t max_cb_dwords;
};

struct HostSymbolMatch {
    const char* name;
    const char* module_name;
    std::uint64_t ps4_vma;
    std::uint32_t ps4_size;
    std::uintptr_t host_func_addr;
    std::uintptr_t host_offset;
    bool via_near_stub;
};

void PatchImage(unsigned char* image, std::uint64_t size);
bool ResolveHostAddress(std::uintptr_t addr, HostSymbolMatch* out) noexcept;
bool ResolveGuestVma(std::uint64_t vma, HostSymbolMatch* out) noexcept;
Telemetry ReadTelemetry() noexcept;
CameraBridgeState ReadCameraBridge() noexcept;
bool QueryShaderManifest(std::uint64_t pgm_hash, std::uint32_t crc32, std::uint32_t code_bytes,
                         ShaderManifestInfo* out) noexcept;
std::uint32_t GetShaderManifestCount() noexcept;

} // namespace BbGoldenEngine
