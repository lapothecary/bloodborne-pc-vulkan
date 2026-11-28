// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 3A (Stage A elimination): Lock-free Direct State Object Registry sealing dense
// content-deduplicated 10-bit IDs for immutable `dantelion_gx` Rasterizer, Blend, and
// DepthStencil state objects (`+0 B` static TLS).

#pragma once

#include <array>
#include <cstdint>
#include <cstring>

#include "common/types.h"
#include "render/native_cmd_stream.h"

namespace Vulkan::DirectStates {

inline constexpr u32 kMaxStates = 1024u; // 1..1023 valid dense IDs (0 = default/invalid)

/// Exact binary match to `Bloodborne::CoreGraphics2::Gnm::CGRasterizerStateObject` (24 B).
struct RasterStateDesc {
    u32 primitive_setup{0};
    u32 clip_control{0};
    u32 slope_scale_depth_bias_bits{0};
    u32 depth_bias_clamp_bits{0};
    u32 depth_bias_bits{0};
    u8 msaa_enable{0};
    u8 scissor_enable{0};
    u16 pad{0};
    bool operator==(const RasterStateDesc&) const = default;
};
static_assert(sizeof(RasterStateDesc) == 24u, "Exact 24-byte CGRasterizerStateObject match");

/// Exact binary match to `Bloodborne::CoreGraphics2::Gnm::CGBlendStateObject` (40 B).
struct BlendStateDesc {
    u32 render_target_channel_mask{0};
    std::array<u32, 8> blend_control{};
    u8 alpha_to_coverage_enable{0};
    u8 pad[3]{};
    bool operator==(const BlendStateDesc&) const = default;
};
static_assert(sizeof(BlendStateDesc) == 40u, "Exact 40-byte CGBlendStateObject match");

/// Exact binary match to `Bloodborne::CoreGraphics2::Gnm::CGDepthStencilStateObject` (12 B).
struct DepthStencilStateDesc {
    u32 depth_control{0};
    u32 stencil_control{0};
    u32 stencil_op_control{0};
    bool operator==(const DepthStencilStateDesc&) const = default;
};
static_assert(sizeof(DepthStencilStateDesc) == 12u, "Exact 12-byte CGDepthStencilStateObject match");

/// Precomputed `SPI_PS_INPUT_CNTL_0..31` usage table for a `(VS exports, PS inputs)` pair.
struct PsUsageDesc {
    u32 count{0};
    std::array<u32, 32> dwords{};
    bool operator==(const PsUsageDesc& o) const noexcept {
        return count == o.count &&
               (count == 0u || std::memcmp(dwords.data(), o.dwords.data(), count * sizeof(u32)) == 0);
    }
};

/// Stage 4A: Precomputed Vulkan dynamic rasterizer state per unique `RasterStateId` (1..10).
/// All Vulkan enum fields store their exact `uint32_t` Vulkan ABI enumerator values so this
/// header stays Vulkan-Hpp-free while Stage B casts them in zero instructions.
struct CompiledRasterDyn {
    u32 cull_mode_vk{0};
    u32 front_face_vk{0};
    float depth_bias_const{0.0f};
    float depth_bias_clamp{0.0f};
    float depth_bias_slope{0.0f};
    bool depth_bias_enable{false};
    bool rasterizer_discard{false};
    /// Bitmask of RS inputs affecting `UpdateViewportScissorState`:
    /// bit0=clip_disable, bit1=clip_space_minus_w, bit2=scissor_enable, bit3=window_offset_enable.
    u8 vp_flags{0};
    u8 pad{0};
};

/// Stage 4A: Precomputed Vulkan dynamic depth-stencil state per unique `DepthStencilStateId` (1..12).
struct CompiledDepthStencilDyn {
    u32 depth_compare_op_vk{0};
    u32 front_fail_vk{0};
    u32 front_pass_vk{0};
    u32 front_zfail_vk{0};
    u32 front_cmp_vk{0};
    u32 back_fail_vk{0};
    u32 back_pass_vk{0};
    u32 back_zfail_vk{0};
    u32 back_cmp_vk{0};
    bool depth_enable{false};
    bool depth_write_enable{false};
    bool depth_bounds_enable{false};
    bool stencil_enable{false};
    bool two_sided{false};
    bool front_uses_op_val{false};
    bool back_uses_op_val{false};
    bool front_all_keep{false};
    bool back_all_keep{false};
    bool front_cmp_checks_ref{false};
    bool back_cmp_checks_ref{false};
    u8 pad{0};
};

/// Lock-free L1 pointer lookup with lazy content-deduplicated registration on first bind.
[[nodiscard]] bb_native_stream::RasterStateId LookupOrRegisterRasterState(const void* rs_obj) noexcept;
[[nodiscard]] bb_native_stream::BlendStateId LookupOrRegisterBlendState(const void* bs_obj) noexcept;
[[nodiscard]] bb_native_stream::DepthStencilStateId LookupOrRegisterDepthStencilState(const void* ds_obj) noexcept;
[[nodiscard]] u16 LookupOrRegisterPsUsage(const void* vs_exports_raw, u32 num_vs_exports,
                                          const void* ps_inputs_raw, u32 num_ps_inputs) noexcept;

/// Stage 5 / Step P2: Lock-free content-only interning from raw `Gcn::Regs` dwords (used for
/// Scaleform UI Raw PM4 draws and batched overrides that have no `CG*StateObject` pointer).
[[nodiscard]] bb_native_stream::RasterStateId InternRasterStateDesc(
    const RasterStateDesc& desc) noexcept;
[[nodiscard]] bb_native_stream::BlendStateId InternBlendStateDesc(
    const BlendStateDesc& desc) noexcept;
[[nodiscard]] bb_native_stream::DepthStencilStateId InternDepthStencilStateDesc(
    const DepthStencilStateDesc& desc) noexcept;

/// Read-only snapshot lookup by dense state ID (lock-free, Stage B / Native Vulkan Recorder).
[[nodiscard]] bool ReadRasterState(bb_native_stream::RasterStateId id, RasterStateDesc& out) noexcept;
[[nodiscard]] bool ReadBlendState(bb_native_stream::BlendStateId id, BlendStateDesc& out) noexcept;
[[nodiscard]] bool ReadDepthStencilState(bb_native_stream::DepthStencilStateId id,
                                         DepthStencilStateDesc& out) noexcept;
[[nodiscard]] const CompiledRasterDyn* GetCompiledRasterDyn(
    bb_native_stream::RasterStateId id) noexcept;
[[nodiscard]] const CompiledDepthStencilDyn* GetCompiledDepthStencilDyn(
    bb_native_stream::DepthStencilStateId id) noexcept;
[[nodiscard]] const PsUsageDesc* GetPsUsagePtr(u16 id) noexcept;

struct RegistryStats {
    u32 raster_unique{0};
    u32 blend_unique{0};
    u32 ds_unique{0};
    u32 ps_usage_unique{0};
    u64 raster_lookups{0};
    u64 blend_lookups{0};
    u64 ds_lookups{0};
    u64 ps_usage_lookups{0};
};

[[nodiscard]] RegistryStats GetStats() noexcept;

} // namespace Vulkan::DirectStates

extern "C" {
uint32_t PS4_SYSV_ABI bb_direct_vk_lookup_raster_state_id(const void* rs_obj) noexcept;
uint32_t PS4_SYSV_ABI bb_direct_vk_lookup_blend_state_id(const void* bs_obj) noexcept;
uint32_t PS4_SYSV_ABI bb_direct_vk_lookup_ds_state_id(const void* ds_obj) noexcept;
uint32_t PS4_SYSV_ABI bb_direct_vk_lookup_ps_usage_id(const void* vs_exports_raw,
                                                      uint32_t num_vs_exports,
                                                      const void* ps_inputs_raw,
                                                      uint32_t num_ps_inputs) noexcept;
}
