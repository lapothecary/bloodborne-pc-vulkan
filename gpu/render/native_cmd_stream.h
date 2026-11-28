// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 3A (Stage A elimination): Flat Native Vulkan 1.3 Command Stream binary contract shared
// between `dantelion_gx` (producer on CPU 1 / GXWorker) and `gpu/` (consumer on Native Vulkan
// Recorder / Stage B). Zero heavy headers, zero static TLS, deterministic 4-byte aligned layout.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace bb_native_stream {

inline constexpr uint32_t kStreamMagic = 0x4E564B31u; // "NVK1"
inline constexpr uint32_t kStreamVersion = 1u;

// Dense runtime handles (0 = unbound / unresolved fallback).
using RasterStateId = uint16_t;       // 1..1023 (Vulkan::DirectStates)
using BlendStateId = uint16_t;        // 1..1023 (Vulkan::DirectStates)
using DepthStencilStateId = uint16_t; // 1..1023 (Vulkan::DirectStates)
using TargetUid = uint16_t;           // 1..4095 (Vulkan::DirectTargets)
using NativePsoId = uint16_t;         // 1..4095 (Vulkan::DirectPso)
using NativeSrvUid = uint16_t;        // 1..65535 (VideoCore::DirectTextures)
using NativeSamplerUid = uint16_t;    // 1..4095 (VideoCore::DirectTextures)

/// Runtime A/B master toggle (`BB_NATIVE_STREAM=1|0`): switches Stage B to the direct
/// `NativeStreamConsumer` for steady-pass 3D draws (default ON; set `=0` for legacy rollback).
inline bool Enabled() noexcept {
    static const bool enabled = []() -> bool {
        const char* env = std::getenv("BB_NATIVE_STREAM");
        return !(env != nullptr && env[0] == '0');
    }();
    return enabled;
}

/// Optional shadow-parity & `specSkip` verifier (`BB_NATIVE_STREAM_SHADOW=1`, zero cost when 0).
inline bool ShadowVerifyEnabled() noexcept {
    static const bool enabled = []() -> bool {
        const char* env = std::getenv("BB_NATIVE_STREAM_SHADOW");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}

/// Complete native command vocabulary replacing PS4 PM4 + `Gcn::Regs` for 97.5% (100% 3D) draws,
/// plus `LegacyPm4Slice` for the ~2.5% Scaleform UI draws at frame end.
enum class CmdOp : uint8_t {
    Noop = 0,               ///< Wrap / alignment skip (`size_dwords` = skipped words incl. header)
    BeginPassTargets = 1,   ///< Bind active Color RT UIDs[0..7] + Depth RT UID + pass dimensions
    SetPipelineState = 2,   ///< Bind VS/PS + dense Raster/Blend/DS state IDs + prim type + PSO hint
    SetDynamicState = 3,    ///< Delta-bitmasked dynamic Vulkan state (viewport/scissor/stencil/bias)
    SetVertexBuffers = 4,   ///< Bind index buffer + N vertex stream descriptors
    BindStageResources = 5, ///< Direct flat stage SRV/Sampler/CB bindings (no PS4 SRT/walker)
    DrawIndexed = 6,        ///< vkCmdDrawIndexed
    DrawAuto = 7,           ///< vkCmdDraw
    DrawIndirect = 8,       ///< vkCmdDrawIndirect / vkCmdDrawIndexedIndirect
    Dispatch = 9,           ///< vkCmdDispatch (compute)
    CopyBuffer = 10,        ///< Direct buffer-to-buffer copy
    FillBuffer = 11,        ///< Direct buffer fill / clear
    CopyImage = 12,         ///< Direct image subresource copy
    ClearTarget = 13,       ///< Direct color / depth-stencil clear
    DecompressTarget = 14,  ///< HTILE / CMASK / FMASK sync hint
    ReleaseLabel = 15,      ///< EOP / EOS guest fence write
    OcclusionQuery = 16,    ///< Native VkQueryPool begin/end/resolve
    TimestampQuery = 17,    ///< Native GPU timestamp write
    NativePostFx = 18,      ///< Native Vulkan PostFX / Tonemap pass
    GenerateMipsSpd = 19,   ///< Native FidelityFX SPD mipmap generation
    LegacyPm4Slice = 20,    ///< Passthrough raw PM4 slice (~2.5% Scaleform UI compatibility bridge)
    EndFrameFlip = 21,      ///< Present / swapchain flip marker
    BeginRenderPass = 22,   ///< Stage 5 P1/P2: Full native pass-start / target-change packet
};

/// 4-byte packet header (`size_dwords` includes this 1-dword header; payload follows immediately).
struct CmdHeader {
    uint8_t op;
    uint8_t flags;
    uint16_t size_dwords;
};
static_assert(sizeof(CmdHeader) == 4u, "CmdHeader must be 4 bytes");

struct BeginPassTargetsPayload {
    TargetUid cb_uid[8];
    TargetUid db_uid;
    uint8_t cb_mask;
    uint8_t aa_samples_log2;
    uint16_t width;
    uint16_t height;
};
static_assert(sizeof(BeginPassTargetsPayload) == 24u, "BeginPassTargetsPayload must be 24 bytes");

/// Bitmask flags for `CmdHeader::flags` on `CmdOp::BeginRenderPass`.
inline constexpr uint8_t kPassFlagHasDepth = 1u << 0;
inline constexpr uint8_t kPassFlagHasCb0Hint = 1u << 1;

/// Variable-length header for `CmdOp::BeginRenderPass` (emitted only on pass start / transition).
/// Followed immediately in the stream by:
///   - `popcount(active_cb_mask)` x 16 dwords (`Gcn::ColorBuffer` 15 dw + `CbDbExtent` 1 dw)
///   - if `(hdr.flags & kPassFlagHasCb0Hint) != 0`: 16 dwords (`color_buffers[0]` + `CbDbExtent`)
///     for depth-only stencil masks (`active_cb_mask == 0`) so `BindDepthAttachment` sees `cb0`
///   - if `(hdr.flags & kPassFlagHasDepth) != 0`: 9 dwords (`Gcn::DepthBuffer` 8 dw + `CbDbExtent` 1 dw)
struct BeginRenderPassHeader {
    TargetUid cb_uid[8];
    TargetUid db_uid;
    uint8_t active_cb_mask;
    uint8_t pass_flags;
    uint32_t color_control;
    uint32_t color_target_mask;
    uint32_t color_shader_mask;
    uint32_t blend_control0;
    uint32_t depth_control;
    uint32_t stencil_control;
    uint32_t stencil_ref_front;
    uint32_t stencil_ref_back;
    uint32_t depth_render_control;
    uint32_t depth_render_override;
    uint32_t depth_clear;
    uint32_t stencil_clear;
    uint32_t depth_view;
    uint32_t depth_htile_data_base;
    uint32_t aa_config;
    uint32_t clipper_control;
    uint32_t mode_control;
    uint32_t viewport_control;
    uint32_t screen_scissor[2];
    uint32_t window_scissor[2];
    uint32_t viewport_scissor0[2];
    uint32_t viewport0[6];
    uint32_t num_indices;
    uint32_t num_instances;
};
static_assert(sizeof(BeginRenderPassHeader) == 148u, "BeginRenderPassHeader must be 148 bytes");

struct SetPipelineStatePayload {
    uint64_t vs_addr;
    uint64_t ps_addr;
    uint32_t vs_rsrc1;
    uint32_t ps_rsrc1;
    RasterStateId raster_id;
    BlendStateId blend_id;
    DepthStencilStateId ds_id;
    NativePsoId pso_id;
    uint8_t prim_type;
    uint8_t stage_enable;
    uint8_t num_interp;
    uint8_t pad;
    uint32_t reserved;
};
static_assert(sizeof(SetPipelineStatePayload) == 40u, "SetPipelineStatePayload must be 40 bytes");

/// Bitmask flags for `CmdHeader::flags` on `CmdOp::SetDynamicState`.
inline constexpr uint8_t kDynFlagViewport = 1u << 0;
inline constexpr uint8_t kDynFlagScissor = 1u << 1;
inline constexpr uint8_t kDynFlagBlendColor = 1u << 2;
inline constexpr uint8_t kDynFlagStencilRef = 1u << 3;
inline constexpr uint8_t kDynFlagDepthBias = 1u << 4;
inline constexpr uint8_t kDynFlagDepthBounds = 1u << 5;
inline constexpr uint8_t kDynFlagLineWidth = 1u << 6;
inline constexpr uint8_t kDynFlagBorderColor = 1u << 7;

struct SetDynamicStatePayload {
    float vp_xscale, vp_xoffset, vp_yscale, vp_yoffset, vp_zmin, vp_zmax;
    int16_t sc_left, sc_top, sc_right, sc_bottom;
    float blend_rgba[4];
    float depth_bias_const, depth_bias_slope, depth_bias_clamp;
    float depth_bounds_min, depth_bounds_max;
    uint32_t stencil_ref_front;
    uint32_t stencil_ref_back;
    float line_width;
    uint32_t ta_bc_base;
};
static_assert(sizeof(SetDynamicStatePayload) == 84u, "SetDynamicStatePayload must be 84 bytes");

/// Bitmask flags for `CmdHeader::flags` on `CmdOp::SetVertexBuffers`.
inline constexpr uint8_t kVbFlagIndexed = 1u << 0;
inline constexpr uint8_t kVbFlagIndex32 = 1u << 1;
inline constexpr uint8_t kVbFlagPrimRestart = 1u << 2;
inline constexpr uint8_t kVbFlagReuseVsharps = 1u << 3;

/// Variable-length header for `CmdOp::SetVertexBuffers`.
/// When `(hdr.flags & kVbFlagReuseVsharps) == 0`, followed immediately by `num_vbs` x 4 dwords
/// (`Gcn::Buffer` V#, 16 B each).
struct SetVertexBuffersHeader {
    uint64_t index_va;
    uint32_t prim_restart_idx;
    uint32_t step_rate_0;
    uint32_t step_rate_1;
    uint16_t num_vbs;
    uint16_t reserved;
};
static_assert(sizeof(SetVertexBuffersHeader) == 24u, "SetVertexBuffersHeader must be 24 bytes");

struct DrawIndexedPayload {
    uint32_t index_count;
    uint32_t instance_count;
    uint32_t first_index;
    int32_t vertex_offset;
    uint32_t first_instance;
};
static_assert(sizeof(DrawIndexedPayload) == 20u, "DrawIndexedPayload must be 20 bytes");

struct DrawAutoPayload {
    uint32_t vertex_count;
    uint32_t instance_count;
    uint32_t first_vertex;
    uint32_t first_instance;
};
static_assert(sizeof(DrawAutoPayload) == 16u, "DrawAutoPayload must be 16 bytes");

struct DrawIndirectPayload {
    uint64_t args_va;
    uint32_t count;
    uint32_t stride;
    uint32_t is_indexed;
    uint32_t reserved;
};
static_assert(sizeof(DrawIndirectPayload) == 24u, "DrawIndirectPayload must be 24 bytes");

/// Variable-length header for `CmdOp::BindStageResources`.
/// `CmdHeader::flags` carries `num_ud_regs` (0..16 scalar user-data dwords for `PushData::ud_regs`).
/// Followed immediately in the stream by:
///   - `num_ud_regs` x `uint32_t` scalar push user-data dwords
///   - `num_srvs` x `NativeSrvBinding` (12 B each: `dw7_tag`, `dw0`, `dw1`)
///   - `num_samplers` x `uint32_t` packed `(slot << 16) | sampler_uid`
///   - `num_cbs` x `NativeConstBinding` (24 B each)
struct BindStageResourcesHeader {
    uint8_t hw_stage; // 0=PS, 1=VS, 2=GS, 3=ES, 4=HS, 5=LS, 6=CS
    uint8_t num_srvs;
    uint8_t num_samplers;
    uint8_t num_cbs;
};
static_assert(sizeof(BindStageResourcesHeader) == 4u, "BindStageResourcesHeader must be 4 bytes");

struct NativeSrvBinding {
    uint32_t dw7_tag; // (kSrvTagMagic << 24) | (gen8 << 16) | uid
    uint32_t dw0;     // T#.dw0 (base_address lo)
    uint32_t dw1;     // T#.dw1 (base_address hi + min_lod + dfmt + nfmt)
};
static_assert(sizeof(NativeSrvBinding) == 12u, "NativeSrvBinding must be 12 bytes");

inline constexpr uint32_t kNativeRingNotCopied = ~0u;
inline constexpr uint32_t kNativeRingPending = ~0u - 1u;

/// Bitmask flags for `NativeConstBinding::flags`.
inline constexpr uint8_t kNativeCbFlagRing = 1u << 0;        // RingBinding entry
inline constexpr uint8_t kNativeCbFlagSpecialNull = 1u << 1; // ClipPlanes disabled -> VK_NULL_HANDLE
inline constexpr uint8_t kNativeCbFlagSpecialBind = 1u << 2; // BdaPagetable / FaultBuffer
inline constexpr uint8_t kNativeCbFlagFormatted = 1u << 3;   // desc.is_formatted
inline constexpr uint8_t kNativeCbFlagStride16 = 1u << 4;    // VS non-special buffer with stride == 16

struct NativeConstBinding {
    uint64_t gpu_va;
    uint32_t byte_size;
    uint32_t ring_offset; // kNativeRingPending, kNativeRingNotCopied, or byte offset into ring_id
    uint16_t slot;
    uint8_t ring_id;      // 0 = constant_ring, 1..N = worker_rings[w], 0xFF = guest/direct buffer
    uint8_t flags;        // kNativeCbFlag*
};
static_assert(sizeof(NativeConstBinding) == 24u, "NativeConstBinding must be 24 bytes");

/// Passthrough slice for Scaleform UI Raw PM4 packets (~2.5% of draws at frame end).
struct LegacyPm4SlicePayload {
    uint64_t pm4_va;
    uint32_t num_dwords;
    uint32_t flags;
};
static_assert(sizeof(LegacyPm4SlicePayload) == 16u, "LegacyPm4SlicePayload must be 16 bytes");

/// Lock-free flat dword chunk writer (zero heap allocations, exact bounds check).
class NativeStreamWriter {
public:
    NativeStreamWriter() noexcept = default;
    NativeStreamWriter(uint32_t* buffer, uint32_t capacity_dwords) noexcept
        : base_{buffer}, cap_{capacity_dwords}, pos_{0u} {}

    void Reset(uint32_t* buffer, uint32_t capacity_dwords) noexcept {
        base_ = buffer;
        cap_ = capacity_dwords;
        pos_ = 0u;
    }

    [[nodiscard]] uint32_t* Reserve(CmdOp op, uint8_t flags, uint16_t payload_dwords) noexcept {
        const uint32_t total = 1u + static_cast<uint32_t>(payload_dwords);
        if (!base_ || total > (cap_ - pos_) || total > 0xFFFFu) [[unlikely]] {
            return nullptr;
        }
        uint32_t* dst = base_ + pos_;
        const CmdHeader hdr{
            static_cast<uint8_t>(op),
            flags,
            static_cast<uint16_t>(total),
        };
        std::memcpy(dst, &hdr, sizeof(hdr));
        pos_ += total;
        return dst + 1u;
    }

    template <typename T>
    bool Emit(CmdOp op, uint8_t flags, const T& payload) noexcept {
        static_assert((sizeof(T) % 4u) == 0u, "Payload size must be a multiple of 4 bytes");
        constexpr uint16_t kPayloadDwords = static_cast<uint16_t>(sizeof(T) / 4u);
        uint32_t* dst = Reserve(op, flags, kPayloadDwords);
        if (!dst) [[unlikely]] {
            return false;
        }
        std::memcpy(dst, &payload, sizeof(T));
        return true;
    }

    bool EmitEmpty(CmdOp op, uint8_t flags = 0u) noexcept {
        return Reserve(op, flags, 0u) != nullptr;
    }

    [[nodiscard]] const uint32_t* Data() const noexcept { return base_; }
    [[nodiscard]] uint32_t SizeDwords() const noexcept { return pos_; }
    [[nodiscard]] uint32_t CapacityDwords() const noexcept { return cap_; }

private:
    uint32_t* base_ = nullptr;
    uint32_t cap_ = 0u;
    uint32_t pos_ = 0u;
};

/// Zero-copy sequential reader over a recorded native command chunk.
class NativeStreamReader {
public:
    NativeStreamReader(const uint32_t* buffer, uint32_t size_dwords) noexcept
        : cur_{buffer}, end_{buffer ? (buffer + size_dwords) : nullptr} {}

    bool Next(CmdHeader& out_hdr, const uint32_t*& out_payload) noexcept {
        while (cur_ < end_) {
            std::memcpy(&out_hdr, cur_, sizeof(out_hdr));
            if (out_hdr.size_dwords == 0u || cur_ + out_hdr.size_dwords > end_) [[unlikely]] {
                cur_ = end_;
                return false;
            }
            out_payload = cur_ + 1u;
            cur_ += out_hdr.size_dwords;
            if (static_cast<CmdOp>(out_hdr.op) == CmdOp::Noop) {
                continue;
            }
            return true;
        }
        return false;
    }

    [[nodiscard]] bool Empty() const noexcept { return cur_ >= end_; }

private:
    const uint32_t* cur_ = nullptr;
    const uint32_t* end_ = nullptr;
};

} // namespace bb_native_stream
