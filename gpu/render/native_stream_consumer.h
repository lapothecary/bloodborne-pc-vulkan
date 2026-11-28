// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 5 / Step 1 (Main Switch): Direct `bb_native_stream` Vulkan consumer on Stage B.
//
// Consumes flat `bb_native_stream` binary packets (`SetPipelineState` -> `SetDynamicState` ->
// `SetVertexBuffers` -> `BindStageResources` -> `DrawIndexed`/`DrawAuto`) directly into Vulkan
// command recording with zero `Gcn::Regs` unpack/copy, zero `ud_snapshots`, zero `flattened_ud_buf`
// replay, and zero `PrepareRenderState`/`MakeBeginSignature`/`BOccSort` on steady-pass 3D draws.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>

#include "common/types.h"
#include "gcn/regs.h"
#include "render/native_cmd_stream.h"

namespace Vulkan {

class GraphicsPipeline;
class Rasterizer;

enum class NativeStageAReject : u8 {
    NotArmed = 0,
    PreGate,
    PassChange,
    SpecialPs,
    ViewportClip,
    StageMeta,
    SpecialBuf,
    SrvMeta,
    SrvTag,
    SmpTag,
    StreamFail,
    Count,
};

struct NativeConsumerStats {
    u64 native_draws = 0;
    u64 legacy_draws = 0;
    u64 srv_fast_hits = 0;
    u64 srv_slow_hits = 0;
    u64 smp_fast_hits = 0;
    u64 smp_slow_hits = 0;
    u64 cb_ring_hits = 0;
    u64 cb_direct_hits = 0;
    u64 dyn_skips = 0;
    u64 dyn_updates = 0;
    u64 mid_pass_rebinds = 0;
    u64 stream_errors = 0;
};

class NativeStreamConsumer {
public:
    NativeStreamConsumer() = default;

    /// Called from Stage A (relaxed atomic, only under `BB_DCB_STATS=1`) whenever a graphics draw
    /// falls back from `PacketKind::NativeDraw` to legacy `DrawPacket`.
    void NoteStageAReject(NativeStageAReject reason) noexcept {
        stage_a_rej_[static_cast<size_t>(reason)].fetch_add(1u, std::memory_order_relaxed);
    }

    /// Called on Stage B whenever a legacy `DrawPacket` establishes/updates pass state.
    void OnLegacyDraw(bool count_draw = true) noexcept;

    /// Called on Stage B on pass-breaking ops (`Task`, `Dispatch`, `Special`, `Drain`).
    void InvalidatePass() noexcept {
        has_dyn_ = false;
        last_dyn_cmd_epoch_ = 0u;
        last_vp_cmd_epoch_ = 0u;
    }

    /// Executes a single flat `bb_native_stream` draw chunk directly into Vulkan commands on
    /// Stage B, returning `true` on success (or `false` on malformed stream).
    bool ExecuteDrawStream(Rasterizer& r, const u32* stream, u32 size_dwords,
                           u16 occ_handoff) noexcept;

    [[nodiscard]] const NativeConsumerStats& CumulativeStats() const noexcept {
        return total_;
    }

    void MaybeLogAudit() noexcept;

private:
    void ApplyNativeDynamicState(Rasterizer& r, const GraphicsPipeline* pipeline, u16 rs_id,
                                 u16 ds_id, Gcn::PrimitiveType prim_type, bool is_indexed,
                                 bool prim_restart_en, u32 prim_restart_idx, bool depth_valid,
                                 bool stencil_valid, u8 dyn_flags) noexcept;

    // Cached Stage B dynamic state maintained purely from delta `SetDynamicState` packets.
    bool has_dyn_ = false;
    bb_native_stream::SetDynamicStatePayload cur_dyn_{};
    u64 last_dyn_cmd_epoch_ = 0;
    u64 last_vp_cmd_epoch_ = 0;
    u16 last_rs_id_ = 0;
    u16 last_bs_id_ = 0;
    u16 last_ds_id_ = 0;
    u8 last_prim_type_ = 0xFFu;
    u8 last_vp_ctrl_bits_ = 0xFFu;
    bool last_is_indexed_ = false;
    bool last_prim_restart_ = false;
    bool last_depth_valid_ = false;
    bool last_stencil_valid_ = false;
    std::array<float, 2> last_vp_scale_{};
    std::array<float, 2> last_vp_jitter_{};
    u64 last_vsharps_hash_ = 0;
    u32 last_vsharps_count_ = 0;
    /// Dedicated Stage-B scratch `Gcn::Regs` populated strictly on `CmdOp::BeginRenderPass`
    /// (~60x/frame) so `r.pipe_regs` stays 100% isolated for legacy HLE bridge draws (`+0 B` TLS).
    std::unique_ptr<Gcn::Regs> pass_regs_;

    NativeConsumerStats total_{};
    NativeConsumerStats window_{};
    u64 window_start_ns_ = 0;
    u32 audit_tick_ = 0;
    std::array<std::atomic<u64>, static_cast<size_t>(NativeStageAReject::Count)> stage_a_rej_{};
};

} // namespace Vulkan
