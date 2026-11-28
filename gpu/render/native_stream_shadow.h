// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 3B-1 / 3B-2 (Stage A elimination): Live NativeCmdStream shadow producer and bit-exact
// `Gcn::Regs` + resource UID parity verifier gated behind `BB_NATIVE_STREAM=1`.
//
// Emits a real flat `bb_native_stream` binary command buffer per graphics submission in exact
// Vulkan execution order (`BeginPassTargets` -> `SetPipelineState` -> `SetDynamicState` ->
// `BindStageResources` -> `DrawIndexed`/`DrawAuto`/`DrawIndirect`, plus `LegacyPm4Slice` for
// Scaleform UI), verifies framing + draw-count + DirectStates bit-for-bit parity against
// `Gcn::Regs`, and audits NativeSrvUid / NativeSamplerUid / NativeVkBuf / NativePsoId coverage
// under `[NativeStream-Audit]`. Zero overhead when `BB_NATIVE_STREAM=0`.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "common/types.h"
#include "gcn/regs.h"
#include "render/native_cmd_stream.h"

namespace Vulkan {

class GraphicsPipeline;

struct NativeStreamParityStats {
    u64 submissions = 0;
    u64 packets_emitted = 0;
    u64 dwords_emitted = 0;
    u64 draws_3d = 0;
    u64 draws_pm4 = 0;
    u64 rs_covered = 0;
    u64 bs_covered = 0;
    u64 ds_covered = 0;
    u64 rt_covered = 0;
    u64 pso_id_covered = 0;
    u64 dyn_packets = 0;
    u64 res_packets = 0;
    u64 srvs_total = 0;
    u64 srvs_uid_hit = 0;
    u64 samplers_total = 0;
    u64 samplers_uid_hit = 0;
    u64 bufs_total = 0;
    u64 bufs_cue = 0;
    u64 bufs_native = 0;
    u64 rs_mismatches = 0;
    u64 bs_mismatches = 0;
    u64 ds_mismatches = 0;
    u64 stream_errors = 0;
};

class NativeStreamShadow {
public:
    static constexpr u32 kInitialCapacityDwords = 131072u; // 512 KiB reusable per-submission chunk

    NativeStreamShadow() = default;

    void BeginSubmission() noexcept;

    /// Stashes DirectVk draw parameters right before `Rasterizer::Draw()` so `CompleteDirectDraw`
    /// emits `BeginPassTargets -> SetPipelineState(pso_id) -> SetDynamicState -> BindStageResources
    /// -> Draw*` in exact Vulkan recording order.
    void PrepareDirectDraw(u32 direct_op, u32 start_index = 0u) noexcept {
        pending_direct_op_ = direct_op;
        pending_start_index_ = start_index;
        has_pending_direct_ = true;
    }

    /// Completes the pending DirectVk draw once `GraphicsPipeline*` + stage flat buffers are
    /// resolved on Stage A.
    void CompleteDirectDraw(const Gcn::Regs& regs, const Gcn::CbDbExtent* cb_extents,
                            const u16* cb_uids, u16 db_uid, u16 rs_id, u16 bs_id, u16 ds_id,
                            const GraphicsPipeline& pipeline) noexcept;

    /// Standalone DirectVk draw emission (without `GraphicsPipeline*`, used by unit tests).
    void RecordDirectDraw(const Gcn::Regs& regs, const Gcn::CbDbExtent* cb_extents,
                          const u16* cb_uids, u16 db_uid, u16 rs_id, u16 bs_id, u16 ds_id,
                          u32 direct_op, u32 start_index = 0u, u16 pso_id = 0u,
                          const GraphicsPipeline* pipeline = nullptr) noexcept;

    /// Emits a `LegacyPm4Slice` command for a Raw PM4 draw (e.g. Scaleform UI).
    void RecordLegacyPm4Draw(const u32* packet, u32 total_dwords) noexcept;

    /// Replays the emitted binary chunk through `NativeStreamReader` to verify framing & draw
    /// count parity, updates window statistics, and emits `[NativeStream-Audit]` every 5 seconds.
    bool EndSubmission(bool log_periodic = true) noexcept;

    [[nodiscard]] const NativeStreamParityStats& CumulativeStats() const noexcept {
        return total_;
    }

    [[nodiscard]] u32 LastStreamDwords() const noexcept {
        return writer_.SizeDwords();
    }

private:
    void VerifyDirectStateParity(const Gcn::Regs& regs, u16 rs_id, u16 bs_id, u16 ds_id) noexcept;
    void EmitDynamicStateDelta(const Gcn::Regs& regs) noexcept;
    void EmitStageResources(const GraphicsPipeline& pipeline) noexcept;
    void MaybeLogAudit() noexcept;

    std::vector<u32> storage_;
    bb_native_stream::NativeStreamWriter writer_{};

    // Per-submission delta caches to avoid redundant state packets.
    bool has_targets_ = false;
    bool has_pipe_ = false;
    bool has_dyn_ = false;
    bb_native_stream::BeginPassTargetsPayload last_targets_{};
    bb_native_stream::SetPipelineStatePayload last_pipe_{};
    bb_native_stream::SetDynamicStatePayload last_dyn_{};

    // Pending DirectVk draw handed from CommandProcessor -> Rasterizer::Draw.
    bool has_pending_direct_ = false;
    u32 pending_direct_op_ = 0;
    u32 pending_start_index_ = 0;

    // Per-submission counters verified on EndSubmission().
    u32 sub_draws_3d_ = 0;
    u32 sub_draws_pm4_ = 0;
    u32 sub_packets_ = 0;
    bool sub_overflow_ = false;
    bool active_ = false;

    u32 logged_mismatches_ = 0;
    NativeStreamParityStats total_{};
    NativeStreamParityStats window_{};
    u64 window_start_ns_ = 0;
};

} // namespace Vulkan
