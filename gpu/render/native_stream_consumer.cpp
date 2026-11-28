// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 5 / Step 1 (Main Switch): Direct `bb_native_stream` Vulkan consumer on Stage B.

#include "render/native_stream_consumer.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "cp/cp_diagnostics.h"
#include "render/direct_pso.h"
#include "render/direct_states.h"
#include "render/gcn_to_vk.h"
#include "render/pipeline.h"
#include "render/raster_internal.h"
#include "render/vk_vsm_blur_batcher.h"
#include "textures/direct_textures.h"
#include "upscale/vk_temporal_upscaler.h"

namespace Vulkan {

namespace {

inline u64 NowSteadyNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

inline bool IsListTopology(Gcn::PrimitiveType type) noexcept {
    switch (GcnToVk::PrimitiveType(type)) {
    case vk::PrimitiveTopology::ePointList:
    case vk::PrimitiveTopology::eLineList:
    case vk::PrimitiveTopology::eTriangleList:
    case vk::PrimitiveTopology::eLineListWithAdjacency:
    case vk::PrimitiveTopology::eTriangleListWithAdjacency:
        return true;
    default:
        return false;
    }
}

inline bool IsPatchListTopology(Gcn::PrimitiveType type) noexcept {
    return type == Gcn::PrimitiveType::PatchPrimitive || type == Gcn::PrimitiveType::QuadList ||
           type == Gcn::PrimitiveType::RectList;
}

} // namespace

void NativeStreamConsumer::OnLegacyDraw(bool count_draw) noexcept {
    if (count_draw) {
        ++total_.legacy_draws;
        ++window_.legacy_draws;
        BbStats::legacy_stream_draws.fetch_add(1u, std::memory_order_relaxed);
    }
    has_dyn_ = false;
    last_dyn_cmd_epoch_ = 0u;
    last_vp_cmd_epoch_ = 0u;
    last_vsharps_hash_ = 0u;
    last_vsharps_count_ = 0u;
    if ((++audit_tick_ & 0xFFu) == 0u) {
        MaybeLogAudit();
    }
}

void NativeStreamConsumer::MaybeLogAudit() noexcept {
    if (!BbCp::Diagnostics::DcbStatsEnabled()) {
        return;
    }
    const u64 now = NowSteadyNs();
    if (window_start_ns_ == 0u) {
        window_start_ns_ = now;
        return;
    }
    const u64 dt_ns = now - window_start_ns_;
    if (dt_ns < 5'000'000'000ULL) {
        return;
    }
    const double inv_s = 1e9 / static_cast<double>(dt_ns);
    const u64 tot_draws = window_.native_draws + window_.legacy_draws;
    const double nat_pct = tot_draws ? (100.0 * double(window_.native_draws) / double(tot_draws))
                                     : 0.0;
    const u64 dyn_tot = window_.dyn_skips + window_.dyn_updates;
    const double dyn_skip_pct =
        dyn_tot ? (100.0 * double(window_.dyn_skips) / double(dyn_tot)) : 0.0;
    std::array<u64, static_cast<size_t>(NativeStageAReject::Count)> rej{};
    for (size_t i = 0; i < rej.size(); ++i) {
        rej[i] = stage_a_rej_[i].exchange(0u, std::memory_order_relaxed);
    }
    std::printf(
        "[NativeStream-Exec] draws: native=%.0f/s (%.1f%%) legacy=%.0f/s | "
        "srv(fast=%.0f/s slow=%.0f/s) smp(fast=%.0f/s slow=%.0f/s) "
        "cb(ring=%.0f/s direct=%.0f/s) | dynSkip=%.1f%% rebinds=%llu err=%llu\n"
        "[NativeStream-Rej] notArmed=%llu preGate=%llu passChg=%llu specPs=%llu vpClip=%llu "
        "stMeta=%llu specBuf=%llu srvMeta=%llu srvTag=%llu smpTag=%llu fail=%llu\n",
        double(window_.native_draws) * inv_s, nat_pct, double(window_.legacy_draws) * inv_s,
        double(window_.srv_fast_hits) * inv_s, double(window_.srv_slow_hits) * inv_s,
        double(window_.smp_fast_hits) * inv_s, double(window_.smp_slow_hits) * inv_s,
        double(window_.cb_ring_hits) * inv_s, double(window_.cb_direct_hits) * inv_s,
        dyn_skip_pct, static_cast<unsigned long long>(window_.mid_pass_rebinds),
        static_cast<unsigned long long>(total_.stream_errors),
        static_cast<unsigned long long>(rej[static_cast<size_t>(NativeStageAReject::NotArmed)]),
        static_cast<unsigned long long>(rej[static_cast<size_t>(NativeStageAReject::PreGate)]),
        static_cast<unsigned long long>(rej[static_cast<size_t>(NativeStageAReject::PassChange)]),
        static_cast<unsigned long long>(rej[static_cast<size_t>(NativeStageAReject::SpecialPs)]),
        static_cast<unsigned long long>(rej[static_cast<size_t>(NativeStageAReject::ViewportClip)]),
        static_cast<unsigned long long>(rej[static_cast<size_t>(NativeStageAReject::StageMeta)]),
        static_cast<unsigned long long>(rej[static_cast<size_t>(NativeStageAReject::SpecialBuf)]),
        static_cast<unsigned long long>(rej[static_cast<size_t>(NativeStageAReject::SrvMeta)]),
        static_cast<unsigned long long>(rej[static_cast<size_t>(NativeStageAReject::SrvTag)]),
        static_cast<unsigned long long>(rej[static_cast<size_t>(NativeStageAReject::SmpTag)]),
        static_cast<unsigned long long>(rej[static_cast<size_t>(NativeStageAReject::StreamFail)]));
    window_ = {};
    window_start_ns_ = now;
}

void NativeStreamConsumer::ApplyNativeDynamicState(
    Rasterizer& r, const GraphicsPipeline* pipeline, u16 rs_id, u16 ds_id,
    Gcn::PrimitiveType prim_type, bool is_indexed, bool prim_restart_en, u32 prim_restart_idx,
    bool depth_valid, bool stencil_valid, u8 dyn_flags) noexcept {
    BB_SECTION(UpdateDynamicState);
    auto& dynamic_state = r.scheduler.GetDynamicState();
    const u64 cur_cmd_epoch = r.scheduler.DescriptorStateEpoch();

    const bool state_unchanged =
        !dynamic_state.AnyDirty() && dyn_flags == 0u &&
        last_dyn_cmd_epoch_ == cur_cmd_epoch &&
        last_rs_id_ == rs_id && last_ds_id_ == ds_id &&
        last_prim_type_ == static_cast<u8>(prim_type) &&
        last_is_indexed_ == is_indexed &&
        last_prim_restart_ == prim_restart_en &&
        last_depth_valid_ == depth_valid &&
        last_stencil_valid_ == stencil_valid &&
        last_vp_scale_ == r.target_scale &&
        last_vp_jitter_ == r.draw_jitter;

    if (state_unchanged) [[likely]] {
        dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
        dynamic_state.SetAttachmentFeedbackLoopEnabled(r.attachment_feedback_loop);
        if (dynamic_state.AnyDirty()) {
            dynamic_state.CommitWith(
                r.instance.IsDepthBoundsSupported(), r.instance.IsDynamicColorWriteMaskSupported(),
                r.instance.IsAttachmentFeedbackLoopLayoutSupported(),
                [&](auto&& command) { r.scheduler.Record(std::move(command)); });
        }
        ++total_.dyn_skips;
        ++window_.dyn_skips;
        return;
    }

    ++total_.dyn_updates;
    ++window_.dyn_updates;
    const auto* const crs = DirectStates::GetCompiledRasterDyn(rs_id);
    const auto* const cds = DirectStates::GetCompiledDepthStencilDyn(ds_id);

    const u8 cur_vp_ctrl_bits = static_cast<u8>(
        (crs ? crs->vp_flags : 0u) |
        (prim_type == Gcn::PrimitiveType::RectList ? 16u : 0u));
    const bool need_vp =
        dynamic_state.ViewportsDirty() ||
        (dyn_flags & (bb_native_stream::kDynFlagViewport | bb_native_stream::kDynFlagScissor)) != 0u ||
        last_vp_cmd_epoch_ != cur_cmd_epoch ||
        last_vp_ctrl_bits_ != cur_vp_ctrl_bits ||
        last_vp_scale_ != r.target_scale ||
        last_vp_jitter_ != r.draw_jitter;

    if (need_vp) {
        const bool clip_disabled = (cur_vp_ctrl_bits & 1u) != 0u ||
                                   prim_type == Gcn::PrimitiveType::RectList;
        const bool unrestricted_depth = r.instance.IsDepthRangeUnrestrictedSupported();
        float min_depth = cur_dyn_.vp_zmin;
        float max_depth = cur_dyn_.vp_zmax;
        if (!unrestricted_depth) {
            min_depth = std::max(min_depth, 0.0f);
            max_depth = std::min(max_depth, 1.0f);
        }

        vk::Viewport vp{};
        vp.minDepth = min_depth;
        vp.maxDepth = max_depth;
        if (clip_disabled) [[unlikely]] {
            vp.x = 0.0f;
            vp.y = 0.0f;
            vp.width = float(std::min<u32>(r.instance.GetMaxViewportWidth(), 16u * 1024u));
            vp.height = float(std::min<u32>(r.instance.GetMaxViewportHeight(), 16u * 1024u));
        } else {
            vp.x = (cur_dyn_.vp_xoffset - cur_dyn_.vp_xscale) * r.target_scale[0] + r.draw_jitter[0];
            vp.y = (cur_dyn_.vp_yoffset - cur_dyn_.vp_yscale) * r.target_scale[1] + r.draw_jitter[1];
            vp.width = cur_dyn_.vp_xscale * 2.0f * r.target_scale[0];
            vp.height = cur_dyn_.vp_yscale * 2.0f * r.target_scale[1];
        }

        const bool scaled = r.target_scale[0] != 1.0f || r.target_scale[1] != 1.0f;
        const s32 raw_x0 = cur_dyn_.sc_left;
        const s32 raw_y0 = cur_dyn_.sc_top;
        const u32 raw_w = (cur_dyn_.sc_right > cur_dyn_.sc_left)
                              ? static_cast<u32>(cur_dyn_.sc_right - cur_dyn_.sc_left)
                              : 0u;
        const u32 raw_h = (cur_dyn_.sc_bottom > cur_dyn_.sc_top)
                              ? static_cast<u32>(cur_dyn_.sc_bottom - cur_dyn_.sc_top)
                              : 0u;
        vk::Rect2D sc{};
        if (scaled) {
            const auto sc_fn = [](s32 v, float f) noexcept {
                return static_cast<s32>(std::lround(float(v) * f));
            };
            const s32 x0 = sc_fn(raw_x0, r.target_scale[0]);
            const s32 y0 = sc_fn(raw_y0, r.target_scale[1]);
            const s32 x1 = sc_fn(raw_x0 + static_cast<s32>(raw_w), r.target_scale[0]);
            const s32 y1 = sc_fn(raw_y0 + static_cast<s32>(raw_h), r.target_scale[1]);
            sc = vk::Rect2D{
                .offset = {x0, y0},
                .extent = {static_cast<u32>(std::max(x1 - x0, 0)),
                           static_cast<u32>(std::max(y1 - y0, 0))},
            };
        } else {
            sc = vk::Rect2D{
                .offset = {raw_x0, raw_y0},
                .extent = {raw_w, raw_h},
            };
        }
        Viewports vps;
        vps.push_back(vp);
        Scissors scs;
        scs.push_back(sc);
        dynamic_state.SetViewports(vps);
        dynamic_state.SetScissors(scs);
        last_vp_cmd_epoch_ = cur_cmd_epoch;
        last_vp_ctrl_bits_ = cur_vp_ctrl_bits;
        last_vp_scale_ = r.target_scale;
        last_vp_jitter_ = r.draw_jitter;
    }

    if (cds && crs) [[likely]] {
        const bool depth_test = cds->depth_enable && depth_valid;
        dynamic_state.SetDepthTestEnabled(depth_test);
        if (depth_test) {
            dynamic_state.SetDepthWriteEnabled(cds->depth_write_enable);
            dynamic_state.SetDepthCompareOp(static_cast<vk::CompareOp>(cds->depth_compare_op_vk));
        }
        dynamic_state.SetDepthBoundsTestEnabled(cds->depth_bounds_enable);
        if (cds->depth_bounds_enable) {
            dynamic_state.SetDepthBounds(cur_dyn_.depth_bounds_min, cur_dyn_.depth_bounds_max);
        }
        dynamic_state.SetDepthBiasEnabled(crs->depth_bias_enable);
        if (crs->depth_bias_enable) {
            dynamic_state.SetDepthBias(crs->depth_bias_const, crs->depth_bias_clamp,
                                       crs->depth_bias_slope);
        }
        const bool stencil_test = cds->stencil_enable && stencil_valid;
        dynamic_state.SetStencilTestEnabled(stencil_test);
        if (stencil_test) {
            const StencilOps front_ops{
                .fail_op = static_cast<vk::StencilOp>(cds->front_fail_vk),
                .pass_op = static_cast<vk::StencilOp>(cds->front_pass_vk),
                .depth_fail_op = static_cast<vk::StencilOp>(cds->front_zfail_vk),
                .compare_op = static_cast<vk::CompareOp>(cds->front_cmp_vk),
            };
            const StencilOps back_ops{
                .fail_op = static_cast<vk::StencilOp>(cds->back_fail_vk),
                .pass_op = static_cast<vk::StencilOp>(cds->back_pass_vk),
                .depth_fail_op = static_cast<vk::StencilOp>(cds->back_zfail_vk),
                .compare_op = static_cast<vk::CompareOp>(cds->back_cmp_vk),
            };
            dynamic_state.SetStencilOps(front_ops, back_ops);
            const u32 f_raw = cur_dyn_.stencil_ref_front;
            const u32 b_raw = cds->two_sided ? cur_dyn_.stencil_ref_back : cur_dyn_.stencil_ref_front;
            const u32 f_test = f_raw & 0xFFu;
            const u32 f_mask = (f_raw >> 8) & 0xFFu;
            const u32 f_wmask = (f_raw >> 16) & 0xFFu;
            const u32 f_op = (f_raw >> 24) & 0xFFu;
            const u32 b_test = b_raw & 0xFFu;
            const u32 b_mask = (b_raw >> 8) & 0xFFu;
            const u32 b_wmask = (b_raw >> 16) & 0xFFu;
            const u32 b_op = (b_raw >> 24) & 0xFFu;
            dynamic_state.SetStencilReferences(cds->front_uses_op_val ? f_op : f_test,
                                               cds->back_uses_op_val ? b_op : b_test);
            dynamic_state.SetStencilWriteMasks(cds->front_all_keep ? 0u : f_wmask,
                                               cds->back_all_keep ? 0u : b_wmask);
            dynamic_state.SetStencilCompareMasks(f_mask, b_mask);
        }
        const bool restart =
            prim_restart_en &&
            (r.instance.IsListRestartSupported() || !IsListTopology(prim_type)) &&
            (r.instance.IsPatchListRestartSupported() || !IsPatchListTopology(prim_type));
        ASSERT_MSG(!is_indexed || !restart || prim_restart_idx == 0xFFFFu ||
                       prim_restart_idx == 0xFFFFFFFFu,
                   "Primitive restart index other than -1 is not supported yet");
        dynamic_state.SetPrimitiveRestartEnabled(restart);
        dynamic_state.SetRasterizerDiscardEnabled(crs->rasterizer_discard);
        dynamic_state.SetCullMode(GcnToVk::IsPrimitiveCulled(prim_type)
                                      ? static_cast<vk::CullModeFlags>(crs->cull_mode_vk)
                                      : vk::CullModeFlagBits::eNone);
        dynamic_state.SetFrontFace(static_cast<vk::FrontFace>(crs->front_face_vk));
    }

    dynamic_state.SetLineWidth(cur_dyn_.line_width);
    const std::array<f32, 4> blend_c{cur_dyn_.blend_rgba[0], cur_dyn_.blend_rgba[1],
                                     cur_dyn_.blend_rgba[2], cur_dyn_.blend_rgba[3]};
    dynamic_state.SetBlendConstants(blend_c);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(r.attachment_feedback_loop);

    dynamic_state.CommitWith(
        r.instance.IsDepthBoundsSupported(), r.instance.IsDynamicColorWriteMaskSupported(),
        r.instance.IsAttachmentFeedbackLoopLayoutSupported(),
        [&](auto&& command) { r.scheduler.Record(std::move(command)); });

    // Invalidate legacy Rasterizer dynamic-state memo epochs so the next legacy DrawPacket
    // always reconciles cleanly with DynamicState.
    r.last_dyn_cmd_epoch = 0;
    r.last_vp_cmd_epoch = 0;

    last_dyn_cmd_epoch_ = cur_cmd_epoch;
    last_rs_id_ = rs_id;
    last_ds_id_ = ds_id;
    last_prim_type_ = static_cast<u8>(prim_type);
    last_is_indexed_ = is_indexed;
    last_prim_restart_ = prim_restart_en;
    last_depth_valid_ = depth_valid;
    last_stencil_valid_ = stencil_valid;
}

bool NativeStreamConsumer::ExecuteDrawStream(Rasterizer& r, const u32* stream, u32 size_dwords,
                                             u16 occ_handoff) noexcept {
    static_assert(sizeof(Gcn::ColorBuffer) == 15u * sizeof(u32));
    static_assert(sizeof(Gcn::DepthBuffer) == 8u * sizeof(u32));
    static_assert(sizeof(r.pipe_regs.screen_scissor) == 2u * sizeof(u32));
    static_assert(sizeof(r.pipe_regs.window_scissor) == 2u * sizeof(u32));
    static_assert(sizeof(r.pipe_regs.viewport_scissors[0]) == 2u * sizeof(u32));
    static_assert(sizeof(r.pipe_regs.viewports[0]) == 6u * sizeof(u32));

    BB_SECTION(DrawRecord);
    bb_native_stream::NativeStreamReader reader(stream, size_dwords);
    bb_native_stream::CmdHeader hdr{};
    const u32* pay = nullptr;

    // RAII guard activated ONLY on `CmdOp::BeginRenderPass` (~5% of draws) so `r.pipe_regs` and
    // `r.pipe_*` metadata stay 100% isolated for legacy HLE bridge draws (`+0 B` overhead on steady
    // pass draws).
    struct PassScopeGuard {
        Rasterizer& rast;
        bool active = false;
        const Gcn::Regs* prev_stage_regs = nullptr;
        std::array<Gcn::CbDbExtent, Gcn::NUM_COLOR_BUFFERS> prev_cb_ext{};
        std::array<u16, Gcn::NUM_COLOR_BUFFERS> prev_cb_uid{};
        Gcn::CbDbExtent prev_db_ext{};
        u16 prev_db_uid = 0u;
        ~PassScopeGuard() {
            if (active) [[unlikely]] {
                Rasterizer::stage_regs = prev_stage_regs;
                rast.pipe_cb_extent = prev_cb_ext;
                rast.pipe_cb_uid = prev_cb_uid;
                rast.pipe_db_extent = prev_db_ext;
                rast.pipe_db_uid = prev_db_uid;
            }
        }
    } pass_guard{r};
    bool has_begin_pass_cmd = false;

    const GraphicsPipeline* pipeline = nullptr;
    bb_native_stream::SetPipelineStatePayload pipe_pay{};
    u8 dyn_flags = 0u;

    // Vertex / index state extracted from `SetVertexBuffers`.
    bool has_vb_cmd = false;
    bool vb_indexed = false;
    bool vb_index32 = false;
    bool vb_prim_restart = false;
    bb_native_stream::SetVertexBuffersHeader vb_hdr{};

    // Stage resource descriptor state.
    bool resources_started = false;
    Shader::Backend::Bindings binding{};
    r.needs_barrier = false;
    const u64 buf_alignment = r.instance.StorageMinAlignment();
    u32 stage_cursor = 0u;
    std::span<const Shader::Info* const> pipe_stages{};
    static const bool all_motion = [] {
        const char* value = std::getenv("BB_OBJECT_MOTION_ALL");
        return value && value[0] == '1';
    }();
    bool motion_gated = !all_motion;
    u64 motion_palette = 0ull;
    u32 ps_first_srv_dw7 = 0u;

    auto begin_resources = [&]() noexcept {
        if (resources_started) {
            return;
        }
        resources_started = true;
        // Keep Upscaler / UI phase tracking 100% synced BEFORE `BindNativeTextures` checks
        // `ui_pass_started` (and AFTER `SetDynamicState` + `BeginRenderPass` have updated state).
        const auto& gkey = pipeline->GetGraphicsKey();
        if (std::popcount(gkey.mrt_mask) == 1) {
            const u64 vs_h = pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash;
            const auto* ps_st = pipe_stages[u32(Shader::SwStage::Fragment)];
            const u64 ps_h = ps_st ? ps_st->pgm_hash : 0ull;
            if (UiComposition::MovieShader(vs_h)) {
                r.ui_pass_started = true;
            }
            if (r.upscaler->Enabled()) {
                r.upscaler->OnDraw(
                    vs_h, ps_h, r.cb_descs[0].first, r.db_desc.first,
                    UiComposition::NativeViewport(cur_dyn_.vp_xscale * 2.0f,
                                                  cur_dyn_.vp_yscale * 2.0f));
                if (r.upscaler->InUiPhase()) {
                    r.ui_pass_started = true;
                }
            }
        } else {
            r.ui_pass_started = false;
        }
        r.set_write_index = 0u;
        r.buffer_infos.clear();
        r.image_infos.clear();
        r.pipeline_is_compute = false;
        r.sampler_lod_bias = 0.0f;
        if (!BbToggle::Disabled(BbToggle::SceneMipBias) &&
            std::popcount(gkey.mrt_mask & 0xFFu) >= 5) {
            r.sampler_lod_bias = r.upscaler->SceneMipBias();
        }
        size_t total_desc = 0u;
        for (const auto* st : pipe_stages) {
            if (st) {
                total_desc += st->buffers.size() + st->images.size() + st->samplers.size();
            }
        }
        if (r.set_writes.size() < total_desc) [[unlikely]] {
            r.set_writes.resize(total_desc);
        }
        r.push_data = {};
        r.push_data.xoffset = cur_dyn_.vp_xoffset;
        r.push_data.xscale = cur_dyn_.vp_xscale;
        r.push_data.yoffset = cur_dyn_.vp_yoffset;
        r.push_data.yscale = cur_dyn_.vp_yscale;
    };

    bool drew = false;
    while (reader.Next(hdr, pay)) {
        const auto op = static_cast<bb_native_stream::CmdOp>(hdr.op);
        switch (op) {
        case bb_native_stream::CmdOp::SetPipelineState: {
            std::memcpy(&pipe_pay, pay, sizeof(pipe_pay));
            pipeline = r.pipeline_cache.LookupPipelineByPsoId(pipe_pay.pso_id);
            if (!pipeline) [[unlikely]] {
                ++total_.stream_errors;
                return false;
            }
            pipe_stages = pipeline->GetStages();
            r.motion_draw = pipeline->GetGraphicsKey().motion_vectors != 0u;
            r.motion_geometry = 0u;
            break;
        }
        case bb_native_stream::CmdOp::SetDynamicState: {
            bb_native_stream::SetDynamicStatePayload next_dyn{};
            std::memcpy(&next_dyn, pay, sizeof(next_dyn));
            const u8 f = hdr.flags;
            if (!has_dyn_ || f == 0xFFu) {
                cur_dyn_ = next_dyn;
                has_dyn_ = true;
                dyn_flags = 0xFFu;
            } else {
                if (f & bb_native_stream::kDynFlagViewport) {
                    cur_dyn_.vp_xscale = next_dyn.vp_xscale;
                    cur_dyn_.vp_xoffset = next_dyn.vp_xoffset;
                    cur_dyn_.vp_yscale = next_dyn.vp_yscale;
                    cur_dyn_.vp_yoffset = next_dyn.vp_yoffset;
                    cur_dyn_.vp_zmin = next_dyn.vp_zmin;
                    cur_dyn_.vp_zmax = next_dyn.vp_zmax;
                }
                if (f & bb_native_stream::kDynFlagScissor) {
                    cur_dyn_.sc_left = next_dyn.sc_left;
                    cur_dyn_.sc_top = next_dyn.sc_top;
                    cur_dyn_.sc_right = next_dyn.sc_right;
                    cur_dyn_.sc_bottom = next_dyn.sc_bottom;
                }
                if (f & bb_native_stream::kDynFlagBlendColor) {
                    std::memcpy(cur_dyn_.blend_rgba, next_dyn.blend_rgba, sizeof(cur_dyn_.blend_rgba));
                }
                if (f & bb_native_stream::kDynFlagStencilRef) {
                    cur_dyn_.stencil_ref_front = next_dyn.stencil_ref_front;
                    cur_dyn_.stencil_ref_back = next_dyn.stencil_ref_back;
                }
                if (f & bb_native_stream::kDynFlagDepthBias) {
                    cur_dyn_.depth_bias_const = next_dyn.depth_bias_const;
                    cur_dyn_.depth_bias_slope = next_dyn.depth_bias_slope;
                    cur_dyn_.depth_bias_clamp = next_dyn.depth_bias_clamp;
                }
                if (f & bb_native_stream::kDynFlagDepthBounds) {
                    cur_dyn_.depth_bounds_min = next_dyn.depth_bounds_min;
                    cur_dyn_.depth_bounds_max = next_dyn.depth_bounds_max;
                }
                if (f & bb_native_stream::kDynFlagLineWidth) {
                    cur_dyn_.line_width = next_dyn.line_width;
                }
                if (f & bb_native_stream::kDynFlagBorderColor) {
                    cur_dyn_.ta_bc_base = next_dyn.ta_bc_base;
                }
                dyn_flags |= f;
            }
            if (resources_started) {
                r.push_data.xoffset = cur_dyn_.vp_xoffset;
                r.push_data.xscale = cur_dyn_.vp_xscale;
                r.push_data.yoffset = cur_dyn_.vp_yoffset;
                r.push_data.yscale = cur_dyn_.vp_yscale;
            }
            break;
        }
        case bb_native_stream::CmdOp::BeginRenderPass: {
            if (!pipeline) [[unlikely]] {
                ++total_.stream_errors;
                return false;
            }
            if (!pass_regs_) [[unlikely]] {
                pass_regs_ = std::make_unique<Gcn::Regs>();
                pass_regs_->SetDefaults();
            }
            bb_native_stream::BeginRenderPassHeader ph{};
            std::memcpy(&ph, pay, sizeof(ph));
            const u32* tp = pay + (sizeof(ph) / sizeof(u32));

            auto& pr = *pass_regs_;
            std::memcpy(&pr.color_control, &ph.color_control, sizeof(u32));
            pr.color_target_mask.raw = ph.color_target_mask;
            pr.color_shader_mask.raw = ph.color_shader_mask;
            std::memcpy(&pr.blend_control[0], &ph.blend_control0, sizeof(u32));
            std::memcpy(&pr.depth_control, &ph.depth_control, sizeof(u32));
            std::memcpy(&pr.stencil_control, &ph.stencil_control, sizeof(u32));
            pr.reg_array[0xA10C] = ph.stencil_ref_front;
            pr.reg_array[0xA10D] = ph.stencil_ref_back;
            std::memcpy(&pr.depth_render_control, &ph.depth_render_control, sizeof(u32));
            std::memcpy(&pr.depth_render_override, &ph.depth_render_override, sizeof(u32));
            std::memcpy(&pr.depth_clear, &ph.depth_clear, sizeof(u32));
            std::memcpy(&pr.stencil_clear, &ph.stencil_clear, sizeof(u32));
            std::memcpy(&pr.depth_view, &ph.depth_view, sizeof(u32));
            std::memcpy(&pr.depth_htile_data_base, &ph.depth_htile_data_base, sizeof(u32));
            std::memcpy(&pr.aa_config, &ph.aa_config, sizeof(u32));
            std::memcpy(&pr.clipper_control, &ph.clipper_control, sizeof(u32));
            std::memcpy(&pr.mode_control, &ph.mode_control, sizeof(u32));
            std::memcpy(&pr.viewport_control, &ph.viewport_control, sizeof(u32));
            std::memcpy(&pr.screen_scissor, ph.screen_scissor, sizeof(ph.screen_scissor));
            std::memcpy(&pr.window_scissor, ph.window_scissor, sizeof(ph.window_scissor));
            std::memcpy(&pr.viewport_scissors[0], ph.viewport_scissor0,
                        sizeof(ph.viewport_scissor0));
            std::memcpy(&pr.viewports[0], ph.viewport0, sizeof(ph.viewport0));
            pr.num_indices = ph.num_indices;
            std::memcpy(&pr.num_instances, &ph.num_instances, sizeof(u32));
            pr.primitive_type = static_cast<Gcn::PrimitiveType>(pipe_pay.prim_type);

            if (!pass_guard.active) {
                pass_guard.active = true;
                pass_guard.prev_stage_regs = Rasterizer::stage_regs;
                pass_guard.prev_cb_ext = r.pipe_cb_extent;
                pass_guard.prev_cb_uid = r.pipe_cb_uid;
                pass_guard.prev_db_ext = r.pipe_db_extent;
                pass_guard.prev_db_uid = r.pipe_db_uid;
            }
            for (u32 c = 0; c < Gcn::NUM_COLOR_BUFFERS; ++c) {
                r.pipe_cb_uid[c] = ph.cb_uid[c];
                if ((ph.active_cb_mask & (1u << c)) != 0u) {
                    std::memcpy(&pr.color_buffers[c], tp, sizeof(Gcn::ColorBuffer));
                    tp += 15u;
                    r.pipe_cb_extent[c].raw = *tp++;
                } else {
                    std::memset(&pr.color_buffers[c], 0, sizeof(Gcn::ColorBuffer));
                    r.pipe_cb_extent[c].raw = 0u;
                }
            }
            if ((hdr.flags & bb_native_stream::kPassFlagHasCb0Hint) != 0u) {
                std::memcpy(&pr.color_buffers[0], tp, sizeof(Gcn::ColorBuffer));
                tp += 15u;
                r.pipe_cb_extent[0].raw = *tp++;
            }
            r.pipe_db_uid = ph.db_uid;
            if ((hdr.flags & bb_native_stream::kPassFlagHasDepth) != 0u) {
                std::memcpy(&pr.depth_buffer, tp, sizeof(Gcn::DepthBuffer));
                tp += 8u;
                r.pipe_db_extent.raw = *tp++;
            } else {
                std::memset(&pr.depth_buffer, 0, sizeof(Gcn::DepthBuffer));
                r.pipe_db_extent.raw = 0u;
            }
            Rasterizer::stage_regs = &pr;
            has_begin_pass_cmd = true;

            if (r.vsm_blur_batcher) {
                if (r.vsm_blur_batcher->HasDeferredCopy()) [[unlikely]] {
                    r.FlushDeferredVsmCopy();
                }
                if (r.vsm_blur_batcher->HasPendingCopyBacks()) [[unlikely]] {
                    const auto& col0 = pr.color_buffers[0];
                    const bool is_vsm_target =
                        col0 && col0.GetDataFmt() == Gcn::DataFormat::Format16_16 &&
                        col0.GetNumberFmt() == Gcn::NumberFormat::Float;
                    if (!is_vsm_target) {
                        r.FlushVsmBlurCopyBacks();
                    }
                }
            }
            r.PrepareRenderState(pipeline);
            if (r.vsm_blur_batcher && r.vsm_blur_batcher->HasPendingCopyBacks() &&
                r.cb_descs[0].first) [[unlikely]] {
                const auto& cb0_range = r.cb_descs[0].second.view_info.range;
                if (r.vsm_blur_batcher->ConflictsWithCopyBack(
                        r.cb_descs[0].first, cb0_range.base.level, cb0_range.base.layer)) {
                    r.FlushVsmBlurCopyBacks();
                }
            }
            break;
        }
        case bb_native_stream::CmdOp::SetVertexBuffers: {
            std::memcpy(&vb_hdr, pay, sizeof(vb_hdr));
            has_vb_cmd = true;
            vb_indexed = (hdr.flags & bb_native_stream::kVbFlagIndexed) != 0u;
            vb_index32 = (hdr.flags & bb_native_stream::kVbFlagIndex32) != 0u;
            vb_prim_restart = (hdr.flags & bb_native_stream::kVbFlagPrimRestart) != 0u;
            const bool reuse_vs = (hdr.flags & bb_native_stream::kVbFlagReuseVsharps) != 0u;
            if (reuse_vs) {
                r.packet_vsharps = {r.stage_b_last_vsharps.data(), vb_hdr.num_vbs};
                r.packet_vsharps_hash =
                    (vb_hdr.num_vbs == last_vsharps_count_) ? last_vsharps_hash_ : 0u;
            } else {
                const auto* vs_in = reinterpret_cast<const Gcn::Buffer*>(pay + 6u);
                if (vb_hdr.num_vbs > 0u && vb_hdr.num_vbs <= r.stage_b_last_vsharps.size()) {
                    std::memcpy(r.stage_b_last_vsharps.data(), vs_in,
                                vb_hdr.num_vbs * sizeof(Gcn::Buffer));
                    r.packet_vsharps = {r.stage_b_last_vsharps.data(), vb_hdr.num_vbs};
                } else {
                    r.packet_vsharps = {vs_in, vb_hdr.num_vbs};
                }
                r.packet_vsharps_hash = 0u;
                last_vsharps_hash_ = 0u;
                last_vsharps_count_ = 0u;
            }
            break;
        }
        case bb_native_stream::CmdOp::BindStageResources: {
            if (!pipeline) [[unlikely]] {
                ++total_.stream_errors;
                return false;
            }
            begin_resources();
            bb_native_stream::BindStageResourcesHeader bh{};
            std::memcpy(&bh, pay, sizeof(bh));
            const u32* p = pay + 1u;
            const u32 num_ud = hdr.flags;

            // Advance `stage_cursor` to the next non-null stage in `pipe_stages`.
            const Shader::Info* stage = nullptr;
            while (stage_cursor < pipe_stages.size()) {
                if (pipe_stages[stage_cursor]) {
                    stage = pipe_stages[stage_cursor++];
                    break;
                }
                ++stage_cursor;
            }
            if (!stage) [[unlikely]] {
                ++total_.stream_errors;
                return false;
            }

            // 1. Inline scalar push user-data dwords (`PushData::ud_regs`).
            if (num_ud != 0u) {
                std::memcpy(&r.push_data.ud_regs[binding.user_data], p, num_ud * sizeof(u32));
                binding.user_data += num_ud;
                p += num_ud;
            }

            // Pointers to SRV, Sampler, and CB slices in the stream.
            const auto* srv_ptr = reinterpret_cast<const bb_native_stream::NativeSrvBinding*>(p);
            if (stage->sw_stage == Shader::SwStage::Fragment && bh.num_srvs != 0u) {
                ps_first_srv_dw7 = srv_ptr[0].dw7_tag;
            }
            p += u32(bh.num_srvs) * 3u;
            const u32* smp_ptr = p;
            p += u32(bh.num_samplers);
            const auto* cb_ptr = reinterpret_cast<const bb_native_stream::NativeConstBinding*>(p);

            // 2. Bind Buffers FIRST (exact Vulkan descriptor order: [buffers][images][samplers]).
            if (bh.num_cbs != 0u) {
                BB_SECTION(BindBuffers);
                if (r.direct_buffers.NeedsProcessEvents(r.scheduler.IsRendering())) [[unlikely]] {
                    r.direct_buffers.ProcessEvents(r.memory);
                }
                const bool track_vs_motion =
                    r.motion_draw && stage->sw_stage == Shader::SwStage::Vertex &&
                    (r.upscaler->Enabled() || r.camera_motion->DebugOverlay());
                for (u32 i = 0; i < bh.num_cbs; ++i) {
                    const auto& cb = cb_ptr[i];
                    const auto& desc = stage->buffers[cb.slot];
                    if ((cb.flags & bb_native_stream::kNativeCbFlagSpecialNull) != 0u) {
                        r.buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                    } else if ((cb.flags & bb_native_stream::kNativeCbFlagSpecialBind) != 0u) {
                        r.BindSpecialBuffer(desc, *stage, binding.buffer, buf_alignment);
                    } else if (cb.ring_id != 0xFFu &&
                               cb.ring_offset < bb_native_stream::kNativeRingPending) [[likely]] {
                        ++total_.cb_ring_hits;
                        ++window_.cb_ring_hits;
                        if (BbCp::Diagnostics::DcbStatsEnabled()) [[unlikely]] {
                            BbCp::Diagnostics::NoteStageBBufAudit(0u, cb.byte_size);
                        }
                        if (!desc.IsSpecial()) {
                            if (cb.byte_size == 864u && r.gbuffer_draw &&
                                !r.camera_motion->PassHasCamera() && r.camera_motion->Enabled()) {
                                if (cb.gpu_va != 0 &&
                                    !r.buffer_cache.IsConstRamRange(cb.gpu_va, 864u) &&
                                    r.memory->ClampRangeSize(cb.gpu_va, 864u) == 864u) {
                                    r.camera_motion->OnConstants(
                                        reinterpret_cast<const float*>(cb.gpu_va));
                                } else {
                                    const u8* ring_data =
                                        cb.ring_id == 0u
                                            ? r.constant_ring->Data(cb.ring_offset)
                                            : r.worker_rings[cb.ring_id - 1u]->Data(cb.ring_offset);
                                    r.camera_motion->OnConstants(
                                        reinterpret_cast<const float*>(ring_data));
                                }
                            }
                            if (track_vs_motion &&
                                (cb.flags & bb_native_stream::kNativeCbFlagStride16) != 0u) {
                                const auto role = Motion::ClassifyBuffer(cb.byte_size);
                                if (role == Motion::BufferRole::Skeleton) {
                                    motion_gated = false;
                                } else if (role == Motion::BufferRole::SmallSkeleton &&
                                           motion_gated) {
                                    const u8* pal_ptr =
                                        cb.ring_id == 0u
                                            ? r.constant_ring->Data(cb.ring_offset)
                                            : r.worker_rings[cb.ring_id - 1u]->Data(cb.ring_offset);
                                    if (pal_ptr) {
                                        motion_palette = XXH3_64bits_withSeed(
                                            pal_ptr, cb.byte_size, motion_palette);
                                    }
                                }
                            }
                            r.push_data.AddOffset(binding.buffer, 0u);
                        }
                        r.buffer_infos.emplace_back(r.RingHandle(cb.ring_id), cb.ring_offset,
                                                    cb.byte_size);
                    } else {
                        ++total_.cb_direct_hits;
                        ++window_.cb_direct_hits;
                        if (cb.ring_offset >= bb_native_stream::kNativeRingPending &&
                            cb.ring_id != 0xFFu && BbCp::Diagnostics::DcbStatsEnabled())
                            [[unlikely]] {
                            BbCp::Diagnostics::NoteStageBBufAudit(1u, cb.byte_size);
                        }
                        if (track_vs_motion && !desc.IsSpecial() &&
                            (cb.flags & bb_native_stream::kNativeCbFlagStride16) != 0u) {
                            const auto role = Motion::ClassifyBuffer(cb.byte_size);
                            if (role == Motion::BufferRole::Skeleton) {
                                motion_gated = false;
                            } else if (role == Motion::BufferRole::SmallSkeleton && motion_gated &&
                                       cb.gpu_va != 0u) {
                                const u8* pal_ptr =
                                    r.direct_buffers.TryGetCpuMirrorPtr(cb.gpu_va, cb.byte_size);
                                if (!pal_ptr &&
                                    !VideoCore::DirectBuffers::OverlapsNativeOwnedRange(
                                        cb.gpu_va, cb.byte_size) &&
                                    r.memory->ClampRangeSize(cb.gpu_va, cb.byte_size) ==
                                        cb.byte_size) {
                                    pal_ptr = reinterpret_cast<const u8*>(cb.gpu_va);
                                }
                                if (pal_ptr) {
                                    motion_palette = XXH3_64bits_withSeed(
                                        pal_ptr, cb.byte_size, motion_palette);
                                }
                            }
                        }
                        Gcn::Buffer synthetic{};
                        synthetic.base_address = cb.gpu_va;
                        synthetic.num_records = cb.byte_size;
                        synthetic.stride = 0u;
                        r.BindGuestBuffer(desc, synthetic, *stage, binding.buffer, r.push_data,
                                          buf_alignment);
                    }
                    r.set_writes[r.set_write_index++] = vk::WriteDescriptorSet{
                        .dstBinding = binding.unified++,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                        .pBufferInfo = &r.buffer_infos.back(),
                    };
                    ++binding.buffer;
                }
            }

            // 3. Bind SRVs & Samplers directly by NativeSrvUid / NativeSamplerUid.
            if (bh.num_srvs != 0u || bh.num_samplers != 0u) {
                u32 srv_fast = 0u, srv_slow = 0u, smp_fast = 0u, smp_slow = 0u;
                r.BindNativeTextures(
                    *stage, {srv_ptr, bh.num_srvs}, {smp_ptr, bh.num_samplers},
                    cur_dyn_.ta_bc_base, binding, r.set_write_index, r.needs_barrier,
                    srv_fast, srv_slow, smp_fast, smp_slow);
                total_.srv_fast_hits += srv_fast;
                window_.srv_fast_hits += srv_fast;
                total_.srv_slow_hits += srv_slow;
                window_.srv_slow_hits += srv_slow;
                total_.smp_fast_hits += smp_fast;
                window_.smp_fast_hits += smp_fast;
                total_.smp_slow_hits += smp_slow;
                window_.smp_slow_hits += smp_slow;
            }
            break;
        }
        case bb_native_stream::CmdOp::DrawIndexed:
        case bb_native_stream::CmdOp::DrawAuto: {
            if (!pipeline) [[unlikely]] {
                ++total_.stream_errors;
                return false;
            }
            begin_resources();
            const bool is_indexed = (op == bb_native_stream::CmdOp::DrawIndexed);
            u32 num_indices = 0u;
            u32 num_instances = 1u;
            u32 first_vertex = 0u;
            u32 first_instance = 0u;
            if (is_indexed) {
                bb_native_stream::DrawIndexedPayload dp{};
                std::memcpy(&dp, pay, sizeof(dp));
                num_indices = dp.index_count;
                num_instances = dp.instance_count;
                first_vertex = static_cast<u32>(dp.vertex_offset);
                first_instance = dp.first_instance;
            } else {
                bb_native_stream::DrawAutoPayload dp{};
                std::memcpy(&dp, pay, sizeof(dp));
                num_indices = dp.vertex_count;
                num_instances = dp.instance_count;
                first_vertex = dp.first_vertex;
                first_instance = dp.first_instance;
            }

            // Pass start (`BeginRenderPass`) or steady-pass (`begin_memo` fast path).
            RenderState pass_state{};
            bool will_begin_pass = false;
            const auto& gkey = pipeline->GetGraphicsKey();
            if (has_begin_pass_cmd) [[unlikely]] {
                pass_state = r.BeginRendering(pipeline);
                will_begin_pass = !r.last_begin_memo_hit && r.scheduler.WillBeginRendering(pass_state);
            } else {
                if (r.vsm_blur_batcher) {
                    if (r.vsm_blur_batcher->HasDeferredCopy()) [[unlikely]] {
                        r.FlushDeferredVsmCopy();
                    }
                    if (r.vsm_blur_batcher->HasPendingCopyBacks()) [[unlikely]] {
                        r.FlushVsmBlurCopyBacks();
                    }
                }
                if (r.upscaler->Enabled() && r.db_desc.first &&
                    r.db_desc.first == r.camera_motion->Depth() && r.cb_descs[0].first &&
                    std::popcount(gkey.mrt_mask) <= 2) {
                    const auto& color = r.texture_cache.GetImage(r.cb_descs[0].first);
                    const auto& depth = r.texture_cache.GetImage(r.db_desc.first);
                    if (color.info.pixel_format == vk::Format::eR16G16B16A16Sfloat &&
                        color.info.size.width == depth.info.size.width &&
                        color.info.size.height == depth.info.size.height) {
                        r.upscaler->OnSceneColor(r.cb_descs[0].first);
                        const auto* cds = DirectStates::GetCompiledDepthStencilDyn(pipe_pay.ds_id);
                        const bool dw = cds ? cds->depth_write_enable : false;
                        if (gkey.blend_controls[0].enable && !dw &&
                            (num_indices > 6u || num_instances > 1u)) {
                            r.upscaler->OnBlendedSceneDraw();
                        } else if (num_indices <= 6u && num_instances <= 1u) {
                            r.upscaler->OnSceneComposite();
                        }
                    }
                }
                r.attachment_feedback_loop = false;
                will_begin_pass =
                    r.needs_barrier || r.runtime.HasPendingBarriers() ||
                    (r.begin_memo.valid && !r.scheduler.IsRenderingWith(r.begin_memo.state));
            }

            // Resolve vertex & index buffers directly from `SetVertexBuffers` (zero `Gcn::Regs`).
            if (has_vb_cmd) {
                const Rasterizer::DirectVertexDrawInfo dv_info{
                    .index_va = vb_hdr.index_va,
                    .num_indices = num_indices,
                    .num_instances = num_instances,
                    .first_vertex = first_vertex,
                    .first_instance = first_instance,
                    .prim_restart_idx = vb_hdr.prim_restart_idx,
                    .is_indexed = is_indexed,
                    .index16 = !vb_index32,
                    .prim_restart = vb_prim_restart,
                };
                r.ResolveVertexBuffersDirect(pipeline, vb_hdr.step_rate_0, vb_hdr.step_rate_1,
                                             &dv_info);
                if (r.packet_vsharps_hash != 0u && vb_hdr.num_vbs > 0u &&
                    vb_hdr.num_vbs <= r.stage_b_last_vsharps.size()) {
                    last_vsharps_hash_ = r.packet_vsharps_hash;
                    last_vsharps_count_ = vb_hdr.num_vbs;
                }
                if (is_indexed) {
                    r.ResolveIndexBufferDirect(vb_hdr.index_va, !vb_index32, num_indices);
                }
            }

            const bool need_flush_barriers =
                r.needs_barrier || (will_begin_pass && r.runtime.HasPendingBarriers());
            if (will_begin_pass && HasPendingHleCopies()) [[unlikely]] {
                FlushPendingHleCopies(r);
            }
            if (need_flush_barriers) [[unlikely]] {
                if (r.scheduler.HasPendingReorderBuckets()) {
                    r.scheduler.FlushReorderBuckets();
                }
                r.runtime.FlushBarriers();
            }

            const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
            // Populate `PushData` scaling, Ultrawide `UiEdgeStretch`, and per-frame FX masks
            // cleanly without touching `Gcn::Regs` or `Shader::Info::ud_snapshots`.
            if (!has_begin_pass_cmd && r.begin_memo.valid) [[likely]] {
                r.push_data.scene_size = r.begin_memo.scene_size;
                r.target_scale = r.begin_memo.target_scale;
                r.ui_placement = r.begin_memo.ui_placement;
            }
            const auto saved_ui_placement = r.ui_placement;
            if (saved_ui_placement.edge_x != 0.0f && !gkey.motion_vectors) [[unlikely]] {
                const auto* crs = DirectStates::GetCompiledRasterDyn(pipe_pay.raster_id);
                const auto* cds = DirectStates::GetCompiledDepthStencilDyn(pipe_pay.ds_id);
                const bool clip_dis = (crs && (crs->vp_flags & 1u) != 0u);
                const bool st_active =
                    (cds && cds->stencil_enable && ((pipe_pay.pad & 2u) != 0u));
                if (!clip_dis && !st_active) {
                    bool big_tex = false;
                    if (ps_first_srv_dw7 != 0u && !BbToggle::Experiment(32)) {
                        const u16 uid = static_cast<u16>(ps_first_srv_dw7 & 0xFFFFu);
                        const u8 gen8 = static_cast<u8>((ps_first_srv_dw7 >> 16) & 0xFFu);
                        Gcn::Image tsharp{};
                        if ((ps_first_srv_dw7 >> 24) == VideoCore::DirectTextures::kSrvTagMagic &&
                            VideoCore::DirectTextures::ReadNativeSrvByUid(uid, gen8, tsharp)) {
                            big_tex = (u32(tsharp.width) + 1u >= 2048u ||
                                       u32(tsharp.height) + 1u >= 2048u);
                        }
                    }
                    if (!big_tex) {
                        r.push_data.motion_param = UiComposition::EdgeParam(
                            saved_ui_placement.edge_x, saved_ui_placement.edge_y,
                            cur_dyn_.vp_yscale > 0.0f);
                    }
                }
            }
            r.ui_placement = {};
            r.ScaleToTarget(pipeline);
            r.ui_placement = saved_ui_placement;

            // Per-object motion vectors for G-Buffer draws (`motion_draw && motion_geometry`)
            // directly from `SetVertexBuffers` + VS `NativeConstBinding` (zero `Gcn::Regs`).
            if (r.motion_draw && r.motion_geometry) {
                r.push_data.motion_param = 0;
                if (r.upscaler->Enabled() || r.camera_motion->DebugOverlay()) {
                    BB_SECTION(Motion);
                    const u32 index_size = vb_index32 ? 4u : 2u;
                    const VAddr index_address = is_indexed ? vb_hdr.index_va : 0u;
                    if (!motion_gated ||
                        r.object_motion->Moving({.shader = vs_info.pgm_hash,
                                                 .geometry = r.motion_geometry,
                                                 .indices = index_address,
                                                 .index_count = num_indices,
                                                 .instances = num_instances,
                                                 .first_instance = first_instance},
                                                motion_palette)) {
                        Motion::VertexRange range{first_vertex, num_indices};
                        u64 topology = 0;
                        if (is_indexed) {
                            range = {};
                            const u64 bytes = u64(num_indices) * index_size;
                            const u8* ib_data =
                                index_address
                                    ? r.direct_buffers.TryGetCpuMirrorPtr(index_address, bytes)
                                    : nullptr;
                            if (!ib_data && index_address &&
                                !VideoCore::DirectBuffers::OverlapsNativeOwnedRange(index_address,
                                                                                    bytes) &&
                                r.memory->ClampRangeSize(index_address, bytes) == bytes) {
                                ib_data = reinterpret_cast<const u8*>(index_address);
                            }
                            if (ib_data) {
                                const auto scanned = r.object_motion->IndexRange(
                                    {index_address, num_indices, index_size, s32(first_vertex),
                                     vb_prim_restart},
                                    [&] {
                                        const auto scan_range =
                                            index_size == 2u
                                                ? Motion::IndexedRange(
                                                      std::span(
                                                          reinterpret_cast<const u16*>(ib_data),
                                                          num_indices),
                                                      s32(first_vertex), vb_prim_restart)
                                                : Motion::IndexedRange(
                                                      std::span(
                                                          reinterpret_cast<const u32*>(ib_data),
                                                          num_indices),
                                                      s32(first_vertex), vb_prim_restart);
                                        return Motion::IndexRangeCache::Result{
                                            scan_range, XXH3_64bits(ib_data, bytes)};
                                    });
                                range = scanned.range;
                                topology = scanned.topology;
                            }
                        }
                        r.push_data.motion_param = r.object_motion->PrepareDraw({
                            .shader = vs_info.pgm_hash,
                            .geometry = r.motion_geometry,
                            .indices = index_address,
                            .topology = topology,
                            .index_count = num_indices,
                            .instances = num_instances,
                            .first_instance = first_instance,
                            .vertices = range,
                        });
                    }
                }
            }

            const bool jitter_eligible =
                vs_info.pgm_hash != 0x000000008747a367ULL &&
                (num_indices > 6u || num_instances > 1u);
            r.draw_jitter = {};
            if (jitter_eligible && r.upscaler->Enabled() && r.db_desc.first &&
                r.db_desc.first == r.camera_motion->Depth()) {
                r.draw_jitter = r.upscaler->Jitter();
            }

            if (will_begin_pass || need_flush_barriers) [[unlikely]] {
                r.scheduler.GetDynamicState().InvalidateViewports();
                if (!has_begin_pass_cmd) {
                    ++total_.mid_pass_rebinds;
                    ++window_.mid_pass_rebinds;
                }
            }

            const bool depth_valid = (pipe_pay.pad & 1u) != 0u;
            const bool stencil_valid = (pipe_pay.pad & 2u) != 0u;
            ApplyNativeDynamicState(
                r, pipeline, pipe_pay.raster_id, pipe_pay.ds_id,
                static_cast<Gcn::PrimitiveType>(pipe_pay.prim_type), is_indexed, vb_prim_restart,
                vb_hdr.prim_restart_idx, depth_valid, stencil_valid, dyn_flags);

            if (has_begin_pass_cmd) [[unlikely]] {
                if (will_begin_pass || need_flush_barriers || !r.scheduler.IsRenderingWith(pass_state)) {
                    r.scheduler.BeginRendering(pass_state, 0u);
                }
                if (r.begin_memo.valid) {
                    r.begin_memo.cmd_epoch = r.scheduler.DescriptorStateEpoch();
                }
            } else if ((will_begin_pass || need_flush_barriers) && r.begin_memo.valid) [[unlikely]] {
                r.scheduler.BeginRendering(r.begin_memo.state, 0u);
                r.begin_memo.cmd_epoch = r.scheduler.DescriptorStateEpoch();
            }

            // Emit vertex & index buffers, push descriptors, bind pipeline (deduped), draw!
            r.EmitVertexBuffers();
            if (is_indexed) {
                BB_SECTION(BDrawEmit);
                r.EmitIndexBuffer();
            }
            {
                BB_SECTION(PipelineBind);
                pipeline->BindResources({r.set_writes.data(), r.set_write_index}, r.push_data,
                                        {r.image_infos.data(), r.image_infos.size()},
                                        {r.buffer_infos.data(), r.buffer_infos.size()}, true);
            }
            {
                BB_SECTION(BDrawEmit);
                const vk::Pipeline handle = pipeline->Handle();
                const u64 cur_epoch = r.scheduler.DescriptorStateEpoch();
                const bool need_bind_pipeline =
                    (r.bound_graphics_epoch != cur_epoch) || (r.bound_graphics_pipeline != handle);
                r.bound_graphics_pipeline = handle;
                r.bound_graphics_epoch = cur_epoch;
                vk::QueryPool occ_pool{};
                u32 q_slot = 0u;
                const bool has_occ_query =
                    is_indexed && r.TryAllocateSceneOccQuery(occ_handoff, occ_pool, q_slot);
                if (is_indexed) {
                    if (has_occ_query) {
                        r.scheduler.Record([handle, need_bind_pipeline, occ_pool, q_slot,
                                            num_indices, num_instances, first_vertex,
                                            first_instance](vk::CommandBuffer cmdbuf) {
                            if (need_bind_pipeline) {
                                cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
                            }
                            cmdbuf.beginQuery(occ_pool, q_slot, {});
                            cmdbuf.drawIndexed(num_indices, num_instances, 0, s32(first_vertex),
                                               first_instance);
                            cmdbuf.endQuery(occ_pool, q_slot);
                        });
                    } else if (need_bind_pipeline) {
                        r.scheduler.Record([handle, num_indices, num_instances, first_vertex,
                                            first_instance](vk::CommandBuffer cmdbuf) {
                            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
                            cmdbuf.drawIndexed(num_indices, num_instances, 0, s32(first_vertex),
                                               first_instance);
                        });
                    } else {
                        r.scheduler.Record([num_indices, num_instances, first_vertex,
                                            first_instance](vk::CommandBuffer cmdbuf) {
                            cmdbuf.drawIndexed(num_indices, num_instances, 0, s32(first_vertex),
                                               first_instance);
                        });
                    }
                } else {
                    if (need_bind_pipeline) {
                        r.scheduler.Record([handle, num_indices, num_instances, first_vertex,
                                            first_instance](vk::CommandBuffer cmdbuf) {
                            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
                            cmdbuf.draw(num_indices, num_instances, first_vertex, first_instance);
                        });
                    } else {
                        r.scheduler.Record([num_indices, num_instances, first_vertex,
                                            first_instance](vk::CommandBuffer cmdbuf) {
                            cmdbuf.draw(num_indices, num_instances, first_vertex, first_instance);
                        });
                    }
                }
            }

            {
                BB_SECTION(Kick);
                r.ResetBindings(false);
                r.scheduler.KickRecording();
            }
            drew = true;
            ++total_.native_draws;
            ++window_.native_draws;
            BbStats::native_stream_draws.fetch_add(1u, std::memory_order_relaxed);
            break;
        }
        default:
            break;
        }
    }

    MaybeLogAudit();
    return drew;
}

} // namespace Vulkan
