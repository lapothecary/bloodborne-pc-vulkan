// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 3B-1 / 3B-2 (Stage A elimination): Live NativeCmdStream shadow producer, Gcn::Regs
// parity verifier, and NativeSrvUid/NativeSamplerUid/NativeVkBuf/NativePsoId coverage audit.

#include "render/native_stream_shadow.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "buffers/direct_buffers.h"
#include "cp/cp_packets.h"
#include "render/direct_states.h"
#include "render/pipeline.h"
#include "textures/direct_textures.h"

namespace Vulkan {

namespace {

inline u64 NowSteadyNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

inline float BitsToFloat(u32 bits) noexcept {
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

inline bb_native_stream::SetDynamicStatePayload ExtractDynamicState(const Gcn::Regs& regs) noexcept {
    bb_native_stream::SetDynamicStatePayload d{};
    const auto& ctl = regs.viewport_control;
    const auto& vp = regs.viewports[0];
    d.vp_xscale = ctl.xscale_enable ? vp.xscale : 1.0f;
    d.vp_xoffset = ctl.xoffset_enable ? vp.xoffset : 0.0f;
    d.vp_yscale = ctl.yscale_enable ? vp.yscale : 1.0f;
    d.vp_yoffset = ctl.yoffset_enable ? vp.yoffset : 0.0f;
    const float zoffset = ctl.zoffset_enable ? vp.zoffset : 0.0f;
    const float zscale = ctl.zscale_enable ? vp.zscale : 1.0f;
    d.vp_zmin = (regs.clipper_control.clip_space == Gcn::ClipSpace::MinusWToW) ? (zoffset - zscale)
                                                                              : zoffset;
    d.vp_zmax = zoffset + zscale;

    const u32 sc0 = regs.reg_array[0xA094];
    const u32 sc1 = regs.reg_array[0xA095];
    d.sc_left = static_cast<int16_t>(sc0 & 0x7FFFu);
    d.sc_top = static_cast<int16_t>((sc0 >> 16) & 0x7FFFu);
    d.sc_right = static_cast<int16_t>(sc1 & 0x7FFFu);
    d.sc_bottom = static_cast<int16_t>((sc1 >> 16) & 0x7FFFu);

    for (u32 i = 0; i < 4u; ++i) {
        d.blend_rgba[i] = BitsToFloat(regs.reg_array[0xA105 + i]);
    }
    d.depth_bias_const = BitsToFloat(regs.reg_array[0xA2E0]);
    d.depth_bias_slope = BitsToFloat(regs.reg_array[0xA2E1]);
    d.depth_bias_clamp = BitsToFloat(regs.reg_array[0xA2DF]);
    d.depth_bounds_min = BitsToFloat(regs.reg_array[0xA008]);
    d.depth_bounds_max = BitsToFloat(regs.reg_array[0xA009]);
    d.stencil_ref_front = regs.reg_array[0xA10C];
    d.stencil_ref_back = regs.reg_array[0xA10D];
    d.line_width = regs.line_control.Width();
    d.ta_bc_base = regs.ta_bc_base.base_addr;
    return d;
}

} // namespace

void NativeStreamShadow::BeginSubmission() noexcept {
    if (storage_.empty()) {
        storage_.resize(kInitialCapacityDwords);
    } else if (sub_overflow_) {
        storage_.resize(storage_.size() * 2u);
    }
    writer_.Reset(storage_.data(), static_cast<u32>(storage_.size()));
    has_targets_ = false;
    has_pipe_ = false;
    has_dyn_ = false;
    has_pending_direct_ = false;
    last_targets_ = {};
    last_pipe_ = {};
    last_dyn_ = {};
    sub_draws_3d_ = 0;
    sub_draws_pm4_ = 0;
    sub_packets_ = 0;
    sub_overflow_ = false;
    active_ = true;
}

void NativeStreamShadow::CompleteDirectDraw(const Gcn::Regs& regs,
                                            const Gcn::CbDbExtent* cb_extents, const u16* cb_uids,
                                            u16 db_uid, u16 rs_id, u16 bs_id, u16 ds_id,
                                            const GraphicsPipeline& pipeline) noexcept {
    if (!has_pending_direct_) {
        return;
    }
    const u32 op = pending_direct_op_;
    const u32 start_idx = pending_start_index_;
    has_pending_direct_ = false;
    RecordDirectDraw(regs, cb_extents, cb_uids, db_uid, rs_id, bs_id, ds_id, op, start_idx,
                     pipeline.GetNativePsoId(), &pipeline);
}

void NativeStreamShadow::VerifyDirectStateParity(const Gcn::Regs& regs, u16 rs_id, u16 bs_id,
                                                 u16 ds_id) noexcept {
    if (rs_id != 0u) {
        ++total_.rs_covered;
        ++window_.rs_covered;
        DirectStates::RasterStateDesc rdesc{};
        if (DirectStates::ReadRasterState(rs_id, rdesc)) {
            const u32 exp_clip = rdesc.primitive_setup & 0x0F7FE03Fu;
            const u32 exp_poly = rdesc.clip_control & 0x00393FFFu;
            const u32 exp_mode0 = (rdesc.msaa_enable & 1u) | ((rdesc.scissor_enable << 1) & 2u);
            const u32 act_clip = regs.reg_array[0xA204] & 0x0F7FE03Fu;
            const u32 act_poly = regs.reg_array[0xA205] & 0x00393FFFu;
            const u32 act_mode0 = regs.reg_array[0xA292] & 3u;
            if (exp_clip != act_clip || exp_poly != act_poly || exp_mode0 != act_mode0) {
                ++total_.rs_mismatches;
                ++window_.rs_mismatches;
                if (logged_mismatches_ < 8u) {
                    ++logged_mismatches_;
                    std::printf(
                        "[NativeStream-Parity] RS mismatch id=%u vs=%#llx ps=%#llx "
                        "A204(exp=%#x act=%#x) A205(exp=%#x act=%#x) A292(exp=%#x act=%#x)\n",
                        u32(rs_id),
                        static_cast<unsigned long long>(regs.vs_program.Address<VAddr>()),
                        static_cast<unsigned long long>(regs.ps_program.Address<VAddr>()), exp_clip,
                        act_clip, exp_poly, act_poly, exp_mode0, act_mode0);
                }
            }
        }
    }

    if (bs_id != 0u) {
        ++total_.bs_covered;
        ++window_.bs_covered;
        DirectStates::BlendStateDesc bdesc{};
        if (DirectStates::ReadBlendState(bs_id, bdesc)) {
            bool ok = (bdesc.render_target_channel_mask == regs.reg_array[0xA08E]);
            u32 bad_slot = 0xFFu;
            for (u32 s = 0; s < 8u && ok; ++s) {
                if ((bdesc.blend_control[s] & 0xFFFF1FFFu) !=
                    (regs.reg_array[0xA1E0 + s] & 0xFFFF1FFFu)) {
                    ok = false;
                    bad_slot = s;
                }
            }
            if (!ok) {
                ++total_.bs_mismatches;
                ++window_.bs_mismatches;
                if (logged_mismatches_ < 8u) {
                    ++logged_mismatches_;
                    std::printf(
                        "[NativeStream-Parity] BS mismatch id=%u vs=%#llx ps=%#llx "
                        "mask(exp=%#x act=%#x) bad_slot=%u\n",
                        u32(bs_id),
                        static_cast<unsigned long long>(regs.vs_program.Address<VAddr>()),
                        static_cast<unsigned long long>(regs.ps_program.Address<VAddr>()),
                        bdesc.render_target_channel_mask, regs.reg_array[0xA08E], bad_slot);
                }
            }
        }
    }

    if (ds_id != 0u) {
        ++total_.ds_covered;
        ++window_.ds_covered;
        DirectStates::DepthStencilStateDesc ddesc{};
        if (DirectStates::ReadDepthStencilState(ds_id, ddesc)) {
            const u32 exp_dc = ddesc.depth_control & 0xC07007FFu;
            const u32 exp_sc = ddesc.stencil_control & 0x00FFFFFFu;
            const u32 exp_so = ddesc.stencil_op_control & 0x00FFFF00u;
            const u32 act_dc = regs.reg_array[0xA200] & 0xC07007FFu;
            const u32 act_sc = regs.reg_array[0xA10B] & 0x00FFFFFFu;
            const u32 act_so = regs.reg_array[0xA10C] & 0x00FFFF00u;
            if (exp_dc != act_dc || exp_sc != act_sc || exp_so != act_so) {
                ++total_.ds_mismatches;
                ++window_.ds_mismatches;
                if (logged_mismatches_ < 8u) {
                    ++logged_mismatches_;
                    std::printf(
                        "[NativeStream-Parity] DS mismatch id=%u vs=%#llx ps=%#llx "
                        "A200(exp=%#x act=%#x) A10B(exp=%#x act=%#x) A10C(exp=%#x act=%#x)\n",
                        u32(ds_id),
                        static_cast<unsigned long long>(regs.vs_program.Address<VAddr>()),
                        static_cast<unsigned long long>(regs.ps_program.Address<VAddr>()), exp_dc,
                        act_dc, exp_sc, act_sc, exp_so, act_so);
                }
            }
        }
    }
}

void NativeStreamShadow::EmitDynamicStateDelta(const Gcn::Regs& regs) noexcept {
    const auto dyn = ExtractDynamicState(regs);
    u8 flags = 0u;
    if (!has_dyn_) {
        flags = 0xFFu;
    } else {
        if (std::memcmp(&dyn.vp_xscale, &last_dyn_.vp_xscale, 6u * sizeof(float)) != 0) {
            flags |= bb_native_stream::kDynFlagViewport;
        }
        if (dyn.sc_left != last_dyn_.sc_left || dyn.sc_top != last_dyn_.sc_top ||
            dyn.sc_right != last_dyn_.sc_right || dyn.sc_bottom != last_dyn_.sc_bottom) {
            flags |= bb_native_stream::kDynFlagScissor;
        }
        if (std::memcmp(dyn.blend_rgba, last_dyn_.blend_rgba, sizeof(dyn.blend_rgba)) != 0) {
            flags |= bb_native_stream::kDynFlagBlendColor;
        }
        if (dyn.stencil_ref_front != last_dyn_.stencil_ref_front ||
            dyn.stencil_ref_back != last_dyn_.stencil_ref_back) {
            flags |= bb_native_stream::kDynFlagStencilRef;
        }
        if (std::memcmp(&dyn.depth_bias_const, &last_dyn_.depth_bias_const, 3u * sizeof(float)) !=
            0) {
            flags |= bb_native_stream::kDynFlagDepthBias;
        }
        if (std::memcmp(&dyn.depth_bounds_min, &last_dyn_.depth_bounds_min, 2u * sizeof(float)) !=
            0) {
            flags |= bb_native_stream::kDynFlagDepthBounds;
        }
        if (dyn.line_width != last_dyn_.line_width) {
            flags |= bb_native_stream::kDynFlagLineWidth;
        }
        if (dyn.ta_bc_base != last_dyn_.ta_bc_base) {
            flags |= bb_native_stream::kDynFlagBorderColor;
        }
    }
    if (flags != 0u) {
        if (writer_.Emit(bb_native_stream::CmdOp::SetDynamicState, flags, dyn)) {
            ++sub_packets_;
            ++total_.dyn_packets;
            ++window_.dyn_packets;
            last_dyn_ = dyn;
            has_dyn_ = true;
        } else {
            sub_overflow_ = true;
        }
    }
}

void NativeStreamShadow::EmitStageResources(const GraphicsPipeline& pipeline) noexcept {
    VAddr cue_base = 0;
    VAddr cue_end = 0;
    VideoCore::DirectBuffers::GetTransientCueHeapBounds(cue_base, cue_end);

    for (const Shader::Info* stage : pipeline.GetStages()) {
        if (!stage) {
            continue;
        }
        const u32 num_srvs = static_cast<u32>(std::min<size_t>(stage->images.size(), 255u));
        const u32 num_smps = static_cast<u32>(std::min<size_t>(stage->samplers.size(), 255u));
        const u32 num_cbs = static_cast<u32>(std::min<size_t>(stage->buffers.size(), 255u));
        if (num_srvs == 0u && num_smps == 0u && num_cbs == 0u) {
            continue;
        }
        const u16 payload_dwords =
            static_cast<u16>(1u + num_srvs * 3u + num_smps + num_cbs * 6u);
        u32* out = writer_.Reserve(bb_native_stream::CmdOp::BindStageResources, 0u, payload_dwords);
        if (!out) {
            sub_overflow_ = true;
            return;
        }
        ++sub_packets_;
        ++total_.res_packets;
        ++window_.res_packets;

        bb_native_stream::BindStageResourcesHeader hdr{};
        hdr.hw_stage = static_cast<u8>(stage->hw_stage);
        hdr.num_srvs = static_cast<u8>(num_srvs);
        hdr.num_samplers = static_cast<u8>(num_smps);
        hdr.num_cbs = static_cast<u8>(num_cbs);
        std::memcpy(out, &hdr, sizeof(hdr));
        out += 1u;

        const u32* const flat_ptr = stage->flattened_ud_buf.data();
        const VAddr pgm_base = stage->pgm_base;

        for (u32 i = 0; i < num_srvs; ++i) {
            Gcn::Image tsharp = stage->images[i].GetSharpDirect(flat_ptr);
            u32 raw_dw[8];
            std::memcpy(raw_dw, &tsharp, sizeof(raw_dw));
            u8 gen8 = 0u;
            const u16 uid = VideoCore::DirectTextures::UnpackAndCleanSrv(tsharp, gen8);
            const bb_native_stream::NativeSrvBinding sb{raw_dw[7], raw_dw[0], raw_dw[1]};
            std::memcpy(out, &sb, sizeof(sb));
            out += 3u;
            ++total_.srvs_total;
            ++window_.srvs_total;
            if (uid != 0u) {
                ++total_.srvs_uid_hit;
                ++window_.srvs_uid_hit;
            }
        }

        for (u32 s = 0; s < num_smps; ++s) {
            Gcn::Sampler ssharp = stage->samplers[s].GetSharpDirect(flat_ptr);
            const u16 uid = VideoCore::DirectTextures::UnpackAndCleanSampler(ssharp);
            *out++ = (s << 16) | u32(uid);
            ++total_.samplers_total;
            ++window_.samplers_total;
            if (uid != 0u) {
                ++total_.samplers_uid_hit;
                ++window_.samplers_uid_hit;
            }
        }

        for (u32 b = 0; b < num_cbs; ++b) {
            const auto& desc = stage->buffers[b];
            const Gcn::Buffer vsharp = desc.GetSharpDirect(flat_ptr, pgm_base);
            const VAddr va = vsharp.base_address;
            const u32 sz = vsharp.GetSize();
            const bool is_cue = (va != 0u && va >= cue_base && va < cue_end);
            const bool is_native =
                (!is_cue && va != 0u && VideoCore::DirectBuffers::IsNativeOwnedAddress(va));
            bb_native_stream::NativeConstBinding cb{};
            cb.gpu_va = va;
            cb.byte_size = sz;
            cb.ring_offset = bb_native_stream::kNativeRingNotCopied;
            cb.slot = static_cast<u16>(b);
            cb.ring_id = 0xFFu;
            cb.flags = static_cast<u8>((is_cue ? 1u : 0u) | (is_native ? 2u : 0u) |
                                       (desc.is_written ? 4u : 0u) | (desc.is_formatted ? 8u : 0u));
            std::memcpy(out, &cb, sizeof(cb));
            out += 6u;
            ++total_.bufs_total;
            ++window_.bufs_total;
            if (is_cue) {
                ++total_.bufs_cue;
                ++window_.bufs_cue;
            } else if (is_native) {
                ++total_.bufs_native;
                ++window_.bufs_native;
            }
        }
    }
}

void NativeStreamShadow::RecordDirectDraw(const Gcn::Regs& regs, const Gcn::CbDbExtent* cb_extents,
                                          const u16* cb_uids, u16 db_uid, u16 rs_id, u16 bs_id,
                                          u16 ds_id, u32 direct_op, u32 start_index, u16 pso_id,
                                          const GraphicsPipeline* pipeline) noexcept {
    if (!active_) {
        BeginSubmission();
    }
    VerifyDirectStateParity(regs, rs_id, bs_id, ds_id);

    if (pso_id != 0u) {
        ++total_.pso_id_covered;
        ++window_.pso_id_covered;
    }

    // 1. BeginPassTargets (delta-emitted on change).
    bb_native_stream::BeginPassTargetsPayload tgt{};
    const u32 target_mask = regs.color_target_mask.raw & regs.color_shader_mask.raw;
    u8 cb_mask = 0u;
    bool any_rt_uid = (db_uid != 0u);
    for (u32 i = 0; i < 8u; ++i) {
        tgt.cb_uid[i] = cb_uids ? cb_uids[i] : 0u;
        if (tgt.cb_uid[i] != 0u) {
            any_rt_uid = true;
        }
        if (((target_mask >> (i * 4u)) & 0xFu) != 0u && bool(regs.color_buffers[i])) {
            cb_mask |= static_cast<u8>(1u << i);
        }
    }
    if (any_rt_uid) {
        ++total_.rt_covered;
        ++window_.rt_covered;
    }
    tgt.db_uid = db_uid;
    tgt.cb_mask = cb_mask;
    tgt.aa_samples_log2 = static_cast<u8>(regs.aa_config.msaa_num_samples & 0x7u);
    u32 w = cb_extents ? cb_extents[0].width : 0u;
    u32 h = cb_extents ? cb_extents[0].height : 0u;
    if ((w == 0u || h == 0u) && bool(regs.color_buffers[0])) {
        w = regs.color_buffers[0].Pitch();
        h = regs.color_buffers[0].Height();
    }
    tgt.width = static_cast<u16>(w);
    tgt.height = static_cast<u16>(h);

    if (!has_targets_ || std::memcmp(&tgt, &last_targets_, sizeof(tgt)) != 0) {
        if (writer_.Emit(bb_native_stream::CmdOp::BeginPassTargets, 0u, tgt)) {
            ++sub_packets_;
            last_targets_ = tgt;
            has_targets_ = true;
        } else {
            sub_overflow_ = true;
        }
    }

    // 2. SetPipelineState (delta-emitted on change).
    bb_native_stream::SetPipelineStatePayload pipe{};
    pipe.vs_addr = regs.vs_program.Address<VAddr>();
    pipe.ps_addr = regs.ps_program.Address<VAddr>();
    pipe.vs_rsrc1 = regs.reg_array[0x2C4A];
    pipe.ps_rsrc1 = regs.reg_array[0x2C0A];
    pipe.raster_id = rs_id;
    pipe.blend_id = bs_id;
    pipe.ds_id = ds_id;
    pipe.pso_id = pso_id;
    pipe.prim_type = static_cast<u8>(u32(regs.primitive_type) & 0x3Fu);
    pipe.stage_enable = static_cast<u8>(regs.stage_enable.raw & 0xFFu);
    pipe.num_interp = static_cast<u8>(regs.num_interp & 0xFFu);
    pipe.pad = 0u;
    pipe.reserved = 0u;

    if (!has_pipe_ || std::memcmp(&pipe, &last_pipe_, sizeof(pipe)) != 0) {
        if (writer_.Emit(bb_native_stream::CmdOp::SetPipelineState, 0u, pipe)) {
            ++sub_packets_;
            last_pipe_ = pipe;
            has_pipe_ = true;
        } else {
            sub_overflow_ = true;
        }
    }

    // 3. SetDynamicState (delta-bitmasked).
    EmitDynamicStateDelta(regs);

    // 4. BindStageResources (flat NativeSrvUid / NativeSamplerUid / NativeConstBinding).
    if (pipeline != nullptr) {
        EmitStageResources(*pipeline);
    }

    // 5. Native draw packet matching DirectVk draw opcode.
    bool emitted = false;
    const auto op = static_cast<BbCp::DirectVk::Op>(direct_op);
    if (op == BbCp::DirectVk::Op::DrawAuto) {
        bb_native_stream::DrawAutoPayload d{};
        d.vertex_count = regs.reg_array[0xC24C];   // VGT_NUM_INDICES
        d.instance_count = regs.reg_array[0xC24D]; // VGT_NUM_INSTANCES
        d.first_vertex = regs.reg_array[0xA102];   // VGT_INDX_OFFSET
        d.first_instance = 0u;
        emitted = writer_.Emit(bb_native_stream::CmdOp::DrawAuto, 0u, d);
    } else if (op == BbCp::DirectVk::Op::DrawIndirect) {
        bb_native_stream::DrawIndirectPayload d{};
        d.args_va = 0u;
        d.count = 1u;
        d.stride = 0u;
        d.is_indexed = 1u;
        d.reserved = 0u;
        emitted = writer_.Emit(bb_native_stream::CmdOp::DrawIndirect, 0u, d);
    } else {
        bb_native_stream::DrawIndexedPayload d{};
        d.index_count = regs.reg_array[0xC24C];   // VGT_NUM_INDICES
        d.instance_count = regs.reg_array[0xC24D]; // VGT_NUM_INSTANCES
        d.first_index = start_index;
        d.vertex_offset = static_cast<int32_t>(regs.reg_array[0xA102]); // VGT_INDX_OFFSET
        d.first_instance = 0u;
        emitted = writer_.Emit(bb_native_stream::CmdOp::DrawIndexed, 0u, d);
    }

    if (emitted) {
        ++sub_packets_;
    } else {
        sub_overflow_ = true;
    }
    ++sub_draws_3d_;
    ++total_.draws_3d;
    ++window_.draws_3d;
}

void NativeStreamShadow::RecordLegacyPm4Draw(const u32* packet, u32 total_dwords) noexcept {
    if (!active_) {
        BeginSubmission();
    }
    has_pending_direct_ = false;
    bb_native_stream::LegacyPm4SlicePayload slice{};
    slice.pm4_va = reinterpret_cast<uintptr_t>(packet);
    slice.num_dwords = total_dwords;
    slice.flags = 0u;
    if (writer_.Emit(bb_native_stream::CmdOp::LegacyPm4Slice, 0u, slice)) {
        ++sub_packets_;
    } else {
        sub_overflow_ = true;
    }
    ++sub_draws_pm4_;
    ++total_.draws_pm4;
    ++window_.draws_pm4;
}

bool NativeStreamShadow::EndSubmission(bool log_periodic) noexcept {
    if (!active_) {
        return true;
    }
    active_ = false;
    has_pending_direct_ = false;
    ++total_.submissions;
    ++window_.submissions;

    const u32 dwords = writer_.SizeDwords();
    total_.dwords_emitted += dwords;
    window_.dwords_emitted += dwords;
    total_.packets_emitted += sub_packets_;
    window_.packets_emitted += sub_packets_;

    bool ok = !sub_overflow_;
    if (ok) {
        bb_native_stream::NativeStreamReader reader{writer_.Data(), dwords};
        bb_native_stream::CmdHeader hdr{};
        const u32* pay = nullptr;
        u32 parsed_pkts = 0u;
        u32 parsed_3d = 0u;
        u32 parsed_pm4 = 0u;
        while (reader.Next(hdr, pay)) {
            ++parsed_pkts;
            const auto op = static_cast<bb_native_stream::CmdOp>(hdr.op);
            if (op == bb_native_stream::CmdOp::DrawIndexed ||
                op == bb_native_stream::CmdOp::DrawAuto ||
                op == bb_native_stream::CmdOp::DrawIndirect) {
                ++parsed_3d;
            } else if (op == bb_native_stream::CmdOp::LegacyPm4Slice) {
                ++parsed_pm4;
            } else if (op == bb_native_stream::CmdOp::BindStageResources) {
                if (hdr.size_dwords < 2u) {
                    ok = false;
                    break;
                }
                bb_native_stream::BindStageResourcesHeader rh{};
                std::memcpy(&rh, pay, sizeof(rh));
                const u32 exp_dw =
                    2u + u32(rh.num_srvs) + u32(rh.num_samplers) + u32(rh.num_cbs) * 4u;
                if (hdr.size_dwords != exp_dw) {
                    ok = false;
                    break;
                }
            }
        }
        if (!reader.Empty() || parsed_pkts != sub_packets_ || parsed_3d != sub_draws_3d_ ||
            parsed_pm4 != sub_draws_pm4_) {
            ok = false;
        }
    }
    if (!ok) {
        ++total_.stream_errors;
        ++window_.stream_errors;
    }
    if (log_periodic) {
        MaybeLogAudit();
    }
    return ok;
}

void NativeStreamShadow::MaybeLogAudit() noexcept {
    const u64 now_ns = NowSteadyNs();
    if (window_start_ns_ == 0u) {
        window_start_ns_ = now_ns;
        return;
    }
    const u64 elapsed_ns = now_ns - window_start_ns_;
    if (elapsed_ns < 5'000'000'000ULL) {
        return;
    }
    const double sec = static_cast<double>(elapsed_ns) * 1e-9;
    const auto pct = [](u64 num, u64 den) -> double {
        return den > 0u ? (100.0 * static_cast<double>(num) / static_cast<double>(den)) : 0.0;
    };
    const double kb_per_sub =
        window_.submissions > 0u
            ? (static_cast<double>(window_.dwords_emitted * 4u) / 1024.0 /
               static_cast<double>(window_.submissions))
            : 0.0;
    const u64 bufs_other = window_.bufs_total - window_.bufs_cue - window_.bufs_native;
    std::printf(
        "[NativeStream-Audit] pkts=%.0f/s (%.1f KiB/sub) | draws: 3D=%.0f/s UI_PM4=%.0f/s | "
        "cov(rs=%.1f%% bs=%.1f%% ds=%.1f%% rt=%.1f%% pso=%.1f%%) | "
        "parityErr(rs=%llu bs=%llu ds=%llu stream=%llu)\n",
        static_cast<double>(window_.packets_emitted) / sec, kb_per_sub,
        static_cast<double>(window_.draws_3d) / sec, static_cast<double>(window_.draws_pm4) / sec,
        pct(window_.rs_covered, window_.draws_3d), pct(window_.bs_covered, window_.draws_3d),
        pct(window_.ds_covered, window_.draws_3d), pct(window_.rt_covered, window_.draws_3d),
        pct(window_.pso_id_covered, window_.draws_3d),
        static_cast<unsigned long long>(total_.rs_mismatches),
        static_cast<unsigned long long>(total_.bs_mismatches),
        static_cast<unsigned long long>(total_.ds_mismatches),
        static_cast<unsigned long long>(total_.stream_errors));
    std::printf(
        "[NativeStream-ResAudit] srvUid=%.1f%% (%llu/%llu) | smpUid=%.1f%% (%llu/%llu) | "
        "buf(cue=%.1f%% native=%.1f%% other=%.1f%% tot=%.0f/s) | dyn=%.0f/s res=%.0f/s\n",
        pct(window_.srvs_uid_hit, window_.srvs_total),
        static_cast<unsigned long long>(window_.srvs_uid_hit),
        static_cast<unsigned long long>(window_.srvs_total),
        pct(window_.samplers_uid_hit, window_.samplers_total),
        static_cast<unsigned long long>(window_.samplers_uid_hit),
        static_cast<unsigned long long>(window_.samplers_total),
        pct(window_.bufs_cue, window_.bufs_total), pct(window_.bufs_native, window_.bufs_total),
        pct(bufs_other, window_.bufs_total), static_cast<double>(window_.bufs_total) / sec,
        static_cast<double>(window_.dyn_packets) / sec,
        static_cast<double>(window_.res_packets) / sec);
    window_ = {};
    window_start_ns_ = now_ns;
}

} // namespace Vulkan
