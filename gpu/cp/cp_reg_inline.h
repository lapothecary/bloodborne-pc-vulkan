// SPDX-License-Identifier: GPL-2.0-or-later
// Step P3: Single-source-of-truth inline register/state helpers shared between
// `command_processor.cpp` and `cp_graphics.cpp` hot loop (eliminates ~560K/s cross-TU calls
// across `SetContextRegsPacked`, `SetIndexState`, `SetVgtControl`, and `SetDirectStateIds`).
#pragma once

#include <cstring>
#include <iterator>

#include "common/types.h"
#include "cp/command_processor.h"
#include "cp/cp_packets.h"
#include "gcn/regs.h"
#include "render/direct_states.h"

namespace BbCp {

[[gnu::always_inline]] inline constexpr bool ContextSpanAffectsPsoSig(u32 ctx_off,
                                                                      u32 count) noexcept {
    if (count == 0u) {
        return false;
    }
    const u32 end = ctx_off + count - 1u;
    return (ctx_off <= 0x003u && end >= 0x003u) || (ctx_off <= 0x017u && end >= 0x010u) ||
           (ctx_off <= 0x08Fu && end >= 0x08Eu) || (ctx_off <= 0x1C5u && end >= 0x191u) ||
           (ctx_off <= 0x1E7u && end >= 0x1E0u) || (ctx_off <= 0x207u && end >= 0x202u) ||
           (ctx_off <= 0x2A9u && end >= 0x286u) || (ctx_off <= 0x2DEu && end >= 0x2ABu) ||
           (ctx_off <= 0x2EAu && end >= 0x2E4u) || (ctx_off <= 0x2F8u && end >= 0x2F8u) ||
           (ctx_off <= 0x38Fu && end >= 0x318u);
}

[[gnu::always_inline]] inline constexpr bool ContextSpanCoveredByDirectState(u32 ctx_off,
                                                                             u32 count) noexcept {
    if (count == 0u) {
        return true;
    }
    const u32 end = ctx_off + count - 1u;
    return (ctx_off == 0x08Eu && end == 0x08Eu) ||
           (ctx_off >= 0x1E0u && end <= 0x1E7u) ||
           (ctx_off >= 0x204u && end <= 0x206u) ||
           (ctx_off == 0x292u && end == 0x292u) ||
           (ctx_off >= 0x2DCu && end <= 0x2DEu);
}

[[gnu::always_inline]] inline void
InvalidateDirectStateOnContextSpan(CommandProcessor::DirectStateTrack& ds, u32 ctx_off,
                                   u32 count) noexcept {
    if (count == 0u) {
        return;
    }
    const u32 end = ctx_off + count - 1u;
    if ((ctx_off <= 0x205u && end >= 0x204u) || (ctx_off <= 0x292u && end >= 0x292u) ||
        (ctx_off <= 0x2E3u && end >= 0x2DFu)) {
        ds.rs_id = 0u;
    }
    if ((ctx_off <= 0x08Eu && end >= 0x08Eu) || (ctx_off <= 0x1E7u && end >= 0x1E0u) ||
        (ctx_off <= 0x2DCu && end >= 0x2DCu)) {
        ds.bs_id = 0u;
    }
    if ((ctx_off <= 0x200u && end >= 0x200u) || (ctx_off <= 0x10Du && end >= 0x10Bu)) {
        ds.ds_id = 0u;
    }
    if (ctx_off <= 0x1B0u && end >= 0x191u) {
        ds.ps_usage_id = 0u;
    }
    if ((ctx_off <= 0x38Fu && end >= 0x318u) || (ctx_off <= 0x017u && end >= 0x010u)) {
        ds.rt_fmt_dirty = 1u;
    }
}

[[gnu::always_inline]] inline void MarkRegInline(Gcn::RegDirty* dirty, u32 index,
                                                 u32 count) noexcept {
    if (dirty) {
        dirty->Mark(index, count);
    }
}

[[gnu::always_inline]] inline void WriteRegInline(Gcn::Regs& regs, Gcn::RegDirty* dirty,
                                                  bool track_all, u32 index, u32 value) noexcept {
    if (regs.reg_array[index] != value) {
        regs.reg_array[index] = value;
        MarkRegInline(dirty, index, 1u);
    } else if (track_all) [[unlikely]] {
        MarkRegInline(dirty, index, 1u);
    }
}

[[gnu::always_inline]] inline bool ApplyContextRegsPackedInline(
    Gcn::Regs& regs, Gcn::RegDirty* dirty, bool track_all, u32 sub_op_word, const u32* pay,
    u32 pay_words, bool& pso_chg, bool& dyn_chg, bool& pso_non_shader_chg, bool& misc_state_chg,
    CommandProcessor::DirectStateTrack* direct_state) noexcept {
    if (pay_words < 2u) {
        return false;
    }
    const bool from_direct_state = (sub_op_word & DirectVk::kContextRegFromDirectState) != 0u;
    const u32 num_spans = (sub_op_word >> 16) & 0x7FFFu;
    u32 idx = 0u;
    bool any_changed = false;
    bool any_dyn_changed = false;
    bool span_pso_changed = false;
    bool span_non_shader_changed = false;
    bool span_misc_changed = false;
    for (u32 s = 0u; s < num_spans; ++s) {
        if (idx >= pay_words) {
            return false;
        }
        const u32 span_hdr = pay[idx++];
        const u32 ctx_off = span_hdr & 0xFFFFu;
        const u32 count = span_hdr >> 16;
        if (count == 0u || idx + count > pay_words ||
            Bank::Context + ctx_off + count > std::size(regs.reg_array)) {
            return false;
        }
        const u32 dst_reg = Bank::Context + ctx_off;
        u32* const dst = &regs.reg_array[dst_reg];
        const u32* const src = pay + idx;
        u32 diff = 0u;
        for (u32 i = 0u; i < count; ++i) {
            const u32 v = src[i];
            diff |= dst[i] ^ v;
            dst[i] = v;
        }
        const bool is_ps_usage_span = (num_spans == 1u && ctx_off == 0x0191u && count <= 32u);
        if (diff != 0u) {
            any_changed = true;
            if (!is_ps_usage_span) {
                any_dyn_changed = true;
            }
            if (!from_direct_state && direct_state) {
                if (is_ps_usage_span) {
                    direct_state->ps_usage_id = 0u;
                } else {
                    InvalidateDirectStateOnContextSpan(*direct_state, ctx_off, count);
                }
            }
            if (ContextSpanAffectsPsoSig(ctx_off, count)) {
                span_pso_changed = true;
                if (!is_ps_usage_span) {
                    span_non_shader_changed = true;
                    if (!(from_direct_state && ContextSpanCoveredByDirectState(ctx_off, count))) {
                        span_misc_changed = true;
                    }
                }
            }
        }
        if ((diff != 0u && !is_ps_usage_span) || track_all) {
            MarkRegInline(dirty, dst_reg, count);
        }
        idx += count;
    }
    if (idx != pay_words) {
        return false;
    }
    if (any_changed) {
        if (any_dyn_changed) {
            dyn_chg = true;
        }
        if (span_pso_changed) {
            pso_chg = true;
            if (span_non_shader_changed) {
                pso_non_shader_chg = true;
                if (span_misc_changed) {
                    misc_state_chg = true;
                }
            }
        }
    }
    return true;
}

[[gnu::always_inline]] inline bool ApplyIndexStateInline(
    Gcn::Regs& regs, Gcn::RegDirty* dirty, bool track_all, const u32* pay, u32 pay_words,
    bool& pso_chg, bool& dyn_chg, bool& pso_non_shader_chg) noexcept {
    if (pay_words != 4u) {
        return false;
    }
    DirectVk::SetIndexStateArgs a{};
    std::memcpy(&a, pay, sizeof(a));
    if (a.index_and_flags & 0x100u) {
        const u32 new_prim = a.prim_type & 0x3Fu;
        if (regs.reg_array[Reg::VgtPrimitiveType] != new_prim) {
            pso_chg = true;
            pso_non_shader_chg = true;
            dyn_chg = true;
        }
        WriteRegInline(regs, dirty, track_all, Reg::VgtPrimitiveType, new_prim);
    }
    if (a.index_and_flags & 0x200u) {
        WriteRegInline(regs, dirty, track_all, Reg::VgtDmaBase, u32(a.ib_va) & ~1u);
        WriteRegInline(regs, dirty, track_all, Reg::VgtDmaBaseHi,
                       (regs.reg_array[Reg::VgtDmaBaseHi] & ~0xFFu) |
                           (u32(a.ib_va >> 32) & 0xFFu));
    }
    if (a.index_and_flags & 0x400u) {
        WriteRegInline(regs, dirty, track_all, Reg::VgtDmaIndexType, a.index_and_flags & 0xFFu);
    }
    return true;
}

[[gnu::always_inline]] inline void ApplyDirectStateIdsFastInline(
    Gcn::Regs& regs, Gcn::RegDirty* dirty, bool track_all, u32 kind, u16 id, bool& pso_chg,
    bool& dyn_chg, bool& pso_non_shader_chg,
    CommandProcessor::DirectStateTrack* direct_state) noexcept {
    auto write_ctx = [&](u32 ctx_off, u32 val, bool affects_pso,
                         bool defer_stage_b = false) noexcept {
        const u32 dst_reg = Bank::Context + ctx_off;
        if (regs.reg_array[dst_reg] != val) {
            regs.reg_array[dst_reg] = val;
            if (!defer_stage_b || track_all) {
                MarkRegInline(dirty, dst_reg, 1u);
            }
            dyn_chg = true;
            if (affects_pso) {
                pso_chg = true;
                pso_non_shader_chg = true;
            }
        } else if (track_all) [[unlikely]] {
            MarkRegInline(dirty, dst_reg, 1u);
        }
    };
    if (kind == 0u) {
        if (direct_state) {
            if (direct_state->rs_id == id && id != 0u && !track_all) [[unlikely]] {
                return;
            }
            direct_state->rs_id = id;
        }
        Vulkan::DirectStates::RasterStateDesc rs{};
        if (id != 0u && Vulkan::DirectStates::ReadRasterState(id, rs)) {
            write_ctx(0x204u, rs.primitive_setup & 0x0F7FE03Fu, true);
            write_ctx(0x205u, rs.clip_control & 0x00393FFFu, true);
            const bool defer_rs = !track_all;
            write_ctx(0x292u, (rs.msaa_enable & 1u) | ((rs.scissor_enable << 1) & 2u), true,
                      defer_rs);
            write_ctx(0x2DFu, rs.depth_bias_clamp_bits, false, defer_rs);
            float db = 0.0f;
            std::memcpy(&db, &rs.depth_bias_bits, sizeof(float));
            const float scaled_db = db * 16.0f;
            u32 sb_bits = 0u;
            std::memcpy(&sb_bits, &scaled_db, sizeof(u32));
            write_ctx(0x2E0u, sb_bits, false, defer_rs);
            write_ctx(0x2E1u, rs.slope_scale_depth_bias_bits, false, defer_rs);
            write_ctx(0x2E2u, sb_bits, false, defer_rs);
            write_ctx(0x2E3u, rs.slope_scale_depth_bias_bits, false, defer_rs);
            if (defer_rs && dirty) {
                dirty->rs_chunks_deferred = true;
            }
        }
    } else if (kind == 1u) {
        if (direct_state) {
            if (direct_state->bs_id == id && id != 0u && !track_all) [[unlikely]] {
                return;
            }
            direct_state->bs_id = id;
        }
        Vulkan::DirectStates::BlendStateDesc bs{};
        if (id != 0u && Vulkan::DirectStates::ReadBlendState(id, bs)) {
            write_ctx(0x2DCu, 0xAA00u | (bs.alpha_to_coverage_enable & 1u), true, !track_all);
            write_ctx(0x08Eu, bs.render_target_channel_mask, true);
            for (u32 slot = 0u; slot < 8u; ++slot) {
                write_ctx(0x1E0u + slot, bs.blend_control[slot] & 0xFFFF1FFFu, true);
            }
        }
    } else if (kind == 2u) {
        if (direct_state) {
            if (direct_state->ds_id == id && id != 0u && !track_all) [[unlikely]] {
                return;
            }
            direct_state->ds_id = id;
        }
        Vulkan::DirectStates::DepthStencilStateDesc ds{};
        if (id != 0u && Vulkan::DirectStates::ReadDepthStencilState(id, ds)) {
            write_ctx(0x200u, ds.depth_control & 0xC07007FFu, false);
            const bool defer_ds = !track_all;
            write_ctx(0x10Bu, ds.stencil_control & 0x00FFFFFFu, false, defer_ds);
            if (defer_ds && dirty) {
                dirty->ds_chunks_deferred = true;
            }
            const u32 st_bits = ds.stencil_op_control & 0x00FFFF00u;
            write_ctx(0x10Cu, (regs.reg_array[Bank::Context + 0x10Cu] & 0xFF0000FFu) | st_bits,
                      false);
            write_ctx(0x10Du, (regs.reg_array[Bank::Context + 0x10Du] & 0xFF0000FFu) | st_bits,
                      false);
        }
    }
}

} // namespace BbCp
