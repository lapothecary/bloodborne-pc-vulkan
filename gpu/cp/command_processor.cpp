// SPDX-License-Identifier: GPL-2.0-or-later
#include <algorithm>
#include <bit>
#include <chrono>
#include <filesystem>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

#include <immintrin.h>
#include <pthread.h>
#include <sys/resource.h>
#ifndef _WIN32
#include <sys/syscall.h>
#endif
#include <csignal>
#include <dlfcn.h>
#include <sys/uio.h>
#include <unistd.h>

#include <fmt/format.h>

#include "bbport_ce_stats.h"
#include "bbport_free_check.h"
#include "bbport_threads.h"
#include "bbport_timeline.h"
#include "bbport_toggles.h"
#include "bbport_write_log.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "bbport_sections.h"
#include "cp/command_processor.h"
#include "cp/cp_diagnostics.h"
#include "cp/cp_packets.h"
#include "cp/cp_tasks.h"
#include "cp/pm4_selftest.h"
#include "platform/interrupts.h"
#include "platform/videoout/videoout.h"
#include "render/vk_draw_pipe.h"
#include "render/vk_draw_prep.h"
#include "render/shader_hle.h"
#include "buffers/buffer_cache.h"
#include "buffers/direct_buffers.h"
#include "render/rasterizer.h"
#include "render/vk_mt_direct_recorder.h"
#include "textures/direct_textures.h"

namespace BbCp {

std::array<u8, 48 * 1024> CommandProcessor::constant_ram;


namespace {

using Vulkan::DrawPipe;

constexpr u32 ComputeShOffset = 0x200; ///< COMPUTE_* in the SH bank
constexpr u32 ComputeWords = sizeof(ComputeProgram) / 4;

[[gnu::always_inline]] inline bool InComputeRange(u32 sh_offset) noexcept {
    return sh_offset >= ComputeShOffset && sh_offset <= ComputeShOffset + ComputeWords;
}

[[gnu::always_inline]] inline u32 BodyWords(u32 header) noexcept {
    return ((header >> 16) & 0x3FFF) + 1;
}

[[gnu::always_inline]] inline void BumpPsoAndDyn(u64* pso_sig_epoch, u64* dyn_reg_epoch) noexcept {
    if (pso_sig_epoch) {
        ++(*pso_sig_epoch);
    }
    if (dyn_reg_epoch) {
        ++(*dyn_reg_epoch);
    }
}

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

[[gnu::always_inline]] inline void MarkReg(Gcn::RegDirty* dirty, u32 index, u32 count) noexcept {
    if (dirty) {
        dirty->Mark(index, count);
    }
}

[[gnu::always_inline]] inline void MarkShReg(Gcn::RegDirty* dirty, bool track_all, u32 index,
                                             u32 count) noexcept {
    if (track_all) [[unlikely]] {
        dirty->Mark(index, count);
    }
}

[[gnu::always_inline]] inline void WriteReg(Gcn::Regs& regs, Gcn::RegDirty* dirty, bool track_all,
                                            u32 index, u32 value) noexcept {
    if (regs.reg_array[index] != value) {
        regs.reg_array[index] = value;
        MarkReg(dirty, index, 1u);
    } else if (track_all) [[unlikely]] {
        MarkReg(dirty, index, 1u);
    }
}

[[gnu::always_inline]] inline bool SetBankRegs(Gcn::Regs& regs, Gcn::RegDirty* dirty,
                                               bool track_all, const u32* body, u32 body_words,
                                               u32 bank) noexcept {
    const u32 index = bank + (body[0] & 0xFFFFu);
    const u32 count = body_words - 1u;
    const u32* src = body + 1u;
    bool any_changed = false;
    for (u32 i = 0; i < count; ++i) {
        if (regs.reg_array[index + i] != src[i]) {
            regs.reg_array[index + i] = src[i];
            any_changed = true;
        }
    }
    if (bank == Bank::Sh) {
        MarkShReg(dirty, track_all, index, count);
    } else if (any_changed || track_all) {
        MarkReg(dirty, index, count);
    }
    return any_changed;
}

[[gnu::always_inline]] inline bool ApplyDirectStateSubOp(Gcn::Regs& regs, Gcn::RegDirty* dirty,
                                                         bool track_all, DirectVk::Op sub_op,
                                                         u32 sub_op_word, const u32* pay,
                                                         u32 pay_words, bool& pso_chg,
                                                         bool& dyn_chg) noexcept {
    switch (sub_op) {
    case DirectVk::Op::SetIndexState: {
        if (pay_words != 4u) {
            return false;
        }
        DirectVk::SetIndexStateArgs a{};
        std::memcpy(&a, pay, sizeof(a));
        if (a.index_and_flags & 0x100u) {
            const u32 new_prim = a.prim_type & 0x3Fu;
            if (regs.reg_array[Reg::VgtPrimitiveType] != new_prim) {
                pso_chg = true;
                dyn_chg = true;
            }
            WriteReg(regs, dirty, track_all, Reg::VgtPrimitiveType, new_prim);
        }
        if (a.index_and_flags & 0x200u) {
            WriteReg(regs, dirty, track_all, Reg::VgtDmaBase, u32(a.ib_va) & ~1u);
            WriteReg(regs, dirty, track_all, Reg::VgtDmaBaseHi,
                     (regs.reg_array[Reg::VgtDmaBaseHi] & ~0xFFu) | (u32(a.ib_va >> 32) & 0xFFu));
        }
        if (a.index_and_flags & 0x400u) {
            WriteReg(regs, dirty, track_all, Reg::VgtDmaIndexType, a.index_and_flags & 0xFFu);
        }
        return true;
    }
    case DirectVk::Op::SetStageUserData: {
        if (pay_words < 1u) {
            return false;
        }
        const u32 sh_offset = pay[0] & 0xFFFFu;
        if (InComputeRange(sh_offset)) {
            return false;
        }
        u32 mask = pay[0] >> 16;
        const u32 count = pay_words - 1u;
        if (static_cast<u32>(std::popcount(mask)) != count) {
            return false;
        }
        if ((mask & (mask + 1u)) == 0u) [[likely]] {
            std::memcpy(&regs.reg_array[Bank::Sh + sh_offset], pay + 1u, count * sizeof(u32));
            MarkShReg(dirty, track_all, Bank::Sh + sh_offset, count);
        } else {
            u32 src = 1u;
            while (mask != 0u) {
                const u32 bit = static_cast<u32>(std::countr_zero(mask));
                mask &= (mask - 1u);
                regs.reg_array[Bank::Sh + sh_offset + bit] = pay[src++];
                MarkShReg(dirty, track_all, Bank::Sh + sh_offset + bit, 1u);
            }
        }
        if ((sh_offset & 0x3Fu) < 0x0Cu) [[unlikely]] {
            pso_chg = true;
        }
        return true;
    }
    case DirectVk::Op::SetContextRegsPacked: {
        if (pay_words < 2u) {
            return false;
        }
        const u32 num_spans = sub_op_word >> 16;
        u32 idx = 0u;
        bool any_changed = false;
        bool span_pso_changed = false;
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
            bool span_changed = false;
            for (u32 i = 0; i < count; ++i) {
                if (regs.reg_array[dst_reg + i] != pay[idx + i]) {
                    regs.reg_array[dst_reg + i] = pay[idx + i];
                    span_changed = true;
                }
            }
            if (span_changed) {
                any_changed = true;
                if (!span_pso_changed && ContextSpanAffectsPsoSig(ctx_off, count)) {
                    span_pso_changed = true;
                }
            }
            if (span_changed || track_all) {
                MarkReg(dirty, dst_reg, count);
            }
            idx += count;
        }
        if (idx != pay_words) {
            return false;
        }
        if (any_changed) {
            dyn_chg = true;
            if (span_pso_changed) {
                pso_chg = true;
            }
        }
        return true;
    }
    case DirectVk::Op::BindVsShader: {
        if (pay_words != 7u) {
            return false;
        }
        const bool only_sh = ((sub_op_word >> 18) & 1u) != 0u;
        bool changed = (regs.reg_array[Bank::Sh + 0x48u] != pay[0]) ||
                       (regs.reg_array[Bank::Sh + 0x49u] != pay[1]) ||
                       (regs.reg_array[Bank::Sh + 0x4Au] != pay[2]) ||
                       (regs.reg_array[Bank::Sh + 0x4Bu] != pay[3]);
        std::memcpy(&regs.reg_array[Bank::Sh + 0x48u], pay, 4u * sizeof(u32));
        MarkShReg(dirty, track_all, Bank::Sh + 0x48u, 4u);
        if (!only_sh) {
            const bool c54 = (regs.reg_array[Bank::Context + 0x1B1u] != pay[4]);
            const bool c56 = (regs.reg_array[Bank::Context + 0x1C3u] != pay[5]);
            const bool c64 = (regs.reg_array[Bank::Context + 0x207u] != pay[6]);
            changed |= c54 || c56 || c64;
            regs.reg_array[Bank::Context + 0x1B1u] = pay[4];
            regs.reg_array[Bank::Context + 0x1C3u] = pay[5];
            regs.reg_array[Bank::Context + 0x207u] = pay[6];
            if (dirty) {
                if (!track_all) [[likely]] {
                    dirty->stage_b_mask[0] |=
                        (c54 ? (1ULL << 54) : 0ULL) | (c56 ? (1ULL << 56) : 0ULL);
                    if (c64) {
                        dirty->stage_b_mask[1] |= (1ULL << 0);
                        dyn_chg = true;
                    }
                } else {
                    MarkReg(dirty, Bank::Context + 0x1B1u, 1u);
                    MarkReg(dirty, Bank::Context + 0x1C3u, 1u);
                    MarkReg(dirty, Bank::Context + 0x207u, 1u);
                }
            } else if (c64) {
                dyn_chg = true;
            }
        }
        if (changed) {
            pso_chg = true;
        }
        return true;
    }
    case DirectVk::Op::BindPsShader: {
        const bool valid = ((sub_op_word >> 16) & 1u) != 0u;
        const bool sdk350 = ((sub_op_word >> 17) & 1u) != 0u;
        const bool only_sh = ((sub_op_word >> 18) & 1u) != 0u;
        if (valid && pay_words == 12u) {
            bool changed = (regs.reg_array[Bank::Sh + 0x08u] != pay[0]) ||
                           (regs.reg_array[Bank::Sh + 0x09u] != pay[1]) ||
                           (regs.reg_array[Bank::Sh + 0x0Au] != pay[2]) ||
                           (regs.reg_array[Bank::Sh + 0x0Bu] != pay[3]);
            std::memcpy(&regs.reg_array[Bank::Sh + 0x08u], pay, 4u * sizeof(u32));
            MarkShReg(dirty, track_all, Bank::Sh + 0x08u, 4u);
            if (!only_sh) {
                const bool c56 = (regs.reg_array[Bank::Context + 0x1C4u] != pay[4]) ||
                                 (regs.reg_array[Bank::Context + 0x1C5u] != pay[5]);
                const bool c54 = (regs.reg_array[Bank::Context + 0x1B3u] != pay[6]) ||
                                 (regs.reg_array[Bank::Context + 0x1B4u] != pay[7]) ||
                                 (regs.reg_array[Bank::Context + 0x1B6u] != pay[8]);
                const bool c55 = (regs.reg_array[Bank::Context + 0x1B8u] != pay[9]);
                const bool c64 = (regs.reg_array[Bank::Context + 0x203u] != pay[10]);
                const bool c17 = (regs.reg_array[Bank::Context + 0x08Fu] != pay[11]);
                changed |= c56 || c54 || c55 || c64 || c17;
                regs.reg_array[Bank::Context + 0x1C4u] = pay[4];
                regs.reg_array[Bank::Context + 0x1C5u] = pay[5];
                regs.reg_array[Bank::Context + 0x1B3u] = pay[6];
                regs.reg_array[Bank::Context + 0x1B4u] = pay[7];
                regs.reg_array[Bank::Context + 0x1B6u] = pay[8];
                regs.reg_array[Bank::Context + 0x1B8u] = pay[9];
                regs.reg_array[Bank::Context + 0x203u] = pay[10];
                regs.reg_array[Bank::Context + 0x08Fu] = pay[11];
                if (dirty) {
                    if (!track_all) [[likely]] {
                        dirty->stage_b_mask[0] |= (c17 ? (1ULL << 17) : 0ULL) |
                                                  (c54 ? (1ULL << 54) : 0ULL) |
                                                  (c55 ? (1ULL << 55) : 0ULL) |
                                                  (c56 ? (1ULL << 56) : 0ULL);
                        if (c64) {
                            dirty->stage_b_mask[1] |= (1ULL << 0);
                        }
                    } else {
                        MarkReg(dirty, Bank::Context + 0x1C4u, 2u);
                        MarkReg(dirty, Bank::Context + 0x1B3u, 2u);
                        MarkReg(dirty, Bank::Context + 0x1B6u, 1u);
                        MarkReg(dirty, Bank::Context + 0x1B8u, 1u);
                        MarkReg(dirty, Bank::Context + 0x203u, 1u);
                        MarkReg(dirty, Bank::Context + 0x08Fu, 1u);
                    }
                }
            }
            if (changed) {
                pso_chg = true;
            }
            return true;
        }
        if (!valid && pay_words == 0u) {
            regs.reg_array[Bank::Sh + 0x08u] = 0u;
            regs.reg_array[Bank::Sh + 0x09u] = 0u;
            MarkShReg(dirty, track_all, Bank::Sh + 0x08u, 2u);
            if (!only_sh) {
                WriteReg(regs, dirty, track_all, Bank::Context + 0x203u, 0u);
            }
            if (sdk350) {
                WriteReg(regs, dirty, track_all, Bank::Context + 0x08Fu, 0xFu);
            }
            pso_chg = true;
            return true;
        }
        return false;
    }
    case DirectVk::Op::BindEsShader:
        if (pay_words != 4u) {
            return false;
        }
        std::memcpy(&regs.reg_array[Bank::Sh + 0xC8u], pay, 4u * sizeof(u32));
        MarkShReg(dirty, track_all, Bank::Sh + 0xC8u, 4u);
        pso_chg = true;
        dyn_chg = true;
        return true;
    case DirectVk::Op::BindGsShader: {
        if (pay_words != 7u) {
            return false;
        }
        const bool only_sh = ((sub_op_word >> 18) & 1u) != 0u;
        std::memcpy(&regs.reg_array[Bank::Sh + 0x88u], pay, 4u * sizeof(u32));
        MarkShReg(dirty, track_all, Bank::Sh + 0x88u, 4u);
        if (!only_sh) {
            WriteReg(regs, dirty, track_all, Bank::Context + 0x2E5u, pay[4]);
            WriteReg(regs, dirty, track_all, Bank::Context + 0x29Bu, pay[5]);
            WriteReg(regs, dirty, track_all, Bank::Context + 0x2E4u, pay[6]);
        }
        pso_chg = true;
        dyn_chg = true;
        return true;
    }
    case DirectVk::Op::BindHsShader: {
        if (pay_words != 8u) {
            return false;
        }
        const bool only_sh = ((sub_op_word >> 18) & 1u) != 0u;
        std::memcpy(&regs.reg_array[Bank::Sh + 0x108u], pay, 4u * sizeof(u32));
        MarkShReg(dirty, track_all, Bank::Sh + 0x108u, 4u);
        if (!only_sh) {
            WriteReg(regs, dirty, track_all, Bank::Context + 0x286u, pay[5]);
            WriteReg(regs, dirty, track_all, Bank::Context + 0x287u, pay[6]);
            WriteReg(regs, dirty, track_all, Bank::Context + 0x2DBu, pay[4]);
            WriteReg(regs, dirty, track_all, Bank::Context + 0x2D6u, pay[7]);
        }
        pso_chg = true;
        dyn_chg = true;
        return true;
    }
    case DirectVk::Op::BindLsShader:
        if (pay_words != 4u) {
            return false;
        }
        std::memcpy(&regs.reg_array[Bank::Sh + 0x148u], pay, 4u * sizeof(u32));
        MarkShReg(dirty, track_all, Bank::Sh + 0x148u, 4u);
        pso_chg = true;
        dyn_chg = true;
        return true;
    case DirectVk::Op::SetVgtControl: {
        if (pay_words != 0u) {
            return false;
        }
        const u32 prim_group_sz_minus_1 = (sub_op_word >> 16) & 0xFFu;
        const u32 partial_vs_wave_mode = (sub_op_word >> 24) & 1u;
        WriteReg(regs, dirty, track_all, Bank::Context + 0x2AAu,
                 (partial_vs_wave_mode << 16) | prim_group_sz_minus_1);
        return true;
    }
    default:
        return false;
    }
}

[[gnu::always_inline]] inline void ApplyDirectDrawIndexedWords(Gcn::Regs& regs,
                                                               Gcn::RegDirty* dirty, bool track_all,
                                                               const u32* dw) noexcept {
    DirectVk::DrawIndexedArgs a{};
    std::memcpy(&a, dw, sizeof(a));
    const u32 base_vtx = static_cast<u32>(a.base_vertex);
    const bool chg_idx_off = (regs.reg_array[Reg::VgtIndxOffset] != base_vtx);
    const bool chg_inst = (regs.reg_array[Reg::VgtNumInstances] != a.instance_count);
    const bool chg_max = (regs.reg_array[Reg::VgtDmaMaxSize] != a.index_count);
    const bool chg_num = (regs.reg_array[Reg::VgtNumIndices] != a.index_count);
    const bool chg_init = (regs.reg_array[Reg::VgtDrawInitiator] != 0u);
    regs.reg_array[Reg::VgtIndxOffset] = base_vtx;
    regs.reg_array[Reg::VgtNumInstances] = a.instance_count;
    regs.reg_array[Reg::VgtDmaMaxSize] = a.index_count;
    regs.reg_array[Reg::VgtNumIndices] = a.index_count;
    regs.reg_array[Reg::VgtDrawInitiator] = 0u;
    if (dirty) {
        if (!track_all) [[likely]] {
            dirty->stage_b_mask[0] |=
                (chg_idx_off ? (1ULL << 32) : 0ULL) | (chg_init ? (1ULL << 63) : 0ULL);
            dirty->stage_b_mask[1] |=
                (chg_max ? (1ULL << (83 - 64)) : 0ULL) |
                ((chg_inst || chg_num) ? (1ULL << (116 - 64)) : 0ULL);
        } else {
            MarkReg(dirty, Reg::VgtIndxOffset, 1u);
            MarkReg(dirty, Reg::VgtNumInstances, 1u);
            MarkReg(dirty, Reg::VgtDmaMaxSize, 1u);
            MarkReg(dirty, Reg::VgtNumIndices, 1u);
            MarkReg(dirty, Reg::VgtDrawInitiator, 1u);
        }
    }
}

[[gnu::always_inline]] inline void ApplyDirectDrawIndexAddrWords(
    Gcn::Regs& regs, Gcn::RegDirty* dirty, bool track_all, const u32* dw, bool& pso_chg,
    bool& dyn_chg) noexcept {
    DirectVk::DrawIndexAddrArgs a{};
    std::memcpy(&a, dw, sizeof(a));
    if (a.prim_and_size & 0x100u) {
        const u32 new_prim = a.prim_and_size & 0x3Fu;
        if (regs.reg_array[Reg::VgtPrimitiveType] != new_prim) {
            pso_chg = true;
            dyn_chg = true;
        }
        WriteReg(regs, dirty, track_all, Reg::VgtPrimitiveType, new_prim);
    }
    if (a.prim_and_size & 0x200u) {
        WriteReg(regs, dirty, track_all, Reg::VgtDmaIndexType, (a.prim_and_size >> 16) & 0xFFu);
    }
    WriteReg(regs, dirty, track_all, Reg::VgtDmaMaxSize, a.index_count);
    WriteReg(regs, dirty, track_all, Reg::VgtDmaBase, u32(a.ib_va) & ~1u);
    WriteReg(regs, dirty, track_all, Reg::VgtDmaBaseHi,
             (regs.reg_array[Reg::VgtDmaBaseHi] & ~0xFFu) | (u32(a.ib_va >> 32) & 0xFFu));
    WriteReg(regs, dirty, track_all, Reg::VgtNumIndices, a.index_count);
    WriteReg(regs, dirty, track_all, Reg::VgtDrawInitiator, 0u);
}

[[gnu::always_inline]] inline void ApplyDirectDrawAutoWords(Gcn::Regs& regs, Gcn::RegDirty* dirty,
                                                            bool track_all,
                                                            const u32* dw) noexcept {
    DirectVk::DrawAutoArgs a{};
    std::memcpy(&a, dw, sizeof(a));
    WriteReg(regs, dirty, track_all, Reg::VgtIndxOffset, static_cast<u32>(a.start_vertex));
    WriteReg(regs, dirty, track_all, Reg::VgtNumInstances, a.instance_count);
    WriteReg(regs, dirty, track_all, Reg::VgtNumIndices, a.vertex_count);
    WriteReg(regs, dirty, track_all, Reg::VgtDrawInitiator, 2u);
}

[[gnu::noinline]] bool ApplyInitDefaultHardwareState(Gcn::Regs& regs, Gcn::RegDirty* dirty,
                                                     bool track_all, u32 op_word,
                                                     u64* pso_sig_epoch,
                                                     u64* dyn_reg_epoch) noexcept {
    const u32 gen = (op_word >> 16) & 0xFFu;
    if (gen > 5u) {
        return false;
    }
    regs.SetDefaults();
    if (dirty) {
        dirty->Clear();
        dirty->reset = true;
    }
    BumpPsoAndDyn(pso_sig_epoch, dyn_reg_epoch);
    if (gen == 4u) {
        WriteReg(regs, dirty, track_all, Reg::VgtNumInstances, 1u);
        WriteReg(regs, dirty, track_all, Bank::Context + 0x102u, 0u);
        WriteReg(regs, dirty, track_all, Bank::Context + 0x202u, 0xCC0010u);
        return true;
    }
    WriteReg(regs, dirty, track_all, Bank::Context + 0x2F9u, 0x2Du);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x282u, 8u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x280u, 0x80008u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x281u, 0xFFFF0000u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x204u, 0u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x206u, 0x43Fu);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x083u, 0xFFFFu);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x317u, 0x10u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x2FAu, 0x3F800000u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x2FCu, 0x3F800000u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x2FBu, 0x3F800000u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x2FDu, 0x3F800000u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x202u, 0xCC0010u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x30Eu, 0xFFFFFFFFu);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x30Fu, 0xFFFFFFFFu);
    WriteReg(regs, dirty, track_all, Reg::VgtNumInstances, 1u);
    if (gen <= 1u) {
        regs.reg_array[Bank::Sh + 0x007u] = 0x1FFu;
        regs.reg_array[Bank::Sh + 0x046u] = 0x1FFu;
        regs.reg_array[Bank::Sh + 0x087u] = 0x1FFu;
        regs.reg_array[Bank::Sh + 0x0C7u] = 0x1FFu;
        regs.reg_array[Bank::Sh + 0x107u] = 0u;
        regs.reg_array[Bank::Sh + 0x147u] = 0x1FFu;
        MarkShReg(dirty, track_all, Bank::Sh + 0x007u, 1u);
        MarkShReg(dirty, track_all, Bank::Sh + 0x046u, 1u);
        MarkShReg(dirty, track_all, Bank::Sh + 0x087u, 1u);
        MarkShReg(dirty, track_all, Bank::Sh + 0x0C7u, 1u);
        MarkShReg(dirty, track_all, Bank::Sh + 0x107u, 1u);
        MarkShReg(dirty, track_all, Bank::Sh + 0x147u, 1u);
    } else if (gen <= 3u) {
        regs.reg_array[Bank::Sh + 0x007u] = 0x1701FFu;
        regs.reg_array[Bank::Sh + 0x046u] = 0x1701FDu;
        regs.reg_array[Bank::Sh + 0x087u] = 0x1701FFu;
        regs.reg_array[Bank::Sh + 0x0C7u] = 0x1701FDu;
        regs.reg_array[Bank::Sh + 0x107u] = 0x17u;
        regs.reg_array[Bank::Sh + 0x147u] = 0x1701FDu;
        regs.reg_array[Bank::Sh + 0x047u] = 0x1Cu;
        MarkShReg(dirty, track_all, Bank::Sh + 0x007u, 1u);
        MarkShReg(dirty, track_all, Bank::Sh + 0x046u, 2u);
        MarkShReg(dirty, track_all, Bank::Sh + 0x087u, 1u);
        MarkShReg(dirty, track_all, Bank::Sh + 0x0C7u, 1u);
        MarkShReg(dirty, track_all, Bank::Sh + 0x107u, 1u);
        MarkShReg(dirty, track_all, Bank::Sh + 0x147u, 1u);
    }
    WriteReg(regs, dirty, track_all, Bank::Context + 0x1B1u, 2u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x101u, 0u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x100u, 0xFFFFFFFFu);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x103u, 0u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x284u, 0u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x290u, 0u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x2AEu, 0u);
    if (gen == 3u || gen == 5u) {
        WriteReg(regs, dirty, track_all, Bank::Context + 0x102u, 0u);
    }
    WriteReg(regs, dirty, track_all, Bank::Context + 0x292u, 0u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x293u, (gen == 0u) ? 0x6000000u : 0x6020000u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x2F8u, 0u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x2DEu, 0x1E9u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x295u, 0x100u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x296u, 0x100u);
    WriteReg(regs, dirty, track_all, Bank::Context + 0x297u, 4u);
    if (gen <= 3u) {
        WriteReg(regs, dirty, track_all, Bank::Uconfig + 0x200u, 0xE0000000u);
    }
    if (gen == 3u || gen == 5u) {
        WriteReg(regs, dirty, track_all, Bank::Context + 0x2AAu, 0xFFu);
    }
    BumpPsoAndDyn(pso_sig_epoch, dyn_reg_epoch);
    return true;
}

[[gnu::noinline]] bool ApplyDrawBundlePacket(Regs& regs, RegDirty* dirty, bool track_all,
                                             u32 op_word, const u32* body, u32 body_words,
                                             u64* pso_sig_epoch, u64* dyn_reg_epoch) noexcept {
    if (body_words < 7u) {
        return false;
    }
    const u32 draw_kind = (op_word >> 16) & 0xFu;
    const u32 sub_count = op_word >> 20;
    const u32 state_end = body_words - 4u;
    u32 cursor = 2u;
    bool pso_chg = false;
    bool dyn_chg = false;
    for (u32 i = 0u; i < sub_count; ++i) {
        if (cursor >= state_end) {
            return false;
        }
        const u32 sub_hdr = body[cursor++];
        const auto sub_op = DirectVk::Op(sub_hdr & 0xFFu);
        const u32 pay_words = (sub_hdr >> 8) & 0xFFu;
        const u32 sub_op_word = sub_hdr & 0xFFFF00FFu;
        if (cursor + pay_words > state_end) {
            return false;
        }
        if (!ApplyDirectStateSubOp(regs, dirty, track_all, sub_op, sub_op_word, body + cursor,
                                   pay_words, pso_chg, dyn_chg)) {
            return false;
        }
        cursor += pay_words;
    }
    if (cursor != state_end) {
        return false;
    }
    const u32* draw_words = body + state_end;
    if (draw_kind == 0u) {
        ApplyDirectDrawIndexedWords(regs, dirty, track_all, draw_words);
    } else if (draw_kind == 1u) {
        ApplyDirectDrawAutoWords(regs, dirty, track_all, draw_words);
    } else if (draw_kind == 2u) {
        ApplyDirectDrawIndexAddrWords(regs, dirty, track_all, draw_words, pso_chg, dyn_chg);
    } else {
        return false;
    }
    if (pso_chg && pso_sig_epoch) {
        ++(*pso_sig_epoch);
    }
    if (dyn_chg && dyn_reg_epoch) {
        ++(*dyn_reg_epoch);
    }
    return true;
}

[[gnu::noinline]] bool ApplySetRenderTargetPacket(Regs& regs, RegDirty* dirty, bool track_all,
                                                  DirectVk::Op vk_op, u32 op_word, const u32* body,
                                                  u32 body_words, u64* pso_sig_epoch,
                                                  u64* dyn_reg_epoch) noexcept {
    if (vk_op == DirectVk::Op::SetRenderTarget) {
        const u32 slot = (op_word >> 16) & 0x7u;
        const bool valid = ((op_word >> 19) & 1u) != 0u;
        if (valid && body_words == 14u) {
            const u32 dst_reg = Bank::Context + 0x318u + slot * 0x0Fu;
            std::memcpy(&regs.reg_array[dst_reg], body + 2u, 11u * sizeof(u32));
            MarkReg(dirty, dst_reg, 11u);
            BumpPsoAndDyn(pso_sig_epoch, dyn_reg_epoch);
            return true;
        }
        if (!valid && body_words == 2u) {
            WriteReg(regs, dirty, track_all, Bank::Context + 0x31Cu + slot * 0x0Fu, 0u);
            BumpPsoAndDyn(pso_sig_epoch, dyn_reg_epoch);
            return true;
        }
        return false;
    }
    const bool valid = ((op_word >> 16) & 1u) != 0u;
    if (valid && body_words == 15u) {
        const u32* dt = body + 2u;
        std::memcpy(&regs.reg_array[Bank::Context + 0x010u], dt, 8u * sizeof(u32));
        MarkReg(dirty, Bank::Context + 0x010u, 8u);
        WriteReg(regs, dirty, track_all, Bank::Context + 0x00Fu, dt[11]);
        WriteReg(regs, dirty, track_all, Bank::Context + 0x002u, dt[8]);
        WriteReg(regs, dirty, track_all, Bank::Context + 0x005u, dt[9]);
        WriteReg(regs, dirty, track_all, Bank::Context + 0x2AFu, dt[10]);
        BumpPsoAndDyn(pso_sig_epoch, dyn_reg_epoch);
        return true;
    }
    if (!valid && body_words == 2u) {
        WriteReg(regs, dirty, track_all, Bank::Context + 0x010u, 0u);
        WriteReg(regs, dirty, track_all, Bank::Context + 0x011u, 0u);
        BumpPsoAndDyn(pso_sig_epoch, dyn_reg_epoch);
        return true;
    }
    return false;
}

} // namespace

// ---- Registers -------------------------------------------------------------------------------

u64 CommandProcessor::HashRegisterPacket(u64 checksum, const u32* words, u32 count) {
    for (u32 i = 0; i < count; ++i) {
        checksum = (checksum ^ words[i]) * 0x9E3779B97F4A7C15ull;
        checksum ^= checksum >> 29;
    }
    return checksum;
}

bool CommandProcessor::InitGnmDefaultRegs(Regs& regs, u32 gen, RegDirty* dirty, bool track_all,
                                          u64* pso_sig_epoch, u64* dyn_reg_epoch) noexcept {
    if (gen > 5u) {
        return false;
    }
    const u32 op_word = u32(DirectVk::Op::InitDefaultHardwareState) | (gen << 16u);
    return ApplyInitDefaultHardwareState(regs, dirty, track_all, op_word, pso_sig_epoch,
                                         dyn_reg_epoch);
}

bool CommandProcessor::ApplyDrawBundleRegs(Regs& regs, RegDirty* dirty, bool track_all,
                                           u32 op_word, const u32* body, u32 body_words,
                                           u64* pso_sig_epoch, u64* dyn_reg_epoch) noexcept {
    return ApplyDrawBundlePacket(regs, dirty, track_all, op_word, body, body_words, pso_sig_epoch,
                                 dyn_reg_epoch);
}

bool CommandProcessor::ApplyGraphicsRegisterPacket(Regs& regs, const u32* packet, u64& checksum,
                                                   RegDirty* dirty, bool compute_checksum,
                                                   u64* pso_sig_epoch, u64* dyn_reg_epoch) {
    const u32 header = packet[0];
    const u32 body_words = BodyWords(header);
    const u32* body = packet + 1;
    const bool track_all = dirty && dirty->track_all_blocks;
    bool handled_completely = false;

    switch (Header{header}.Opcode()) {
    case Op::ClearState:
        regs.SetDefaults();
        if (dirty) {
            dirty->Clear();
            dirty->reset = true;
        }
        BumpPsoAndDyn(pso_sig_epoch, dyn_reg_epoch);
        break;
    case Op::SetConfigReg:
        if (SetBankRegs(regs, dirty, track_all, body, body_words, Bank::Config)) {
            BumpPsoAndDyn(pso_sig_epoch, dyn_reg_epoch);
        }
        handled_completely = true;
        break;
    case Op::SetContextReg: {
        const u32 ctx_off = body[0] & 0xFFFFu;
        const u32 count = body_words - 1u;
        const u32 dst_reg = Bank::Context + ctx_off;
        const u32* src = body + 1u;
        bool any_changed = false;
        for (u32 i = 0; i < count; ++i) {
            if (regs.reg_array[dst_reg + i] != src[i]) {
                regs.reg_array[dst_reg + i] = src[i];
                any_changed = true;
            }
        }
        if (any_changed || track_all) {
            MarkReg(dirty, dst_reg, count);
        }
        if (any_changed) {
            if (dyn_reg_epoch) {
                ++(*dyn_reg_epoch);
            }
            if (pso_sig_epoch && ContextSpanAffectsPsoSig(ctx_off, count)) {
                ++(*pso_sig_epoch);
            }
        }
        break;
    }
    case Op::SetShReg: {
        // The compute program goes to the queue's compute state, not the file.
        const u32 sh_off = body[0] & 0xFFFFu;
        if (InComputeRange(sh_off)) {
            return false;
        }
        const u32 count = body_words - 1u;
        const u32 dst_reg = Bank::Sh + sh_off;
        const u32* src = body + 1u;
        bool any_changed = false;
        for (u32 i = 0; i < count; ++i) {
            if (regs.reg_array[dst_reg + i] != src[i]) {
                regs.reg_array[dst_reg + i] = src[i];
                any_changed = true;
            }
        }
        MarkShReg(dirty, track_all, dst_reg, count);
        if (any_changed && pso_sig_epoch && (sh_off & 0x3Fu) < 0x0Cu) {
            ++(*pso_sig_epoch);
        }
        handled_completely = true;
        break;
    }
    case Op::SetUconfigReg:
        if (SetBankRegs(regs, dirty, track_all, body, body_words, Bank::Uconfig)) {
            BumpPsoAndDyn(pso_sig_epoch, dyn_reg_epoch);
        }
        handled_completely = true;
        break;
    case Op::IndexType:
        WriteReg(regs, dirty, track_all, Reg::VgtDmaIndexType, body[0]);
        handled_completely = true;
        break;
    case Op::DrawIndex2:
        WriteReg(regs, dirty, track_all, Reg::VgtDmaMaxSize, body[0]);
        WriteReg(regs, dirty, track_all, Reg::VgtDmaBase, body[1]);
        WriteReg(regs, dirty, track_all, Reg::VgtDmaBaseHi,
                 (regs.reg_array[Reg::VgtDmaBaseHi] & ~0xFFu) | (body[2] & 0xFFu));
        WriteReg(regs, dirty, track_all, Reg::VgtNumIndices, body[3]);
        WriteReg(regs, dirty, track_all, Reg::VgtDrawInitiator, body[4]);
        break;
    case Op::DrawIndexOffset2:
        WriteReg(regs, dirty, track_all, Reg::VgtDmaMaxSize, body[0]);
        WriteReg(regs, dirty, track_all, Reg::VgtNumIndices, body[2]);
        WriteReg(regs, dirty, track_all, Reg::VgtDrawInitiator, body[3]);
        break;
    case Op::DrawIndexAuto:
        WriteReg(regs, dirty, track_all, Reg::VgtNumIndices, body[0]);
        WriteReg(regs, dirty, track_all, Reg::VgtDrawInitiator, body[1]);
        break;
    case Op::NumInstances:
        WriteReg(regs, dirty, track_all, Reg::VgtNumInstances, body[0]);
        handled_completely = true;
        break;
    case Op::IndexBase:
        WriteReg(regs, dirty, track_all, Reg::VgtDmaBase, body[0]);
        WriteReg(regs, dirty, track_all, Reg::VgtDmaBaseHi,
                 (regs.reg_array[Reg::VgtDmaBaseHi] & ~0xFFu) | (body[1] & 0xFFu));
        handled_completely = true;
        break;
    case Op::IndexBufferSize:
        WriteReg(regs, dirty, track_all, Reg::VgtNumIndices, body[0]);
        handled_completely = true;
        break;
    case Op::EventWrite:
        if ((body[0] & 0x3F) != EventWrite::SoVgtStreamoutFlush) {
            return false;
        }
        WriteReg(regs, dirty, track_all, Reg::CpStrmoutCntl,
                 regs.reg_array[Reg::CpStrmoutCntl] | 1u);
        BumpPsoAndDyn(pso_sig_epoch, dyn_reg_epoch);
        break;
    case Op::Nop: {
        if (body_words < 2 || body[0] != NopPayload::DirectVulkan) {
            return false;
        }
        const u32 op_word = body[1];
        const auto vk_op = DirectVk::Op(op_word & 0xFFFFu);
        switch (vk_op) {
        case DirectVk::Op::SetIndexState:
        case DirectVk::Op::SetStageUserData:
        case DirectVk::Op::SetContextRegsPacked:
        case DirectVk::Op::BindVsShader:
        case DirectVk::Op::BindPsShader:
        case DirectVk::Op::BindEsShader:
        case DirectVk::Op::BindGsShader:
        case DirectVk::Op::BindHsShader:
        case DirectVk::Op::BindLsShader:
        case DirectVk::Op::SetVgtControl: {
            bool pso_chg = false;
            bool dyn_chg = false;
            if (!ApplyDirectStateSubOp(regs, dirty, track_all, vk_op, op_word, body + 2u,
                                       body_words - 2u, pso_chg, dyn_chg)) {
                return false;
            }
            if (pso_chg && pso_sig_epoch) {
                ++(*pso_sig_epoch);
            }
            if (dyn_chg && dyn_reg_epoch) {
                ++(*dyn_reg_epoch);
            }
            handled_completely = true;
            break;
        }
        case DirectVk::Op::DrawIndexed:
            if (body_words != 6u) {
                return false;
            }
            ApplyDirectDrawIndexedWords(regs, dirty, track_all, body + 2u);
            break;
        case DirectVk::Op::DrawIndexAddr: {
            if (body_words != 6u) {
                return false;
            }
            bool pso_chg = false;
            bool dyn_chg = false;
            ApplyDirectDrawIndexAddrWords(regs, dirty, track_all, body + 2u, pso_chg, dyn_chg);
            if (pso_chg && pso_sig_epoch) {
                ++(*pso_sig_epoch);
            }
            if (dyn_chg && dyn_reg_epoch) {
                ++(*dyn_reg_epoch);
            }
            break;
        }
        case DirectVk::Op::DrawAuto:
            if (body_words != 5u) {
                return false;
            }
            ApplyDirectDrawAutoWords(regs, dirty, track_all, body + 2u);
            break;
        case DirectVk::Op::DrawBundle:
            if (!ApplyDrawBundlePacket(regs, dirty, track_all, op_word, body, body_words,
                                       pso_sig_epoch, dyn_reg_epoch)) [[unlikely]] {
                return false;
            }
            break;
        case DirectVk::Op::SetRenderTarget:
        case DirectVk::Op::SetDepthRenderTarget:
            if (!ApplySetRenderTargetPacket(regs, dirty, track_all, vk_op, op_word, body,
                                            body_words, pso_sig_epoch, dyn_reg_epoch)) {
                return false;
            }
            break;
        case DirectVk::Op::InitDefaultHardwareState:
            if (body_words != 2u || !ApplyInitDefaultHardwareState(regs, dirty, track_all, op_word,
                                                                   pso_sig_epoch, dyn_reg_epoch)) {
                return false;
            }
            handled_completely = true;
            break;
        case DirectVk::Op::ExecuteSecondaryCmds: {
            if (body_words != 6u) {
                return false;
            }
            DirectVk::ExecuteSecondaryArgs args{};
            std::memcpy(&args, body + 2u, sizeof(args));
            if ((args.occ_and_flags & DirectVk::kExecSecondaryFlagSyncRegs) == 0u) {
                return false;
            }
            if (auto* rec = Vulkan::VkMtDirectRecorder::ActiveInstance()) {
                if (const auto* meta = rec->LookupCompletedPass(args.worker_and_draws & 0xFFu,
                                                                args.cmd_seq);
                    meta && meta->has_end_sync) {
                    meta->ApplyEndSyncToRegs(regs, dirty, track_all);
                    BumpPsoAndDyn(pso_sig_epoch, dyn_reg_epoch);
                }
            }
            break;
        }
        default:
            return false;
        }
        break;
    }
    default:
        return false;
    }
    if (compute_checksum) {
        checksum = HashRegisterPacket(checksum, packet, body_words + 1);
    }
    return handled_completely;
}

// ---- Thread ----------------------------------------------------------------------------------

CommandProcessor::CommandProcessor()
    : guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    const char* desc_env = std::getenv("BB_DRAW_DESC");
    const char* verify_env = std::getenv("BB_PIPE_VERIFY");
    const bool use_draw_desc = !(desc_env && desc_env[0] == '0') &&
                               !(verify_env && std::strtoul(verify_env, nullptr, 10) != 0);
    pipe_dirty.track_all_blocks = !use_draw_desc;
    thread = std::jthread{[this](std::stop_token stop) { Run(stop); }};
    if (const char* env = std::getenv("BB_STALL_REPORT"); !(env && env[0] == '0')) {
        stall_watch = std::jthread{[this](std::stop_token stop) { StallWatch(stop); }};
    }
}

CommandProcessor::~CommandProcessor() {
    if (stall_watch.joinable()) {
        stall_watch.request_stop();
        stall_watch.join();
    }
    thread.request_stop();
    {
        std::scoped_lock lock{submit_mutex};
        submit_cv.notify_all();
    }
    thread.join();
}

void CommandProcessor::SubmitDone(u64 frame) noexcept {
    std::scoped_lock lock{submit_mutex};
    if (frame != 0) {
        frame_ends.emplace_back(submissions_total, frame);
        has_pending_frame_end.store(true, std::memory_order_release);
    } else {
        ++submissions_total;
        work_retired = false;
    }
    submit_done = true;
    submit_cv.notify_one();
}

void CommandProcessor::WaitGpuIdle() noexcept {
    std::unique_lock lock{submit_mutex};
    submit_cv.wait(lock, [this] { return num_submits == 0; });
}

bool CommandProcessor::IsGpuIdle() const {
    static const bool use_retired = [] {
        const char* env = std::getenv("BB_WORK_RETIRED");
        return !(env && env[0] == '0');
    }();
    // BB_SUBMIT_LOCK=decode with honest labels: decoded is enough.
    return num_submits == 0 && (work_retired || !use_retired || BbGnm::SubmitLockOnDecode());
}

void CommandProcessor::RunCommands() {
    while (num_commands) {
        // Commands touch the caches and record into the scheduler: the draw recording thread
        // must be idle, for each one (one arriving between two checks ran beside it before).
        if (rasterizer) {
            rasterizer->DrainDrawPipe(DrawPipe::ReasonCommands);
        }
        Common::UniqueFunction<void> command;
        {
            std::scoped_lock lock{submit_mutex};
            command = std::move(commands.front());
            commands.pop_front();
            --num_commands;
        }
        command();
    }
}

void CommandProcessor::WaitForWork(std::stop_token stop) {
    const auto has_work = [this] { return num_commands || num_submits || submit_done; };
    const auto idle_start = std::chrono::steady_clock::now();
    u64 flush_ns = 0;
    // BB_HONEST_LABELS: fences (and GPU idle) wait for work not submitted yet; flush unsubmitted
    // signals after a brief 20 us coalesce window instead of sleeping 200+ us in a kernel futex.
    if (rasterizer && Vulkan::Rasterizer::HonestLabels() &&
        (rasterizer->HasUnsubmittedSignals() || !rasterizer->DrawPipeIdle())) {
        static const auto idle_wait = std::chrono::microseconds([] {
            const char* env = std::getenv("BB_HONEST_IDLE_US");
            return env ? std::strtoll(env, nullptr, 10) : 200ll;
        }());
        static const bool spin_wait = [] {
            const char* env = std::getenv("BB_GPU_SPIN_US");
            return env && std::atoi(env) > 0;
        }();
        bool got_work = has_work();
        if (!got_work && idle_wait.count() > 0) {
            if (spin_wait) {
                const auto until = std::chrono::steady_clock::now() + idle_wait;
                for (u32 s = 1; !has_work() && !stop.stop_requested(); ++s) {
                    _mm_pause();
                    if (!(s & 63u) && std::chrono::steady_clock::now() >= until) {
                        break;
                    }
                }
                got_work = has_work();
            } else {
                std::unique_lock lock{submit_mutex};
                got_work = submit_cv.wait_for(lock, idle_wait, has_work);
            }
        }
        if (!got_work) {
            const auto flush_start = std::chrono::steady_clock::now();
            if (rasterizer->PipeSignals()) {
                // The recording thread submits once it gets there (waiting for it to run dry,
                // tens of ms in heavy scenes, held up the guest's next submission).
                rasterizer->RequestSignalFlush();
            } else {
                rasterizer->Flush();
            }
            flush_ns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - flush_start)
                               .count());
        }
    }
    BbTimeline::Note(BbTimeline::DecoderIdleBegin);
    // Avoid empty PAUSE busy-spin waste by default (`BB_GPU_SPIN_US` opts in).
    static const auto spin_time = std::chrono::microseconds([] {
        if (const char* env = std::getenv("BB_GPU_SPIN_US")) {
            return std::max(0, std::atoi(env));
        }
        return 0;
    }());
    if (spin_time.count() > 0) {
        const auto spin_until = std::chrono::steady_clock::now() + spin_time;
        for (u32 spins = 1; !has_work() && !stop.stop_requested(); ++spins) {
            _mm_pause();
            if (!(spins & 255) && std::chrono::steady_clock::now() >= spin_until) {
                break;
            }
        }
    }
    {
        std::unique_lock lock{submit_mutex};
        Common::CondvarWait(submit_cv, lock, stop, has_work);
    }
    const u64 waited_ns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - idle_start)
                                  .count());
    BbStats::gpu_idle_ns.fetch_add(waited_ns > flush_ns ? waited_ns - flush_ns : 0,
                                   std::memory_order_relaxed);
    BbTimeline::Note(BbTimeline::DecoderIdleEnd);
}

void CommandProcessor::StallWatch(std::stop_token stop) {
    Common::SetCurrentThreadName("bb:StallWatch");
    BbThreads::PinToHelperDomain();
    BbThreads::ApplyProcessThreadTopology();
    using Clock = std::chrono::steady_clock;
    constexpr auto Limit = std::chrono::seconds(5);
    u64 last_submitted = ~0ull, last_decoded = ~0ull;
    auto submitted_at = Clock::now(), decoded_at = Clock::now();
    bool reported = false;
    while (!stop.stop_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        BbThreads::ApplyProcessThreadTopology();
        u64 submitted, decoded, pending;
        {
            std::scoped_lock lock{submit_mutex};
            submitted = submissions_total;
            decoded = decoded_total;
            pending = num_submits;
        }
        const auto now = Clock::now();
        if (submitted != last_submitted) {
            last_submitted = submitted;
            submitted_at = now;
            reported = false;
        }
        if (decoded != last_decoded || pending == 0) {
            last_decoded = decoded;
            decoded_at = now;
        }
        if (reported || submitted == 0) {
            continue;
        }
        if (now - decoded_at >= Limit) {
            reported = true;
            ReportStall("the queues have made no progress for 5 s");
        } else if (now - submitted_at >= Limit) {
            reported = true;
            ReportStall("the guest has submitted nothing for 5 s");
        }
    }
}

void CommandProcessor::ReportStall(const char* why) {
    std::string out;
    {
        std::scoped_lock lock{submit_mutex};
        out = fmt::format("GPU stall report: {} (submitted {}, decoded {}, waiting {}, commands "
                          "{})\n",
                          why, submissions_total, decoded_total, num_submits.load(),
                          num_commands.load());
    }
    // What each queue is at (read beside the command thread: a snapshot, maybe torn).
    for (u32 q = 0; q < num_mapped_queues && q < NumTotalQueues; ++q) {
        Queue& queue = queues[q];
        std::scoped_lock lock{queue.mutex};
        if (q != GfxQueueId && asc_queues.is_allocated(Common::SlotId{q - 1})) {
            const auto& ring = asc_queues[Common::SlotId{q - 1}];
            u32 read = 0;
            const bool readable = Diagnostics::ReadGuest(ring.read_addr, &read, sizeof(read));
            out += fmt::format("  compute queue {} (pipe {}): ring {:#x}, {} dwords, read pointer "
                               "{}{}, {} submissions queued\n",
                               q, ring.pipe_id, ring.map_addr, ring.ring_size_dw,
                               readable ? fmt::format("{}", read) : std::string("?"),
                               ring.tmp_dwords ? fmt::format(" ({} dwords of a split packet)",
                                                             ring.tmp_dwords)
                                               : std::string{},
                               queue.works.size());
        } else if (!queue.works.empty()) {
            out += fmt::format("  queue {}: {} submissions queued\n", q, queue.works.size());
        }
        if (queue.works.empty()) {
            continue;
        }
        const Work& work = *queue.works.front();
        if (!work.started || work.walk.top < 0) {
            out += "    first one not started\n";
            continue;
        }
        const Level& level = work.walk.levels[work.walk.top];
        u32 header = 0;
        const bool readable =
            level.at < level.end && Diagnostics::ReadGuest(level.at, &header, sizeof(header));
        out += fmt::format("    first one at depth {}: {:#x}, dword {} of {}, packet {}{}\n",
                           work.walk.top, reinterpret_cast<uintptr_t>(level.at),
                           level.at - level.begin, level.end - level.begin,
                           readable && (header >> 30) == 3
                               ? Diagnostics::OpName(u32(Header{header}.Opcode()))
                               : fmt::format("{:#x}", header),
                           work.waiting_at ? " (waiting)" : "");
    }
    if (rasterizer) {
        out += rasterizer->DescribeSignals();
    }
    out += "  last labels written for the guest (oldest first):\n";
    out += Diagnostics::RecentLabels(40);
    // The command thread's frames (the runtime's SIGUSR2 handler prints its frame-pointer chain
    // to stderr; addresses minus this library's base give the source lines).
    Dl_info self{};
    if (dladdr(reinterpret_cast<void*>(&BodyWords), &self)) {
        out += fmt::format("  command thread {} stack follows ({} at {:#x}):\n", thread_tid,
                           self.dli_fname ? self.dli_fname : "?",
                           reinterpret_cast<uintptr_t>(self.dli_fbase));
    }
    std::fputs(out.c_str(), stdout);
    std::fflush(stdout);
#ifndef _WIN32
    if (thread_tid != 0) {
        syscall(SYS_tgkill, getpid(), thread_tid, SIGUSR2);
    }
#endif
#ifndef _WIN32
    // Every thread of the process prints its frames and registers (the runtime's SIGUSR2 handler):
    // when the guest itself stops (its job system, a lock), they say what each one waits for.
    // BB_STALL_THREADS=0: the command thread only.
    static const bool all_threads = [] {
        const char* env = std::getenv("BB_STALL_THREADS");
        return !(env && env[0] == '0');
    }();
    if (all_threads && !std::getenv("BB_STALL_THREADS_DONE")) {
        setenv("BB_STALL_THREADS_DONE", "1", 1); // once per run
        std::fputs("  all threads follow (frames: image-relative, then registers and stack):\n", stdout);
        std::fflush(stdout);
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator("/proc/self/task", ec)) {
            const long tid = std::strtol(entry.path().filename().c_str(), nullptr, 10);
            if (tid > 0 && tid != long(thread_tid)) {
                syscall(SYS_tgkill, getpid(), tid, SIGUSR2);
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
            }
        }
    }
#endif
}

void CommandProcessor::Run(std::stop_token stop) {
    Common::SetCurrentThreadName("bb:GpuComm");
    BbSections::command_thread = true;
    BbThreads::PinToGpuCommCore();
    BbThreads::ApplyProcessThreadTopology();
    if (clockid_t clock; pthread_getcpuclockid(pthread_self(), &clock) == 0) {
        BbStats::gpu_thread_clock.store(int(clock));
    }
    thread_id = std::this_thread::get_id();
    thread_tid = u32(gettid());

    while (!stop.stop_requested()) {
        WaitForWork(stop);
        if (stop.stop_requested()) {
            break;
        }
        // The queues in turn: each one's front submission runs until it is done or waits.
        current_queue = -1;
        while (num_submits || num_commands) {
            RunCommands();
            current_queue = int((current_queue + 1) % num_mapped_queues);
            Queue& queue = queues[current_queue];
            Work* work;
            {
                std::scoped_lock lock{queue.mutex};
                if (queue.works.empty()) {
                    continue;
                }
                work = queue.works.front().get();
            }
            if (work != last_decoded) {
                if (has_pending_frame_end.load(std::memory_order_acquire)) {
                    SignalDecodedFrames();
                }
                last_decoded = work;
                BbTimeline::Note(BbTimeline::DecodeBegin, u64(current_queue), decoded_total);
                if (rasterizer && (VideoCore::DirectTextures::HasPendingEvents() ||
                                   VideoCore::DirectBuffers::HasPendingEvents())) {
                    if (rasterizer->PipeSignals() ||
                        (rasterizer->PipeOrder() &&
                         !BbToggle::Disabled(BbToggle::PipelinedTasks))) {
                        static constexpr u8 none = 0;
                        rasterizer->RunInOrder(
                            [](Vulkan::Rasterizer& r, const u8*) {
                                r.ProcessDirectTextureEvents();
                            },
                            &none, 0, BbToggle::PipelinedTasks, false);
                    } else {
                        rasterizer->DrainDrawPipe();
                        rasterizer->ProcessDirectTextureEvents();
                    }
                }
            }
            Status status;
            {
                BB_SECTION(ADecode);
                status = current_queue == GfxQueueId ? RunGraphics(*work) : RunCompute(*work);
            }
            if (status == Status::Wait) {
                _mm_pause(); // the other queues' turn
                continue;
            }
            BbTimeline::Note(BbTimeline::DecodeEnd, u64(current_queue), decoded_total + 1);
            bool frame_end;
            {
                std::scoped_lock lock{queue.mutex};
                queue.works.pop_front();
                last_decoded = nullptr;
                const u32 remaining = --num_submits;
                std::scoped_lock submit_lock{submit_mutex};
                ++decoded_total;
                frame_end = !frame_ends.empty() && frame_ends.front().first <= decoded_total;
                if (remaining == 0 || frame_end) {
                    submit_cv.notify_all();
                }
            }
            if (frame_end) {
                SignalDecodedFrames(); // BB_SUBMIT_LOCK=frame: a frame's last submission decoded
            }
        }
        {
            BB_SECTION(ADecode);
            AfterBatch();
        }
    }
}

void CommandProcessor::AfterBatch() {
    SignalDecodedFrames();
    const bool pipe_signals = rasterizer && rasterizer->PipeSignals();
    const bool pipe_order =
        rasterizer && rasterizer->PipeOrder() &&
        !BbToggle::Disabled(BbToggle::PipelinedTasks);
    const bool was_submit_done = submit_done.exchange(false, std::memory_order_acq_rel);

    // The guest takes GPU idle (sceGnmSubmitDone) as its work being done and frees the objects
    // that hold its fence labels: they must be written before, by the draw recording thread and
    // the fences it deferred. (BB_PIPE_SIGNALS: that thread does both in order, below.)
    if (rasterizer && !pipe_signals && !pipe_order) {
        rasterizer->DrainDrawPipe(DrawPipe::ReasonSubmissionEnd);
        rasterizer->WaitDeferredSignals();
    }

    if (was_submit_done && (pipe_signals || pipe_order)) {
        // BB_PIPE_SIGNALS: the frame's cache work runs on the recording thread after the
        // frame (this thread decodes the next one meanwhile).
        static constexpr u8 none = 0;
        rasterizer->RunInOrder(
            [](Vulkan::Rasterizer& r, const u8*) {
                r.OnSubmit();
                if (r.HasPendingWork()) {
                    r.Flush();
                }
                // BB_EXP_FINISH_FRAMES=1 (experiment): the GPU finishes each frame before the next.
                static const bool finish = std::getenv("BB_EXP_FINISH_FRAMES") != nullptr;
                if (finish) {
                    r.Finish();
                }
            },
            &none, 0, BbToggle::PipelinedTasks, false);
    }

    // GPU idle only if nothing was submitted meanwhile: the drain and the deferred fences take
    // milliseconds, and a submission plus sceGnmSubmitDone in that time would release the lock for
    // work not even decoded (late fences in the guest's heap, command buffers reused as garbage).
    if (rasterizer && Vulkan::Rasterizer::HonestLabels()) {
        u64 generation;
        bool decoded;
        {
            std::scoped_lock lock{submit_mutex};
            generation = submissions_total;
            decoded = num_submits == 0;
        }
        // BB_SUBMIT_LOCK=decode: everything is decoded; the next submission need not wait for
        // the GPU (the idle interrupt still does).
        if (BbGnm::SubmitLockOnDecode() && decoded) {
            BbGnm::ReleaseSubmissionLock();
        }
        if (pipe_signals) {
            struct IdleSignal {
                CommandProcessor* self;
                u64 generation;
            } const signal{this, generation};
            rasterizer->RunInOrder(
                [](Vulkan::Rasterizer& r, const u8* data) {
                    IdleSignal idle;
                    std::memcpy(&idle, data, sizeof(idle));
                    r.WaitDeferredSignals();
                    r.SignalAfterGpu([idle] { idle.self->SignalGpuIdle(idle.generation); });
                    r.SubmitForSignals(); // at once if the GPU has run dry
                    r.WaitPendingFrame();
                },
                &signal, sizeof(signal), BbToggle::PipelinedTasks, false);
            return;
        }
        rasterizer->SignalAfterGpu([this, generation] { SignalGpuIdle(generation); });
        if (was_submit_done && !pipe_order) {
            rasterizer->OnSubmit();
            if (rasterizer->HasPendingWork()) {
                rasterizer->Flush();
            } else {
                rasterizer->SubmitForSignals();
            }
        }
        rasterizer->WaitPendingFrame();
        return;
    }

    if (was_submit_done && rasterizer && !pipe_signals && !pipe_order) {
        rasterizer->OnSubmit();
        if (rasterizer->HasPendingWork()) {
            rasterizer->Flush();
        }
    }
    if (pipe_order) {
        u64 generation;
        {
            std::scoped_lock lock{submit_mutex};
            generation = submissions_total;
        }
        struct IdleSignal {
            CommandProcessor* self;
            u64 generation;
            bool wait_frame;
        } const signal{this, generation, was_submit_done};
        rasterizer->RunInOrder(
            [](Vulkan::Rasterizer& r, const u8* data) {
                IdleSignal idle;
                std::memcpy(&idle, data, sizeof(idle));
                r.WaitDeferredSignals();
                idle.self->SignalGpuIdle(idle.generation);
                if (idle.wait_frame) {
                    r.WaitPendingFrame();
                }
            },
            &signal, sizeof(signal), BbToggle::PipelinedTasks, false);
        return;
    }
    bool idle;
    {
        std::scoped_lock lock{submit_mutex};
        idle = num_submits == 0;
        if (idle) {
            work_retired = true;
        }
    }
    if (idle) {
        BbIrq::Raise(BbIrq::Line::GpuIdle);
    }
    if (rasterizer && was_submit_done) {
        rasterizer->WaitPendingFrame();
    }
}

void CommandProcessor::SignalDecodedFrames() {
    u64 frame = 0;
    {
        std::scoped_lock lock{submit_mutex};
        while (!frame_ends.empty() && frame_ends.front().first <= decoded_total) {
            frame = frame_ends.front().second;
            frame_ends.pop_front();
        }
        if (frame_ends.empty()) {
            has_pending_frame_end.store(false, std::memory_order_release);
        }
    }
    if (frame == 0) {
        return;
    }
    const bool do_on_submit = submit_done.exchange(false, std::memory_order_acq_rel);
    if (!rasterizer) {
        BbGnm::NoteFramesRetired(frame);
        return;
    }
    struct FrameSignal {
        u64 retired;
        bool on_submit;
    } const sig{frame, do_on_submit};
    // Everything of the frame recorded, then submitted: finished once the GPU has executed it
    // (HonestLabels), or once Stage B has recorded/flushed all commands and EoP labels of frame F
    // in the 0.3 VRAM model (where all transient guest data is copied out into ConstantRing /
    // stream_buffer / native VkBuffers before Flush, while WaitPendingFrame bounds GPU to 2-FIF).
    if (rasterizer->PipeSignals()) {
        rasterizer->RunInOrder(
            [](Vulkan::Rasterizer& r, const u8* data) {
                FrameSignal s;
                std::memcpy(&s, data, sizeof(s));
                if (s.on_submit) {
                    r.OnSubmit();
                }
                BbTimeline::Note(BbTimeline::PipeTask, 7, r.CurrentTick());
                r.SignalAfterGpu([retired = s.retired] { BbGnm::NoteFramesRetired(retired); });
                r.Flush();
            },
            &sig, sizeof(sig), BbToggle::PipelinedTasks, false);
        return;
    }
    if (!Vulkan::Rasterizer::HonestLabels()) {
        if (rasterizer->PipeOrder() && !BbToggle::Disabled(BbToggle::PipelinedTasks)) {
            rasterizer->RunInOrder(
                [](Vulkan::Rasterizer& r, const u8* data) {
                    FrameSignal s;
                    std::memcpy(&s, data, sizeof(s));
                    if (s.on_submit) {
                        r.OnSubmit();
                    }
                    if (r.HasPendingWork()) {
                        r.Flush();
                    }
                    BbTimeline::Note(BbTimeline::PipeTask, 7, r.CurrentTick());
                    r.WaitDeferredSignals();
                    BbGnm::NoteFramesRetired(s.retired);
                    if (s.on_submit) {
                        r.WaitPendingFrame();
                    }
                },
                &sig, sizeof(sig), BbToggle::PipelinedTasks, false);
            return;
        }
        rasterizer->DrainDrawPipe(DrawPipe::ReasonSubmissionEnd);
        if (do_on_submit) {
            rasterizer->OnSubmit();
            if (rasterizer->HasPendingWork()) {
                rasterizer->Flush();
            }
        }
        rasterizer->WaitDeferredSignals();
        BbGnm::NoteFramesRetired(frame);
        if (do_on_submit) {
            rasterizer->WaitPendingFrame();
        }
        return;
    }
    rasterizer->DrainDrawPipe(DrawPipe::ReasonSubmissionEnd);
    if (do_on_submit) {
        rasterizer->OnSubmit();
    }
    rasterizer->SignalAfterGpu([frame] { BbGnm::NoteFramesRetired(frame); });
    rasterizer->Flush();
}

void CommandProcessor::SignalGpuIdle(u64 generation) {
    bool idle;
    {
        std::scoped_lock lock{submit_mutex};
        idle = num_submits == 0 && submissions_total == generation;
        if (idle) {
            work_retired = true;
        }
    }
    if (idle) {
        BbIrq::Raise(BbIrq::Line::GpuIdle);
    }
}

// ---- Submissions -----------------------------------------------------------------------------

void CommandProcessor::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb, bool force_copy) {
    Queue& queue = queues[GfxQueueId];
    auto work = std::make_unique<Work>();
    work->queue = GfxQueueId;
    // BB_COPY_GPU_BUFFERS: one copy per submission, alive until it is decoded (the guest reused
    // the memory before this thread got to it: "PM4 type 0" while an area loads).
    if (force_copy || EmulatorSettings.IsCopyGpuBuffers()) {
        auto copy = Diagnostics::AcquireSubmittedCopy(dcb.size() + ccb.size());
        std::memcpy(copy->data.data(), dcb.data(), dcb.size_bytes());
        if (!ccb.empty()) {
            std::memcpy(copy->data.data() + dcb.size(), ccb.data(), ccb.size_bytes());
        }
        copy->guest_dcb = force_copy ? copy->data.data() : dcb.data();
        copy->dcb_dwords = dcb.size();
        copy->guest_ccb = force_copy ? copy->data.data() + dcb.size() : ccb.data();
        copy->ccb_dwords = ccb.size();
        copy->submitted = std::chrono::steady_clock::now();
        if (Diagnostics::PreflightInspectEnabled()) {
            Diagnostics::InspectSubmittedCopy(*copy);
        }
        Diagnostics::RememberSubmittedCopy(copy);
        dcb = {copy->data.data(), copy->dcb_dwords};
        ccb = {copy->data.data() + copy->dcb_dwords, copy->ccb_dwords};
        work->copy = std::move(copy);
    }
    work->dcb = dcb;
    work->ccb = ccb;
    // N3.4a: Zero-copy intake sharing `work->copy`; flattening, draw counting and PreparedDraw[]
    // allocation run asynchronously on bb:DrawScan (CCD1) off the guest GXRenderThread.
    auto* draw_prep = rasterizer ? &rasterizer->GetDrawPreparation() : nullptr;
    auto prepared = draw_prep ? draw_prep->CreateRaw(dcb, work->copy) : nullptr;
    {
        // Numbering, enqueueing and queueing under one lock keep the three orders the same.
        std::scoped_lock lock{queue.mutex};
        if (draw_prep) {
            work->seq = gfx_submit_seq++;
            draw_prep->Enqueue(work->seq, std::move(prepared));
        }
        BbFreeCheck::NoteSubmit(work->seq, work->copy ? work->copy->guest_dcb : dcb.data(),
                                dcb.size_bytes());
        queue.works.push_back(std::move(work));
    }
    std::scoped_lock lock{submit_mutex};
    ++num_submits;
    ++submissions_total;
    BbTimeline::Note(BbTimeline::GuestSubmit, 0, submissions_total);
    work_retired = false;
    submit_cv.notify_one();
}

void CommandProcessor::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "bad compute queue {}", gnm_vqid);
    BbStats::asc_submits.fetch_add(1, std::memory_order_relaxed);
    Queue& queue = queues[gnm_vqid];
    auto work = std::make_unique<Work>();
    work->queue = gnm_vqid;
    work->dcb = acb;
    {
        std::scoped_lock lock{queue.mutex};
        queue.works.push_back(std::move(work));
    }
    std::scoped_lock lock{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    ++submissions_total;
    work_retired = false;
    submit_cv.notify_one();
}

} // namespace BbCp
