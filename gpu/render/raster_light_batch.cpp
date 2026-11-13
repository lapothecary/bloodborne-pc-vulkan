// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 8: Stage-A batcher for Bloodborne's deferred local-light tile quads (`GXLightAcc`).

#include "render/vk_light_tile_batch.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "buffers/direct_buffers.h"
#include "common/alignment.h"
#include "cp/cp_tasks.h"
#include "render/raster_internal.h"
#include "render/vk_vsm_shadow_memo.h"
#include "render/vk_vfx_ui_batch.h"
#include "shader/backend/spirv/native_shaders.h"
#include "upscale/vk_scene_resolution.h"
#include "../../dantelion_gx/include/gx_direct_vulkan.h"

extern "C" bool BbLookupMappedEnd(u64 address, uintptr_t& out_end);

namespace Vulkan {

namespace {

constexpr u64 kLightAccVsHash = 0x000000008747a367ULL;

constexpr s32 RegWordToStageBSlot(u32 reg_word) noexcept {
    if (reg_word >= 0xA000u && reg_word < 0xA000u + 114u * 8u) {
        return static_cast<s32>(reg_word - 0xA000u);
    }
    if (reg_word >= 0xC038u && reg_word < 0xC040u) {
        return static_cast<s32>(114u * 8u + (reg_word - 0xC038u));
    }
    if (reg_word >= 0xC240u && reg_word < 0xC240u + 3u * 8u) {
        return static_cast<s32>(115u * 8u + (reg_word - 0xC240u));
    }
    return -1;
}

void ZeroStageBWordRange(LightTileBatcher::StageBChunks& c, u32 first_word,
                         u32 word_count) noexcept {
    for (u32 w = first_word; w < first_word + word_count; ++w) {
        const s32 slot = RegWordToStageBSlot(w);
        if (slot >= 0) {
            c[static_cast<size_t>(slot)] = 0u;
        }
    }
}

void CaptureStageBChunks(const Gcn::Regs& regs, LightTileBatcher::StageBChunks& out) noexcept {
    for (u32 bit = 0; bit < Gcn::RegDirty::StageBNumChunks; ++bit) {
        std::memcpy(&out[bit * Gcn::RegDirty::StageBChunkWords],
                    regs.reg_array.data() + Gcn::RegDirty::StageBChunkWordOffset(bit),
                    Gcn::RegDirty::StageBChunkWords * sizeof(u32));
    }
}

void RestoreStageBChunks(Gcn::Regs& regs, const LightTileBatcher::StageBChunks& in) noexcept {
    for (u32 bit = 0; bit < Gcn::RegDirty::StageBNumChunks; ++bit) {
        std::memcpy(regs.reg_array.data() + Gcn::RegDirty::StageBChunkWordOffset(bit),
                    &in[bit * Gcn::RegDirty::StageBChunkWords],
                    Gcn::RegDirty::StageBChunkWords * sizeof(u32));
    }
}

void NormalizeStageBChunksForPass(LightTileBatcher::StageBChunks& c) noexcept {
    // Per-tile depth bounds and scissors:
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, depth_bounds_min) / 4u), 2u);
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, screen_scissor) / 4u), 2u);
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, viewport_scissors) / 4u), 2u);

    // Mask out only `depth_bounds_enable` in DB_DEPTH_CONTROL (`depth_test`, `stencil`, etc. stay):
    {
        Gcn::DepthControl db_mask{};
        db_mask.depth_bounds_enable = 1u;
        const s32 dc_slot =
            RegWordToStageBSlot(static_cast<u32>(offsetof(Gcn::Regs, depth_control) / 4u));
        if (dc_slot >= 0) {
            c[static_cast<size_t>(dc_slot)] &= ~std::bit_cast<u32>(db_mask);
        }
    }

    // Per-draw counts / initiator:
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, index_offset) / 4u), 1u);
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, index_base_address) / 4u), 2u);
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, draw_initiator) / 4u), 1u);
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, num_indices) / 4u), 1u);

    // PS/VS shader-linkage & DB_SHADER_CONTROL registers (`[0xA191, 0xA1C6)` + `0xA203`): baked
    // into `GraphicsPipelineKey`, never read dynamically on Stage B, and may differ across PS
    // permutations in the same `GXLightAcc` pass.
    ZeroStageBWordRange(c, 0xA191u, 0xA1C6u - 0xA191u);
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, depth_shader_control) / 4u), 1u);
}

void MarkStageBDiffDirty(const LightTileBatcher::StageBChunks& prev,
                         const LightTileBatcher::StageBChunks& next,
                         Gcn::RegDirty& dirty) noexcept {
    for (u32 bit = 0; bit < Gcn::RegDirty::StageBNumChunks; ++bit) {
        if (std::memcmp(&prev[bit * Gcn::RegDirty::StageBChunkWords],
                        &next[bit * Gcn::RegDirty::StageBChunkWords],
                        Gcn::RegDirty::StageBChunkWords * sizeof(u32)) != 0) {
            if (bit < 64u) {
                dirty.stage_b_mask[0] |= (1ULL << bit);
            } else {
                dirty.stage_b_mask[1] |= (1ULL << (bit - 64u));
            }
            const u32 word_off = Gcn::RegDirty::StageBChunkWordOffset(bit);
            dirty.blocks.set(word_off / Gcn::RegDirty::BlockWords);
        }
    }
}

bool IsCommutativeLightBlend(const GraphicsPipeline* pipeline) noexcept {
    const auto& key = pipeline->GetGraphicsKey();
    for (u32 cb = 0; cb < Gcn::NUM_COLOR_BUFFERS; ++cb) {
        if ((key.mrt_mask & (1u << cb)) == 0u || u32(key.write_masks[cb]) == 0u) {
            continue;
        }
        const auto& bl = key.blend_controls[cb];
        if (!bl.enable) {
            return false;
        }
        // All 34 GXLightAcc local-light fragment shaders output constant `.w = 1.0` on all active
        // MRTs, so both `One + One` and `SrcAlpha + One` (`0x60010104`) evaluate to exact pure
        // additive `Src.rgb * 1.0 + Dst.rgb * 1.0` (100% associative & commutative across tiles).
        const bool unit_src = bl.color_src_factor == Gcn::BlendControl::BlendFactor::One ||
                              bl.color_src_factor == Gcn::BlendControl::BlendFactor::SrcAlpha;
        if (bl.color_func != Gcn::BlendControl::BlendFunc::Add || !unit_src ||
            bl.color_dst_factor != Gcn::BlendControl::BlendFactor::One) {
            return false;
        }
        const bool writes_alpha = (u32(key.write_masks[cb]) & 0x8u) != 0u;
        if (writes_alpha && bl.separate_alpha_blend) {
            // In GXLightAcc all active fragments write `frag_color.a == 1.0` (demoted fragments
            // discard before output), so `(One, Add, One)`, `(One, Add, Zero)`, and
            // `(Zero, Add, One)` are all order-independent across light tiles.
            const bool add_one_one =
                bl.alpha_func == Gcn::BlendControl::BlendFunc::Add &&
                bl.alpha_src_factor == Gcn::BlendControl::BlendFactor::One &&
                bl.alpha_dst_factor == Gcn::BlendControl::BlendFactor::One;
            const bool keep_dst =
                bl.alpha_func == Gcn::BlendControl::BlendFunc::Add &&
                bl.alpha_src_factor == Gcn::BlendControl::BlendFactor::Zero &&
                bl.alpha_dst_factor == Gcn::BlendControl::BlendFactor::One;
            const bool const_one =
                bl.alpha_func == Gcn::BlendControl::BlendFunc::Add &&
                bl.alpha_src_factor == Gcn::BlendControl::BlendFactor::One &&
                bl.alpha_dst_factor == Gcn::BlendControl::BlendFactor::Zero;
            if (!add_one_one && !keep_dst && !const_one) {
                return false;
            }
        }
    }
    return true;
}

size_t ShadowPageSet(u64 page2k) noexcept {
    const u64 h = page2k ^ ((page2k * 0x9E3779B97F4A7C15ULL) >> 32);
    return static_cast<size_t>(h & (LightTileBatcher::kShadowSets - 1u));
}

Gcn::Scissor CombinedScissor(const Gcn::Regs& regs) noexcept {
    const bool offset = !regs.window_scissor.window_offset_disable;
    const s16 offset_x = offset ? regs.window_offset.window_x_offset : 0;
    const s16 offset_y = offset ? regs.window_offset.window_y_offset : 0;
    const auto top_left = [](s16 screen, s16 window, s16 generic, s16 window_offset) {
        return std::max({screen, s16(window + window_offset), s16(generic + window_offset)});
    };
    const auto bottom_right = [](s16 screen, s16 window, s16 generic, s16 window_offset) {
        return std::min({screen, s16(window + window_offset), s16(generic + window_offset)});
    };
    Gcn::Scissor scissor{};
    scissor.top_left_x =
        top_left(regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
                 s16(regs.generic_scissor.top_left_x), offset_x);
    scissor.top_left_y =
        top_left(regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
                 s16(regs.generic_scissor.top_left_y), offset_y);
    scissor.bottom_right_x =
        bottom_right(regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
                     regs.generic_scissor.bottom_right_x, offset_x);
    scissor.bottom_right_y =
        bottom_right(regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
                     regs.generic_scissor.bottom_right_y, offset_y);
    return scissor;
}

bool BatchLogEnabled() noexcept {
    static const bool on = [] {
        const char* env = std::getenv("BB_LIGHT_TILE_BATCH_LOG");
        return env && env[0] == '1';
    }();
    return on;
}

} // namespace

LightTileBatcher::LightTileBatcher()
    : shadow_meta(kShadowSlots), shadow_bytes(kShadowSlots * kMaxShadowBytes, 0u),
      coarse_epoch_256k(kCoarseBuckets, 0u) {
    for (auto& g : groups) {
        g.vertices.reserve(512);
        g.ssbo2_dwords.reserve(4096);
        g.ps_images.reserve(20);
        g.ps_samplers.reserve(4);
        for (auto& st : g.stages) {
            st.user_data.reserve(16);
            st.flattened_ud_buf.reserve(64);
        }
    }
}

bool LightTileBatcher::Enabled() noexcept {
    static const bool on = [] {
        const char* env = std::getenv("BB_LIGHT_TILE_BATCH");
        return !(env && env[0] == '0');
    }();
    return on;
}

bool LightTileBatcher::IsBatchedLightAccPsHash(u64 pgm_hash) noexcept {
    switch (pgm_hash) {
    case 0x0000000002cd8f9bULL:
    case 0x0000000007d07cf3ULL:
    case 0x000000000b2ace4cULL:
    case 0x000000000ef50047ULL:
    case 0x00000000149432ffULL:
    case 0x00000000186d69d5ULL:
    case 0x00000000201cb43dULL:
    case 0x0000000022b357f3ULL:
    case 0x000000003105494cULL:
    case 0x00000000370174f7ULL:
    case 0x00000000505ebdbeULL:
    case 0x000000005d7150daULL:
    case 0x000000005de04266ULL:
    case 0x000000005f76eb3dULL:
    case 0x00000000600f86b0ULL:
    case 0x000000006b42fc89ULL:
    case 0x00000000799bfa32ULL:
    case 0x000000007cc2c044ULL:
    case 0x00000000896e7242ULL:
    case 0x0000000089f9ee3eULL:
    case 0x000000008d5387d9ULL:
    case 0x000000008ea5be99ULL:
    case 0x000000009f9ef267ULL:
    case 0x00000000a22c7f71ULL:
    case 0x00000000b59b4c4eULL:
    case 0x00000000b5b52d63ULL:
    case 0x00000000bae5454eULL:
    case 0x00000000c6edb19aULL:
    case 0x00000000d22cbfd1ULL:
    case 0x00000000d75fe2ffULL:
    case 0x00000000e96447c3ULL:
    case 0x00000000f2531727ULL:
    case 0x00000000f40da260ULL:
    case 0x00000000f81596cdULL:
        return true;
    default:
        return false;
    }
}

bool IsReadableCpuSourceRange(VAddr addr, u64 size) noexcept {
    if (!addr || size == 0u) {
        return false;
    }
    if (VideoCore::DirectBuffers::IsTransientCueHeap(addr, size) ||
        VideoCore::DirectBuffers::IsTransientAuxCpuRing(addr, size)) {
        return true;
    }
    uintptr_t mapped_end = 0;
    return BbLookupMappedEnd(addr, mapped_end) && size <= (mapped_end - addr);
}

bool LightTileBatcher::IsBatchSafeCpuConstantCopyTask(Rasterizer::OrderedTask task,
                                                      const void* data,
                                                      u32 size) noexcept {
    if (task == &BbCp::Tasks::DirectVkReleaseLabelsBatchTask ||
        task == &BbCp::Tasks::DirectVkReleaseU32LabelsBatchTask) {
        return data != nullptr && size >= sizeof(u32);
    }
    if (task != &BbCp::Tasks::DirectVkCopyBatchTask || !data || size < sizeof(u32)) {
        return false;
    }
    u32 op_word = 0;
    std::memcpy(&op_word, data, sizeof(op_word));
    const u32 count = op_word >> 16;
    if (count == 0u || count > 32u ||
        size != (1u + count * 6u) * static_cast<u32>(sizeof(u32))) {
        return false;
    }
    const auto* raw = static_cast<const u8*>(data) + sizeof(u32);
    for (u32 i = 0; i < count; ++i) {
        Rasterizer::DirectCopyRegion reg{};
        std::memcpy(&reg, raw + i * sizeof(Rasterizer::DirectCopyRegion), sizeof(reg));
        if (reg.byte_size == 0u) {
            continue;
        }
        if (reg.reserved != gx_direct_vk::kCopyFlagKnownCpuSource ||
            reg.byte_size > kMaxShadowBytes) {
            return false;
        }
    }
    return true;
}

void LightTileBatcher::NoteStageACpuCopyBatch(const u32* raw_regions, u32 count) noexcept {
    if (!raw_regions || count == 0u || count > 32u ||
        (!Enabled() && !VsmShadowMemo::Enabled() && !VfxUiBatcher::Enabled())) {
        return;
    }
    for (u32 i = 0; i < count; ++i) {
        Rasterizer::DirectCopyRegion reg{};
        std::memcpy(&reg, raw_regions + i * 6u, sizeof(reg));
        if (!reg.dst_va || reg.byte_size == 0u) {
            continue;
        }
        if (reg.reserved != gx_direct_vk::kCopyFlagKnownCpuSource || !reg.src_va ||
            reg.byte_size > kMaxShadowBytes ||
            !IsReadableCpuSourceRange(reg.src_va, reg.byte_size)) {
            InvalidateCpuShadow(reg.dst_va, reg.byte_size);
            continue;
        }
        // Invalidate any overlapping partial entries at different base addresses first:
        InvalidateCpuShadow(reg.dst_va, reg.byte_size);

        const size_t set_base = ShadowPageSet(reg.dst_va >> 11) * kShadowWays;
        size_t victim = set_base;
        for (size_t w = 0; w < kShadowWays; ++w) {
            const auto& slot = shadow_meta[set_base + w];
            if (slot.byte_size == 0u || slot.dst_va == reg.dst_va) {
                victim = set_base + w;
                break;
            }
            if (slot.last_use < shadow_meta[victim].last_use) {
                victim = set_base + w;
            }
        }
        auto& dst_slot = shadow_meta[victim];
        dst_slot.dst_va = reg.dst_va;
        dst_slot.byte_size = reg.byte_size;
        dst_slot.last_use = ++shadow_clock;
        dst_slot.coarse_e0 = CoarseEpochAt(reg.dst_va);
        dst_slot.coarse_e1 = CoarseEpochAt(reg.dst_va + reg.byte_size - 1u);
        shadow_min_va = std::min(shadow_min_va, reg.dst_va);
        shadow_max_va = std::max(shadow_max_va, reg.dst_va + reg.byte_size);
        std::memcpy(&shadow_bytes[victim * kMaxShadowBytes],
                    reinterpret_cast<const void*>(reg.src_va), reg.byte_size);
    }
}

void LightTileBatcher::InvalidateCpuShadow(VAddr addr, u64 size) noexcept {
    if (!addr || size == 0u || shadow_max_va == 0u) {
        return;
    }
    const VAddr end = (size > (~0ULL - addr)) ? ~0ULL : (addr + size);
    if (addr >= shadow_max_va || end <= shadow_min_va) {
        return;
    }
    // Small writes (`<= 32 KiB` = 16 x 2 KiB pages): exact O(1) set-associative check without
    // disturbing neighboring CBs in the same 256 KiB coarse slab.
    const u64 p_lo = (addr >= kMaxShadowBytes ? (addr - (kMaxShadowBytes - 1u)) : 0u) >> 11;
    const u64 p_hi = (end - 1u) >> 11;
    if ((p_hi - p_lo) < 16u) {
        for (u64 p = p_lo; p <= p_hi; ++p) {
            const size_t set_base = ShadowPageSet(p) * kShadowWays;
            for (size_t w = 0; w < kShadowWays; ++w) {
                auto& slot = shadow_meta[set_base + w];
                if (slot.byte_size != 0u) {
                    const VAddr s_end = slot.dst_va + slot.byte_size;
                    if (addr < s_end && slot.dst_va < end) {
                        slot.byte_size = 0u;
                        slot.dst_va = 0u;
                    }
                }
            }
        }
        return;
    }
    // Large writes (`> 32 KiB`, e.g. 2 MiB dynamic rings / compute UAVs): O(1) 256 KiB coarse
    // epoch bump instead of scanning all 2,048 `shadow_meta` slots!
    const u64 g_lo = addr >> kCoarseShift;
    const u64 g_hi = (end - 1u) >> kCoarseShift;
    if ((g_hi - g_lo) <= 512u) {
        const u32 stamp = ++coarse_clock;
        for (u64 g = g_lo; g <= g_hi; ++g) {
            coarse_epoch_256k[g & (kCoarseBuckets - 1u)] = stamp;
        }
    } else {
        coarse_wipe_bias += 1u;
    }
}

const u8* LightTileBatcher::LookupCpuShadow(VAddr addr, u32 size,
                                            u32* out_entry_remaining) noexcept {
    if (!addr || size == 0u || size > kMaxShadowBytes || shadow_max_va == 0u ||
        addr >= shadow_max_va || (addr + size) <= shadow_min_va) {
        return nullptr;
    }
    const u64 page = addr >> 11;
    // Fast exact-base check in `page` set first:
    {
        const size_t set_base = ShadowPageSet(page) * kShadowWays;
        for (size_t w = 0; w < kShadowWays; ++w) {
            auto& slot = shadow_meta[set_base + w];
            if (slot.byte_size != 0u && slot.dst_va == addr && slot.byte_size >= size) {
                if (slot.coarse_e0 != CoarseEpochAt(slot.dst_va) ||
                    slot.coarse_e1 != CoarseEpochAt(slot.dst_va + slot.byte_size - 1u)) {
                    slot.byte_size = 0u;
                    slot.dst_va = 0u;
                    continue;
                }
                slot.last_use = ++shadow_clock;
                if (out_entry_remaining) {
                    *out_entry_remaining = slot.byte_size;
                }
                return &shadow_bytes[(set_base + w) * kMaxShadowBytes];
            }
        }
    }
    // Interior sub-range check: since `slot.byte_size <= 2048` and `size >= 1`, `slot.dst_va`
    // can only reside on `page` or `page - 1` (`<= 32` slots total, O(1)).
    for (u32 step = 0; step < 2u; ++step) {
        if (step == 1u && page == 0u) {
            break;
        }
        const size_t set_base = ShadowPageSet(page - step) * kShadowWays;
        for (size_t w = 0; w < kShadowWays; ++w) {
            auto& slot = shadow_meta[set_base + w];
            if (slot.byte_size != 0u && addr >= slot.dst_va) {
                const u64 sub_off = addr - slot.dst_va;
                if (sub_off < slot.byte_size && (slot.byte_size - sub_off) >= size) {
                    if (slot.coarse_e0 != CoarseEpochAt(slot.dst_va) ||
                        slot.coarse_e1 != CoarseEpochAt(slot.dst_va + slot.byte_size - 1u)) {
                        slot.byte_size = 0u;
                        slot.dst_va = 0u;
                        continue;
                    }
                    slot.last_use = ++shadow_clock;
                    if (out_entry_remaining) {
                        *out_entry_remaining = static_cast<u32>(slot.byte_size - sub_off);
                    }
                    return &shadow_bytes[(set_base + w) * kMaxShadowBytes + sub_off];
                }
            }
        }
    }
    return nullptr;
}

bool LightTileBatcher::TryBatchDraw(Rasterizer& rasterizer, const GraphicsPipeline* pipeline,
                                    const PreparedDraw* /*used_prepared*/, bool is_indexed,
                                    u32 index_offset) {
    if (!Enabled() || flushing || !pipeline || is_indexed || index_offset != 0u ||
        !rasterizer.OnStageA() || !rasterizer.constant_ring ||
        BbToggle::Disabled(BbToggle::ConstantRing) ||
        !Shader::Backend::SPIRV::NativeShadersEnabled() ||
        Shader::Backend::SPIRV::NativeShadersCustomRange()) {
        return false;
    }
    static const bool guest_in_place = VideoCore::GuestInPlace();
    static const bool pipe_verify = [] {
        const char* env = std::getenv("BB_PIPE_VERIFY");
        return env && std::strtoul(env, nullptr, 10) != 0u;
    }();
    static const bool draw_desc_on = [] {
        const char* env = std::getenv("BB_DRAW_DESC");
        return !(env && env[0] == '0');
    }();
    if (guest_in_place || pipe_verify || !draw_desc_on || FrameCapture::Active() ||
        BbCeStats::Enabled() || rasterizer.liverpool->pipe_dirty.reset) {
        return false;
    }

    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    if (vs.pgm_hash != kLightAccVsHash || vs.uses_dma) {
        return false;
    }
    const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    if (!ps || !IsBatchedLightAccPsHash(ps->pgm_hash) || ps->uses_dma) {
        return false;
    }
    for (u32 st = 0; st < Shader::MaxStageTypes; ++st) {
        const auto* s = pipeline->GetStages()[st];
        if (s && st != u32(Shader::SwStage::Fragment) && st != u32(Shader::SwStage::Vertex)) {
            return false;
        }
    }

    ++stat_seen;
    auto fail = [&](u64& bucket) noexcept {
        ++bucket;
        ++stat_fallbacks;
        if (!summary_logged && stat_seen >= 2000u) {
            summary_logged = true;
            std::printf(
                "[LightTileBatcher] Summary(seen=%llu): batched=%llu empty=%llu emitted=%llu "
                "task_flushes=%llu fallbacks=%llu (regs=%llu layout=%llu cb0=%llu s1=%llu s2=%llu "
                "vb=%llu quad=%llu)\n",
                static_cast<unsigned long long>(stat_seen),
                static_cast<unsigned long long>(stat_tiles_batched),
                static_cast<unsigned long long>(stat_tiles_empty),
                static_cast<unsigned long long>(stat_batches_emitted),
                static_cast<unsigned long long>(stat_task_flushes),
                static_cast<unsigned long long>(stat_fallbacks),
                static_cast<unsigned long long>(fb_regs),
                static_cast<unsigned long long>(fb_layout),
                static_cast<unsigned long long>(fb_cb0),
                static_cast<unsigned long long>(fb_ssbo1),
                static_cast<unsigned long long>(fb_ssbo2),
                static_cast<unsigned long long>(fb_vb),
                static_cast<unsigned long long>(fb_quad));
            std::fflush(stdout);
        }
        return false;
    };

    const auto& regs = rasterizer.liverpool->regs;
    if (regs.num_indices != 4u || regs.num_instances.NumInstances() != 1u ||
        regs.primitive_type != Gcn::PrimitiveType::TriangleStrip ||
        regs.stage_enable.raw != Gcn::ShaderStageEnable::VgtStages::Vs ||
        regs.IsClipDisabled() || regs.polygon_control.NeedsBias() ||
        regs.depth_control.depth_write_enable || regs.WritesStencil() ||
        regs.clipper_control.user_clip_plane_enable != 0u ||
        regs.vs_output_control.clip_distance_enable != 0u ||
        regs.vs_output_control.cull_distance_enable != 0u) {
        return fail(fb_regs);
    }

    const auto& fetch = pipeline->GetFetchShader();
    if (!fetch || fetch->attributes.size() != 2u) {
        return fail(fb_layout);
    }
    // Ensure >= 2 MiB headroom in ConstantRing so `Flush()` never fails mid-batch.
    if (!rasterizer.constant_ring->HasFreeSpace(2ULL << 20, 256u)) {
        if (HasPendingTiles()) {
            Flush(rasterizer);
        }
        if (!rasterizer.constant_ring->HasFreeSpace(2ULL << 20, 256u)) {
            return fail(fb_layout);
        }
    }

    // Locate buffer descriptors: exactly 1 non-special in VS (`cbLightAccVtxParam`, ignoring
    // unused `BufferType::ClipPlanes` when `user_clip_plane_enable == 0`) and 2 non-special in PS
    // (`ssbo_1`, `ssbo_2`).
    u32 vs_buf_idx = ~0u;
    u32 vs_non_special = 0u;
    for (u32 i = 0; i < vs.buffers.size(); ++i) {
        if (vs.buffers[i].buffer_type == Shader::BufferType::ClipPlanes) {
            continue;
        }
        if (vs.buffers[i].IsSpecial() || vs.buffers[i].is_written || vs_non_special >= 1u) {
            return fail(fb_layout);
        }
        vs_buf_idx = i;
        ++vs_non_special;
    }
    u32 ps_buf_idx[2]{~0u, ~0u};
    u32 ps_non_special = 0u;
    for (u32 i = 0; i < ps->buffers.size(); ++i) {
        if (ps->buffers[i].IsSpecial() || ps->buffers[i].is_written || ps_non_special >= 2u) {
            return fail(fb_layout);
        }
        ps_buf_idx[ps_non_special++] = i;
    }
    if (vs_non_special != 1u || ps_non_special != 2u) {
        return fail(fb_layout);
    }

    const auto vs_cb0_sharp = vs.buffers[vs_buf_idx].GetSharp(vs);
    const auto ps_b0_sharp = ps->buffers[ps_buf_idx[0]].GetSharp(*ps);
    const auto ps_b1_sharp = ps->buffers[ps_buf_idx[1]].GetSharp(*ps);
    if (!vs_cb0_sharp.base_address || !ps_b0_sharp.base_address || !ps_b1_sharp.base_address) {
        return fail(fb_layout);
    }

    auto resolve_cb_bytes = [&](const Gcn::Buffer& sharp, u32 min_bytes,
                                u32& out_size) -> const u8* {
        const VAddr addr = sharp.base_address;
        u64 raw_sz = sharp.GetSize();
        if (VideoCore::DirectBuffers::IsNativeOwnedAddress(addr)) {
            raw_sz = VideoCore::DirectBuffers::ClampNativeGpuReadSize(addr, raw_sz);
        } else if (VideoCore::DirectBuffers::IsTransientCueHeap(addr, 16u)) {
            raw_sz = VideoCore::DirectBuffers::ClampTransientCueReadSize(addr, raw_sz);
        }
        u32 shadow_rem = 0u;
        if (const u8* sh = LookupCpuShadow(addr, min_bytes, &shadow_rem)) {
            const u64 bounded = (raw_sz >= min_bytes && raw_sz <= kMaxShadowBytes)
                                    ? std::min<u64>(raw_sz, shadow_rem)
                                    : u64(shadow_rem);
            if (bounded >= min_bytes && bounded <= kMaxShadowBytes) {
                out_size = static_cast<u32>(bounded);
                return sh;
            }
        }
        if (raw_sz < min_bytes || raw_sz > kMaxShadowBytes) {
            return nullptr;
        }
        if (VideoCore::DirectBuffers::IsTransientCueHeap(addr, raw_sz) ||
            VideoCore::DirectBuffers::IsTransientAuxCpuRing(addr, raw_sz)) {
            out_size = static_cast<u32>(raw_sz);
            return reinterpret_cast<const u8*>(addr);
        }
        // Native-owned `DEFAULT` ConstantBuffers have stale guest RAM unless updated via
        // `CopyBufferBatch` (which `LookupCpuShadow` catches above).
        if (VideoCore::DirectBuffers::IsNativeOwnedAddress(addr)) {
            return nullptr;
        }
        uintptr_t mapped_end = 0;
        if (!BbLookupMappedEnd(addr, mapped_end) || raw_sz > (mapped_end - addr) ||
            rasterizer.PendingWriteOverlaps(addr, raw_sz) ||
            rasterizer.buffer_cache.IsRegionGpuModified(addr, raw_sz)) {
            return nullptr;
        }
        out_size = static_cast<u32>(raw_sz);
        return reinterpret_cast<const u8*>(addr);
    };

    u32 vs_cb0_bytes = 0u;
    u32 ssbo1_bytes = 0u;
    u32 ssbo2_bytes = 0u;
    const u8* vs_cb0_ptr = resolve_cb_bytes(vs_cb0_sharp, 4u, vs_cb0_bytes);
    if (!vs_cb0_ptr) {
        return fail(fb_cb0);
    }
    const u8* ssbo1_ptr = resolve_cb_bytes(ps_b0_sharp, 64u, ssbo1_bytes);
    if (!ssbo1_ptr) {
        return fail(fb_ssbo1);
    }
    const u8* ssbo2_ptr = resolve_cb_bytes(ps_b1_sharp, 16u, ssbo2_bytes);
    if (!ssbo2_ptr) {
        return fail(fb_ssbo2);
    }

    float tile_vtx_z = 0.0f;
    std::memcpy(&tile_vtx_z, vs_cb0_ptr, sizeof(float));
    if (!std::isfinite(tile_vtx_z)) {
        return fail(fb_cb0);
    }

    // Read and validate the 4 input fullscreen quad vertices `Q[0..3]`.
    const auto vs_ud = vs.UserData();
    Gcn::Buffer orig_vsharps[2]{
        fetch->attributes[0].GetSharpDirect(vs_ud),
        fetch->attributes[1].GetSharpDirect(vs_ud),
    };
    const u32 sem0 = fetch->attributes[0].semantic;
    const u32 sem1 = fetch->attributes[1].semantic;
    if (!((sem0 == 0u && sem1 == 1u) || (sem0 == 1u && sem1 == 0u))) {
        return fail(fb_vb);
    }
    const u32 pos_slot = (sem0 == 0u) ? 0u : 1u;
    const u32 uv_slot = 1u - pos_slot;
    const auto& pos_vb = orig_vsharps[pos_slot];
    const auto& uv_vb = orig_vsharps[uv_slot];
    if (pos_vb.GetDataFmt() != Gcn::DataFormat::Format32_32_32_32 ||
        pos_vb.GetNumberFmt() != Gcn::NumberFormat::Float ||
        uv_vb.GetDataFmt() != Gcn::DataFormat::Format32_32 ||
        uv_vb.GetNumberFmt() != Gcn::NumberFormat::Float) {
        return fail(fb_vb);
    }
    const u32 pos_stride = pos_vb.GetStride();
    const u32 uv_stride = uv_vb.GetStride();
    if (pos_stride < 16u || pos_stride > 256u || uv_stride < 8u || uv_stride > 256u) {
        return fail(fb_vb);
    }
    const u32 pos_span = 3u * pos_stride + 16u;
    const u32 uv_span = 3u * uv_stride + 8u;
    auto resolve_vb_bytes = [&](VAddr addr, u32 span) -> const u8* {
        if (!addr || span > kMaxShadowBytes) {
            return nullptr;
        }
        if (const u8* sh = LookupCpuShadow(addr, span)) {
            return sh;
        }
        if (VideoCore::DirectBuffers::IsTransientCueHeap(addr, span) ||
            VideoCore::DirectBuffers::IsTransientAuxCpuRing(addr, span)) {
            return reinterpret_cast<const u8*>(addr);
        }
        uintptr_t mapped_end = 0;
        if (!BbLookupMappedEnd(addr, mapped_end) || span > (mapped_end - addr)) {
            return nullptr;
        }
        // Immutable native-owned quad VBs (`pShaderRepo + 0x9a0`, `usageType == 1`) store their
        // initial CPU data directly in guest RAM at `addr`, while neighbor `DEFAULT` CBs on the
        // same 4 KiB page mark `PendingWriteOverlaps` / `IsRegionGpuModified` at 4 KiB granularity.
        if (VideoCore::DirectBuffers::IsNativeOwnedAddress(addr) &&
            VideoCore::DirectBuffers::ClampNativeGpuReadSize(addr, span) >= span) {
            return reinterpret_cast<const u8*>(addr);
        }
        if (rasterizer.PendingWriteOverlaps(addr, span) ||
            rasterizer.buffer_cache.IsRegionGpuModified(addr, span)) {
            return nullptr;
        }
        return reinterpret_cast<const u8*>(addr);
    };
    const u8* pos_bytes = resolve_vb_bytes(pos_vb.base_address, pos_span);
    const u8* uv_bytes = resolve_vb_bytes(uv_vb.base_address, uv_span);
    if (!pos_bytes || !uv_bytes) {
        return fail(fb_vb);
    }

    float q_pos[4][4]{};
    float q_uv[4][2]{};
    float qx_min = 1e30f, qx_max = -1e30f;
    float qy_min = 1e30f, qy_max = -1e30f;
    for (u32 k = 0; k < 4u; ++k) {
        std::memcpy(q_pos[k], pos_bytes + k * pos_stride, 16u);
        std::memcpy(q_uv[k], uv_bytes + k * uv_stride, 8u);
        if (!std::isfinite(q_pos[k][0]) || !std::isfinite(q_pos[k][1]) ||
            !std::isfinite(q_pos[k][2]) || !std::isfinite(q_pos[k][3]) ||
            !std::isfinite(q_uv[k][0]) || !std::isfinite(q_uv[k][1]) ||
            std::abs(q_pos[k][3] - 1.0f) > 1e-4f) {
            return fail(fb_quad);
        }
        qx_min = std::min(qx_min, q_pos[k][0]);
        qx_max = std::max(qx_max, q_pos[k][0]);
        qy_min = std::min(qy_min, q_pos[k][1]);
        qy_max = std::max(qy_max, q_pos[k][1]);
    }
    if ((qx_max - qx_min) <= 1e-3f || (qy_max - qy_min) <= 1e-3f) {
        return fail(fb_quad);
    }

    u32 corner_mask = 0u;
    u32 corner_vtx[4]{0, 0, 0, 0};
    bool vtx_x_hi[4]{};
    bool vtx_y_hi[4]{};
    for (u32 k = 0; k < 4u; ++k) {
        const bool x_lo = std::abs(q_pos[k][0] - qx_min) <= 1e-4f;
        const bool x_hi = std::abs(q_pos[k][0] - qx_max) <= 1e-4f;
        const bool y_lo = std::abs(q_pos[k][1] - qy_min) <= 1e-4f;
        const bool y_hi = std::abs(q_pos[k][1] - qy_max) <= 1e-4f;
        if ((!x_lo && !x_hi) || (!y_lo && !y_hi)) {
            return fail(fb_quad);
        }
        vtx_x_hi[k] = x_hi;
        vtx_y_hi[k] = y_hi;
        const u32 c = (x_hi ? 1u : 0u) | (y_hi ? 2u : 0u);
        corner_mask |= (1u << c);
        corner_vtx[c] = k;
    }
    if (corner_mask != 0xFu) {
        return fail(fb_quad);
    }
    const float u_at_xlo = q_uv[corner_vtx[0]][0];
    const float u_at_xhi = q_uv[corner_vtx[1]][0];
    const float v_at_ylo = q_uv[corner_vtx[0]][1];
    const float v_at_yhi = q_uv[corner_vtx[2]][1];
    if (std::abs(q_uv[corner_vtx[2]][0] - u_at_xlo) > 1e-4f ||
        std::abs(q_uv[corner_vtx[3]][0] - u_at_xhi) > 1e-4f ||
        std::abs(q_uv[corner_vtx[1]][1] - v_at_ylo) > 1e-4f ||
        std::abs(q_uv[corner_vtx[3]][1] - v_at_yhi) > 1e-4f) {
        return fail(fb_quad);
    }

    const auto& vp = regs.viewports[0];
    const auto& ctl = regs.viewport_control;
    const float xoffset = ctl.xoffset_enable ? vp.xoffset : 0.0f;
    const float xscale = ctl.xscale_enable ? vp.xscale : 1.0f;
    const float yoffset = ctl.yoffset_enable ? vp.yoffset : 0.0f;
    const float yscale = ctl.yscale_enable ? vp.yscale : 1.0f;
    if (!std::isfinite(xscale) || !std::isfinite(yscale) ||
        std::abs(xscale) <= 1e-3f || std::abs(yscale) <= 1e-3f) {
        ++stat_fallbacks;
        return false;
    }

    // Check pass-level state compatibility against any open light-tile batch pass:
    StageBChunks cur_base_chunks{};
    CaptureStageBChunks(regs, cur_base_chunks);
    StageBChunks cur_norm_chunks = cur_base_chunks;
    NormalizeStageBChunksForPass(cur_norm_chunks);
    const bool cur_commutative = IsCommutativeLightBlend(pipeline);

    if (pass_active) {
        const bool same_pass =
            cur_commutative == pass_commutative_blend &&
            rasterizer.liverpool->last_db_uid == pass_db_uid &&
            rasterizer.liverpool->last_db_extent.raw == pass_db_extent.raw &&
            std::memcmp(rasterizer.liverpool->last_cb_uid.data(), pass_cb_uid.data(),
                        sizeof(pass_cb_uid)) == 0 &&
            std::memcmp(rasterizer.liverpool->last_cb_extent.data(), pass_cb_extent.data(),
                        sizeof(pass_cb_extent)) == 0 &&
            std::memcmp(cur_norm_chunks.data(), pass_norm_chunks.data(),
                        sizeof(StageBChunks)) == 0;
        if (!same_pass) {
            Flush(rasterizer);
        }
    }

    // Compute exact effective tile scissor `[sx0, sx1) x [sy0, sy1)` (matches `raster_state.cpp`).
    Gcn::Scissor eff_sc = CombinedScissor(regs);
    const bool use_vp_sc = regs.mode_control.vport_scissor_enable != 0u;
    if (use_vp_sc) {
        const auto& own = regs.viewport_scissors[0];
        eff_sc.top_left_x = std::max(eff_sc.top_left_x, s16(own.top_left_x));
        eff_sc.top_left_y = std::max(eff_sc.top_left_y, s16(own.top_left_y));
        eff_sc.bottom_right_x =
            std::min(Gcn::Scissor::Clamp(eff_sc.bottom_right_x), own.bottom_right_x);
        eff_sc.bottom_right_y =
            std::min(Gcn::Scissor::Clamp(eff_sc.bottom_right_y), own.bottom_right_y);
    }
    const s32 sx0 = eff_sc.top_left_x;
    const s32 sy0 = eff_sc.top_left_y;
    const s32 sx1 = sx0 + static_cast<s32>(eff_sc.GetWidth());
    const s32 sy1 = sy0 + static_cast<s32>(eff_sc.GetHeight());

    const float z_bounds_min = regs.depth_control.depth_bounds_enable ? regs.depth_bounds_min : -2.0f;
    const float z_bounds_max = regs.depth_control.depth_bounds_enable ? regs.depth_bounds_max : 2.0f;

    const float wx0 = qx_min * xscale + xoffset;
    const float wx1 = qx_max * xscale + xoffset;
    const float wy0 = qy_min * yscale + yoffset;
    const float wy1 = qy_max * yscale + yoffset;
    const float w_left = std::min(wx0, wx1);
    const float w_right = std::max(wx0, wx1);
    const float w_top = std::min(wy0, wy1);
    const float w_bot = std::max(wy0, wy1);

    float sc_x0 = static_cast<float>(sx0);
    float sc_x1 = static_cast<float>(sx1);
    float sc_y0 = static_cast<float>(sy0);
    float sc_y1 = static_cast<float>(sy1);
    if (rasterizer.scene_targets && rasterizer.scene_targets->Reduced()) {
        const auto sz = rasterizer.scene_targets->Size();
        const auto bs = rasterizer.scene_targets->BaseSize();
        if (sz.width != 0u && sz.height != 0u && bs.width != 0u && bs.height != 0u) {
            const float sx_scale = static_cast<float>(sz.width) / static_cast<float>(bs.width);
            const float sy_scale = static_cast<float>(sz.height) / static_cast<float>(bs.height);
            sc_x0 = static_cast<float>(std::lround(sc_x0 * sx_scale)) / sx_scale;
            sc_x1 = static_cast<float>(std::lround(sc_x1 * sx_scale)) / sx_scale;
            sc_y0 = static_cast<float>(std::lround(sc_y0 * sy_scale)) / sy_scale;
            sc_y1 = static_cast<float>(std::lround(sc_y1 * sy_scale)) / sy_scale;
        }
    }

    const float cx0 = std::max(sc_x0, w_left);
    const float cx1 = std::min(sc_x1, w_right);
    const float cy0 = std::max(sc_y0, w_top);
    const float cy1 = std::min(sc_y1, w_bot);

    if (sx1 <= sx0 || sy1 <= sy0 || cx1 <= cx0 || cy1 <= cy0 || z_bounds_min > z_bounds_max) {
        // Zero pixels covered: absorb this draw completely!
        BbStats::draws.fetch_sub(1, std::memory_order_relaxed);
        ++stat_tiles_empty;
        return true;
    }

    // Extract PS `T#` and `S#` descriptors and hash `(pipeline, ssbo_1, T#[], S#[])`.
    std::array<Gcn::Image, 32> cand_images{};
    std::array<Gcn::Sampler, 16> cand_samplers{};
    if (ps->images.size() > cand_images.size() || ps->samplers.size() > cand_samplers.size()) {
        ++stat_fallbacks;
        return false;
    }
    const u32 num_ps_img = static_cast<u32>(ps->images.size());
    const u32 num_ps_smp = static_cast<u32>(ps->samplers.size());
    for (u32 i = 0; i < num_ps_img; ++i) {
        if (ps->images[i].is_written) {
            ++stat_fallbacks;
            return false;
        }
        cand_images[i] = ps->images[i].GetSharp(*ps);
    }
    for (u32 i = 0; i < num_ps_smp; ++i) {
        cand_samplers[i] = ps->samplers[i].GetSharp(*ps);
    }

    u64 res_hash = reinterpret_cast<uintptr_t>(pipeline) ^ (u64(ssbo1_bytes) * 0x9E3779B97F4A7C15ULL);
    res_hash = XXH3_64bits_withSeed(ssbo1_ptr, ssbo1_bytes, res_hash);
    if (num_ps_img != 0u) {
        res_hash = XXH3_64bits_withSeed(cand_images.data(), num_ps_img * sizeof(Gcn::Image),
                                        res_hash);
    }
    if (num_ps_smp != 0u) {
        res_hash = XXH3_64bits_withSeed(cand_samplers.data(), num_ps_smp * sizeof(Gcn::Sampler),
                                        res_hash);
    }

    const u32 aligned_ssbo2_bytes = Common::AlignUp(ssbo2_bytes, 16u);
    const u32 add_ssbo2_dwords = 4u + (aligned_ssbo2_bytes >> 2u);

    auto group_matches = [&](const BatchGroup& g) noexcept {
        if (g.pipeline != pipeline || g.resource_hash != res_hash ||
            g.ssbo1_bytes != ssbo1_bytes || g.ps_images.size() != num_ps_img ||
            g.ps_samplers.size() != num_ps_smp || g.has_vport_scissor != use_vp_sc) {
            return false;
        }
        if (use_vp_sc &&
            g.union_vport_scissor0.window_offset_disable !=
                regs.viewport_scissors[0].window_offset_disable) {
            return false;
        }
        if (g.ssbo2_dwords.size() + add_ssbo2_dwords > kMaxGroupSsbo2Dwords) {
            return false;
        }
        if (std::memcmp(g.ssbo1.data(), ssbo1_ptr, ssbo1_bytes) != 0) {
            return false;
        }
        if (num_ps_img != 0u &&
            std::memcmp(g.ps_images.data(), cand_images.data(),
                        num_ps_img * sizeof(Gcn::Image)) != 0) {
            return false;
        }
        if (num_ps_smp != 0u &&
            std::memcmp(g.ps_samplers.data(), cand_samplers.data(),
                        num_ps_smp * sizeof(Gcn::Sampler)) != 0) {
            return false;
        }
        return true;
    };

    BatchGroup* target_group = nullptr;
    if (pass_active && active_groups != 0u) {
        if (cur_commutative) {
            for (u32 i = 0; i < active_groups; ++i) {
                if (group_matches(groups[i])) {
                    target_group = &groups[i];
                    break;
                }
            }
        } else {
            if (group_matches(groups[active_groups - 1u])) {
                target_group = &groups[active_groups - 1u];
            } else {
                Flush(rasterizer);
            }
        }
    }

    if (!target_group && active_groups >= kMaxGroupsPerPass) {
        Flush(rasterizer);
    }

    if (!pass_active) {
        pass_active = true;
        pass_commutative_blend = cur_commutative;
        pass_norm_chunks = cur_norm_chunks;
        pass_cb_extent = rasterizer.liverpool->last_cb_extent;
        pass_db_extent = rasterizer.liverpool->last_db_extent;
        pass_cb_uid = rasterizer.liverpool->last_cb_uid;
        pass_db_uid = rasterizer.liverpool->last_db_uid;
    }

    if (!target_group) {
        target_group = &groups[active_groups++];
        target_group->Reset();
        target_group->pipeline = pipeline;
        target_group->resource_hash = res_hash;
        target_group->ssbo1_bytes = ssbo1_bytes;
        std::memcpy(target_group->ssbo1.data(), ssbo1_ptr, ssbo1_bytes);
        target_group->ps_images.assign(cand_images.begin(), cand_images.begin() + num_ps_img);
        target_group->ps_samplers.assign(cand_samplers.begin(), cand_samplers.begin() + num_ps_smp);
        target_group->orig_vsharps = {orig_vsharps[0], orig_vsharps[1]};
        target_group->attr_semantics = {sem0, sem1};
        target_group->base_chunks = cur_base_chunks;
        target_group->has_vport_scissor = use_vp_sc;
        target_group->union_screen_scissor = regs.screen_scissor;
        if (use_vp_sc) {
            target_group->union_vport_scissor0 = regs.viewport_scissors[0];
        }

        u32 slot = 0u;
        for (const auto* st : pipeline->GetStages()) {
            if (!st) {
                continue;
            }
            auto& snap = target_group->stages[slot++];
            snap.pgm_base = st->pgm_base;
            snap.user_data.assign(st->user_data.begin(), st->user_data.end());
            snap.flattened_ud_buf.assign(st->flattened_ud_buf.begin(), st->flattened_ud_buf.end());
            if (st == &vs) {
                snap.non_special_buf_indices[0] = vs_buf_idx;
                snap.num_non_special_bufs = 1u;
            } else {
                snap.non_special_buf_indices[0] = ps_buf_idx[0];
                snap.non_special_buf_indices[1] = ps_buf_idx[1];
                snap.num_non_special_bufs = 2u;
            }
        }
        target_group->num_stages = slot;
    } else {
        target_group->union_screen_scissor.top_left_x =
            std::min(target_group->union_screen_scissor.top_left_x, regs.screen_scissor.top_left_x);
        target_group->union_screen_scissor.top_left_y =
            std::min(target_group->union_screen_scissor.top_left_y, regs.screen_scissor.top_left_y);
        target_group->union_screen_scissor.bottom_right_x = std::max(
            target_group->union_screen_scissor.bottom_right_x, regs.screen_scissor.bottom_right_x);
        target_group->union_screen_scissor.bottom_right_y = std::max(
            target_group->union_screen_scissor.bottom_right_y, regs.screen_scissor.bottom_right_y);
        if (use_vp_sc) {
            const auto& own = regs.viewport_scissors[0];
            target_group->union_vport_scissor0.top_left_x =
                std::min<u16>(target_group->union_vport_scissor0.top_left_x, own.top_left_x);
            target_group->union_vport_scissor0.top_left_y =
                std::min<u16>(target_group->union_vport_scissor0.top_left_y, own.top_left_y);
            target_group->union_vport_scissor0.bottom_right_x = std::max<u16>(
                target_group->union_vport_scissor0.bottom_right_x, own.bottom_right_x);
            target_group->union_vport_scissor0.bottom_right_y = std::max<u16>(
                target_group->union_vport_scissor0.bottom_right_y, own.bottom_right_y);
        }
    }

    // Append `[z_bounds_min, z_bounds_max, 0, 0]` + `AlignUp(ssbo2_bytes, 16)` to `ssbo2_dwords`.
    const u32 tile_v4_idx = static_cast<u32>(target_group->ssbo2_dwords.size() >> 2u);
    const size_t base_dw = target_group->ssbo2_dwords.size();
    target_group->ssbo2_dwords.resize(base_dw + add_ssbo2_dwords, 0u);
    target_group->ssbo2_dwords[base_dw + 0u] = std::bit_cast<u32>(z_bounds_min);
    target_group->ssbo2_dwords[base_dw + 1u] = std::bit_cast<u32>(z_bounds_max);
    target_group->ssbo2_dwords[base_dw + 2u] = 0u;
    target_group->ssbo2_dwords[base_dw + 3u] = 0u;
    std::memcpy(&target_group->ssbo2_dwords[base_dw + 4u], ssbo2_ptr, ssbo2_bytes);

    // Compute exact NDC `(tx_lo, tx_hi) x (ty_lo, ty_hi)` and bilinear `(u_lo, u_hi) x (v_lo, v_hi)`.
    const float tx_lo = ((xscale >= 0.0f ? cx0 : cx1) - xoffset) / xscale;
    const float tx_hi = ((xscale >= 0.0f ? cx1 : cx0) - xoffset) / xscale;
    const float ty_lo = ((yscale >= 0.0f ? cy0 : cy1) - yoffset) / yscale;
    const float ty_hi = ((yscale >= 0.0f ? cy1 : cy0) - yoffset) / yscale;

    const float inv_qx_span = 1.0f / (qx_max - qx_min);
    const float inv_qy_span = 1.0f / (qy_max - qy_min);
    const float fx_lo = (tx_lo - qx_min) * inv_qx_span;
    const float fx_hi = (tx_hi - qx_min) * inv_qx_span;
    const float fy_lo = (ty_lo - qy_min) * inv_qy_span;
    const float fy_hi = (ty_hi - qy_min) * inv_qy_span;

    const float u_lo = u_at_xlo + (u_at_xhi - u_at_xlo) * fx_lo;
    const float u_hi = u_at_xlo + (u_at_xhi - u_at_xlo) * fx_hi;
    const float v_lo = v_at_ylo + (v_at_yhi - v_at_ylo) * fy_lo;
    const float v_hi = v_at_ylo + (v_at_yhi - v_at_ylo) * fy_hi;

    const float encoded_w = -static_cast<float>(tile_v4_idx + 1u);
    TileVertex t_verts[4]{};
    for (u32 k = 0; k < 4u; ++k) {
        t_verts[k] = TileVertex{
            .x = vtx_x_hi[k] ? tx_hi : tx_lo,
            .y = vtx_y_hi[k] ? ty_hi : ty_lo,
            .z = tile_vtx_z,
            .w = encoded_w,
            .u = vtx_x_hi[k] ? u_hi : u_lo,
            .v = vtx_y_hi[k] ? v_hi : v_lo,
        };
    }

    if (target_group->tile_count == 0u) {
        target_group->vertices.insert(target_group->vertices.end(), std::begin(t_verts),
                                      std::end(t_verts));
    } else {
        // 2 degenerate duplicate vertices `[prev_t3, t0]` stitch consecutive `TriangleStrip` quads
        // with 6*t even vertex offset (preserving exact front/back winding parity) and 0-area seams.
        target_group->vertices.push_back(target_group->last_t3);
        target_group->vertices.push_back(t_verts[0]);
        target_group->vertices.insert(target_group->vertices.end(), std::begin(t_verts),
                                      std::end(t_verts));
    }
    target_group->last_t3 = t_verts[3];
    ++target_group->tile_count;
    ++pending_tiles;
    ++stat_tiles_batched;
    BbStats::draws.fetch_sub(1, std::memory_order_relaxed);
    return true;
}

void LightTileBatcher::Flush(Rasterizer& rasterizer) {
    if (flushing || !pass_active || active_groups == 0u) {
        pass_active = false;
        pending_tiles = 0u;
        active_groups = 0u;
        return;
    }
    flushing = true;

    auto& lp = *rasterizer.liverpool;
    StageBChunks saved_chunks{};
    CaptureStageBChunks(lp.regs, saved_chunks);
    const Gcn::RegDirty saved_dirty = lp.pipe_dirty;
    const auto saved_cb_extent = lp.last_cb_extent;
    const auto saved_db_extent = lp.last_db_extent;
    const auto saved_cb_uid = lp.last_cb_uid;
    const auto saved_db_uid = lp.last_db_uid;

    lp.last_cb_extent = pass_cb_extent;
    lp.last_db_extent = pass_db_extent;
    lp.last_cb_uid = pass_cb_uid;
    lp.last_db_uid = pass_db_uid;

    const u64 cb_align = std::max<u64>(rasterizer.instance.StorageMinAlignment(),
                                       rasterizer.instance.UniformMinAlignment());

    // Shared dummy 16-byte `cb0` (`[0, 1, 1, 1]`) for batched `vs_8747a367` (`EmitLightTileBatchPatch`
    // uses per-vertex `in_attr0.z` / `in_attr0.w`, while the unpatched instructions still load `cb0[0]`).
    static constexpr float kDummyVsCb0[4]{0.0f, 1.0f, 1.0f, 1.0f};
    const auto dummy_vs_off = rasterizer.constant_ring->Allocate(sizeof(kDummyVsCb0), cb_align);

    StageBChunks prev_emitted_chunks{};
    bool any_emitted = false;

    for (u32 gi = 0; gi < active_groups; ++gi) {
        auto& g = groups[gi];
        if (g.vertices.empty() || g.ssbo2_dwords.empty() || !dummy_vs_off) {
            g.Reset();
            continue;
        }
        if (!any_emitted) {
            std::memcpy(rasterizer.constant_ring->Data(*dummy_vs_off), kDummyVsCb0,
                        sizeof(kDummyVsCb0));
            rasterizer.constant_ring->Flush(*dummy_vs_off, sizeof(kDummyVsCb0));
        }

        const u64 vtx_bytes = g.vertices.size() * sizeof(TileVertex);
        const u64 ssbo1_alloc_bytes = Common::AlignUp<u64>(g.ssbo1_bytes, 16u);
        const u64 ssbo2_bytes = g.ssbo2_dwords.size() * sizeof(u32);

        const auto vtx_off = rasterizer.constant_ring->Allocate(vtx_bytes, 64u);
        const auto ssbo1_off = rasterizer.constant_ring->Allocate(ssbo1_alloc_bytes, cb_align);
        const auto ssbo2_off = rasterizer.constant_ring->Allocate(ssbo2_bytes, cb_align);
        if (!vtx_off || !ssbo1_off || !ssbo2_off) {
            g.Reset();
            continue;
        }

        std::memcpy(rasterizer.constant_ring->Data(*vtx_off), g.vertices.data(), vtx_bytes);
        rasterizer.constant_ring->Flush(*vtx_off, vtx_bytes);

        u8* dst_s1 = rasterizer.constant_ring->Data(*ssbo1_off);
        std::memcpy(dst_s1, g.ssbo1.data(), g.ssbo1_bytes);
        if (ssbo1_alloc_bytes > g.ssbo1_bytes) {
            std::memset(dst_s1 + g.ssbo1_bytes, 0, ssbo1_alloc_bytes - g.ssbo1_bytes);
        }
        rasterizer.constant_ring->Flush(*ssbo1_off, ssbo1_alloc_bytes);

        std::memcpy(rasterizer.constant_ring->Data(*ssbo2_off), g.ssbo2_dwords.data(), ssbo2_bytes);
        rasterizer.constant_ring->Flush(*ssbo2_off, ssbo2_bytes);

        // Build `vsharps_override` pointing directly into `ConstantRing` via `kConstRingVertexBase`.
        const u32 num_verts = static_cast<u32>(g.vertices.size());
        std::array<Gcn::Buffer, 2> vsharps_ovr = g.orig_vsharps;
        for (u32 a = 0; a < 2u; ++a) {
            vsharps_ovr[a].base_address =
                kConstRingVertexBase + *vtx_off + (g.attr_semantics[a] == 1u ? 16ULL : 0ULL);
            vsharps_ovr[a].stride = sizeof(TileVertex);
            vsharps_ovr[a].num_records = num_verts;
            vsharps_ovr[a].swizzle_enable = 0u;
        }

        // Build per-stage `RingBinding` and `BatchedLightStageOverride` arrays.
        std::array<std::array<Rasterizer::RingBinding, 2>, Shader::MaxStageTypes> stage_rings{};
        std::array<BatchedLightStageOverride, Shader::MaxStageTypes> stage_ovrs{};
        for (u32 s = 0; s < g.num_stages; ++s) {
            const auto& snap = g.stages[s];
            if (snap.num_non_special_bufs == 1u) {
                stage_rings[s][0] = Rasterizer::RingBinding{
                    .index = snap.non_special_buf_indices[0],
                    .size = sizeof(kDummyVsCb0),
                    .offset = *dummy_vs_off,
                    .address = 0u,
                    .ring = 0u,
                    .always_note = 0u,
                };
            } else if (snap.num_non_special_bufs == 2u) {
                stage_rings[s][0] = Rasterizer::RingBinding{
                    .index = snap.non_special_buf_indices[0],
                    .size = static_cast<u32>(ssbo1_alloc_bytes),
                    .offset = *ssbo1_off,
                    .address = 0u,
                    .ring = 0u,
                    .always_note = 0u,
                };
                stage_rings[s][1] = Rasterizer::RingBinding{
                    .index = snap.non_special_buf_indices[1],
                    .size = static_cast<u32>(ssbo2_bytes),
                    .offset = *ssbo2_off,
                    .address = 0u,
                    .ring = 0u,
                    .always_note = 0u,
                };
            }
            stage_ovrs[s] = BatchedLightStageOverride{
                .pgm_base = snap.pgm_base,
                .user_data = snap.user_data,
                .flattened_ud_buf = snap.flattened_ud_buf,
                .ring_bindings = std::span<const Rasterizer::RingBinding>{
                    stage_rings[s].data(), snap.num_non_special_bufs},
            };
        }

        RestoreStageBChunks(lp.regs, g.base_chunks);
        lp.regs.num_indices = num_verts;
        lp.regs.index_offset = 0u;
        lp.regs.primitive_type = Gcn::PrimitiveType::TriangleStrip;
        lp.regs.depth_control.depth_bounds_enable = 0u;
        lp.regs.screen_scissor = g.union_screen_scissor;
        if (g.has_vport_scissor) {
            lp.regs.viewport_scissors[0] = g.union_vport_scissor0;
        }

        StageBChunks cur_chunks{};
        CaptureStageBChunks(lp.regs, cur_chunks);
        if (!any_emitted) {
            lp.pipe_dirty = saved_dirty;
            MarkStageBDiffDirty(saved_chunks, cur_chunks, lp.pipe_dirty);
        } else {
            lp.pipe_dirty.Clear();
            MarkStageBDiffDirty(prev_emitted_chunks, cur_chunks, lp.pipe_dirty);
        }

        const BatchedLightDrawOverride draw_ovr{
            .stages = std::span<const BatchedLightStageOverride>{stage_ovrs.data(), g.num_stages},
            .vsharps = vsharps_ovr,
        };
        rasterizer.PostDraw(g.pipeline, nullptr, false, 0u, nullptr, nullptr, &draw_ovr);
        BbStats::draws.fetch_add(1, std::memory_order_relaxed);
        ++stat_batches_emitted;
        prev_emitted_chunks = cur_chunks;
        any_emitted = true;
        g.Reset();
    }

    RestoreStageBChunks(lp.regs, saved_chunks);
    lp.last_cb_extent = saved_cb_extent;
    lp.last_db_extent = saved_db_extent;
    lp.last_cb_uid = saved_cb_uid;
    lp.last_db_uid = saved_db_uid;
    if (any_emitted) {
        lp.pipe_dirty.Clear();
        lp.pipe_dirty.blocks |= saved_dirty.blocks;
        MarkStageBDiffDirty(prev_emitted_chunks, saved_chunks, lp.pipe_dirty);
    } else {
        lp.pipe_dirty = saved_dirty;
    }

    if (!first_active_logged && pending_tiles >= 16u && any_emitted) {
        first_active_logged = true;
        std::printf("[LightTileBatcher] Active: first pass coalesced %u local-light tiles into %u "
                    "batched draws (fallbacks=%llu)\n",
                    pending_tiles, active_groups,
                    static_cast<unsigned long long>(stat_fallbacks));
        std::fflush(stdout);
    }

    if (!summary_logged && stat_seen >= 2000u) {
        summary_logged = true;
        std::printf(
            "[LightTileBatcher] Summary(seen=%llu): batched=%llu empty=%llu emitted=%llu "
            "task_flushes=%llu fallbacks=%llu (regs=%llu layout=%llu cb0=%llu s1=%llu s2=%llu "
            "vb=%llu quad=%llu)\n",
            static_cast<unsigned long long>(stat_seen),
            static_cast<unsigned long long>(stat_tiles_batched),
            static_cast<unsigned long long>(stat_tiles_empty),
            static_cast<unsigned long long>(stat_batches_emitted),
            static_cast<unsigned long long>(stat_task_flushes),
            static_cast<unsigned long long>(stat_fallbacks),
            static_cast<unsigned long long>(fb_regs),
            static_cast<unsigned long long>(fb_layout),
            static_cast<unsigned long long>(fb_cb0),
            static_cast<unsigned long long>(fb_ssbo1),
            static_cast<unsigned long long>(fb_ssbo2),
            static_cast<unsigned long long>(fb_vb),
            static_cast<unsigned long long>(fb_quad));
        std::fflush(stdout);
    }

    if (BatchLogEnabled()) {
        static auto last_log = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        if (now - last_log >= std::chrono::seconds(5)) {
            last_log = now;
            std::printf("[LightTileBatcher] 5s: tiles_batched=%llu batches_emitted=%llu "
                        "empty_culled=%llu fallbacks=%llu\n",
                        static_cast<unsigned long long>(stat_tiles_batched),
                        static_cast<unsigned long long>(stat_batches_emitted),
                        static_cast<unsigned long long>(stat_tiles_empty),
                        static_cast<unsigned long long>(stat_fallbacks));
            stat_tiles_batched = 0;
            stat_batches_emitted = 0;
            stat_tiles_empty = 0;
            stat_fallbacks = 0;
        }
    }

    active_groups = 0u;
    pending_tiles = 0u;
    pass_active = false;
    flushing = false;
}

} // namespace Vulkan
