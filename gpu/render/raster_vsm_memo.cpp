// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 9 (Target A): Stage-A Memoization for Local VSM Point/Spot Shadow Maps (`R16G16Sfloat`
// `32x32..512x512` cubemap faces & 2D spot maps) and their `Sample13Weight` 13-tap Gaussian blurs.

#include "render/vk_vsm_shadow_memo.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "buffers/direct_buffers.h"
#include "cp/cp_tasks.h"
#include "render/direct_targets.h"
#include "render/raster_internal.h"
#include "render/vk_light_tile_batch.h"
#include "textures/direct_textures.h"
#include "../../dantelion_gx/include/gx_direct_vulkan.h"

extern "C" bool BbLookupMappedEnd(u64 address, uintptr_t& out_end);

namespace Vulkan {

namespace {

constexpr u64 kSample13BlurVsHash = 0x000000000b0acf50ULL;
constexpr u64 kSample13BlurPsHash = 0x000000000f61fd27ULL;

// 16 GiB GPU address space `[0x1000000000, 0x1400000000)` partitioned into 262,144 exact 64-KiB
// granules (`1.0 MiB` static BSS, 0% hash collisions, lock-free O(1) write epoch tracking).
constexpr VAddr kGpuSpaceBase = 0x1000000000ULL;
constexpr VAddr kGpuSpaceEnd = 0x1400000000ULL;
constexpr u64 kPage64kShift = 16u;
constexpr size_t kNumPage64kSlots =
    static_cast<size_t>((kGpuSpaceEnd - kGpuSpaceBase) >> kPage64kShift); // 262,144

std::array<std::atomic<u32>, kNumPage64kSlots> g_page64k_write_epoch{};
std::atomic<u32> g_fallback_write_epoch{1u};

// Lock-free 256-slot retired-texture ring drained by Stage A.
constexpr size_t kRetireRingSize = 256u;
struct RetireSlot {
    std::atomic<u64> seq{0};
    VAddr addr = 0;
    u64 size = 0;
};
std::array<RetireSlot, kRetireRingSize> g_retire_ring{};
std::atomic<u64> g_retire_write_seq{0};

bool MemoLogEnabled() noexcept {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_VSM_SHADOW_MEMO_LOG");
        return env && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

constexpr bool IsVsmDim(u32 dim) noexcept {
    return dim == 32u || dim == 64u || dim == 128u || dim == 256u || dim == 512u;
}

void CaptureStageBChunks(const Gcn::Regs& regs, VsmShadowMemo::StageBChunks& out) noexcept {
    for (u32 bit = 0; bit < Gcn::RegDirty::StageBNumChunks; ++bit) {
        std::memcpy(&out[bit * Gcn::RegDirty::StageBChunkWords],
                    regs.reg_array.data() + Gcn::RegDirty::StageBChunkWordOffset(bit),
                    Gcn::RegDirty::StageBChunkWords * sizeof(u32));
    }
}

void RestoreStageBChunks(Gcn::Regs& regs, const VsmShadowMemo::StageBChunks& in) noexcept {
    for (u32 bit = 0; bit < Gcn::RegDirty::StageBNumChunks; ++bit) {
        std::memcpy(regs.reg_array.data() + Gcn::RegDirty::StageBChunkWordOffset(bit),
                    &in[bit * Gcn::RegDirty::StageBChunkWords],
                    Gcn::RegDirty::StageBChunkWords * sizeof(u32));
    }
}

void MarkStageBDiffDirty(const VsmShadowMemo::StageBChunks& prev,
                         const VsmShadowMemo::StageBChunks& next,
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

inline u64 MixU64(u64 h, u64 v) noexcept {
    v ^= v >> 33;
    v *= 0xff51afd7ed558ccdULL;
    v ^= v >> 33;
    v *= 0xc4ceb9fe1a85ec53ULL;
    v ^= v >> 33;
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

inline u64 HashConstantBufferBytes(u64 seed, const u8* ptr, u64 byte_size) noexcept {
    // Verified across all 7 VSM shaders (`vs 0x167daba6`, `0x5ecabf03`, `0xbc49f900`,
    // `0x288c510c`, `0x48ecf308` + `ps 0x1ae03d75`, `0x12483d6d`): `cbSceneParam` (`864` B =
    // `216` dwords) is only read at `dword 1` (`[0..32)` near/far/viewport) and `dwords 183, 187,
    // 191, 200..215` (`[720..864)` shadow camera eye + `4x4` `viewProj`). Bytes `[32..720)` hold
    // global main-camera / frame-timer data that changes when rotating the player camera.
    if (byte_size == 864u) {
        u64 h = XXH3_64bits_withSeed(ptr, 32u, seed);
        h = XXH3_64bits_withSeed(ptr + 720u, 144u, h);
        return h;
    }
    return XXH3_64bits_withSeed(ptr, static_cast<size_t>(byte_size), seed);
}

u64 ComputeEmptyFaceDigest(const u32 clear_rgba[4], u32 dim) noexcept {
    u64 h = 0x56534d5f454d5054ULL; // "VSM_EMPT"
    h = MixU64(h, dim);
    h = XXH3_64bits_withSeed(clear_rgba, sizeof(u32) * 4u, h);
    return h;
}

struct ReplayScope {
    bool& flag;
    explicit ReplayScope(bool& f) noexcept : flag{f} { flag = true; }
    ~ReplayScope() noexcept { flag = false; }
};

} // namespace

VsmShadowMemo::VsmShadowMemo() {
    bound_ranges_init:
    pending_face.bound_ranges.reserve(64);
    for (auto& d : pending_face.draws) {
        d.vsharps.reserve(16);
        for (auto& st : d.stages) {
            st.user_data.reserve(16);
            st.flattened_ud_buf.reserve(48);
        }
    }
    (void)&&bound_ranges_init;
}

bool VsmShadowMemo::Enabled() noexcept {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_VSM_SHADOW_MEMO");
        return !env || (env[0] != '\0' && env[0] != '0');
    }();
    return enabled;
}

void VsmShadowMemo::NoteBufferWriteGlobal(VAddr addr, u64 byte_size) noexcept {
    if (!addr || byte_size == 0u) {
        return;
    }
    const VAddr end = (byte_size > (~0ULL - addr)) ? ~0ULL : (addr + byte_size);
    if (addr < kGpuSpaceEnd && end > kGpuSpaceBase) {
        const VAddr clamped_lo = std::max(addr, kGpuSpaceBase);
        const VAddr clamped_hi = std::min(end, kGpuSpaceEnd);
        if (clamped_hi > clamped_lo) {
            const size_t p_lo = static_cast<size_t>((clamped_lo - kGpuSpaceBase) >> kPage64kShift);
            const size_t p_hi =
                static_cast<size_t>((clamped_hi - 1u - kGpuSpaceBase) >> kPage64kShift);
            // Cap per-call sweep to 4096 pages (256 MiB) so pathological bulk unmaps remain O(1).
            if (p_hi >= p_lo && (p_hi - p_lo) <= 4096u) {
                for (size_t p = p_lo; p <= p_hi; ++p) {
                    g_page64k_write_epoch[p].fetch_add(1u, std::memory_order_relaxed);
                }
                return;
            }
        }
    }
    g_fallback_write_epoch.fetch_add(1u, std::memory_order_relaxed);
}

u64 VsmShadowMemo::QueryPage64kWriteDigest(VAddr addr, u64 byte_size) noexcept {
    if (!addr || byte_size == 0u) {
        return 0ULL;
    }
    const VAddr end = (byte_size > (~0ULL - addr)) ? ~0ULL : (addr + byte_size);
    if (addr >= kGpuSpaceBase && end <= kGpuSpaceEnd && end > addr) {
        const size_t p_lo = static_cast<size_t>((addr - kGpuSpaceBase) >> kPage64kShift);
        const size_t p_hi = static_cast<size_t>((end - 1u - kGpuSpaceBase) >> kPage64kShift);
        if ((p_hi - p_lo) <= 1024u) { // up to 64 MiB single VB/IB
            u64 h = MixU64(0x504736344b5f4550ULL, p_hi - p_lo + 1u);
            for (size_t p = p_lo; p <= p_hi; ++p) {
                const u32 ep = g_page64k_write_epoch[p].load(std::memory_order_relaxed);
                h = MixU64(h, (static_cast<u64>(p) << 32) | ep);
            }
            return h;
        }
    }
    return MixU64(0x46414c4c4241434bULL,
                  g_fallback_write_epoch.load(std::memory_order_relaxed));
}

void VsmShadowMemo::NoteRetiredTextureRangeGlobal(VAddr addr, u64 byte_size) noexcept {
    if (!addr || byte_size == 0u) {
        return;
    }
    const u64 ticket = g_retire_write_seq.fetch_add(1u, std::memory_order_relaxed) + 1u;
    auto& slot = g_retire_ring[ticket & (kRetireRingSize - 1u)];
    slot.seq.store(0u, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    slot.addr = addr;
    slot.size = byte_size;
    slot.seq.store(ticket, std::memory_order_release);
}

void VsmShadowMemo::DrainRetiredRanges() noexcept {
    const u64 head = g_retire_write_seq.load(std::memory_order_acquire);
    if (head == last_seen_retire_seq) {
        return;
    }
    if (head - last_seen_retire_seq > kRetireRingSize) {
        // Ring wrapped before Stage A drained it: conservatively invalidate all cached faces/links.
        for (auto& f : faces) {
            f.valid = false;
            f.skip_blur_chain = false;
            f.was_blurred = false;
        }
        for (auto& s : scratch_links) {
            s.valid = false;
        }
        last_seen_retire_seq = head;
        return;
    }
    for (u64 seq = last_seen_retire_seq + 1u; seq <= head; ++seq) {
        const auto& slot = g_retire_ring[seq & (kRetireRingSize - 1u)];
        if (slot.seq.load(std::memory_order_acquire) == seq) {
            InvalidateOverlappingMemory(slot.addr, slot.size);
        } else {
            // Concurrent writer still mid-store: invalidate conservatively so no stale hit occurs.
            for (auto& f : faces) {
                f.valid = false;
                f.skip_blur_chain = false;
            }
            break;
        }
    }
    last_seen_retire_seq = head;
}

void VsmShadowMemo::InvalidateOverlappingMemory(VAddr addr, u64 byte_size) noexcept {
    if (!addr || byte_size == 0u) {
        return;
    }
    const VAddr end = (byte_size > (~0ULL - addr)) ? ~0ULL : (addr + byte_size);
    auto overlaps = [&](VAddr base, u64 span) noexcept {
        if (!base || span == 0u) {
            return false;
        }
        const VAddr b_end = (span > (~0ULL - base)) ? ~0ULL : (base + span);
        return addr < b_end && base < end;
    };
    for (auto& f : faces) {
        if (!f.allocated) {
            continue;
        }
        const u64 span = f.slice_bytes ? f.slice_bytes : (u64(f.dim) * u64(f.dim) * 4ULL);
        const u64 base_span = span * u64(f.slice + 1u);
        if (overlaps(f.slice_va, span) || overlaps(f.base_va, base_span)) {
            f.valid = false;
            f.was_blurred = false;
            f.saw_blur_after_face = false;
            f.skip_blur_chain = false;
        }
    }
    for (auto& s : scratch_links) {
        if (!s.valid) {
            continue;
        }
        const u64 span = u64(s.dim) * u64(s.dim) * 4ULL;
        if (overlaps(s.slice_va, span) || overlaps(s.base_va, span * u64(s.slice + 1u))) {
            s.valid = false;
        }
    }
    for (auto& kb : known_vsm_bases) {
        if (kb >= addr && kb < end) {
            kb = 0;
        }
    }
}

void VsmShadowMemo::MarkKnownVsmBase(VAddr base_va) noexcept {
    if (!base_va) {
        return;
    }
    for (VAddr v : known_vsm_bases) {
        if (v == base_va) {
            break;
        }
    }
    bool found = false;
    for (VAddr v : known_vsm_bases) {
        if (v == base_va) {
            found = true;
            break;
        }
    }
    if (!found) {
        known_vsm_bases[known_vsm_cursor & (known_vsm_bases.size() - 1u)] = base_va;
        ++known_vsm_cursor;
    }
    for (auto& f : faces) {
        if (f.allocated && (f.base_va == base_va || f.slice_va == base_va)) {
            f.is_known_vsm = true;
        }
    }
}

namespace {

bool ParseVsmImageSharp(const u64 raw_words[4], VAddr hint_slice_va, u32 hint_slice_bytes,
                        u32& out_dim, VAddr& out_base_va, u32& out_slice, VAddr& out_slice_va,
                        u32& out_slice_bytes) noexcept {
    Gcn::Image img{};
    std::memcpy(&img, raw_words, sizeof(img));
    reinterpret_cast<u32*>(&img)[7] = 0u; // strip DirectTextures::kSrvTagMagic
    if (!img.Valid() || img.IsMsaa() || img.base_level != 0u) {
        return false;
    }
    if (img.GetDataFmt() != Gcn::DataFormat::Format16_16 ||
        img.GetNumberFmt() != Gcn::NumberFormat::Float) {
        return false;
    }
    const u32 w = static_cast<u32>(img.width) + 1u;
    const u32 h = static_cast<u32>(img.height) + 1u;
    if (w != h || !IsVsmDim(w)) {
        return false;
    }
    const VAddr base_va = img.Address();
    if (!base_va) {
        return false;
    }
    const u32 slice = static_cast<u32>(img.base_array);
    const u32 min_bytes = w * h * 4u;
    out_dim = w;
    out_base_va = base_va;
    out_slice = slice;
    out_slice_bytes = (hint_slice_bytes >= min_bytes) ? hint_slice_bytes : min_bytes;
    if (hint_slice_va != 0u &&
        ((slice == 0u && hint_slice_va == base_va) || (slice > 0u && hint_slice_va > base_va))) {
        out_slice_va = hint_slice_va;
    } else if (slice == 0u) {
        out_slice_va = base_va;
    } else {
        out_slice_va = 0u;
    }
    return true;
}

bool ResolveColorRtvTarget(const Gcn::Regs& regs, Gcn::CbDbExtent cb_extent, u16 cb_uid,
                           u32& out_dim, VAddr& out_base_va, u32& out_slice, VAddr& out_slice_va,
                           u32& out_slice_bytes) noexcept {
    const auto& col_buf = regs.color_buffers[0];
    if (!col_buf || col_buf.NumSamples() != 1u || col_buf.NumSlices() < 1u) {
        return false;
    }
    if (col_buf.GetDataFmt() != Gcn::DataFormat::Format16_16 ||
        col_buf.GetNumberFmt() != Gcn::NumberFormat::Float) {
        return false;
    }
    const u32 w = cb_extent.Valid() ? u32(cb_extent.width) : col_buf.Pitch();
    const u32 h = cb_extent.Valid() ? u32(cb_extent.height) : col_buf.Height();
    if (w != h || !IsVsmDim(w)) {
        return false;
    }
    const u32 slice_bytes = col_buf.GetColorSliceSize();
    const VAddr rtv_addr = col_buf.Address();
    const u32 rtv_slice = col_buf.BaseSlice();
    if (!rtv_addr || slice_bytes == 0u) {
        return false;
    }
    // Single-slice target requirement (`view.slice_max == view.slice_start`).
    if (!col_buf.info.linear_general && col_buf.view.slice_max != col_buf.view.slice_start) {
        return false;
    }

    VAddr base_va = rtv_addr;
    u32 slice = rtv_slice;
    VAddr slice_va = rtv_addr + u64(rtv_slice) * u64(slice_bytes);

    if (cb_uid != 0u && cb_uid < DirectTargets::kMaxTargets) {
        DirectTargets::TargetSnapshot snap{};
        if (DirectTargets::ReadTargetSlot(cb_uid, snap) && !snap.is_depth) {
            if (snap.mip_level != 0u || snap.slice_count != 1u) {
                return false;
            }
            if (snap.has_parent_tsharp) {
                Gcn::Image parent_img{};
                std::memcpy(&parent_img, snap.parent_tsharp.data(), sizeof(parent_img));
                reinterpret_cast<u32*>(&parent_img)[7] = 0u;
                if (parent_img.Valid() && parent_img.Address() != 0u) {
                    const VAddr parent_base = parent_img.Address();
                    if (rtv_slice == 0u && snap.first_slice > 0u && rtv_addr >= parent_base) {
                        base_va = parent_base;
                        slice = snap.first_slice;
                        slice_va = rtv_addr;
                    } else if (rtv_addr == parent_base) {
                        base_va = parent_base;
                        slice = rtv_slice;
                        slice_va = parent_base + u64(slice) * u64(slice_bytes);
                    }
                }
            }
        }
    }

    out_dim = w;
    out_base_va = base_va;
    out_slice = slice;
    out_slice_va = slice_va;
    out_slice_bytes = slice_bytes;
    return true;
}

} // namespace

s32 VsmShadowMemo::FindFaceIndex(const VsmTargetKey& key) const noexcept {
    if (!key.valid) {
        return -1;
    }
    for (size_t i = 0; i < kMaxFaces; ++i) {
        const auto& f = faces[i];
        if (!f.allocated || f.dim != key.dim) {
            continue;
        }
        if (f.base_va == key.base_va && f.slice == key.slice) {
            return static_cast<s32>(i);
        }
        if (f.slice_va != 0u && key.slice_va != 0u && f.slice_va == key.slice_va) {
            return static_cast<s32>(i);
        }
    }
    return -1;
}

void VsmShadowMemo::EnrichFaceKey(FaceEntry& f, const VsmTargetKey& key) noexcept {
    if (key.slice_va != 0u && f.slice_va == 0u) {
        f.slice_va = key.slice_va;
    }
    if (key.slice_bytes > f.slice_bytes) {
        f.slice_bytes = key.slice_bytes;
    }
    // Prefer canonical parent `(base_va, slice)` when `f` was initially allocated from a
    // slice-offset RTV `(slice_va, 0)`.
    if (key.base_va != 0u && key.slice > 0u && f.slice == 0u && f.base_va == f.slice_va) {
        f.base_va = key.base_va;
        f.slice = key.slice;
    }
    if (!f.is_known_vsm && f.base_va != 0u) {
        for (VAddr kb : known_vsm_bases) {
            if (kb != 0u && kb == f.base_va) {
                f.is_known_vsm = true;
                break;
            }
        }
    }
}

u32 VsmShadowMemo::GetOrAllocateFaceIndex(const VsmTargetKey& key, u64 cur_frame) noexcept {
    if (const s32 existing = FindFaceIndex(key); existing >= 0) {
        EnrichFaceKey(faces[static_cast<size_t>(existing)], key);
        faces[static_cast<size_t>(existing)].last_seen_frame = cur_frame;
        return static_cast<u32>(existing);
    }
    size_t victim = 0u;
    bool found_free = false;
    for (size_t i = 0; i < kMaxFaces; ++i) {
        if (!faces[i].allocated) {
            victim = i;
            found_free = true;
            break;
        }
    }
    if (!found_free) {
        for (size_t i = 0; i < kMaxFaces; ++i) {
            if (faces[i].has_deferred_clear ||
                (pending_face.active && pending_face.face_idx == i)) {
                continue;
            }
            if (faces[victim].has_deferred_clear ||
                (pending_face.active && pending_face.face_idx == victim) ||
                faces[i].last_seen_frame < faces[victim].last_seen_frame) {
                victim = i;
            }
        }
    }
    auto& f = faces[victim];
    f = FaceEntry{};
    f.allocated = true;
    f.base_va = key.base_va;
    f.slice = key.slice;
    f.slice_va = key.slice_va;
    f.slice_bytes = key.slice_bytes;
    f.dim = key.dim;
    f.last_seen_frame = cur_frame;
    EnrichFaceKey(f, key);
    return static_cast<u32>(victim);
}

void VsmShadowMemo::SetScratchLink(const VsmTargetKey& scratch_key, u32 face_idx, bool skip,
                                   bool from_blur, u64 cur_frame) noexcept {
    size_t slot_idx = 0u;
    bool matched = false;
    for (size_t i = 0; i < kMaxScratchLinks; ++i) {
        const auto& s = scratch_links[i];
        if (s.valid && s.dim == scratch_key.dim &&
            ((s.base_va == scratch_key.base_va && s.slice == scratch_key.slice) ||
             (s.slice_va != 0u && scratch_key.slice_va != 0u &&
              s.slice_va == scratch_key.slice_va))) {
            slot_idx = i;
            matched = true;
            break;
        }
    }
    if (!matched) {
        for (size_t i = 0; i < kMaxScratchLinks; ++i) {
            if (!scratch_links[i].valid) {
                slot_idx = i;
                matched = true;
                break;
            }
            if (scratch_links[i].last_frame < scratch_links[slot_idx].last_frame) {
                slot_idx = i;
            }
        }
    }
    auto& dst = scratch_links[slot_idx];
    dst.valid = true;
    dst.skip = skip;
    dst.from_blur = from_blur;
    dst.face_idx = face_idx;
    dst.base_va = scratch_key.base_va;
    dst.slice = scratch_key.slice;
    dst.slice_va = scratch_key.slice_va;
    dst.dim = scratch_key.dim;
    dst.last_frame = cur_frame;
}

const VsmShadowMemo::ScratchLink* VsmShadowMemo::FindScratchLink(
    const VsmTargetKey& key) const noexcept {
    if (!key.valid) {
        return nullptr;
    }
    for (const auto& s : scratch_links) {
        if (!s.valid || s.dim != key.dim) {
            continue;
        }
        if ((s.base_va == key.base_va && s.slice == key.slice) ||
            (s.slice_va != 0u && key.slice_va != 0u && s.slice_va == key.slice_va)) {
            return &s;
        }
    }
    return nullptr;
}

void VsmShadowMemo::EmitClearColorNow(Rasterizer& rasterizer,
                                      const Rasterizer::DirectClearColorArgs& args) {
    ReplayScope scope{replaying};
    if (rasterizer.RunInOrder(&BbCp::Tasks::DirectVkClearColorTask, &args, sizeof(args),
                              BbToggle::PipelinedMemoryWrites, false)) {
        if (args.dst_va && args.byte_size) {
            rasterizer.NotePendingGpuWrite(args.dst_va, args.byte_size);
        }
    }
}

void VsmShadowMemo::EmitClearDepthNow(Rasterizer& rasterizer,
                                      const Rasterizer::DirectClearDepthStencilArgs& args) {
    ReplayScope scope{replaying};
    if (rasterizer.RunInOrder(&BbCp::Tasks::DirectVkClearDepthStencilTask, &args, sizeof(args),
                              BbToggle::PipelinedMemoryWrites, false)) {
        if (args.dst_va && args.dst_byte_size) {
            rasterizer.NotePendingGpuWrite(args.dst_va, args.dst_byte_size);
        }
    }
}

void VsmShadowMemo::FlushPendingDepthClear(Rasterizer& rasterizer) {
    if (!has_pending_depth_clear) {
        return;
    }
    const auto args = pending_depth_clear_args;
    has_pending_depth_clear = false;
    EmitClearDepthNow(rasterizer, args);
}

void VsmShadowMemo::FinalizeDeferredClear(Rasterizer& rasterizer, u32 face_idx) {
    if (face_idx >= kMaxFaces) {
        return;
    }
    auto& f = faces[face_idx];
    if (!f.has_deferred_clear) {
        return;
    }
    const auto clear_args = f.deferred_clear_args;
    f.has_deferred_clear = false;
    if (num_deferred_clears > 0u) {
        --num_deferred_clears;
    }

    // Only memoize 0-draw empty clears on targets proven to belong to a VSM shadow map/cubemap.
    if (!f.is_known_vsm) {
        f.valid = false;
        f.skip_blur_chain = false;
        EmitClearColorNow(rasterizer, clear_args);
        return;
    }

    const u64 cur_frame = BbStats::frame_number.load(std::memory_order_relaxed);
    const u64 empty_digest = ComputeEmptyFaceDigest(clear_args.clear_rgba, f.dim);
    const bool blur_consistent = (f.was_blurred == f.saw_blur_after_face);
    const bool aged_out =
        (cur_frame > f.last_render_frame) &&
        ((cur_frame - f.last_render_frame) >= kMaxPeriodicRefreshFrames) &&
        (last_periodic_refresh_frame != cur_frame);

    ++stat_faces_evaluated;
    if (f.valid && f.num_last_draws == 0u && blur_consistent && !aged_out &&
        f.geom_digest == empty_digest) {
        // 0-draw empty frustum face HIT: skip ClearColor (and subsequent Blur/Copy if blurred).
        f.skip_blur_chain = f.was_blurred;
        f.saw_blur_after_face = false;
        ++stat_face_hits;
        ++stat_empty_hits;
        ++stat_clears_saved;
        MaybeLogTelemetry();
        return;
    }

    if (aged_out) {
        last_periodic_refresh_frame = cur_frame;
    }
    EmitClearColorNow(rasterizer, clear_args);
    f.valid = true;
    f.num_last_draws = 0u;
    f.geom_digest = empty_digest;
    f.was_blurred = false;
    f.saw_blur_after_face = false;
    f.skip_blur_chain = false;
    f.last_render_frame = cur_frame;
    ++stat_face_misses;
    MaybeLogTelemetry();
}

void VsmShadowMemo::FlushAllDeferredClears(Rasterizer& rasterizer) {
    if (num_deferred_clears == 0u) {
        return;
    }
    for (u32 i = 0; i < kMaxFaces && num_deferred_clears != 0u; ++i) {
        if (faces[i].allocated && faces[i].has_deferred_clear) {
            FinalizeDeferredClear(rasterizer, i);
        }
    }
    num_deferred_clears = 0u;
}

void VsmShadowMemo::FlushAll(Rasterizer& rasterizer) {
    if (replaying) {
        return;
    }
    passthrough_target = {};
    if (pending_face.active) {
        FinishPendingFace(rasterizer, false);
    }
    if (has_pending_depth_clear) {
        FlushPendingDepthClear(rasterizer);
    }
    if (num_deferred_clears != 0u) {
        FlushAllDeferredClears(rasterizer);
    }
}

bool VsmShadowMemo::TryInterceptClearColor(Rasterizer& rasterizer,
                                           const Rasterizer::DirectClearColorArgs& args) {
    if (!Enabled() || replaying) {
        return false;
    }
    DrainRetiredRanges();

    if (args.num_bits != 32u || !args.dst_va || args.byte_size == 0u) {
        FlushAll(rasterizer);
        InvalidateOverlappingMemory(args.dst_va, args.byte_size);
        return false;
    }

    VsmTargetKey key{};
    if (!ParseVsmImageSharp(args.tsharp, args.dst_va, args.byte_size, key.dim, key.base_va,
                            key.slice, key.slice_va, key.slice_bytes)) {
        FlushAll(rasterizer);
        InvalidateOverlappingMemory(args.dst_va, args.byte_size);
        return false;
    }
    key.valid = true;
    passthrough_target = {};

    if (pending_face.active) {
        FinishPendingFace(rasterizer, false);
    }
    if (has_pending_depth_clear) {
        FlushPendingDepthClear(rasterizer);
    }

    const u64 cur_frame = BbStats::frame_number.load(std::memory_order_relaxed);
    const u32 idx = GetOrAllocateFaceIndex(key, cur_frame);
    auto& f = faces[idx];
    if (f.has_deferred_clear) {
        FinalizeDeferredClear(rasterizer, idx);
    }
    f.deferred_clear_args = args;
    f.has_deferred_clear = true;
    ++num_deferred_clears;
    return true;
}

bool VsmShadowMemo::TryInterceptClearDepth(Rasterizer& rasterizer,
                                           const Rasterizer::DirectClearDepthStencilArgs& args) {
    if (!Enabled() || replaying) {
        return false;
    }
    DrainRetiredRanges();

    Gcn::CbDbExtent ext{};
    ext.raw = args.cb_db_extent;
    const u32 w = ext.Valid() ? u32(ext.width) : 0u;
    const u32 h = ext.Valid() ? u32(ext.height) : 0u;
    if (!args.dst_va || args.dst_byte_size == 0u || args.slice_count != 1u || w != h ||
        !IsVsmDim(w)) {
        FlushAll(rasterizer);
        return false;
    }

    if (pending_face.active) {
        FinishPendingFace(rasterizer, false);
    }
    if (has_pending_depth_clear) {
        FlushPendingDepthClear(rasterizer);
    }
    pending_depth_clear_args = args;
    has_pending_depth_clear = true;
    passthrough_target = {};
    return true;
}

void VsmShadowMemo::NotifyBeforeCopyBatch(Rasterizer& rasterizer, const u32* raw_regions,
                                          u32 count) {
    if (!Enabled() || replaying || !pending_face.active || pending_face.num_draws == 0u ||
        !raw_regions || count == 0u) {
        return;
    }
    for (u32 i = 0; i < count; ++i) {
        Rasterizer::DirectCopyRegion reg{};
        std::memcpy(&reg, raw_regions + i * 6u, sizeof(reg));
        if (!reg.dst_va || reg.byte_size == 0u) {
            continue;
        }
        const VAddr c_end = reg.dst_va + reg.byte_size;
        for (const auto& br : pending_face.bound_ranges) {
            if (reg.dst_va < (br.addr + br.size) && br.addr < c_end) {
                // Same buffer overwritten mid-face: flush buffered draws before this copy runs!
                FinishPendingFace(rasterizer, /*force_miss=*/true);
                return;
            }
        }
    }
}

void VsmShadowMemo::InvalidateTargetRangeOnStageA(Rasterizer& rasterizer, VAddr dst_va,
                                                  u64 byte_size) {
    if (replaying) {
        return;
    }
    FlushAll(rasterizer);
    InvalidateOverlappingMemory(dst_va, byte_size);
}

bool VsmShadowMemo::TryInterceptCopyTextureRegion(
    Rasterizer& rasterizer, const Rasterizer::DirectCopyTextureRegionArgs& args) {
    if (!Enabled() || replaying) {
        return false;
    }
    DrainRetiredRanges();
    passthrough_target = {};

    if (pending_face.active) {
        FinishPendingFace(rasterizer, false);
    }
    if (has_pending_depth_clear) {
        FlushPendingDepthClear(rasterizer);
    }

    VsmTargetKey src_key{};
    VsmTargetKey dst_key{};
    // `CopyTextureRegionArgs` passes whole-resource `src_va/dst_va` (`slice 0` base) and total
    // `src_byte_size/dst_byte_size` (`6 * slice_bytes` for cubemaps), so pass `0u, 0u` hints here.
    const bool src_ok = ParseVsmImageSharp(args.srv_tsharp, 0u, 0u, src_key.dim, src_key.base_va,
                                           src_key.slice, src_key.slice_va, src_key.slice_bytes);
    const bool dst_ok = ParseVsmImageSharp(args.uav_tsharp, 0u, 0u, dst_key.dim, dst_key.base_va,
                                           dst_key.slice, dst_key.slice_va, dst_key.slice_bytes);
    src_key.valid = src_ok;
    dst_key.valid = dst_ok;

    if (src_ok && dst_ok && src_key.dim == dst_key.dim && args.src_x == 0u && args.src_y == 0u &&
        args.dst_x == 0u && args.dst_y == 0u && args.copy_w == src_key.dim &&
        args.copy_h == src_key.dim && args.copy_d == 1u) {
        const u64 cur_frame = BbStats::frame_number.load(std::memory_order_relaxed);
        const s32 src_face_idx = FindFaceIndex(src_key);
        const s32 dst_face_idx = FindFaceIndex(dst_key);

        // Topology 1 (`Copy(F -> S)` for Cubemap In-Place blur):
        if (src_face_idx >= 0 && src_face_idx != dst_face_idx) {
            auto& f = faces[static_cast<size_t>(src_face_idx)];
            EnrichFaceKey(f, src_key);
            if (f.has_deferred_clear) {
                FinalizeDeferredClear(rasterizer, static_cast<u32>(src_face_idx));
            }
            f.saw_blur_after_face = true;
            SetScratchLink(dst_key, static_cast<u32>(src_face_idx), f.skip_blur_chain,
                           /*from_blur=*/false, cur_frame);
            if (f.skip_blur_chain) {
                ++stat_copies_saved;
                return true;
            }
            return false;
        }

        // Topology 2 (`Copy(S -> F)` for Spot Copy-Back blur):
        if (const ScratchLink* link = FindScratchLink(src_key);
            link && link->from_blur && link->face_idx < kMaxFaces) {
            auto& f = faces[link->face_idx];
            if (f.allocated && f.dim == dst_key.dim &&
                ((f.base_va == dst_key.base_va && f.slice == dst_key.slice) ||
                 (f.slice_va != 0u && dst_key.slice_va != 0u && f.slice_va == dst_key.slice_va))) {
                EnrichFaceKey(f, dst_key);
                if (f.has_deferred_clear) {
                    f.has_deferred_clear = false;
                    if (num_deferred_clears > 0u) {
                        --num_deferred_clears;
                    }
                }
                MarkKnownVsmBase(f.base_va);
                f.is_known_vsm = true;
                f.saw_blur_after_face = true;
                if (link->skip && f.skip_blur_chain) {
                    ++stat_copies_saved;
                    return true;
                }
                f.was_blurred = true;
                return false;
            }
        }
    }

    FlushAllDeferredClears(rasterizer);
    InvalidateOverlappingMemory(args.dst_va, args.dst_byte_size);
    return false;
}

bool VsmShadowMemo::TryHandleDraw(Rasterizer& rasterizer, const GraphicsPipeline* pipeline,
                                  const PreparedDraw* used_prepared, bool is_indexed,
                                  u32 index_offset) {
    if (!Enabled() || replaying || !pipeline || !rasterizer.draw_pipe || !rasterizer.liverpool ||
        FrameCapture::Active() || (rasterizer.occlusion && rasterizer.occlusion->Events() != 0u)) {
        FlushAll(rasterizer);
        return false;
    }

    DrainRetiredRanges();

    const auto stages = pipeline->GetStages();
    const auto* vs = stages[static_cast<size_t>(Shader::SwStage::Vertex)];
    const auto* ps = stages[static_cast<size_t>(Shader::SwStage::Fragment)];
    bool has_extra_stages = false;
    for (size_t s = 0; s < stages.size(); ++s) {
        if (s != static_cast<size_t>(Shader::SwStage::Vertex) &&
            s != static_cast<size_t>(Shader::SwStage::Fragment) && stages[s] != nullptr) {
            has_extra_stages = true;
            break;
        }
    }

    auto& lp = *rasterizer.liverpool;
    const auto& regs = lp.regs;
    const auto& gkey = pipeline->GetGraphicsKey();

    // -------------------------------------------------------------------------
    // Branch 1: `Sample13Weight.ppo` (`vs 0x0b0acf50 + ps 0x0f61fd27`) VSM Blur
    // -------------------------------------------------------------------------
    if (!has_extra_stages && vs && ps && vs->pgm_hash == kSample13BlurVsHash &&
        ps->pgm_hash == kSample13BlurPsHash) {
        passthrough_target = {};
        if (pending_face.active) {
            FinishPendingFace(rasterizer, false);
        }
        if (has_pending_depth_clear) {
            FlushPendingDepthClear(rasterizer);
        }

        VsmTargetKey dst_key{};
        const bool dst_ok =
            gkey.mrt_mask == 1u && !regs.blend_control[0].enable &&
            ResolveColorRtvTarget(regs, lp.last_cb_extent[0], lp.last_cb_uid[0], dst_key.dim,
                                  dst_key.base_va, dst_key.slice, dst_key.slice_va,
                                  dst_key.slice_bytes);
        dst_key.valid = dst_ok;

        VsmTargetKey src_key{};
        bool src_ok = false;
        if (dst_ok && ps->images.size() == 1u) {
            Gcn::Image src_img = ps->images[0].GetSharp(*ps);
            u64 raw_words[4]{};
            std::memcpy(raw_words, &src_img, sizeof(raw_words));
            src_ok = ParseVsmImageSharp(raw_words, 0u, 0u, src_key.dim, src_key.base_va,
                                        src_key.slice, src_key.slice_va, src_key.slice_bytes);
            src_key.valid = src_ok;
        }

        if (dst_ok && src_ok && dst_key.dim == src_key.dim) {
            const u64 cur_frame = BbStats::frame_number.load(std::memory_order_relaxed);
            const s32 src_face_idx = FindFaceIndex(src_key);
            const s32 dst_face_idx = FindFaceIndex(dst_key);

            // Pass H (`Blur(F -> S)` for Spot Copy-Back or Spot Ping-Pong):
            if (src_face_idx >= 0 && src_face_idx != dst_face_idx) {
                auto& f = faces[static_cast<size_t>(src_face_idx)];
                EnrichFaceKey(f, src_key);
                MarkKnownVsmBase(f.base_va);
                f.is_known_vsm = true;
                if (f.has_deferred_clear) {
                    FinalizeDeferredClear(rasterizer, static_cast<u32>(src_face_idx));
                }
                f.saw_blur_after_face = true;
                SetScratchLink(dst_key, static_cast<u32>(src_face_idx), f.skip_blur_chain,
                               /*from_blur=*/true, cur_frame);
                if (f.skip_blur_chain) {
                    ++stat_blur_draws_saved;
                    BbStats::draws.fetch_sub(1, std::memory_order_relaxed);
                    return true;
                }
                return false;
            }

            // Pass H/V (`Blur(S -> F)` or intermediate `Blur(S1 -> S2)` ping-pong):
            if (const ScratchLink* link = FindScratchLink(src_key);
                link && link->face_idx < kMaxFaces) {
                const u32 linked_face_idx = link->face_idx;
                const bool linked_skip = link->skip;
                auto& f = faces[linked_face_idx];
                if (f.allocated && f.dim == dst_key.dim &&
                    ((f.base_va == dst_key.base_va && f.slice == dst_key.slice) ||
                     (f.slice_va != 0u && dst_key.slice_va != 0u &&
                      f.slice_va == dst_key.slice_va))) {
                    EnrichFaceKey(f, dst_key);
                    if (f.has_deferred_clear) {
                        f.has_deferred_clear = false;
                        if (num_deferred_clears > 0u) {
                            --num_deferred_clears;
                        }
                    }
                    MarkKnownVsmBase(f.base_va);
                    f.is_known_vsm = true;
                    f.saw_blur_after_face = true;
                    if (linked_skip && f.skip_blur_chain) {
                        ++stat_blur_draws_saved;
                        BbStats::draws.fetch_sub(1, std::memory_order_relaxed);
                        return true;
                    }
                    f.was_blurred = true;
                    return false;
                }
                if (f.allocated && dst_face_idx < 0 &&
                    !(src_key.base_va == dst_key.base_va && src_key.slice == dst_key.slice)) {
                    f.saw_blur_after_face = true;
                    const bool skip_now = linked_skip && f.skip_blur_chain;
                    SetScratchLink(dst_key, linked_face_idx, skip_now, /*from_blur=*/true,
                                   cur_frame);
                    if (skip_now) {
                        ++stat_blur_draws_saved;
                        BbStats::draws.fetch_sub(1, std::memory_order_relaxed);
                        return true;
                    }
                    return false;
                }
            }
        }

        FlushAllDeferredClears(rasterizer);
        if (dst_ok) {
            InvalidateOverlappingMemory(dst_key.slice_va ? dst_key.slice_va : dst_key.base_va,
                                        dst_key.slice_bytes);
        }
        return false;
    }

    // -------------------------------------------------------------------------
    // Branch 2: 3D Local VSM Moment Geometry Draw (`R16G16Sfloat` + Depth)
    // -------------------------------------------------------------------------
    VsmTargetKey target{};
    const bool is_vsm_geom_cand =
        !has_extra_stages && vs && ps && ps->pgm_hash != kSample13BlurPsHash &&
        !vs->has_storage_images && !ps->has_storage_images && !vs->uses_dma && !ps->uses_dma &&
        gkey.mrt_mask == 1u && u32(gkey.write_masks[0]) != 0u && !regs.blend_control[0].enable &&
        regs.color_control.mode == Gcn::ColorControl::OperationMode::Normal &&
        regs.depth_control.depth_enable && regs.depth_buffer.DepthAddress() != 0u &&
        ResolveColorRtvTarget(regs, lp.last_cb_extent[0], lp.last_cb_uid[0], target.dim,
                              target.base_va, target.slice, target.slice_va, target.slice_bytes);
    target.valid = is_vsm_geom_cand;

    if (!is_vsm_geom_cand) {
        FlushAll(rasterizer);
        if (regs.color_buffers[0]) {
            InvalidateOverlappingMemory(regs.color_buffers[0].Address(),
                                        regs.color_buffers[0].GetColorSliceSize());
        }
        return false;
    }

    // Flush open light-tile / VFX-UI batches before starting/appending a VSM shadow pass.
    if (rasterizer.HasPendingLightTiles()) {
        rasterizer.FlushLightTileBatches();
    }
    if (rasterizer.HasPendingVfxUiBatches()) {
        rasterizer.FlushVfxUiBatches();
    }

    const u64 cur_frame = BbStats::frame_number.load(std::memory_order_relaxed);
    const VAddr db_addr = regs.depth_buffer.DepthAddress();
    const u32 db_slice = regs.depth_view.slice_start;

    // If a different face is open (or a fresh depth clear arrived between faces), finish it first.
    if (pending_face.active) {
        const bool same_target =
            pending_face.target.dim == target.dim &&
            ((pending_face.target.base_va == target.base_va &&
              pending_face.target.slice == target.slice) ||
             (pending_face.target.slice_va != 0u && target.slice_va != 0u &&
              pending_face.target.slice_va == target.slice_va)) &&
            pending_face.db_addr == db_addr && pending_face.db_slice == db_slice &&
            !has_pending_depth_clear && !regs.depth_render_control.depth_clear_enable;
        if (!same_target) {
            FinishPendingFace(rasterizer, false);
        }
    }

    const u32 face_idx = GetOrAllocateFaceIndex(target, cur_frame);
    MarkKnownVsmBase(faces[face_idx].base_va);
    faces[face_idx].is_known_vsm = true;

    if (passthrough_target.valid) {
        const bool same_pt =
            passthrough_target.dim == target.dim &&
            ((passthrough_target.base_va == target.base_va &&
              passthrough_target.slice == target.slice) ||
             (passthrough_target.slice_va != 0u && target.slice_va != 0u &&
              passthrough_target.slice_va == target.slice_va)) &&
            !has_pending_depth_clear && !regs.depth_render_control.depth_clear_enable &&
            !faces[face_idx].has_deferred_clear;
        if (same_pt) {
            faces[face_idx].valid = false;
            faces[face_idx].skip_blur_chain = false;
            return false;
        }
        passthrough_target = {};
    }

    if (!pending_face.active) {
        pending_face.active = true;
        pending_face.unmemoizable = false;
        pending_face.has_depth_clear = false;
        pending_face.face_idx = face_idx;
        pending_face.target = target;
        pending_face.db_addr = db_addr;
        pending_face.db_slice = db_slice;
        pending_face.running_digest = MixU64(0x56534d5f46414345ULL, target.dim);
        pending_face.num_draws = 0u;
        pending_face.bound_ranges.clear();
        pending_face.saved_initial_dirty = lp.pipe_dirty;
        CaptureStageBChunks(regs, pending_face.saved_initial_chunks);

        if (has_pending_depth_clear) {
            if (pending_depth_clear_args.dst_va == db_addr) {
                pending_face.has_depth_clear = true;
                pending_face.depth_clear_args = pending_depth_clear_args;
                has_pending_depth_clear = false;
            } else {
                FlushPendingDepthClear(rasterizer);
            }
        }
    }

    if (pending_face.num_draws >= kMaxDrawsPerFace) {
        FinishPendingFace(rasterizer, /*force_miss=*/true);
        return false;
    }

    // Hash this draw's pipeline, raster/depth/viewport state, CBs, T#/S#, VBs, and IB:
    u32 vtx_off = regs.index_offset;
    u32 inst_off = 0u;
    if (const auto& fetch_sh = pipeline->GetFetchShader(); fetch_sh) {
        if (vtx_off == 0u && fetch_sh->vertex_offset_sgpr != -1) {
            vtx_off = vs->UserData()[fetch_sh->vertex_offset_sgpr];
        }
        if (fetch_sh->instance_offset_sgpr != -1) {
            inst_off = vs->UserData()[fetch_sh->instance_offset_sgpr];
        }
    }
    u64 dh = pending_face.running_digest;
    dh = MixU64(dh, reinterpret_cast<uintptr_t>(pipeline));
    dh = MixU64(dh, vs->pgm_hash);
    dh = MixU64(dh, ps->pgm_hash);
    dh = MixU64(dh, (u64(is_indexed) << 63) | (u64(regs.num_instances.NumInstances()) << 32) |
                        u64(regs.num_indices));
    dh = MixU64(dh, (u64(vtx_off) << 32) | u64(inst_off));
    dh = MixU64(dh, regs.draw_initiator);
    dh = MixU64(dh, std::bit_cast<u32>(regs.depth_control));
    dh = MixU64(dh, std::bit_cast<u32>(regs.depth_render_control));
    dh = MixU64(dh, std::bit_cast<u32>(regs.polygon_control));
    dh = MixU64(dh, std::bit_cast<u32>(regs.clipper_control));
    dh = MixU64(dh, std::bit_cast<u32>(regs.viewport_control));
    dh = XXH3_64bits_withSeed(&regs.viewports[0], sizeof(regs.viewports[0]), dh);
    dh = XXH3_64bits_withSeed(&regs.screen_scissor, sizeof(regs.screen_scissor), dh);
    dh = XXH3_64bits_withSeed(&regs.viewport_scissors[0], sizeof(regs.viewport_scissors[0]), dh);

    const u64 topo_gen = VideoCore::DirectBuffers::ExactBufferTopologyGen();
    const bool cue_gpu_modified = VideoCore::DirectBuffers::IsCueHeapEverGpuModified();
    VAddr cue_base = 0;
    VAddr cue_end = 0;
    VideoCore::DirectBuffers::GetTransientCueHeapBounds(cue_base, cue_end);

    auto hash_stage_resources = [&](const Shader::Info& st) {
        for (u32 bi = 0; bi < st.buffers.size(); ++bi) {
            const auto& desc = st.buffers[bi];
            if (desc.IsSpecial()) {
                continue;
            }
            if (desc.is_written) {
                pending_face.unmemoizable = true;
                return;
            }
            const Gcn::Buffer sharp = desc.GetSharp(st);
            const VAddr addr = sharp.base_address;
            u64 raw_sz = sharp.GetSize();
            // Mix slot & format metadata ONLY (never transient `cueHeap` ring `addr`!).
            dh = MixU64(dh, (u64(bi) << 48) ^ u64(sharp.GetStride()) ^
                                (u64(sharp.data_format) << 16) ^ (u64(sharp.num_format) << 24));
            if (!addr || raw_sz == 0u) {
                continue;
            }
            const bool in_cue = (addr >= cue_base && addr < cue_end);
            if (in_cue) {
                raw_sz = (raw_sz > 65536u)
                             ? VideoCore::DirectBuffers::ClampTransientCueReadSize(addr, raw_sz)
                             : std::min<u64>(raw_sz, cue_end - addr);
            } else if (VideoCore::DirectBuffers::IsNativeOwnedAddress(addr)) {
                raw_sz = VideoCore::DirectBuffers::ClampNativeGpuReadSize(addr, raw_sz);
            }
            if (raw_sz == 0u) {
                continue;
            }
            const u64 bounded_track_sz = std::min<u64>(raw_sz, 64ULL << 20);
            pending_face.bound_ranges.push_back({addr, bounded_track_sz});

            // Tier 1: Stage-A CPU shadow (`<= 2048 B` `CopyBufferBatch` dynamic CBs).
            if (rasterizer.light_tile_batcher && raw_sz <= LightTileBatcher::kMaxShadowBytes) {
                u32 shadow_rem = 0u;
                if (const u8* sh = rasterizer.light_tile_batcher->LookupCpuShadow(
                        addr, 4u, &shadow_rem)) {
                    const u64 use_sz = std::min<u64>(raw_sz, shadow_rem);
                    if (use_sz >= 4u) {
                        dh = MixU64(dh, use_sz);
                        dh = HashConstantBufferBytes(dh, sh, use_sz);
                        continue;
                    }
                }
            }

            // Tier 2: Transient `cueHeap` or `auxCpuRing` CPU-written constant/matrix buffers.
            if ((!cue_gpu_modified && in_cue) ||
                VideoCore::DirectBuffers::IsTransientAuxCpuRing(addr,
                                                                std::min<u64>(raw_sz, 16u))) {
                if (raw_sz <= 131072u) {
                    dh = MixU64(dh, raw_sz);
                    dh = HashConstantBufferBytes(dh, reinterpret_cast<const u8*>(addr), raw_sz);
                    continue;
                }
                pending_face.unmemoizable = true;
                return;
            }

            // Tier 3: Settled native `GXGnmBuffer` (`!PendingWriteOverlaps`) via 64-KiB epoch.
            if (VideoCore::DirectBuffers::IsNativeOwnedAddress(addr)) {
                if (rasterizer.PendingWriteOverlaps(addr, bounded_track_sz)) {
                    // Large (`> 2048 B`) dynamic CB updated this frame (e.g. large bone palette):
                    pending_face.unmemoizable = true;
                    return;
                }
                dh = MixU64(dh, addr);
                dh = MixU64(dh, bounded_track_sz);
                dh = MixU64(dh, topo_gen);
                dh = MixU64(dh, QueryPage64kWriteDigest(addr, bounded_track_sz));
                continue;
            }

            // Tier 4: Mapped read-only guest RAM not modified by GPU.
            uintptr_t mapped_end = 0;
            if (raw_sz <= 8192u && BbLookupMappedEnd(addr, mapped_end) &&
                raw_sz <= (mapped_end - addr) && !rasterizer.PendingWriteOverlaps(addr, raw_sz) &&
                !rasterizer.buffer_cache.IsRegionGpuModified(addr, raw_sz)) {
                dh = MixU64(dh, raw_sz);
                dh = HashConstantBufferBytes(dh, reinterpret_cast<const u8*>(addr), raw_sz);
                continue;
            }

            pending_face.unmemoizable = true;
            return;
        }

        for (u32 ti = 0; ti < st.images.size(); ++ti) {
            if (st.images[ti].is_written) {
                pending_face.unmemoizable = true;
                return;
            }
            Gcn::Image img = st.images[ti].GetSharp(st);
            reinterpret_cast<u32*>(&img)[7] = 0u;
            dh = XXH3_64bits_withSeed(&img, sizeof(img), dh);
        }
        for (u32 si = 0; si < st.samplers.size(); ++si) {
            Gcn::Sampler smp = st.samplers[si].GetSharp(st);
            (void)VideoCore::DirectTextures::UnpackAndCleanSampler(smp);
            dh = XXH3_64bits_withSeed(&smp, sizeof(smp), dh);
        }
    };

    hash_stage_resources(*vs);
    if (!pending_face.unmemoizable) {
        hash_stage_resources(*ps);
    }

    // Snapshot `BufferedDraw` & hash VBs + IB:
    BufferedDraw& bd = pending_face.draws[pending_face.num_draws];
    bd.pipeline = pipeline;
    bd.is_indexed = is_indexed;
    bd.index_offset = index_offset;
    bd.num_stages = 0u;
    for (const auto* st : stages) {
        if (!st) {
            continue;
        }
        auto& dst_st = bd.stages[bd.num_stages++];
        dst_st.pgm_base = st->pgm_base;
        dst_st.user_data.assign(st->user_data.begin(), st->user_data.end());
        dst_st.flattened_ud_buf = st->flattened_ud_buf;
    }

    bd.vsharps.clear();
    if (const auto& fetch = pipeline->GetFetchShader(); fetch) {
        const auto* fast_vs = rasterizer.pipeline_cache.PeekFastVsharps(pipeline);
        if (fast_vs && fast_vs->size() == fetch->attributes.size()) {
            bd.vsharps.assign(fast_vs->begin(), fast_vs->end());
        } else {
            const auto vs_ud = vs->UserData();
            for (const auto& attrib : fetch->attributes) {
                bd.vsharps.push_back(attrib.GetSharpDirect(vs_ud));
            }
        }
    }

    if (!pending_face.unmemoizable) {
        for (const auto& vb : bd.vsharps) {
            if (!vb.base_address || vb.GetSize() == 0u) {
                dh = MixU64(dh, 0x56425f4e554c4c30ULL);
                continue;
            }
            u64 vb_sz = vb.GetSize();
            const bool vb_in_cue = (vb.base_address >= cue_base && vb.base_address < cue_end);
            if ((!cue_gpu_modified && vb_in_cue) ||
                VideoCore::DirectBuffers::IsTransientAuxCpuRing(vb.base_address,
                                                                std::min<u64>(vb_sz, 16u))) {
                if (vb_in_cue) {
                    vb_sz = (vb_sz > 65536u) ? VideoCore::DirectBuffers::ClampTransientCueReadSize(
                                                   vb.base_address, vb_sz)
                                             : std::min<u64>(vb_sz, cue_end - vb.base_address);
                }
                pending_face.bound_ranges.push_back({vb.base_address, vb_sz});
                // Strip transient ring `base_address` from `V#` metadata (e.g. `vs_167daba6`
                // per-instance `vs_in_attr1` stream in `cueHeap`) and hash CPU bytes directly!
                Gcn::Buffer vb_clean = vb;
                vb_clean.base_address = 0u;
                dh = XXH3_64bits_withSeed(&vb_clean, sizeof(vb_clean), dh);
                dh = MixU64(dh, vb_sz);
                if (vb_sz <= 65536u) {
                    dh = XXH3_64bits_withSeed(reinterpret_cast<const void*>(vb.base_address),
                                              static_cast<size_t>(vb_sz), dh);
                } else {
                    pending_face.unmemoizable = true;
                    break;
                }
            } else if (VideoCore::DirectBuffers::IsNativeOwnedAddress(vb.base_address)) {
                dh = XXH3_64bits_withSeed(&vb, sizeof(vb), dh);
                vb_sz = VideoCore::DirectBuffers::ClampNativeGpuReadSize(vb.base_address, vb_sz);
                const u64 bounded_vb = std::min<u64>(vb_sz, 64ULL << 20);
                pending_face.bound_ranges.push_back({vb.base_address, bounded_vb});
                if (rasterizer.PendingWriteOverlaps(vb.base_address, bounded_vb)) {
                    // Dynamic / cloth VB updated this frame via CopyBufferBatch: force 0-lag MISS!
                    pending_face.unmemoizable = true;
                    break;
                }
                dh = MixU64(dh, topo_gen);
                dh = MixU64(dh, QueryPage64kWriteDigest(vb.base_address, bounded_vb));
            } else {
                dh = XXH3_64bits_withSeed(&vb, sizeof(vb), dh);
                const u64 bounded_vb = std::min<u64>(vb_sz, 64ULL << 20);
                if (rasterizer.PendingWriteOverlaps(vb.base_address, bounded_vb) ||
                    rasterizer.buffer_cache.IsRegionGpuModified(vb.base_address, bounded_vb)) {
                    pending_face.unmemoizable = true;
                    break;
                }
                dh = MixU64(dh, QueryPage64kWriteDigest(vb.base_address, bounded_vb));
            }
        }
    }

    if (!pending_face.unmemoizable && is_indexed) {
        const u32 idx_elem =
            (regs.index_buffer_type.index_type == Gcn::IndexType::Index16) ? 2u : 4u;
        const VAddr ib_addr =
            regs.index_base_address.Address<VAddr>() + u64(index_offset) * idx_elem;
        const u64 ib_bytes = u64(regs.num_indices) * idx_elem;
        dh = MixU64(dh, (u64(idx_elem) << 56) | ib_bytes);
        if (ib_addr && ib_bytes != 0u) {
            pending_face.bound_ranges.push_back({ib_addr, ib_bytes});
            const bool ib_in_cue = (ib_addr >= cue_base && (cue_end - ib_addr) >= ib_bytes);
            if ((!cue_gpu_modified && ib_in_cue) ||
                VideoCore::DirectBuffers::IsTransientAuxCpuRing(ib_addr, ib_bytes)) {
                if (ib_bytes <= 65536u) {
                    dh = XXH3_64bits_withSeed(reinterpret_cast<const void*>(ib_addr),
                                              static_cast<size_t>(ib_bytes), dh);
                } else {
                    pending_face.unmemoizable = true;
                }
            } else if (VideoCore::DirectBuffers::IsNativeOwnedAddress(ib_addr)) {
                dh = MixU64(dh, ib_addr);
                if (rasterizer.PendingWriteOverlaps(ib_addr, ib_bytes)) {
                    pending_face.unmemoizable = true;
                } else {
                    dh = MixU64(dh, topo_gen);
                    dh = MixU64(dh, QueryPage64kWriteDigest(ib_addr, ib_bytes));
                }
            } else if (rasterizer.PendingWriteOverlaps(ib_addr, ib_bytes) ||
                       rasterizer.buffer_cache.IsRegionGpuModified(ib_addr, ib_bytes)) {
                pending_face.unmemoizable = true;
            } else {
                dh = MixU64(dh, ib_addr);
                dh = MixU64(dh, QueryPage64kWriteDigest(ib_addr, ib_bytes));
            }
        }
    }

    CaptureStageBChunks(regs, bd.regs_chunks);
    bd.cb_extent = lp.last_cb_extent;
    bd.db_extent = lp.last_db_extent;
    bd.cb_uid = lp.last_cb_uid;
    bd.db_uid = lp.last_db_uid;

    pending_face.running_digest = dh;
    ++pending_face.num_draws;
    BbStats::draws.fetch_sub(1, std::memory_order_relaxed);
    return true;
}

void VsmShadowMemo::FinishPendingFace(Rasterizer& rasterizer, bool force_miss) {
    if (!pending_face.active) {
        return;
    }
    pending_face.active = false;

    const u32 face_idx = pending_face.face_idx;
    auto& f = faces[face_idx];
    const u64 cur_frame = BbStats::frame_number.load(std::memory_order_relaxed);

    u64 final_digest = MixU64(pending_face.running_digest, pending_face.num_draws);
    if (f.has_deferred_clear) {
        final_digest = XXH3_64bits_withSeed(f.deferred_clear_args.clear_rgba, sizeof(u32) * 4u,
                                            final_digest);
    } else {
        final_digest = MixU64(final_digest, 0x4e4f5f434c523030ULL);
    }
    if (pending_face.has_depth_clear) {
        final_digest = MixU64(final_digest,
                              std::bit_cast<u32>(pending_face.depth_clear_args.depth_clear));
        final_digest = MixU64(final_digest, pending_face.depth_clear_args.stencil_and_flags);
    }

    const bool blur_consistent = (f.was_blurred == f.saw_blur_after_face);
    const bool aged_out =
        (cur_frame > f.last_render_frame) &&
        ((cur_frame - f.last_render_frame) >= kMaxPeriodicRefreshFrames) &&
        (last_periodic_refresh_frame != cur_frame);

    const bool can_hit = !force_miss && !pending_face.unmemoizable && f.valid &&
                         f.num_last_draws == pending_face.num_draws && blur_consistent &&
                         !aged_out && f.geom_digest == final_digest;

    ++stat_faces_evaluated;
    if (pending_face.unmemoizable) {
        ++stat_unmemoizable;
    }

    if (can_hit) {
        // HIT: discard deferred clears and buffered geometry draws; arm blur-chain skip if blurred!
        if (f.has_deferred_clear) {
            f.has_deferred_clear = false;
            if (num_deferred_clears > 0u) {
                --num_deferred_clears;
            }
            ++stat_clears_saved;
        }
        if (pending_face.has_depth_clear) {
            pending_face.has_depth_clear = false;
            ++stat_clears_saved;
        }
        f.skip_blur_chain = f.was_blurred;
        f.saw_blur_after_face = false;
        stat_geom_draws_saved += pending_face.num_draws;
        ++stat_face_hits;
        if (!first_active_logged && stat_geom_draws_saved >= 8u) {
            first_active_logged = true;
            std::printf("[VsmShadowMemo] Active: skipping unchanged local VSM shadow faces "
                        "(hits=%llu misses=%llu geom_draws_saved=%llu)\n",
                        static_cast<unsigned long long>(stat_face_hits),
                        static_cast<unsigned long long>(stat_face_misses),
                        static_cast<unsigned long long>(stat_geom_draws_saved));
            std::fflush(stdout);
        }
        MaybeLogTelemetry();
        pending_face.num_draws = 0u;
        pending_face.bound_ranges.clear();
        return;
    }

    // MISS: emit deferred clears and replay buffered geometry draws in exact stream order!
    if (aged_out && !force_miss && !pending_face.unmemoizable) {
        last_periodic_refresh_frame = cur_frame;
    }

    if (f.has_deferred_clear) {
        const auto clr = f.deferred_clear_args;
        f.has_deferred_clear = false;
        if (num_deferred_clears > 0u) {
            --num_deferred_clears;
        }
        EmitClearColorNow(rasterizer, clr);
    }
    if (pending_face.has_depth_clear) {
        pending_face.has_depth_clear = false;
        EmitClearDepthNow(rasterizer, pending_face.depth_clear_args);
    }

    {
        ReplayScope scope{replaying};
        auto& lp = *rasterizer.liverpool;
        StageBChunks caller_chunks{};
        CaptureStageBChunks(lp.regs, caller_chunks);
        const auto caller_cb_extent = lp.last_cb_extent;
        const auto caller_db_extent = lp.last_db_extent;
        const auto caller_cb_uid = lp.last_cb_uid;
        const auto caller_db_uid = lp.last_db_uid;
        const Gcn::RegDirty caller_dirty = lp.pipe_dirty;

        StageBChunks prev_emitted_chunks{};
        bool any_emitted = false;
        std::array<BatchedLightStageOverride, Shader::MaxStageTypes> stage_ovrs{};

        for (u32 i = 0; i < pending_face.num_draws; ++i) {
            auto& d = pending_face.draws[i];
            RestoreStageBChunks(lp.regs, d.regs_chunks);
            lp.last_cb_extent = d.cb_extent;
            lp.last_db_extent = d.db_extent;
            lp.last_cb_uid = d.cb_uid;
            lp.last_db_uid = d.db_uid;

            if (!any_emitted) {
                lp.pipe_dirty = pending_face.saved_initial_dirty;
                MarkStageBDiffDirty(pending_face.saved_initial_chunks, d.regs_chunks,
                                    lp.pipe_dirty);
            } else {
                lp.pipe_dirty.Clear();
                MarkStageBDiffDirty(prev_emitted_chunks, d.regs_chunks, lp.pipe_dirty);
            }

            for (u32 s = 0; s < d.num_stages; ++s) {
                stage_ovrs[s] = BatchedLightStageOverride{
                    .pgm_base = d.stages[s].pgm_base,
                    .user_data = d.stages[s].user_data,
                    .flattened_ud_buf = d.stages[s].flattened_ud_buf,
                    .ring_bindings = {},
                    .collect_ring_bindings = true,
                };
            }
            const BatchedLightDrawOverride draw_ovr{
                .stages = std::span<const BatchedLightStageOverride>{stage_ovrs.data(),
                                                                     d.num_stages},
                .vsharps = d.vsharps,
                .note_guest_vb_ib_reads = true,
            };
            rasterizer.PostDraw(d.pipeline, nullptr, d.is_indexed, d.index_offset, nullptr,
                                nullptr, &draw_ovr);
            BbStats::draws.fetch_add(1, std::memory_order_relaxed);
            prev_emitted_chunks = d.regs_chunks;
            any_emitted = true;
        }

        RestoreStageBChunks(lp.regs, caller_chunks);
        lp.last_cb_extent = caller_cb_extent;
        lp.last_db_extent = caller_db_extent;
        lp.last_cb_uid = caller_cb_uid;
        lp.last_db_uid = caller_db_uid;
        if (any_emitted) {
            lp.pipe_dirty.Clear();
            lp.pipe_dirty.blocks |= caller_dirty.blocks;
            MarkStageBDiffDirty(prev_emitted_chunks, caller_chunks, lp.pipe_dirty);
        } else {
            lp.pipe_dirty = caller_dirty;
        }
    }

    if (force_miss || pending_face.unmemoizable) {
        if (force_miss) {
            passthrough_target = pending_face.target;
        }
        f.valid = false;
        f.was_blurred = false;
        f.saw_blur_after_face = false;
        f.skip_blur_chain = false;
    } else {
        f.valid = true;
        f.num_last_draws = pending_face.num_draws;
        f.geom_digest = final_digest;
        f.was_blurred = false;
        f.saw_blur_after_face = false;
        f.skip_blur_chain = false;
        f.last_render_frame = cur_frame;
    }
    ++stat_face_misses;
    MaybeLogTelemetry();
    pending_face.num_draws = 0u;
    pending_face.bound_ranges.clear();
}

void VsmShadowMemo::MaybeLogTelemetry() noexcept {
    if (!summary_logged && stat_faces_evaluated >= 300u) {
        summary_logged = true;
        std::printf("[VsmShadowMemo] Summary(faces=%llu): hits=%llu misses=%llu empty_hits=%llu "
                    "unmemo=%llu geom_draws_saved=%llu blur_draws_saved=%llu copies_saved=%llu "
                    "clears_saved=%llu\n",
                    static_cast<unsigned long long>(stat_faces_evaluated),
                    static_cast<unsigned long long>(stat_face_hits),
                    static_cast<unsigned long long>(stat_face_misses),
                    static_cast<unsigned long long>(stat_empty_hits),
                    static_cast<unsigned long long>(stat_unmemoizable),
                    static_cast<unsigned long long>(stat_geom_draws_saved),
                    static_cast<unsigned long long>(stat_blur_draws_saved),
                    static_cast<unsigned long long>(stat_copies_saved),
                    static_cast<unsigned long long>(stat_clears_saved));
        std::fflush(stdout);
    }

    if (MemoLogEnabled()) {
        static auto last_log = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        if (now - last_log >= std::chrono::seconds(5)) {
            last_log = now;
            std::printf("[VsmShadowMemo] 5s: hits=%llu misses=%llu empty_hits=%llu unmemo=%llu "
                        "geom_saved=%llu blur_saved=%llu copies_saved=%llu clears_saved=%llu\n",
                        static_cast<unsigned long long>(stat_face_hits),
                        static_cast<unsigned long long>(stat_face_misses),
                        static_cast<unsigned long long>(stat_empty_hits),
                        static_cast<unsigned long long>(stat_unmemoizable),
                        static_cast<unsigned long long>(stat_geom_draws_saved),
                        static_cast<unsigned long long>(stat_blur_draws_saved),
                        static_cast<unsigned long long>(stat_copies_saved),
                        static_cast<unsigned long long>(stat_clears_saved));
            std::fflush(stdout);
        }
    }
}

} // namespace Vulkan
