// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 11 (Target C): Soft-Particle VFX, Scaleform GFx 2D HUD & Small-Draw Run Batcher on Stage A.

#include "render/vk_vfx_ui_batch.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "buffers/direct_buffers.h"
#include "common/alignment.h"
#include "cp/cp_tasks.h"
#include "render/raster_internal.h"
#include "shader/backend/spirv/native_shaders.h"
#include "../../dantelion_gx/include/gx_direct_vulkan.h"

extern "C" bool BbLookupMappedEnd(u64 address, uintptr_t& out_end);

namespace Vulkan {

namespace {

constexpr u64 kLightAccVsHash = 0x000000008747a367ULL;
constexpr u64 kVsmShadowVsHash = 0x00000000167daba6ULL;

constexpr u64 kScaleform2404VsHash = 0x0000000024042a9bULL;
constexpr u64 kScaleformE94BVsHash = 0x00000000e94b5b06ULL;
constexpr u64 kScaleformA400VsHash = 0x00000000a400228bULL;

constexpr u64 kFfxTessLsHash = 0x000000004cd21a0fULL;
constexpr u64 kFfxQuadVsHash = 0x0000000041ae600fULL;

constexpr bool IsTargetVsOrLsHash(u64 h) noexcept {
    return h == kScaleform2404VsHash || h == kScaleformE94BVsHash || h == kScaleformA400VsHash ||
           h == kFfxTessLsHash || h == kFfxQuadVsHash;
}

constexpr bool IsFfxHsHash(u64 h) noexcept {
    return h == 0x000000008453cd1cULL || h == 0x00000000bc234799ULL ||
           h == 0x00000000d67db0efULL;
}

constexpr bool IsFfxDsHash(u64 h) noexcept {
    return h == 0x000000007f03e945ULL || h == 0x000000000c1888bdULL ||
           h == 0x000000000c6f6c39ULL || h == 0x0000000052d756dfULL ||
           h == 0x00000000da35f466ULL;
}

constexpr bool IsFfxSoftPsHash(u64 h) noexcept {
    return h == 0x0000000071c96598ULL || h == 0x000000005e224cb6ULL ||
           h == 0x00000000d729c469ULL || h == 0x0000000020739372ULL;
}

constexpr bool IsFfxSoft2MrtPsHash(u64 h) noexcept {
    return h == 0x00000000d729c469ULL || h == 0x0000000020739372ULL;
}

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

void ZeroStageBWordRange(VfxUiBatcher::StageBChunks& c, u32 first_word, u32 word_count) noexcept {
    for (u32 w = first_word; w < first_word + word_count; ++w) {
        const s32 slot = RegWordToStageBSlot(w);
        if (slot >= 0) {
            c[static_cast<size_t>(slot)] = 0u;
        }
    }
}

void CaptureStageBChunks(const Gcn::Regs& regs, VfxUiBatcher::StageBChunks& out) noexcept {
    for (u32 bit = 0; bit < Gcn::RegDirty::StageBNumChunks; ++bit) {
        std::memcpy(&out[bit * Gcn::RegDirty::StageBChunkWords],
                    regs.reg_array.data() + Gcn::RegDirty::StageBChunkWordOffset(bit),
                    Gcn::RegDirty::StageBChunkWords * sizeof(u32));
    }
}

void RestoreStageBChunks(Gcn::Regs& regs, const VfxUiBatcher::StageBChunks& in) noexcept {
    for (u32 bit = 0; bit < Gcn::RegDirty::StageBNumChunks; ++bit) {
        std::memcpy(regs.reg_array.data() + Gcn::RegDirty::StageBChunkWordOffset(bit),
                    &in[bit * Gcn::RegDirty::StageBChunkWords],
                    Gcn::RegDirty::StageBChunkWords * sizeof(u32));
    }
}

void NormalizeStageBChunksForRun(VfxUiBatcher::StageBChunks& c) noexcept {
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, index_offset) / 4u), 1u);
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, index_base_address) / 4u), 2u);
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, index_buffer_type) / 4u), 1u);
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, draw_initiator) / 4u), 1u);
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, num_indices) / 4u), 1u);
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, enable_primitive_restart) / 4u),
                        1u);
    ZeroStageBWordRange(c, static_cast<u32>(offsetof(Gcn::Regs, primitive_restart_index) / 4u), 1u);
}

void MarkStageBDiffDirty(const VfxUiBatcher::StageBChunks& prev,
                         const VfxUiBatcher::StageBChunks& next, Gcn::RegDirty& dirty) noexcept {
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

bool BatchLogEnabled() noexcept {
    static const bool on = [] {
        const char* env = std::getenv("BB_VFX_UI_BATCH_LOG");
        return env && env[0] == '1';
    }();
    return on;
}

float HalfToFloat(u16 h) noexcept {
    const u32 sign = static_cast<u32>(h & 0x8000u) << 16u;
    const u32 exp = (h >> 10u) & 0x1Fu;
    const u32 mant = h & 0x03FFu;
    u32 bits = 0u;
    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;
        } else {
            u32 m = mant;
            s32 e = -14;
            while ((m & 0x0400u) == 0u) {
                m <<= 1u;
                --e;
            }
            m &= 0x03FFu;
            bits = sign | (static_cast<u32>(e + 127) << 23u) | (m << 13u);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (mant << 13u);
    } else {
        bits = sign | ((exp + 112u) << 23u) | (mant << 13u);
    }
    return std::bit_cast<float>(bits);
}

u32 DataFormatByteSize(Gcn::DataFormat dfmt) noexcept {
    using DF = Gcn::DataFormat;
    switch (dfmt) {
    case DF::Format8:
        return 1u;
    case DF::Format16:
    case DF::Format8_8:
        return 2u;
    case DF::Format32:
    case DF::Format16_16:
    case DF::Format8_8_8_8:
    case DF::Format10_11_11:
    case DF::Format11_11_10:
    case DF::Format10_10_10_2:
    case DF::Format2_10_10_10:
        return 4u;
    case DF::Format32_32:
    case DF::Format16_16_16_16:
        return 8u;
    case DF::Format32_32_32:
        return 12u;
    case DF::Format32_32_32_32:
        return 16u;
    default:
        return 0u;
    }
}

float ConvertScalarFloat(u32 raw_u, s32 raw_s, u32 bits, Gcn::NumberFormat nfmt) noexcept {
    using NF = Gcn::NumberFormat;
    const u32 max_u = (bits == 32u) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    const s32 max_s = (bits == 32u) ? 0x7FFFFFFF : static_cast<s32>((1u << (bits - 1u)) - 1u);
    switch (nfmt) {
    case NF::Unorm:
        return static_cast<float>(raw_u) / static_cast<float>(max_u);
    case NF::Snorm:
        return std::max(-1.0f, static_cast<float>(raw_s) / static_cast<float>(max_s));
    case NF::Uscaled:
        return static_cast<float>(raw_u);
    case NF::Sscaled:
        return static_cast<float>(raw_s);
    case NF::Float:
        if (bits == 32u) {
            return std::bit_cast<float>(raw_u);
        }
        if (bits == 16u) {
            return HalfToFloat(static_cast<u16>(raw_u));
        }
        return 0.0f;
    default:
        return 0.0f;
    }
}

bool DecodeAttrFloat4(const Gcn::Buffer& vsharp, const u8* elem_ptr,
                      std::array<float, 4>& out) noexcept {
    using DF = Gcn::DataFormat;
    using NF = Gcn::NumberFormat;
    const DF dfmt = Gcn::DataFormat(vsharp.data_format);
    const NF nfmt = Gcn::NumberFormat(vsharp.num_format);
    if (Gcn::GetNumberClass(vsharp.GetNumberFmt()) != Gcn::NumberClass::Float) {
        return false;
    }
    if (vsharp.GetNumberConversion() != Gcn::NumberConversion::None) {
        return false;
    }

    std::array<float, 4> raw{0.0f, 0.0f, 0.0f, 1.0f};
    switch (dfmt) {
    case DF::Format32:
    case DF::Format32_32:
    case DF::Format32_32_32:
    case DF::Format32_32_32_32: {
        const u32 comps = (dfmt == DF::Format32)          ? 1u
                          : (dfmt == DF::Format32_32)     ? 2u
                          : (dfmt == DF::Format32_32_32)  ? 3u
                                                          : 4u;
        for (u32 c = 0; c < comps; ++c) {
            u32 u = 0;
            std::memcpy(&u, elem_ptr + c * 4u, 4u);
            raw[c] = ConvertScalarFloat(u, static_cast<s32>(u), 32u, nfmt);
        }
        break;
    }
    case DF::Format16:
    case DF::Format16_16:
    case DF::Format16_16_16_16: {
        const u32 comps = (dfmt == DF::Format16)             ? 1u
                          : (dfmt == DF::Format16_16)        ? 2u
                                                             : 4u;
        for (u32 c = 0; c < comps; ++c) {
            u16 u = 0;
            std::memcpy(&u, elem_ptr + c * 2u, 2u);
            raw[c] = ConvertScalarFloat(u, static_cast<s16>(u), 16u, nfmt);
        }
        break;
    }
    case DF::Format8:
    case DF::Format8_8:
    case DF::Format8_8_8_8: {
        const u32 comps = (dfmt == DF::Format8)         ? 1u
                          : (dfmt == DF::Format8_8)     ? 2u
                                                        : 4u;
        for (u32 c = 0; c < comps; ++c) {
            const u8 u = elem_ptr[c];
            raw[c] = ConvertScalarFloat(u, static_cast<s8>(u), 8u, nfmt);
        }
        break;
    }
    default:
        return false;
    }

    const Gcn::CompSwizzle sels[4]{
        Gcn::CompSwizzle(vsharp.dst_sel_x),
        Gcn::CompSwizzle(vsharp.dst_sel_y),
        Gcn::CompSwizzle(vsharp.dst_sel_z),
        Gcn::CompSwizzle(vsharp.dst_sel_w),
    };
    for (u32 i = 0; i < 4u; ++i) {
        switch (sels[i]) {
        case Gcn::CompSwizzle::Zero:
            out[i] = 0.0f;
            break;
        case Gcn::CompSwizzle::One:
            out[i] = 1.0f;
            break;
        case Gcn::CompSwizzle::Red:
            out[i] = raw[0];
            break;
        case Gcn::CompSwizzle::Green:
            out[i] = raw[1];
            break;
        case Gcn::CompSwizzle::Blue:
            out[i] = raw[2];
            break;
        case Gcn::CompSwizzle::Alpha:
            out[i] = raw[3];
            break;
        default:
            return false;
        }
    }
    return true;
}

bool DecodeAttrBitcastU32X(const Gcn::Buffer& vsharp, const u8* elem_ptr, u32& out_u32) noexcept {
    using DF = Gcn::DataFormat;
    const auto nclass = Gcn::GetNumberClass(vsharp.GetNumberFmt());
    if (nclass == Gcn::NumberClass::Float) {
        std::array<float, 4> f{};
        if (!DecodeAttrFloat4(vsharp, elem_ptr, f)) {
            return false;
        }
        out_u32 = std::bit_cast<u32>(f[0]);
        return true;
    }
    if (vsharp.GetNumberConversion() != Gcn::NumberConversion::None) {
        return false;
    }
    const DF dfmt = Gcn::DataFormat(vsharp.data_format);
    std::array<u32, 4> raw{0u, 0u, 0u, 1u};
    const bool is_signed = (nclass == Gcn::NumberClass::Sint);
    switch (dfmt) {
    case DF::Format8:
    case DF::Format8_8:
    case DF::Format8_8_8_8: {
        const u32 comps = (dfmt == DF::Format8) ? 1u : (dfmt == DF::Format8_8) ? 2u : 4u;
        for (u32 c = 0; c < comps; ++c) {
            const u8 b = elem_ptr[c];
            raw[c] = is_signed ? static_cast<u32>(static_cast<s32>(static_cast<s8>(b)))
                               : static_cast<u32>(b);
        }
        break;
    }
    case DF::Format16:
    case DF::Format16_16:
    case DF::Format16_16_16_16: {
        const u32 comps = (dfmt == DF::Format16) ? 1u : (dfmt == DF::Format16_16) ? 2u : 4u;
        for (u32 c = 0; c < comps; ++c) {
            u16 w = 0;
            std::memcpy(&w, elem_ptr + c * 2u, 2u);
            raw[c] = is_signed ? static_cast<u32>(static_cast<s32>(static_cast<s16>(w)))
                               : static_cast<u32>(w);
        }
        break;
    }
    case DF::Format32:
    case DF::Format32_32:
    case DF::Format32_32_32:
    case DF::Format32_32_32_32: {
        const u32 comps = (dfmt == DF::Format32)          ? 1u
                          : (dfmt == DF::Format32_32)     ? 2u
                          : (dfmt == DF::Format32_32_32)  ? 3u
                                                          : 4u;
        for (u32 c = 0; c < comps; ++c) {
            std::memcpy(&raw[c], elem_ptr + c * 4u, 4u);
        }
        break;
    }
    default:
        return false;
    }
    switch (Gcn::CompSwizzle(vsharp.dst_sel_x)) {
    case Gcn::CompSwizzle::Zero:
        out_u32 = 0u;
        return true;
    case Gcn::CompSwizzle::One:
        out_u32 = 1u;
        return true;
    case Gcn::CompSwizzle::Red:
        out_u32 = raw[0];
        return true;
    case Gcn::CompSwizzle::Green:
        out_u32 = raw[1];
        return true;
    case Gcn::CompSwizzle::Blue:
        out_u32 = raw[2];
        return true;
    case Gcn::CompSwizzle::Alpha:
        out_u32 = raw[3];
        return true;
    default:
        return false;
    }
}

Gcn::Buffer MakeVec4BufferDesc(u32 num_verts, Gcn::NumberFormat nfmt) noexcept {
    Gcn::Buffer buf{};
    buf.base_address = 0;
    buf.stride = 16u;
    buf.num_records = num_verts;
    buf.dst_sel_x = u32(Gcn::CompSwizzle::Red);
    buf.dst_sel_y = u32(Gcn::CompSwizzle::Green);
    buf.dst_sel_z = u32(Gcn::CompSwizzle::Blue);
    buf.dst_sel_w = u32(Gcn::CompSwizzle::Alpha);
    buf.num_format = u32(nfmt);
    buf.data_format = u32(Gcn::DataFormat::Format32_32_32_32);
    return buf;
}

bool DwordsEqualAt(const u8* a, size_t a_len, const u8* b, size_t b_len,
                   std::span<const u32> dword_indices) noexcept {
    for (const u32 idx : dword_indices) {
        const size_t byte_off = static_cast<size_t>(idx) * 4u;
        if (byte_off + 4u > a_len || byte_off + 4u > b_len) {
            return false;
        }
        if (std::memcmp(a + byte_off, b + byte_off, 4u) != 0) {
            return false;
        }
    }
    return true;
}

bool DwordRangeEqual(const u8* a, size_t a_len, const u8* b, size_t b_len, u32 first_dw,
                     u32 count_dw) noexcept {
    const size_t off = static_cast<size_t>(first_dw) * 4u;
    const size_t bytes = static_cast<size_t>(count_dw) * 4u;
    if (off + bytes > a_len || off + bytes > b_len) {
        return false;
    }
    return std::memcmp(a + off, b + off, bytes) == 0;
}

bool CompareFfxBufferBytes(u64 stage_hash, u32 buf_ord, const u8* a, size_t a_len, const u8* b,
                           size_t b_len) noexcept {
    if (IsFfxHsHash(stage_hash)) {
        return buf_ord == 0u && DwordRangeEqual(a, a_len, b, b_len, 0u, 4u);
    }
    if (IsFfxDsHash(stage_hash)) {
        if (buf_ord == 0u) {
            static constexpr u32 kDsTessDwords[]{1u, 2u, 3u, 4u, 6u};
            return DwordsEqualAt(a, a_len, b, b_len, kDsTessDwords);
        }
        if (buf_ord == 1u) {
            static constexpr u32 kDsCommonSparse[]{19u, 23u, 27u};
            return DwordRangeEqual(a, a_len, b, b_len, 0u, 16u) &&
                   DwordsEqualAt(a, a_len, b, b_len, kDsCommonSparse) &&
                   DwordRangeEqual(a, a_len, b, b_len, 64u, 32u);
        }
        return false;
    }
    if (stage_hash == kFfxQuadVsHash) {
        if (buf_ord != 0u) {
            return false;
        }
        static constexpr u32 kVsQuadSparse[]{19u, 23u, 27u, 31u, 72u, 73u};
        return DwordRangeEqual(a, a_len, b, b_len, 0u, 8u) &&
               DwordRangeEqual(a, a_len, b, b_len, 12u, 4u) &&
               DwordsEqualAt(a, a_len, b, b_len, kVsQuadSparse) &&
               DwordRangeEqual(a, a_len, b, b_len, 80u, 4u) &&
               DwordRangeEqual(a, a_len, b, b_len, 96u, 4u);
    }
    if (IsFfxSoftPsHash(stage_hash)) {
        if (buf_ord == 0u) {
            // Note: dwords[36..39] (tint) are already normalized to (1,1,1,1) in both buffers!
            static constexpr u32 kPsCommon1Mrt[]{16u, 17u, 18u, 43u, 47u, 51u, 95u};
            if (!DwordsEqualAt(a, a_len, b, b_len, kPsCommon1Mrt)) {
                return false;
            }
            if (IsFfxSoft2MrtPsHash(stage_hash) &&
                !DwordRangeEqual(a, a_len, b, b_len, 88u, 4u)) {
                return false;
            }
            return true;
        }
        if (buf_ord == 1u) {
            return DwordRangeEqual(a, a_len, b, b_len, 4u, 16u);
        }
        return false;
    }
    return a_len == b_len && (a_len == 0u || std::memcmp(a, b, a_len) == 0);
}

} // namespace

VfxUiBatcher::VfxUiBatcher()
    : small_staging_map(kCpuStagingSlots), coarse_epoch_256k(kCoarseBuckets, 0u) {
    open.sf2404_cb_dwords.reserve(4096);
    for (auto& a : open.attrs) {
        a.bytes.reserve(8192);
    }
    for (auto& st : open.stages) {
        st.user_data.reserve(16);
        st.flattened_ud_buf.reserve(64);
        st.images.reserve(8);
        st.samplers.reserve(4);
        for (auto& b : st.bufs) {
            b.bytes.reserve(512);
        }
    }
}

bool VfxUiBatcher::Enabled() noexcept {
    static const bool on = [] {
        const char* env = std::getenv("BB_VFX_UI_BATCH");
        return !(env && env[0] == '0');
    }();
    return on;
}

bool VfxUiBatcher::IsBatchSafeTask(Rasterizer::OrderedTask task, const void* data,
                                   u32 size) noexcept {
    if (task == &BbCp::Tasks::DirectVkReleaseLabelsBatchTask ||
        task == &BbCp::Tasks::DirectVkReleaseU32LabelsBatchTask) {
        return data != nullptr && size >= sizeof(u32);
    }
    if (task == &Rasterizer::NoteCommandWriteTask || task == &BbCp::Tasks::WriteDataTask) {
        return data != nullptr && size >= sizeof(u64) * 2u;
    }
    if (task != &BbCp::Tasks::DirectVkCopyBatchTask || !data || size < sizeof(u32)) {
        return false;
    }
    u32 count = 0;
    std::memcpy(&count, data, sizeof(u32));
    if (count == 0u || count > 32u ||
        size < sizeof(u32) + count * sizeof(Rasterizer::DirectCopyRegion)) {
        return false;
    }
    const auto* raw = static_cast<const u8*>(data) + sizeof(u32);
    for (u32 i = 0; i < count; ++i) {
        Rasterizer::DirectCopyRegion reg{};
        std::memcpy(&reg, raw + i * sizeof(Rasterizer::DirectCopyRegion), sizeof(reg));
        if (reg.byte_size == 0u) {
            continue;
        }
        if (reg.reserved != gx_direct_vk::kCopyFlagKnownCpuSource) {
            return false;
        }
    }
    return true;
}

void VfxUiBatcher::RecomputeLargeStagingBounds() noexcept {
    VAddr min_v = ~VAddr{0};
    VAddr max_v = 0;
    u32 active = 0;
    for (const auto& s : large_staging) {
        if (s.byte_size == 0u) {
            continue;
        }
        min_v = std::min(min_v, s.dst_va);
        max_v = std::max(max_v, s.dst_va + s.byte_size);
        ++active;
    }
    large_min_va = min_v;
    large_max_va = max_v;
    large_active_count = active;
}

void VfxUiBatcher::NoteStageACpuCopyBatch(const u32* raw_regions, u32 count) noexcept {
    if (!raw_regions || count == 0u || count > 32u || !Enabled()) {
        return;
    }
    for (u32 i = 0; i < count; ++i) {
        Rasterizer::DirectCopyRegion reg{};
        std::memcpy(&reg, raw_regions + i * 6u, sizeof(reg));
        if (!reg.dst_va || reg.byte_size == 0u) {
            continue;
        }
        InvalidateCpuShadow(reg.dst_va, reg.byte_size);
        if (reg.reserved != gx_direct_vk::kCopyFlagKnownCpuSource || !reg.src_va ||
            !IsReadableCpuSourceRange(reg.src_va, reg.byte_size)) {
            continue;
        }
        const u32 stamp = ++staging_clock;
        if (reg.byte_size <= kSmallStagingMaxBytes) {
            const size_t set_base = SmallStagingSet(reg.dst_va >> 12) * kCpuStagingWays;
            size_t victim = set_base;
            for (size_t w = 0; w < kCpuStagingWays; ++w) {
                const auto& s = small_staging_map[set_base + w];
                if (s.byte_size == 0u || s.dst_va == reg.dst_va) {
                    victim = set_base + w;
                    break;
                }
                if (s.last_use < small_staging_map[victim].last_use) {
                    victim = set_base + w;
                }
            }
            auto& slot = small_staging_map[victim];
            slot.dst_va = reg.dst_va;
            slot.src_va = reg.src_va;
            slot.byte_size = reg.byte_size;
            slot.last_use = stamp;
            slot.coarse_e0 = CoarseEpochAt(reg.dst_va);
            slot.coarse_e1 = CoarseEpochAt(reg.dst_va + reg.byte_size - 1u);
        } else {
            size_t victim = 0u;
            for (size_t s = 0; s < large_staging.size(); ++s) {
                if (large_staging[s].byte_size == 0u || large_staging[s].dst_va == reg.dst_va) {
                    victim = s;
                    break;
                }
                if (large_staging[s].last_use < large_staging[victim].last_use) {
                    victim = s;
                }
            }
            auto& slot = large_staging[victim];
            slot.dst_va = reg.dst_va;
            slot.src_va = reg.src_va;
            slot.byte_size = reg.byte_size;
            slot.last_use = stamp;
            RecomputeLargeStagingBounds();
        }
    }
}

void VfxUiBatcher::InvalidateCpuShadow(VAddr addr, u64 size) noexcept {
    if (!addr || size == 0u) {
        return;
    }
    VideoCore::DirectBuffers::ClearCpuImmutableRange(addr, size);
    const VAddr end = (size > (~0ULL - addr)) ? ~0ULL : (addr + size);
    const u64 span = end - addr;

    // 1. O(1) invalidation of `<= 4 KiB` staging hash table:
    if (span <= 32768u) {
        const VAddr p_lo =
            (addr >= kSmallStagingMaxBytes) ? ((addr - kSmallStagingMaxBytes) >> 12) : 0ULL;
        const VAddr p_hi = (end - 1u) >> 12;
        for (VAddr p = p_lo; p <= p_hi; ++p) {
            const size_t set_base = SmallStagingSet(p) * kCpuStagingWays;
            for (size_t w = 0; w < kCpuStagingWays; ++w) {
                auto& slot = small_staging_map[set_base + w];
                if (slot.byte_size != 0u && slot.dst_va < end &&
                    addr < (slot.dst_va + slot.byte_size)) {
                    slot.byte_size = 0u;
                }
            }
        }
    } else {
        const VAddr g_lo = addr >> kCoarseShift;
        const VAddr g_hi = (end - 1u) >> kCoarseShift;
        if ((g_hi - g_lo) < 512u) {
            u32 epoch = ++coarse_clock;
            if (epoch == 0u) [[unlikely]] {
                epoch = ++coarse_clock;
            }
            for (VAddr g = g_lo; g <= g_hi; ++g) {
                coarse_epoch_256k[static_cast<size_t>(g & (kCoarseBuckets - 1u))] = epoch;
            }
        } else {
            ++coarse_wipe_bias;
        }
    }

    // 2. Tight-bounded `32`-slot check for `> 4 KiB` dynamic VB/IB rings (shrinks bounds on clear):
    if (large_active_count != 0u && addr < large_max_va && end > large_min_va) {
        bool cleared_any = false;
        for (auto& slot : large_staging) {
            if (slot.byte_size == 0u) {
                continue;
            }
            if (addr < (slot.dst_va + slot.byte_size) && slot.dst_va < end) {
                slot = CpuStagingSlot{};
                cleared_any = true;
            }
        }
        if (cleared_any) {
            RecomputeLargeStagingBounds();
        }
    }
}

const u8* VfxUiBatcher::ResolveCpuReadableRange(Rasterizer& rasterizer, VAddr addr,
                                                u64 byte_size) noexcept {
    if (!addr || byte_size == 0u) {
        return nullptr;
    }
    if (VideoCore::DirectBuffers::IsTransientCueHeap(addr, byte_size) ||
        VideoCore::DirectBuffers::IsTransientAuxCpuRing(addr, byte_size)) {
        return reinterpret_cast<const u8*>(addr);
    }
    u32 best_use = 0u;
    const u8* best_ptr = nullptr;
    if (byte_size <= kSmallStagingMaxBytes) {
        const VAddr page = addr >> 12;
        for (u32 step = 0; step < 2u; ++step) {
            if (step == 1u && page == 0u) {
                break;
            }
            const size_t set_base = SmallStagingSet(page - step) * kCpuStagingWays;
            for (size_t w = 0; w < kCpuStagingWays; ++w) {
                auto& slot = small_staging_map[set_base + w];
                if (slot.byte_size == 0u || addr < slot.dst_va) {
                    continue;
                }
                const u64 sub_off = addr - slot.dst_va;
                if (sub_off >= slot.byte_size || byte_size > (slot.byte_size - sub_off)) {
                    continue;
                }
                if (slot.coarse_e0 != CoarseEpochAt(slot.dst_va) ||
                    slot.coarse_e1 != CoarseEpochAt(slot.dst_va + slot.byte_size - 1u)) [[unlikely]] {
                    slot.byte_size = 0u;
                    continue;
                }
                if (slot.last_use >= best_use) {
                    const VAddr src = slot.src_va + sub_off;
                    if (IsReadableCpuSourceRange(src, byte_size)) {
                        best_use = slot.last_use;
                        best_ptr = reinterpret_cast<const u8*>(src);
                    }
                }
            }
        }
    }
    if (large_active_count != 0u && addr >= large_min_va && addr < large_max_va &&
        byte_size <= (large_max_va - addr)) {
        for (const auto& slot : large_staging) {
            if (slot.byte_size == 0u || addr < slot.dst_va) {
                continue;
            }
            const u64 sub_off = addr - slot.dst_va;
            if (sub_off < slot.byte_size && byte_size <= (slot.byte_size - sub_off) &&
                slot.last_use >= best_use) {
                const VAddr src = slot.src_va + sub_off;
                if (IsReadableCpuSourceRange(src, byte_size)) {
                    best_use = slot.last_use;
                    best_ptr = reinterpret_cast<const u8*>(src);
                }
            }
        }
    }
    if (best_ptr) {
        return best_ptr;
    }
    if (byte_size <= LightTileBatcher::kMaxShadowBytes && rasterizer.light_tile_batcher) {
        if (const u8* sh =
                rasterizer.light_tile_batcher->LookupCpuShadow(addr, static_cast<u32>(byte_size))) {
            return sh;
        }
    }
    if (VideoCore::DirectBuffers::IsCpuReadableImmutableRange(addr, byte_size)) {
        return reinterpret_cast<const u8*>(addr);
    }
    if (!VideoCore::DirectBuffers::OverlapsNativeOwnedRange(addr, byte_size) &&
        !rasterizer.PendingWriteOverlaps(addr, byte_size) &&
        IsReadableCpuSourceRange(addr, byte_size)) {
        return reinterpret_cast<const u8*>(addr);
    }
    return nullptr;
}

bool VfxUiBatcher::TryBatchDraw(Rasterizer& rasterizer, const GraphicsPipeline* pipeline,
                                const PreparedDraw* /*used_prepared*/, bool is_indexed,
                                u32 index_offset) {
    if (!Enabled() || flushing || !pipeline || pipeline->IsCompute() ||
        !rasterizer.constant_ring || BbToggle::Disabled(BbToggle::ConstantRing) ||
        !rasterizer.instance.IsVertexInputDynamicState()) {
        if (HasPendingDraws()) {
            Flush(rasterizer);
        }
        return false;
    }

    auto& lp = *rasterizer.liverpool;
    const auto& regs = lp.regs;
    const u32 num_indices = regs.num_indices;
    if (num_indices == 0u || num_indices > kMaxSmallDrawIndices ||
        regs.num_instances.NumInstances() != 1u || regs.depth_render_control.depth_clear_enable ||
        regs.depth_render_control.stencil_clear_enable ||
        (rasterizer.occlusion && rasterizer.occlusion->Events() != 0)) {
        if (HasPendingDraws()) {
            Flush(rasterizer);
        }
        return false;
    }

    const auto stages_span = pipeline->GetStages();
    const Shader::Info* vs_stage = stages_span[u32(Shader::SwStage::Vertex)];
    const Shader::Info* gs_stage = stages_span[u32(Shader::SwStage::Geometry)];
    const Shader::Info* hs_stage = stages_span[u32(Shader::SwStage::TessellationControl)];
    const Shader::Info* ds_stage = stages_span[u32(Shader::SwStage::TessellationEval)];
    const Shader::Info* ps_stage = stages_span[u32(Shader::SwStage::Fragment)];
    if (!vs_stage || gs_stage) {
        if (HasPendingDraws()) {
            Flush(rasterizer);
        }
        return false;
    }

    const u64 vs_hash = vs_stage->pgm_hash;
    // Instant ~2 ns reject for all non-VFX / non-Scaleform draws (95%+ of engine draws):
    if (!IsTargetVsOrLsHash(vs_hash)) {
        if (HasPendingDraws()) {
            Flush(rasterizer);
        }
        return false;
    }

    const auto& fetch = pipeline->GetFetchShader();
    if (!fetch || fetch->attributes.empty() || fetch->attributes.size() > kMaxAttrs) {
        if (HasPendingDraws()) {
            Flush(rasterizer);
        }
        return false;
    }
    for (const auto& attr : fetch->attributes) {
        if (attr.GetStepRate() != Shader::Gcn::VertexAttribute::InstanceIdType::None) {
            if (HasPendingDraws()) {
                Flush(rasterizer);
            }
            return false;
        }
    }
    if (fetch->instance_offset_sgpr != -1 &&
        static_cast<size_t>(fetch->instance_offset_sgpr) < vs_stage->UserData().size() &&
        vs_stage->UserData()[fetch->instance_offset_sgpr] != 0u) {
        if (HasPendingDraws()) {
            Flush(rasterizer);
        }
        return false;
    }

    for (const auto* st : stages_span) {
        if (!st) {
            continue;
        }
        if (st->uses_dma || st->has_storage_images) {
            if (HasPendingDraws()) {
                Flush(rasterizer);
            }
            return false;
        }
        u32 ns_count = 0u;
        for (const auto& desc : st->buffers) {
            if (desc.IsSpecial()) {
                continue;
            }
            if (desc.is_written || desc.is_formatted) {
                if (HasPendingDraws()) {
                    Flush(rasterizer);
                }
                return false;
            }
            if (++ns_count > kMaxNonSpecialBufs) {
                if (HasPendingDraws()) {
                    Flush(rasterizer);
                }
                return false;
            }
        }
    }

    const auto& key = pipeline->GetGraphicsKey();
    const Gcn::PrimitiveType prim = regs.primitive_type;
    if (prim != key.prim_type) {
        if (HasPendingDraws()) {
            Flush(rasterizer);
        }
        return false;
    }
    u32 patch_cp = 1u;
    switch (prim) {
    case Gcn::PrimitiveType::TriangleList:
    case Gcn::PrimitiveType::RectList:
        if ((num_indices % 3u) != 0u) {
            if (HasPendingDraws()) {
                Flush(rasterizer);
            }
            return false;
        }
        break;
    case Gcn::PrimitiveType::QuadList:
        if ((num_indices % 4u) != 0u) {
            if (HasPendingDraws()) {
                Flush(rasterizer);
            }
            return false;
        }
        break;
    case Gcn::PrimitiveType::TriangleStrip:
        if (num_indices < 3u) {
            if (HasPendingDraws()) {
                Flush(rasterizer);
            }
            return false;
        }
        break;
    case Gcn::PrimitiveType::PatchPrimitive:
        patch_cp = std::max<u32>(key.patch_control_points, 1u);
        if (!hs_stage || !ds_stage || (num_indices % patch_cp) != 0u) {
            if (HasPendingDraws()) {
                Flush(rasterizer);
            }
            return false;
        }
        break;
    default:
        if (HasPendingDraws()) {
            Flush(rasterizer);
        }
        return false;
    }

    ++stat_candidates_seen;

    // Classify candidate mode (only exact Scaleform GFx & GXFfx soft-particle targets):
    Mode cand_mode = Mode::None;
    const u64 ps_hash = ps_stage ? ps_stage->pgm_hash : 0ULL;
    if (!hs_stage && !ds_stage && vs_hash == kScaleform2404VsHash &&
        fetch->attributes.size() == 2u && fetch->attributes[0].semantic == 0u &&
        fetch->attributes[1].semantic == 1u && Shader::Backend::SPIRV::NativeShadersEnabled() &&
        !Shader::Backend::SPIRV::NativeShadersCustomRange()) {
        cand_mode = Mode::Scaleform2404;
    } else if (!hs_stage && !ds_stage && vs_hash == kScaleformE94BVsHash &&
               fetch->attributes.size() == 3u && fetch->attributes[0].semantic == 0u &&
               fetch->attributes[1].semantic == 1u && fetch->attributes[2].semantic == 2u) {
        cand_mode = Mode::ScaleformE94B;
    } else if (!hs_stage && !ds_stage && vs_hash == kScaleformA400VsHash &&
               fetch->attributes.size() == 2u && fetch->attributes[0].semantic == 0u &&
               fetch->attributes[1].semantic == 1u) {
        cand_mode = Mode::ScaleformA400;
    } else if (ps_stage && IsFfxSoftPsHash(ps_hash) &&
               ((vs_hash == kFfxTessLsHash && hs_stage && ds_stage &&
                 IsFfxHsHash(hs_stage->pgm_hash) && IsFfxDsHash(ds_stage->pgm_hash)) ||
                (vs_hash == kFfxQuadVsHash && !hs_stage && !ds_stage)) &&
               fetch->attributes.size() >= 2u && fetch->attributes[1].semantic == 1u) {
        cand_mode = Mode::FfxTintBaked;
    }
    if (cand_mode == Mode::None) {
        ++fb_layout;
        if (HasPendingDraws()) {
            Flush(rasterizer);
        }
        return false;
    }

    StageBChunks raw_chunks{};
    CaptureStageBChunks(regs, raw_chunks);
    StageBChunks norm_chunks = raw_chunks;
    NormalizeStageBChunksForRun(norm_chunks);

    // Check if an open batch must flush before inspecting CBs/VBs due to pipeline / state change:
    if (open.active) {
        StageBChunks open_norm = open.base_chunks;
        NormalizeStageBChunksForRun(open_norm);
        const bool same_pipe_state =
            open.pipeline == pipeline && open.mode == cand_mode && open.prim_type == prim &&
            std::memcmp(&open.cb_extent, &lp.last_cb_extent, sizeof(open.cb_extent)) == 0 &&
            open.db_extent.raw == lp.last_db_extent.raw &&
            open.cb_uid == lp.last_cb_uid && open.db_uid == lp.last_db_uid &&
            std::memcmp(open_norm.data(), norm_chunks.data(), sizeof(StageBChunks)) == 0;
        if (!same_pipe_state) {
            ++fb_regs;
            Flush(rasterizer);
        } else if (open.num_verts + num_indices + 4u > kMaxMergedVertices ||
                   (cand_mode == Mode::Scaleform2404 &&
                    open.sf2404_quads >= kMaxScaleform2404Quads)) {
            Flush(rasterizer);
        }
    }

    // Read stage descriptors & constant buffers on Stage A (zero locks!):
    struct LocalBuf {
        u32 sharp_idx = 0;
        const u8* ptr = nullptr;
        u32 size = 0;
    };
    struct LocalStage {
        const Shader::Info* info = nullptr;
        u32 num_bufs = 0;
        std::array<LocalBuf, kMaxNonSpecialBufs> bufs{};
        std::array<Gcn::Image, 24> images{};
        u32 num_images = 0;
        std::array<Gcn::Sampler, 16> samplers{};
        u32 num_samplers = 0;
    };
    std::array<LocalStage, Shader::MaxStageTypes> local_stages{};
    u32 num_active_stages = 0u;
    // Note: `Pipeline::GetStages()` is ordered `[Fragment(0), TessControl(1), TessEval(2),
    // Vertex(3), Geometry(4)]`, so `VS/LS` is NOT `slot 0` whenever `PS` is active!
    u32 vs_slot = ~0u;
    u32 hs_slot = ~0u;
    u32 ds_slot = ~0u;
    u32 ps_slot = ~0u;

    for (const auto* st : stages_span) {
        if (!st) {
            continue;
        }
        if (st->images.size() > 24u || st->samplers.size() > 16u) {
            ++fb_layout;
            if (open.active) {
                Flush(rasterizer);
            }
            return false;
        }
        const u32 slot = num_active_stages++;
        if (st == vs_stage) {
            vs_slot = slot;
        }
        if (st == hs_stage) {
            hs_slot = slot;
        }
        if (st == ds_stage) {
            ds_slot = slot;
        }
        if (st == ps_stage) {
            ps_slot = slot;
        }
        auto& ls = local_stages[slot];
        ls.info = st;
        for (u32 idx = 0; idx < st->buffers.size(); ++idx) {
            const auto& desc = st->buffers[idx];
            if (desc.IsSpecial()) {
                continue;
            }
            const Gcn::Buffer sharp = desc.GetSharp(*st);
            if (!sharp.base_address || sharp.GetSize() == 0u || sharp.swizzle_enable ||
                sharp.add_tid_enable) {
                ++fb_cb;
                if (open.active) {
                    Flush(rasterizer);
                }
                return false;
            }
            const u64 clamped_sz = VideoCore::DirectBuffers::ClampNativeGpuReadSize(
                sharp.base_address, sharp.GetSize());
            if (clamped_sz == 0u || clamped_sz > 16384u) {
                ++fb_cb;
                if (open.active) {
                    Flush(rasterizer);
                }
                return false;
            }
            const u8* ptr = ResolveCpuReadableRange(rasterizer, sharp.base_address, clamped_sz);
            if (!ptr) {
                ++fb_cb;
                if (open.active) {
                    Flush(rasterizer);
                }
                return false;
            }
            ls.bufs[ls.num_bufs++] = LocalBuf{
                .sharp_idx = idx,
                .ptr = ptr,
                .size = static_cast<u32>(clamped_sz),
            };
        }
        for (const auto& img_desc : st->images) {
            ls.images[ls.num_images++] = img_desc.GetSharp(*st);
        }
        for (const auto& samp_desc : st->samplers) {
            ls.samplers[ls.num_samplers++] = samp_desc.GetSharp(*st);
        }
    }

    // Validate mode-specific buffer counts & sizes using exact `SwStage` slots:
    std::array<float, 4> ffx_tint{1.0f, 1.0f, 1.0f, 1.0f};
    bool mode_bufs_ok = (vs_slot != ~0u);
    if (mode_bufs_ok && cand_mode == Mode::Scaleform2404) {
        mode_bufs_ok = local_stages[vs_slot].num_bufs == 1u &&
                       local_stages[vs_slot].bufs[0].size >= 96u;
    } else if (mode_bufs_ok && cand_mode == Mode::ScaleformE94B) {
        mode_bufs_ok = local_stages[vs_slot].num_bufs == 1u &&
                       local_stages[vs_slot].bufs[0].size >= 64u;
    } else if (mode_bufs_ok && cand_mode == Mode::ScaleformA400) {
        mode_bufs_ok = local_stages[vs_slot].num_bufs == 1u &&
                       local_stages[vs_slot].bufs[0].size >= 32u;
    } else if (mode_bufs_ok && cand_mode == Mode::FfxTintBaked) {
        if (ps_slot == ~0u) {
            mode_bufs_ok = false;
        } else {
            const auto& ps_ls = local_stages[ps_slot];
            bool ok = ps_ls.num_bufs == 2u && ps_ls.bufs[0].size >= 384u &&
                      ps_ls.bufs[1].size >= 80u;
            if (ok && vs_hash == kFfxTessLsHash) {
                ok = num_active_stages == 4u && hs_slot != ~0u && ds_slot != ~0u &&
                     local_stages[vs_slot].num_bufs == 0u &&
                     (local_stages[hs_slot].num_bufs == 0u ||
                      (local_stages[hs_slot].num_bufs == 1u &&
                       local_stages[hs_slot].bufs[0].size >= 16u)) &&
                     local_stages[ds_slot].num_bufs == 2u &&
                     local_stages[ds_slot].bufs[0].size >= 28u &&
                     local_stages[ds_slot].bufs[1].size >= 384u;
            } else if (ok && vs_hash == kFfxQuadVsHash) {
                ok = num_active_stages == 2u && local_stages[vs_slot].num_bufs == 1u &&
                     local_stages[vs_slot].bufs[0].size >= 400u;
            } else {
                ok = false;
            }
            if (ok) {
                std::memcpy(ffx_tint.data(), ps_ls.bufs[0].ptr + 36u * 4u, 16u);
            }
            mode_bufs_ok = ok;
        }
    }
    if (!mode_bufs_ok) {
        ++fb_cb;
        if (open.active) {
            Flush(rasterizer);
        }
        return false;
    }

    // Check T#/S# & CB compatibility with open batch (if active):
    if (open.active) {
        bool compat = (open.num_stages == num_active_stages && open.vs_slot == vs_slot &&
                       open.ffx_ps_slot == ps_slot);
        for (u32 s = 0; compat && s < num_active_stages; ++s) {
            const auto& ost = open.stages[s];
            const auto& lst = local_stages[s];
            if (ost.info != lst.info || ost.pgm_base != lst.info->ProgramBase() ||
                ost.images.size() != lst.num_images || ost.samplers.size() != lst.num_samplers ||
                ost.num_bufs != lst.num_bufs) {
                compat = false;
                break;
            }
            if (lst.num_images > 0u &&
                std::memcmp(ost.images.data(), lst.images.data(),
                            lst.num_images * sizeof(Gcn::Image)) != 0) {
                compat = false;
                break;
            }
            if (lst.num_samplers > 0u &&
                std::memcmp(ost.samplers.data(), lst.samplers.data(),
                            lst.num_samplers * sizeof(Gcn::Sampler)) != 0) {
                compat = false;
                break;
            }
            for (u32 b = 0; b < lst.num_bufs; ++b) {
                if (ost.bufs[b].sharp_idx != lst.bufs[b].sharp_idx) {
                    compat = false;
                    break;
                }
                // VS CB0 in Scaleform modes is packed/pre-transformed per draw!
                if (s == vs_slot && b == 0u &&
                    (cand_mode == Mode::Scaleform2404 || cand_mode == Mode::ScaleformE94B ||
                     cand_mode == Mode::ScaleformA400)) {
                    continue;
                }
                if (cand_mode == Mode::FfxTintBaked) {
                    if (s == ps_slot && b == 0u) {
                        static constexpr u32 kPsCommon1Mrt[]{16u, 17u, 18u, 43u, 47u, 51u, 95u};
                        if (!DwordsEqualAt(ost.bufs[b].bytes.data(), ost.bufs[b].bytes.size(),
                                           lst.bufs[b].ptr, lst.bufs[b].size, kPsCommon1Mrt)) {
                            compat = false;
                            break;
                        }
                        if (IsFfxSoft2MrtPsHash(ps_hash) &&
                            !DwordRangeEqual(ost.bufs[b].bytes.data(), ost.bufs[b].bytes.size(),
                                             lst.bufs[b].ptr, lst.bufs[b].size, 88u, 4u)) {
                            compat = false;
                            break;
                        }
                        continue;
                    }
                    if (!CompareFfxBufferBytes(lst.info->pgm_hash, b, ost.bufs[b].bytes.data(),
                                               ost.bufs[b].bytes.size(), lst.bufs[b].ptr,
                                               lst.bufs[b].size)) {
                        compat = false;
                        break;
                    }
                    continue;
                }
                if (ost.bufs[b].bytes.size() != lst.bufs[b].size ||
                    std::memcmp(ost.bufs[b].bytes.data(), lst.bufs[b].ptr, lst.bufs[b].size) != 0) {
                    compat = false;
                    break;
                }
            }
        }
        if (!compat) {
            ++fb_cb;
            Flush(rasterizer);
        }
    }

    // Resolve index stream into expanded vertex indices:
    u32 vertex_offset = regs.index_offset;
    if (vertex_offset == 0u && fetch->vertex_offset_sgpr != -1 &&
        static_cast<size_t>(fetch->vertex_offset_sgpr) < vs_stage->UserData().size()) {
        vertex_offset = vs_stage->UserData()[fetch->vertex_offset_sgpr];
    }

    std::array<u32, kMaxSmallDrawIndices> exp_indices{};
    std::array<u8, kMaxSmallDrawIndices> exp_restart{};
    u32 max_vtx_idx = 0u;
    bool any_valid_idx = false;
    const bool prim_restart_en = is_indexed && regs.enable_primitive_restart != 0u;
    const u32 restart_val = regs.primitive_restart_index;

    if (is_indexed) {
        const bool is_u16 = (regs.index_buffer_type.index_type == Gcn::IndexType::Index16);
        const u32 idx_sz = is_u16 ? 2u : 4u;
        const VAddr ib_addr =
            regs.index_base_address.Address<VAddr>() + static_cast<u64>(index_offset) * idx_sz;
        const u64 ib_bytes = static_cast<u64>(num_indices) * idx_sz;
        const u8* ib_ptr = ResolveCpuReadableRange(rasterizer, ib_addr, ib_bytes);
        if (!ib_ptr) {
            ++fb_ib;
            if (open.active) {
                Flush(rasterizer);
            }
            return false;
        }
        const u32 restart_cmp = is_u16 ? (restart_val & 0xFFFFu) : restart_val;
        for (u32 i = 0; i < num_indices; ++i) {
            u32 raw_idx = 0u;
            if (is_u16) {
                u16 v16 = 0;
                std::memcpy(&v16, ib_ptr + i * 2u, 2u);
                raw_idx = v16;
            } else {
                std::memcpy(&raw_idx, ib_ptr + i * 4u, 4u);
            }
            if (prim_restart_en && raw_idx == restart_cmp) {
                exp_restart[i] = 1u;
                continue;
            }
            const u32 vtx = raw_idx + vertex_offset;
            exp_indices[i] = vtx;
            max_vtx_idx = any_valid_idx ? std::max(max_vtx_idx, vtx) : vtx;
            any_valid_idx = true;
        }
    } else {
        for (u32 i = 0; i < num_indices; ++i) {
            const u32 vtx = vertex_offset + i;
            exp_indices[i] = vtx;
            max_vtx_idx = any_valid_idx ? std::max(max_vtx_idx, vtx) : vtx;
            any_valid_idx = true;
        }
    }

    if (!any_valid_idx) {
        if (open.active) {
            Flush(rasterizer);
        }
        return false;
    }

    // Resolve all vertex attribute V# streams on Stage A:
    const u32 num_attrs = static_cast<u32>(fetch->attributes.size());
    std::array<Gcn::Buffer, kMaxAttrs> attr_sharps{};
    std::array<const u8*, kMaxAttrs> attr_ptrs{};
    std::array<u32, kMaxAttrs> attr_elem_bytes{};
    std::array<u32, kMaxAttrs> attr_src_strides{};

    for (u32 a = 0; a < num_attrs; ++a) {
        const Gcn::Buffer vsharp = fetch->attributes[a].GetSharp(*vs_stage);
        const u32 fmt_bytes = DataFormatByteSize(Gcn::DataFormat(vsharp.data_format));
        const u32 src_stride = vsharp.GetStride();
        if (!vsharp.base_address || fmt_bytes == 0u || fmt_bytes > 16u || vsharp.swizzle_enable ||
            vsharp.add_tid_enable || (src_stride == 0u && max_vtx_idx > 0u) || src_stride > 256u) {
            ++fb_vb;
            if (open.active) {
                Flush(rasterizer);
            }
            return false;
        }
        if (vsharp.num_records != 0u && src_stride != 0u && max_vtx_idx >= vsharp.num_records) {
            ++fb_vb;
            if (open.active) {
                Flush(rasterizer);
            }
            return false;
        }
        const u64 needed_bytes = static_cast<u64>(max_vtx_idx) * src_stride + fmt_bytes;
        const u8* vb_ptr = ResolveCpuReadableRange(rasterizer, vsharp.base_address, needed_bytes);
        if (!vb_ptr) {
            ++fb_vb;
            if (open.active) {
                Flush(rasterizer);
            }
            return false;
        }
        attr_sharps[a] = vsharp;
        attr_ptrs[a] = vb_ptr;
        attr_elem_bytes[a] = fmt_bytes;
        attr_src_strides[a] = src_stride != 0u ? src_stride : fmt_bytes;
    }

    // De-interleave each attribute into a compact 4-byte-aligned `<= 16 B` stream in ConstantRing:
    std::array<Gcn::Buffer, kMaxAttrs> out_sharps = attr_sharps;
    std::array<u32, kMaxAttrs> out_strides{};
    for (u32 a = 0; a < num_attrs; ++a) {
        out_strides[a] = Common::AlignUp(attr_elem_bytes[a], 4u);
        out_sharps[a].stride = out_strides[a];
    }

    bool mode_attrs_ok = true;
    if (cand_mode == Mode::Scaleform2404) {
        if (Gcn::GetNumberClass(attr_sharps[1].GetNumberFmt()) != Gcn::NumberClass::Float ||
            attr_sharps[1].GetNumberConversion() != Gcn::NumberConversion::None) {
            mode_attrs_ok = false;
        } else {
            out_sharps[1] = MakeVec4BufferDesc(0u, Gcn::NumberFormat::Float);
            out_strides[1] = 16u;
        }
    } else if (cand_mode == Mode::ScaleformE94B) {
        if (Gcn::GetNumberClass(attr_sharps[0].GetNumberFmt()) != Gcn::NumberClass::Float ||
            Gcn::GetNumberClass(attr_sharps[1].GetNumberFmt()) != Gcn::NumberClass::Float ||
            Gcn::GetNumberClass(attr_sharps[2].GetNumberFmt()) != Gcn::NumberClass::Float ||
            attr_sharps[0].GetNumberConversion() != Gcn::NumberConversion::None ||
            attr_sharps[1].GetNumberConversion() != Gcn::NumberConversion::None ||
            attr_sharps[2].GetNumberConversion() != Gcn::NumberConversion::None) {
            mode_attrs_ok = false;
        } else {
            for (u32 a = 0; a < 3u; ++a) {
                out_sharps[a] = MakeVec4BufferDesc(0u, Gcn::NumberFormat::Float);
                out_strides[a] = 16u;
            }
        }
    } else if (cand_mode == Mode::ScaleformA400) {
        const auto nc1 = Gcn::GetNumberClass(attr_sharps[1].GetNumberFmt());
        if (Gcn::GetNumberClass(attr_sharps[0].GetNumberFmt()) != Gcn::NumberClass::Float ||
            attr_sharps[0].GetNumberConversion() != Gcn::NumberConversion::None ||
            attr_sharps[1].GetNumberConversion() != Gcn::NumberConversion::None) {
            mode_attrs_ok = false;
        } else {
            const Gcn::NumberFormat out_nf1 =
                (nc1 == Gcn::NumberClass::Sint)   ? Gcn::NumberFormat::Sint
                : (nc1 == Gcn::NumberClass::Uint) ? Gcn::NumberFormat::Uint
                                                  : Gcn::NumberFormat::Float;
            out_sharps[0] = MakeVec4BufferDesc(0u, Gcn::NumberFormat::Float);
            out_strides[0] = 16u;
            out_sharps[1] = MakeVec4BufferDesc(0u, out_nf1);
            out_strides[1] = 16u;
        }
    } else if (cand_mode == Mode::FfxTintBaked) {
        const auto ds1 = attr_sharps[1].DstSelect();
        if (Gcn::GetNumberClass(attr_sharps[1].GetNumberFmt()) != Gcn::NumberClass::Float ||
            attr_sharps[1].GetNumberConversion() != Gcn::NumberConversion::None ||
            ds1.r != Gcn::CompSwizzle::Red || ds1.g != Gcn::CompSwizzle::Green ||
            ds1.b != Gcn::CompSwizzle::Blue || ds1.a != Gcn::CompSwizzle::Alpha) {
            mode_attrs_ok = false;
        } else {
            out_sharps[1] = MakeVec4BufferDesc(0u, Gcn::NumberFormat::Float);
            out_strides[1] = 16u;
        }
    }
    if (!mode_attrs_ok) {
        ++fb_vb;
        if (open.active) {
            Flush(rasterizer);
        }
        return false;
    }

    if (open.active) {
        bool vsharp_compat = (open.mode == cand_mode && open.num_attrs == num_attrs);
        for (u32 a = 0; vsharp_compat && a < num_attrs; ++a) {
            const auto& oa = open.attrs[a].out_vsharp;
            const auto& ca = out_sharps[a];
            if (oa.stride != ca.stride || oa.data_format != ca.data_format ||
                oa.num_format != ca.num_format || oa.dst_sel_x != ca.dst_sel_x ||
                oa.dst_sel_y != ca.dst_sel_y || oa.dst_sel_z != ca.dst_sel_z ||
                oa.dst_sel_w != ca.dst_sel_w) {
                vsharp_compat = false;
            }
        }
        if (!vsharp_compat) {
            ++fb_vb;
            Flush(rasterizer);
        }
    }

    const u32 cur_sf2404_quad_idx = open.active ? open.sf2404_quads : 0u;
    const float* sf_cb_f32 =
        (vs_slot != ~0u && local_stages[vs_slot].num_bufs >= 1u)
            ? reinterpret_cast<const float*>(local_stages[vs_slot].bufs[0].ptr)
            : nullptr;
    const u32 sf_cb_dwords =
        (vs_slot != ~0u && local_stages[vs_slot].num_bufs >= 1u)
            ? (local_stages[vs_slot].bufs[0].size >> 2u)
            : 0u;

    auto encode_vertex_attr = [&](u32 vtx_idx, u32 attr_idx,
                                  std::array<u8, 64>& dst_rec) noexcept -> bool {
        const u8* src_elem =
            attr_ptrs[attr_idx] + static_cast<size_t>(vtx_idx) * attr_src_strides[attr_idx];
        if (cand_mode == Mode::Scaleform2404 && attr_idx == 1u) {
            std::array<float, 4> p{};
            if (!DecodeAttrFloat4(attr_sharps[1], src_elem, p)) {
                return false;
            }
            const float out_v[4]{
                p[0],
                p[1],
                0.0f,
                -static_cast<float>(cur_sf2404_quad_idx + 1u),
            };
            std::memcpy(dst_rec.data(), out_v, 16u);
            return true;
        }
        if (cand_mode == Mode::ScaleformE94B) {
            std::array<float, 4> f{};
            if (!DecodeAttrFloat4(attr_sharps[attr_idx], src_elem, f)) {
                return false;
            }
            float out_v[4]{};
            if (attr_idx == 0u) {
                for (u32 c = 0; c < 4u; ++c) {
                    out_v[c] = f[c] * sf_cb_f32[4u + c] + sf_cb_f32[c];
                }
            } else if (attr_idx == 1u) {
                out_v[0] = f[0];
                out_v[1] = f[1];
                out_v[2] = 0.0f;
                out_v[3] = 1.0f;
            } else {
                const float px = f[0];
                const float py = f[1];
                out_v[0] = sf_cb_f32[8u] * px + sf_cb_f32[9u] * py + sf_cb_f32[11u];
                out_v[1] = sf_cb_f32[12u] * px + sf_cb_f32[13u] * py + sf_cb_f32[15u];
                out_v[2] = 0.0f;
                out_v[3] = 1.0f;
            }
            std::memcpy(dst_rec.data(), out_v, 16u);
            return true;
        }
        if (cand_mode == Mode::ScaleformA400) {
            if (attr_idx == 1u) {
                std::memset(dst_rec.data(), 0, 16u);
                return true;
            }
            std::array<float, 4> p{};
            if (!DecodeAttrFloat4(attr_sharps[0], src_elem, p)) {
                return false;
            }
            const u8* src_m =
                attr_ptrs[1] + static_cast<size_t>(vtx_idx) * attr_src_strides[1];
            u32 m_idx = 0u;
            if (!DecodeAttrBitcastU32X(attr_sharps[1], src_m, m_idx)) {
                return false;
            }
            const float base_r = static_cast<float>(m_idx << 1u) + 0.1f;
            if (!(base_r >= 0.0f && base_r < 16384.0f)) {
                return false;
            }
            const u32 r0 = static_cast<u32>(static_cast<s32>(base_r));
            const u32 r1 = static_cast<u32>(static_cast<s32>(base_r + 1.0f));
            if (r1 * 4u + 4u > sf_cb_dwords) {
                return false;
            }
            const float* row0 = sf_cb_f32 + r0 * 4u;
            const float* row1 = sf_cb_f32 + r1 * 4u;
            const float out_v[4]{
                row0[0] * p[0] + row0[1] * p[1] + row0[3],
                row1[0] * p[0] + row1[1] * p[1] + row1[3],
                0.0f,
                1.0f,
            };
            std::memcpy(dst_rec.data(), out_v, 16u);
            return true;
        }
        if (cand_mode == Mode::FfxTintBaked && attr_idx == 1u) {
            std::array<float, 4> c{};
            if (!DecodeAttrFloat4(attr_sharps[1], src_elem, c)) {
                return false;
            }
            // Both 0x4cd21a0f and 0x41ae600f swizzle vs_in_attr1.zyxw into fs_in_attr0_p:
            const float out_v[4]{
                c[0] * ffx_tint[2],
                c[1] * ffx_tint[1],
                c[2] * ffx_tint[0],
                c[3] * ffx_tint[3],
            };
            std::memcpy(dst_rec.data(), out_v, 16u);
            return true;
        }

        const u32 stride = out_strides[attr_idx];
        const u32 elem_sz = attr_elem_bytes[attr_idx];
        std::memcpy(dst_rec.data(), src_elem, elem_sz);
        if (stride > elem_sz) {
            std::memset(dst_rec.data() + elem_sz, 0, stride - elem_sz);
        }
        return true;
    };

    // Pre-validate all referenced vertices before mutating `open`:
    for (u32 i = 0; i < num_indices; ++i) {
        if (exp_restart[i]) {
            continue;
        }
        std::array<u8, 64> tmp{};
        for (u32 a = 0; a < num_attrs; ++a) {
            if (!encode_vertex_attr(exp_indices[i], a, tmp)) {
                ++fb_vb;
                if (open.active) {
                    Flush(rasterizer);
                }
                return false;
            }
        }
    }

    // Initialize `open` if starting a new sequential batch:
    if (!open.active) {
        open.Reset();
        open.active = true;
        open.mode = cand_mode;
        open.pipeline = pipeline;
        open.prim_type = prim;
        open.patch_cp_count = patch_cp;
        open.num_attrs = num_attrs;
        open.num_stages = num_active_stages;
        open.vs_slot = vs_slot;
        open.ffx_ps_slot = ps_slot;
        open.base_chunks = raw_chunks;
        open.cb_extent = lp.last_cb_extent;
        open.db_extent = lp.last_db_extent;
        open.cb_uid = lp.last_cb_uid;
        open.db_uid = lp.last_db_uid;

        for (u32 a = 0; a < num_attrs; ++a) {
            open.attrs[a].out_vsharp = out_sharps[a];
            open.attrs[a].stride = out_strides[a];
        }
        for (u32 s = 0; s < num_active_stages; ++s) {
            auto& dst_st = open.stages[s];
            const auto& src_st = local_stages[s];
            dst_st.info = src_st.info;
            dst_st.pgm_hash = src_st.info->pgm_hash;
            dst_st.pgm_base = src_st.info->ProgramBase();
            const auto ud = src_st.info->UserData();
            const auto flat = src_st.info->FlatUserData();
            dst_st.user_data.assign(ud.begin(), ud.end());
            dst_st.flattened_ud_buf.assign(flat.begin(), flat.end());
            if (s == vs_slot && fetch->vertex_offset_sgpr != -1) {
                const size_t sgpr = static_cast<size_t>(fetch->vertex_offset_sgpr);
                if (sgpr < dst_st.user_data.size()) {
                    dst_st.user_data[sgpr] = 0u;
                }
                if (sgpr < dst_st.flattened_ud_buf.size()) {
                    dst_st.flattened_ud_buf[sgpr] = 0u;
                }
            }
            dst_st.num_bufs = src_st.num_bufs;
            for (u32 b = 0; b < src_st.num_bufs; ++b) {
                dst_st.bufs[b].sharp_idx = src_st.bufs[b].sharp_idx;
                if (s == vs_slot && b == 0u && cand_mode == Mode::ScaleformE94B) {
                    static constexpr float kIdentityE94B[16]{
                        0.0f, 0.0f, 0.0f, 0.0f, // cx_add
                        1.0f, 1.0f, 1.0f, 1.0f, // cx_mul
                        1.0f, 0.0f, 0.0f, 0.0f, // pos_row0
                        0.0f, 1.0f, 0.0f, 0.0f, // pos_row1
                    };
                    dst_st.bufs[b].bytes.resize(sizeof(kIdentityE94B));
                    std::memcpy(dst_st.bufs[b].bytes.data(), kIdentityE94B, sizeof(kIdentityE94B));
                } else if (s == vs_slot && b == 0u && cand_mode == Mode::ScaleformA400) {
                    static constexpr float kIdentityA400[8]{
                        1.0f, 0.0f, 0.0f, 0.0f, // row0
                        0.0f, 1.0f, 0.0f, 0.0f, // row1
                    };
                    dst_st.bufs[b].bytes.resize(sizeof(kIdentityA400));
                    std::memcpy(dst_st.bufs[b].bytes.data(), kIdentityA400, sizeof(kIdentityA400));
                } else if (s == vs_slot && b == 0u && cand_mode == Mode::Scaleform2404) {
                    dst_st.bufs[b].bytes.clear();
                } else {
                    dst_st.bufs[b].bytes.assign(src_st.bufs[b].ptr,
                                                src_st.bufs[b].ptr + src_st.bufs[b].size);
                    if (cand_mode == Mode::FfxTintBaked && s == ps_slot && b == 0u &&
                        dst_st.bufs[b].bytes.size() >= 40u * 4u) {
                        static constexpr float kUnitTint[4]{1.0f, 1.0f, 1.0f, 1.0f};
                        std::memcpy(dst_st.bufs[b].bytes.data() + 36u * 4u, kUnitTint, 16u);
                    }
                }
            }
            dst_st.images.assign(src_st.images.data(), src_st.images.data() + src_st.num_images);
            dst_st.samplers.assign(src_st.samplers.data(),
                                   src_st.samplers.data() + src_st.num_samplers);
        }
    }

    // Append vertices (with exact degenerate stitching for TriangleStrip!):
    auto push_vtx = [&](u32 vtx_idx) {
        std::array<u8, 64> rec{};
        for (u32 a = 0; a < num_attrs; ++a) {
            encode_vertex_attr(vtx_idx, a, rec);
            auto& stream = open.attrs[a];
            stream.bytes.insert(stream.bytes.end(), rec.data(), rec.data() + stream.stride);
            std::memcpy(stream.last_vtx.data(), rec.data(), stream.stride);
        }
        ++open.num_verts;
    };

    auto push_repeat_last = [&]() {
        for (u32 a = 0; a < num_attrs; ++a) {
            auto& stream = open.attrs[a];
            stream.bytes.insert(stream.bytes.end(), stream.last_vtx.data(),
                                stream.last_vtx.data() + stream.stride);
        }
        ++open.num_verts;
    };

    u32 appended_prims = 0u;
    if (prim == Gcn::PrimitiveType::TriangleStrip) {
        u32 seg_start = 0u;
        while (seg_start < num_indices) {
            while (seg_start < num_indices && exp_restart[seg_start]) {
                ++seg_start;
            }
            u32 seg_end = seg_start;
            while (seg_end < num_indices && !exp_restart[seg_end]) {
                ++seg_end;
            }
            const u32 seg_len = seg_end - seg_start;
            if (seg_len >= 3u) {
                if (open.num_verts > 0u) {
                    // Preserve even vertex offset parity so front/back triangle winding is exact!
                    const u32 prev_copies = ((open.num_verts & 1u) == 0u) ? 1u : 2u;
                    for (u32 c = 0; c < prev_copies; ++c) {
                        push_repeat_last();
                    }
                    push_vtx(exp_indices[seg_start]);
                }
                for (u32 k = seg_start; k < seg_end; ++k) {
                    push_vtx(exp_indices[k]);
                }
                ++appended_prims;
            }
            seg_start = seg_end + 1u;
        }
    } else {
        for (u32 i = 0; i < num_indices; ++i) {
            if (!exp_restart[i]) {
                push_vtx(exp_indices[i]);
                ++appended_prims;
            }
        }
    }

    if (appended_prims == 0u) {
        if (open.num_draws == 0u) {
            open.Reset();
        }
        return false;
    }

    if (cand_mode == Mode::Scaleform2404) {
        const size_t base_dw = open.sf2404_cb_dwords.size();
        open.sf2404_cb_dwords.resize(base_dw + 24u);
        std::memcpy(&open.sf2404_cb_dwords[base_dw], local_stages[vs_slot].bufs[0].ptr,
                    24u * sizeof(u32));
        ++open.sf2404_quads;
    }

    ++open.num_draws;
    if (cand_mode == Mode::FfxTintBaked) {
        ++stat_ffx_draws_in;
        if (open.num_draws >= 2u) {
            ++stat_ffx_draws_saved;
        }
    } else if (cand_mode == Mode::Scaleform2404 || cand_mode == Mode::ScaleformE94B ||
               cand_mode == Mode::ScaleformA400) {
        ++stat_ui_draws_in;
        if (open.num_draws >= 2u) {
            ++stat_ui_draws_saved;
        }
    } else {
        ++stat_eq_draws_in;
        if (open.num_draws >= 2u) {
            ++stat_eq_draws_saved;
        }
    }

    BbStats::draws.fetch_sub(1, std::memory_order_relaxed);
    return true;
}

void VfxUiBatcher::Flush(Rasterizer& rasterizer) {
    if (flushing || !open.active) {
        open.Reset();
        return;
    }
    if (open.num_verts == 0u || !open.pipeline) {
        open.Reset();
        return;
    }
    flushing = true;

    const u64 cb_align = std::max<u64>(rasterizer.instance.StorageMinAlignment(),
                                       rasterizer.instance.UniformMinAlignment());

    // 1. Allocate all vertex attribute streams in ConstantRing:
    std::array<Gcn::Buffer, kMaxAttrs> vsharps_ovr{};
    bool alloc_ok = true;
    for (u32 a = 0; a < open.num_attrs; ++a) {
        auto& stream = open.attrs[a];
        const u64 vb_bytes = stream.bytes.size();
        const auto vb_off = rasterizer.constant_ring->Allocate(vb_bytes, 16u);
        if (!vb_off) {
            alloc_ok = false;
            break;
        }
        std::memcpy(rasterizer.constant_ring->Data(*vb_off), stream.bytes.data(), vb_bytes);
        rasterizer.constant_ring->Flush(*vb_off, vb_bytes);
        Gcn::Buffer vs_desc = stream.out_vsharp;
        vs_desc.base_address = kConstRingVertexBase + *vb_off;
        vs_desc.num_records = open.num_verts;
        vsharps_ovr[a] = vs_desc;
    }

    // 2. Allocate all non-special stage CBs in ConstantRing:
    std::array<std::array<Rasterizer::RingBinding, Shader::NUM_BUFFERS>, Shader::MaxStageTypes>
        stage_rings{};
    std::array<BatchedLightStageOverride, Shader::MaxStageTypes> stage_ovrs{};
    for (u32 s = 0; alloc_ok && s < open.num_stages; ++s) {
        const auto& st = open.stages[s];
        for (u32 b = 0; b < st.num_bufs; ++b) {
            const u8* cb_src = nullptr;
            u64 cb_bytes = 0u;
            if (s == open.vs_slot && b == 0u && open.mode == Mode::Scaleform2404) {
                cb_src = reinterpret_cast<const u8*>(open.sf2404_cb_dwords.data());
                cb_bytes = open.sf2404_cb_dwords.size() * sizeof(u32);
            } else {
                cb_src = st.bufs[b].bytes.data();
                cb_bytes = st.bufs[b].bytes.size();
            }
            const u64 alloc_bytes = std::max<u64>(Common::AlignUp(cb_bytes, 16ULL), 16ULL);
            const auto cb_off = rasterizer.constant_ring->Allocate(alloc_bytes, cb_align);
            if (!cb_off) {
                alloc_ok = false;
                break;
            }
            u8* dst = rasterizer.constant_ring->Data(*cb_off);
            if (cb_bytes > 0u) {
                std::memcpy(dst, cb_src, cb_bytes);
            }
            if (alloc_bytes > cb_bytes) {
                std::memset(dst + cb_bytes, 0, alloc_bytes - cb_bytes);
            }
            rasterizer.constant_ring->Flush(*cb_off, alloc_bytes);
            stage_rings[s][b] = Rasterizer::RingBinding{
                .index = st.bufs[b].sharp_idx,
                .size = static_cast<u32>(alloc_bytes),
                .offset = *cb_off,
                .address = 0u,
                .ring = 0u,
                .always_note = 0u,
            };
        }
        stage_ovrs[s] = BatchedLightStageOverride{
            .pgm_base = st.pgm_base,
            .user_data = st.user_data,
            .flattened_ud_buf = st.flattened_ud_buf,
            .ring_bindings =
                std::span<const Rasterizer::RingBinding>{stage_rings[s].data(), st.num_bufs},
        };
    }

    if (alloc_ok) {
        auto& lp = *rasterizer.liverpool;
        StageBChunks saved_chunks{};
        CaptureStageBChunks(lp.regs, saved_chunks);
        const Gcn::RegDirty saved_dirty = lp.pipe_dirty;
        const auto saved_cb_extent = lp.last_cb_extent;
        const auto saved_db_extent = lp.last_db_extent;
        const auto saved_cb_uid = lp.last_cb_uid;
        const auto saved_db_uid = lp.last_db_uid;

        RestoreStageBChunks(lp.regs, open.base_chunks);
        lp.regs.num_indices = open.num_verts;
        lp.regs.index_offset = 0u;
        lp.regs.primitive_type = open.prim_type;
        lp.regs.enable_primitive_restart = 0u;
        lp.last_cb_extent = open.cb_extent;
        lp.last_db_extent = open.db_extent;
        lp.last_cb_uid = open.cb_uid;
        lp.last_db_uid = open.db_uid;

        StageBChunks cur_chunks{};
        CaptureStageBChunks(lp.regs, cur_chunks);
        lp.pipe_dirty = saved_dirty;
        MarkStageBDiffDirty(saved_chunks, cur_chunks, lp.pipe_dirty);

        const BatchedLightDrawOverride draw_ovr{
            .stages = std::span<const BatchedLightStageOverride>{stage_ovrs.data(), open.num_stages},
            .vsharps = std::span<const Gcn::Buffer>{vsharps_ovr.data(), open.num_attrs},
            .note_guest_vb_ib_reads = false,
        };
        rasterizer.PostDraw(open.pipeline, nullptr, false, 0u, nullptr, nullptr, &draw_ovr);
        BbStats::draws.fetch_add(1, std::memory_order_relaxed);
        if (open.num_draws >= 2u) {
            ++stat_batches_emitted;
        } else {
            ++stat_single_passthrough;
        }

        RestoreStageBChunks(lp.regs, saved_chunks);
        lp.last_cb_extent = saved_cb_extent;
        lp.last_db_extent = saved_db_extent;
        lp.last_cb_uid = saved_cb_uid;
        lp.last_db_uid = saved_db_uid;
        lp.pipe_dirty.Clear();
        lp.pipe_dirty.blocks |= saved_dirty.blocks;
        lp.pipe_dirty.stage_b_mask[0] |= saved_dirty.stage_b_mask[0];
        lp.pipe_dirty.stage_b_mask[1] |= saved_dirty.stage_b_mask[1];
        MarkStageBDiffDirty(cur_chunks, saved_chunks, lp.pipe_dirty);

        const u64 total_saved =
            stat_ffx_draws_saved + stat_ui_draws_saved + stat_eq_draws_saved;
        if (!first_active_logged && open.num_draws >= 3u) {
            first_active_logged = true;
            std::printf(
                "[VfxUiBatcher] Active: coalesced %u sequential %s draws (%u verts) into 1 draw "
                "(total_saved=%llu)\n",
                open.num_draws,
                open.mode == Mode::FfxTintBaked    ? "FFX-SoftParticle"
                : open.mode == Mode::Scaleform2404 ? "Scaleform-2404"
                : open.mode == Mode::ScaleformE94B ? "Scaleform-E94B"
                : open.mode == Mode::ScaleformA400 ? "Scaleform-A400"
                                                   : "SmallEq",
                open.num_verts, static_cast<unsigned long long>(total_saved));
            std::fflush(stdout);
        }
        if ((!summary_logged && stat_candidates_seen >= 400u) ||
            (BatchLogEnabled() && (stat_candidates_seen % 2000u) < open.num_draws)) {
            summary_logged = true;
            std::printf(
                "[VfxUiBatcher] Summary(seen=%llu): saved=%llu (ffx=%llu/%llu ui=%llu/%llu "
                "eq=%llu/%llu) multi_emitted=%llu singles=%llu task_flushes=%llu "
                "splits=(regs=%llu layout=%llu cb=%llu vb=%llu ib=%llu)\n",
                static_cast<unsigned long long>(stat_candidates_seen),
                static_cast<unsigned long long>(total_saved),
                static_cast<unsigned long long>(stat_ffx_draws_saved),
                static_cast<unsigned long long>(stat_ffx_draws_in),
                static_cast<unsigned long long>(stat_ui_draws_saved),
                static_cast<unsigned long long>(stat_ui_draws_in),
                static_cast<unsigned long long>(stat_eq_draws_saved),
                static_cast<unsigned long long>(stat_eq_draws_in),
                static_cast<unsigned long long>(stat_batches_emitted),
                static_cast<unsigned long long>(stat_single_passthrough),
                static_cast<unsigned long long>(stat_task_flushes),
                static_cast<unsigned long long>(fb_regs),
                static_cast<unsigned long long>(fb_layout),
                static_cast<unsigned long long>(fb_cb),
                static_cast<unsigned long long>(fb_vb),
                static_cast<unsigned long long>(fb_ib));
            std::fflush(stdout);
        }
    }

    open.Reset();
    flushing = false;
}

} // namespace Vulkan
