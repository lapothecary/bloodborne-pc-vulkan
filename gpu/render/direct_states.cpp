// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 3A (Stage A elimination): Lock-free Direct State Object Registry implementation.

#include "render/direct_states.h"

#include <atomic>
#include <bit>
#include <cstring>
#include <mutex>

#include "gcn/regs_raster.h"
#include "gcn/regs_targets.h"
#include "render/gcn_to_vk.h"

namespace Vulkan::DirectStates {

namespace {

CompiledRasterDyn CompileRasterDyn(const RasterStateDesc& rs) noexcept {
    CompiledRasterDyn out{};
    // NOTE: In Bloodborne's `CGRasterizerStateObject`, field +0x00 (`primitive_setup`) holds
    // `PA_CL_CLIP_CNTL` (0xA204, `Gcn::ClipperControl`, mask 0x0F7FE03F) and field +0x04
    // (`clip_control`) holds `PA_SU_SC_MODE_CNTL` (0xA205, `Gcn::PolygonControl`, mask 0x00393FFF).
    const u32 cc_bits = rs.primitive_setup & 0x0F7FE03Fu;
    const u32 pc_bits = rs.clip_control & 0x00393FFFu;
    Gcn::ClipperControl cc{};
    Gcn::PolygonControl pc{};
    std::memcpy(&cc, &cc_bits, sizeof(cc));
    std::memcpy(&pc, &pc_bits, sizeof(pc));
    out.cull_mode_vk = static_cast<u32>(GcnToVk::CullMode(pc.CullingMode()));
    out.front_face_vk = static_cast<u32>(GcnToVk::FrontFace(pc.front_face));
    out.depth_bias_const = std::bit_cast<float>(rs.slope_scale_depth_bias_bits);
    out.depth_bias_clamp = std::bit_cast<float>(rs.depth_bias_clamp_bits);
    const float scaled_db = std::bit_cast<float>(rs.depth_bias_bits) * 16.0f;
    out.depth_bias_slope = scaled_db / 16.0f;
    out.depth_bias_enable = pc.NeedsBias();
    out.rasterizer_discard = cc.dx_rasterization_kill != 0u;
    out.vp_flags = static_cast<u8>((cc.clip_disable != 0u ? 1u : 0u) |
                                   (cc.clip_space == Gcn::ClipSpace::MinusWToW ? 2u : 0u) |
                                   ((rs.scissor_enable & 1u) != 0u ? 4u : 0u) |
                                   (pc.enable_window_offset != 0u ? 8u : 0u));
    return out;
}

CompiledDepthStencilDyn CompileDepthStencilDyn(const DepthStencilStateDesc& ds) noexcept {
    CompiledDepthStencilDyn out{};
    const u32 dc_bits = ds.depth_control & 0xC07007FFu;
    const u32 sc_bits = ds.stencil_control & 0x00FFFFFFu;
    Gcn::DepthControl dc{};
    Gcn::StencilControl sc{};
    std::memcpy(&dc, &dc_bits, sizeof(dc));
    std::memcpy(&sc, &sc_bits, sizeof(sc));
    out.depth_enable = dc.depth_enable != 0u;
    out.depth_write_enable = dc.depth_write_enable != 0u;
    out.depth_bounds_enable = dc.depth_bounds_enable != 0u;
    out.depth_compare_op_vk = static_cast<u32>(GcnToVk::CompareOp(dc.depth_func));
    out.stencil_enable = dc.stencil_enable != 0u;
    out.two_sided = dc.backface_enable != 0u;

    out.front_fail_vk = static_cast<u32>(GcnToVk::StencilOp(sc.stencil_fail_front));
    out.front_pass_vk = static_cast<u32>(GcnToVk::StencilOp(sc.stencil_zpass_front));
    out.front_zfail_vk = static_cast<u32>(GcnToVk::StencilOp(sc.stencil_zfail_front));
    out.front_cmp_vk = static_cast<u32>(GcnToVk::CompareOp(dc.stencil_ref_func));

    auto uses_op = [](Gcn::StencilFunc a, Gcn::StencilFunc b, Gcn::StencilFunc c) noexcept {
        return a == Gcn::StencilFunc::ReplaceOp || b == Gcn::StencilFunc::ReplaceOp ||
               c == Gcn::StencilFunc::ReplaceOp;
    };
    auto all_keep = [](Gcn::StencilFunc a, Gcn::StencilFunc b, Gcn::StencilFunc c) noexcept {
        return a == Gcn::StencilFunc::Keep && b == Gcn::StencilFunc::Keep &&
               c == Gcn::StencilFunc::Keep;
    };

    out.front_uses_op_val =
        uses_op(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
    out.front_all_keep =
        all_keep(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
    out.front_cmp_checks_ref =
        dc.stencil_ref_func != Gcn::CompareFunc::Always &&
        dc.stencil_ref_func != Gcn::CompareFunc::Never;

    if (out.two_sided) {
        out.back_fail_vk = static_cast<u32>(GcnToVk::StencilOp(sc.stencil_fail_back));
        out.back_pass_vk = static_cast<u32>(GcnToVk::StencilOp(sc.stencil_zpass_back));
        out.back_zfail_vk = static_cast<u32>(GcnToVk::StencilOp(sc.stencil_zfail_back));
        out.back_cmp_vk = static_cast<u32>(GcnToVk::CompareOp(dc.stencil_bf_func));
        out.back_uses_op_val =
            uses_op(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back);
        out.back_all_keep =
            all_keep(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back);
        out.back_cmp_checks_ref =
            dc.stencil_bf_func != Gcn::CompareFunc::Always &&
            dc.stencil_bf_func != Gcn::CompareFunc::Never;
    } else {
        out.back_fail_vk = out.front_fail_vk;
        out.back_pass_vk = out.front_pass_vk;
        out.back_zfail_vk = out.front_zfail_vk;
        out.back_cmp_vk = out.front_cmp_vk;
        out.back_uses_op_val = out.front_uses_op_val;
        out.back_all_keep = out.front_all_keep;
        out.back_cmp_checks_ref = out.front_cmp_checks_ref;
    }
    return out;
}

struct NoCompiledDyn {};
inline NoCompiledDyn CompileNoDyn(const BlendStateDesc&) noexcept {
    return {};
}

constexpr u32 kLookupBuckets = 512u;
constexpr u32 kLookupMask = kLookupBuckets - 1u;

struct LookupBucket {
    std::atomic<u64> way0{0};
    std::atomic<u64> way1{0};
};

inline bool IsCurrentThreadStackPtr(const void* ptr) noexcept {
    alignas(16) const u8 anchor = 0;
    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t sp = reinterpret_cast<uintptr_t>(&anchor);
    const uintptr_t diff = (p >= sp) ? (p - sp) : (sp - p);
    return diff < (64u << 10);
}

inline u64 PackPtrAndId(const void* ptr, u16 id) noexcept {
    const u64 shifted = (reinterpret_cast<uintptr_t>(ptr) >> 2) << 16;
    return shifted | static_cast<u64>(id & 0x3FFu);
}

inline u32 BucketIndex(const void* ptr) noexcept {
    const u64 x = reinterpret_cast<uintptr_t>(ptr) >> 4;
    return static_cast<u32>((x ^ (x >> 9)) & kLookupMask);
}

template <typename DescT, typename CompiledT = NoCompiledDyn,
          CompiledT (*CompileFn)(const DescT&) noexcept = nullptr>
struct StateTable {
    std::mutex mutex;
    std::atomic<u32> count{0}; // number of unique states in slots[1..count]
    std::atomic<u64> lookups{0};
    std::array<DescT, kMaxStates> slots{};
    std::array<CompiledT, kMaxStates> compiled_slots{};
    std::array<LookupBucket, kLookupBuckets> ptr_l1{};

    [[nodiscard]] u16 LookupOrRegister(const void* obj_ptr, const DescT& normalized_desc) noexcept {
        lookups.fetch_add(1u, std::memory_order_relaxed);
        const bool is_stack = IsCurrentThreadStackPtr(obj_ptr);
        const u64 target_hi = (reinterpret_cast<uintptr_t>(obj_ptr) >> 2) << 16;
        const u32 bidx = BucketIndex(obj_ptr);

        if (!is_stack) [[likely]] {
            const auto& bucket = ptr_l1[bidx];
            const u64 w0 = bucket.way0.load(std::memory_order_acquire);
            const u16 id0 = static_cast<u16>(w0 & 0x3FFu);
            if ((w0 & ~0xFFFFULL) == target_hi && id0 != 0u && slots[id0] == normalized_desc)
                [[likely]] {
                return id0;
            }
            const u64 w1 = bucket.way1.load(std::memory_order_acquire);
            const u16 id1 = static_cast<u16>(w1 & 0x3FFu);
            if ((w1 & ~0xFFFFULL) == target_hi && id1 != 0u && slots[id1] == normalized_desc) {
                return id1;
            }
        }

        std::scoped_lock lk{mutex};
        const u32 cur_n = count.load(std::memory_order_relaxed);
        u16 id = 0u;
        for (u32 i = 1u; i <= cur_n; ++i) {
            if (slots[i] == normalized_desc) {
                id = static_cast<u16>(i);
                break;
            }
        }
        if (id == 0u) {
            if (cur_n + 1u >= kMaxStates) {
                return 0u;
            }
            const u32 next_id = cur_n + 1u;
            slots[next_id] = normalized_desc;
            if constexpr (CompileFn != nullptr) {
                compiled_slots[next_id] = CompileFn(normalized_desc);
            }
            count.store(next_id, std::memory_order_release);
            id = static_cast<u16>(next_id);
        }
        if (!is_stack) {
            auto& bucket = ptr_l1[bidx];
            const u64 packed = PackPtrAndId(obj_ptr, id);
            const u64 old0 = bucket.way0.load(std::memory_order_relaxed);
            if ((old0 & ~0xFFFFULL) == target_hi || old0 == 0u) {
                bucket.way0.store(packed, std::memory_order_release);
            } else {
                bucket.way1.store(old0, std::memory_order_release);
                bucket.way0.store(packed, std::memory_order_release);
            }
        }
        return id;
    }

    [[nodiscard]] u16 InternDesc(const DescT& normalized_desc) noexcept {
        lookups.fetch_add(1u, std::memory_order_relaxed);
        const u32 cur_n = count.load(std::memory_order_acquire);
        for (u32 i = 1u; i <= cur_n; ++i) {
            if (slots[i] == normalized_desc) [[likely]] {
                return static_cast<u16>(i);
            }
        }
        std::scoped_lock lk{mutex};
        const u32 locked_n = count.load(std::memory_order_relaxed);
        for (u32 i = 1u; i <= locked_n; ++i) {
            if (slots[i] == normalized_desc) {
                return static_cast<u16>(i);
            }
        }
        if (locked_n + 1u >= kMaxStates) {
            return 0u;
        }
        const u32 next_id = locked_n + 1u;
        slots[next_id] = normalized_desc;
        if constexpr (CompileFn != nullptr) {
            compiled_slots[next_id] = CompileFn(normalized_desc);
        }
        count.store(next_id, std::memory_order_release);
        return static_cast<u16>(next_id);
    }

    [[nodiscard]] bool Read(u16 id, DescT& out) const noexcept {
        if (id == 0u || id > count.load(std::memory_order_acquire)) {
            return false;
        }
        out = slots[id];
        return true;
    }

    [[nodiscard]] const CompiledT* GetCompiled(u16 id) const noexcept {
        if (id == 0u || id > count.load(std::memory_order_acquire)) {
            return nullptr;
        }
        return &compiled_slots[id];
    }
};

struct PsUsageL1Slot {
    std::atomic<u32> seq{0};
    u32 counts_packed{0};
    const void* vs_ptr{nullptr};
    const void* ps_ptr{nullptr};
    u64 vs_head{0};
    u64 vs_tail{0};
    u64 ps_head{0};
    u64 ps_tail{0};
    u16 id{0};
};

inline void ReadHeadTailBytes(const void* ptr, u32 bytes, u64& head, u64& tail) noexcept {
    head = 0ULL;
    tail = 0ULL;
    if (!ptr || bytes == 0u) {
        return;
    }
    const auto* p = static_cast<const u8*>(ptr);
    if (bytes >= 8u) {
        std::memcpy(&head, p, 8u);
        std::memcpy(&tail, p + (bytes - 8u), 8u);
    } else {
        std::memcpy(&head, p, bytes);
        tail = head;
    }
}

struct PsUsageTable {
    static constexpr u32 kL1Size = 1024u;
    static constexpr u32 kL1Mask = kL1Size - 1u;

    std::mutex mutex;
    std::atomic<u32> count{0};
    std::atomic<u64> lookups{0};
    std::array<PsUsageDesc, kMaxStates> slots{};
    std::array<PsUsageL1Slot, kL1Size> l1{};

    [[nodiscard]] u16 LookupOrRegister(const void* vs_exports_raw, u32 num_vs_exports,
                                       const void* ps_inputs_raw, u32 num_ps_inputs) noexcept {
        if (!ps_inputs_raw || num_ps_inputs == 0u || num_ps_inputs > 32u ||
            num_vs_exports > 64u) {
            return 0u;
        }
        lookups.fetch_add(1u, std::memory_order_relaxed);
        const u32 counts_packed = (num_vs_exports & 0xFFFFu) | ((num_ps_inputs & 0xFFFFu) << 16);
        u64 vh = 0, vt = 0, ph = 0, pt = 0;
        ReadHeadTailBytes(vs_exports_raw, num_vs_exports * 2u, vh, vt);
        ReadHeadTailBytes(ps_inputs_raw, num_ps_inputs * 2u, ph, pt);

        const u64 h = (reinterpret_cast<uintptr_t>(vs_exports_raw) * 0x9E3779B97F4A7C15ULL) ^
                      (reinterpret_cast<uintptr_t>(ps_inputs_raw) * 0xBF58476D1CE4E5B9ULL) ^
                      (u64(counts_packed) << 7);
        const u32 bidx = static_cast<u32>((h ^ (h >> 17)) & kL1Mask);
        const bool is_stack =
            IsCurrentThreadStackPtr(vs_exports_raw) || IsCurrentThreadStackPtr(ps_inputs_raw);

        if (!is_stack) [[likely]] {
            const auto& e = l1[bidx];
            const u32 s0 = e.seq.load(std::memory_order_acquire);
            if (s0 != 0u && (s0 & 1u) == 0u) [[likely]] {
                const void* ev = e.vs_ptr;
                const void* ep = e.ps_ptr;
                const u32 ec = e.counts_packed;
                const u64 evh = e.vs_head;
                const u64 evt = e.vs_tail;
                const u64 eph = e.ps_head;
                const u64 ept = e.ps_tail;
                const u16 eid = e.id;
                std::atomic_thread_fence(std::memory_order_acquire);
                if (e.seq.load(std::memory_order_relaxed) == s0 && ev == vs_exports_raw &&
                    ep == ps_inputs_raw && ec == counts_packed && evh == vh && evt == vt &&
                    eph == ph && ept == pt && eid != 0u) [[likely]] {
                    return eid;
                }
            }
        }

        // Exact PS4 `generatePsShaderUsageTable` (`0x10873a0`) computation.
        PsUsageDesc desc{};
        desc.count = num_ps_inputs;
        const auto* vs_exp = static_cast<const u8*>(vs_exports_raw);
        const auto* ps_in_u16 = static_cast<const u16*>(ps_inputs_raw);
        const auto* ps_in_u8 = static_cast<const u8*>(ps_inputs_raw);
        for (u32 i = 0u; i < num_ps_inputs; ++i) {
            u32 match_val = 0x20u;
            const u8 target_sem = ps_in_u8[i * 2u];
            if (vs_exp != nullptr) {
                for (u32 k = 0u; k < num_vs_exports; ++k) {
                    if (vs_exp[k * 2u] == target_sem) {
                        match_val = vs_exp[k * 2u + 1u];
                        break;
                    }
                }
            }
            const u32 ps_w = ps_in_u16[i];
            const u32 lo6 = (((ps_w >> 7) & 0x1E0u) | match_val) & 0x3Fu;
            const u32 mid2 = lo6 | (((ps_w >> 8) & 0x3u) << 8);
            const u32 bit10 = (((ps_w >> 12) | (ps_w >> 10)) & 0x1u) << 10;
            desc.dwords[i] = mid2 | bit10;
        }

        std::scoped_lock lk{mutex};
        const u32 cur_n = count.load(std::memory_order_relaxed);
        u16 id = 0u;
        for (u32 i = 1u; i <= cur_n; ++i) {
            if (slots[i] == desc) {
                id = static_cast<u16>(i);
                break;
            }
        }
        if (id == 0u) {
            if (cur_n + 1u >= kMaxStates) {
                return 0u;
            }
            const u32 next_id = cur_n + 1u;
            slots[next_id] = desc;
            count.store(next_id, std::memory_order_release);
            id = static_cast<u16>(next_id);
        }
        if (!is_stack) {
            auto& e = l1[bidx];
            const u32 s0 = e.seq.load(std::memory_order_relaxed);
            e.seq.store(s0 + 1u, std::memory_order_release);
            std::atomic_thread_fence(std::memory_order_release);
            e.vs_ptr = vs_exports_raw;
            e.ps_ptr = ps_inputs_raw;
            e.counts_packed = counts_packed;
            e.vs_head = vh;
            e.vs_tail = vt;
            e.ps_head = ph;
            e.ps_tail = pt;
            e.id = id;
            e.seq.store(s0 + 2u, std::memory_order_release);
        }
        return id;
    }

    [[nodiscard]] const PsUsageDesc* GetPtr(u16 id) const noexcept {
        if (id == 0u || id > count.load(std::memory_order_acquire)) {
            return nullptr;
        }
        return &slots[id];
    }
};

struct AllTables {
    StateTable<RasterStateDesc, CompiledRasterDyn, &CompileRasterDyn> raster;
    StateTable<BlendStateDesc, NoCompiledDyn, &CompileNoDyn> blend;
    StateTable<DepthStencilStateDesc, CompiledDepthStencilDyn, &CompileDepthStencilDyn> ds;
    PsUsageTable ps_usage;
};

AllTables& Tables() noexcept {
    static AllTables tables{};
    return tables;
}

} // namespace

bb_native_stream::RasterStateId LookupOrRegisterRasterState(const void* rs_obj) noexcept {
    if (!rs_obj) {
        return 0u;
    }
    RasterStateDesc desc{};
    std::memcpy(&desc, rs_obj, sizeof(desc));
    desc.pad = 0u;
    return Tables().raster.LookupOrRegister(rs_obj, desc);
}

bb_native_stream::BlendStateId LookupOrRegisterBlendState(const void* bs_obj) noexcept {
    if (!bs_obj) {
        return 0u;
    }
    BlendStateDesc desc{};
    std::memcpy(&desc, bs_obj, 37u); // 4B mask + 32B blend_control[8] + 1B alpha_to_coverage
    desc.pad[0] = desc.pad[1] = desc.pad[2] = 0u;
    return Tables().blend.LookupOrRegister(bs_obj, desc);
}

bb_native_stream::DepthStencilStateId LookupOrRegisterDepthStencilState(const void* ds_obj) noexcept {
    if (!ds_obj) {
        return 0u;
    }
    DepthStencilStateDesc desc{};
    std::memcpy(&desc, ds_obj, sizeof(desc));
    return Tables().ds.LookupOrRegister(ds_obj, desc);
}

bb_native_stream::RasterStateId InternRasterStateDesc(const RasterStateDesc& desc) noexcept {
    RasterStateDesc norm = desc;
    norm.pad = 0u;
    return Tables().raster.InternDesc(norm);
}

bb_native_stream::BlendStateId InternBlendStateDesc(const BlendStateDesc& desc) noexcept {
    BlendStateDesc norm = desc;
    norm.pad[0] = norm.pad[1] = norm.pad[2] = 0u;
    return Tables().blend.InternDesc(norm);
}

bb_native_stream::DepthStencilStateId InternDepthStencilStateDesc(
    const DepthStencilStateDesc& desc) noexcept {
    return Tables().ds.InternDesc(desc);
}

bool ReadRasterState(bb_native_stream::RasterStateId id, RasterStateDesc& out) noexcept {
    return Tables().raster.Read(id, out);
}

bool ReadBlendState(bb_native_stream::BlendStateId id, BlendStateDesc& out) noexcept {
    return Tables().blend.Read(id, out);
}

bool ReadDepthStencilState(bb_native_stream::DepthStencilStateId id,
                           DepthStencilStateDesc& out) noexcept {
    return Tables().ds.Read(id, out);
}

const CompiledRasterDyn* GetCompiledRasterDyn(bb_native_stream::RasterStateId id) noexcept {
    return Tables().raster.GetCompiled(id);
}

const CompiledDepthStencilDyn* GetCompiledDepthStencilDyn(
    bb_native_stream::DepthStencilStateId id) noexcept {
    return Tables().ds.GetCompiled(id);
}

u16 LookupOrRegisterPsUsage(const void* vs_exports_raw, u32 num_vs_exports,
                            const void* ps_inputs_raw, u32 num_ps_inputs) noexcept {
    return Tables().ps_usage.LookupOrRegister(vs_exports_raw, num_vs_exports, ps_inputs_raw,
                                              num_ps_inputs);
}

const PsUsageDesc* GetPsUsagePtr(u16 id) noexcept {
    return Tables().ps_usage.GetPtr(id);
}

RegistryStats GetStats() noexcept {
    const auto& t = Tables();
    return RegistryStats{
        .raster_unique = t.raster.count.load(std::memory_order_relaxed),
        .blend_unique = t.blend.count.load(std::memory_order_relaxed),
        .ds_unique = t.ds.count.load(std::memory_order_relaxed),
        .ps_usage_unique = t.ps_usage.count.load(std::memory_order_relaxed),
        .raster_lookups = t.raster.lookups.load(std::memory_order_relaxed),
        .blend_lookups = t.blend.lookups.load(std::memory_order_relaxed),
        .ds_lookups = t.ds.lookups.load(std::memory_order_relaxed),
        .ps_usage_lookups = t.ps_usage.lookups.load(std::memory_order_relaxed),
    };
}

} // namespace Vulkan::DirectStates

extern "C" {

uint32_t PS4_SYSV_ABI bb_direct_vk_lookup_raster_state_id(const void* rs_obj) noexcept {
    return Vulkan::DirectStates::LookupOrRegisterRasterState(rs_obj);
}

uint32_t PS4_SYSV_ABI bb_direct_vk_lookup_blend_state_id(const void* bs_obj) noexcept {
    return Vulkan::DirectStates::LookupOrRegisterBlendState(bs_obj);
}

uint32_t PS4_SYSV_ABI bb_direct_vk_lookup_ds_state_id(const void* ds_obj) noexcept {
    return Vulkan::DirectStates::LookupOrRegisterDepthStencilState(ds_obj);
}

uint32_t PS4_SYSV_ABI bb_direct_vk_lookup_ps_usage_id(const void* vs_exports_raw,
                                                      uint32_t num_vs_exports,
                                                      const void* ps_inputs_raw,
                                                      uint32_t num_ps_inputs) noexcept {
    return Vulkan::DirectStates::LookupOrRegisterPsUsage(vs_exports_raw, num_vs_exports,
                                                         ps_inputs_raw, num_ps_inputs);
}

} // extern "C"
