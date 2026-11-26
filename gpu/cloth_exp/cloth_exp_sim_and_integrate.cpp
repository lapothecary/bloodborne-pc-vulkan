// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// File: gpu/cloth_exp/cloth_exp_sim_and_integrate.cpp
//
// 100% 1-to-1 C++20 decompilation of PS4 eboot.bin Batch 3 (6 functions,
// 12,555 B PS4 code): pre-step damping/action gathering, substep simulation
// driver, reference-bone motion transfer, Verlet force/damping integrator,
// triangle/convex pinch & stuck-particle release, and SIMD cloth normal update.
// Zero stubs, exact SIMD/register/memory semantics, exact hkMonitorStream tags.
// ============================================================================

#include "cloth_exp/cloth_exp_sim_and_integrate.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "cloth_exp/cloth_exp_runtime.h"

namespace BbClothExp {

namespace {

using GetActiveCfgFn = u8*(PS4_SYSV_ABI*)(u8*);

struct alignas(16) HkTransform4x4 {
    __m128 col[4];
};

[[gnu::always_inline]] inline __m128 HkMaxFloatVec() noexcept {
    return _mm_set1_ps(std::bit_cast<float>(0x7f7fffeeu)); // 0x2bd5460
}

[[gnu::always_inline]] inline __m128 HkMinFloatVec() noexcept {
    return _mm_set1_ps(std::bit_cast<float>(0xff7fffeeu)); // 0x2bd5460 ^ 0x80000000
}

[[gnu::always_inline]] inline void SetIdentityTransform(HkTransform4x4& out) noexcept {
    out.col[0] = _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f);
    out.col[1] = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
    out.col[2] = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
    out.col[3] = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
}

[[gnu::always_inline]] inline __m128 NormalizeQuatNR(__m128 q) noexcept {
    const __m128 half_v = _mm_set1_ps(0.5f);  // 0x2bd5eb0
    const __m128 three_v = _mm_set1_ps(3.0f); // 0x2bd5ee0
    const __m128 d = _mm_dp_ps(q, q, 0xff);
    const __m128 r = _mm_rsqrt_ps(d);
    const __m128 nr =
        _mm_mul_ps(_mm_mul_ps(r, half_v), _mm_sub_ps(three_v, _mm_mul_ps(d, _mm_mul_ps(r, r))));
    return _mm_mul_ps(q, nr);
}

/// Appends active `hclAction*` objects (`vtbl[0x38/8](action) != 0`) into `sim + 0x168`.
[[gnu::always_inline]] inline void CollectActiveActionsFromArray(u8* sim,
                                                                 u8* const* const* arr_slot,
                                                                 const std::int32_t* cnt_slot) noexcept {
    if (*cnt_slot <= 0) {
        return;
    }
    using IsActiveFn = u8(PS4_SYSV_ABI*)(u8*);
    using ReserveMoreFn = void(PS4_SYSV_ABI*)(void*, void*, int);

    for (std::int32_t b = 0; b < *cnt_slot; ++b) {
        u8* action = (*arr_slot)[b];
        void* const* vtbl = *reinterpret_cast<void* const* const*>(action);
        if (reinterpret_cast<IsActiveFn>(vtbl[0x38 / 8])(action) != 0u) {
            u8* const* cur_src = *arr_slot;
            std::int32_t sz = *reinterpret_cast<const std::int32_t*>(sim + 0x170);
            const std::int32_t cap =
                *reinterpret_cast<const std::int32_t*>(sim + 0x174) & 0x3fffffff;
            if (sz == cap) {
                ResolveFn<ReserveMoreFn>(0x0419100ULL)(
                    reinterpret_cast<void*>(ResolveVma(0x54018b0ULL)), sim + 0x168, 8);
                sz = *reinterpret_cast<const std::int32_t*>(sim + 0x170);
            }
            u8** dst = *reinterpret_cast<u8** const*>(sim + 0x168);
            dst[sz] = cur_src[b];
            ++(*reinterpret_cast<std::int32_t*>(sim + 0x170));
        }
    }
}

/// Computes effective damping `[sim + 0x10c]` matching `0x76fd50..0x76feee` / `0x76fea2..0x76ff36`.
[[gnu::always_inline]] inline float ComputeEffectiveDampingPS4(float damp, float substep_dt) noexcept {
    if (!(1.0f > damp)) {
        // Matches `vucomiss 1.0f, damp; jbe -> 0.0f` (including NaN).
        return 0.0f;
    }
    if (damp == 0.0f) {
        return 1.0f;
    }
    using PowfFn = float(PS4_SYSV_ABI*)(float, float);
    return ResolveFn<PowfFn>(0x2bbe948ULL)(1.0f - damp, substep_dt);
}

/// Linear blend ramp used in `transferMotion` (`0x7713a7..0x771423` & `0x77146e..0x77150b`).
[[gnu::always_inline]] inline float EvalTransferRamp(float speed, float min_spd, float max_spd,
                                                     float min_blend, float max_blend) noexcept {
    if (speed >= max_spd) {
        return max_blend;
    }
    if (!(speed > min_spd)) {
        return min_blend;
    }
    const float span = max_spd - min_spd;
    constexpr float kSpanEps = std::bit_cast<float>(0x3727c5acu); // 0x47b3360 (~1e-5f)
    const float inv_span = (std::fabs(span) > kSpanEps) ? (1.0f / span) : 0.0f;
    return min_blend + ((max_blend - min_blend) * (speed - min_spd)) * inv_span;
}

} // namespace

extern "C" {

// ============================================================================
// 31. hclSimulateOperator::preStep @ 0x076fb90 (1717 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclSimulateOperator_preStep(u8* op, u8* step_info) noexcept {
    u8* ctx0 = *reinterpret_cast<u8* const*>(step_info + 0x10);
    const std::int32_t sim_idx = *reinterpret_cast<const std::int32_t*>(op + 0x1c);
    u8* sim = (*reinterpret_cast<u8* const* const*>(ctx0 + 0x20))[sim_idx];

    const float slowmo = (*reinterpret_cast<const std::int32_t*>(sim + 0x1cc) == 1)
                             ? 1.0f
                             : *reinterpret_cast<const float*>(sim + 0x1d0);

    const u8* cfg = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
    std::int32_t substeps = *reinterpret_cast<const std::int32_t*>(cfg + 0x18);
    if (substeps == 0) {
        substeps = *reinterpret_cast<const std::int32_t*>(op + 0x20);
    }

    const float step_dt = *reinterpret_cast<const float*>(step_info + 0x00);
    float prev_sdt = *reinterpret_cast<const float*>(sim + 0x108);

    if (prev_sdt == 0.0f) {
        const u8* cfg_init = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
        if (cfg_init[0x1e] != 0u) {
            const u8* sd = *reinterpret_cast<const u8* const*>(sim + 0x10);
            const std::int32_t tset_idx = *reinterpret_cast<const std::int32_t*>(sd + 0x150);
            const std::int32_t bone_idx = *reinterpret_cast<const std::int32_t*>(sd + 0x154);

            HkTransform4x4 bone_mat{};
            SetIdentityTransform(bone_mat);

            const u8* ctx_cur = *reinterpret_cast<const u8* const*>(step_info + 0x10);
            const u8* tset = (*reinterpret_cast<const u8* const* const*>(ctx_cur + 0x40))[tset_idx];
            const __m128* src_m = reinterpret_cast<const __m128*>(
                *reinterpret_cast<const u8* const*>(tset + 0x10) +
                (std::size_t(bone_idx) << 6));
            bone_mat.col[0] = _mm_load_ps(reinterpret_cast<const float*>(src_m + 0));
            bone_mat.col[1] = _mm_load_ps(reinterpret_cast<const float*>(src_m + 1));
            bone_mat.col[2] = _mm_load_ps(reinterpret_cast<const float*>(src_m + 2));
            bone_mat.col[3] = _mm_load_ps(reinterpret_cast<const float*>(src_m + 3));

            using SetQTransformFn = void(PS4_SYSV_ABI*)(void*, const void*);
            ResolveFn<SetQTransformFn>(0x043b8c0ULL)(sim + 0x120, &bone_mat);
        }
        prev_sdt = *reinterpret_cast<const float*>(sim + 0x108);
    }

    const float new_sdt = step_dt / (slowmo * static_cast<float>(substeps));
    if (prev_sdt != new_sdt) {
        if (prev_sdt == 0.0f) {
            *reinterpret_cast<float*>(sim + 0x108) = new_sdt;
            MonitorPushTimer(0x47b3319ULL);
            const u8* cfg_d = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
            const float damp = *reinterpret_cast<const float*>(cfg_d + 0x10);
            *reinterpret_cast<float*>(sim + 0x10c) = ComputeEffectiveDampingPS4(damp, new_sdt);
            MonitorPushTimer(0x47b317fULL);
        } else {
            MonitorPushTimer(0x47b32fcULL);
            const std::int32_t n_part = *reinterpret_cast<const std::int32_t*>(sim + 0x20);
            if (n_part != 0) {
                const __m128 alpha = _mm_set1_ps(1.0f - (new_sdt / prev_sdt));
                for (std::int32_t i = 0; i < n_part; ++i) {
                    u8* cur = *reinterpret_cast<u8* const*>(sim + 0x18);
                    u8* prev = *reinterpret_cast<u8* const*>(sim + 0x28);
                    const std::size_t off = std::size_t(i) << 4;
                    const __m128 c = _mm_load_ps(reinterpret_cast<const float*>(cur + off));
                    const __m128 p = _mm_load_ps(reinterpret_cast<const float*>(prev + off));
                    _mm_store_ps(reinterpret_cast<float*>(prev + off),
                                 _mm_add_ps(p, _mm_mul_ps(alpha, _mm_sub_ps(c, p))));
                }
            }
            *reinterpret_cast<float*>(sim + 0x108) = new_sdt;
            MonitorPushTimer(0x47b317fULL);

            MonitorPushTimer(0x47b3319ULL);
            const u8* cfg_d = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
            const float damp = *reinterpret_cast<const float*>(cfg_d + 0x10);
            *reinterpret_cast<float*>(sim + 0x10c) = ComputeEffectiveDampingPS4(damp, new_sdt);
            MonitorPushTimer(0x47b317fULL);
        }
    }

    // Rebuild sim->m_activeActions (+0x168/+0x170/+0x174) from all 5 action scopes.
    *reinterpret_cast<std::int32_t*>(sim + 0x170) = 0;

    u8* world = *reinterpret_cast<u8* const*>(step_info + 0x08);
    CollectActiveActionsFromArray(sim, reinterpret_cast<u8* const* const*>(world + 0x30),
                                  reinterpret_cast<const std::int32_t*>(world + 0x38));

    u8* ctx1 = *reinterpret_cast<u8* const*>(step_info + 0x10);
    u8* cloth_data = *reinterpret_cast<u8* const*>(ctx1 + 0x10);
    CollectActiveActionsFromArray(sim, reinterpret_cast<u8* const* const*>(cloth_data + 0x68),
                                  reinterpret_cast<const std::int32_t*>(cloth_data + 0x70));

    u8* sim_data = *reinterpret_cast<u8* const*>(sim + 0x10);
    CollectActiveActionsFromArray(sim, reinterpret_cast<u8* const* const*>(sim_data + 0xe8),
                                  reinterpret_cast<const std::int32_t*>(sim_data + 0xf0));

    u8* ctx2 = *reinterpret_cast<u8* const*>(step_info + 0x10);
    CollectActiveActionsFromArray(sim, reinterpret_cast<u8* const* const*>(ctx2 + 0x50),
                                  reinterpret_cast<const std::int32_t*>(ctx2 + 0x58));

    CollectActiveActionsFromArray(sim, reinterpret_cast<u8* const* const*>(sim + 0x110),
                                  reinterpret_cast<const std::int32_t*>(sim + 0x118));
}

// ============================================================================
// 32. hclSimulateOperator::simulate @ 0x076e370 (3276 B)
// ============================================================================
int PS4_SYSV_ABI ClothExp_hclSimulateOperator_simulate(u8* op, u8* ctx, float dt) noexcept {
    MonitorPushTimer(0x47b31afULL);

    const std::int32_t sim_idx = *reinterpret_cast<const std::int32_t*>(op + 0x1c);
    u8* sim = (*reinterpret_cast<u8* const* const*>(ctx + 0x20))[sim_idx];
    dt = dt / (*reinterpret_cast<const float*>(sim + 0x1d0));
    u8* sim_data = *reinterpret_cast<u8* const*>(sim + 0x10);

    const u8* cfg0 = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
    if (cfg0[0x1e] != 0u) {
        const std::int32_t tset_idx = *reinterpret_cast<const std::int32_t*>(sim_data + 0x150);
        const std::int32_t bone_idx = *reinterpret_cast<const std::int32_t*>(sim_data + 0x154);
        const u8* tset = (*reinterpret_cast<const u8* const* const*>(ctx + 0x40))[tset_idx];
        const u8* bone_ptr =
            *reinterpret_cast<const u8* const*>(tset + 0x10) + (std::size_t(bone_idx) << 6);

        HkTransform4x4 new_ref{};
        using SetQTransformFn = void(PS4_SYSV_ABI*)(void*, const void*);
        using TransferMotionFn =
            void(PS4_SYSV_ABI*)(const u8*, u8*, const void*, const void*, float);
        ResolveFn<SetQTransformFn>(0x043b8c0ULL)(&new_ref, bone_ptr);
        ResolveFn<TransferMotionFn>(0x0771290ULL)(op, sim, sim + 0x120, &new_ref, dt);

        _mm_store_ps(reinterpret_cast<float*>(sim + 0x120), new_ref.col[0]);
        _mm_store_ps(reinterpret_cast<float*>(sim + 0x130), new_ref.col[1]);
        _mm_store_ps(reinterpret_cast<float*>(sim + 0x140), new_ref.col[2]);
        _mm_store_ps(reinterpret_cast<float*>(sim + 0x150), new_ref.col[3]);
    }

    // Drive per-instance collidables from bone transform set (`sim_data + 0x80 >= 0`).
    if (*reinterpret_cast<const std::int32_t*>(sim_data + 0x80) >= 0) {
        MonitorPushTimer(0x47b31baULL);
        if (*reinterpret_cast<const std::int32_t*>(sim + 0xd8) > 0) {
            const std::int32_t col_tset_idx =
                *reinterpret_cast<const std::int32_t*>(sim_data + 0x80);
            const u8* col_tset =
                (*reinterpret_cast<const u8* const* const*>(ctx + 0x40))[col_tset_idx];
            using SetColTransformFn =
                void(PS4_SYSV_ABI*)(u8*, const float*, const float*, int, float);

            for (std::int32_t c = 0; c < *reinterpret_cast<const std::int32_t*>(sim + 0xd8); ++c) {
                u8* col = (*reinterpret_cast<u8* const* const*>(sim + 0xd0))[c];
                const std::int32_t b_idx =
                    (*reinterpret_cast<const std::int32_t* const*>(sim_data + 0x88))[c];
                const __m128i* off_ptr = reinterpret_cast<const __m128i*>(
                    *reinterpret_cast<const u8* const*>(sim_data + 0x98) +
                    std::size_t(c) * 0x40u);
                const __m128* bone_m = reinterpret_cast<const __m128*>(
                    *reinterpret_cast<const u8* const*>(col_tset + 0x10) +
                    (std::size_t(b_idx) << 6));

                const __m128 o0 = _mm_castsi128_ps(_mm_load_si128(off_ptr + 0));
                const __m128 o1 = _mm_castsi128_ps(_mm_load_si128(off_ptr + 1));
                const __m128 o2 = _mm_castsi128_ps(_mm_load_si128(off_ptr + 2));
                const __m128 o3 = _mm_castsi128_ps(_mm_load_si128(off_ptr + 3));

                const __m128 b0 = _mm_load_ps(reinterpret_cast<const float*>(bone_m + 0));
                const __m128 b1 = _mm_load_ps(reinterpret_cast<const float*>(bone_m + 1));
                const __m128 b2 = _mm_load_ps(reinterpret_cast<const float*>(bone_m + 2));
                const __m128 b3 = _mm_load_ps(reinterpret_cast<const float*>(bone_m + 3));

                HkTransform4x4 world_m{};
                world_m.col[0] = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(o0, o0, 0x00), b0),
                               _mm_mul_ps(_mm_shuffle_ps(o0, o0, 0x55), b1)),
                    _mm_mul_ps(_mm_shuffle_ps(o0, o0, 0xaa), b2));
                world_m.col[1] = _mm_add_ps(
                    _mm_mul_ps(b2, _mm_shuffle_ps(o1, o1, 0xaa)),
                    _mm_add_ps(_mm_mul_ps(b0, _mm_shuffle_ps(o1, o1, 0x00)),
                               _mm_mul_ps(b1, _mm_shuffle_ps(o1, o1, 0x55))));
                world_m.col[2] = _mm_add_ps(
                    _mm_mul_ps(b2, _mm_shuffle_ps(o2, o2, 0xaa)),
                    _mm_add_ps(_mm_mul_ps(b0, _mm_shuffle_ps(o2, o2, 0x00)),
                               _mm_mul_ps(b1, _mm_shuffle_ps(o2, o2, 0x55))));
                world_m.col[3] = _mm_add_ps(
                    _mm_mul_ps(b2, _mm_shuffle_ps(o3, o3, 0xaa)),
                    _mm_add_ps(_mm_mul_ps(b1, _mm_shuffle_ps(o3, o3, 0x55)),
                               _mm_add_ps(_mm_mul_ps(b0, _mm_shuffle_ps(o3, o3, 0x00)), b3)));

                ResolveFn<SetColTransformFn>(0x0767cf0ULL)(
                    col, reinterpret_cast<const float*>(&world_m),
                    reinterpret_cast<const float*>(col + 0x20), 0, dt);
            }
        }
        MonitorPushTimer(0x47b317fULL);
    }

    // Determine substep count & assemble active collidables array (`sim + 0x1c0`).
    const u8* cfg1 = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
    std::int32_t substeps = *reinterpret_cast<const std::int32_t*>(cfg1 + 0x18);
    if (substeps == 0) {
        substeps = *reinterpret_cast<const std::int32_t*>(op + 0x20);
    }
    const float f_substeps = static_cast<float>(substeps);
    const float substep_dt = dt / f_substeps;

    const std::int32_t n_inst_col = *reinterpret_cast<const std::int32_t*>(sim + 0xd8);
    const std::int32_t n_world_col = *reinterpret_cast<const std::int32_t*>(sim + 0xe8);

    const u8* cfg2 = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
    bool has_landscape = false;
    if (cfg2[0x1d] != 0u) {
        const u8* tri_src = *reinterpret_cast<const u8* const*>(sim + 0xf8);
        const u8* cvx_src = *reinterpret_cast<const u8* const*>(sim + 0x100);
        const bool any_src =
            (tri_src && *reinterpret_cast<const std::int32_t*>(tri_src + 0x18) != 0) ||
            (cvx_src && *reinterpret_cast<const std::int32_t*>(cvx_src + 0x18) != 0);
        if (any_src) {
            const u8* sd_chk = *reinterpret_cast<const u8* const*>(sim + 0x10);
            if (*reinterpret_cast<const std::int32_t*>(sd_chk + 0x148) != 0 &&
                *reinterpret_cast<const u8* const*>(sim + 0xf0) != nullptr) {
                has_landscape = true;
            }
        }
    }

    const std::int32_t base_col_count = n_inst_col + n_world_col;
    const std::int32_t total_col_count = base_col_count + (has_landscape ? 1 : 0);

    if (*reinterpret_cast<const std::int32_t*>(sim + 0x1bc) < total_col_count) {
        u8* old_buf = *reinterpret_cast<u8* const*>(sim + 0x1c0);
        u8** router_slot = GuestTlsSlot<u8*>(0x53e40f8ULL);
        u8* temp_alloc = *reinterpret_cast<u8* const*>(*router_slot + 0x58);
        ResolveFn<void(PS4_SYSV_ABI*)(u8*, void*)>(0x0429a90ULL)(temp_alloc, old_buf);
        u8* temp_alloc2 = *reinterpret_cast<u8* const*>(*router_slot + 0x58);
        u8* new_buf = ResolveFn<u8*(PS4_SYSV_ABI*)(u8*, int)>(0x0429a40ULL)(
            temp_alloc2, total_col_count * 0x90);
        *reinterpret_cast<u8**>(sim + 0x1c0) = new_buf;
        *reinterpret_cast<std::int32_t*>(sim + 0x1bc) = total_col_count;
    }
    *reinterpret_cast<std::int32_t*>(sim + 0x1b8) = total_col_count;

    using HkMemCpyFn = void(PS4_SYSV_ABI*)(void*, const void*, int);
    if (base_col_count > 0) {
        for (std::int32_t i = 0; i < base_col_count; ++i) {
            u8* dst = *reinterpret_cast<u8* const*>(sim + 0x1c0) + std::size_t(i) * 0x90u;
            const u8* src =
                (i < n_inst_col)
                    ? (*reinterpret_cast<const u8* const* const*>(sim + 0xd0))[i]
                    : (*reinterpret_cast<const u8* const* const*>(sim + 0xe0))[i - n_inst_col];
            ResolveFn<HkMemCpyFn>(0x0446140ULL)(dst, src, 0x90);
        }
    }
    if (has_landscape) {
        u8* dst =
            *reinterpret_cast<u8* const*>(sim + 0x1c0) + std::size_t(base_col_count) * 0x90u;
        const u8* src = *reinterpret_cast<const u8* const*>(sim + 0xf0);
        ResolveFn<HkMemCpyFn>(0x0446140ULL)(dst, src, 0x90);
    }

    // LIFO temporary buffers: saved fixed-particle pairs (`32 B` each) & pinch planes (`16 B` each).
    u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
    u8* prev_pos = *reinterpret_cast<u8* const*>(sim + 0x28);

    const std::int32_t n_fixed_init = *reinterpret_cast<const std::int32_t*>(sim_data + 0x50);
    const std::int32_t bytes_fixed = ((n_fixed_init << 5) + 0x7f) & ~0x7f;
    u8* fixed_pairs_buf = HkLifoAlloc(bytes_fixed);

    if (*reinterpret_cast<const std::int32_t*>(sim_data + 0x50) > 0) {
        __m128* dst_pair = reinterpret_cast<__m128*>(fixed_pairs_buf);
        for (std::int32_t i = 0; i < *reinterpret_cast<const std::int32_t*>(sim_data + 0x50); ++i) {
            const std::size_t p_off =
                std::size_t((*reinterpret_cast<const std::uint16_t* const*>(sim_data + 0x48))[i])
                << 4;
            _mm_store_ps(reinterpret_cast<float*>(dst_pair + 0),
                         _mm_load_ps(reinterpret_cast<const float*>(prev_pos + p_off)));
            _mm_store_ps(reinterpret_cast<float*>(dst_pair + 1),
                         _mm_load_ps(reinterpret_cast<const float*>(cur_pos + p_off)));
            dst_pair += 2;
        }
    }

    const std::int32_t n_part_init = *reinterpret_cast<const std::int32_t*>(sim + 0x20);
    const std::int32_t bytes_planes = ((n_part_init << 4) + 0x7f) & ~0x7f;
    u8* pinch_planes_buf = HkLifoAlloc(bytes_planes);

    // Substep simulation loop (`0x76e96d..0x76ee68`).
    if (substeps != 0) {
        using MatToQuatFn = void(PS4_SYSV_ABI*)(__m128*, const void*);
        using QuatToMatFn = void(PS4_SYSV_ABI*)(void*, const __m128*);
        using PinchReleaseFn = void(PS4_SYSV_ABI*)(const u8*, u8*, u8*);
        using IntegrateFn = void(PS4_SYSV_ABI*)(const u8*, u8*, u8*, float);
        using SolveFn = void(PS4_SYSV_ABI*)(const u8*, u8*, u8*, int, float);

        const __m128 sdt_vec = _mm_set1_ps(substep_dt);
        const __m128 half_sdt_vec = _mm_set1_ps(0.5f * substep_dt);

        for (std::int32_t s = 0; s < substeps; ++s) {
            MonitorPushTimer(0x47b31ceULL);
            if (total_col_count > 0) {
                for (std::int32_t k = 0; k < total_col_count; ++k) {
                    u8* col =
                        *reinterpret_cast<u8* const*>(sim + 0x1c0) + std::size_t(k) * 0x90u;

                    // Advance linear position: col.pos (+0x50) += substep_dt * col.linVel (+0x60)
                    const __m128 lin_v =
                        _mm_load_ps(reinterpret_cast<const float*>(col + 0x60));
                    const __m128 old_t =
                        _mm_load_ps(reinterpret_cast<const float*>(col + 0x50));
                    _mm_store_ps(reinterpret_cast<float*>(col + 0x50),
                                 _mm_add_ps(_mm_mul_ps(sdt_vec, lin_v), old_t));

                    // Advance rotation (`col + 0x20..0x4f`) via Havok half-angle polynomial.
                    alignas(16) __m128 q_cur{};
                    ResolveFn<MatToQuatFn>(0x0449900ULL)(&q_cur, col + 0x20);

                    const __m128 half_w = _mm_mul_ps(
                        half_sdt_vec, _mm_load_ps(reinterpret_cast<const float*>(col + 0x70)));
                    const float x = _mm_cvtss_f32(_mm_dp_ps(half_w, half_w, 0x7f));
                    constexpr float kC0 = std::bit_cast<float>(0x3ecf817au); // 0x47b3138
                    constexpr float kC1 = std::bit_cast<float>(0xbeaac436u); // 0x47b313c
                    constexpr float kC2 = std::bit_cast<float>(0xbe05a965u); // 0x47b3140
                    constexpr float kC3 = std::bit_cast<float>(0xbc93704cu); // 0x47b3144
                    const float t = x * kC0;
                    const float t2 = t * t;
                    const float w_comp = ((x * kC1 + 1.0f) + t2 * kC2) + (x * kC3) * t2;
                    const __m128 dq = _mm_insert_ps(half_w, _mm_set_ss(w_comp), 0x30);

                    // Quaternion product q_next = dq * q_cur (0x76ead3..0x76eb1b).
                    const __m128 q1 = q_cur;
                    const __m128 dq_w = _mm_shuffle_ps(dq, dq, 0xff);
                    const __m128 q1_w = _mm_shuffle_ps(q1, q1, 0xff);
                    __m128 cross = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(q1, q1, 0xc9), dq),
                                              _mm_mul_ps(q1, _mm_shuffle_ps(dq, dq, 0xc9)));
                    cross = _mm_shuffle_ps(cross, cross, 0xc9);
                    const __m128 xyz = _mm_add_ps(
                        _mm_mul_ps(q1_w, dq), _mm_add_ps(_mm_mul_ps(q1, dq_w), cross));
                    const __m128 prod = _mm_mul_ps(q1, dq);
                    const __m128 w_new =
                        _mm_sub_ps(_mm_shuffle_ps(prod, prod, 0xff), _mm_dp_ps(dq, q1, 0x7f));
                    alignas(16) __m128 q_next = NormalizeQuatNR(_mm_blend_ps(xyz, w_new, 0x08));

                    ResolveFn<QuatToMatFn>(0x0429b00ULL)(col + 0x20, &q_next);
                }
            }
            MonitorPushTimer(0x47b317fULL);

            ResolveFn<PinchReleaseFn>(0x076dcc0ULL)(op, sim, pinch_planes_buf);

            if (*reinterpret_cast<const std::int32_t*>(sim_data + 0x50) > 0) {
                const float frac = static_cast<float>(s + 1) / f_substeps;
                const __m128 w_prev = _mm_set1_ps(1.0f - frac);
                const __m128 w_cur = _mm_set1_ps(frac);
                const __m128* src_pair = reinterpret_cast<const __m128*>(fixed_pairs_buf);

                for (std::int32_t i = 0;
                     i < *reinterpret_cast<const std::int32_t*>(sim_data + 0x50); ++i) {
                    const __m128 p_interp = _mm_add_ps(
                        _mm_mul_ps(w_cur, _mm_load_ps(reinterpret_cast<const float*>(src_pair + 1))),
                        _mm_mul_ps(w_prev, _mm_load_ps(reinterpret_cast<const float*>(src_pair + 0))));
                    const std::size_t p_off = std::size_t(
                        (*reinterpret_cast<const std::uint16_t* const*>(sim_data + 0x48))[i])
                        << 4;
                    _mm_store_ps(reinterpret_cast<float*>(prev_pos + p_off), p_interp);
                    _mm_store_ps(reinterpret_cast<float*>(cur_pos + p_off), p_interp);
                    src_pair += 2;
                }
            }

            *reinterpret_cast<std::int32_t*>(sim + 0x1b0) = 0;
            ResolveFn<IntegrateFn>(0x0773df0ULL)(op, ctx, sim, substep_dt);

            if (*reinterpret_cast<const std::int32_t*>(op + 0x30) > 0) {
                ResolveFn<SolveFn>(0x0772180ULL)(op, ctx, sim, s, substep_dt);
            } else {
                ResolveFn<SolveFn>(0x0771f60ULL)(op, ctx, sim, s, substep_dt);
            }
        }
    }

    // Update AABBs (`0x76ee69..0x76eebb`).
    if (sim[0x1c8] != 0u) {
        ResolveFn<void(PS4_SYSV_ABI*)(u8*)>(0x0748ab0ULL)(sim);
    } else {
        ResolveFn<void(PS4_SYSV_ABI*)(u8*)>(0x0748a00ULL)(sim);
        const __m128 hi = HkMaxFloatVec();
        const __m128 lo = HkMinFloatVec();
        _mm_store_ps(reinterpret_cast<float*>(sim + 0x70), hi);
        _mm_store_ps(reinterpret_cast<float*>(sim + 0x80), lo);
        _mm_store_ps(reinterpret_cast<float*>(sim + 0x90), hi);
        _mm_store_ps(reinterpret_cast<float*>(sim + 0xa0), lo);
    }

    // Write back final substepped transforms to per-instance collidables (`0x76eebd..0x76ef3b`).
    if (*reinterpret_cast<const std::int32_t*>(sim_data + 0x80) >= 0 &&
        *reinterpret_cast<const std::int32_t*>(sim + 0xd8) > 0) {
        for (std::int32_t i = 0; i < *reinterpret_cast<const std::int32_t*>(sim + 0xd8); ++i) {
            u8* dst_col = (*reinterpret_cast<u8* const* const*>(sim + 0xd0))[i];
            const u8* src_col =
                *reinterpret_cast<const u8* const*>(sim + 0x1c0) + std::size_t(i) * 0x90u;
            _mm_store_ps(reinterpret_cast<float*>(dst_col + 0x20),
                         _mm_load_ps(reinterpret_cast<const float*>(src_col + 0x20)));
            _mm_store_ps(reinterpret_cast<float*>(dst_col + 0x30),
                         _mm_load_ps(reinterpret_cast<const float*>(src_col + 0x30)));
            _mm_store_ps(reinterpret_cast<float*>(dst_col + 0x40),
                         _mm_load_ps(reinterpret_cast<const float*>(src_col + 0x40)));
            _mm_store_ps(reinterpret_cast<float*>(dst_col + 0x50),
                         _mm_load_ps(reinterpret_cast<const float*>(src_col + 0x50)));
        }
    }

    MonitorPushTimer(0x47b317fULL);

    const u8* sd_end = *reinterpret_cast<const u8* const*>(sim + 0x10);
    if (sd_end[0x14c] != 0u) {
        ResolveFn<void(PS4_SYSV_ABI*)(const u8*, u8*)>(0x076f040ULL)(op, sim);
    }

    HkLifoFree(pinch_planes_buf, bytes_planes);
    HkLifoFree(fixed_pairs_buf, bytes_fixed);
    return 0;
}

// ============================================================================
// 33. hclSimulateOperator::transferMotion @ 0x0771290 (2054 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclSimulateOperator_transferMotion(const u8* /*op*/, u8* sim,
                                                              const void* old_xform,
                                                              const void* new_xform,
                                                              float dt) noexcept {
    MonitorPushTimer(0x47b3380ULL);

    const u8* sd = *reinterpret_cast<const u8* const*>(sim + 0x10);
    const float inv_dt = 1.0f / dt;

    using SetInverseFn = void(PS4_SYSV_ABI*)(void*, const void*);
    using SetMulFn = void(PS4_SYSV_ABI*)(void*, const void*, const void*);
    using MatToQuatFn = void(PS4_SYSV_ABI*)(__m128*, const void*);
    using QuatAngleFn = float(PS4_SYSV_ABI*)(const __m128*);
    using QuatAxisAngleFn = void(PS4_SYSV_ABI*)(__m128*, const __m128*, float);
    using QuatToMatFn = void(PS4_SYSV_ABI*)(void*, const __m128*);
    using TransformPosFn = void(PS4_SYSV_ABI*)(__m128*, const void*, const __m128*);

    HkTransform4x4 inv_old{};
    HkTransform4x4 delta_xform{};
    ResolveFn<SetInverseFn>(0x043b980ULL)(&inv_old, old_xform);
    ResolveFn<SetMulFn>(0x043b9f0ULL)(&delta_xform, &inv_old, new_xform);

    float trans_frac = 0.0f;
    if (sd[0x158] != 0u) {
        const __m128 lin_v = _mm_mul_ps(_mm_set1_ps(inv_dt), delta_xform.col[3]);
        const __m128 v_sq = _mm_dp_ps(lin_v, lin_v, 0x7f);
        const __m128 le_zero = _mm_cmple_ps(v_sq, _mm_setzero_ps());
        const __m128 r = _mm_rsqrt_ps(v_sq);
        const __m128 nr = _mm_mul_ps(_mm_mul_ps(r, _mm_set1_ps(0.5f)),
                                     _mm_sub_ps(_mm_set1_ps(3.0f), _mm_mul_ps(v_sq, _mm_mul_ps(r, r))));
        const float lin_speed = _mm_cvtss_f32(_mm_andnot_ps(le_zero, _mm_mul_ps(v_sq, nr)));

        trans_frac = EvalTransferRamp(lin_speed,
                                      *reinterpret_cast<const float*>(sd + 0x15c),
                                      *reinterpret_cast<const float*>(sd + 0x160),
                                      *reinterpret_cast<const float*>(sd + 0x164),
                                      *reinterpret_cast<const float*>(sd + 0x168));
    }

    alignas(16) __m128 delta_quat{};
    ResolveFn<MatToQuatFn>(0x0449900ULL)(&delta_quat, &delta_xform);

    float rot_frac = 0.0f;
    if (sd[0x16c] != 0u) {
        const float angle_rad = ResolveFn<QuatAngleFn>(0x04497d0ULL)(&delta_quat);
        constexpr float kRadToDeg = std::bit_cast<float>(0x42652ee0u); // 0x47b335c (57.295776f)
        const float deg_per_sec = (inv_dt * angle_rad) * kRadToDeg;

        rot_frac = EvalTransferRamp(deg_per_sec,
                                    *reinterpret_cast<const float*>(sd + 0x170),
                                    *reinterpret_cast<const float*>(sd + 0x174),
                                    *reinterpret_cast<const float*>(sd + 0x178),
                                    *reinterpret_cast<const float*>(sd + 0x17c));
    }

    if (trans_frac > 0.0f || rot_frac > 0.0f) {
        HkTransform4x4 local_blend{};
        SetIdentityTransform(local_blend);
        local_blend.col[3] = _mm_mul_ps(_mm_set1_ps(trans_frac), delta_xform.col[3]);

        const __m128 dq_v = delta_quat;
        const __m128 axis_sq = _mm_dp_ps(dq_v, dq_v, 0x7f);
        constexpr float kAxisSqEps = std::bit_cast<float>(0x28800000u); // 0x2bd5490 (~1.42e-14f)
        if (_mm_cvtss_f32(axis_sq) > kAxisSqEps) {
            const __m128 zero_v = _mm_setzero_ps();
            const __m128 le_zero = _mm_cmple_ps(axis_sq, zero_v);
            const __m128 r = _mm_rsqrt_ps(axis_sq);
            const __m128 nr = _mm_andnot_ps(
                le_zero,
                _mm_mul_ps(_mm_mul_ps(r, _mm_set1_ps(0.5f)),
                           _mm_sub_ps(_mm_set1_ps(3.0f), _mm_mul_ps(axis_sq, _mm_mul_ps(r, r)))));
            const __m128 axis = _mm_mul_ps(dq_v, nr);
            const __m128 w_bcast = _mm_shuffle_ps(dq_v, dq_v, 0xff);
            const __m128i sign_mask = _mm_slli_epi32(
                _mm_srli_epi32(_mm_castps_si128(_mm_cmplt_ps(w_bcast, zero_v)), 31), 31);
            alignas(16) const __m128 signed_axis =
                _mm_xor_ps(axis, _mm_castsi128_ps(sign_mask));

            const float full_angle = ResolveFn<QuatAngleFn>(0x04497d0ULL)(&delta_quat);
            alignas(16) __m128 scaled_quat{};
            ResolveFn<QuatAxisAngleFn>(0x0448e80ULL)(&scaled_quat, &signed_axis,
                                                     rot_frac * full_angle);
            ResolveFn<QuatToMatFn>(0x0429b00ULL)(&local_blend, &scaled_quat);
        }

        HkTransform4x4 tmp_xform{};
        HkTransform4x4 world_motion{};
        ResolveFn<SetMulFn>(0x043b9f0ULL)(&tmp_xform, old_xform, &local_blend);
        ResolveFn<SetMulFn>(0x043b9f0ULL)(&world_motion, &tmp_xform, &inv_old);

        const u8* sd_cur = *reinterpret_cast<const u8* const*>(sim + 0x10);
        __m128* cur_pos = *reinterpret_cast<__m128* const*>(sim + 0x18);
        __m128* prev_pos = *reinterpret_cast<__m128* const*>(sim + 0x28);
        const u8* pdata = *reinterpret_cast<const u8* const*>(sd_cur + 0x38);
        const std::uint32_t n_part = *reinterpret_cast<const std::uint32_t*>(sd_cur + 0x40);
        const __m128 zero_v = _mm_setzero_ps();

        for (std::uint32_t i = 0u; i < n_part; ++i) {
            alignas(16) const __m128 orig_c =
                _mm_load_ps(reinterpret_cast<const float*>(cur_pos + i));
            alignas(16) const __m128 orig_p =
                _mm_load_ps(reinterpret_cast<const float*>(prev_pos + i));
            const __m128 fixed_mask = _mm_cmpeq_ps(
                _mm_set1_ps(*reinterpret_cast<const float*>(pdata + std::size_t(i) * 0x10u)),
                zero_v);

            alignas(16) __m128 xf_c{};
            alignas(16) __m128 xf_p{};
            ResolveFn<TransformPosFn>(0x043c040ULL)(&xf_c, &world_motion, &orig_c);
            ResolveFn<TransformPosFn>(0x043c040ULL)(&xf_p, &world_motion, &orig_p);

            _mm_store_ps(reinterpret_cast<float*>(cur_pos + i),
                         _mm_blendv_ps(xf_c, orig_c, fixed_mask));
            _mm_store_ps(reinterpret_cast<float*>(prev_pos + i),
                         _mm_blendv_ps(xf_p, orig_p, fixed_mask));
        }
    }

    MonitorPushTimer(0x47b3392ULL);
}

// ============================================================================
// 34. hclSimulateOperator::integrate @ 0x0773df0 (1130 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclSimulateOperator_integrate(const u8* /*op*/, u8* /*ctx*/, u8* sim,
                                                         float substep_dt) noexcept {
    u8* guard_flag = reinterpret_cast<u8*>(ResolveVma(0x54435c0ULL));
    if (*guard_flag == 0u) {
        ResolveFn<void(PS4_SYSV_ABI*)()>(0x07703b0ULL)();
        ResolveFn<void(PS4_SYSV_ABI*)(u8*)>(0x07703c0ULL)(guard_flag);
        if (*guard_flag == 0u) {
            return;
        }
    }

    MonitorPushTimer(0x47b35b4ULL);

    const u8* sd = *reinterpret_cast<const u8* const*>(sim + 0x10);
    const u8* cfg = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
    const __m128 eff_damp = _mm_set1_ps(*reinterpret_cast<const float*>(sim + 0x10c));
    const std::int32_t n_part = *reinterpret_cast<const std::int32_t*>(sim + 0x20);

    const std::int32_t force_bytes = ((n_part << 4) + 0x7f) & ~0x7f;
    u8* raw_forces = nullptr;
    __m128* forces = nullptr;
    if (n_part != 0) {
        raw_forces = HkLifoAlloc(force_bytes);
        if (n_part > 0) {
            const __m128 zero_v = _mm_setzero_ps();
            __m128* f_out = reinterpret_cast<__m128*>(raw_forces);
            for (std::int32_t i = 0; i < n_part; ++i) {
                _mm_store_ps(reinterpret_cast<float*>(f_out + i), zero_v);
            }
        }
        forces = reinterpret_cast<__m128*>(raw_forces);
    }

    MonitorPushTimer(0x47b35c0ULL);

    if (*reinterpret_cast<const std::int32_t*>(sim + 0x170) > 0) {
        using ActionApplyFn = void(PS4_SYSV_ABI*)(u8*, u8*, __m128*, float);
        for (std::int32_t b = 0; b < *reinterpret_cast<const std::int32_t*>(sim + 0x170); ++b) {
            u8* action = (*reinterpret_cast<u8* const* const*>(sim + 0x168))[b];
            if (action != nullptr) {
                void* const* vtbl = *reinterpret_cast<void* const* const*>(action);
                reinterpret_cast<ActionApplyFn>(vtbl[0x28 / 8])(action, sim, forces, substep_dt);
            }
        }
    }

    const __m128 dt_sq = _mm_set1_ps(substep_dt * substep_dt);
    MonitorPushTimer(0x47b35caULL);

    const u8* pdata = *reinterpret_cast<const u8* const*>(sd + 0x38);
    __m128* cur = *reinterpret_cast<__m128* const*>(sim + 0x18);
    __m128* prev = *reinterpret_cast<__m128* const*>(sim + 0x28);
    const __m128 gravity = _mm_load_ps(reinterpret_cast<const float*>(cfg + 0x00));

    for (std::int32_t i = 0; i < n_part; ++i) {
        const u8* pd = pdata + std::size_t(i) * 0x10u;
        const __m128 mass = _mm_set1_ps(*reinterpret_cast<const float*>(pd + 0x00));
        const __m128 inv_mass = _mm_set1_ps(*reinterpret_cast<const float*>(pd + 0x04));
        const __m128 c_pos = _mm_load_ps(reinterpret_cast<const float*>(cur + i));
        const __m128 p_pos = _mm_load_ps(reinterpret_cast<const float*>(prev + i));

        const __m128 f_tot = _mm_add_ps(_mm_mul_ps(mass, gravity),
                                        _mm_load_ps(reinterpret_cast<const float*>(forces + i)));
        const __m128 accel_step = _mm_mul_ps(dt_sq, _mm_mul_ps(inv_mass, f_tot));
        const __m128 inertial =
            _mm_add_ps(c_pos, _mm_mul_ps(eff_damp, _mm_sub_ps(c_pos, p_pos)));

        _mm_store_ps(reinterpret_cast<float*>(cur + i), _mm_add_ps(inertial, accel_step));
        _mm_store_ps(reinterpret_cast<float*>(prev + i), c_pos);
    }

    MonitorPushTimer(0x47b35caULL);
    HkLifoFree(raw_forces, force_bytes);
}

// ============================================================================
// 35. hclSimulateOperator::pinchAndRelease @ 0x076dcc0 (1496 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclSimulateOperator_pinchAndRelease(const u8* /*op*/, u8* sim,
                                                               u8* planes_buf) noexcept {
    const u8* cfg = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
    if (cfg[0x1d] == 0u) {
        return;
    }

    const u8* tri_src = *reinterpret_cast<const u8* const*>(sim + 0xf8);
    const u8* cvx_src = *reinterpret_cast<const u8* const*>(sim + 0x100);
    const bool any_src =
        (tri_src && *reinterpret_cast<const std::int32_t*>(tri_src + 0x18) != 0) ||
        (cvx_src && *reinterpret_cast<const std::int32_t*>(cvx_src + 0x18) != 0);
    if (!any_src) {
        return;
    }

    const u8* sd0 = *reinterpret_cast<const u8* const*>(sim + 0x10);
    if (*reinterpret_cast<const std::int32_t*>(sd0 + 0x148) == 0) {
        return;
    }
    u8* landscape_col = *reinterpret_cast<u8* const*>(sim + 0xf0);
    if (landscape_col == nullptr) {
        return;
    }

    MonitorPushTimer(0x47b3155ULL);

    landscape_col = *reinterpret_cast<u8* const*>(sim + 0xf0);
    tri_src = *reinterpret_cast<const u8* const*>(sim + 0xf8);
    cvx_src = *reinterpret_cast<const u8* const*>(sim + 0x100);

    u8* lshape = *reinterpret_cast<u8* const*>(landscape_col + 0x88);
    *reinterpret_cast<u8**>(lshape + 0x10) = planes_buf;

    const u8* sd1 = *reinterpret_cast<const u8* const*>(sim + 0x10);
    const std::int32_t max_items = *reinterpret_cast<const std::int32_t*>(sd1 + 0x148) + 4;
    const std::int32_t work_bytes = ((max_items << 4) + 0x7f) & ~0x7f;

    struct alignas(16) WorkPtrHolder {
        u8* ptr;
        std::int32_t max_items;
        std::int32_t pad;
    } work_holder{HkLifoAlloc(work_bytes), max_items, 0};

    bool tri_valid = false;
    if (tri_src != nullptr) {
        const __m128 t_min = _mm_load_ps(reinterpret_cast<const float*>(tri_src + 0x20));
        const __m128 t_max = _mm_load_ps(reinterpret_cast<const float*>(tri_src + 0x30));
        if ((_mm_movemask_ps(_mm_cmplt_ps(t_max, t_min)) & 0x7) == 0) {
            tri_valid = (*reinterpret_cast<const std::int32_t*>(tri_src + 0x18) != 0);
        }
    }

    bool cvx_valid = false;
    if (cvx_src != nullptr) {
        const __m128 c_min = _mm_load_ps(reinterpret_cast<const float*>(cvx_src + 0x20));
        const __m128 c_max = _mm_load_ps(reinterpret_cast<const float*>(cvx_src + 0x30));
        if ((_mm_movemask_ps(_mm_cmplt_ps(c_max, c_min)) & 0x7) == 0) {
            cvx_valid = (*reinterpret_cast<const std::int32_t*>(cvx_src + 0x18) != 0);
        }
    }

    struct alignas(16) AabbPair {
        __m128 min_v;
        __m128 max_v;
    } merged_aabb{HkMaxFloatVec(), HkMinFloatVec()};

    if (tri_valid) {
        merged_aabb.min_v = _mm_min_ps(
            merged_aabb.min_v, _mm_load_ps(reinterpret_cast<const float*>(tri_src + 0x20)));
        merged_aabb.max_v = _mm_max_ps(
            merged_aabb.max_v, _mm_load_ps(reinterpret_cast<const float*>(tri_src + 0x30)));
    }
    if (cvx_valid) {
        merged_aabb.min_v = _mm_min_ps(
            merged_aabb.min_v, _mm_load_ps(reinterpret_cast<const float*>(cvx_src + 0x20)));
        merged_aabb.max_v = _mm_max_ps(
            merged_aabb.max_v, _mm_load_ps(reinterpret_cast<const float*>(cvx_src + 0x30)));
    }

    std::int32_t num_active = 0;
    using GatherPinchFn =
        void(PS4_SYSV_ABI*)(u8*, const AabbPair*, u8*, WorkPtrHolder*, std::int32_t*);
    ResolveFn<GatherPinchFn>(0x07738d0ULL)(sim, &merged_aabb, planes_buf, &work_holder,
                                           &num_active);

    if (num_active > 0) {
        if (tri_valid) {
            MonitorPushTimer(0x47b316cULL);
            std::int32_t rem = *reinterpret_cast<const std::int32_t*>(tri_src + 0x18);
            if (rem > 0) {
                const u8* tri_ptr = *reinterpret_cast<const u8* const*>(tri_src + 0x10);
                using CollideTrisFn = void(PS4_SYSV_ABI*)(u8*, const u8*, int, WorkPtrHolder*,
                                                          int, u8*);
                while (rem > 0) {
                    const std::int32_t batch = std::min<std::int32_t>(rem, 0xfff0);
                    ResolveFn<CollideTrisFn>(0x07739e0ULL)(sim, tri_ptr, batch, &work_holder,
                                                           num_active, planes_buf);
                    rem -= batch;
                    tri_ptr += std::size_t(batch) * 0x30u;
                }
            }
            MonitorPushTimer(0x47b317fULL);
        }

        if (cvx_valid) {
            MonitorPushTimer(0x47b3182ULL);
            const std::int32_t n_cvx = *reinterpret_cast<const std::int32_t*>(cvx_src + 0x18);
            if (n_cvx != 0) {
                const u8* cvx_ptr = *reinterpret_cast<const u8* const*>(cvx_src + 0x10);
                const __m128i xyz_mask = _mm_setr_epi32(-1, -1, -1, 0);
                using CollideConvexFn =
                    void(PS4_SYSV_ABI*)(u8*, const u8*, WorkPtrHolder*, int, u8*);

                for (std::int32_t i = 0; i < n_cvx; ++i, cvx_ptr += 0x70u) {
                    const __m128 cvx_min =
                        _mm_load_ps(reinterpret_cast<const float*>(cvx_ptr + 0x50));
                    const __m128 cvx_max =
                        _mm_load_ps(reinterpret_cast<const float*>(cvx_ptr + 0x60));
                    const __m128 sim_min =
                        _mm_load_ps(reinterpret_cast<const float*>(sim + 0x90));
                    const __m128 sim_max =
                        _mm_load_ps(reinterpret_cast<const float*>(sim + 0xa0));

                    const __m128i cmp_both = _mm_and_si128(
                        _mm_castps_si128(
                            _mm_and_ps(_mm_cmple_ps(cvx_min, sim_max),
                                       _mm_cmple_ps(sim_min, cvx_max))),
                        xyz_mask);
                    if (_mm_testc_si128(cmp_both, xyz_mask) != 0) {
                        ResolveFn<CollideConvexFn>(0x0773b50ULL)(sim, cvx_ptr, &work_holder,
                                                                 num_active, planes_buf);
                    }
                }
            }
            MonitorPushTimer(0x47b317fULL);
        }
    }

    const std::int32_t free_bytes = ((work_holder.max_items << 4) + 0x7f) & ~0x7f;
    HkLifoFree(work_holder.ptr, free_bytes);

    ResolveFn<void(PS4_SYSV_ABI*)(u8*, u8*)>(0x0773d10ULL)(sim, planes_buf);
    MonitorPushTimer(0x47b317fULL);

    MonitorPushTimer(0x47b3194ULL);
    const u8* sd2 = *reinterpret_cast<const u8* const*>(sim + 0x10);
    if (sd2[0x138] != 0u) {
        alignas(16) const __m128 max_stretch =
            _mm_set1_ps(*reinterpret_cast<const float*>(sd2 + 0x13c));
        using ReleaseStuckFn = void(PS4_SYSV_ABI*)(u8*, u8*, const __m128*, u8*);

        if (*reinterpret_cast<const std::int32_t*>(sd2 + 0xc0) > 0) {
            for (std::int32_t b = 0; b < *reinterpret_cast<const std::int32_t*>(sd2 + 0xc0); ++b) {
                u8* cset = (*reinterpret_cast<u8* const* const*>(sd2 + 0xb8))[b];
                const std::uint32_t ctype =
                    *reinterpret_cast<const std::uint32_t*>(cset + 0x18);
                if (ctype == 0x0du) {
                    ResolveFn<ReleaseStuckFn>(0x0772480ULL)(sim, cset, &max_stretch, planes_buf);
                } else if (ctype == 0x01u) {
                    ResolveFn<ReleaseStuckFn>(0x07723f0ULL)(sim, cset, &max_stretch, planes_buf);
                }
                sd2 = *reinterpret_cast<const u8* const*>(sim + 0x10);
            }
        }
    }
    MonitorPushTimer(0x47b317fULL);
}

// ============================================================================
// 36. hclSimulateOperator::updateParticleNormals @ 0x076f040 (2882 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclSimulateOperator_updateParticleNormals(const u8* /*op*/,
                                                                     u8* sim) noexcept {
    MonitorPushTimer(0x47b31e4ULL);

    const u8* sd = *reinterpret_cast<const u8* const*>(sim + 0x10);
    const __m128* cur_pos = *reinterpret_cast<const __m128* const*>(sim + 0x18);
    const std::int64_t n_indices = *reinterpret_cast<const std::int32_t*>(sd + 0x60);
    const std::int32_t n_normals = *reinterpret_cast<const std::int32_t*>(sd + 0x40);
    __m128* normals = *reinterpret_cast<__m128* const*>(sim + 0x38);

    const __m128 zero_v = _mm_setzero_ps();
    for (std::int32_t i = 0; i < n_normals; ++i) {
        _mm_store_ps(reinterpret_cast<float*>(normals + i), zero_v);
    }

    const std::int32_t n_tris = static_cast<std::int32_t>(n_indices / 3);
    const std::uint16_t* tri_idx = *reinterpret_cast<const std::uint16_t* const*>(sd + 0x58);
    const u8* flip_bytes = *reinterpret_cast<const u8* const*>(sd + 0x68);

    const __m128 one_v = _mm_set1_ps(1.0f);  // 0x2bd5280
    const __m128 four_v = _mm_set1_ps(4.0f); // 0x47b3120 (exact PS4 constant!)

    for (std::int32_t t = 0; t < n_tris; ++t) {
        const std::size_t i0 = tri_idx[std::size_t(t) * 3u + 0u];
        const std::size_t i1 = tri_idx[std::size_t(t) * 3u + 1u];
        const std::size_t i2 = tri_idx[std::size_t(t) * 3u + 2u];

        const __m128 p0 = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + i0));
        const __m128 p1 = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + i1));
        const __m128 p2 = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + i2));

        const __m128 e1 = _mm_sub_ps(p1, p0);
        const __m128 e2 = _mm_sub_ps(p2, p0);
        __m128 cross = _mm_sub_ps(_mm_mul_ps(e1, _mm_shuffle_ps(e2, e2, 0xc9)),
                                  _mm_mul_ps(e2, _mm_shuffle_ps(e1, e1, 0xc9)));
        cross = _mm_shuffle_ps(cross, cross, 0xc9);

        const int bit =
            static_cast<int>((flip_bytes[std::size_t(t) >> 3] >> (7 - (t & 7))) & 1u);
        const __m128 bit_f = _mm_cvtepi32_ps(_mm_set1_epi32(bit));
        const __m128 scale = _mm_sub_ps(one_v, _mm_mul_ps(bit_f, four_v));
        const __m128 n_tri = _mm_mul_ps(cross, scale);

        _mm_store_ps(reinterpret_cast<float*>(normals + i0),
                     _mm_add_ps(n_tri, _mm_load_ps(reinterpret_cast<const float*>(normals + i0))));
        _mm_store_ps(reinterpret_cast<float*>(normals + i1),
                     _mm_add_ps(n_tri, _mm_load_ps(reinterpret_cast<const float*>(normals + i1))));
        _mm_store_ps(reinterpret_cast<float*>(normals + i2),
                     _mm_add_ps(n_tri, _mm_load_ps(reinterpret_cast<const float*>(normals + i2))));
    }

    for (std::int32_t k = 0; k < n_normals; ++k) {
        const __m128 n = _mm_load_ps(reinterpret_cast<const float*>(normals + k));
        const __m128 len_sq = _mm_dp_ps(n, n, 0x7f);
        const __m128 inv_len =
            _mm_andnot_ps(_mm_cmple_ps(len_sq, zero_v), _mm_rsqrt_ps(len_sq));
        _mm_store_ps(reinterpret_cast<float*>(normals + k), _mm_mul_ps(n, inv_len));
    }

    MonitorPushTimer(0x47b317fULL);
}

} // extern "C"

} // namespace BbClothExp
