// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// File: gpu/cloth_exp/cloth_exp_constraints_and_col.cpp
//
// 100% 1-to-1 C++20 decompilation of PS4 eboot.bin Batch 2 (16 functions,
// 8,668 B PS4 code): Collidable motion velocity extraction, constraint solver
// drivers, multi-pass pinch & shape collision dispatch, and all 11 Havok Cloth
// constraint set solvers.
// Zero stubs, exact SIMD/register/memory semantics, exact hkMonitorStream tags.
// ============================================================================

#include "cloth_exp/cloth_exp_constraints_and_col.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "cloth_exp/cloth_exp_runtime.h"

namespace BbClothExp {

namespace {

using GetActiveCfgFn = u8*(PS4_SYSV_ABI*)(u8*);
using StiffnessFactorFn = float(PS4_SYSV_ABI*)(const u8*, int, int, int, float, float);
using ConstraintApplyFn = void(PS4_SYSV_ABI*)(u8*, u8*, u8*, int, int, float);
using CollideParticlesFn = void(PS4_SYSV_ABI*)(const u8*, u8*, u8*, float);
using CollideInternalFn = void(PS4_SYSV_ABI*)(u8*, int, int, int, float);
using MxApplyBatchFn = void(PS4_SYSV_ABI*)(const u8*, u8*, const __m128*);

/// Exact `0x2bd5480` (`[0x2bd5250 + 0x230]`): `0x34000000` (`1.1920928955078125e-07f`).
[[gnu::always_inline]] inline float HkFloatEps() noexcept {
    return std::bit_cast<float>(0x34000000u);
}

[[gnu::always_inline]] inline __m128 HkFloatEpsVec() noexcept {
    return _mm_set1_ps(HkFloatEps());
}

/// 1-iteration Newton-Raphson `rsqrt` normalization matching `0x767d72..0x767dd6`.
[[gnu::always_inline]] inline __m128 NormalizeQuatNR(__m128 q) noexcept {
    const __m128 half_v = _mm_set1_ps(0.5f);  // 0x2bd5eb0
    const __m128 three_v = _mm_set1_ps(3.0f); // 0x2bd5ee0
    const __m128 d = _mm_dp_ps(q, q, 0xff);
    const __m128 r = _mm_rsqrt_ps(d);
    const __m128 nr =
        _mm_mul_ps(_mm_mul_ps(r, half_v), _mm_sub_ps(three_v, _mm_mul_ps(d, _mm_mul_ps(r, r))));
    return _mm_mul_ps(q, nr);
}

/// Sphere-cage local range solver loop (`0x6fb350` aligned / `0x6fb450` unaligned).
template <bool AlignedVtx>
[[gnu::always_inline]] inline void SolveSphereLocalRangeLoop(const u8* cset, u8* cur_pos,
                                                             const u8* ref_buf,
                                                             __m128 stiff_vec) noexcept {
    const std::int32_t count = *reinterpret_cast<const std::int32_t*>(cset + 0x28);
    if (count <= 0) {
        return;
    }

    const __m128 eps_v = HkFloatEpsVec(); // [0x2bd5250 + 0x230]
    const __m128 zero_v = _mm_setzero_ps();
    const __m128 col0 = _mm_load_ps(reinterpret_cast<const float*>(ref_buf + 0x80));
    const __m128 col1 = _mm_load_ps(reinterpret_cast<const float*>(ref_buf + 0x90));
    const __m128 col2 = _mm_load_ps(reinterpret_cast<const float*>(ref_buf + 0xa0));
    const __m128 col3 = _mm_load_ps(reinterpret_cast<const float*>(ref_buf + 0xb0));

    for (std::int32_t i = 0; i < count; ++i) {
        const u8* entries = *reinterpret_cast<const u8* const*>(cset + 0x20);
        const u8* vtx_base = *reinterpret_cast<const u8* const*>(ref_buf + 0x10);
        const u8* entry = entries + std::size_t(i) * 0x10u;

        const std::size_t ref_vtx_idx = *reinterpret_cast<const std::uint16_t*>(entry + 0x02);
        const std::size_t stride = *reinterpret_cast<const u8*>(ref_buf + 0x1c);
        const auto* vtx_ptr = reinterpret_cast<const __m128i*>(vtx_base + ref_vtx_idx * stride);
        const __m128 v =
            AlignedVtx ? _mm_castsi128_ps(_mm_load_si128(vtx_ptr))
                       : _mm_castsi128_ps(_mm_lddqu_si128(vtx_ptr));

        const __m128 max_dist = _mm_set1_ps(*reinterpret_cast<const float*>(entry + 0x04));
        const __m128 vx = _mm_shuffle_ps(v, v, 0x00);
        const __m128 vy = _mm_shuffle_ps(v, v, 0x55);
        const __m128 vz = _mm_shuffle_ps(v, v, 0xaa);

        const __m128 wp = _mm_add_ps(
            _mm_add_ps(_mm_add_ps(_mm_mul_ps(vx, col0), col3), _mm_mul_ps(vy, col1)),
            _mm_mul_ps(vz, col2));

        const std::size_t p_off =
            std::size_t(*reinterpret_cast<const std::uint16_t*>(entry + 0x00)) << 4;
        const __m128 p = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + p_off));
        const __m128 diff = _mm_add_ps(eps_v, _mm_sub_ps(p, wp));
        const __m128 len_sq = _mm_dp_ps(diff, diff, 0x7f);
        const __m128 le_zero = _mm_cmple_ps(len_sq, zero_v);
        const __m128 inv_len = _mm_andnot_ps(le_zero, _mm_rsqrt_ps(len_sq));
        const __m128 dir = _mm_mul_ps(diff, inv_len);
        const __m128 len = _mm_mul_ps(len_sq, inv_len);
        const __m128 corr = _mm_min_ps(_mm_mul_ps(stiff_vec, _mm_sub_ps(max_dist, len)), zero_v);
        const __m128 new_p = _mm_add_ps(p, _mm_mul_ps(corr, dir));
        _mm_store_ps(reinterpret_cast<float*>(cur_pos + p_off), new_p);
    }
}

} // namespace

extern "C" {

// ============================================================================
// 15. hclCollidable::setTransform @ 0x0767cf0 (541 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclCollidable_setTransform(u8* col, const float* new_mat,
                                                      const float* old_mat, int normalize_quats,
                                                      float dt) noexcept {
    // Store old transform into col + 0x20..0x5f and linear displacement into col + 0x60.
    _mm_store_ps(reinterpret_cast<float*>(col + 0x20), _mm_load_ps(old_mat + 0));
    _mm_store_ps(reinterpret_cast<float*>(col + 0x30), _mm_load_ps(old_mat + 4));
    _mm_store_ps(reinterpret_cast<float*>(col + 0x40), _mm_load_ps(old_mat + 8));
    _mm_store_ps(reinterpret_cast<float*>(col + 0x50), _mm_load_ps(old_mat + 12));
    _mm_store_ps(reinterpret_cast<float*>(col + 0x60),
                 _mm_sub_ps(_mm_load_ps(new_mat + 12), _mm_load_ps(old_mat + 12)));

    using MatToQuatFn = void(PS4_SYSV_ABI*)(__m128*, const void*);
    using QuatAngleFn = float(PS4_SYSV_ABI*)(const __m128*);

    alignas(16) __m128 q_new{};
    alignas(16) __m128 q_old{};
    ResolveFn<MatToQuatFn>(0x0449900ULL)(&q_new, new_mat);
    ResolveFn<MatToQuatFn>(0x0449900ULL)(&q_old, col + 0x20);

    __m128 q0 = q_new;
    __m128 q1 = q_old;
    if (static_cast<u8>(normalize_quats) != 0u) {
        q0 = NormalizeQuatNR(q0);
        q_new = q0;
        q1 = NormalizeQuatNR(q1);
        q_old = q1;
    }

    // Exact delta quaternion dq = q0 * conj(q1) (0x767de2..0x767e23).
    __m128 x2 = _mm_shuffle_ps(q0, q0, 0xc9);
    x2 = _mm_mul_ps(q1, x2);
    __m128 x3 = _mm_shuffle_ps(q1, q1, 0xc9);
    x3 = _mm_mul_ps(q0, x3);
    x2 = _mm_sub_ps(x2, x3);
    x2 = _mm_shuffle_ps(x2, x2, 0xc9);
    x3 = _mm_shuffle_ps(q0, q0, 0xff);
    x3 = _mm_mul_ps(x3, q1);
    x2 = _mm_sub_ps(x2, x3);
    x3 = _mm_shuffle_ps(q1, q1, 0xff);
    x3 = _mm_mul_ps(q0, x3);
    x2 = _mm_add_ps(x3, x2);
    const __m128 dot4 = _mm_dp_ps(q0, q1, 0xff);
    alignas(16) __m128 dq = _mm_blend_ps(x2, dot4, 0x08);

    const float angle = ResolveFn<QuatAngleFn>(0x04497d0ULL)(&dq);
    bool wrote_ang_vel = false;
    if (angle > 0.0f) {
        const __m128 dq_v = dq;
        const __m128 axis_sq = _mm_dp_ps(dq_v, dq_v, 0x7f);
        if (_mm_cvtss_f32(axis_sq) > HkFloatEps()) {
            const __m128 zero_v = _mm_setzero_ps();
            const __m128 le_zero = _mm_cmple_ps(axis_sq, zero_v);
            const __m128 r = _mm_rsqrt_ps(axis_sq);
            const __m128 half_v = _mm_set1_ps(0.5f);
            const __m128 three_v = _mm_set1_ps(3.0f);
            const __m128 nr = _mm_andnot_ps(
                le_zero,
                _mm_mul_ps(_mm_mul_ps(r, half_v),
                           _mm_sub_ps(three_v, _mm_mul_ps(axis_sq, _mm_mul_ps(r, r)))));
            const __m128 axis = _mm_mul_ps(dq_v, nr);
            const __m128 w_bcast = _mm_shuffle_ps(dq_v, dq_v, 0xff);
            const __m128i sign_mask = _mm_slli_epi32(
                _mm_srli_epi32(_mm_castps_si128(_mm_cmplt_ps(w_bcast, zero_v)), 31), 31);
            const __m128 signed_axis = _mm_xor_ps(axis, _mm_castsi128_ps(sign_mask));
            const __m128 ang_speed = _mm_set1_ps(angle / dt);
            _mm_store_ps(reinterpret_cast<float*>(col + 0x70), _mm_mul_ps(ang_speed, signed_axis));
            wrote_ang_vel = true;
        }
    }
    if (!wrote_ang_vel) {
        _mm_store_ps(reinterpret_cast<float*>(col + 0x70), _mm_setzero_ps());
    }

    const __m128 inv_dt = _mm_set1_ps(1.0f / dt);
    _mm_store_ps(reinterpret_cast<float*>(col + 0x60),
                 _mm_mul_ps(inv_dt, _mm_load_ps(reinterpret_cast<const float*>(col + 0x60))));
}

// ============================================================================
// 16. hclSimulateOperator::solveConstraints @ 0x0771f60 (537 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclSimulateOperator_solveConstraints(const u8* op, u8* ctx, u8* sim,
                                                                int substep_idx,
                                                                float substep_dt) noexcept {
    MonitorPushTimer(0x47b33dcULL);

    const std::int32_t n_part = *reinterpret_cast<const std::int32_t*>(sim + 0x20);
    if (n_part >= 8) {
        const std::int32_t blocks = n_part / 8;
        const u8* p = *reinterpret_cast<const u8* const*>(sim + 0x18);
        for (std::int32_t d = 0; d < blocks; ++d, p += 0x80u) {
            _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_NTA);
        }
    }

    int mode = *reinterpret_cast<const std::int32_t*>(sim + 0x1cc);
    const float f0 = *reinterpret_cast<const float*>(sim + 0x1d0);
    const float f1 = *reinterpret_cast<const float*>(sim + 0x1d4);
    if (mode == 1) {
        mode = (op[0x38] != 0u) ? 2 : 1;
    }

    const u8* cfg = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
    int substeps = *reinterpret_cast<const std::int32_t*>(cfg + 0x18);
    if (substeps == 0) {
        substeps = *reinterpret_cast<const std::int32_t*>(op + 0x20);
    }

    int use_factor = 0;
    if (mode != 1) {
        if (substeps != 1) {
            use_factor = 1;
        } else {
            use_factor = (f0 != 1.0f || f1 != 1.0f) ? 1 : 0;
        }
    }

    if (*reinterpret_cast<const std::int32_t*>(op + 0x24) > 0) {
        u8* sim_data = *reinterpret_cast<u8* const*>(sim + 0x10);
        for (std::int32_t iter = 0; iter < *reinterpret_cast<const std::int32_t*>(op + 0x24);
             ++iter) {
            const std::int32_t n_csets = *reinterpret_cast<const std::int32_t*>(sim_data + 0xc0);
            for (std::int32_t i = 0; i < n_csets; ++i) {
                u8* cset = (*reinterpret_cast<u8* const* const*>(sim_data + 0xb8))[i];
                const float factor = ResolveFn<StiffnessFactorFn>(0x1167a80ULL)(
                    cset, mode, substep_idx, substeps, f0, f1);
                void* const* vtbl = *reinterpret_cast<void* const* const*>(cset);
                reinterpret_cast<ConstraintApplyFn>(vtbl[0x50 / 8])(cset, sim, ctx, i, use_factor,
                                                                    factor);
            }
            ResolveFn<CollideParticlesFn>(0x0774260ULL)(op, ctx, sim, substep_dt);
        }
    }

    MonitorPushTimer(0x47b33e4ULL);
}

// ============================================================================
// 17. hclSimulateOperator::solveConstraintsExec @ 0x0772180 (620 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclSimulateOperator_solveConstraintsExec(const u8* op, u8* ctx, u8* sim,
                                                                    int substep_idx,
                                                                    float substep_dt) noexcept {
    MonitorPushTimer(0x47b33dcULL);

    u8* sim_data = *reinterpret_cast<u8* const*>(sim + 0x10);
    int mode = *reinterpret_cast<const std::int32_t*>(sim + 0x1cc);
    const float f0 = *reinterpret_cast<const float*>(sim + 0x1d0);
    const float f1 = *reinterpret_cast<const float*>(sim + 0x1d4);
    if (mode == 1) {
        mode = (op[0x38] != 0u) ? 2 : 1;
    }

    const u8* cfg = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
    int substeps = *reinterpret_cast<const std::int32_t*>(cfg + 0x18);
    if (substeps == 0) {
        substeps = *reinterpret_cast<const std::int32_t*>(op + 0x20);
    }

    int use_factor = 0;
    if (mode != 1) {
        if (substeps != 1) {
            use_factor = 1;
        } else {
            use_factor = (f0 != 1.0f || f1 != 1.0f) ? 1 : 0;
        }
    }

    const std::int32_t n_iter_init = *reinterpret_cast<const std::int32_t*>(op + 0x24);
    if (n_iter_init > 0) {
        for (std::int32_t iter = 0; iter < *reinterpret_cast<const std::int32_t*>(op + 0x24);
             ++iter) {
            for (std::int32_t k = 0; k < *reinterpret_cast<const std::int32_t*>(op + 0x30); ++k) {
                const std::int32_t exec_id =
                    (*reinterpret_cast<const std::int32_t* const*>(op + 0x28))[k];
                if (exec_id == -1) {
                    ResolveFn<CollideParticlesFn>(0x0774260ULL)(op, ctx, sim, substep_dt);
                } else {
                    u8* cset = (*reinterpret_cast<u8* const* const*>(sim_data + 0xb8))[exec_id];
                    const float factor = ResolveFn<StiffnessFactorFn>(0x1167a80ULL)(
                        cset, mode, substep_idx, substeps, f0, f1);
                    const std::int32_t idx_arg =
                        (*reinterpret_cast<const std::int32_t* const*>(op + 0x28))[k];
                    void* const* vtbl = *reinterpret_cast<void* const* const*>(cset);
                    reinterpret_cast<ConstraintApplyFn>(vtbl[0x50 / 8])(cset, sim, ctx, idx_arg,
                                                                        use_factor, factor);
                }
            }
        }
    }

    const u8* cfg2 = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
    if (cfg2[0x1c] != 0u) {
        u8* sd = *reinterpret_cast<u8* const*>(sim + 0x10);
        if (*reinterpret_cast<const std::int32_t*>(sd + 0xd0) > 0) {
            using UpdateTransitionFn = void(PS4_SYSV_ABI*)(u8*, float);
            for (std::int32_t b = 0; b < *reinterpret_cast<const std::int32_t*>(sd + 0xd0); ++b) {
                const u8* ap_cset = (*reinterpret_cast<const u8* const* const*>(sd + 0xc8))[b];
                if (*reinterpret_cast<const std::uint32_t*>(ap_cset + 0x18) == 0x13u) {
                    const std::int32_t n_map =
                        *reinterpret_cast<const std::int32_t*>(sim + 0xc8);
                    const u8* map_base = *reinterpret_cast<const u8* const*>(sim + 0xc0);
                    u8* trans_inst = nullptr;
                    for (std::int32_t m = 0; m < n_map; ++m) {
                        const u8* entry = map_base + std::size_t(m) * 0x10u;
                        if (*reinterpret_cast<const std::int32_t*>(entry) == b) {
                            trans_inst = *reinterpret_cast<u8* const*>(entry + 0x08);
                            break;
                        }
                    }
                    ResolveFn<UpdateTransitionFn>(0x0771c80ULL)(trans_inst, substep_dt);
                    sd = *reinterpret_cast<u8* const*>(sim + 0x10);
                }
            }
        }
    }

    MonitorPushTimer(0x47b33e4ULL);
}

// ============================================================================
// 18. hclSimulateOperator::collideParticles @ 0x0774260 (1261 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclSimulateOperator_collideParticles(const u8* /*op*/, u8* ctx, u8* sim,
                                                                float substep_dt) noexcept {
    u8* guard_flag = reinterpret_cast<u8*>(ResolveVma(0x54435c4ULL));
    if (*guard_flag == 0u) {
        ResolveFn<void(PS4_SYSV_ABI*)()>(0x07703b0ULL)();
        ResolveFn<void(PS4_SYSV_ABI*)(u8*)>(0x07703c0ULL)(guard_flag);
        if (*guard_flag == 0u) {
            return;
        }
    }

    MonitorPushTimer(0x47b3880ULL);

    const u8* cfg = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
    const std::int32_t n_col = *reinterpret_cast<const std::int32_t*>(sim + 0x1b8);
    std::int32_t pinch_count = 0;
    if (cfg[0x1c] != 0u && n_col > 0) {
        const u8* col_base = *reinterpret_cast<const u8* const*>(sim + 0x1c0);
        for (std::int32_t i = 0; i < n_col; ++i) {
            if (col_base[std::size_t(i) * 0x90u + 0x80u] != 0u) {
                ++pinch_count;
            }
        }
    }

    if (cfg[0x1c] == 0u || n_col <= 0 || pinch_count <= 1) {
        ResolveFn<CollideInternalFn>(0x0774750ULL)(sim, 1, 1, 0, substep_dt);
        MonitorPushTimer(0x47b38a9ULL);
        return;
    }

    // Multi-collider pinch detection & contact plane resolution path (0x7743ba..0x7746f3).
    MonitorPushTimer(0x47b3894ULL);

    const u8* sd = *reinterpret_cast<const u8* const*>(sim + 0x10);
    const std::uint16_t min_p = *reinterpret_cast<const std::uint16_t*>(sd + 0x128);
    const std::uint16_t max_p = *reinterpret_cast<const std::uint16_t*>(sd + 0x12a);
    const std::uint16_t span16 =
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(max_p + 1u) - min_p);
    const std::uint32_t aligned_bytes = (static_cast<std::uint32_t>(span16) + 0xfu) & 0x1fff0u;
    const std::uint32_t vec_count = aligned_bytes >> 4;
    if (vec_count != 0u) {
        const __m128 zero_v = _mm_setzero_ps();
        __m128* flags_a = *reinterpret_cast<__m128* const*>(sim + 0x198);
        for (std::uint32_t i = 0; i < vec_count; ++i) {
            _mm_store_ps(reinterpret_cast<float*>(flags_a + i), zero_v);
        }
        __m128* flags_b = *reinterpret_cast<__m128* const*>(sim + 0x1a0);
        for (std::uint32_t i = 0; i < vec_count; ++i) {
            _mm_store_ps(reinterpret_cast<float*>(flags_b + i), zero_v);
        }
    }
    const std::uint32_t dword_count = aligned_bytes >> 2;
    if (dword_count != 0u) {
        void* contact_ids = *reinterpret_cast<void* const*>(sim + 0x1a8);
        std::memset(contact_ids, 0xff, std::size_t(dword_count) * 4u);
    }

    MonitorPushTimer(0x47b38a9ULL);
    MonitorPushTimer(0x47b38acULL);

    ResolveFn<CollideInternalFn>(0x0774750ULL)(sim, 1, 0, 1, substep_dt);

    MonitorPushTimer(0x47b38a9ULL);
    MonitorPushTimer(0x47b38cfULL);

    const u8* sd2 = *reinterpret_cast<const u8* const*>(sim + 0x10);
    const std::uint16_t p_lo = *reinterpret_cast<const std::uint16_t*>(sd2 + 0x128);
    const std::uint16_t p_hi = *reinterpret_cast<const std::uint16_t*>(sd2 + 0x12a);
    if (p_lo <= p_hi) {
        const __m128 dt_vec = _mm_set1_ps(substep_dt);
        std::uint16_t rel = 0u;
        for (;;) {
            const std::size_t idx = rel;
            const u8* has_contact = *reinterpret_cast<const u8* const*>(sim + 0x198);
            if (has_contact[idx] != 0u) {
                const std::size_t p_idx = static_cast<std::uint16_t>(p_lo + rel);
                const u8* contacts = *reinterpret_cast<const u8* const*>(sim + 0x188);
                u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
                u8* prev_pos = *reinterpret_cast<u8* const*>(sim + 0x28);
                const u8* pdata = *reinterpret_cast<const u8* const*>(sd2 + 0x38);

                const std::size_t p_off = p_idx << 4;
                const std::size_t c_off = idx * 0x30u;

                const __m128 friction =
                    _mm_set1_ps(*reinterpret_cast<const float*>(pdata + p_off + 0x0c));
                const __m128 radius =
                    _mm_set1_ps(*reinterpret_cast<const float*>(pdata + p_off + 0x08));
                const __m128 pos = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + p_off));
                const __m128 old_p = _mm_load_ps(reinterpret_cast<const float*>(prev_pos + p_off));
                const __m128 c_pt =
                    _mm_load_ps(reinterpret_cast<const float*>(contacts + c_off + 0x00));
                const __m128 c_nrm =
                    _mm_load_ps(reinterpret_cast<const float*>(contacts + c_off + 0x10));
                const __m128 c_vel =
                    _mm_load_ps(reinterpret_cast<const float*>(contacts + c_off + 0x20));

                const __m128 dist = _mm_dp_ps(_mm_sub_ps(pos, c_pt), c_nrm, 0x7f);
                const __m128 new_p = _mm_add_ps(pos, _mm_mul_ps(c_nrm, _mm_sub_ps(radius, dist)));
                __m128 rel_disp = _mm_sub_ps(_mm_sub_ps(new_p, old_p), _mm_mul_ps(dt_vec, c_vel));
                const __m128 n_comp = _mm_mul_ps(c_nrm, _mm_dp_ps(rel_disp, c_nrm, 0x7f));
                rel_disp = _mm_sub_ps(rel_disp, n_comp);
                const __m128 new_old_p = _mm_add_ps(old_p, _mm_mul_ps(friction, rel_disp));

                _mm_store_ps(reinterpret_cast<float*>(cur_pos + p_off), new_p);
                _mm_store_ps(reinterpret_cast<float*>(prev_pos + p_off), new_old_p);
            }

            const std::uint16_t next_p = static_cast<std::uint16_t>(p_lo + idx + 1u);
            rel = static_cast<std::uint16_t>(idx + 1u);
            if (next_p > p_hi) {
                break;
            }
        }
    }

    MonitorPushTimer(0x47b38a9ULL);

    ResolveFn<CollideInternalFn>(0x0774750ULL)(sim, 0, 1, 0, substep_dt);

    u8* sd3 = *reinterpret_cast<u8* const*>(sim + 0x10);
    if (*reinterpret_cast<const std::int32_t*>(sd3 + 0xd0) > 0) {
        for (std::int32_t b = 0; b < *reinterpret_cast<const std::int32_t*>(sd3 + 0xd0); ++b) {
            u8* cset = (*reinterpret_cast<u8* const* const*>(sd3 + 0xc8))[b];
            void* const* vtbl = *reinterpret_cast<void* const* const*>(cset);
            reinterpret_cast<ConstraintApplyFn>(vtbl[0x50 / 8])(cset, sim, ctx, b, 0, 1.0f);
            sd3 = *reinterpret_cast<u8* const*>(sim + 0x10);
        }
    }

    MonitorPushTimer(0x47b38a9ULL);
}

// ============================================================================
// 19. hclSimClothInstance::collideParticlesInternal @ 0x0774750 (968 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclSimClothInstance_collideParticlesInternal(u8* sim, int do_pinch,
                                                                        int do_non_pinch,
                                                                        int record_contacts,
                                                                        float substep_dt) noexcept {
    if (*reinterpret_cast<const std::int32_t*>(sim + 0x1b8) <= 0) {
        return;
    }

    const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
    const std::int32_t static_col_count = *reinterpret_cast<const std::int32_t*>(sim_data + 0xb0);
    const bool pinch_pass = (static_cast<u8>(do_pinch) != 0u);
    const bool non_pinch_pass = (static_cast<u8>(do_non_pinch) != 0u);
    const int rec_arg = static_cast<int>(static_cast<u8>(record_contacts));

    using ShapeCollideFn = void(PS4_SYSV_ABI*)(int, int, u8*, int, int, float);

    for (std::int32_t i = 0; i < *reinterpret_cast<const std::int32_t*>(sim + 0x1b8); ++i) {
        const u8* col =
            *reinterpret_cast<const u8* const*>(sim + 0x1c0) + std::size_t(i) * 0x90u;
        const u8 is_pinch = col[0x80];

        if (pinch_pass) {
            if (is_pinch == 0u && !non_pinch_pass) {
                continue;
            }
        } else {
            if (is_pinch != 0u || !non_pinch_pass) {
                continue;
            }
        }

        const u8* shape = *reinterpret_cast<const u8* const*>(col + 0x88);
        const std::uint32_t shape_type = *reinterpret_cast<const std::uint32_t*>(shape + 0x0c);
        const int pinch_arg = (is_pinch != 0u) ? 1 : 0;
        const int is_dyn = (i >= static_col_count) ? 1 : 0;

        switch (shape_type) {
        case 0u:
            ResolveFn<ShapeCollideFn>(0x257f000ULL)(i, is_dyn, sim, pinch_arg, rec_arg, substep_dt);
            break;
        case 1u:
            ResolveFn<ShapeCollideFn>(0x257e620ULL)(i, is_dyn, sim, pinch_arg, rec_arg, substep_dt);
            break;
        case 2u:
            ResolveFn<ShapeCollideFn>(0x257bc20ULL)(i, is_dyn, sim, pinch_arg, rec_arg, substep_dt);
            break;
        case 3u:
            ResolveFn<ShapeCollideFn>(0x257c600ULL)(i, is_dyn, sim, pinch_arg, rec_arg, substep_dt);
            break;
        case 5u:
            ResolveFn<ShapeCollideFn>(0x257d140ULL)(i, is_dyn, sim, pinch_arg, rec_arg, substep_dt);
            break;
        case 9u:
            ResolveFn<ShapeCollideFn>(0x257dc40ULL)(i, is_dyn, sim, pinch_arg, rec_arg, substep_dt);
            break;
        case 10u:
            // Shape 10 explicitly passes esi = 0 in both jump tables (0x7748ec / 0x774a5b).
            ResolveFn<ShapeCollideFn>(0x257f9e0ULL)(i, 0, sim, pinch_arg, rec_arg, substep_dt);
            break;
        case 11u:
            ResolveFn<ShapeCollideFn>(0x25803d0ULL)(i, is_dyn, sim, pinch_arg, rec_arg, substep_dt);
            break;
        default:
            break;
        }
    }
}

// ============================================================================
// 20. hclLocalRangeConstraintSet::apply @ 0x06fb220 (1547 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclLocalRangeConstraintSet_apply(const u8* cset, u8* sim, u8* ctx,
                                                            int /*cset_idx*/, int /*use_factor*/,
                                                            float factor) noexcept {
    if (!(factor > 0.0f)) {
        return;
    }

    const std::int32_t buf_idx = *reinterpret_cast<const std::int32_t*>(cset + 0x30);
    u8* const* bufs = *reinterpret_cast<u8* const* const*>(ctx + 0x30);
    const u8* raw_buf = bufs[buf_idx];
    const std::int32_t shadow_idx = *reinterpret_cast<const std::int32_t*>(raw_buf + 0x100);
    const u8* ref_buf = bufs[shadow_idx];

    u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
    u8* prev_pos = *reinterpret_cast<u8* const*>(sim + 0x28);

    MonitorPushTimer(0x47acae2ULL);

    const std::uint32_t pos_aligned = *reinterpret_cast<const std::uint32_t*>(ref_buf + 0x20) & 1u;
    if (cset[0x3c] != 0u && *reinterpret_cast<const u8* const*>(ref_buf + 0x38) != nullptr) {
        using NormalRangeFn = void(PS4_SYSV_ABI*)(const u8*, u8*, u8*, const u8*, std::uint32_t,
                                                  const __m128*);
        const std::uint32_t nrm_flags = *reinterpret_cast<const std::uint32_t*>(ref_buf + 0x48);
        const std::uint32_t shape_mode = *reinterpret_cast<const std::uint32_t*>(cset + 0x38);
        alignas(16) const __m128 fac_vec = _mm_set1_ps(factor);

        if ((nrm_flags & pos_aligned) != 0u) {
            ResolveFn<NormalRangeFn>(0x2553210ULL)(cset, cur_pos, prev_pos, ref_buf, shape_mode,
                                                   &fac_vec);
        } else if (pos_aligned != 0u) {
            ResolveFn<NormalRangeFn>(0x2553510ULL)(cset, cur_pos, prev_pos, ref_buf, shape_mode,
                                                   &fac_vec);
        } else if ((nrm_flags & 1u) != 0u) {
            ResolveFn<NormalRangeFn>(0x2553810ULL)(cset, cur_pos, prev_pos, ref_buf, shape_mode,
                                                   &fac_vec);
        } else {
            ResolveFn<NormalRangeFn>(0x2553b10ULL)(cset, cur_pos, prev_pos, ref_buf, shape_mode,
                                                   &fac_vec);
        }
    } else {
        const __m128 stiff_vec =
            _mm_set1_ps(factor * (*reinterpret_cast<const float*>(cset + 0x34)));
        if (pos_aligned != 0u) {
            SolveSphereLocalRangeLoop<true>(cset, cur_pos, ref_buf, stiff_vec);
        } else {
            SolveSphereLocalRangeLoop<false>(cset, cur_pos, ref_buf, stiff_vec);
        }
    }

    MonitorPushTimer(0x47acafcULL);
}

// ============================================================================
// 21. hclStretchLinkConstraintSet::apply @ 0x071ac50 (270 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclStretchLinkConstraintSet_apply(const u8* cset, u8* sim, u8* /*ctx*/,
                                                             int /*cset_idx*/, int /*use_factor*/,
                                                             float factor) noexcept {
    if (!(factor > 0.0f)) {
        return;
    }

    u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
    MonitorPushTimer(0x47af18cULL);

    const std::int32_t count = *reinterpret_cast<const std::int32_t*>(cset + 0x28);
    if (count > 0) {
        const u8* link = *reinterpret_cast<const u8* const*>(cset + 0x20);
        const __m128 fac_vec = _mm_set1_ps(factor);
        const __m128 zero_v = _mm_setzero_ps();

        for (std::int32_t i = 0; i < count; ++i, link += 0x0cu) {
            const __m128 stiff = _mm_set1_ps(*reinterpret_cast<const float*>(link + 0x08));
            const __m128 rest = _mm_set1_ps(*reinterpret_cast<const float*>(link + 0x04));
            const std::size_t off_b =
                std::size_t(*reinterpret_cast<const std::uint16_t*>(link + 0x02)) << 4;
            const std::size_t off_a =
                std::size_t(*reinterpret_cast<const std::uint16_t*>(link + 0x00)) << 4;

            const __m128 pos_b = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_b));
            const __m128 diff =
                _mm_sub_ps(pos_b, _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_a)));
            const __m128 len_sq = _mm_dp_ps(diff, diff, 0x7f);
            const __m128 inv_len =
                _mm_andnot_ps(_mm_cmple_ps(len_sq, zero_v), _mm_rsqrt_ps(len_sq));
            const __m128 dir = _mm_mul_ps(diff, inv_len);
            const __m128 len = _mm_mul_ps(len_sq, inv_len);
            const __m128 delta = _mm_min_ps(_mm_sub_ps(rest, len), zero_v);
            const __m128 scale = _mm_mul_ps(fac_vec, _mm_mul_ps(stiff, delta));
            _mm_store_ps(reinterpret_cast<float*>(cur_pos + off_b),
                         _mm_add_ps(pos_b, _mm_mul_ps(scale, dir)));
        }
    }

    MonitorPushTimer(0x47af1a2ULL);
}

// ============================================================================
// 22. hclStretchLinkConstraintSetMx::apply @ 0x0720dc0 (224 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclStretchLinkConstraintSetMx_apply(const u8* cset, u8* sim, u8* /*ctx*/,
                                                               int /*cset_idx*/, int /*use_factor*/,
                                                               float factor) noexcept {
    if (!(factor > 0.0f)) {
        return;
    }

    u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
    MonitorPushTimer(0x47afd28ULL);

    alignas(16) const __m128 fac_batch = _mm_set1_ps(factor);
    ResolveFn<MxApplyBatchFn>(0x2556440ULL)(cset, cur_pos, &fac_batch);

    alignas(16) const __m128 fac_single = _mm_set1_ps(factor);
    ResolveFn<MxApplyBatchFn>(0x0748c70ULL)(cset, cur_pos, &fac_single);

    MonitorPushTimer(0x47afd53ULL);
}

// ============================================================================
// 23. hclStandardLinkConstraintSet::apply @ 0x07422b0 (329 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclStandardLinkConstraintSet_apply(const u8* cset, u8* sim, u8* /*ctx*/,
                                                              int /*cset_idx*/, int /*use_factor*/,
                                                              float factor) noexcept {
    if (!(factor > 0.0f)) {
        return;
    }

    const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
    u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
    const u8* pdata = *reinterpret_cast<const u8* const*>(sim_data + 0x38);
    MonitorPushTimer(0x47b1ce0ULL);

    const std::int32_t count = *reinterpret_cast<const std::int32_t*>(cset + 0x28);
    if (count > 0) {
        const u8* link = *reinterpret_cast<const u8* const*>(cset + 0x20);
        const __m128 fac_vec = _mm_set1_ps(factor);
        const __m128 zero_v = _mm_setzero_ps();

        for (std::int32_t i = 0; i < count; ++i, link += 0x0cu) {
            const std::size_t off_b =
                std::size_t(*reinterpret_cast<const std::uint16_t*>(link + 0x02)) << 4;
            const std::size_t off_a =
                std::size_t(*reinterpret_cast<const std::uint16_t*>(link + 0x00)) << 4;

            const __m128 pos_b = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_b));
            const __m128 pos_a = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_a));
            const __m128 diff = _mm_sub_ps(pos_b, pos_a);
            const __m128 len_sq = _mm_dp_ps(diff, diff, 0x7f);
            const __m128 inv_len =
                _mm_andnot_ps(_mm_cmple_ps(len_sq, zero_v), _mm_rsqrt_ps(len_sq));
            const __m128 dir = _mm_mul_ps(diff, inv_len);
            const __m128 len = _mm_mul_ps(len_sq, inv_len);

            const __m128 inv_mass_a =
                _mm_set1_ps(*reinterpret_cast<const float*>(pdata + off_a + 0x04));
            const __m128 inv_mass_b =
                _mm_set1_ps(*reinterpret_cast<const float*>(pdata + off_b + 0x04));
            const __m128 stiff = _mm_set1_ps(*reinterpret_cast<const float*>(link + 0x08));
            const __m128 rest = _mm_set1_ps(*reinterpret_cast<const float*>(link + 0x04));

            const __m128 imp =
                _mm_mul_ps(dir, _mm_mul_ps(fac_vec, _mm_mul_ps(stiff, _mm_sub_ps(len, rest))));
            _mm_store_ps(reinterpret_cast<float*>(cur_pos + off_a),
                         _mm_add_ps(pos_a, _mm_mul_ps(inv_mass_a, imp)));
            const __m128 pos_b_reload =
                _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_b));
            _mm_store_ps(reinterpret_cast<float*>(cur_pos + off_b),
                         _mm_sub_ps(pos_b_reload, _mm_mul_ps(inv_mass_b, imp)));
        }
    }

    MonitorPushTimer(0x47b1ceeULL);
}

// ============================================================================
// 24. hclStandardLinkConstraintSetMx::apply @ 0x07420c0 (224 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclStandardLinkConstraintSetMx_apply(const u8* cset, u8* sim, u8* /*ctx*/,
                                                                int /*cset_idx*/, int /*use_factor*/,
                                                                float factor) noexcept {
    if (!(factor > 0.0f)) {
        return;
    }

    u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
    MonitorPushTimer(0x47b1cb0ULL);

    alignas(16) const __m128 fac_batch = _mm_set1_ps(factor);
    ResolveFn<MxApplyBatchFn>(0x255fa20ULL)(cset, cur_pos, &fac_batch);

    alignas(16) const __m128 fac_single = _mm_set1_ps(factor);
    ResolveFn<MxApplyBatchFn>(0x0767810ULL)(cset, cur_pos, &fac_single);

    MonitorPushTimer(0x47b1cddULL);
}

// ============================================================================
// 25. hclBendStiffnessConstraintSet::apply @ 0x07386e0 (471 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclBendStiffnessConstraintSet_apply(const u8* cset, u8* sim, u8* /*ctx*/,
                                                               int /*cset_idx*/, int /*use_factor*/,
                                                               float factor) noexcept {
    if (!(factor > 0.0f)) {
        return;
    }

    const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
    u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
    const u8* pdata = *reinterpret_cast<const u8* const*>(sim_data + 0x38);
    MonitorPushTimer(0x47b1ad2ULL);

    const std::int32_t count = *reinterpret_cast<const std::int32_t*>(cset + 0x28);
    const u8* links = *reinterpret_cast<const u8* const*>(cset + 0x20);
    alignas(16) const __m128 fac_vec = _mm_set1_ps(factor);

    if (cset[0x30] != 0u) {
        using RestPoseBendFn =
            void(PS4_SYSV_ABI*)(int, u8*, const u8*, const u8*, const __m128*);
        ResolveFn<RestPoseBendFn>(0x07384f0ULL)(count, cur_pos, pdata, links, &fac_vec);
    } else if (count > 0) {
        const u8* link = links;
        for (std::int32_t i = 0; i < count; ++i, link += 0x20u) {
            const std::size_t off_a =
                std::size_t(*reinterpret_cast<const std::uint16_t*>(link + 0x18)) << 4;
            const std::size_t off_b =
                std::size_t(*reinterpret_cast<const std::uint16_t*>(link + 0x1a)) << 4;
            const std::size_t off_c =
                std::size_t(*reinterpret_cast<const std::uint16_t*>(link + 0x1c)) << 4;
            const std::size_t off_d =
                std::size_t(*reinterpret_cast<const std::uint16_t*>(link + 0x1e)) << 4;

            const __m128 w_a = _mm_set1_ps(*reinterpret_cast<const float*>(link + 0x00));
            const __m128 w_b = _mm_set1_ps(*reinterpret_cast<const float*>(link + 0x04));
            const __m128 w_c = _mm_set1_ps(*reinterpret_cast<const float*>(link + 0x08));
            const __m128 w_d = _mm_set1_ps(*reinterpret_cast<const float*>(link + 0x0c));

            const __m128 pos_a = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_a));
            __m128 sum = _mm_mul_ps(w_a, pos_a);
            sum = _mm_add_ps(sum, _mm_mul_ps(w_b, _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_b))));
            sum = _mm_add_ps(sum, _mm_mul_ps(w_c, _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_c))));
            sum = _mm_add_ps(sum, _mm_mul_ps(w_d, _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_d))));

            const __m128 stiff = _mm_mul_ps(
                fac_vec, _mm_set1_ps(*reinterpret_cast<const float*>(link + 0x10)));

            const __m128 k_a = _mm_mul_ps(
                _mm_set1_ps(*reinterpret_cast<const float*>(pdata + off_a + 0x04)),
                _mm_mul_ps(w_a, stiff));
            const __m128 k_b = _mm_mul_ps(
                _mm_set1_ps(*reinterpret_cast<const float*>(pdata + off_b + 0x04)),
                _mm_mul_ps(w_b, stiff));
            const __m128 k_c = _mm_mul_ps(
                _mm_mul_ps(w_c, stiff),
                _mm_set1_ps(*reinterpret_cast<const float*>(pdata + off_c + 0x04)));
            const __m128 k_d = _mm_mul_ps(
                _mm_mul_ps(w_d, stiff),
                _mm_set1_ps(*reinterpret_cast<const float*>(pdata + off_d + 0x04)));

            _mm_store_ps(reinterpret_cast<float*>(cur_pos + off_a),
                         _mm_add_ps(pos_a, _mm_mul_ps(sum, k_a)));
            _mm_store_ps(
                reinterpret_cast<float*>(cur_pos + off_b),
                _mm_add_ps(_mm_mul_ps(sum, k_b),
                           _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_b))));
            _mm_store_ps(
                reinterpret_cast<float*>(cur_pos + off_c),
                _mm_add_ps(_mm_mul_ps(sum, k_c),
                           _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_c))));
            _mm_store_ps(
                reinterpret_cast<float*>(cur_pos + off_d),
                _mm_add_ps(_mm_mul_ps(sum, k_d),
                           _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_d))));
        }
    }

    MonitorPushTimer(0x47b1ae9ULL);
}

// ============================================================================
// 26. hclBendStiffnessConstraintSetMx::apply @ 0x0741f20 (284 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclBendStiffnessConstraintSetMx_apply(const u8* cset, u8* sim, u8* /*ctx*/,
                                                                 int /*cset_idx*/, int /*use_factor*/,
                                                                 float factor) noexcept {
    if (!(factor > 0.0f)) {
        return;
    }

    u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
    MonitorPushTimer(0x47b1c80ULL);

    if (cset[0x40] != 0u) {
        alignas(16) const __m128 fac_batch = _mm_set1_ps(factor);
        ResolveFn<MxApplyBatchFn>(0x255e5d0ULL)(cset, cur_pos, &fac_batch);

        alignas(16) const __m128 fac_single = _mm_set1_ps(factor);
        ResolveFn<MxApplyBatchFn>(0x07675a0ULL)(cset, cur_pos, &fac_single);
    } else {
        alignas(16) const __m128 fac_batch = _mm_set1_ps(factor);
        ResolveFn<MxApplyBatchFn>(0x255e010ULL)(cset, cur_pos, &fac_batch);

        alignas(16) const __m128 fac_single = _mm_set1_ps(factor);
        ResolveFn<MxApplyBatchFn>(0x0767410ULL)(cset, cur_pos, &fac_single);
    }

    MonitorPushTimer(0x47b1cadULL);
}

// ============================================================================
// 27. hclBendLinkConstraintSet::apply @ 0x07464d0 (415 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclBendLinkConstraintSet_apply(const u8* cset, u8* sim, u8* /*ctx*/,
                                                          int /*cset_idx*/, int /*use_factor*/,
                                                          float factor) noexcept {
    if (!(factor > 0.0f)) {
        return;
    }

    const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
    u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
    const u8* pdata = *reinterpret_cast<const u8* const*>(sim_data + 0x38);
    MonitorPushTimer(0x47b1e70ULL);

    const std::int32_t count = *reinterpret_cast<const std::int32_t*>(cset + 0x28);
    if (count > 0) {
        const u8* link = *reinterpret_cast<const u8* const*>(cset + 0x20);
        const __m128 fac_vec = _mm_set1_ps(factor);
        const __m128 zero_v = _mm_setzero_ps();

        for (std::int32_t i = 0; i < count; ++i, link += 0x14u) {
            _mm_prefetch(reinterpret_cast<const char*>(link + 0x50u), _MM_HINT_NTA);
            if (i + 2 < count) {
                const std::size_t pf_a =
                    std::size_t(*reinterpret_cast<const std::uint16_t*>(link + 0x28u)) << 4;
                const std::size_t pf_b =
                    std::size_t(*reinterpret_cast<const std::uint16_t*>(link + 0x2au)) << 4;
                _mm_prefetch(reinterpret_cast<const char*>(cur_pos + pf_a), _MM_HINT_NTA);
                _mm_prefetch(reinterpret_cast<const char*>(cur_pos + pf_b), _MM_HINT_NTA);
            }

            const std::size_t off_b =
                std::size_t(*reinterpret_cast<const std::uint16_t*>(link + 0x02)) << 4;
            const std::size_t off_a =
                std::size_t(*reinterpret_cast<const std::uint16_t*>(link + 0x00)) << 4;

            const __m128 pos_b = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_b));
            const __m128 pos_a = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_a));
            const __m128 diff = _mm_sub_ps(pos_b, pos_a);
            const __m128 len_sq = _mm_dp_ps(diff, diff, 0x7f);
            const __m128 inv_len =
                _mm_andnot_ps(_mm_cmple_ps(len_sq, zero_v), _mm_rsqrt_ps(len_sq));
            const __m128 dir = _mm_mul_ps(diff, inv_len);
            const __m128 len = _mm_mul_ps(len_sq, inv_len);

            const __m128 inv_mass_a =
                _mm_set1_ps(*reinterpret_cast<const float*>(pdata + off_a + 0x04));
            const __m128 inv_mass_b =
                _mm_set1_ps(*reinterpret_cast<const float*>(pdata + off_b + 0x04));

            const __m128 stretch_stiff = _mm_set1_ps(*reinterpret_cast<const float*>(link + 0x10));
            const __m128 bend_stiff = _mm_set1_ps(*reinterpret_cast<const float*>(link + 0x0c));
            const __m128 bend_min_len = _mm_set1_ps(*reinterpret_cast<const float*>(link + 0x04));
            const __m128 stretch_max_len =
                _mm_set1_ps(*reinterpret_cast<const float*>(link + 0x08));

            const __m128 err_bend = _mm_max_ps(zero_v, _mm_sub_ps(bend_min_len, len));
            const __m128 err_stretch = _mm_max_ps(zero_v, _mm_sub_ps(len, stretch_max_len));
            const __m128 force = _mm_sub_ps(_mm_mul_ps(stretch_stiff, err_stretch),
                                            _mm_mul_ps(bend_stiff, err_bend));
            const __m128 imp = _mm_mul_ps(dir, _mm_mul_ps(fac_vec, force));

            _mm_store_ps(reinterpret_cast<float*>(cur_pos + off_a),
                         _mm_add_ps(pos_a, _mm_mul_ps(inv_mass_a, imp)));
            const __m128 pos_b_reload =
                _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_b));
            _mm_store_ps(reinterpret_cast<float*>(cur_pos + off_b),
                         _mm_sub_ps(pos_b_reload, _mm_mul_ps(inv_mass_b, imp)));
        }
    }

    MonitorPushTimer(0x47b1e83ULL);
}

// ============================================================================
// 28. hclBendLinkConstraintSetMx::apply @ 0x072b2d0 (224 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclBendLinkConstraintSetMx_apply(const u8* cset, u8* sim, u8* /*ctx*/,
                                                            int /*cset_idx*/, int /*use_factor*/,
                                                            float factor) noexcept {
    if (!(factor > 0.0f)) {
        return;
    }

    u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
    MonitorPushTimer(0x47b1113ULL);

    alignas(16) const __m128 fac_batch = _mm_set1_ps(factor);
    ResolveFn<MxApplyBatchFn>(0x2556db0ULL)(cset, cur_pos, &fac_batch);

    alignas(16) const __m128 fac_single = _mm_set1_ps(factor);
    ResolveFn<MxApplyBatchFn>(0x07411e0ULL)(cset, cur_pos, &fac_single);

    MonitorPushTimer(0x47b1138ULL);
}

// ============================================================================
// 29. hclCompressibleLinkConstraintSet::apply @ 0x07308c0 (364 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclCompressibleLinkConstraintSet_apply(const u8* cset, u8* sim,
                                                                  u8* /*ctx*/, int /*cset_idx*/,
                                                                  int /*use_factor*/,
                                                                  float factor) noexcept {
    if (!(factor > 0.0f)) {
        return;
    }

    const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
    u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
    const u8* pdata = *reinterpret_cast<const u8* const*>(sim_data + 0x38);
    MonitorPushTimer(0x47b1303ULL);

    const std::int32_t count = *reinterpret_cast<const std::int32_t*>(cset + 0x28);
    if (count > 0) {
        const u8* link = *reinterpret_cast<const u8* const*>(cset + 0x20);
        const __m128 fac_vec = _mm_set1_ps(factor);
        const __m128 zero_v = _mm_setzero_ps();

        for (std::int32_t i = 0; i < count; ++i, link += 0x10u) {
            const std::size_t off_b =
                std::size_t(*reinterpret_cast<const std::uint16_t*>(link + 0x02)) << 4;
            const std::size_t off_a =
                std::size_t(*reinterpret_cast<const std::uint16_t*>(link + 0x00)) << 4;

            const __m128 pos_b = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_b));
            const __m128 pos_a = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_a));
            const __m128 diff = _mm_sub_ps(pos_b, pos_a);
            const __m128 len_sq = _mm_dp_ps(diff, diff, 0x7f);
            const __m128 inv_len =
                _mm_andnot_ps(_mm_cmple_ps(len_sq, zero_v), _mm_rsqrt_ps(len_sq));
            const __m128 dir = _mm_mul_ps(diff, inv_len);
            const __m128 len = _mm_mul_ps(len_sq, inv_len);

            const __m128 inv_mass_a =
                _mm_set1_ps(*reinterpret_cast<const float*>(pdata + off_a + 0x04));
            const __m128 inv_mass_b =
                _mm_set1_ps(*reinterpret_cast<const float*>(pdata + off_b + 0x04));
            const __m128 stiff = _mm_set1_ps(*reinterpret_cast<const float*>(link + 0x0c));
            const float rest_len = *reinterpret_cast<const float*>(link + 0x04);
            const float comp_limit = *reinterpret_cast<const float*>(link + 0x08);
            const float len_f = _mm_cvtss_f32(len);

            const bool below_comp = !(len_f >= comp_limit);
            const bool above_rest = (len_f > rest_len);
            if (!above_rest && !below_comp) {
                continue;
            }

            const __m128 target = below_comp ? _mm_set1_ps(comp_limit) : _mm_set1_ps(rest_len);
            const __m128 imp =
                _mm_mul_ps(dir, _mm_mul_ps(fac_vec, _mm_mul_ps(stiff, _mm_sub_ps(len, target))));
            _mm_store_ps(reinterpret_cast<float*>(cur_pos + off_a),
                         _mm_add_ps(pos_a, _mm_mul_ps(inv_mass_a, imp)));
            const __m128 pos_b_reload =
                _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off_b));
            _mm_store_ps(reinterpret_cast<float*>(cur_pos + off_b),
                         _mm_sub_ps(pos_b_reload, _mm_mul_ps(inv_mass_b, imp)));
        }
    }

    MonitorPushTimer(0x47b131eULL);
}

// ============================================================================
// 30. hclVolumeConstraint::apply @ 0x076c7e0 (389 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclVolumeConstraint_apply(const u8* cset, u8* sim, u8* /*ctx*/,
                                                     int cset_idx, int /*use_factor*/,
                                                     float factor) noexcept {
    if (!(factor > 0.0f)) {
        return;
    }

    u8* inst_data = nullptr;
    const std::int32_t n_map = *reinterpret_cast<const std::int32_t*>(sim + 0xb8);
    const u8* map_base = *reinterpret_cast<const u8* const*>(sim + 0xb0);
    for (std::int32_t m = 0; m < n_map; ++m) {
        const u8* entry = map_base + std::size_t(m) * 0x10u;
        if (*reinterpret_cast<const std::int32_t*>(entry) == cset_idx) {
            inst_data = *reinterpret_cast<u8* const*>(entry + 0x08);
            break;
        }
    }

    MonitorPushTimer(0x47b300eULL);

    u8* frame_xform = inst_data + 0x10u;
    using CalcVolumeCenterFn = void(PS4_SYSV_ABI*)(const u8*, u8*, u8*);
    using CalcVolumeDeltaFn = void(PS4_SYSV_ABI*)(const u8*, u8*, u8*, u8*);
    using TransformPosFn = void(PS4_SYSV_ABI*)(__m128*, const u8*, const u8*);

    ResolveFn<CalcVolumeCenterFn>(0x076d910ULL)(
        cset, frame_xform, *reinterpret_cast<u8* const*>(sim + 0x18));
    ResolveFn<CalcVolumeDeltaFn>(0x076d960ULL)(
        cset, frame_xform, inst_data + 0x50u, *reinterpret_cast<u8* const*>(sim + 0x18));

    const std::int32_t n_apply = *reinterpret_cast<const std::int32_t*>(cset + 0x38);
    if (n_apply > 0) {
        u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
        const u8* elem = *reinterpret_cast<const u8* const*>(cset + 0x30);
        for (std::int32_t i = 0; i < n_apply; ++i, elem += 0x20u) {
            alignas(16) __m128 world_pt{};
            ResolveFn<TransformPosFn>(0x043c040ULL)(&world_pt, frame_xform, elem);

            const std::size_t off =
                std::size_t(*reinterpret_cast<const std::uint16_t*>(elem + 0x10)) << 4;
            const __m128 p = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + off));
            const float stiff = factor * (*reinterpret_cast<const float*>(elem + 0x14));
            const __m128 new_p =
                _mm_add_ps(p, _mm_mul_ps(_mm_sub_ps(world_pt, p), _mm_set1_ps(stiff)));
            _mm_store_ps(reinterpret_cast<float*>(cur_pos + off), new_p);
        }
    }

    MonitorPushTimer(0x47b3023ULL);
}

} // extern "C"

} // namespace BbClothExp
