// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// File: gpu/cloth_exp/cloth_exp_local_fp64.cpp
//
// Mode 1.1: Canonical Havok Cloth Upgraded to Character-Local FP64 Coordinates
//           + Explicit FP64 Velocity + Inelastic Cage/Tether Projection
//           (True Native Any-FPS: 30..500+ FPS & Multi-Frame FG 2x/3x/4x).
//
// Zero static `thread_local` additions (`+0 B` `PT_TLS`), portable `x86-64-v3`.
// ============================================================================

#include "cloth_exp/cloth_exp_local_fp64.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "cloth_exp/cloth_exp_constraints_and_col.h"
#include "cloth_exp/cloth_exp_runtime.h"
#include "cloth_exp/cloth_exp_sim_and_integrate.h"
#include "cloth_exp/cloth_exp_world_and_ops.h"
#include "shim/bbport_cloth_interp.h"

namespace BbClothExp {

namespace {

using GetActiveCfgFn = u8*(PS4_SYSV_ABI*)(u8*);

struct alignas(16) HkTransform4x4 {
    __m128 col[4];
};

struct LocalFp64InstanceState {
    std::mutex instance_mtx;
    const u8* sim_data = nullptr;
    std::int32_t n_part = 0;
    bool initialized = false;
    bool has_move_pins = false;
    std::uint64_t last_world_seq = 0;

    double anchor_w[3]{0.0, 0.0, 0.0};
    double smooth_dt = 1.0 / 60.0;

    std::vector<double> x_loc;          // n_part * 3 (local FP64 position = world - anchor_w)
    std::vector<double> v_world;        // n_part * 3 (explicit FP64 physical velocity in m/s)
    std::vector<double> pin_world_cur;  // n_part * 3 (exact FP64 world target from MoveParticles)
    std::vector<double> pin_world_prev; // n_part * 3 (previous step FP64 world target)
    std::vector<u8> pin_mask;           // bit 0: invMass==0, bit 1: m_fixedParticles, bit 2: moved
    std::vector<float> pre_cset_pos;    // n_part * 4 scratch for inelastic cage/tether projection
};

std::shared_mutex g_fp64_map_mtx;
std::unordered_map<const u8*, std::unique_ptr<LocalFp64InstanceState>> g_fp64_states;
std::uint64_t g_fp64_world_seq = 0;

LocalFp64InstanceState* GetOrCreateInstanceState(const u8* sim) noexcept {
    {
        std::shared_lock<std::shared_mutex> rlk(g_fp64_map_mtx);
        auto it = g_fp64_states.find(sim);
        if (it != g_fp64_states.end()) {
            return it->second.get();
        }
    }
    std::unique_lock<std::shared_mutex> wlk(g_fp64_map_mtx);
    auto& slot = g_fp64_states[sim];
    if (!slot) {
        slot = std::make_unique<LocalFp64InstanceState>();
    }
    return slot.get();
}

[[gnu::always_inline]] inline __m128 NormalizeQuatNR(__m128 q) noexcept {
    const __m128 half_v = _mm_set1_ps(0.5f);
    const __m128 three_v = _mm_set1_ps(3.0f);
    const __m128 d = _mm_dp_ps(q, q, 0xff);
    const __m128 r = _mm_rsqrt_ps(d);
    const __m128 nr =
        _mm_mul_ps(_mm_mul_ps(r, half_v), _mm_sub_ps(three_v, _mm_mul_ps(d, _mm_mul_ps(r, r))));
    return _mm_mul_ps(q, nr);
}

[[gnu::always_inline]] inline __m128 HkMaxFloatVec() noexcept {
    return _mm_set1_ps(std::bit_cast<float>(0x7f7fffeeu));
}

[[gnu::always_inline]] inline __m128 HkMinFloatVec() noexcept {
    return _mm_set1_ps(std::bit_cast<float>(0xff7fffeeu));
}

[[gnu::always_inline]] inline __m128 HkFloatEpsVec() noexcept {
    return _mm_set1_ps(std::bit_cast<float>(0x34000000u));
}

void CollectActiveActionsFromArray(u8* sim, u8* const* const* arr_slot,
                                   const std::int32_t* count_slot) noexcept {
    if (*count_slot <= 0) {
        return;
    }
    using IsActiveFn = u8(PS4_SYSV_ABI*)(u8*);
    using ReserveMoreFn = void(PS4_SYSV_ABI*)(void*, void*, int);
    for (std::int32_t b = 0; b < *count_slot; ++b) {
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

void EnsureStateTopologyLocked(LocalFp64InstanceState& st, const u8* sim_data,
                               std::int32_t n_part, const float* cur_pos) noexcept {
    if (st.sim_data == sim_data && st.n_part == n_part &&
        static_cast<std::int32_t>(st.pin_mask.size()) == n_part) {
        return;
    }
    st.sim_data = sim_data;
    st.n_part = n_part;
    st.initialized = false;
    st.has_move_pins = false;
    st.smooth_dt = 1.0 / 60.0;

    const std::size_t n3 = static_cast<std::size_t>(n_part) * 3u;
    const std::size_t n4 = static_cast<std::size_t>(n_part) * 4u;
    st.x_loc.assign(n3, 0.0);
    st.v_world.assign(n3, 0.0);
    st.pin_world_cur.assign(n3, 0.0);
    st.pin_world_prev.assign(n3, 0.0);
    st.pin_mask.assign(static_cast<std::size_t>(n_part), 0u);
    st.pre_cset_pos.assign(n4, 0.0f);

    const u8* pdatas = *reinterpret_cast<const u8* const*>(sim_data + 0x38);
    const std::int32_t n_pdatas = *reinterpret_cast<const std::int32_t*>(sim_data + 0x40);
    if (pdatas && n_pdatas >= n_part) {
        for (std::int32_t i = 0; i < n_part; ++i) {
            const float inv_m =
                *reinterpret_cast<const float*>(pdatas + std::size_t(i) * 16u + 4u);
            if (inv_m == 0.0f) {
                st.pin_mask[std::size_t(i)] |= 1u;
            }
        }
    }
    const std::uint16_t* fix_idx = *reinterpret_cast<const std::uint16_t* const*>(sim_data + 0x48);
    const std::int32_t n_fix = *reinterpret_cast<const std::int32_t*>(sim_data + 0x50);
    if (fix_idx && n_fix > 0 && n_fix <= n_part) {
        for (std::int32_t k = 0; k < n_fix; ++k) {
            const std::uint16_t pi = fix_idx[k];
            if (static_cast<std::int32_t>(pi) < n_part) {
                st.pin_mask[pi] |= 2u;
            }
        }
    }
    if (cur_pos) {
        for (std::int32_t i = 0; i < n_part; ++i) {
            st.pin_world_cur[std::size_t(i) * 3u + 0u] = static_cast<double>(cur_pos[i * 4 + 0]);
            st.pin_world_cur[std::size_t(i) * 3u + 1u] = static_cast<double>(cur_pos[i * 4 + 1]);
            st.pin_world_cur[std::size_t(i) * 3u + 2u] = static_cast<double>(cur_pos[i * 4 + 2]);
        }
        st.pin_world_prev = st.pin_world_cur;
    }
}

/// Evaluates `transferMotion` (`0x0771290`) in exact FP64 on free particles (`x_loc` & `v_world`).
void ApplyTransferMotionFp64(LocalFp64InstanceState& st, const u8* sim_data,
                             const void* old_xform_ptr, const void* new_xform_ptr,
                             float dt) noexcept {
    if (!(dt > 1e-6f)) {
        return;
    }
    const float inv_dt = 1.0f / dt;

    using SetInverseFn = void(PS4_SYSV_ABI*)(void*, const void*);
    using SetMulFn = void(PS4_SYSV_ABI*)(void*, const void*, const void*);
    using MatToQuatFn = void(PS4_SYSV_ABI*)(__m128*, const void*);
    using QuatAngleFn = float(PS4_SYSV_ABI*)(const __m128*);
    using QuatAxisAngleFn = void(PS4_SYSV_ABI*)(__m128*, const __m128*, float);
    using QuatToMatFn = void(PS4_SYSV_ABI*)(void*, const __m128*);

    HkTransform4x4 inv_old{};
    HkTransform4x4 delta_xform{};
    ResolveFn<SetInverseFn>(0x043b980ULL)(&inv_old, old_xform_ptr);
    ResolveFn<SetMulFn>(0x043b9f0ULL)(&delta_xform, &inv_old, new_xform_ptr);

    alignas(16) float dx_col3[4]{};
    _mm_store_ps(dx_col3, delta_xform.col[3]);

    float trans_frac = 0.0f;
    if (sim_data[0x158] != 0u) {
        const double vx = static_cast<double>(dx_col3[0]) * static_cast<double>(inv_dt);
        const double vy = static_cast<double>(dx_col3[1]) * static_cast<double>(inv_dt);
        const double vz = static_cast<double>(dx_col3[2]) * static_cast<double>(inv_dt);
        const float lin_speed = static_cast<float>(std::sqrt(vx * vx + vy * vy + vz * vz));

        const float min_s = *reinterpret_cast<const float*>(sim_data + 0x15c);
        const float max_s = *reinterpret_cast<const float*>(sim_data + 0x160);
        const float min_b = *reinterpret_cast<const float*>(sim_data + 0x164);
        const float max_b = *reinterpret_cast<const float*>(sim_data + 0x168);
        if (lin_speed >= max_s) {
            trans_frac = max_b;
        } else if (!(lin_speed > min_s)) {
            trans_frac = min_b;
        } else {
            const float span = max_s - min_s;
            const float inv_s = (std::fabs(span) > 1e-5f) ? (1.0f / span) : 0.0f;
            trans_frac = min_b + ((max_b - min_b) * (lin_speed - min_s)) * inv_s;
        }
    }

    alignas(16) __m128 delta_quat{};
    ResolveFn<MatToQuatFn>(0x0449900ULL)(&delta_quat, &delta_xform);

    float rot_frac = 0.0f;
    if (sim_data[0x16c] != 0u) {
        const float angle_rad = ResolveFn<QuatAngleFn>(0x04497d0ULL)(&delta_quat);
        const float deg_per_sec = (inv_dt * angle_rad) * 57.295776f;
        const float min_s = *reinterpret_cast<const float*>(sim_data + 0x170);
        const float max_s = *reinterpret_cast<const float*>(sim_data + 0x174);
        const float min_b = *reinterpret_cast<const float*>(sim_data + 0x178);
        const float max_b = *reinterpret_cast<const float*>(sim_data + 0x17c);
        if (deg_per_sec >= max_s) {
            rot_frac = max_b;
        } else if (!(deg_per_sec > min_s)) {
            rot_frac = min_b;
        } else {
            const float span = max_s - min_s;
            const float inv_s = (std::fabs(span) > 1e-5f) ? (1.0f / span) : 0.0f;
            rot_frac = min_b + ((max_b - min_b) * (deg_per_sec - min_s)) * inv_s;
        }
    }

    if (!(trans_frac > 0.0f || rot_frac > 0.0f)) {
        return;
    }

    HkTransform4x4 local_blend{};
    local_blend.col[0] = _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f);
    local_blend.col[1] = _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f);
    local_blend.col[2] = _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f);
    local_blend.col[3] = _mm_mul_ps(_mm_set1_ps(trans_frac), delta_xform.col[3]);

    const __m128 dq_v = delta_quat;
    const __m128 axis_sq = _mm_dp_ps(dq_v, dq_v, 0x7f);
    if (_mm_cvtss_f32(axis_sq) > 1.42e-14f && rot_frac > 0.0f) {
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
        alignas(16) const __m128 signed_axis = _mm_xor_ps(axis, _mm_castsi128_ps(sign_mask));
        const float full_angle = ResolveFn<QuatAngleFn>(0x04497d0ULL)(&delta_quat);
        alignas(16) __m128 scaled_quat{};
        ResolveFn<QuatAxisAngleFn>(0x0448e80ULL)(&scaled_quat, &signed_axis, rot_frac * full_angle);
        ResolveFn<QuatToMatFn>(0x0429b00ULL)(&local_blend, &scaled_quat);
    }

    // Compute world-space rotation `R_wm = R_old * R_blend * R_old^T` and pivot in FP64
    // relative to `anchor_w` so `500 m` world translation never pollutes `x_loc`!
    const float* old_m = static_cast<const float*>(old_xform_ptr);
    const float* new_m = static_cast<const float*>(new_xform_ptr);
    alignas(16) float lb_m[16]{};
    std::memcpy(lb_m, &local_blend, 64);

    // R_tmp = R_old * R_lb; R_wm = R_tmp * R_old^T
    double r_old[3][3]{};
    double r_lb[3][3]{};
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 3; ++r) {
            r_old[r][c] = static_cast<double>(old_m[c * 4 + r]);
            r_lb[r][c] = static_cast<double>(lb_m[c * 4 + r]);
        }
    }
    double r_tmp[3][3]{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            r_tmp[r][c] =
                r_old[r][0] * r_lb[0][c] + r_old[r][1] * r_lb[1][c] + r_old[r][2] * r_lb[2][c];
        }
    }
    double r_wm[3][3]{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            r_wm[r][c] =
                r_tmp[r][0] * r_old[c][0] + r_tmp[r][1] * r_old[c][1] + r_tmp[r][2] * r_old[c][2];
        }
    }

    // Exact FP64 world translation delta `t_blend_w = trans_frac * (new_pos - old_pos)`
    // (avoids `inv_old` FP32 `500 m` cancellation!).
    const double tf_d = static_cast<double>(trans_frac);
    const double t_blend_w[3] = {
        tf_d * (static_cast<double>(new_m[12]) - static_cast<double>(old_m[12])),
        tf_d * (static_cast<double>(new_m[13]) - static_cast<double>(old_m[13])),
        tf_d * (static_cast<double>(new_m[14]) - static_cast<double>(old_m[14])),
    };
    const double piv_loc[3] = {
        static_cast<double>(old_m[12]) - st.anchor_w[0],
        static_cast<double>(old_m[13]) - st.anchor_w[1],
        static_cast<double>(old_m[14]) - st.anchor_w[2],
    };

    const u8* pdata = *reinterpret_cast<const u8* const*>(sim_data + 0x38);
    for (std::int32_t i = 0; i < st.n_part; ++i) {
        const float mass = *reinterpret_cast<const float*>(pdata + std::size_t(i) * 16u);
        if (mass == 0.0f || st.pin_mask[std::size_t(i)] != 0u) {
            continue;
        }
        double* x = &st.x_loc[std::size_t(i) * 3u];
        double* v = &st.v_world[std::size_t(i) * 3u];

        const double rx = x[0] - piv_loc[0];
        const double ry = x[1] - piv_loc[1];
        const double rz = x[2] - piv_loc[2];

        x[0] = piv_loc[0] + (r_wm[0][0] * rx + r_wm[0][1] * ry + r_wm[0][2] * rz) + t_blend_w[0];
        x[1] = piv_loc[1] + (r_wm[1][0] * rx + r_wm[1][1] * ry + r_wm[1][2] * rz) + t_blend_w[1];
        x[2] = piv_loc[2] + (r_wm[2][0] * rx + r_wm[2][1] * ry + r_wm[2][2] * rz) + t_blend_w[2];

        const double vx = v[0], vy = v[1], vz = v[2];
        v[0] = r_wm[0][0] * vx + r_wm[0][1] * vy + r_wm[0][2] * vz;
        v[1] = r_wm[1][0] * vx + r_wm[1][1] * vy + r_wm[1][2] * vz;
        v[2] = r_wm[2][0] * vx + r_wm[2][1] * vy + r_wm[2][2] * vz;
    }
}

/// Runs `hclLocalRangeConstraintSet` (`type == 5`) in `anchor_w`-centered local coordinates
/// using a stack-local `ref_buf` header copy (`col3 -= anchor_w`), followed by exact inelastic
/// velocity projection so cage pushouts never inject `dx / sdt` bounce velocity.
void ApplyLocalRangeInLocalSpace(LocalFp64InstanceState& st, const u8* cset, u8* sim, u8* ctx,
                                 float factor) noexcept {
    if (!(factor > 0.0f)) {
        return;
    }

    const std::int32_t buf_idx = *reinterpret_cast<const std::int32_t*>(cset + 0x30);
    u8* const* bufs = *reinterpret_cast<u8* const* const*>(ctx + 0x30);
    const u8* raw_buf = bufs[buf_idx];
    const std::int32_t shadow_idx = *reinterpret_cast<const std::int32_t*>(raw_buf + 0x100);
    const u8* ref_buf = bufs[shadow_idx];

    alignas(16) u8 local_ref_buf[0x110]{};
    std::memcpy(local_ref_buf, ref_buf, 0x108u);
    const float* orig_c3 = reinterpret_cast<const float*>(ref_buf + 0xb0);
    float* loc_c3 = reinterpret_cast<float*>(local_ref_buf + 0xb0);
    loc_c3[0] = static_cast<float>(static_cast<double>(orig_c3[0]) - st.anchor_w[0]);
    loc_c3[1] = static_cast<float>(static_cast<double>(orig_c3[1]) - st.anchor_w[1]);
    loc_c3[2] = static_cast<float>(static_cast<double>(orig_c3[2]) - st.anchor_w[2]);
    loc_c3[3] = orig_c3[3];

    u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
    u8* prev_pos = *reinterpret_cast<u8* const*>(sim + 0x28);

    std::memcpy(st.pre_cset_pos.data(), cur_pos, static_cast<std::size_t>(st.n_part) * 16u);

    const std::uint32_t pos_aligned =
        *reinterpret_cast<const std::uint32_t*>(local_ref_buf + 0x20) & 1u;
    if (cset[0x3c] != 0u && *reinterpret_cast<const u8* const*>(local_ref_buf + 0x38) != nullptr) {
        using NormalRangeFn = void(PS4_SYSV_ABI*)(const u8*, u8*, u8*, const u8*, std::uint32_t,
                                                  const __m128*);
        const std::uint32_t nrm_flags =
            *reinterpret_cast<const std::uint32_t*>(local_ref_buf + 0x48);
        const std::uint32_t shape_mode = *reinterpret_cast<const std::uint32_t*>(cset + 0x38);
        alignas(16) const __m128 fac_vec = _mm_set1_ps(factor);

        if ((nrm_flags & pos_aligned) != 0u) {
            ResolveFn<NormalRangeFn>(0x2553210ULL)(cset, cur_pos, prev_pos, local_ref_buf,
                                                   shape_mode, &fac_vec);
        } else if (pos_aligned != 0u) {
            ResolveFn<NormalRangeFn>(0x2553510ULL)(cset, cur_pos, prev_pos, local_ref_buf,
                                                   shape_mode, &fac_vec);
        } else if ((nrm_flags & 1u) != 0u) {
            ResolveFn<NormalRangeFn>(0x2553810ULL)(cset, cur_pos, prev_pos, local_ref_buf,
                                                   shape_mode, &fac_vec);
        } else {
            ResolveFn<NormalRangeFn>(0x2553b10ULL)(cset, cur_pos, prev_pos, local_ref_buf,
                                                   shape_mode, &fac_vec);
        }
    } else {
        const std::int32_t count = *reinterpret_cast<const std::int32_t*>(cset + 0x28);
        if (count > 0) {
            const __m128 stiff_vec =
                _mm_set1_ps(factor * (*reinterpret_cast<const float*>(cset + 0x34)));
            const __m128 eps_v = HkFloatEpsVec();
            const __m128 zero_v = _mm_setzero_ps();
            const __m128 col0 = _mm_load_ps(reinterpret_cast<const float*>(local_ref_buf + 0x80));
            const __m128 col1 = _mm_load_ps(reinterpret_cast<const float*>(local_ref_buf + 0x90));
            const __m128 col2 = _mm_load_ps(reinterpret_cast<const float*>(local_ref_buf + 0xa0));
            const __m128 col3 = _mm_load_ps(reinterpret_cast<const float*>(local_ref_buf + 0xb0));

            for (std::int32_t i = 0; i < count; ++i) {
                const u8* entries = *reinterpret_cast<const u8* const*>(cset + 0x20);
                const u8* vtx_base = *reinterpret_cast<const u8* const*>(local_ref_buf + 0x10);
                const u8* entry = entries + std::size_t(i) * 0x10u;
                const std::size_t ref_vtx_idx =
                    *reinterpret_cast<const std::uint16_t*>(entry + 0x02);
                const std::size_t stride = *reinterpret_cast<const u8*>(local_ref_buf + 0x1c);
                const auto* vtx_ptr =
                    reinterpret_cast<const __m128i*>(vtx_base + ref_vtx_idx * stride);
                const __m128 v = (pos_aligned != 0u) ? _mm_castsi128_ps(_mm_load_si128(vtx_ptr))
                                                     : _mm_castsi128_ps(_mm_lddqu_si128(vtx_ptr));

                const __m128 max_dist =
                    _mm_set1_ps(*reinterpret_cast<const float*>(entry + 0x04));
                const __m128 wp = _mm_add_ps(
                    _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(v, v, 0x00), col0), col3),
                               _mm_mul_ps(_mm_shuffle_ps(v, v, 0x55), col1)),
                    _mm_mul_ps(_mm_shuffle_ps(v, v, 0xaa), col2));

                const std::size_t p_off =
                    std::size_t(*reinterpret_cast<const std::uint16_t*>(entry + 0x00)) << 4;
                const __m128 p = _mm_load_ps(reinterpret_cast<const float*>(cur_pos + p_off));
                const __m128 diff = _mm_add_ps(eps_v, _mm_sub_ps(p, wp));
                const __m128 len_sq = _mm_dp_ps(diff, diff, 0x7f);
                const __m128 le_zero = _mm_cmple_ps(len_sq, zero_v);
                const __m128 inv_len = _mm_andnot_ps(le_zero, _mm_rsqrt_ps(len_sq));
                const __m128 dir = _mm_mul_ps(diff, inv_len);
                const __m128 len = _mm_mul_ps(len_sq, inv_len);
                const __m128 corr =
                    _mm_min_ps(_mm_mul_ps(stiff_vec, _mm_sub_ps(max_dist, len)), zero_v);
                _mm_store_ps(reinterpret_cast<float*>(cur_pos + p_off),
                             _mm_add_ps(p, _mm_mul_ps(corr, dir)));
            }
        }
    }

    float* cur_f = reinterpret_cast<float*>(cur_pos);
    float* prev_f = reinterpret_cast<float*>(prev_pos);
    for (std::int32_t i = 0; i < st.n_part; ++i) {
        if (st.pin_mask[std::size_t(i)] == 0u) {
            ApplyInelasticConstraintProjection3D(&st.pre_cset_pos[std::size_t(i) * 4u],
                                                 &cur_f[i * 4], &prev_f[i * 4]);
        }
    }
}

float ComputeMode11ConstraintFactor(const u8* cset, int mode, int substep_idx, int eff_substeps,
                                    int ps4_substeps, float f0, float f1,
                                    double substep_ratio) noexcept {
    const std::uint32_t type = *reinterpret_cast<const std::uint32_t*>(cset + 0x18);
    const bool adapts = (mode == 2) && !(ps4_substeps == 1 && f0 == 1.0f && f1 == 1.0f);

    if (type == 8u) { // kTransition
        return adapts ? (static_cast<float>(substep_idx + 1) / static_cast<float>(eff_substeps))
                      : 1.0f;
    }

    float game_base = 1.0f;
    if (adapts && type != 5u && type != 10u) {
        const float x = static_cast<float>(ps4_substeps) * f0 * f1;
        if (x != 1.0f && x > 0.0f) {
            constexpr float kClothExp = -1.725000023841858f;
            game_base = std::pow(x, kClothExp);
        }
    }

    const float clamped_ratio = static_cast<float>(std::clamp(substep_ratio, 1.0 / 300.0, 1.5));
    switch (type) {
    case 2u: // kStretchLink (SIGGRAPH 2012 LRA tether)
        return game_base;
    case 5u: { // kLocalRange — active EVERY substep so StretchLink never diverges!
        const float cset_stiff = *reinterpret_cast<const float*>(cset + 0x34);
        return BbClothInterp::ComputeClothExponentialRelaxationFactor(cset_stiff, clamped_ratio);
    }
    case 6u:  // kVolume
    case 10u: // kVolumeAlt
        return 1.0f;
    default:  // Elastic weave & bend springs (StandardLink, BendStiffness, BendLink, CompressibleLink)
        return game_base * BbClothInterp::ComputeClothElasticStiffnessScale(clamped_ratio, 1.725f);
    }
}

void DispatchSingleConstraintSetLocalFp64(LocalFp64InstanceState& st, u8* cset, u8* sim, u8* ctx,
                                          int cset_idx, int use_factor, float factor) noexcept {
    if (!(factor > 0.0f)) {
        return;
    }
    const std::uint32_t type = *reinterpret_cast<const std::uint32_t*>(cset + 0x18);
    if (type == 5u) {
        ApplyLocalRangeInLocalSpace(st, cset, sim, ctx, factor);
        return;
    }

    using ConstraintApplyFn = void(PS4_SYSV_ABI*)(const u8*, u8*, u8*, int, int, float);
    void* const* vtbl = *reinterpret_cast<void* const* const*>(cset);

    if (type == 2u) {
        // Hard LRA StretchLink tether: apply inelastic velocity projection so tether pull-back
        // preserves tangential swing speed without adding inward `dx / sdt` bounce velocity.
        u8* cur_pos = *reinterpret_cast<u8* const*>(sim + 0x18);
        u8* prev_pos = *reinterpret_cast<u8* const*>(sim + 0x28);
        std::memcpy(st.pre_cset_pos.data(), cur_pos, static_cast<std::size_t>(st.n_part) * 16u);
        reinterpret_cast<ConstraintApplyFn>(vtbl[0x50 / 8])(cset, sim, ctx, cset_idx, use_factor,
                                                            factor);
        float* cur_f = reinterpret_cast<float*>(cur_pos);
        float* prev_f = reinterpret_cast<float*>(prev_pos);
        for (std::int32_t i = 0; i < st.n_part; ++i) {
            if (st.pin_mask[std::size_t(i)] == 0u) {
                ApplyInelasticConstraintProjection3D(&st.pre_cset_pos[std::size_t(i) * 4u],
                                                     &cur_f[i * 4], &prev_f[i * 4]);
            }
        }
        return;
    }

    reinterpret_cast<ConstraintApplyFn>(vtbl[0x50 / 8])(cset, sim, ctx, cset_idx, use_factor,
                                                        factor);
}

} // namespace

void LocalFp64_RecordMoveParticles(const u8* move_op, u8* sim, const u8* ref_buf) noexcept {
    if (!move_op || !sim || !ref_buf) {
        return;
    }
    const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
    const std::int32_t n_part = *reinterpret_cast<const std::int32_t*>(sim + 0x20);
    const float* cur_pos = *reinterpret_cast<const float* const*>(sim + 0x18);
    if (!sim_data || !cur_pos || n_part <= 0 || n_part > 16384) {
        return;
    }

    LocalFp64InstanceState* st = GetOrCreateInstanceState(sim);
    std::lock_guard<std::mutex> lk(st->instance_mtx);
    EnsureStateTopologyLocked(*st, sim_data, n_part, cur_pos);

    const std::int32_t count = *reinterpret_cast<const std::int32_t*>(move_op + 0x28);
    if (count <= 0) {
        return;
    }

    const float* c0 = reinterpret_cast<const float*>(ref_buf + 0x80);
    const float* c1 = reinterpret_cast<const float*>(ref_buf + 0x90);
    const float* c2 = reinterpret_cast<const float*>(ref_buf + 0xa0);
    const float* c3 = reinterpret_cast<const float*>(ref_buf + 0xb0);

    const double m00 = c0[0], m10 = c0[1], m20 = c0[2];
    const double m01 = c1[0], m11 = c1[1], m21 = c1[2];
    const double m02 = c2[0], m12 = c2[1], m22 = c2[2];
    const double tx = c3[0], ty = c3[1], tz = c3[2];

    const std::uint16_t* pairs = *reinterpret_cast<const std::uint16_t* const*>(move_op + 0x20);
    const u8* vtx_base = *reinterpret_cast<const u8* const*>(ref_buf + 0x10);
    const std::size_t stride = *reinterpret_cast<const u8*>(ref_buf + 0x1c);

    for (std::int32_t i = 0; i < count; ++i) {
        const std::size_t vtx_idx = pairs[std::size_t(i) * 2u + 0u];
        const std::size_t p_idx = pairs[std::size_t(i) * 2u + 1u];
        if (static_cast<std::int32_t>(p_idx) >= n_part) {
            continue;
        }
        float v[4]{};
        std::memcpy(v, vtx_base + vtx_idx * stride, 16);
        const double vx = v[0], vy = v[1], vz = v[2];

        st->pin_world_cur[p_idx * 3u + 0u] = tx + m00 * vx + m01 * vy + m02 * vz;
        st->pin_world_cur[p_idx * 3u + 1u] = ty + m10 * vx + m11 * vy + m12 * vz;
        st->pin_world_cur[p_idx * 3u + 2u] = tz + m20 * vx + m21 * vy + m22 * vz;
        st->pin_mask[p_idx] |= 4u;
    }
    st->has_move_pins = true;
}

void LocalFp64_OnWorldStepBegin(std::uint64_t world_seq) noexcept {
    g_fp64_world_seq = world_seq;
    if ((world_seq & 63u) != 0u) {
        return;
    }
    std::unique_lock<std::shared_mutex> wlk(g_fp64_map_mtx);
    for (auto it = g_fp64_states.begin(); it != g_fp64_states.end();) {
        if (world_seq - it->second->last_world_seq > 180u) {
            it = g_fp64_states.erase(it);
        } else {
            ++it;
        }
    }
}

bool LocalFp64_CopyParticleVelocities(const u8* sim, __m128* out_vel, std::int32_t max_n) noexcept {
    if (!sim || !out_vel || max_n <= 0) {
        return false;
    }
    LocalFp64InstanceState* st = nullptr;
    {
        std::shared_lock<std::shared_mutex> rlk(g_fp64_map_mtx);
        auto it = g_fp64_states.find(sim);
        if (it == g_fp64_states.end() || !it->second || !it->second->initialized) {
            return false;
        }
        st = it->second.get();
    }
    const std::int32_t n = std::min(max_n, st->n_part);
    for (std::int32_t i = 0; i < n; ++i) {
        const double* v = &st->v_world[std::size_t(i) * 3u];
        out_vel[i] = _mm_setr_ps(static_cast<float>(v[0]), static_cast<float>(v[1]),
                                 static_cast<float>(v[2]), 0.0f);
    }
    return true;
}

void LocalFp64_PreStep(u8* op, u8* step_info) noexcept {
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
    const float prev_sdt = *reinterpret_cast<const float*>(sim + 0x108);

    if (prev_sdt == 0.0f && cfg[0x1e] != 0u) {
        const u8* sd = *reinterpret_cast<const u8* const*>(sim + 0x10);
        const std::int32_t tset_idx = *reinterpret_cast<const std::int32_t*>(sd + 0x150);
        const std::int32_t bone_idx = *reinterpret_cast<const std::int32_t*>(sd + 0x154);
        const u8* tset = (*reinterpret_cast<const u8* const* const*>(ctx0 + 0x40))[tset_idx];
        const __m128* src_m = reinterpret_cast<const __m128*>(
            *reinterpret_cast<const u8* const*>(tset + 0x10) + (std::size_t(bone_idx) << 6));
        HkTransform4x4 bone_mat{};
        bone_mat.col[0] = _mm_load_ps(reinterpret_cast<const float*>(src_m + 0));
        bone_mat.col[1] = _mm_load_ps(reinterpret_cast<const float*>(src_m + 1));
        bone_mat.col[2] = _mm_load_ps(reinterpret_cast<const float*>(src_m + 2));
        bone_mat.col[3] = _mm_load_ps(reinterpret_cast<const float*>(src_m + 3));
        using SetQTransformFn = void(PS4_SYSV_ABI*)(void*, const void*);
        ResolveFn<SetQTransformFn>(0x043b8c0ULL)(sim + 0x120, &bone_mat);
    }

    // Store current substep_dt & damping without running Havok's noisy FP32 `prev_pos` rescale loop!
    const float new_sdt = step_dt / std::max(slowmo * static_cast<float>(std::max(1, substeps)), 1e-4f);
    *reinterpret_cast<float*>(sim + 0x108) = new_sdt;
    const float damp = *reinterpret_cast<const float*>(cfg + 0x10);
    *reinterpret_cast<float*>(sim + 0x10c) =
        (damp >= 1.0f) ? 0.0f : ((damp <= 0.0f) ? 1.0f : std::pow(1.0f - damp, new_sdt));

    // Rebuild sim->m_activeActions (+0x168/+0x170/+0x174) from all 5 action scopes.
    *reinterpret_cast<std::int32_t*>(sim + 0x170) = 0;
    u8* world = *reinterpret_cast<u8* const*>(step_info + 0x08);
    CollectActiveActionsFromArray(sim, reinterpret_cast<u8* const* const*>(world + 0x30),
                                  reinterpret_cast<const std::int32_t*>(world + 0x38));
    u8* cloth_data = *reinterpret_cast<u8* const*>(ctx0 + 0x10);
    CollectActiveActionsFromArray(sim, reinterpret_cast<u8* const* const*>(cloth_data + 0x68),
                                  reinterpret_cast<const std::int32_t*>(cloth_data + 0x70));
    u8* sim_data = *reinterpret_cast<u8* const*>(sim + 0x10);
    CollectActiveActionsFromArray(sim, reinterpret_cast<u8* const* const*>(sim_data + 0xe8),
                                  reinterpret_cast<const std::int32_t*>(sim_data + 0xf0));
    CollectActiveActionsFromArray(sim, reinterpret_cast<u8* const* const*>(ctx0 + 0x50),
                                  reinterpret_cast<const std::int32_t*>(ctx0 + 0x58));
    CollectActiveActionsFromArray(sim, reinterpret_cast<u8* const* const*>(sim + 0x110),
                                  reinterpret_cast<const std::int32_t*>(sim + 0x118));
}

int LocalFp64_Simulate(u8* op, u8* ctx, float raw_dt) noexcept {
    MonitorPushTimer(0x47b31afULL);

    const std::int32_t sim_idx = *reinterpret_cast<const std::int32_t*>(op + 0x1c);
    u8* sim = (*reinterpret_cast<u8* const* const*>(ctx + 0x20))[sim_idx];
    u8* sim_data = *reinterpret_cast<u8* const*>(sim + 0x10);
    const std::int32_t n_part = *reinterpret_cast<const std::int32_t*>(sim + 0x20);
    float* cur_pos_f = *reinterpret_cast<float* const*>(sim + 0x18);
    float* prev_pos_f = *reinterpret_cast<float* const*>(sim + 0x28);

    if (!sim_data || !cur_pos_f || !prev_pos_f || n_part <= 0 || n_part > 16384) {
        return 0;
    }

    const float slowmo = std::max(*reinterpret_cast<const float*>(sim + 0x1d0), 1e-3f);
    const float dt = std::clamp(raw_dt / slowmo, 1.0f / 1000.0f, 1.0f / 20.0f);

    LocalFp64InstanceState* st = GetOrCreateInstanceState(sim);
    std::lock_guard<std::mutex> lk(st->instance_mtx);
    EnsureStateTopologyLocked(*st, sim_data, n_part, cur_pos_f);
    const bool resumed_after_pause_or_switch =
        (st->initialized && g_fp64_world_seq > st->last_world_seq + 2u);
    st->last_world_seq = g_fp64_world_seq;

    // Refresh pinned world positions for any pinned particle not driven by MoveParticles.
    for (std::int32_t i = 0; i < n_part; ++i) {
        const u8 pm = st->pin_mask[std::size_t(i)];
        if (pm != 0u && (pm & 4u) == 0u) {
            st->pin_world_cur[std::size_t(i) * 3u + 0u] = static_cast<double>(cur_pos_f[i * 4 + 0]);
            st->pin_world_cur[std::size_t(i) * 3u + 1u] = static_cast<double>(cur_pos_f[i * 4 + 1]);
            st->pin_world_cur[std::size_t(i) * 3u + 2u] = static_cast<double>(cur_pos_f[i * 4 + 2]);
        }
    }

    // Compute exact FP64 character-centered anchor `new_anchor` from pinned centroid.
    double new_anchor[3]{0.0, 0.0, 0.0};
    std::int32_t pin_cnt = 0;
    for (std::int32_t i = 0; i < n_part; ++i) {
        if (st->pin_mask[std::size_t(i)] != 0u) {
            new_anchor[0] += st->pin_world_cur[std::size_t(i) * 3u + 0u];
            new_anchor[1] += st->pin_world_cur[std::size_t(i) * 3u + 1u];
            new_anchor[2] += st->pin_world_cur[std::size_t(i) * 3u + 2u];
            ++pin_cnt;
        }
    }
    if (pin_cnt > 0) {
        const double inv_cnt = 1.0 / static_cast<double>(pin_cnt);
        new_anchor[0] *= inv_cnt;
        new_anchor[1] *= inv_cnt;
        new_anchor[2] *= inv_cnt;
    } else {
        new_anchor[0] = static_cast<double>(cur_pos_f[0]);
        new_anchor[1] = static_cast<double>(cur_pos_f[1]);
        new_anchor[2] = static_cast<double>(cur_pos_f[2]);
    }

    const double d_anc[3] = {
        new_anchor[0] - st->anchor_w[0],
        new_anchor[1] - st->anchor_w[1],
        new_anchor[2] - st->anchor_w[2],
    };
    const double d_anc_sq = d_anc[0] * d_anc[0] + d_anc[1] * d_anc[1] + d_anc[2] * d_anc[2];
    const bool teleported = (!st->initialized || resumed_after_pause_or_switch || d_anc_sq > 9.0);

    st->anchor_w[0] = new_anchor[0];
    st->anchor_w[1] = new_anchor[1];
    st->anchor_w[2] = new_anchor[2];

    if (teleported) {
        for (std::int32_t i = 0; i < n_part; ++i) {
            if (st->pin_mask[std::size_t(i)] != 0u) {
                st->x_loc[std::size_t(i) * 3u + 0u] =
                    st->pin_world_cur[std::size_t(i) * 3u + 0u] - st->anchor_w[0];
                st->x_loc[std::size_t(i) * 3u + 1u] =
                    st->pin_world_cur[std::size_t(i) * 3u + 1u] - st->anchor_w[1];
                st->x_loc[std::size_t(i) * 3u + 2u] =
                    st->pin_world_cur[std::size_t(i) * 3u + 2u] - st->anchor_w[2];
            } else {
                st->x_loc[std::size_t(i) * 3u + 0u] =
                    static_cast<double>(cur_pos_f[i * 4 + 0]) - st->anchor_w[0];
                st->x_loc[std::size_t(i) * 3u + 1u] =
                    static_cast<double>(cur_pos_f[i * 4 + 1]) - st->anchor_w[1];
                st->x_loc[std::size_t(i) * 3u + 2u] =
                    static_cast<double>(cur_pos_f[i * 4 + 2]) - st->anchor_w[2];
            }
            st->v_world[std::size_t(i) * 3u + 0u] = 0.0;
            st->v_world[std::size_t(i) * 3u + 1u] = 0.0;
            st->v_world[std::size_t(i) * 3u + 2u] = 0.0;
        }
        st->pin_world_prev = st->pin_world_cur;
        st->smooth_dt = static_cast<double>(dt);
        st->initialized = true;
    } else {
        // Exact lossless FP64 coordinate-origin shift (`x_world = old_anchor + x_old = new_anchor + x_new`).
        for (std::int32_t i = 0; i < n_part; ++i) {
            if (st->pin_mask[std::size_t(i)] == 0u) {
                st->x_loc[std::size_t(i) * 3u + 0u] -= d_anc[0];
                st->x_loc[std::size_t(i) * 3u + 1u] -= d_anc[1];
                st->x_loc[std::size_t(i) * 3u + 2u] -= d_anc[2];
            }
        }
    }

    // 1. Reference-bone motion transfer in exact FP64 (`cfg0[0x1e] != 0`).
    const u8* cfg0 = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
    if (cfg0[0x1e] != 0u) {
        const std::int32_t tset_idx = *reinterpret_cast<const std::int32_t*>(sim_data + 0x150);
        const std::int32_t bone_idx = *reinterpret_cast<const std::int32_t*>(sim_data + 0x154);
        const u8* tset = (*reinterpret_cast<const u8* const* const*>(ctx + 0x40))[tset_idx];
        const u8* bone_ptr =
            *reinterpret_cast<const u8* const*>(tset + 0x10) + (std::size_t(bone_idx) << 6);

        HkTransform4x4 new_ref{};
        using SetQTransformFn = void(PS4_SYSV_ABI*)(void*, const void*);
        ResolveFn<SetQTransformFn>(0x043b8c0ULL)(&new_ref, bone_ptr);
        if (!teleported) {
            ApplyTransferMotionFp64(*st, sim_data, sim + 0x120, &new_ref, dt);
        }
        _mm_store_ps(reinterpret_cast<float*>(sim + 0x120), new_ref.col[0]);
        _mm_store_ps(reinterpret_cast<float*>(sim + 0x130), new_ref.col[1]);
        _mm_store_ps(reinterpret_cast<float*>(sim + 0x140), new_ref.col[2]);
        _mm_store_ps(reinterpret_cast<float*>(sim + 0x150), new_ref.col[3]);
    }

    // 2. Drive per-instance collidables from skeleton (`sim_data + 0x80 >= 0`), computing
    //    `linVel` in FP64 to avoid 500 m world-space subtraction quantization.
    const std::int32_t n_inst_col = *reinterpret_cast<const std::int32_t*>(sim + 0xd8);
    std::vector<HkTransform4x4> saved_world_cols;
    if (*reinterpret_cast<const std::int32_t*>(sim_data + 0x80) >= 0 && n_inst_col > 0) {
        MonitorPushTimer(0x47b31baULL);
        saved_world_cols.resize(static_cast<std::size_t>(n_inst_col));
        const std::int32_t col_tset_idx = *reinterpret_cast<const std::int32_t*>(sim_data + 0x80);
        const u8* col_tset = (*reinterpret_cast<const u8* const* const*>(ctx + 0x40))[col_tset_idx];
        using SetColTransformFn =
            void(PS4_SYSV_ABI*)(u8*, const float*, const float*, int, float);

        for (std::int32_t c = 0; c < n_inst_col; ++c) {
            u8* col = (*reinterpret_cast<u8* const* const*>(sim + 0xd0))[c];
            const std::int32_t b_idx =
                (*reinterpret_cast<const std::int32_t* const*>(sim_data + 0x88))[c];
            const __m128i* off_ptr = reinterpret_cast<const __m128i*>(
                *reinterpret_cast<const u8* const*>(sim_data + 0x98) + std::size_t(c) * 0x40u);
            const __m128* bone_m = reinterpret_cast<const __m128*>(
                *reinterpret_cast<const u8* const*>(col_tset + 0x10) + (std::size_t(b_idx) << 6));

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

            // Compute collider world position in FP64 (`bone_t + bone_R * off_t`) for exact velocity.
            alignas(16) float b0f[4], b1f[4], b2f[4], b3f[4], o3f[4];
            _mm_store_ps(b0f, b0);
            _mm_store_ps(b1f, b1);
            _mm_store_ps(b2f, b2);
            _mm_store_ps(b3f, b3);
            _mm_store_ps(o3f, o3);
            const double new_col_w[3] = {
                double(b3f[0]) + double(b0f[0]) * double(o3f[0]) + double(b1f[0]) * double(o3f[1]) +
                    double(b2f[0]) * double(o3f[2]),
                double(b3f[1]) + double(b0f[1]) * double(o3f[0]) + double(b1f[1]) * double(o3f[1]) +
                    double(b2f[1]) * double(o3f[2]),
                double(b3f[2]) + double(b0f[2]) * double(o3f[0]) + double(b1f[2]) * double(o3f[1]) +
                    double(b2f[2]) * double(o3f[2]),
            };
            world_m.col[3] = _mm_setr_ps(static_cast<float>(new_col_w[0]),
                                         static_cast<float>(new_col_w[1]),
                                         static_cast<float>(new_col_w[2]), 1.0f);
            saved_world_cols[std::size_t(c)] = world_m;

            const float* old_col_p = reinterpret_cast<const float*>(col + 0x50);
            const double old_col_w[3] = {old_col_p[0], old_col_p[1], old_col_p[2]};

            ResolveFn<SetColTransformFn>(0x0767cf0ULL)(
                col, reinterpret_cast<const float*>(&world_m),
                reinterpret_cast<const float*>(col + 0x20), 0, dt);

            if (teleported) {
                _mm_store_ps(reinterpret_cast<float*>(col + 0x60), _mm_setzero_ps());
                _mm_store_ps(reinterpret_cast<float*>(col + 0x70), _mm_setzero_ps());
                _mm_store_ps(reinterpret_cast<float*>(col + 0x20), world_m.col[0]);
                _mm_store_ps(reinterpret_cast<float*>(col + 0x30), world_m.col[1]);
                _mm_store_ps(reinterpret_cast<float*>(col + 0x40), world_m.col[2]);
                _mm_store_ps(reinterpret_cast<float*>(col + 0x50), world_m.col[3]);
            } else {
                const double inv_dt_d = 1.0 / static_cast<double>(dt);
                _mm_store_ps(
                    reinterpret_cast<float*>(col + 0x60),
                    _mm_setr_ps(static_cast<float>((new_col_w[0] - old_col_w[0]) * inv_dt_d),
                                static_cast<float>((new_col_w[1] - old_col_w[1]) * inv_dt_d),
                                static_cast<float>((new_col_w[2] - old_col_w[2]) * inv_dt_d),
                                0.0f));
            }
        }
        MonitorPushTimer(0x47b317fULL);
    }

    // 3. Assemble `sim->m_activeCollidables` (`+0x1c0`) and shift `col + 0x50` by `-anchor_w`
    //    so all 8 Havok capsule/convex/sphere collision kernels operate in local space!
    const std::int32_t n_world_col = *reinterpret_cast<const std::int32_t*>(sim + 0xe8);
    const u8* cfg2 = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
    bool has_landscape = false;
    if (cfg2[0x1d] != 0u) {
        const u8* tri_src = *reinterpret_cast<const u8* const*>(sim + 0xf8);
        const u8* cvx_src = *reinterpret_cast<const u8* const*>(sim + 0x100);
        const bool any_src =
            (tri_src && *reinterpret_cast<const std::int32_t*>(tri_src + 0x18) != 0) ||
            (cvx_src && *reinterpret_cast<const std::int32_t*>(cvx_src + 0x18) != 0);
        if (any_src && *reinterpret_cast<const std::int32_t*>(sim_data + 0x148) != 0 &&
            *reinterpret_cast<const u8* const*>(sim + 0xf0) != nullptr) {
            has_landscape = true;
        }
    }

    const std::int32_t base_col_count = n_inst_col + n_world_col;
    const std::int32_t total_col_count = base_col_count + (has_landscape ? 1 : 0);

    if (*reinterpret_cast<const std::int32_t*>(sim + 0x1bc) < total_col_count) {
        u8* old_buf = *reinterpret_cast<u8* const*>(sim + 0x1c0);
        u8* temp_alloc = GetHkTempAllocator();
        ResolveFn<void(PS4_SYSV_ABI*)(u8*, void*)>(0x0429a90ULL)(temp_alloc, old_buf);
        u8* new_buf = ResolveFn<u8*(PS4_SYSV_ABI*)(u8*, int)>(0x0429a40ULL)(
            GetHkTempAllocator(), total_col_count * 0x90);
        *reinterpret_cast<u8**>(sim + 0x1c0) = new_buf;
        *reinterpret_cast<std::int32_t*>(sim + 0x1bc) = total_col_count;
    }
    *reinterpret_cast<std::int32_t*>(sim + 0x1b8) = total_col_count;

    using HkMemCpyFn = void(PS4_SYSV_ABI*)(void*, const void*, int);
    for (std::int32_t i = 0; i < base_col_count; ++i) {
        u8* dst = *reinterpret_cast<u8* const*>(sim + 0x1c0) + std::size_t(i) * 0x90u;
        const u8* src = (i < n_inst_col)
                            ? (*reinterpret_cast<const u8* const* const*>(sim + 0xd0))[i]
                            : (*reinterpret_cast<const u8* const* const*>(sim + 0xe0))[i - n_inst_col];
        ResolveFn<HkMemCpyFn>(0x0446140ULL)(dst, src, 0x90);
        float* p_col = reinterpret_cast<float*>(dst + 0x50);
        p_col[0] = static_cast<float>(static_cast<double>(p_col[0]) - st->anchor_w[0]);
        p_col[1] = static_cast<float>(static_cast<double>(p_col[1]) - st->anchor_w[1]);
        p_col[2] = static_cast<float>(static_cast<double>(p_col[2]) - st->anchor_w[2]);
    }
    if (has_landscape) {
        u8* dst = *reinterpret_cast<u8* const*>(sim + 0x1c0) + std::size_t(base_col_count) * 0x90u;
        const u8* src = *reinterpret_cast<const u8* const*>(sim + 0xf0);
        ResolveFn<HkMemCpyFn>(0x0446140ULL)(dst, src, 0x90);
        float* p_col = reinterpret_cast<float*>(dst + 0x50);
        p_col[0] = static_cast<float>(static_cast<double>(p_col[0]) - st->anchor_w[0]);
        p_col[1] = static_cast<float>(static_cast<double>(p_col[1]) - st->anchor_w[1]);
        p_col[2] = static_cast<float>(static_cast<double>(p_col[2]) - st->anchor_w[2]);
    }

    // 4. Adaptive substep sizing & jitter-free smooth_dt stiffness ratio.
    const u8* cfg1 = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
    std::int32_t ps4_substeps = *reinterpret_cast<const std::int32_t*>(cfg1 + 0x18);
    if (ps4_substeps == 0) {
        ps4_substeps = *reinterpret_cast<const std::int32_t*>(op + 0x20);
    }
    ps4_substeps = std::max(1, ps4_substeps);

    const double ps4_sdt = (1.0 / 30.0) / static_cast<double>(ps4_substeps);
    const int eff_substeps = std::clamp(
        static_cast<int>(std::ceil(static_cast<double>(dt) / ps4_sdt - 0.05)), 1, ps4_substeps);
    const double sdt_d = static_cast<double>(dt) / static_cast<double>(eff_substeps);
    const float substep_dt = static_cast<float>(sdt_d);
    *reinterpret_cast<float*>(sim + 0x108) = substep_dt;

    const double ema_alpha = std::clamp(static_cast<double>(dt) / (1.0 / 30.0), 0.05, 1.0);
    st->smooth_dt += (static_cast<double>(dt) - st->smooth_dt) * ema_alpha;
    const double smooth_sdt = st->smooth_dt / static_cast<double>(eff_substeps);
    const double substep_ratio = smooth_sdt / ps4_sdt;

    const float damp_cfg = *reinterpret_cast<const float*>(cfg1 + 0x10);
    const double base_damp =
        (damp_cfg >= 1.0f) ? 0.0
                           : ((damp_cfg <= 0.0f) ? 1.0 : std::pow(1.0 - double(damp_cfg), sdt_d));
    // Gentle continuous-time PBD micro-relaxation damping (0.96 per 1/30 s at >60 FPS, 1.0 at 30 FPS).
    const double hf_blend = std::clamp((1.0 - smooth_sdt * 30.0) / 0.5, 0.0, 1.0);
    const double hf_damp = std::pow(0.96, sdt_d * 30.0 * hf_blend);
    const double total_vel_damp = base_damp * hf_damp;

    const std::int32_t bytes_planes = ((n_part << 4) + 0x7f) & ~0x7f;
    u8* pinch_planes_buf = HkLifoAlloc(bytes_planes);
    const std::int32_t force_bytes = ((n_part << 4) + 0x7f) & ~0x7f;
    u8* raw_forces = HkLifoAlloc(force_bytes);
    __m128* forces = reinterpret_cast<__m128*>(raw_forces);

    const float* grav_f = reinterpret_cast<const float*>(cfg1 + 0x00);
    const double grav_d[3] = {grav_f[0], grav_f[1], grav_f[2]};
    const u8* pdata = *reinterpret_cast<const u8* const*>(sim_data + 0x38);

    int mode = *reinterpret_cast<const std::int32_t*>(sim + 0x1cc);
    const float f0 = *reinterpret_cast<const float*>(sim + 0x1d0);
    const float f1 = *reinterpret_cast<const float*>(sim + 0x1d4);
    if (mode == 1) {
        mode = (op[0x38] != 0u) ? 2 : 1;
    }
    const int use_factor = 1;
    const std::int32_t n_cset_iters = *reinterpret_cast<const std::int32_t*>(op + 0x24);
    const std::int32_t n_execs = *reinterpret_cast<const std::int32_t*>(op + 0x30);

    using MatToQuatFn = void(PS4_SYSV_ABI*)(__m128*, const void*);
    using QuatToMatFn = void(PS4_SYSV_ABI*)(void*, const __m128*);
    using PinchReleaseFn = void(PS4_SYSV_ABI*)(const u8*, u8*, u8*);
    using ActionApplyFn = void(PS4_SYSV_ABI*)(u8*, u8*, __m128*, float);
    using CollideParticlesFn = void(PS4_SYSV_ABI*)(const u8*, u8*, u8*, float);
    using UpdateTransitionFn = void(PS4_SYSV_ABI*)(u8*, float);

    const __m128 sdt_vec = _mm_set1_ps(substep_dt);
    const __m128 half_sdt_vec = _mm_set1_ps(0.5f * substep_dt);

    // 5. Substep simulation loop in Character-Local FP64 + Local SIMD Coordinates.
    for (int s = 0; s < eff_substeps; ++s) {
        // 5a. Advance local collidables (`col + 0x20..0x5f`).
        if (total_col_count > 0) {
            for (std::int32_t k = 0; k < total_col_count; ++k) {
                u8* col = *reinterpret_cast<u8* const*>(sim + 0x1c0) + std::size_t(k) * 0x90u;
                const __m128 lin_v = _mm_load_ps(reinterpret_cast<const float*>(col + 0x60));
                const __m128 old_t = _mm_load_ps(reinterpret_cast<const float*>(col + 0x50));
                _mm_store_ps(reinterpret_cast<float*>(col + 0x50),
                             _mm_add_ps(_mm_mul_ps(sdt_vec, lin_v), old_t));

                alignas(16) __m128 q_cur{};
                ResolveFn<MatToQuatFn>(0x0449900ULL)(&q_cur, col + 0x20);
                const __m128 half_w = _mm_mul_ps(
                    half_sdt_vec, _mm_load_ps(reinterpret_cast<const float*>(col + 0x70)));
                const float x = _mm_cvtss_f32(_mm_dp_ps(half_w, half_w, 0x7f));
                constexpr float kC0 = std::bit_cast<float>(0x3ecf817au);
                constexpr float kC1 = std::bit_cast<float>(0xbeaac436u);
                constexpr float kC2 = std::bit_cast<float>(0xbe05a965u);
                constexpr float kC3 = std::bit_cast<float>(0xbc93704cu);
                const float t = x * kC0;
                const float t2 = t * t;
                const float w_comp = ((x * kC1 + 1.0f) + t2 * kC2) + (x * kC3) * t2;
                const __m128 dq = _mm_insert_ps(half_w, _mm_set_ss(w_comp), 0x30);

                const __m128 q1 = q_cur;
                const __m128 dq_w = _mm_shuffle_ps(dq, dq, 0xff);
                const __m128 q1_w = _mm_shuffle_ps(q1, q1, 0xff);
                __m128 cross = _mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(q1, q1, 0xc9), dq),
                                          _mm_mul_ps(q1, _mm_shuffle_ps(dq, dq, 0xc9)));
                cross = _mm_shuffle_ps(cross, cross, 0xc9);
                const __m128 xyz =
                    _mm_add_ps(_mm_mul_ps(q1_w, dq), _mm_add_ps(_mm_mul_ps(q1, dq_w), cross));
                const __m128 prod = _mm_mul_ps(q1, dq);
                const __m128 w_new =
                    _mm_sub_ps(_mm_shuffle_ps(prod, prod, 0xff), _mm_dp_ps(dq, q1, 0x7f));
                alignas(16) __m128 q_next = NormalizeQuatNR(_mm_blend_ps(xyz, w_new, 0x08));
                ResolveFn<QuatToMatFn>(0x0429b00ULL)(col + 0x20, &q_next);
            }
        }

        // 5b. Interpolate pinned particle positions in exact FP64 across substeps.
        const double frac = static_cast<double>(s + 1) / static_cast<double>(eff_substeps);
        const double one_minus_frac = 1.0 - frac;
        for (std::int32_t i = 0; i < n_part; ++i) {
            if (st->pin_mask[std::size_t(i)] != 0u) {
                const std::size_t i3 = std::size_t(i) * 3u;
                const double wx =
                    one_minus_frac * st->pin_world_prev[i3 + 0u] + frac * st->pin_world_cur[i3 + 0u];
                const double wy =
                    one_minus_frac * st->pin_world_prev[i3 + 1u] + frac * st->pin_world_cur[i3 + 1u];
                const double wz =
                    one_minus_frac * st->pin_world_prev[i3 + 2u] + frac * st->pin_world_cur[i3 + 2u];
                st->x_loc[i3 + 0u] = wx - st->anchor_w[0];
                st->x_loc[i3 + 1u] = wy - st->anchor_w[1];
                st->x_loc[i3 + 2u] = wz - st->anchor_w[2];
            }
        }

        // 5c. Evaluate world-space wind actions (`SprjWindAction_apply` reads world pos for spatial
        //     XZ wave phase & point-wind attenuation, and reads exact FP64 `v_world` via
        //     `LocalFp64_CopyParticleVelocities`!).
        std::memset(forces, 0, static_cast<std::size_t>(n_part) * 16u);
        if (*reinterpret_cast<const std::int32_t*>(sim + 0x170) > 0) {
            for (std::int32_t i = 0; i < n_part; ++i) {
                const std::size_t i3 = std::size_t(i) * 3u;
                const float wx = static_cast<float>(st->anchor_w[0] + st->x_loc[i3 + 0u]);
                const float wy = static_cast<float>(st->anchor_w[1] + st->x_loc[i3 + 1u]);
                const float wz = static_cast<float>(st->anchor_w[2] + st->x_loc[i3 + 2u]);
                cur_pos_f[i * 4 + 0] = wx;
                cur_pos_f[i * 4 + 1] = wy;
                cur_pos_f[i * 4 + 2] = wz;
                cur_pos_f[i * 4 + 3] = 0.0f;
                prev_pos_f[i * 4 + 0] =
                    static_cast<float>(double(wx) - st->v_world[i3 + 0u] * sdt_d);
                prev_pos_f[i * 4 + 1] =
                    static_cast<float>(double(wy) - st->v_world[i3 + 1u] * sdt_d);
                prev_pos_f[i * 4 + 2] =
                    static_cast<float>(double(wz) - st->v_world[i3 + 2u] * sdt_d);
                prev_pos_f[i * 4 + 3] = 0.0f;
            }
            for (std::int32_t b = 0; b < *reinterpret_cast<const std::int32_t*>(sim + 0x170); ++b) {
                u8* action = (*reinterpret_cast<u8* const* const*>(sim + 0x168))[b];
                if (action != nullptr) {
                    void* const* vtbl = *reinterpret_cast<void* const* const*>(action);
                    reinterpret_cast<ActionApplyFn>(vtbl[0x28 / 8])(action, sim, forces, substep_dt);
                }
            }
        }

        // 5d. Symplectic FP64 integration -> local SIMD buffers (`|coord| < 1.5 m` = 60 nm ULP!).
        *reinterpret_cast<std::int32_t*>(sim + 0x1b0) = 0;
        for (std::int32_t i = 0; i < n_part; ++i) {
            const std::size_t i3 = std::size_t(i) * 3u;
            if (st->pin_mask[std::size_t(i)] != 0u) {
                const float lx = static_cast<float>(st->x_loc[i3 + 0u]);
                const float ly = static_cast<float>(st->x_loc[i3 + 1u]);
                const float lz = static_cast<float>(st->x_loc[i3 + 2u]);
                cur_pos_f[i * 4 + 0] = lx;
                cur_pos_f[i * 4 + 1] = ly;
                cur_pos_f[i * 4 + 2] = lz;
                cur_pos_f[i * 4 + 3] = 0.0f;
                prev_pos_f[i * 4 + 0] = lx;
                prev_pos_f[i * 4 + 1] = ly;
                prev_pos_f[i * 4 + 2] = lz;
                prev_pos_f[i * 4 + 3] = 0.0f;
                continue;
            }

            const u8* pd = pdata + std::size_t(i) * 16u;
            const double inv_mass = static_cast<double>(*reinterpret_cast<const float*>(pd + 0x04));
            const float* f_vec = reinterpret_cast<const float*>(&forces[i]);

            double vx = st->v_world[i3 + 0u] * total_vel_damp +
                        (grav_d[0] + inv_mass * static_cast<double>(f_vec[0])) * sdt_d;
            double vy = st->v_world[i3 + 1u] * total_vel_damp +
                        (grav_d[1] + inv_mass * static_cast<double>(f_vec[1])) * sdt_d;
            double vz = st->v_world[i3 + 2u] * total_vel_damp +
                        (grav_d[2] + inv_mass * static_cast<double>(f_vec[2])) * sdt_d;

            const double x0 = st->x_loc[i3 + 0u];
            const double y0 = st->x_loc[i3 + 1u];
            const double z0 = st->x_loc[i3 + 2u];

            const double xp = x0 + vx * sdt_d;
            const double yp = y0 + vy * sdt_d;
            const double zp = z0 + vz * sdt_d;

            // `prev_pos_f` stores local `x0`, `cur_pos_f` stores local `xp = x0 + v*sdt`
            // so `cur - prev == v * sdt` in 60-nm local precision for Havok's capsule collision!
            prev_pos_f[i * 4 + 0] = static_cast<float>(x0);
            prev_pos_f[i * 4 + 1] = static_cast<float>(y0);
            prev_pos_f[i * 4 + 2] = static_cast<float>(z0);
            prev_pos_f[i * 4 + 3] = 0.0f;

            cur_pos_f[i * 4 + 0] = static_cast<float>(xp);
            cur_pos_f[i * 4 + 1] = static_cast<float>(yp);
            cur_pos_f[i * 4 + 2] = static_cast<float>(zp);
            cur_pos_f[i * 4 + 3] = 0.0f;
        }

        // 5e. Landscape pinch/release + full constraint & capsule collision solve in local space.
        ResolveFn<PinchReleaseFn>(0x076dcc0ULL)(op, sim, pinch_planes_buf);

        if (n_cset_iters > 0) {
            if (n_execs > 0) {
                for (std::int32_t iter = 0; iter < n_cset_iters; ++iter) {
                    for (std::int32_t k = 0; k < n_execs; ++k) {
                        const std::int32_t exec_id =
                            (*reinterpret_cast<const std::int32_t* const*>(op + 0x28))[k];
                        if (exec_id == -1) {
                            ResolveFn<CollideParticlesFn>(0x0774260ULL)(op, ctx, sim, substep_dt);
                        } else {
                            u8* cset =
                                (*reinterpret_cast<u8* const* const*>(sim_data + 0xb8))[exec_id];
                            const float factor = ComputeMode11ConstraintFactor(
                                cset, mode, s, eff_substeps, ps4_substeps, f0, f1, substep_ratio);
                            DispatchSingleConstraintSetLocalFp64(*st, cset, sim, ctx, exec_id,
                                                                 use_factor, factor);
                        }
                    }
                }
                const u8* cfg_tr = ResolveFn<GetActiveCfgFn>(0x0748c50ULL)(sim);
                if (cfg_tr[0x1c] != 0u && *reinterpret_cast<const std::int32_t*>(sim_data + 0xd0) > 0) {
                    for (std::int32_t b = 0;
                         b < *reinterpret_cast<const std::int32_t*>(sim_data + 0xd0); ++b) {
                        const u8* ap_cset =
                            (*reinterpret_cast<const u8* const* const*>(sim_data + 0xc8))[b];
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
                            if (trans_inst) {
                                ResolveFn<UpdateTransitionFn>(0x0771c80ULL)(trans_inst, substep_dt);
                            }
                        }
                    }
                }
            } else {
                const std::int32_t n_csets =
                    *reinterpret_cast<const std::int32_t*>(sim_data + 0xc0);
                for (std::int32_t iter = 0; iter < n_cset_iters; ++iter) {
                    for (std::int32_t i = 0; i < n_csets; ++i) {
                        u8* cset = (*reinterpret_cast<u8* const* const*>(sim_data + 0xb8))[i];
                        const float factor = ComputeMode11ConstraintFactor(
                            cset, mode, s, eff_substeps, ps4_substeps, f0, f1, substep_ratio);
                        DispatchSingleConstraintSetLocalFp64(*st, cset, sim, ctx, i, use_factor,
                                                             factor);
                    }
                    ResolveFn<CollideParticlesFn>(0x0774260ULL)(op, ctx, sim, substep_dt);
                }
            }
        }

        // 5f. Recover post-constraint FP64 local positions & inelastic velocities.
        const double inv_sdt_d = 1.0 / sdt_d;
        constexpr double kMaxVelSq = 40.0 * 40.0;
        for (std::int32_t i = 0; i < n_part; ++i) {
            if (st->pin_mask[std::size_t(i)] != 0u) {
                continue;
            }
            const std::size_t i3 = std::size_t(i) * 3u;
            const double nx = static_cast<double>(cur_pos_f[i * 4 + 0]);
            const double ny = static_cast<double>(cur_pos_f[i * 4 + 1]);
            const double nz = static_cast<double>(cur_pos_f[i * 4 + 2]);
            const double px = static_cast<double>(prev_pos_f[i * 4 + 0]);
            const double py = static_cast<double>(prev_pos_f[i * 4 + 1]);
            const double pz = static_cast<double>(prev_pos_f[i * 4 + 2]);

            if (!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz) ||
                !std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) {
                continue;
            }

            st->x_loc[i3 + 0u] = nx;
            st->x_loc[i3 + 1u] = ny;
            st->x_loc[i3 + 2u] = nz;

            double vx = (nx - px) * inv_sdt_d;
            double vy = (ny - py) * inv_sdt_d;
            double vz = (nz - pz) * inv_sdt_d;
            const double v2 = vx * vx + vy * vy + vz * vz;
            if (v2 > kMaxVelSq) {
                const double sc = 40.0 / std::sqrt(v2);
                vx *= sc;
                vy *= sc;
                vz *= sc;
            }
            st->v_world[i3 + 0u] = vx;
            st->v_world[i3 + 1u] = vy;
            st->v_world[i3 + 2u] = vz;
        }
    }

    st->pin_world_prev = st->pin_world_cur;

    // 6. Compute cloth normals in local coordinates (zero 500 m FP32 edge cancellation!).
    if (sim_data[0x14c] != 0u) {
        ResolveFn<void(PS4_SYSV_ABI*)(const u8*, u8*)>(0x076f040ULL)(op, sim);
    }

    // 7. Reconstruct world-space `m_currentPositions` & `m_previousPositions` for skinning & AABB.
    const double dt_d = static_cast<double>(dt);
    for (std::int32_t i = 0; i < n_part; ++i) {
        const std::size_t i3 = std::size_t(i) * 3u;
        const double wx = st->anchor_w[0] + st->x_loc[i3 + 0u];
        const double wy = st->anchor_w[1] + st->x_loc[i3 + 1u];
        const double wz = st->anchor_w[2] + st->x_loc[i3 + 2u];
        cur_pos_f[i * 4 + 0] = static_cast<float>(wx);
        cur_pos_f[i * 4 + 1] = static_cast<float>(wy);
        cur_pos_f[i * 4 + 2] = static_cast<float>(wz);
        cur_pos_f[i * 4 + 3] = 0.0f;

        if (st->pin_mask[std::size_t(i)] != 0u) {
            prev_pos_f[i * 4 + 0] = static_cast<float>(wx);
            prev_pos_f[i * 4 + 1] = static_cast<float>(wy);
            prev_pos_f[i * 4 + 2] = static_cast<float>(wz);
            prev_pos_f[i * 4 + 3] = 0.0f;
        } else {
            prev_pos_f[i * 4 + 0] = static_cast<float>(wx - st->v_world[i3 + 0u] * dt_d);
            prev_pos_f[i * 4 + 1] = static_cast<float>(wy - st->v_world[i3 + 1u] * dt_d);
            prev_pos_f[i * 4 + 2] = static_cast<float>(wz - st->v_world[i3 + 2u] * dt_d);
            prev_pos_f[i * 4 + 3] = 0.0f;
        }
    }

    // 8. Update world-space AABBs and write back world transforms to per-instance collidables.
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

    if (!saved_world_cols.empty()) {
        for (std::int32_t i = 0; i < n_inst_col; ++i) {
            u8* dst_col = (*reinterpret_cast<u8* const* const*>(sim + 0xd0))[i];
            const HkTransform4x4& wm = saved_world_cols[std::size_t(i)];
            _mm_store_ps(reinterpret_cast<float*>(dst_col + 0x20), wm.col[0]);
            _mm_store_ps(reinterpret_cast<float*>(dst_col + 0x30), wm.col[1]);
            _mm_store_ps(reinterpret_cast<float*>(dst_col + 0x40), wm.col[2]);
            _mm_store_ps(reinterpret_cast<float*>(dst_col + 0x50), wm.col[3]);
        }
    }

    MonitorPushTimer(0x47b317fULL);
    HkLifoFree(raw_forces, force_bytes);
    HkLifoFree(pinch_planes_buf, bytes_planes);
    return 0;
}

} // namespace BbClothExp
