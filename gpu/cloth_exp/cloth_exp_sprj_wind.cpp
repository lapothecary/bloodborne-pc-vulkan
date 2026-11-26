// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// File: gpu/cloth_exp/cloth_exp_sprj_wind.cpp
//
// Exact 1-to-1 C++20 machine-logic port of FromSoftware's SprjClothWind
// simulation action (`SprjWindAction_apply`, `SprjWindEvalWorldDir`,
// `SprjPointWind_apply`, `SprjDirWind_apply` — 10,196 B in `out/boot.bin`).
// Zero stubs, zero TLS additions, strict x86-64-v3 AVX/SSE intrinsics.
// ============================================================================

#include "cloth_exp/cloth_exp_sprj_wind.h"

#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

#include "cloth_exp/cloth_exp_local_fp64.h"
#include "cloth_exp/cloth_exp_registry.h"
#include "cloth_exp/cloth_exp_runtime.h"
#include "cloth_exp/cloth_exp_xpbd.h"
#include "shim/bbport_cloth_interp.h"
#include "shim/bbport_frame_rate.h"

namespace BbClothExp {

extern std::atomic<std::uint64_t> g_cloth_exp_dispatch_counter;

namespace {

// Exact PS4 Xorshift128 step + modulo-(2^23 + 1) uniform float in [0.0f, 1.0f]
// matching `0x10523db..0x1052447`, `0x1052bcc..0x1052c38`, `0x1053f14..0x105418e`.
inline float XorShift128NextUnitFloat(std::uint32_t s[4]) noexcept {
    const std::uint32_t x = s[0];
    const std::uint32_t t = x ^ (x << 11);
    s[0] = s[1];
    s[1] = s[2];
    const std::uint32_t w = s[3];
    s[2] = w;
    const std::uint32_t nw = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    s[3] = nw;

    const std::uint64_t prod = static_cast<std::uint64_t>(nw) * 0x00000000fffffe01ULL;
    const std::uint32_t quot = static_cast<std::uint32_t>(prod >> 55);
    const std::uint32_t neg_q = 0u - (quot * 0x00800001u);
    const std::uint32_t fbits = nw + neg_q + 0x3f800000u;
    return std::bit_cast<float>(fbits) + (-1.0f);
}

// Exact PS4 Sony Vectormath SIMD sincos + ZYX Euler quaternion + 3x3 rotation of `vec`
// matching `0x10525cc..0x1052abf` and `0x1052dde..0x10531f8` down to 0 ULP.
inline __m128 RotateVecByEulerXYZ_ExactPS4(__m128 euler_in, __m128 vec) noexcept {
    // Pack {euler.x, euler.y, euler.z, 0.0f} * 0.5f
    const __m128 e_y = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(euler_in), 0x01));
    __m128 e_z = _mm_movehl_ps(euler_in, euler_in);
    const __m128 e_xy = _mm_unpacklo_ps(euler_in, e_y);
    const __m128i zero_i = _mm_setzero_si128();
    const __m128 zero_f = _mm_castsi128_ps(zero_i);
    e_z = _mm_move_ss(zero_f, e_z);
    const __m128 half_ang = _mm_mul_ps(_mm_movelh_ps(e_xy, e_z), _mm_set1_ps(0.5f));

    // Cody-Waite range reduction by pi/2
    const __m128 sign_mask = _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0x80000000u)));
    const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
    const __m128 two_over_pi = _mm_castsi128_ps(_mm_set1_epi32(0x3f22f983));
    const __m128 half_signed = _mm_or_ps(_mm_and_ps(half_ang, sign_mask), _mm_set1_ps(0.5f));
    const __m128i q = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(half_ang, two_over_pi), half_signed));
    const __m128i one_i = _mm_set1_epi32(1);
    const __m128i two_i = _mm_set1_epi32(2);
    const __m128i three_i = _mm_set1_epi32(3);
    const __m128i q_cos = _mm_add_epi32(_mm_and_si128(q, three_i), one_i);

    const __m128 qf = _mm_cvtepi32_ps(q);
    const __m128 pio2_hi = _mm_castsi128_ps(_mm_set1_epi32(0x3fc90fda));
    const __m128 pio2_lo = _mm_castsi128_ps(_mm_set1_epi32(0x33a22168));
    const __m128 r_hi = _mm_sub_ps(half_ang, _mm_mul_ps(qf, pio2_hi));
    const __m128 r = _mm_sub_ps(r_hi, _mm_mul_ps(qf, pio2_lo));
    const __m128 r2 = _mm_mul_ps(r, r);
    const __m128 r3 = _mm_mul_ps(r, r2);

    // Cosine & Sine polynomials
    __m128 c = _mm_add_ps(_mm_mul_ps(r2, _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0xbab24993u)))),
                          _mm_castsi128_ps(_mm_set1_epi32(0x3d2aa036)));
    __m128 s = _mm_add_ps(_mm_mul_ps(r2, _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0xb94c8c6eu)))),
                          _mm_castsi128_ps(_mm_set1_epi32(0x3c088342)));
    c = _mm_add_ps(_mm_mul_ps(r2, c), _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0xbeffffdfu))));
    s = _mm_add_ps(_mm_mul_ps(r2, s), _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0xbe2aaaa1u))));
    c = _mm_add_ps(_mm_mul_ps(r2, c), _mm_set1_ps(1.0f));
    s = _mm_add_ps(r, _mm_mul_ps(r3, s));

    const __m128 sin_even = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(q, one_i), zero_i));
    const __m128 cos_even = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(q_cos, one_i), zero_i));
    const __m128 sin_mag = _mm_blendv_ps(c, s, sin_even);
    const __m128 cos_mag = _mm_blendv_ps(c, s, cos_even);

    const __m128 sin_pos = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(q, two_i), zero_i));
    const __m128 cos_pos = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(q_cos, two_i), zero_i));
    __m128 sin_v = _mm_blendv_ps(_mm_xor_ps(sin_mag, sign_mask), sin_mag, sin_pos);
    __m128 cos_v = _mm_blendv_ps(_mm_xor_ps(cos_mag, sign_mask), cos_mag, cos_pos);

    const __m128 is_inf = _mm_castsi128_ps(
        _mm_cmpeq_epi32(_mm_castps_si128(_mm_and_ps(half_ang, abs_mask)), _mm_set1_epi32(0x7f800000)));
    const __m128 is_zero = _mm_cmpeq_ps(half_ang, zero_f);
    sin_v = _mm_or_ps(_mm_blendv_ps(sin_v, half_ang, is_zero), is_inf);
    cos_v = _mm_or_ps(cos_v, is_inf);

    // Sony Vectormath byte-table permutations (`0x472bc90`, `0x472bca0`, `0x472bcb0`):
    // p0 = {sx, cx, cx, cx}, p1 = {cy, sy, cy, cy}, p2 = {cz, cz, sz, cz}
    // n0 = {cx, sx, sx, sx}, n1 = {sy, cy, sy, sy}, n2 = {sz, sz, cz, sz}
    const __m128 cx4 = _mm_shuffle_ps(cos_v, cos_v, 0x00);
    const __m128 sx4 = _mm_shuffle_ps(sin_v, sin_v, 0x00);
    const __m128 cy4 = _mm_shuffle_ps(cos_v, cos_v, 0x55);
    const __m128 sy4 = _mm_shuffle_ps(sin_v, sin_v, 0x55);
    const __m128 cz4 = _mm_shuffle_ps(cos_v, cos_v, 0xaa);
    const __m128 sz4 = _mm_shuffle_ps(sin_v, sin_v, 0xaa);

    const __m128 p0 = _mm_move_ss(cx4, sin_v);
    const __m128 p1 = _mm_blend_ps(cy4, sy4, 0x02);
    const __m128 p2 = _mm_blend_ps(cz4, sz4, 0x04);

    const __m128 n0 = _mm_move_ss(sx4, cos_v);
    const __m128 n1 = _mm_blend_ps(sy4, cy4, 0x02);
    const __m128 n2 = _mm_blend_ps(sz4, cz4, 0x04);

    const __m128 sign_quat = _mm_setr_ps(1.0f, -1.0f, -1.0f, 1.0f); // 0x47381e0
    const __m128 term0 = _mm_mul_ps(_mm_mul_ps(p0, p1), p2);
    const __m128 term1 = _mm_mul_ps(n2, _mm_mul_ps(n1, _mm_mul_ps(n0, sign_quat)));
    const __m128 quat = _mm_add_ps(term0, term1);

    // Quaternion to 3x3 basis columns (`0x1052a07..0x1052a88`)
    const __m128 q2 = _mm_add_ps(quat, quat);
    const __m128 qq2 = _mm_mul_ps(quat, q2);
    const __m128 mask_xyz = _mm_castsi128_ps(_mm_setr_epi32(-1, -1, -1, 0));
    const __m128 one_xyz = _mm_setr_ps(1.0f, 1.0f, 1.0f, 0.0f);

    const __m128 d_a = _mm_and_ps(_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(qq2), 0xc1)), mask_xyz);
    const __m128 d_b = _mm_and_ps(_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(qq2), 0xda)), mask_xyz);
    const __m128 diag = _mm_sub_ps(_mm_sub_ps(one_xyz, d_a), d_b);

    const __m128 cross = _mm_mul_ps(
        _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(quat), 0xd0)),
        _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(q2), 0xe6)));
    const __m128 wmul = _mm_mul_ps(
        _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(quat), 0xff)),
        _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(q2), 0xc9)));

    const __m128 sum_cw = _mm_add_ps(cross, wmul);
    const __m128 dif_cw = _mm_sub_ps(cross, wmul);

    __m128 sh_3 = _mm_shuffle_ps(sum_cw, dif_cw, 0x49);
    sh_3 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(sh_3), 0x78));
    __m128 sh_0 = _mm_move_ss(dif_cw, sum_cw);
    sh_0 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(sh_0), 0x08));

    __m128 col0 = _mm_shuffle_ps(diag, sh_3, 0x4c);
    col0 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(col0), 0x78));
    __m128 col1 = _mm_shuffle_ps(diag, sh_3, 0xed);
    col1 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(col1), 0x72));
    const __m128 col2 = _mm_shuffle_ps(sh_0, diag, 0xe4);

    const __m128 vx = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(vec), 0x00));
    const __m128 vy = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(vec), 0x55));
    const __m128 vz = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(vec), 0xaa));
    const __m128 vw = _mm_mul_ps(
        _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(vec), 0xff)),
        _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f));

    const __m128 xy = _mm_add_ps(_mm_mul_ps(vx, col0), _mm_mul_ps(vy, col1));
    const __m128 zw = _mm_add_ps(vw, _mm_mul_ps(vz, col2));
    return _mm_add_ps(zw, xy);
}

// 1-iteration Newton-Raphson rsqrt unit-normalize3 (`0x1053323..0x1053366`)
inline __m128 FastNormalize3_OneStep(__m128 v) noexcept {
    const __m128 d2 = _mm_dp_ps(v, v, 0x7f);
    const __m128 le0 = _mm_cmple_ps(d2, _mm_setzero_ps());
    const __m128 r0 = _mm_rsqrt_ps(d2);
    const __m128 half_r = _mm_mul_ps(r0, _mm_set1_ps(0.5f));
    const __m128 nr = _mm_sub_ps(_mm_set1_ps(3.0f), _mm_mul_ps(r0, _mm_mul_ps(d2, r0)));
    const __m128 inv_len = _mm_andnot_ps(le0, _mm_mul_ps(half_r, nr));
    return _mm_mul_ps(v, inv_len);
}

// 1-iteration Newton-Raphson length3 (`0x105353b..0x1053574`)
inline __m128 FastLength3_OneStep(__m128 v) noexcept {
    const __m128 d2 = _mm_dp_ps(v, v, 0x7f);
    const __m128 le0 = _mm_cmple_ps(d2, _mm_setzero_ps());
    const __m128 r0 = _mm_rsqrt_ps(d2);
    const __m128 half_r = _mm_mul_ps(r0, _mm_set1_ps(0.5f));
    const __m128 nr = _mm_sub_ps(_mm_set1_ps(3.0f), _mm_mul_ps(r0, _mm_mul_ps(d2, r0)));
    const __m128 len = _mm_mul_ps(d2, _mm_mul_ps(half_r, nr));
    return _mm_andnot_ps(le0, len);
}

// 2-iteration Newton-Raphson sqrt (`0x1053bd4..0x1053c28` and `0x10540d7..0x1054127`)
inline __m128 FastSqrt_TwoStep(__m128 d2) noexcept {
    const __m128 zero = _mm_setzero_ps();
    const __m128 half = _mm_set1_ps(0.5f);
    const __m128 three = _mm_set1_ps(3.0f);
    __m128 r = _mm_rsqrt_ps(d2);
    const __m128 is_zero = _mm_or_ps(_mm_cmpeq_ps(d2, zero), _mm_cmpeq_ps(r, zero));

    __m128 hr = _mm_mul_ps(r, half);
    r = _mm_mul_ps(hr, _mm_sub_ps(three, _mm_mul_ps(r, _mm_mul_ps(r, d2))));
    hr = _mm_mul_ps(r, half);
    r = _mm_mul_ps(hr, _mm_sub_ps(three, _mm_mul_ps(r, _mm_mul_ps(r, d2))));

    return _mm_blendv_ps(_mm_mul_ps(d2, r), d2, is_zero);
}

inline bool ShouldReusePrevParticleWindForce(float sub_rate, float idx_f) noexcept {
    using ModffFn = float(PS4_SYSV_ABI*)(float, float*);
    float int_part = 0.0f;
    const float frac = ResolveFn<ModffFn>(0x2bbf0a8ULL)(sub_rate * idx_f, &int_part);
    const float sum = sub_rate + frac;
    // Exact `vucomiss 1.0f, sum; jbe reuse`: reuse when `1.0f <= sum` or `sum` is NaN
    return !(1.0f > sum);
}

} // namespace

extern "C" {

// ============================================================================
// [36] ClothExp_SprjWindAction_apply @ 0x1051fa0 (844 B)
// ============================================================================
bool PS4_SYSV_ABI ClothExp_SprjWindAction_apply(
    u8* action, u8* sim, __m128* forces, float substep_dt) noexcept {
    g_cloth_exp_dispatch_counter.fetch_add(1, std::memory_order_relaxed);

    if (action[0x48] == 0) {
        return false;
    }
    if (*reinterpret_cast<const u8*>(ResolveVma(0x5122788ULL)) == 0) {
        return false;
    }
    const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
    const float density = *reinterpret_cast<const float*>(sim_data + 0x78);
    // Exact `vucomiss density, 0.0f; jne/jp proceed; else return false`
    if (density == 0.0f) {
        return false;
    }

    u8* const* mgr_slot = reinterpret_cast<u8* const*>(ResolveVma(0x54b46f0ULL));
    u8* mgr = *mgr_slot;
    if (!mgr) {
        using AssertLogFn = void(PS4_SYSV_ABI*)(const char*, int, const char*, const char*);
        ResolveFn<AssertLogFn>(0x20b55b0ULL)(
            reinterpret_cast<const char*>(ResolveVma(0x48ce5a4ULL)),
            0xb1,
            reinterpret_cast<const char*>(ResolveVma(0x48ce5f1ULL)),
            reinterpret_cast<const char*>(ResolveVma(0x48ce648ULL)));
        mgr = *mgr_slot;
    }

    using QueryWindFn = std::int64_t(PS4_SYSV_ABI*)(u8*, u8*);
    std::int64_t wind_handle = ResolveFn<QueryWindFn>(0x0fd5ab0ULL)(
        *reinterpret_cast<u8* const*>(mgr + 0x10), sim);
    if (wind_handle == -1) {
        return false;
    }

    alignas(16) __m128 aabb[2];
    aabb[0] = _mm_load_ps(reinterpret_cast<const float*>(sim + 0x50));
    aabb[1] = _mm_load_ps(reinterpret_cast<const float*>(sim + 0x60));

    using UpdateWindAabbFn = void(PS4_SYSV_ABI*)(std::int64_t*, const __m128*);
    using GetWindListFn = u8*(PS4_SYSV_ABI*)(std::int64_t*);
    ResolveFn<UpdateWindAabbFn>(0x2313810ULL)(&wind_handle, aabb);
    u8* list = ResolveFn<GetWindListFn>(0x2313840ULL)(&wind_handle);

    u8** cur = *reinterpret_cast<u8***>(list + 0x08);
    u8** const end = *reinterpret_cast<u8***>(list + 0x10);
    if (cur == end) {
        return true;
    }

    using IsEnabledFn = bool(PS4_SYSV_ABI*)(u8*);
    using GetTypeFn = int(PS4_SYSV_ABI*)(u8*);
    using GetSubTypeFn = u8(PS4_SYSV_ABI*)(u8*);
    using GetPhaseFn = float(PS4_SYSV_ABI*)(u8*);
    using GetVecFn = __m128(PS4_SYSV_ABI*)(u8*);
    using PointWindFn = bool(PS4_SYSV_ABI*)(const u8*, u8*, __m128*, u8*, float);
    using EvalDirFn = __m128(PS4_SYSV_ABI*)(const u8*, u8*, std::uint64_t);
    using DirWindFn = bool(PS4_SYSV_ABI*)(const u8*, u8*, __m128*, const __m128*, const float*, float);

    std::int32_t dir_count = 0;
    __m128 param_acc = _mm_setzero_ps();
    __m128 dir_max = _mm_setzero_ps();
    __m128 dir_min = _mm_setzero_ps();

    for (; cur != end; ++cur) {
        u8* entry = *cur;
        if (!ResolveFn<IsEnabledFn>(0x2318bf0ULL)(entry)) {
            continue;
        }
        int wtype = ResolveFn<GetTypeFn>(0x23187d0ULL)(entry);
        if (wtype == 0) {
            wtype = static_cast<int>(ResolveFn<GetSubTypeFn>(0x23188a0ULL)(entry));
        }

        if (static_cast<std::uint32_t>(wtype - 1) < 2u) {
            if (action[0xae] == 0) {
                continue;
            }
            const float ph = ResolveFn<GetPhaseFn>(0x2318840ULL)(entry);
            const __m128 v = ResolveFn<GetVecFn>(0x2318b20ULL)(entry);
            const __m128 v_hi = _mm_movehl_ps(v, v);
            const __m128 scaled_z = _mm_mul_ss(_mm_set_ss(ph), v_hi);
            const __m128 ins = _mm_insert_ps(v, scaled_z, 0x20);
            const __m128 mod_v = _mm_blend_ps(ins, v, 0x08);
            param_acc = _mm_add_ps(param_acc, mod_v);

            ResolveFn<PointWindFn>(0x1052af0ULL)(action, sim, forces, entry, substep_dt);
        } else if (wtype == 0) {
            if (action[0xaf] == 0) {
                continue;
            }
            const float ph = ResolveFn<GetPhaseFn>(0x2318840ULL)(entry);
            const __m128 v = ResolveFn<GetVecFn>(0x2318b20ULL)(entry);
            const __m128 v_hi = _mm_movehl_ps(v, v);
            const __m128 scaled_z = _mm_mul_ss(_mm_set_ss(ph), v_hi);
            const __m128 ins = _mm_insert_ps(v, scaled_z, 0x20);
            const __m128 mod_v = _mm_blend_ps(ins, v, 0x08);
            param_acc = _mm_add_ps(param_acc, mod_v);

            const __m128 d = ResolveFn<EvalDirFn>(0x10522f0ULL)(action, entry, 0ULL);
            dir_max = _mm_max_ps(dir_max, d);
            dir_min = _mm_min_ps(dir_min, d);
            ++dir_count;
        }
    }

    if (dir_count > 0) {
        const __m128 cnt_f = _mm_cvtepi32_ps(_mm_set1_epi32(dir_count));
        const __m128 avg_param_v = _mm_mul_ps(_mm_div_ps(_mm_set1_ps(1.0f), cnt_f), param_acc);
        alignas(16) float avg_param[4];
        _mm_store_ps(avg_param, avg_param_v);
        alignas(16) const __m128 sum_dir = _mm_add_ps(dir_max, dir_min);
        ResolveFn<DirWindFn>(0x1053b90ULL)(action, sim, forces, &sum_dir, avg_param, substep_dt);
    }

    return true;
}

// ============================================================================
// [37] ClothExp_SprjWindEvalWorldDir @ 0x10522f0 (2,037 B)
// ============================================================================
static __m128 EvalWorldDirSingleSeed(
    const u8* action, u8* wind_entry, std::uint64_t opt_seed_tag) noexcept {

    using GetPhaseFn = float(PS4_SYSV_ABI*)(u8*);
    using FmodfFn = float(PS4_SYSV_ABI*)(float, float);
    using RngInitFn = void(PS4_SYSV_ABI*)(std::uint32_t*, std::uint32_t);
    using RngSeedFn = void(PS4_SYSV_ABI*)(std::uint32_t*, std::int64_t);
    using RngDtorFn = void(PS4_SYSV_ABI*)(std::uint32_t*);
    using PerlinCtorFn = void(PS4_SYSV_ABI*)(u8*);
    using PerlinDtorFn = void(PS4_SYSV_ABI*)(u8*);
    using PerlinFbmFn = float(PS4_SYSV_ABI*)(u8*, int, float, float, float, float, float);
    using GetFloatRangeFn = void(PS4_SYSV_ABI*)(u8*, float*, float*);
    using GetMulFn = float(PS4_SYSV_ABI*)(u8*);
    using GetMtxFn = void*(PS4_SYSV_ABI*)(__m128*, u8*);
    using GetVecRangeFn = void(PS4_SYSV_ABI*)(u8*, __m128*, __m128*);

    const float phase = ResolveFn<GetPhaseFn>(0x2318840ULL)(wind_entry);

    alignas(16) std::uint32_t rng_buf[8]{};
    rng_buf[0] = 0x075bcd15u;
    std::uint32_t* rng_state = &rng_buf[1];
    ResolveFn<RngInitFn>(0x20f8110ULL)(rng_state, 0x075bcd15u);

    const float rem = ResolveFn<FmodfFn>(0x20b54f0ULL)(phase, 1000.0f);
    std::int64_t seed64 = static_cast<std::int64_t>(rem * 1000.0f);
    if ((opt_seed_tag >> 32) == 0x5eed0001ULL) {
        seed64 = static_cast<std::int64_t>(static_cast<std::int32_t>(opt_seed_tag & 0xffffffffu));
    }
    rng_buf[0] = static_cast<std::uint32_t>(seed64);
    ResolveFn<RngSeedFn>(0x20f81c0ULL)(rng_state, seed64);

    alignas(16) u8 perlin[32]{};
    ResolveFn<PerlinCtorFn>(0x2685930ULL)(perlin);

    const float amp_scale = *reinterpret_cast<const float*>(action + 0x98);
    const float freq = *reinterpret_cast<const float*>(action + 0x9c);
    const float pers = *reinterpret_cast<const float*>(action + 0xa0);

    float gust_min = 0.0f;
    float gust_max = 0.0f;
    ResolveFn<GetFloatRangeFn>(0x2318760ULL)(wind_entry, &gust_min, &gust_max);
    const float u_gust = XorShift128NextUnitFloat(rng_state);
    const float gust_p = gust_min + (gust_max - gust_min) * u_gust;

    float spd_min = 0.0f;
    float spd_max = 0.0f;
    ResolveFn<GetFloatRangeFn>(0x2318690ULL)(wind_entry, &spd_min, &spd_max);

    const float py = gust_p * 6.0f + 400.0f;
    const float pz = gust_p + 150.0f;
    const float n_spd_raw =
        ResolveFn<PerlinFbmFn>(0x2685e30ULL)(perlin, 4, gust_p, py, pz, freq, pers);
    const float n_spd = std::bit_cast<float>(std::bit_cast<std::uint32_t>(amp_scale * n_spd_raw) & 0x7fffffffu);
    const float spd_mul = ResolveFn<GetMulFn>(0x2318a40ULL)(wind_entry);

    alignas(16) __m128 mtx[4];
    ResolveFn<GetMtxFn>(0x2318520ULL)(mtx, wind_entry);
    mtx[3] = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);

    alignas(16) __m128 ang_min = _mm_setzero_ps();
    alignas(16) __m128 ang_max = _mm_setzero_ps();
    ResolveFn<GetVecRangeFn>(0x23185c0ULL)(wind_entry, &ang_min, &ang_max);
    const __m128 ang_span = _mm_sub_ps(ang_max, ang_min);

    const float n_ang_raw =
        ResolveFn<PerlinFbmFn>(0x2685e30ULL)(perlin, 4, gust_p, py, pz, freq, pers);
    const float n_ang = std::bit_cast<float>(std::bit_cast<std::uint32_t>(amp_scale * n_ang_raw) & 0x7fffffffu);
    const __m128 euler = _mm_add_ps(_mm_mul_ps(ang_span, _mm_set1_ps(n_ang)), ang_min);

    const float final_spd = ((spd_max - spd_min) * n_spd + spd_min) * spd_mul;
    const __m128 base_fwd = mtx[2];

    ResolveFn<PerlinDtorFn>(0x2685940ULL)(perlin);
    ResolveFn<RngDtorFn>(0x20f8290ULL)(rng_state);

    const __m128 rot_dir = RotateVecByEulerXYZ_ExactPS4(euler, base_fwd);
    return _mm_mul_ps(_mm_set1_ps(final_spd), rot_dir);
}

__m128 PS4_SYSV_ABI ClothExp_SprjWindEvalWorldDir(
    const u8* action, u8* wind_entry, std::uint64_t opt_seed_tag) noexcept {
    if (!IsNativeFpsMode() || (opt_seed_tag >> 32) == 0x5eed0001ULL) {
        return EvalWorldDirSingleSeed(action, wind_entry, opt_seed_tag);
    }

    constexpr std::uint64_t kSeedTag = 0x5eed000100000000ULL;
    std::uint32_t seed0 = 0;
    std::uint32_t seed1 = 0;
    float smooth = 0.0f;
    float frames = 1.0f;
    BbFrameRate::GetClothWindSmoothParams(seed0, seed1, smooth, frames);

    const __m128 m0 = EvalWorldDirSingleSeed(action, wind_entry, kSeedTag | seed0);
    if (!(smooth > 1e-6f && smooth < 1.0f)) {
        return m0;
    }
    const __m128 m1 = EvalWorldDirSingleSeed(action, wind_entry, kSeedTag | seed1);
    alignas(16) float v0[4]{}, v1[4]{}, out[4]{};
    _mm_store_ps(v0, m0);
    _mm_store_ps(v1, m1);
    BbClothInterp::BlendClothWindVectors3D(v0, v1, smooth, out);
    return _mm_load_ps(out);
}

// ============================================================================
// [38] ClothExp_SprjPointWind_apply @ 0x1052af0 (4,244 B)
// ============================================================================
bool PS4_SYSV_ABI ClothExp_SprjPointWind_apply(
    const u8* action, u8* sim, __m128* forces, u8* wind_entry, float substep_dt) noexcept {

    using GetPhaseFn = float(PS4_SYSV_ABI*)(u8*);
    using GetSubTypeFn = u8(PS4_SYSV_ABI*)(u8*);
    using FmodfFn = float(PS4_SYSV_ABI*)(float, float);
    using RngInitFn = void(PS4_SYSV_ABI*)(std::uint32_t*, std::uint32_t);
    using RngSeedFn = void(PS4_SYSV_ABI*)(std::uint32_t*, std::int64_t);
    using RngDtorFn = void(PS4_SYSV_ABI*)(std::uint32_t*);
    using PerlinCtorFn = void(PS4_SYSV_ABI*)(u8*);
    using PerlinDtorFn = void(PS4_SYSV_ABI*)(u8*);
    using PerlinFbmFn = float(PS4_SYSV_ABI*)(u8*, int, float, float, float, float, float);
    using GetFloatRangeFn = void(PS4_SYSV_ABI*)(u8*, float*, float*);
    using GetMulFn = float(PS4_SYSV_ABI*)(u8*);
    using GetMtxFn = void*(PS4_SYSV_ABI*)(__m128*, u8*);
    using GetVecRangeFn = void(PS4_SYSV_ABI*)(u8*, __m128*, __m128*);
    using GetPosFn = __m128(PS4_SYSV_ABI*)(u8*);
    using HasRadiusFn = u8(PS4_SYSV_ABI*)(u8*);
    using GetRadiusFn = float(PS4_SYSV_ABI*)(u8*);

    const float phase = ResolveFn<GetPhaseFn>(0x2318840ULL)(wind_entry);
    const u8 sub_mode = ResolveFn<GetSubTypeFn>(0x23188a0ULL)(wind_entry);

    const bool native_fps = IsNativeFpsMode();
    std::uint32_t seed0 = 0, seed1 = 0;
    float smooth_t = 0.0f, cloth_frames = 1.0f;
    if (native_fps) {
        BbFrameRate::GetClothWindSmoothParams(seed0, seed1, smooth_t, cloth_frames);
    }

    float gust_min = 0.0f;
    float gust_max = 0.0f;
    ResolveFn<GetFloatRangeFn>(0x2318760ULL)(wind_entry, &gust_min, &gust_max);

    float spd_min = 0.0f;
    float spd_max = 0.0f;
    ResolveFn<GetFloatRangeFn>(0x2318690ULL)(wind_entry, &spd_min, &spd_max);
    const float spd_span = spd_max - spd_min;
    const float spd_mul = ResolveFn<GetMulFn>(0x2318a40ULL)(wind_entry);

    alignas(16) __m128 mtx[4]{};
    alignas(16) __m128 ang_min = _mm_setzero_ps();
    alignas(16) __m128 ang_max = _mm_setzero_ps();
    if (sub_mode != 0) {
        ResolveFn<GetMtxFn>(0x2318520ULL)(mtx, wind_entry);
        mtx[3] = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
        ResolveFn<GetVecRangeFn>(0x23185c0ULL)(wind_entry, &ang_min, &ang_max);
    }
    const __m128 ang_span = _mm_sub_ps(ang_max, ang_min);

    constexpr float kPointFreq = 0.1f;      // 0x48cdf4c
    constexpr float kPointPers = 0.3f;      // 0x48cdf50
    constexpr float kPointAmpScale = 5.0f;  // 0x48cdf54

    auto eval_point_gust = [&](std::int64_t seed_val, float& out_spd, __m128& out_dir) noexcept {
        alignas(16) std::uint32_t local_rng_buf[8]{};
        local_rng_buf[0] = 0x075bcd15u;
        std::uint32_t* local_rng = &local_rng_buf[1];
        ResolveFn<RngInitFn>(0x20f8110ULL)(local_rng, 0x075bcd15u);
        local_rng_buf[0] = static_cast<std::uint32_t>(seed_val);
        ResolveFn<RngSeedFn>(0x20f81c0ULL)(local_rng, seed_val);

        alignas(16) u8 local_perlin[32]{};
        ResolveFn<PerlinCtorFn>(0x2685930ULL)(local_perlin);

        const float u_gust = XorShift128NextUnitFloat(local_rng);
        const float gust_p = gust_min + (gust_max - gust_min) * u_gust;
        const float py = gust_p * 6.0f + 400.0f;
        const float pz = gust_p + 150.0f;

        const float n_spd_raw =
            ResolveFn<PerlinFbmFn>(0x2685e30ULL)(local_perlin, 4, gust_p, py, pz, kPointFreq, kPointPers);
        const float n_spd =
            std::bit_cast<float>(std::bit_cast<std::uint32_t>(n_spd_raw * kPointAmpScale) & 0x7fffffffu);
        out_spd = (spd_span * n_spd + spd_min) * spd_mul;

        out_dir = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
        if (sub_mode != 0) {
            const float n_ang_raw =
                ResolveFn<PerlinFbmFn>(0x2685e30ULL)(local_perlin, 4, gust_p, py, pz, kPointFreq, kPointPers);
            const float n_ang =
                std::bit_cast<float>(std::bit_cast<std::uint32_t>(n_ang_raw * kPointAmpScale) & 0x7fffffffu);
            const __m128 euler = _mm_add_ps(_mm_mul_ps(ang_span, _mm_set1_ps(n_ang)), ang_min);
            out_dir = RotateVecByEulerXYZ_ExactPS4(euler, mtx[2]);
        }

        ResolveFn<PerlinDtorFn>(0x2685940ULL)(local_perlin);
        ResolveFn<RngDtorFn>(0x20f8290ULL)(local_rng);
    };

    float final_spd = 0.0f;
    __m128 world_dir = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
    if (!native_fps) {
        const float rem = ResolveFn<FmodfFn>(0x20b54f0ULL)(phase, 1000.0f);
        eval_point_gust(static_cast<std::int64_t>(rem * 1000.0f), final_spd, world_dir);
    } else {
        float spd0 = 0.0f;
        __m128 dir0 = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
        eval_point_gust(static_cast<std::int64_t>(static_cast<std::int32_t>(seed0)), spd0, dir0);
        if (!(smooth_t > 1e-6f) || seed0 == seed1) {
            final_spd = spd0;
            world_dir = dir0;
        } else {
            float spd1 = 0.0f;
            __m128 dir1 = _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
            eval_point_gust(static_cast<std::int64_t>(static_cast<std::int32_t>(seed1)), spd1, dir1);
            final_spd = spd0 + smooth_t * (spd1 - spd0);
            if (sub_mode != 0) {
                alignas(16) float d0_arr[4], d1_arr[4], dout_arr[4]{0.0f, 0.0f, 0.0f, 1.0f};
                _mm_store_ps(d0_arr, dir0);
                _mm_store_ps(d1_arr, dir1);
                BbClothInterp::BlendClothWindVectors3D(d0_arr, d1_arr, smooth_t, dout_arr);
                dout_arr[3] = 1.0f;
                world_dir = _mm_load_ps(dout_arr);
            }
        }
    }

    const __m128 wind_pos = ResolveFn<GetPosFn>(0x23182b0ULL)(wind_entry);
    u8 has_atten = ResolveFn<HasRadiusFn>(0x2318970ULL)(wind_entry);
    const float radius = ResolveFn<GetRadiusFn>(0x23189e0ULL)(wind_entry);

    float inv_radius = 1.0f;
    if (has_atten != 0) {
        if (radius > 0.0f) {
            inv_radius = 1.0f / radius;
        } else {
            has_atten = 0;
            inv_radius = 1.0f;
        }
    }

    const std::int32_t num_p = *reinterpret_cast<const std::int32_t*>(sim + 0x20);
    const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
    const __m128* cur_pos = *reinterpret_cast<const __m128* const*>(sim + 0x18);
    const float inv_density = 1.0f / *reinterpret_cast<const float*>(sim_data + 0x78);
    const __m128* pdatas = *reinterpret_cast<const __m128* const*>(sim_data + 0x38);
    const __m128* prev_pos = *reinterpret_cast<const __m128* const*>(sim + 0x28);
    const float drag_coeff = *reinterpret_cast<const float*>(action + 0x94);
    const float sub_rate = (action[0xac] != 0) ? *reinterpret_cast<const float*>(action + 0xa4) : 0.0f;

    const __m128 spd_vec4 = _mm_set1_ps(final_spd);
    const __m128 scaled_world_dir = _mm_mul_ps(spd_vec4, world_dir);
    const float inv_dt = 1.0f / substep_dt;
    const __m128 neg_inv_dt_vec = _mm_set1_ps(-inv_dt);

    __m128 cone_fac = _mm_set1_ps(1.0f);
    if (sub_mode != 0) {
        const __m128 d0 = _mm_sub_ps(cur_pos[0], wind_pos);
        const __m128 u_p0 = FastNormalize3_OneStep(d0);
        const __m128 u_wd = FastNormalize3_OneStep(scaled_world_dir);
        const float dp = _mm_cvtss_f32(_mm_dp_ps(u_wd, u_p0, 0x7f));
        const __m128 c_raw = _mm_set1_ps(dp * 0.8f + 0.2f);
        const __m128 ones = _mm_set1_ps(1.0f);
        const __m128 c_lt1 = _mm_cmplt_ps(c_raw, ones);
        cone_fac = _mm_max_ps(_mm_setzero_ps(), _mm_blendv_ps(ones, c_raw, c_lt1));
    }
    const float cone_fac_s = _mm_cvtss_f32(cone_fac);

    const std::int32_t vel_bytes = native_fps ? (((num_p << 4) + 0x7f) & ~0x7f) : 0;
    __m128* fp64_vel =
        (native_fps && num_p > 0) ? reinterpret_cast<__m128*>(HkLifoAlloc(vel_bytes)) : nullptr;
    const bool have_fp64_vel =
        (fp64_vel != nullptr) &&
        (IsXpbdMode() ? Xpbd_CopyParticleVelocities(sim, fp64_vel, num_p)
                      : LocalFp64_CopyParticleVelocities(sim, fp64_vel, num_p));
    const __m128 zero_ps = _mm_setzero_ps();

    auto neg_particle_vel = [&](std::int32_t idx, __m128 pos_v) noexcept -> __m128 {
        return have_fp64_vel ? _mm_sub_ps(zero_ps, fp64_vel[idx])
                             : _mm_mul_ps(neg_inv_dt_vec, _mm_sub_ps(pos_v, prev_pos[idx]));
    };

    const std::int32_t num_normals = *reinterpret_cast<const std::int32_t*>(sim + 0x40);
    if (num_normals == num_p) {
        const __m128* normals = *reinterpret_cast<const __m128* const*>(sim + 0x38);
        if (has_atten != 0) {
            if (num_p > 0) {
                __m128 prev_force = _mm_setzero_ps();
                float idx_f = 0.0f;
                for (std::int32_t i = 0; i < num_p; ++i, idx_f += 1.0f) {
                    if (i == 0 || i == num_p - 1 || !ShouldReusePrevParticleWindForce(sub_rate, idx_f)) {
                        const __m128 pos = cur_pos[i];
                        const __m128 diff = _mm_sub_ps(pos, wind_pos);
                        const float dist = _mm_cvtss_f32(FastLength3_OneStep(diff));
                        const float nr = inv_radius * dist;
                        float cl = 0.0f;
                        if (!(0.0f > nr)) {
                            cl = (nr < 1.0f) ? nr : 1.0f;
                        }
                        const float atten = cone_fac_s * (1.0f - cl);
                        if (!(atten > 0.0f)) {
                            prev_force = _mm_setzero_ps();
                        } else {
                            const __m128 wind_v = (sub_mode != 0)
                                ? scaled_world_dir
                                : _mm_mul_ps(_mm_set1_ps(final_spd * atten), diff);
                            const float ndot_raw = _mm_cvtss_f32(_mm_dp_ps(diff, normals[i], 0x7f));
                            const float ndot = std::bit_cast<float>(std::bit_cast<std::uint32_t>(ndot_raw) & 0x7fffffffu);
                            const float mass_i = _mm_cvtss_f32(pdatas[i]);
                            const __m128 rel_v = _mm_add_ps(wind_v, neg_particle_vel(i, pos));
                            const float f_scale = (inv_density * mass_i) * (drag_coeff * ndot);
                            prev_force = _mm_mul_ps(rel_v, _mm_set1_ps(f_scale));
                        }
                    }
                    forces[i] = _mm_add_ps(prev_force, forces[i]);
                }
            }
        } else {
            if (num_p > 0) {
                const __m128 dir_cone = _mm_mul_ps(scaled_world_dir, cone_fac);
                const __m128 spd_cone = _mm_mul_ps(spd_vec4, cone_fac);
                __m128 prev_force = _mm_setzero_ps();
                float idx_f = 0.0f;
                for (std::int32_t i = 0; i < num_p; ++i, idx_f += 1.0f) {
                    if (i == 0 || i == num_p - 1 || !ShouldReusePrevParticleWindForce(sub_rate, idx_f)) {
                        const __m128 pos = cur_pos[i];
                        const __m128 diff = _mm_sub_ps(pos, wind_pos);
                        const __m128 wind_v = (sub_mode != 0) ? dir_cone : _mm_mul_ps(spd_cone, diff);
                        const float ndot_raw = _mm_cvtss_f32(_mm_dp_ps(diff, normals[i], 0x7f));
                        const float ndot = std::bit_cast<float>(std::bit_cast<std::uint32_t>(ndot_raw) & 0x7fffffffu);
                        const float mass_i = _mm_cvtss_f32(pdatas[i]);
                        const __m128 rel_v = _mm_add_ps(wind_v, neg_particle_vel(i, pos));
                        const float f_scale = (inv_density * mass_i) * (drag_coeff * ndot);
                        prev_force = _mm_mul_ps(rel_v, _mm_set1_ps(f_scale));
                    }
                    forces[i] = _mm_add_ps(prev_force, forces[i]);
                }
            }
        }
    } else {
        constexpr float kNoNormalDragMul = 0.699999988079071f; // 0x48cdf64 (0.7f)
        const float eff_drag = drag_coeff * kNoNormalDragMul;
        if (has_atten != 0) {
            if (num_p > 0) {
                __m128 prev_force = _mm_setzero_ps();
                float idx_f = 0.0f;
                for (std::int32_t i = 0; i < num_p; ++i, idx_f += 1.0f) {
                    if (i == 0 || i == num_p - 1 || !ShouldReusePrevParticleWindForce(sub_rate, idx_f)) {
                        const __m128 pos = cur_pos[i];
                        const __m128 diff = _mm_sub_ps(pos, wind_pos);
                        const float dist = _mm_cvtss_f32(FastLength3_OneStep(diff));
                        const float nr = inv_radius * dist;
                        float cl = 0.0f;
                        if (!(0.0f > nr)) {
                            cl = (nr < 1.0f) ? nr : 1.0f;
                        }
                        const float atten = cone_fac_s * (1.0f - cl);
                        if (!(atten > 0.0f)) {
                            prev_force = _mm_setzero_ps();
                        } else {
                            const __m128 wind_v = _mm_mul_ps(_mm_set1_ps(final_spd * atten), diff);
                            const float mass_i = _mm_cvtss_f32(pdatas[i]);
                            const __m128 rel_v = _mm_add_ps(wind_v, neg_particle_vel(i, pos));
                            const float f_scale = eff_drag * (inv_density * mass_i);
                            prev_force = _mm_mul_ps(rel_v, _mm_set1_ps(f_scale));
                        }
                    }
                    forces[i] = _mm_add_ps(prev_force, forces[i]);
                }
            }
        } else {
            if (num_p > 0) {
                __m128 prev_force = _mm_setzero_ps();
                float idx_f = 0.0f;
                for (std::int32_t i = 0; i < num_p; ++i, idx_f += 1.0f) {
                    if (i == 0 || i == num_p - 1 || !ShouldReusePrevParticleWindForce(sub_rate, idx_f)) {
                        // Exact `0x1053af0..0x1053b16` instruction sequence
                        const float dm = inv_density * _mm_cvtss_f32(pdatas[i]);
                        const __m128 dm_ss = _mm_set_ss(dm);
                        __m128 v = _mm_add_ps(neg_particle_vel(i, cur_pos[i]), dm_ss);
                        v = _mm_mul_ps(cone_fac, v);
                        prev_force = _mm_mul_ps(_mm_set1_ps(eff_drag * dm), v);
                    }
                    forces[i] = _mm_add_ps(prev_force, forces[i]);
                }
            }
        }
    }

    if (fp64_vel != nullptr) {
        HkLifoFree(fp64_vel, vel_bytes);
    }
    return true;
}

// ============================================================================
// [39] ClothExp_SprjDirWind_apply @ 0x1053b90 (3,071 B)
// ============================================================================
bool PS4_SYSV_ABI ClothExp_SprjDirWind_apply(
    const u8* action,
    u8* sim,
    __m128* forces,
    const __m128* sum_dir_ptr,
    const float* avg_param_ptr,
    float substep_dt) noexcept {

    using FmodfFn = float(PS4_SYSV_ABI*)(float, float);
    using RngInitFn = void(PS4_SYSV_ABI*)(std::uint32_t*, std::uint32_t);
    using RngSeedFn = void(PS4_SYSV_ABI*)(std::uint32_t*, std::int64_t);
    using RngDtorFn = void(PS4_SYSV_ABI*)(std::uint32_t*);
    using FastSinfFn = float(PS4_SYSV_ABI*)(float);

    const __m128 raw_dir = _mm_load_ps(reinterpret_cast<const float*>(sum_dir_ptr));
    const __m128 d2_lanes = _mm_mul_ps(raw_dir, raw_dir);
    const __m128 len2_3d = _mm_add_ps(
        _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(d2_lanes), 0xaa)),
        _mm_add_ps(
            _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(d2_lanes), 0x00)),
            _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(d2_lanes), 0x55))));
    const __m128 wind_spd_vec = FastSqrt_TwoStep(len2_3d);
    const float wind_spd = _mm_cvtss_f32(wind_spd_vec);

    // Mask W to 0 and safe-normalize (`0x1053c36..0x1053ce1`, `0x1053d64`)
    const __m128 zero = _mm_setzero_ps();
    const __m128 mask_xyz = _mm_castsi128_ps(_mm_setr_epi32(-1, -1, -1, 0));
    const __m128 dir_xyz = _mm_or_ps(_mm_and_ps(raw_dir, mask_xyz), _mm_andnot_ps(mask_xyz, zero));
    const __m128 sq4 = _mm_mul_ps(dir_xyz, dir_xyz);
    const __m128 sq_hi = _mm_movehl_ps(sq4, sq4);
    __m128 sum4 = _mm_add_ps(sq4, sq_hi);
    sum4 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(sum4), 0x40));
    const __m128 sum4_sh = _mm_shuffle_ps(sq_hi, sum4, 0xf0);
    sum4 = _mm_add_ps(sum4, sum4_sh);
    const __m128 len2_4 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(sum4), 0xaa));
    const __m128 len4 = FastSqrt_TwoStep(len2_4);

    const __m128 nz_mask = _mm_cmpneq_ps(zero, len4);
    const __m128 ninf_mask = _mm_cmpneq_ps(len2_4, _mm_castsi128_ps(_mm_set1_epi32(0x7f800000)));
    const __m128 qnan_vec = _mm_castsi128_ps(_mm_set1_epi32(0x7fc00000));
    const __m128 unit_dir = _mm_or_ps(
        _mm_andnot_ps(ninf_mask, qnan_vec),
        _mm_and_ps(_mm_and_ps(_mm_div_ps(dir_xyz, len4), nz_mask), ninf_mask));

    const float p0 = avg_param_ptr[0];
    const float p1 = avg_param_ptr[1];
    const float p2_phase = avg_param_ptr[2];
    const float p3_wavenum = avg_param_ptr[3];

    const std::int32_t num_p = *reinterpret_cast<const std::int32_t*>(sim + 0x20);
    const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
    const float inv_density = 1.0f / *reinterpret_cast<const float*>(sim_data + 0x78);
    const float sub_rate = (action[0xad] != 0) ? *reinterpret_cast<const float*>(action + 0xa8) : 0.0f;
    const float inv_dt = 1.0f / substep_dt;
    const __m128 neg_inv_dt_vec = _mm_set1_ps(-inv_dt);

    const __m128* pdatas = *reinterpret_cast<const __m128* const*>(sim_data + 0x38);
    const __m128* cur_pos = *reinterpret_cast<const __m128* const*>(sim + 0x18);
    const __m128* prev_pos = *reinterpret_cast<const __m128* const*>(sim + 0x28);
    const float drag_coeff = *reinterpret_cast<const float*>(action + 0x94);

    const bool native_fps = IsNativeFpsMode();
    std::uint32_t seed0 = 0, seed1 = 0;
    float smooth_t = 0.0f, cloth_frames = 1.0f;
    if (native_fps) {
        BbFrameRate::GetClothWindSmoothParams(seed0, seed1, smooth_t, cloth_frames);
    }

    std::uint32_t rng_state0[4]{};
    std::uint32_t rng_state1[4]{};
    alignas(16) std::uint32_t rng_buf[8]{};
    rng_buf[0] = 0x075bcd15u;
    std::uint32_t* rng_state = &rng_buf[1];
    ResolveFn<RngInitFn>(0x20f8110ULL)(rng_state, 0x075bcd15u);

    const float rem = ResolveFn<FmodfFn>(0x20b54f0ULL)(p2_phase, 1000.0f);
    const std::int64_t seed64 = native_fps ? static_cast<std::int64_t>(static_cast<std::int32_t>(seed0))
                                           : static_cast<std::int64_t>(rem * 1000.0f);
    rng_buf[0] = static_cast<std::uint32_t>(seed64);
    ResolveFn<RngSeedFn>(0x20f81c0ULL)(rng_state, seed64);
    if (native_fps) {
        std::memcpy(rng_state0, rng_state, sizeof(rng_state0));
        alignas(16) std::uint32_t tmp_rng[8]{};
        tmp_rng[0] = 0x075bcd15u;
        ResolveFn<RngInitFn>(0x20f8110ULL)(&tmp_rng[1], 0x075bcd15u);
        ResolveFn<RngSeedFn>(0x20f81c0ULL)(
            &tmp_rng[1], static_cast<std::int64_t>(static_cast<std::int32_t>(seed1)));
        std::memcpy(rng_state1, &tmp_rng[1], sizeof(rng_state1));
        ResolveFn<RngDtorFn>(0x20f8290ULL)(&tmp_rng[1]);
    }

    const std::int32_t vel_bytes = native_fps ? (((num_p << 4) + 0x7f) & ~0x7f) : 0;
    __m128* fp64_vel =
        (native_fps && num_p > 0) ? reinterpret_cast<__m128*>(HkLifoAlloc(vel_bytes)) : nullptr;
    const bool have_fp64_vel =
        (fp64_vel != nullptr) &&
        (IsXpbdMode() ? Xpbd_CopyParticleVelocities(sim, fp64_vel, num_p)
                      : LocalFp64_CopyParticleVelocities(sim, fp64_vel, num_p));

    constexpr float kTurbAddScale = 0.20000000298023224f;  // 0x48cdf7c (0.2f)
    constexpr float kTurbAddBias = 0.05000000074505806f;   // 0x48cdf80 (0.05f)
    constexpr float kTurbMulScale = 0.6000000238418579f;   // 0x48cdf84 (0.6f)
    constexpr float kTurbMulBias = -0.10000000149011612f;  // 0x48cdf88 (-0.1f)
    constexpr float kWaveDivDeg = 180.0f;                  // 0x48cdf8c
    constexpr float kNegPi = -3.1415927410125732f;         // 0x48cdf90
    constexpr float kNoNormalDragMul = 0.699999988079071f; // 0x48cdf74 (0.7f)

    const std::int32_t num_normals = *reinterpret_cast<const std::int32_t*>(sim + 0x40);
    const bool has_normals = (num_normals == num_p);
    const __m128* normals = has_normals ? *reinterpret_cast<const __m128* const*>(sim + 0x38) : nullptr;
    const float const_drag = drag_coeff * kNoNormalDragMul;

    if (num_p > 0) {
        __m128 prev_force = _mm_setzero_ps();
        float idx_f = 0.0f;
        const __m128 dt30_vec = _mm_set1_ps(1.0f / 30.0f);
        for (std::int32_t i = 0; i < num_p; ++i, idx_f += 1.0f) {
            if (i == 0 || i == num_p - 1 || !ShouldReusePrevParticleWindForce(sub_rate, idx_f)) {
                float eff_drag = const_drag;
                if (has_normals) {
                    const float nd = _mm_cvtss_f32(_mm_dp_ps(unit_dir, normals[i], 0x7f));
                    const float abs_nd = (nd > 0.0f) ? nd : (0.0f - nd);
                    eff_drag = drag_coeff * abs_nd;
                }

                const float dm = inv_density * _mm_cvtss_f32(pdatas[i]);
                const __m128 pos = cur_pos[i];
                const __m128 raw_disp = _mm_sub_ps(pos, prev_pos[i]);
                // In Native-FPS Mode 1.1, use exact FP64 velocity for both 30-FPS turbulence
                // displacement (`v * 1/30 s`) and aerodynamic drag (`-v`), avoiding FP32 noise!
                const __m128 disp =
                    have_fp64_vel ? _mm_mul_ps(fp64_vel[i], dt30_vec) : raw_disp;

                auto next_turb_u = [&]() noexcept -> float {
                    if (!native_fps) {
                        return XorShift128NextUnitFloat(rng_state);
                    }
                    const float a = XorShift128NextUnitFloat(rng_state0);
                    if (!(smooth_t > 1e-6f)) {
                        return a;
                    }
                    const float b = XorShift128NextUnitFloat(rng_state1);
                    return a + smooth_t * (b - a);
                };

                const float u0 = next_turb_u();
                const float tx = _mm_cvtss_f32(disp) + (u0 * kTurbAddScale + kTurbAddBias);
                const float u1 = next_turb_u();
                const float turb_x = tx * (u1 * kTurbMulScale + kTurbMulBias);

                const float disp_y = _mm_cvtss_f32(
                    _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(disp), 0x01)));
                const float u2 = next_turb_u();
                const float ty = disp_y + (u2 * kTurbAddScale + kTurbAddBias);
                const float u3 = next_turb_u();
                const float turb_y = ty * (u3 * kTurbMulScale + kTurbMulBias);

                // Exact XZ distance `sqrt(pos.x^2 + pos.z^2)` (`0x10540b4..0x1054127`)
                __m128 xz = _mm_shuffle_ps(pos, zero, 0xd8);
                xz = _mm_mul_ps(xz, xz);
                xz = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(xz), 0xd8));
                xz = _mm_hadd_ps(xz, xz);
                xz = _mm_hadd_ps(xz, xz);
                const float xz_dist = _mm_cvtss_f32(FastSqrt_TwoStep(xz));

                const float disp_z = _mm_cvtss_f32(_mm_movehl_ps(disp, disp));
                const float u4 = next_turb_u();
                const float tz = disp_z + (u4 * kTurbAddScale + kTurbAddBias);
                const float u5 = next_turb_u();
                const float turb_z = tz * (u5 * kTurbMulScale + kTurbMulBias);

                const __m128 turb_vec = _mm_movelh_ps(
                    _mm_unpacklo_ps(_mm_set_ss(turb_x), _mm_set_ss(turb_y)),
                    _mm_move_ss(zero, _mm_set_ss(turb_z)));

                const float angle = (p2_phase + p3_wavenum * xz_dist) / kWaveDivDeg + kNegPi;
                const float s_val = ResolveFn<FastSinfFn>(0x20b51e0ULL)(angle);
                const float w_raw = s_val * 0.5f + 0.5f;
                float w = 0.0f;
                if (!(0.0f > w_raw)) {
                    w = (w_raw < 1.0f) ? w_raw : 1.0f;
                }
                const float spd_factor = p0 * (1.0f - w) + p1 * w;

                const __m128 wind_v = _mm_add_ps(
                    turb_vec, _mm_mul_ps(unit_dir, _mm_set1_ps(wind_spd * spd_factor)));
                const __m128 neg_part_v = have_fp64_vel ? _mm_sub_ps(zero, fp64_vel[i])
                                                        : _mm_mul_ps(neg_inv_dt_vec, raw_disp);
                const __m128 rel_v = _mm_add_ps(neg_part_v, wind_v);
                prev_force = _mm_mul_ps(_mm_set1_ps(eff_drag * dm), rel_v);
            }
            forces[i] = _mm_add_ps(prev_force, forces[i]);
        }
    }

    if (fp64_vel != nullptr) {
        HkLifoFree(fp64_vel, vel_bytes);
    }
    ResolveFn<RngDtorFn>(0x20f8290ULL)(rng_state);
    return true;
}

} // extern "C"

} // namespace BbClothExp
