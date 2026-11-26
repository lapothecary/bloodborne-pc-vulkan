// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// File: gpu/cloth_exp/cloth_exp_world_and_ops.cpp
//
// 100% 1-to-1 C++20 decompilation of PS4 eboot.bin Batch 1 (14 functions).
// Zero stubs, exact register/memory semantics, exact hkMonitorStream tags.
// ============================================================================

#include "cloth_exp/cloth_exp_world_and_ops.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "cloth_exp/cloth_exp_runtime.h"

namespace BbClothExp {

namespace {

[[gnu::always_inline]] inline __m128 HkMaxFloatVec() noexcept {
    // Exact bit pattern from eboot.bin 0x2bd5460 (+0x210): 0x7f7fffee (3.40282002e+38f).
    const float v = std::bit_cast<float>(0x7f7fffeeu);
    return _mm_set1_ps(v);
}

[[gnu::always_inline]] inline __m128 HkMinFloatVec() noexcept {
    // Exact XOR 0x80000000 of 0x7f7fffee: 0xff7fffee (-3.40282002e+38f).
    const float v = std::bit_cast<float>(0xff7fffeeu);
    return _mm_set1_ps(v);
}

[[gnu::always_inline]] inline u8* ResolveShadowBuffer(u8* const* bufs, std::int32_t idx) noexcept {
    u8* b = bufs[idx];
    const std::int32_t sh = *reinterpret_cast<const std::int32_t*>(b + 0x100);
    return bufs[sh];
}

/// Converts a 32-bit PRNG output (`0x29add90`) to `[0.0f, 1.0f)` using FromSoftware's exact
/// `mod 0x800001` mantissa packing (`0xfd8213..0xfd823c`).
[[gnu::always_inline]] inline float RandU32ToUnitFloat(std::uint32_t r) noexcept {
    // In x86-64 `mov edx, 0xfffffe01` zero-extends to 64-bit `0x00000000fffffe01ULL`,
    // computing exact unsigned quotient `q = r / 0x00800001u` via `>> 55`.
    const std::uint64_t uprod =
        static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(0x00000000fffffe01ULL);
    const std::uint32_t q = static_cast<std::uint32_t>(uprod >> 55);
    const std::uint32_t rem_sub = static_cast<std::uint32_t>(0u - q * 0x00800001u);
    const std::uint32_t bits = r + rem_sub + 0x3f800000u;
    return std::bit_cast<float>(bits) + (-1.0f);
}

} // namespace

std::atomic<std::uint64_t> g_cloth_exp_dispatch_counter{0};

std::uint64_t ReadDispatchCounterInternal() noexcept {
    return g_cloth_exp_dispatch_counter.load(std::memory_order_relaxed);
}

void ResetDispatchCounterInternal() noexcept {
    g_cloth_exp_dispatch_counter.store(0, std::memory_order_relaxed);
}

extern "C" {

// ============================================================================
// 1. hclSimClothInstance::getActiveConfig @ 0x0748c50 (21 B)
// ============================================================================
u8* PS4_SYSV_ABI ClothExp_hclSimClothInstance_getActiveConfig(u8* sim) noexcept {
    u8* override_cfg = *reinterpret_cast<u8* const*>(sim + 0x1d8);
    if (override_cfg != nullptr) {
        return override_cfg;
    }
    return *reinterpret_cast<u8* const*>(sim + 0x10) + 0x10;
}

// ============================================================================
// 2. hclSimClothInstance::updateParticlesAabb @ 0x0748a00 (161 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclSimClothInstance_updateParticlesAabb(u8* sim) noexcept {
    MonitorPushTimer(0x47b2060ULL);
    __m128 aabb_min = HkMaxFloatVec();
    __m128 aabb_max = HkMinFloatVec();
    const std::int32_t n = *reinterpret_cast<const std::int32_t*>(sim + 0x20);
    if (n > 0) {
        const __m128* pos = *reinterpret_cast<const __m128* const*>(sim + 0x18);
        for (std::int32_t i = 0; i < n; ++i) {
            const __m128 p = _mm_load_ps(reinterpret_cast<const float*>(pos + i));
            aabb_min = _mm_min_ps(aabb_min, p);
            aabb_max = _mm_max_ps(aabb_max, p);
        }
    }
    _mm_store_ps(reinterpret_cast<float*>(sim + 0x50), aabb_min);
    _mm_store_ps(reinterpret_cast<float*>(sim + 0x60), aabb_max);
    MonitorPushTimer(0x47b2087ULL);
}

// ============================================================================
// 3. hclSimClothInstance::updateAabbs @ 0x0748ab0 (410 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclSimClothInstance_updateAabbs(u8* sim) noexcept {
    MonitorPushTimer(0x47b2078ULL);
    __m128 pinch_min = HkMaxFloatVec();
    __m128 pinch_max = HkMinFloatVec();
    __m128 col_min = pinch_min;
    __m128 col_max = pinch_max;
    __m128 part_min = pinch_min;
    __m128 part_max = pinch_max;

    const std::int32_t n = *reinterpret_cast<const std::int32_t*>(sim + 0x20);
    u8* sim_data = *reinterpret_cast<u8* const*>(sim + 0x10);
    if (n > 0) {
        const u8* pdata_bytes = *reinterpret_cast<const u8* const*>(sim_data + 0x38) + 0x04;
        const std::int32_t* col_masks =
            *reinterpret_cast<const std::int32_t* const*>(sim_data + 0xf8);
        const __m128* cur = *reinterpret_cast<const __m128* const*>(sim + 0x18);
        const __m128* prev = *reinterpret_cast<const __m128* const*>(sim + 0x28);
        const __m128 half_v = _mm_set1_ps(0.5f);
        const __m128 one_half_v = _mm_set1_ps(1.5f);

        for (std::int32_t i = 0; i < n; ++i) {
            const __m128 p = _mm_load_ps(reinterpret_cast<const float*>(cur + i));
            part_min = _mm_min_ps(part_min, p);
            part_max = _mm_max_ps(part_max, p);

            const std::int32_t mask = col_masks[i];
            const float inv_mass =
                *reinterpret_cast<const float*>(pdata_bytes + std::size_t(i) * 0x10u);
            if (mask != 0 || inv_mass != 0.0f) {
                const __m128 q = _mm_load_ps(reinterpret_cast<const float*>(prev + i));
                const __m128 d = _mm_sub_ps(p, q);
                const __m128 p05 = _mm_add_ps(q, _mm_mul_ps(d, half_v));
                const __m128 p15 = _mm_add_ps(q, _mm_mul_ps(d, one_half_v));
                col_min = _mm_min_ps(_mm_min_ps(col_min, p05), p15);
                col_max = _mm_max_ps(_mm_max_ps(col_max, p05), p15);
                if (mask < 0) {
                    pinch_min = _mm_min_ps(_mm_min_ps(pinch_min, p05), p15);
                    pinch_max = _mm_max_ps(_mm_max_ps(pinch_max, p05), p15);
                }
            }
        }
    }

    const u8* cfg = *reinterpret_cast<const u8* const*>(sim + 0x1d8);
    if (!cfg) {
        cfg = sim_data + 0x10;
    }
    const __m128 rad = _mm_set1_ps(*reinterpret_cast<const float*>(cfg + 0x14));
    col_max = _mm_add_ps(col_max, rad);
    col_min = _mm_sub_ps(col_min, rad);
    const __m128 pinch_max_out = _mm_add_ps(pinch_max, rad);
    pinch_min = _mm_sub_ps(pinch_min, rad);

    _mm_store_ps(reinterpret_cast<float*>(sim + 0x50), part_min);
    _mm_store_ps(reinterpret_cast<float*>(sim + 0x60), part_max);
    _mm_store_ps(reinterpret_cast<float*>(sim + 0x90), pinch_min);
    _mm_store_ps(reinterpret_cast<float*>(sim + 0xa0), pinch_max_out);
    _mm_store_ps(reinterpret_cast<float*>(sim + 0x70), col_min);
    _mm_store_ps(reinterpret_cast<float*>(sim + 0x80), col_max);
    MonitorPushTimer(0x47b2087ULL);
}

// ============================================================================
// 4. hclConstraintStiffnessFactor @ 0x1167a80 (166 B)
// ============================================================================
float PS4_SYSV_ABI ClothExp_hclConstraintStiffnessFactor(const u8* cset, int mode, int iter,
                                                         int substeps, float factor0,
                                                         float factor1) noexcept {
    if (mode != 2) {
        return 1.0f;
    }
    if (substeps == 1 && factor0 == 1.0f && factor1 == 1.0f) {
        return 1.0f;
    }
    const float f_sub = static_cast<float>(substeps);
    constexpr float kClothExp = -1.725000023841858f; // 0x48d7bbc
    const float res = std::pow(f_sub * factor0 * factor1, kClothExp);
    const std::uint32_t type = *reinterpret_cast<const std::uint32_t*>(cset + 0x18);
    if (type == 8u) {
        return static_cast<float>(iter + 1) / f_sub;
    }
    if (type == 5u || type == 10u) {
        return (substeps - 1 == iter) ? 1.0f : 0.0f;
    }
    return res;
}

// ============================================================================
// 5-7. hclMoveParticlesOperator @ 0x06fffb0 / 0x06fffc0 / 0x0700180
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclMoveParticlesOperator_moveSomeSIMD(const u8* op, u8* sim,
                                                                 const u8* ref_buf) noexcept {
    MonitorPushTimer(0x47ad017ULL);
    const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
    const bool has_fixed = (*reinterpret_cast<const std::int32_t*>(sim_data + 0x50) != 0);
    const std::int32_t count = *reinterpret_cast<const std::int32_t*>(op + 0x28);

    if (count > 0) {
        const __m128 col0 = _mm_load_ps(reinterpret_cast<const float*>(ref_buf + 0x80));
        const __m128 col1 = _mm_load_ps(reinterpret_cast<const float*>(ref_buf + 0x90));
        const __m128 col2 = _mm_load_ps(reinterpret_cast<const float*>(ref_buf + 0xa0));
        const __m128 col3 = _mm_load_ps(reinterpret_cast<const float*>(ref_buf + 0xb0));

        if (has_fixed) {
            for (std::int32_t i = 0; i < count; ++i) {
                const std::uint16_t* pairs = *reinterpret_cast<const std::uint16_t* const*>(op + 0x20);
                const u8* vtx_base = *reinterpret_cast<const u8* const*>(ref_buf + 0x10);
                const std::size_t vtx_idx = pairs[std::size_t(i) * 2u + 0u];
                const std::size_t stride = *reinterpret_cast<const u8*>(ref_buf + 0x1c);
                const __m128 v = _mm_castsi128_ps(
                    _mm_load_si128(reinterpret_cast<const __m128i*>(vtx_base + vtx_idx * stride)));
                const __m128 vx = _mm_shuffle_ps(v, v, 0x00);
                const __m128 vy = _mm_shuffle_ps(v, v, 0x55);
                const __m128 vz = _mm_shuffle_ps(v, v, 0xaa);
                const __m128 wp = _mm_add_ps(
                    _mm_add_ps(_mm_add_ps(_mm_mul_ps(vx, col0), col3), _mm_mul_ps(vy, col1)),
                    _mm_mul_ps(vz, col2));

                const std::size_t part_idx = pairs[std::size_t(i) * 2u + 1u];
                __m128* cur = *reinterpret_cast<__m128* const*>(sim + 0x18);
                __m128* prev = *reinterpret_cast<__m128* const*>(sim + 0x28);
                prev[part_idx] = cur[part_idx];
                cur[part_idx] = wp;
            }
        } else {
            for (std::int32_t i = 0; i < count; ++i) {
                const std::uint16_t* pairs = *reinterpret_cast<const std::uint16_t* const*>(op + 0x20);
                const u8* vtx_base = *reinterpret_cast<const u8* const*>(ref_buf + 0x10);
                const std::size_t vtx_idx = pairs[std::size_t(i) * 2u + 0u];
                const std::size_t stride = *reinterpret_cast<const u8*>(ref_buf + 0x1c);
                const __m128 v = _mm_castsi128_ps(
                    _mm_load_si128(reinterpret_cast<const __m128i*>(vtx_base + vtx_idx * stride)));
                const __m128 vx = _mm_shuffle_ps(v, v, 0x00);
                const __m128 vy = _mm_shuffle_ps(v, v, 0x55);
                const __m128 vz = _mm_shuffle_ps(v, v, 0xaa);
                const __m128 wp = _mm_add_ps(
                    _mm_add_ps(_mm_add_ps(_mm_mul_ps(vx, col0), col3), _mm_mul_ps(vy, col1)),
                    _mm_mul_ps(vz, col2));

                const std::size_t part_idx = pairs[std::size_t(i) * 2u + 1u];
                __m128* prev = *reinterpret_cast<__m128* const*>(sim + 0x28);
                __m128* cur = *reinterpret_cast<__m128* const*>(sim + 0x18);
                prev[part_idx] = wp;
                cur[part_idx] = wp;
            }
        }
    }
    MonitorPushTimer(0x47ad014ULL);
}

void PS4_SYSV_ABI ClothExp_hclMoveParticlesOperator_moveSomeScalar(const u8* op, u8* sim,
                                                                   const u8* ref_buf) noexcept {
    MonitorPushTimer(0x47acffeULL);
    const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
    const bool has_fixed = (*reinterpret_cast<const std::int32_t*>(sim_data + 0x50) != 0);
    const std::int32_t count = *reinterpret_cast<const std::int32_t*>(op + 0x28);

    if (count > 0) {
        const __m128 col0 = _mm_load_ps(reinterpret_cast<const float*>(ref_buf + 0x80));
        const __m128 col1 = _mm_load_ps(reinterpret_cast<const float*>(ref_buf + 0x90));
        const __m128 col2 = _mm_load_ps(reinterpret_cast<const float*>(ref_buf + 0xa0));
        const __m128 col3 = _mm_load_ps(reinterpret_cast<const float*>(ref_buf + 0xb0));

        if (has_fixed) {
            for (std::int32_t i = 0; i < count; ++i) {
                const std::uint16_t* pairs = *reinterpret_cast<const std::uint16_t* const*>(op + 0x20);
                const u8* vtx_base = *reinterpret_cast<const u8* const*>(ref_buf + 0x10);
                const std::size_t vtx_idx = pairs[std::size_t(i) * 2u + 0u];
                const std::size_t stride = *reinterpret_cast<const u8*>(ref_buf + 0x1c);
                const __m128 v = _mm_castsi128_ps(
                    _mm_lddqu_si128(reinterpret_cast<const __m128i*>(vtx_base + vtx_idx * stride)));
                const __m128 vx = _mm_shuffle_ps(v, v, 0x00);
                const __m128 vy = _mm_shuffle_ps(v, v, 0x55);
                const __m128 vz = _mm_shuffle_ps(v, v, 0xaa);
                const __m128 wp = _mm_add_ps(
                    _mm_add_ps(_mm_add_ps(col3, _mm_mul_ps(vx, col0)), _mm_mul_ps(vy, col1)),
                    _mm_mul_ps(vz, col2));

                const std::size_t part_idx = pairs[std::size_t(i) * 2u + 1u];
                __m128* cur = *reinterpret_cast<__m128* const*>(sim + 0x18);
                __m128* prev = *reinterpret_cast<__m128* const*>(sim + 0x28);
                prev[part_idx] = cur[part_idx];
                cur[part_idx] = wp;
            }
        } else {
            for (std::int32_t i = 0; i < count; ++i) {
                const std::uint16_t* pairs = *reinterpret_cast<const std::uint16_t* const*>(op + 0x20);
                const u8* vtx_base = *reinterpret_cast<const u8* const*>(ref_buf + 0x10);
                const std::size_t vtx_idx = pairs[std::size_t(i) * 2u + 0u];
                const std::size_t stride = *reinterpret_cast<const u8*>(ref_buf + 0x1c);
                const __m128 v = _mm_castsi128_ps(
                    _mm_lddqu_si128(reinterpret_cast<const __m128i*>(vtx_base + vtx_idx * stride)));
                const __m128 vx = _mm_shuffle_ps(v, v, 0x00);
                const __m128 vy = _mm_shuffle_ps(v, v, 0x55);
                const __m128 vz = _mm_shuffle_ps(v, v, 0xaa);
                const __m128 wp = _mm_add_ps(
                    _mm_add_ps(_mm_add_ps(_mm_mul_ps(vx, col0), col3), _mm_mul_ps(vy, col1)),
                    _mm_mul_ps(vz, col2));

                const std::size_t part_idx = pairs[std::size_t(i) * 2u + 1u];
                __m128* cur = *reinterpret_cast<__m128* const*>(sim + 0x18);
                __m128* prev = *reinterpret_cast<__m128* const*>(sim + 0x28);
                cur[part_idx] = wp;
                prev[part_idx] = wp;
            }
        }
    }
    MonitorPushTimer(0x47ad014ULL);
}

void PS4_SYSV_ABI ClothExp_hclMoveParticlesOperator_process(const u8* op, u8* sim,
                                                            const u8* ref_buf) noexcept {
    if ((ref_buf[0x20] & 0x01u) != 0u) {
        ResolveFn<decltype(&ClothExp_hclMoveParticlesOperator_moveSomeSIMD)>(0x06fffc0ULL)(
            op, sim, ref_buf);
    } else {
        ResolveFn<decltype(&ClothExp_hclMoveParticlesOperator_moveSomeScalar)>(0x0700180ULL)(
            op, sim, ref_buf);
    }
}

// ============================================================================
// 8-9. hclWorld::step (0x0769450) & hclWorld::addClothJobs (0x07694d0)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclWorld_addClothJobs(u8* world, u8* job_queue, int num_jobs,
                                                 u8* scratch, float dt) noexcept {
    MonitorPushTimer(0x47b2c51ULL); // TtInitialize Step
    using InitStepFn = void(PS4_SYSV_ABI*)(u8*, u8*, int);
    ResolveFn<InitStepFn>(0x07687b0ULL)(world, scratch, num_jobs);

    MonitorPushTimer(0x47b2c62ULL); // TtPre Simulation Callbacks
    const std::int32_t n_listeners = *reinterpret_cast<const std::int32_t*>(world + 0x48);
    if (n_listeners > 0) {
        using PreSimCb = void(PS4_SYSV_ABI*)(u8*, u8*, float);
        for (std::int32_t i = 0; i < n_listeners; ++i) {
            u8* const* listeners = *reinterpret_cast<u8* const* const*>(world + 0x40);
            u8* lst = listeners[i];
            const auto* vtbl = *reinterpret_cast<void* const* const*>(lst);
            reinterpret_cast<PreSimCb>(vtbl[0x30 / sizeof(void*)])(lst, world, dt);
        }
    }

    MonitorPushTimer(0x47b2c4eULL); // Et
    MonitorPushTimer(0x47b2c7dULL); // TtAdd Cloth Jobs

    const std::int32_t n_insts = *reinterpret_cast<const std::int32_t*>(world + 0x18);
    if (n_insts > 0) {
        struct alignas(16) HclClothJob {
            u8 m_jobSubType;         // +0x00 (= 0)
            u8 m_jobType;            // +0x01 (= 8: HK_JOB_TYPE_CLOTH)
            u8 m_code;               // +0x02 (= 2)
            u8 m_pad3;               // +0x03
            std::uint16_t m_size;    // +0x04 (= 0x30)
            std::uint16_t m_thread;  // +0x06 (= 0xffff)
            float m_timeStep;        // +0x08
            std::uint32_t m_padC;    // +0x0c
            u8* m_world;             // +0x10
            u8* m_clothInstance;     // +0x18
            u8* m_scratch;           // +0x20 (= nullptr)
            u8 m_doSimulations;      // +0x28 (= world[0x81])
            u8 m_pad29[3];           // +0x29..0x2b
            std::uint32_t m_tailZero;// +0x2c (= 0)
        };
        static_assert(sizeof(HclClothJob) == 0x30, "HclClothJob must be 48 bytes");

        using AddJobFn = void(PS4_SYSV_ABI*)(u8*, HclClothJob*, int);
        const auto add_job = ResolveFn<AddJobFn>(0x04210c0ULL);

        for (std::int32_t i = 0; i < n_insts; ++i) {
            u8* const* insts = *reinterpret_cast<u8* const* const*>(world + 0x10);
            u8* inst = insts[i];
            const u8* cloth_data = *reinterpret_cast<const u8* const*>(inst + 0x10);
            const std::int32_t state_idx = *reinterpret_cast<const std::int32_t*>(inst + 0x18);
            const u8* const* states = *reinterpret_cast<const u8* const* const*>(cloth_data + 0x58);
            const u8* state = states[state_idx];
            if (*reinterpret_cast<const std::int32_t*>(state + 0x20) == 0) {
                *reinterpret_cast<std::int32_t*>(inst + 0x1c) = 0;
                continue;
            }
            HclClothJob job{};
            job.m_jobSubType = 0u;
            job.m_jobType = 8u;
            job.m_code = 2u;
            job.m_size = 0x30u;
            job.m_thread = 0xffffu;
            job.m_timeStep = dt;
            job.m_world = world;
            job.m_clothInstance = inst;
            job.m_scratch = nullptr;
            job.m_doSimulations = world[0x81];
            job.m_tailZero = 0u;
            add_job(job_queue, &job, 1);
        }
    }

    MonitorPushTimer(0x47b2c4eULL); // Et
    MonitorPushTimer(0x47b2c4eULL); // Et
}

void PS4_SYSV_ABI ClothExp_hclWorld_step(u8* world, u8* job_queue, int num_jobs, u8* scratch,
                                         float dt) noexcept {
    const std::int32_t n_insts = *reinterpret_cast<const std::int32_t*>(world + 0x18);
    if (n_insts > 0) {
        for (std::int32_t i = 0; i < n_insts; ++i) {
            u8* const* insts = *reinterpret_cast<u8* const* const*>(world + 0x10);
            u8* inst = insts[i];
            const u8* cloth_data = *reinterpret_cast<const u8* const*>(inst + 0x10);
            const std::int32_t state_idx = *reinterpret_cast<const std::int32_t*>(inst + 0x18);
            const u8* const* states = *reinterpret_cast<const u8* const* const*>(cloth_data + 0x58);
            const u8* state = states[state_idx];
            std::int32_t n_ops = *reinterpret_cast<const std::int32_t*>(state + 0x20);
            for (std::int32_t k = 0; k < n_ops; ++k) {
                const std::int32_t* op_idxs =
                    *reinterpret_cast<const std::int32_t* const*>(state + 0x18);
                u8* const* all_ops = *reinterpret_cast<u8* const* const*>(cloth_data + 0x48);
                u8* op = all_ops[op_idxs[k]];
                if (*reinterpret_cast<const std::int32_t*>(op + 0x18) == 1) {
                    op[0x38] = 0u;
                    n_ops = *reinterpret_cast<const std::int32_t*>(state + 0x20);
                }
            }
        }
    }
    ResolveFn<decltype(&ClothExp_hclWorld_addClothJobs)>(0x07694d0ULL)(
        world, job_queue, num_jobs, scratch, dt);
}

// ============================================================================
// 10. hclWorld::stepSingleThread @ 0x0769190 (701 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclWorld_stepSingleThread(u8* world, u8* scratch, float dt) noexcept {
    using InitStepFn = void(PS4_SYSV_ABI*)(u8*, u8*, int);
    using AllocScratchFn = u8*(PS4_SYSV_ABI*)(u8*);
    using PrepareFn = void(PS4_SYSV_ABI*)(u8*, void*);
    using ReleaseFn = void(PS4_SYSV_ABI*)(u8*, void*);
    using PrePostSimCb = void(PS4_SYSV_ABI*)(u8*, u8*, float);

    ResolveFn<InitStepFn>(0x07687b0ULL)(world, scratch, 1);

    std::int32_t n_listeners = *reinterpret_cast<const std::int32_t*>(world + 0x48);
    for (std::int32_t i = 0; i < n_listeners; ++i) {
        u8* const* listeners = *reinterpret_cast<u8* const* const*>(world + 0x40);
        u8* lst = listeners[i];
        const auto* vtbl = *reinterpret_cast<void* const* const*>(lst);
        reinterpret_cast<PrePostSimCb>(vtbl[0x30 / sizeof(void*)])(lst, world, dt);
        n_listeners = *reinterpret_cast<const std::int32_t*>(world + 0x48);
    }

    const std::int32_t n_insts = *reinterpret_cast<const std::int32_t*>(world + 0x18);
    if (n_insts > 0) {
        struct alignas(8) HclStepInfo {
            float m_timeStep;        // +0x00 ([rbp-0x58])
            std::uint32_t m_pad4;    // +0x04
            u8* m_world;             // +0x08 ([rbp-0x50])
            u8* m_clothInstance;     // +0x10 ([rbp-0x48])
            u8* m_scratchBuffer;     // +0x18 ([rbp-0x40])
            u8 m_doSimulations;      // +0x20 ([rbp-0x38])
            u8 m_pad21[3];           // +0x21
            std::uint32_t m_state0;  // +0x24 ([rbp-0x34] = 0)
        };
        static_assert(sizeof(HclStepInfo) == 0x28, "HclStepInfo must be 40 bytes");

        for (std::int32_t idx = 0; idx < n_insts; ++idx) {
            u8* const* insts = *reinterpret_cast<u8* const* const*>(world + 0x10);
            u8* inst = insts[idx];
            const u8* cloth_data = *reinterpret_cast<const u8* const*>(inst + 0x10);
            const std::int32_t state_idx = *reinterpret_cast<const std::int32_t*>(inst + 0x18);
            const u8* const* states = *reinterpret_cast<const u8* const* const*>(cloth_data + 0x58);
            const u8* state = states[state_idx];
            if (*reinterpret_cast<const std::int32_t*>(state + 0x20) == 0) {
                continue;
            }

            HclStepInfo step_info{};
            step_info.m_timeStep = dt;
            step_info.m_world = world;
            step_info.m_clothInstance = inst;
            step_info.m_scratchBuffer = ResolveFn<AllocScratchFn>(0x0768940ULL)(world);
            step_info.m_doSimulations = world[0x81];
            step_info.m_state0 = 0u;

            const u8* cd_reload = *reinterpret_cast<const u8* const*>(inst + 0x10);
            const std::int32_t st_reload = *reinterpret_cast<const std::int32_t*>(inst + 0x18);
            const u8* cur_state =
                (*reinterpret_cast<const u8* const* const*>(cd_reload + 0x58))[st_reload];

            const std::uint64_t name_ptr =
                *reinterpret_cast<const std::uint64_t*>(cd_reload + 0x10) & ~std::uint64_t(1u);
            MonitorPushObjectTimer(0x47b2c37ULL, name_ptr);

            ResolveFn<PrepareFn>(0x075ac80ULL)(inst, &step_info);
            MonitorPushTimer(0x47b2c3aULL); // TtExecute Operators

            std::int32_t n_ops = *reinterpret_cast<const std::int32_t*>(cur_state + 0x20);
            if (n_ops > 0) {
                for (std::int32_t k = 0; k < n_ops; ++k) {
                    const std::int32_t* op_idxs =
                        *reinterpret_cast<const std::int32_t* const*>(cur_state + 0x18);
                    u8* const* all_ops = *reinterpret_cast<u8* const* const*>(cd_reload + 0x48);
                    u8* op = all_ops[op_idxs[k]];
                    if (*reinterpret_cast<const std::int32_t*>(op + 0x18) == 1) {
                        op[0x38] = 0u;
                        n_ops = *reinterpret_cast<const std::int32_t*>(cur_state + 0x20);
                    }
                }
                for (std::int32_t k = 0; k < *reinterpret_cast<const std::int32_t*>(cur_state + 0x20);
                     ++k) {
                    const std::int32_t* op_idxs =
                        *reinterpret_cast<const std::int32_t* const*>(cur_state + 0x18);
                    u8* const* all_ops = *reinterpret_cast<u8* const* const*>(cd_reload + 0x48);
                    u8* op = all_ops[op_idxs[k]];
                    ResolveFn<decltype(&ClothExp_hclOperator_execute)>(0x076a960ULL)(
                        op, reinterpret_cast<u8*>(&step_info));
                }
            }

            MonitorPushTimer(0x47b2c4eULL); // Et
            ResolveFn<ReleaseFn>(0x075b1a0ULL)(inst, &step_info);
            MonitorPushTimer(0x47b2c4eULL); // Et
        }
    }

    const std::int32_t n_post = *reinterpret_cast<const std::int32_t*>(world + 0x48);
    for (std::int32_t i = 0; i < n_post; ++i) {
        u8* const* listeners = *reinterpret_cast<u8* const* const*>(world + 0x40);
        u8* lst = listeners[i];
        const auto* vtbl = *reinterpret_cast<void* const* const*>(lst);
        reinterpret_cast<PrePostSimCb>(vtbl[0x38 / sizeof(void*)])(lst, world, dt);
    }
}

// ============================================================================
// 11. hclOperator::execute @ 0x076a960 (1436 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_hclOperator_execute(u8* op, u8* step_info) noexcept {
    g_cloth_exp_dispatch_counter.fetch_add(1u, std::memory_order_relaxed);

    const std::uint32_t op_type = *reinterpret_cast<const std::uint32_t*>(op + 0x18);
    u8* ctx = *reinterpret_cast<u8* const*>(step_info + 0x10);
    u8* const* bufs = ctx ? *reinterpret_cast<u8* const* const*>(ctx + 0x30) : nullptr;
    u8* const* tsets = ctx ? *reinterpret_cast<u8* const* const*>(ctx + 0x40) : nullptr;

    using Op2BufFn = void(PS4_SYSV_ABI*)(u8*, u8*, u8*);
    using Op3BufFn = void(PS4_SYSV_ABI*)(u8*, u8*, u8*, u8*);
    using Op1BufFn = void(PS4_SYSV_ABI*)(u8*, u8*);
    using OpBoneDeformFn = void(PS4_SYSV_ABI*)(u8*, u8*, u8*, u8*, u8*);
    using SimulateFn = int(PS4_SYSV_ABI*)(u8*, u8*, float);

    switch (op_type) {
    case 1u: { // hclSimulateOperator::simulate (honors 0x076ab3c call target when FIXED30 is active)
        const float dt = *reinterpret_cast<const float*>(step_info + 0x00);
        const u8* call_site = reinterpret_cast<const u8*>(ResolveVma(0x076ab3cULL));
        if (call_site[0] == 0xe8u) {
            std::int32_t rel32 = 0;
            std::memcpy(&rel32, call_site + 1u, 4u);
            const auto target =
                ResolveVma(0x076ab41ULL) + static_cast<std::intptr_t>(rel32);
            (void)reinterpret_cast<SimulateFn>(target)(op, ctx, dt);
        } else {
            (void)ResolveFn<SimulateFn>(0x076e370ULL)(op, ctx, dt);
        }
        return;
    }
    case 2u: { // hclGatherAllVerticesOperator
        u8* b0 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x30));
        u8* b1 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x34));
        ResolveFn<Op2BufFn>(0x07370b0ULL)(op, b0, b1);
        return;
    }
    case 3u: { // hclGatherSomeVerticesOperator
        u8* b0 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x30));
        u8* b1 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x34));
        ResolveFn<Op2BufFn>(0x0742fb0ULL)(op, b0, b1);
        return;
    }
    case 4u: { // hclCopyVerticesOperator
        u8* b0 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x1c));
        u8* b1 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x20));
        ResolveFn<Op2BufFn>(0x0742470ULL)(op, b0, b1);
        return;
    }
    case 5u: { // hclSimpleMeshMeshDeformOperator
        u8* b0 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x50));
        u8* b1 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x54));
        ResolveFn<Op2BufFn>(0x076b150ULL)(op, b0, b1);
        return;
    }
    case 8u: { // hclSimpleMeshBoneDeformOperator (Skin_outer)
        u8* b_in = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x64));
        u8* b_out = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x68));
        u8* ts = tsets[*reinterpret_cast<const std::int32_t*>(op + 0x6c)];
        u8* world = *reinterpret_cast<u8* const*>(step_info + 0x08);
        if (op[0x75] != 0u) {
            ResolveFn<OpBoneDeformFn>(0x075d440ULL)(op, b_in, ts, b_out, world);
        } else {
            ResolveFn<OpBoneDeformFn>(0x075ce50ULL)(op, b_in, ts, b_out, world);
        }
        return;
    }
    case 10u: { // hclBlendSomeVerticesOperator
        u8* b0 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x30));
        u8* b1 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x34));
        u8* b2 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x38));
        ResolveFn<Op3BufFn>(0x07413a0ULL)(op, b0, b1, b2);
        return;
    }
    case 11u: { // hclMoveParticlesOperator
        u8* const* sims = *reinterpret_cast<u8* const* const*>(ctx + 0x20);
        u8* ref_buf = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x34));
        u8* sim = sims[*reinterpret_cast<const std::int32_t*>(op + 0x30)];
        ClothExp_hclMoveParticlesOperator_process(op, sim, ref_buf);
        return;
    }
    case 12u: {
        u8* b0 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x80));
        ResolveFn<Op1BufFn>(0x0748f40ULL)(op, b0);
        return;
    }
    case 13u: {
        u8* b0 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0xb0));
        ResolveFn<Op1BufFn>(0x0738030ULL)(op, b0);
        return;
    }
    case 14u: { // hclInputConvertOperator (direct buffer indices, no +0x100 shadow)
        u8* b0 = bufs[*reinterpret_cast<const std::int32_t*>(op + 0x1c)];
        u8* b1 = bufs[*reinterpret_cast<const std::int32_t*>(op + 0x20)];
        ResolveFn<Op2BufFn>(0x0710100ULL)(op, b0, b1);
        return;
    }
    case 15u: { // hclOutputConvertOperator (swapped b1, b0 order at 0x76ad3e!)
        u8* b_rdx = bufs[*reinterpret_cast<const std::int32_t*>(op + 0x1c)];
        u8* b_rsi = bufs[*reinterpret_cast<const std::int32_t*>(op + 0x20)];
        ResolveFn<Op2BufFn>(0x07164b0ULL)(op, b_rsi, b_rdx);
        return;
    }
    case 16u: {
        u8* b0 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x1c));
        u8* ts = tsets[*reinterpret_cast<const std::int32_t*>(op + 0x20)];
        ResolveFn<Op2BufFn>(0x076ccd0ULL)(op, b0, ts);
        return;
    }
    case 17u: {
        u8* b0 = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x1c));
        u8* ts = tsets[*reinterpret_cast<const std::int32_t*>(op + 0x20)];
        ResolveFn<Op2BufFn>(0x076bd60ULL)(op, b0, ts);
        return;
    }
    case 18u:
    case 19u:
    case 20u:
    case 21u: { // Bone-Space Skin (P / PN / PNT / PNTB) — note: rsi=tset, rdx=shadowBuf!
        u8* b_rdx = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x30));
        u8* ts_rsi = tsets[*reinterpret_cast<const std::int32_t*>(op + 0x34)];
        constexpr std::uintptr_t kTargets[4] = {
            0x07579c0ULL, 0x0700c60ULL, 0x07210e0ULL, 0x0749440ULL};
        ResolveFn<Op2BufFn>(kTargets[op_type - 18u])(op, ts_rsi, b_rdx);
        return;
    }
    case 22u:
    case 23u:
    case 24u:
    case 25u: { // Object-Space Skin (P / PN / PNT / PNTB) — op +0x40/+0x44, rsi=tset, rdx=shadowBuf
        u8* b_rdx = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x40));
        u8* ts_rsi = tsets[*reinterpret_cast<const std::int32_t*>(op + 0x44)];
        constexpr std::uintptr_t kTargets[4] = {
            0x0728830ULL, 0x075e140ULL, 0x07623d0ULL, 0x0739ef0ULL};
        ResolveFn<Op2BufFn>(kTargets[op_type - 22u])(op, ts_rsi, b_rdx);
        return;
    }
    case 26u:
    case 27u:
    case 28u:
    case 29u: { // Bone-Space Mesh-Mesh Deform (P / PN / PNT / PNTB)
        u8* b_rsi = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x1c));
        u8* b_rdx = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x20));
        constexpr std::uintptr_t kTargets[4] = {
            0x0718d70ULL, 0x06fbb10ULL, 0x0711350ULL, 0x0750f30ULL};
        ResolveFn<Op2BufFn>(kTargets[op_type - 26u])(op, b_rsi, b_rdx);
        return;
    }
    case 30u:
    case 31u:
    case 32u:
    case 33u: { // Object-Space Mesh-Mesh Deform (P / PN / PNT / PNTB)
        u8* b_rsi = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x1c));
        u8* b_rdx = ResolveShadowBuffer(bufs, *reinterpret_cast<const std::int32_t*>(op + 0x20));
        constexpr std::uintptr_t kTargets[4] = {
            0x0743cb0ULL, 0x071b890ULL, 0x0731ab0ULL, 0x07071a0ULL};
        ResolveFn<Op2BufFn>(kTargets[op_type - 30u])(op, b_rsi, b_rdx);
        return;
    }
    default: {
        // Exact match for 0x76a9e6..0x76aa38 (hkOstream + hkError::messageAssert 0x20238843)
        alignas(16) u8 ostream_buf[0x18]{};
        alignas(16) char msg_buf[0x200]{};
        using OstreamCtorFn = void(PS4_SYSV_ABI*)(u8*, char*, int);
        using OstreamPutsFn = void(PS4_SYSV_ABI*)(u8*, const char*);
        using MsgAssertFn = int(PS4_SYSV_ABI*)(std::uint32_t, const char*, const char*, int);
        using OstreamDtorFn = void(PS4_SYSV_ABI*)(u8*);
        ResolveFn<OstreamCtorFn>(0x0419350ULL)(ostream_buf, msg_buf, 0x200);
        ResolveFn<OstreamPutsFn>(0x0425170ULL)(
            ostream_buf, reinterpret_cast<const char*>(ResolveVma(0x47b2e9eULL)));
        const int brk = ResolveFn<MsgAssertFn>(0x0419670ULL)(
            0x20238843u, msg_buf, reinterpret_cast<const char*>(ResolveVma(0x47b2eafULL)), 0x15a);
        if (brk != 0) {
            __builtin_trap();
        }
        ResolveFn<OstreamDtorFn>(0x0419460ULL)(ostream_buf);
        return;
    }
    }
}

// ============================================================================
// 12. FD4HavokCloth::step @ 0x0fd5440 (346 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_FD4HavokCloth_step(u8* self, float dt) noexcept {
    if (!(dt > 0.0f)) {
        return;
    }
    *reinterpret_cast<float*>(self + 0x144) = dt;

    u8* sub_mgr = *reinterpret_cast<u8* const*>(self + 0x18);
    if (sub_mgr != nullptr) {
        using SubStepFn = void(PS4_SYSV_ABI*)(u8*, float);
        ResolveFn<SubStepFn>(0x1176450ULL)(sub_mgr, dt);
    }

    using IsIdleFn = bool(PS4_SYSV_ABI*)(u8*);
    using PreSyncFn = void(PS4_SYSV_ABI*)(u8*);
    using CalcBufSizeFn = int(PS4_SYSV_ABI*)(u8*, int);
    using AllocFn = u8*(PS4_SYSV_ABI*)(u8*, int);
    using DeallocFn = void(PS4_SYSV_ABI*)(u8*, u8*);

    u8* world = *reinterpret_cast<u8* const*>(self + 0x10);
    const bool ok = ResolveFn<IsIdleFn>(0x076a020ULL)(world);
    self[0x140] = ok ? 0u : 1u;
    if (!ok) {
        ResolveFn<PreSyncFn>(0x076a030ULL)(*reinterpret_cast<u8* const*>(self + 0x10));
    }

    if (self[0x30] != 0u) {
        self[0x141] = 1u;
        u8* thread_pool = *reinterpret_cast<u8* const*>(self + 0x28);
        const auto* tp_vtbl = *reinterpret_cast<void* const* const*>(thread_pool);
        using GetNumThreadsFn = int(PS4_SYSV_ABI*)(u8*);
        const int n_threads = reinterpret_cast<GetNumThreadsFn>(tp_vtbl[0x48 / sizeof(void*)])(
            thread_pool);
        const int num_jobs = n_threads * 2 + 1;

        const int buf_bytes = ResolveFn<CalcBufSizeFn>(0x0768770ULL)(
            *reinterpret_cast<u8* const*>(self + 0x10), num_jobs);
        u8* scratch = ResolveFn<AllocFn>(0x0429a40ULL)(GetHkTempAllocator(), buf_bytes);
        *reinterpret_cast<u8**>(self + 0x138) = scratch;

        ResolveFn<decltype(&ClothExp_hclWorld_step)>(0x0769450ULL)(
            *reinterpret_cast<u8* const*>(self + 0x10),
            *reinterpret_cast<u8* const*>(self + 0x20),
            num_jobs, scratch, dt);

        u8* job_queue = *reinterpret_cast<u8* const*>(self + 0x20);
        u8* tp = *reinterpret_cast<u8* const*>(self + 0x28);
        const auto* tp_vt2 = *reinterpret_cast<void* const* const*>(tp);
        using ProcessJobQueueFn = void(PS4_SYSV_ABI*)(u8*, u8*, int);
        reinterpret_cast<ProcessJobQueueFn>(tp_vt2[0x28 / sizeof(void*)])(tp, job_queue, 0x0f);
        return;
    }

    const int buf_bytes = ResolveFn<CalcBufSizeFn>(0x0768770ULL)(
        *reinterpret_cast<u8* const*>(self + 0x10), 1);
    u8* scratch = ResolveFn<AllocFn>(0x0429a40ULL)(GetHkTempAllocator(), buf_bytes);
    ResolveFn<decltype(&ClothExp_hclWorld_stepSingleThread)>(0x0769190ULL)(
        *reinterpret_cast<u8* const*>(self + 0x10), scratch, dt);
    ResolveFn<DeallocFn>(0x0429a90ULL)(GetHkTempAllocator(), scratch);
}

// ============================================================================
// 13. FD4HavokCloth::queryWind @ 0x0fd5ab0 (255 B)
// ============================================================================
std::int64_t PS4_SYSV_ABI ClothExp_FD4HavokCloth_queryWind(u8* self, const void* target) noexcept {
    u8* const* begin_grp = *reinterpret_cast<u8* const* const*>(self + 0x50);
    u8* const* end_grp = *reinterpret_cast<u8* const* const*>(self + 0x58);
    const std::size_t n_groups = static_cast<std::size_t>(end_grp - begin_grp);
    if (n_groups == 0u) {
        return -1;
    }

    using QueryWindIdxFn = std::int64_t(PS4_SYSV_ABI*)(u8*, int);
    const auto query_idx = ResolveFn<QueryWindIdxFn>(0x104f7c0ULL);

    for (std::size_t g = 0; g < n_groups; ++g) {
        u8* const* grp_arr = *reinterpret_cast<u8* const* const*>(self + 0x50);
        u8* grp = grp_arr[g];
        u8* const* b0 = *reinterpret_cast<u8* const* const*>(grp + 0x20);
        u8* const* e0 = *reinterpret_cast<u8* const* const*>(grp + 0x28);
        const int count = static_cast<int>(e0 - b0);
        for (int i = 0; i < count; ++i) {
            u8* const* b_cur = *reinterpret_cast<u8* const* const*>(grp + 0x20);
            u8* const* e_cur = *reinterpret_cast<u8* const* const*>(grp + 0x28);
            const int cur_len = static_cast<int>(e_cur - b_cur);
            const u8* holder = nullptr;
            if (i < cur_len) {
                holder = *reinterpret_cast<const u8* const*>(b_cur[i]);
            }
            const std::int64_t wind_id = query_idx(grp, i);
            if (holder != nullptr && wind_id != -1) {
                const std::int32_t n_refs = *reinterpret_cast<const std::int32_t*>(holder + 0x28);
                const void* const* refs =
                    *reinterpret_cast<const void* const* const*>(holder + 0x20);
                for (std::int32_t k = 0; k < n_refs; ++k) {
                    if (refs[k] == target) {
                        return wind_id;
                    }
                }
            }
        }
    }
    return -1;
}

// ============================================================================
// 14. FD4Wind::step @ 0x0fd8120 (560 B)
// ============================================================================
void PS4_SYSV_ABI ClothExp_FD4Wind_step(u8* self, const u8* step_ctx) noexcept {
    auto& p0_min = *reinterpret_cast<float*>(self + 0x190);
    auto& p0_max = *reinterpret_cast<float*>(self + 0x194);
    auto& y0_min = *reinterpret_cast<float*>(self + 0x198);
    auto& y0_max = *reinterpret_cast<float*>(self + 0x19c);
    auto& t0_min = *reinterpret_cast<float*>(self + 0x1a0);
    auto& t0_max = *reinterpret_cast<float*>(self + 0x1a4);
    auto& s0_min = *reinterpret_cast<float*>(self + 0x1a8);
    auto& s0_max = *reinterpret_cast<float*>(self + 0x1ac);

    const float p_lo = std::min(p0_min, p0_max);
    const float p_hi = std::max(p0_min, p0_max);
    const float y_lo = std::min(y0_min, y0_max);
    const float y_hi = std::max(y0_min, y0_max);
    const float t_lo = std::min(t0_min, t0_max);
    const float t_hi = std::max(t0_min, t0_max);
    const float s_lo = std::min(s0_min, s0_max);
    const float s_hi = std::max(s0_min, s0_max);

    p0_min = p_lo;
    p0_max = p_hi;
    y0_min = y_lo;
    y0_max = y_hi;
    t0_min = t_lo;
    t0_max = t_hi;
    s0_min = s_lo;
    s0_max = s_hi;

    using NextRandU32Fn = std::uint32_t(PS4_SYSV_ABI*)(u8*);
    const auto next_rand = ResolveFn<NextRandU32Fn>(0x29add90ULL);
    u8* rng = self + 0x1f4;

    float& timer = *reinterpret_cast<float*>(self + 0x1c0);
    float& period = *reinterpret_cast<float*>(self + 0x1bc);
    const float dt = *reinterpret_cast<const float*>(step_ctx + 0x08);
    timer += dt;

    float use_p_lo = p_lo;
    float use_p_hi = p_hi;
    if (timer > period) {
        timer = 0.0f;
        const float u_per = RandU32ToUnitFloat(next_rand(rng));
        period = t_lo + (t_hi - t_lo) * u_per;
        use_p_lo = p0_min;
        use_p_hi = p0_max;
    }

    const float u_pitch = RandU32ToUnitFloat(next_rand(rng));
    *reinterpret_cast<float*>(self + 0x1b4) = use_p_lo + (use_p_hi - use_p_lo) * u_pitch;

    const float cur_y_lo = y0_min;
    const float cur_y_hi = y0_max;
    const float u_yaw = RandU32ToUnitFloat(next_rand(rng));
    *reinterpret_cast<float*>(self + 0x1b8) = cur_y_lo + (cur_y_hi - cur_y_lo) * u_yaw;

    using UpdateBasisFn = void(PS4_SYSV_ABI*)(u8*);
    ResolveFn<UpdateBasisFn>(0x0fd7c80ULL)(self);
}

} // extern "C"

} // namespace BbClothExp
