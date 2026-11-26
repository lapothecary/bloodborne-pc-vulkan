// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// File: gpu/cloth_exp/cloth_exp_local_fp64.h
//
// Mode 1.1: Canonical Havok Cloth Upgraded to Character-Local FP64 Coordinates
//           + Explicit FP64 Velocity + Inelastic Cage/Tether Projection
//           (True Native Any-FPS: 30..500+ FPS & Multi-Frame FG 2x/3x/4x).
// ============================================================================
#pragma once

#include <cmath>
#include <cstdint>
#include <immintrin.h>

#include "common/types.h"

namespace BbClothExp {

/// Pure-math inelastic velocity projection helper (unit-tested offline in `camera-motion-test`):
/// Given `pre_pos` (particle position before a hard one-sided constraint such as `LocalRange`
/// or `StretchLink` LRA tether), `post_pos` (clamped position), and `prev_pos` (`post_pos - v*sdt`
/// reference), shifts `prev_pos` along the clamp normal `u = (post - pre) / |post - pre|` whenever
/// `dot(post - prev, u) > max(0, dot(pre - prev, u))`.
/// Eliminates the `dx / sdt` (`7..22 m/s` at 500 FPS) phantom Verlet velocity explosion from
/// conflicting 2015 `.hkx` hard constraints while preserving 100% of tangential pendulum motion.
inline void ApplyInelasticConstraintProjection3D(
    const float pre_pos[3], const float post_pos[3], float prev_pos[3]) noexcept {
    const double dx = static_cast<double>(post_pos[0]) - static_cast<double>(pre_pos[0]);
    const double dy = static_cast<double>(post_pos[1]) - static_cast<double>(pre_pos[1]);
    const double dz = static_cast<double>(post_pos[2]) - static_cast<double>(pre_pos[2]);
    const double d2 = dx * dx + dy * dy + dz * dz;
    if (!(d2 > 1e-16)) {
        return;
    }
    const double inv_d = 1.0 / std::sqrt(d2);
    const double ux = dx * inv_d;
    const double uy = dy * inv_d;
    const double uz = dz * inv_d;

    const double pre_disp_n =
        (static_cast<double>(pre_pos[0]) - static_cast<double>(prev_pos[0])) * ux +
        (static_cast<double>(pre_pos[1]) - static_cast<double>(prev_pos[1])) * uy +
        (static_cast<double>(pre_pos[2]) - static_cast<double>(prev_pos[2])) * uz;
    const double post_disp_n =
        (static_cast<double>(post_pos[0]) - static_cast<double>(prev_pos[0])) * ux +
        (static_cast<double>(post_pos[1]) - static_cast<double>(prev_pos[1])) * uy +
        (static_cast<double>(post_pos[2]) - static_cast<double>(prev_pos[2])) * uz;

    const double allowed_n = (pre_disp_n > 0.0) ? pre_disp_n : 0.0;
    const double excess_n = post_disp_n - allowed_n;
    if (excess_n > 0.0) {
        prev_pos[0] = static_cast<float>(static_cast<double>(prev_pos[0]) + ux * excess_n);
        prev_pos[1] = static_cast<float>(static_cast<double>(prev_pos[1]) + uy * excess_n);
        prev_pos[2] = static_cast<float>(static_cast<double>(prev_pos[2]) + uz * excess_n);
    }
}

/// Records the pinned-vertex `hclMoveParticlesOperator` binding for `sim` and computes exact
/// FP64 world->local pinned targets (eliminating 500 m world-space FP32 `col3` ULP staircase).
void LocalFp64_RecordMoveParticles(const u8* move_op, u8* sim, const u8* ref_buf) noexcept;

/// Prunes stale `hclSimClothInstance*` entries once per 64 frames right after `waitForStepComplete`.
void LocalFp64_OnWorldStepBegin(std::uint64_t world_seq) noexcept;

/// Mode 1.1 replacement for `hclSimulateOperator::preStep` (`0x076fb90`).
void LocalFp64_PreStep(u8* op, u8* step_info) noexcept;

/// Mode 1.1 replacement for `hclSimulateOperator::simulate` (`0x076e370`).
int LocalFp64_Simulate(u8* op, u8* ctx, float dt) noexcept;

/// Thread-safe velocity query used by `04_SprjWind` in Native-FPS modes (`Mode 1.1` / `1.2`)
/// so aerodynamic drag & directional turbulence read exact `FP64` particle velocity `v_i` (m/s)
/// instead of noisy world-space `(cur - prev) / sdt`. Returns `false` if `sim` has no FP64 state.
bool LocalFp64_CopyParticleVelocities(const u8* sim, __m128* out_vel, std::int32_t max_n) noexcept;

} // namespace BbClothExp
