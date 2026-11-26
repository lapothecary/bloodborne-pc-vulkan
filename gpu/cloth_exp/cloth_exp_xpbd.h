// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// File: gpu/cloth_exp/cloth_exp_xpbd.h
//
// Mode 1.2: Next-Gen Small-Steps FP64 XPBD Cloth Solver (Native Any-FPS)
//   - Character-anchor-centered FP64 positions & explicit FP64 velocities
//   - Exact binary compilation of all PS4 Havok scalar + 16-wide Mx SoA
//     constraint sets (`StandardLink`, `StretchLink`, `CompressibleLink`,
//     `BendLink`, `BendStiffness`, `LocalRange`) into compliant XPBD
//     constraints (`alpha = (w_a + w_b) * h0^2 * (1 - k) / k`) with Lagrange
//     multiplier `lambda` accumulation and Rayleigh fiber damping
//   - Per-triangle 3D fluid-cloth aerodynamic lift, normal pressure cushion,
//     and trailing-edge billow during sprints, quicksteps, rolls, and strikes
//   - Analytical swept Sphere / Plane / Capsule / Tapered-Capsule CCD in FP64
//     with Coulomb contact friction + local-space convex/landscape fallback.
// ============================================================================
#pragma once

#include <cmath>
#include <cstdint>
#include <immintrin.h>

#include "common/types.h"

namespace BbClothExp {

struct XpbdTuningParams {
    float compliance_scale = 2.50f; // Curated natural drape default (1.0 = raw .hkx; >1 softer bend)
    float aero_lift_scale = 0.25f;  // Curated restrained air cushion (avoids cape ballooning)
    float damping_scale = 1.50f;    // Curated heavier gothic wool/leather settling inertia
};

XpbdTuningParams Xpbd_GetTuning() noexcept;
void Xpbd_SetTuning(const XpbdTuningParams& params) noexcept;
void Xpbd_ResetTuningDefaults() noexcept;

/// Analytical conversion from Havok's authored per-PS4-substep normalized PBD stiffness
/// `k_unit` in `(0, 1]` at reference substep `ps4_sdt` to physical XPBD compliance `alpha` (m/N):
///   `alpha = w_sum * ps4_sdt^2 * (1 - k_unit) / k_unit`.
/// Makes physical Hookean stiffness `K = 1 / alpha` strictly invariant to frame rate and substeps!
inline double ComputeXpbdComplianceFromHavok(double k_unit, double w_sum,
                                             double ps4_sdt,
                                             double compliance_scale = 1.0) noexcept {
    if (!(w_sum > 1e-12)) {
        return 0.0;
    }
    const double k_clamped = (k_unit < 1e-4) ? 1e-4 : ((k_unit > 0.99995) ? 0.99995 : k_unit);
    if (k_clamped >= 0.9995) {
        // Authored as inextensible structural/tether fiber (`k == 1.0`).
        return 0.0;
    }
    const double base_alpha = w_sum * (ps4_sdt * ps4_sdt) * ((1.0 - k_clamped) / k_clamped);
    return base_alpha * ((compliance_scale > 0.05) ? compliance_scale : 0.05);
}

/// Quadrature-floored XPBD compliance matching Havok Cloth across multi-row curved garments
/// (`bd_m_1300` etc.): preserves 90-100% of authored `.hkx` stiffness on `< 1.0` leather/wool bends
/// (`hypot(intrinsic, stencil_compliance)` instead of `intrinsic + stencil_compliance`, which
/// previously softened `k = 0.6..0.9` coat panels by 2.5-11x!) while smoothly flooring `k >= 1.0`
/// links at `stencil_compliance` so conical capes/skirts never suffer `alpha = 0` membrane locking.
inline double ComputeXpbdMeshCalibratedCompliance(double k_raw, double w_sum,
                                                  double ps4_sdt,
                                                  double stencil_compliance,
                                                  double compliance_scale = 1.0) noexcept {
    if (!(w_sum > 1e-12)) {
        return 0.0;
    }
    const double kr = (k_raw < 1e-4) ? 1e-4 : k_raw;
    const double intrinsic = (kr < 1.0) ? ((1.0 - kr) / kr) : 0.0;
    const double sc = (compliance_scale > 0.05) ? compliance_scale : 0.05;
    const double eff_c = std::hypot(intrinsic, stencil_compliance);
    return w_sum * (ps4_sdt * ps4_sdt) * eff_c * sc;
}

/// Single bilateral/unilateral 2-particle XPBD distance constraint step in FP64 with Lagrange
/// multiplier `io_lambda` accumulation and optional Rayleigh velocity damping along the fiber.
/// Exposed inline for deterministic offline unit verification in `camera-motion-test`.
inline void SolveXpbdDistanceLinkFp64(double* x_a, double* x_b, const double* x_old_a,
                                      const double* x_old_b, double w_a, double w_b,
                                      double target_len, double alpha, double rayleigh_beta,
                                      double sdt, bool unilateral_max, bool unilateral_min,
                                      double& io_lambda) noexcept {
    const double w_sum = w_a + w_b;
    if (!(w_sum > 1e-12)) {
        return;
    }
    const double dx = x_b[0] - x_a[0];
    const double dy = x_b[1] - x_a[1];
    const double dz = x_b[2] - x_a[2];
    const double len2 = dx * dx + dy * dy + dz * dz;
    if (!(len2 > 1e-18)) {
        return;
    }
    const double len = std::sqrt(len2);
    const double C = len - target_len;
    if (unilateral_max && !(C > 0.0) && !(io_lambda < 0.0)) {
        return;
    }
    if (unilateral_min && !(C < 0.0) && !(io_lambda > 0.0)) {
        return;
    }

    const double inv_len = 1.0 / len;
    const double nx = dx * inv_len;
    const double ny = dy * inv_len;
    const double nz = dz * inv_len;

    const double alpha_tilde = alpha / (sdt * sdt);
    const double gamma = (alpha_tilde > 0.0 && rayleigh_beta > 0.0)
                             ? (alpha_tilde * rayleigh_beta * sdt)
                             : 0.0;
    double vel_term = 0.0;
    if (gamma > 0.0 && x_old_a != nullptr && x_old_b != nullptr) {
        const double rel_disp_n = ((x_b[0] - x_old_b[0]) - (x_a[0] - x_old_a[0])) * nx +
                                  ((x_b[1] - x_old_b[1]) - (x_a[1] - x_old_a[1])) * ny +
                                  ((x_b[2] - x_old_b[2]) - (x_a[2] - x_old_a[2])) * nz;
        vel_term = gamma * rel_disp_n;
    }

    const double denom = (1.0 + gamma) * w_sum + alpha_tilde;
    if (!(denom > 1e-18)) {
        return;
    }
    double d_lambda = -(C + alpha_tilde * io_lambda + vel_term) / denom;
    if (unilateral_max && (io_lambda + d_lambda > 0.0)) {
        d_lambda = -io_lambda;
    } else if (unilateral_min && (io_lambda + d_lambda < 0.0)) {
        d_lambda = -io_lambda;
    }
    io_lambda += d_lambda;

    // Note: C = |x_b - x_a| - L0 -> grad_b C = +n, grad_a C = -n:
    //   dx_a = w_a * grad_a C * d_lambda = -w_a * d_lambda * n
    //   dx_b = w_b * grad_b C * d_lambda = +w_b * d_lambda * n
    if (w_a > 0.0) {
        const double sa = -w_a * d_lambda;
        x_a[0] += sa * nx;
        x_a[1] += sa * ny;
        x_a[2] += sa * nz;
    }
    if (w_b > 0.0) {
        const double sb = w_b * d_lambda;
        x_b[0] += sb * nx;
        x_b[1] += sb * ny;
        x_b[2] += sb * nz;
    }
}

/// Clamps artificial normal recoil velocity (`e = 0` Galilean-invariant inelastic contact /
/// unilateral tether law) in FP64:
/// When a unilateral constraint or collider pushes `x` from `pre_x` to `post_x` along
/// `u = (post_x - pre_x) / |post_x - pre_x|`, ensures `(post_x - io_old_x) . u` never exceeds
/// `max(0, (pre_x - ref_old_x) . u, max_driver_step)`, preserving 100% of tangential motion and
/// friction while eliminating `push / sdt` lateral slingshot oscillations on belt straps & skirts.
inline void ClampXpbdInelasticPushFp64(const double pre_x[3], const double post_x[3],
                                       const double ref_old_x[3], double io_old_x[3],
                                       double max_driver_step = 0.0) noexcept {
    const double cdx = post_x[0] - pre_x[0];
    const double cdy = post_x[1] - pre_x[1];
    const double cdz = post_x[2] - pre_x[2];
    const double cd2 = cdx * cdx + cdy * cdy + cdz * cdz;
    if (!(cd2 > 1e-24)) {
        return;
    }
    const double inv_cd = 1.0 / std::sqrt(cd2);
    const double ux = cdx * inv_cd;
    const double uy = cdy * inv_cd;
    const double uz = cdz * inv_cd;

    const double pre_n = (pre_x[0] - ref_old_x[0]) * ux + (pre_x[1] - ref_old_x[1]) * uy +
                         (pre_x[2] - ref_old_x[2]) * uz;
    const double post_n = (post_x[0] - io_old_x[0]) * ux + (post_x[1] - io_old_x[1]) * uy +
                          (post_x[2] - io_old_x[2]) * uz;
    const double pre_pos = (pre_n > 0.0) ? pre_n : 0.0;
    const double allowed = (max_driver_step > pre_pos) ? max_driver_step : pre_pos;
    if (post_n > allowed) {
        const double corr = post_n - allowed;
        io_old_x[0] += ux * corr;
        io_old_x[1] += uy * corr;
        io_old_x[2] += uz * corr;
    }
}

/// Selects Small-Steps XPBD substep count in `[2, 6]` (`target_sdt = 1/180 s`) with `+/-15%`
/// deadband hysteresis around `prev_substeps` so minor frame-time variance (e.g. 85-95 FPS FG base
/// rate or 170-185 FPS native) never toggles `2 <-> 3` substeps from frame to frame.
inline int SelectXpbdSubstepsWithHysteresis(double smooth_dt, int prev_substeps) noexcept {
    constexpr double kTargetSdt = 1.0 / 180.0; // 5.556 ms
    const int raw_cld = static_cast<int>(std::ceil(smooth_dt / kTargetSdt));
    const int raw_sub = (raw_cld < 2) ? 2 : ((raw_cld > 6) ? 6 : raw_cld);
    if (prev_substeps < 2 || prev_substeps > 6) {
        return raw_sub;
    }
    const double upper_dt = static_cast<double>(prev_substeps) * kTargetSdt * 1.15;
    const double lower_dt = static_cast<double>(prev_substeps - 1) * kTargetSdt * 0.85;
    if (smooth_dt > upper_dt || (prev_substeps > 2 && smooth_dt < lower_dt)) {
        return raw_sub;
    }
    return prev_substeps;
}

/// Classifies narrow 1D/ribbon accessories (belts, straps, chains, pouches) and micro `MeshDeform`
/// proxy rigs (such as `bd_m_2200` `BD_M_2200_belt` `Sim[3]`: `n_part = 3`, `n_tris = 1`, `0`
/// `StretchLink`s driving `hclMeshMeshDeformOperator` with `scaleNormalBehaviour = 1`) versus wide
/// 2D garment sheets (coats, capes, skirts):
///   - Proxy triangles & narrow ribbons (`n_tris <= 32` or `>= 45%` open boundary edges, or small
///     layers lacking `StretchLink`) do not suffer from 2D conical skirt membrane locking and MUST
///     keep inextensible leather `StandardLink` compliance (`0.004`) so `MeshDeform` triangles never
///     inflate 3D belt geometry or bounce like rubber!
inline bool IsXpbdStrapOrProxyTopology(std::int32_t n_part, std::int32_t n_tris,
                                       std::int32_t boundary_edges, std::int32_t total_edges,
                                       bool has_stretch_links) noexcept {
    (void)n_part;
    if (n_tris <= 32) {
        return true;
    }
    if (!has_stretch_links) {
        // Any layer authored without `hclStretchLinkConstraintSet` (belts, micro-proxies, or
        // boss flesh/tentacle meshes) relies solely on `StandardLink` to prevent elongation.
        return true;
    }
    if (total_edges > 0 && (boundary_edges * 100 >= total_edges * 45)) {
        return true;
    }
    return false;
}

/// Maps `XPBD Fabric Softness (Compliance)` (`compliance_scale`) onto in-plane `StandardLink`
/// fiber compliance (while `BendStiffness` & `BendLink` `bendMin` receive the full
/// `max(0.05, compliance_scale)`):
///   - Real cloth/leather softness (e.g. relaxing a T-pose shoulder capelet onto the arms) is
///     dihedral bending suppleness, NOT rubber-band stretching of fabric length/width!
///   - Narrow belts/straps/proxies (`is_strap_or_proxy == true`) never soften `StandardLink` above
///     `1.0`, eliminating belt size growth & rubber-banding at high `Compliance`.
///   - Wide garment sheets allow at most a mild `12%` shear cushion above `1.0` while honoring `< 1`
///     stiffening in full.
inline double ComputeXpbdEffectiveWeaveScale(double compliance_scale,
                                             bool is_strap_or_proxy) noexcept {
    const double c = (compliance_scale > 0.05) ? compliance_scale : 0.05;
    if (is_strap_or_proxy) {
        return (c < 1.0) ? c : 1.0;
    }
    return (c <= 1.0) ? c : (1.0 + 0.12 * (c - 1.0));
}

/// Records pinned-vertex `hclMoveParticlesOperator` targets in exact FP64 for Mode 1.2.
void Xpbd_RecordMoveParticles(const u8* move_op, u8* sim, const u8* ref_buf) noexcept;

/// Prunes stale XPBD cloth instance states once per 64 world steps.
void Xpbd_OnWorldStepBegin(std::uint64_t world_seq) noexcept;

/// Mode 1.2 `hclSimulateOperator::preStep` replacement.
void Xpbd_PreStep(u8* op, u8* step_info) noexcept;

/// Mode 1.2 `hclSimulateOperator::simulate` replacement.
int Xpbd_Simulate(u8* op, u8* ctx, float dt) noexcept;

/// Thread-safe FP64 particle velocity readout for `04_SprjWind` in Mode 1.2.
bool Xpbd_CopyParticleVelocities(const u8* sim, __m128* out_vel, std::int32_t max_n) noexcept;

} // namespace BbClothExp
