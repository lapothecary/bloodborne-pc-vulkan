// SPDX-License-Identifier: GPL-2.0-or-later
// Sealed C++20 Havok Cloth native arbitrary-FPS constraint scaling & optional fixed-30 helpers.
//
// Verified eboot.bin VMAs (PS4 ELF, image_base + offset):
//   - ClothStiffnessFactor:                    0x1167a80 (callers: 0x7720e4, 0x7722f5; exp -1.725f at 0x48d7bbc)
//   - hclStandardLinkConstraintSet (type=1):   finish 0x705fbe, vtable 0x526aa10, apply 0x705fe0
//   - hclStretchLinkConstraintSet (type=2):    finish 0x6f9fee, vtable 0x5266aa0, apply 0x6fa010
//   - hclLocalRangeConstraintSet (type=5):     finish 0x6fad1e, vtable 0x5263750, apply 0x6fb220 -> 0x2553210
//   - hclVolumeConstraint (type=6 / 10):       finish 0x70ea7e / 0x70697e
//   - hclTransitionConstraintSet (type=8):     finish 0x7064be
//   - hclBendStiffnessConstraintSetMx (16):    finish 0x705cee, vtable 0x526a930, apply 0x705d00
//   - hclCompressibleLinkConstraintSet (20):   finish 0x6fa43e, vtable 0x5269ca0, apply 0x6fa450
//   - hclClothInstance::step:                  0x76a960 (caller of 0x76e370 at 0x76ab3c)
//   - hclSimulateOperator::simulate:           0x76e370
//   - hclSimulateOperator::preStep:            0x76fb90 (damping pow at 0x76fee6, dt rescale at 0x76fdd8)
//   - hclSimClothInstance::transferMotion:     0x771290
//   - hclTransformSet/matrix transpose-inv:    0x43b980 (called from 0x771320)
//   - FD4ClothWorld::step:                     0xfd5440 (post-wait sync at 0xfd54a2)
//   - SprjWindAction::apply:                   0x1051fa0 (vtable 0x573bb00)
//   - SprjWindEvalWorldDir (sub_10522f0):      0x10522f0 (1000 Hz XorShift gust at 0x10523db..0x1052448)
//   - SprjPointWind:                           0x1052af0 (gust at 0x1052bcc..0x1052c39, inv_dt at 0x10532f2)
//   - SprjDirWind:                             0x1053b90 (0x1053b90..0x105478f)
//   - XorShift128::seed:                       0x20f81c0 / 0x2686010 (0x5deece66dULL * s + 0xbULL LCG)
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace BbClothInterp {

/// Verified byte offsets inside PS4 Havok Cloth runtime & setup structures (`eboot.bin`).
struct HclClothInstanceOffsets {
    static constexpr std::size_t kClothData = 0x10;
    static constexpr std::size_t kCurrentState = 0x18;
    static constexpr std::size_t kSimClothInstancesData = 0x20;
    static constexpr std::size_t kSimClothInstancesSize = 0x28;
    static constexpr std::size_t kBuffersData = 0x30;
    static constexpr std::size_t kBuffersSize = 0x38;
    static constexpr std::size_t kTransformSetsData = 0x40;
    static constexpr std::size_t kTransformSetsSize = 0x48;
};

struct HclSimClothInstanceOffsets {
    static constexpr std::size_t kSimClothData = 0x10;
    static constexpr std::size_t kCurrentPositions = 0x18;
    static constexpr std::size_t kNumParticles = 0x20;
    static constexpr std::size_t kPreviousPositions = 0x28;
    static constexpr std::size_t kParticleNormalsData = 0x38;
    static constexpr std::size_t kParticleNormalsSize = 0x40;
    static constexpr std::size_t kCollidablesData = 0xd0;
    static constexpr std::size_t kCollidablesSize = 0xd8;
    static constexpr std::size_t kPrevSubstepDt = 0x108;
    static constexpr std::size_t kEffectiveDamping = 0x10c;
    static constexpr std::size_t kPrevTransferMotionTransform = 0x120;
    static constexpr std::size_t kActiveCollidablesSize = 0x1b8;
    static constexpr std::size_t kActiveCollidablesData = 0x1c0;
    static constexpr std::size_t kAabbMode = 0x1c8;
    static constexpr std::size_t kSlowMotionMode = 0x1cc;
    static constexpr std::size_t kSlowMotionFactor = 0x1d0;
};

struct HclSimClothDataOffsets {
    static constexpr std::size_t kGravity = 0x10;
    static constexpr std::size_t kGlobalDampingPerSecond = 0x20;
    static constexpr std::size_t kParticleDatasData = 0x38;
    static constexpr std::size_t kParticleDatasSize = 0x40;
    static constexpr std::size_t kFixedParticlesData = 0x48;
    static constexpr std::size_t kFixedParticlesSize = 0x50;
    static constexpr std::size_t kTotalMass = 0x78;
    static constexpr std::size_t kCollidableTransformSetIndex = 0x80;
    static constexpr std::size_t kCollidableTransformMapData = 0x88;
    static constexpr std::size_t kCollidableTransformMapSize = 0x90;
    static constexpr std::size_t kCollidableTransformOffsetsData = 0x98;
    static constexpr std::size_t kStaticConstraintSetsData = 0xb8;
    static constexpr std::size_t kStaticConstraintSetsSize = 0xc0;
    static constexpr std::size_t kAntiPinchConstraintSetsData = 0xc8;
    static constexpr std::size_t kAntiPinchConstraintSetsSize = 0xd0;
    static constexpr std::size_t kDoNormals = 0x14c;
    static constexpr std::size_t kTransferTransformSetIndex = 0x150;
    static constexpr std::size_t kTransferTransformIndex = 0x154;
    static constexpr std::size_t kTransferTranslationMotion = 0x158;
    static constexpr std::size_t kMinTranslationSpeed = 0x15c;
    static constexpr std::size_t kMaxTranslationSpeed = 0x160;
};

struct HclSimulateOperatorOffsets {
    static constexpr std::size_t kSimClothIndex = 0x1c;
    static constexpr std::size_t kSubSteps = 0x20;
    static constexpr std::size_t kConstraintIterations = 0x24;
    static constexpr std::size_t kConstraintExecutionsData = 0x28;
    static constexpr std::size_t kConstraintExecutionsSize = 0x30;
};

struct HclCollidableOffsets {
    static constexpr std::size_t kStride = 0x90;
    static constexpr std::size_t kTransform = 0x20;
    static constexpr std::size_t kLinearVelocity = 0x60;
    static constexpr std::size_t kAngularVelocity = 0x70;
};

struct SprjWindActionOffsets {
    static constexpr std::size_t kDragCoeff = 0x94;
    static constexpr std::size_t kPointCullStep = 0xa4;
    static constexpr std::size_t kDirCullStep = 0xa8;
    static constexpr std::size_t kPointCullEnabled = 0xac;
    static constexpr std::size_t kDirCullEnabled = 0xad;
};

struct HclBufferOffsets {
    static constexpr std::size_t kBufferToWorldTransform = 0x80;
    static constexpr std::size_t kWorldToBufferTransform = 0xc0;
    static constexpr std::size_t kShadowBufferIndex = 0x100;
};

struct HclTransformSetOffsets {
    static constexpr std::size_t kMatricesData = 0x10;
    static constexpr std::size_t kMatricesSize = 0x18;
};

struct alignas(16) ClothMat4 {
    float m[16]{};
};
static_assert(sizeof(ClothMat4) == 64 && alignof(ClothMat4) == 16,
              "ClothMat4 must match 16-byte aligned hkMatrix4 layout");

struct alignas(16) ClonedHclTransformSet {
    std::uint8_t bytes[64]{};
};
static_assert(sizeof(ClonedHclTransformSet) == 64 && alignof(ClonedHclTransformSet) == 16,
              "ClonedHclTransformSet must hold a full hkTransformSet header safely");

inline void InitClonedHclTransformSet(ClonedHclTransformSet& dst, const std::uint8_t* src,
                                      ClothMat4* matrices) noexcept {
    std::memset(dst.bytes, 0, sizeof(dst.bytes));
    if (src) {
        const std::size_t copy_bytes =
            ((reinterpret_cast<std::uintptr_t>(src) & 0xFFFu) <= 0xFC0u) ? 64u : 32u;
        std::memcpy(dst.bytes, src, copy_bytes);
    }
    std::memcpy(dst.bytes + HclTransformSetOffsets::kMatricesData, &matrices, sizeof(matrices));
}

struct ClothFixed30Tick {
    bool do_step = true;
    float alpha = 1.0f;
    float step_beta = 1.0f;
    float step_dt = 1.0f / 30.0f;
};

/// Advances the fixed-30 Hz cloth accumulator for base-frame duration `cdt` and computes:
///  - `do_step`: whether a 1/30 s Havok Cloth Verlet/constraint solve runs on this base frame
///  - `step_beta`: fractional position `[0..1]` within `[t_prev, t_curr]` where the exact 30 Hz
///    tick crossed `Frame30` (used to interpolate character poses/colliders at exact `t_sim`)
///  - `alpha`: post-step fractional position `[0..1]` for smooth CPU vertex interpolation `[prev, curr]`
inline ClothFixed30Tick ComputeClothFixed30Tick(double& accum_s,
                                                std::uint32_t& steady_30_streak, float cdt,
                                                float frame30 = 1.0f / 30.0f) noexcept {
    ClothFixed30Tick out{};
    out.step_dt = frame30;

    if (cdt >= frame30 * 0.96f) {
        steady_30_streak = std::min<std::uint32_t>(steady_30_streak + 1u, 16u);
    } else {
        steady_30_streak = 0u;
    }

    // Native ~30 FPS steady-state bypass (zero latency). Require a short streak (or empty accum)
    // so an isolated ~33 ms hitch frame at 42.5..85 FPS keeps accumulating cleanly without
    // jumping alpha forward to 1.0 and backward on the next frame.
    if ((cdt >= frame30 * 0.96f) && (steady_30_streak >= 4u || accum_s <= 1e-6)) {
        accum_s = 0.0;
        out.do_step = true;
        out.alpha = 1.0f;
        out.step_beta = 1.0f;
        out.step_dt = cdt;
        return out;
    }

    const double prev_accum = std::clamp(accum_s, 0.0, double(frame30) * 0.999999);
    const double next_accum = prev_accum + double(cdt);
    if (next_accum >= double(frame30)) {
        const double needed = std::max(0.0, double(frame30) - prev_accum);
        out.step_beta =
            std::clamp(static_cast<float>(needed / std::max<double>(cdt, 1e-6)), 0.0f, 1.0f);
        accum_s = std::min(next_accum - double(frame30), double(frame30) * 0.999);
        out.do_step = true;
        out.step_dt = frame30;
    } else {
        accum_s = next_accum;
        out.do_step = false;
        out.step_beta = 1.0f;
        out.step_dt = frame30;
    }
    out.alpha = std::clamp(static_cast<float>(accum_s / double(frame30)), 0.0f, 1.0f);
    return out;
}

/// Interpolates two column-major 4x4 affine matrices `m0` -> `m1` at fraction `beta` in `[0, 1]`,
/// using shortest-arc unit-quaternion SLERP for the orthonormalized 3x3 rotation part (supporting
/// mirrored reflections consistently if `det < 0`), linear column-scale blending, and linear
/// translation blending. Falls back cleanly to `m1` on non-finite, degenerate, or >1.5 m teleports.
inline void LerpTransformMatrix4x4(const float* m0, const float* m1, float beta,
                                   float* out) noexcept {
    if (!m0 || !m1 || !out) {
        return;
    }
    if (beta >= 0.9999f) {
        std::memcpy(out, m1, 64);
        return;
    }
    for (int k = 0; k < 16; ++k) {
        if (!std::isfinite(m0[k]) || !std::isfinite(m1[k])) {
            std::memcpy(out, m1, 64);
            return;
        }
    }
    const float dx = m1[12] - m0[12];
    const float dy = m1[13] - m0[13];
    const float dz = m1[14] - m0[14];
    if (dx * dx + dy * dy + dz * dz > 2.25f) {
        std::memcpy(out, m1, 64);
        return;
    }
    if (beta <= 0.0001f) {
        std::memcpy(out, m0, 64);
        return;
    }

    auto col_len = [](const float* c) noexcept -> float {
        return std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    };
    const float s0_a = col_len(m0 + 0);
    const float s1_a = col_len(m0 + 4);
    const float s2_a = col_len(m0 + 8);
    const float s0_b = col_len(m1 + 0);
    const float s1_b = col_len(m1 + 4);
    const float s2_b = col_len(m1 + 8);
    if (s0_a < 1e-4f || s1_a < 1e-4f || s2_a < 1e-4f || s0_b < 1e-4f || s1_b < 1e-4f ||
        s2_b < 1e-4f || s0_a > 100.0f || s1_a > 100.0f || s2_a > 100.0f || s0_b > 100.0f ||
        s1_b > 100.0f || s2_b > 100.0f) {
        std::memcpy(out, m1, 64);
        return;
    }

    float a0[3] = {m0[0] / s0_a, m0[1] / s0_a, m0[2] / s0_a};
    float a1[3] = {m0[4] / s1_a, m0[5] / s1_a, m0[6] / s1_a};
    float a2[3] = {m0[8] / s2_a, m0[9] / s2_a, m0[10] / s2_a};

    float b0[3] = {m1[0] / s0_b, m1[1] / s0_b, m1[2] / s0_b};
    float b1[3] = {m1[4] / s1_b, m1[5] / s1_b, m1[6] / s1_b};
    float b2[3] = {m1[8] / s2_b, m1[9] / s2_b, m1[10] / s2_b};

    auto det3 = [](const float c0[3], const float c1[3], const float c2[3]) noexcept -> float {
        const float cx = c1[1] * c2[2] - c1[2] * c2[1];
        const float cy = c1[2] * c2[0] - c1[0] * c2[2];
        const float cz = c1[0] * c2[1] - c1[1] * c2[0];
        return c0[0] * cx + c0[1] * cy + c0[2] * cz;
    };
    const float det_a = det3(a0, a1, a2);
    const float det_b = det3(b0, b1, b2);
    if (det_a * det_b <= 0.0f || std::fabs(det_b) < 0.5f || std::fabs(det_a) < 0.5f) {
        std::memcpy(out, m1, 64);
        return;
    }
    const bool flip_sign = (det_b < 0.0f);
    if (flip_sign) {
        a2[0] = -a2[0];
        a2[1] = -a2[1];
        a2[2] = -a2[2];
        b2[0] = -b2[0];
        b2[1] = -b2[1];
        b2[2] = -b2[2];
    }

    auto mat_to_quat = [](const float c0[3], const float c1[3], const float c2[3],
                          float q[4]) noexcept {
        const float r00 = c0[0], r10 = c0[1], r20 = c0[2];
        const float r01 = c1[0], r11 = c1[1], r21 = c1[2];
        const float r02 = c2[0], r12 = c2[1], r22 = c2[2];
        const float tr = r00 + r11 + r22;
        if (tr > 0.0f) {
            const float s = std::sqrt(tr + 1.0f) * 2.0f;
            q[3] = 0.25f * s;
            q[0] = (r21 - r12) / s;
            q[1] = (r02 - r20) / s;
            q[2] = (r10 - r01) / s;
        } else if (r00 > r11 && r00 > r22) {
            const float s = std::sqrt(1.0f + r00 - r11 - r22) * 2.0f;
            q[3] = (r21 - r12) / s;
            q[0] = 0.25f * s;
            q[1] = (r01 + r10) / s;
            q[2] = (r02 + r20) / s;
        } else if (r11 > r22) {
            const float s = std::sqrt(1.0f + r11 - r00 - r22) * 2.0f;
            q[3] = (r02 - r20) / s;
            q[0] = (r01 + r10) / s;
            q[1] = 0.25f * s;
            q[2] = (r12 + r21) / s;
        } else {
            const float s = std::sqrt(1.0f + r22 - r00 - r11) * 2.0f;
            q[3] = (r10 - r01) / s;
            q[0] = (r02 + r20) / s;
            q[1] = (r12 + r21) / s;
            q[2] = 0.25f * s;
        }
        const float inv_n =
            1.0f / std::max(std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]),
                            1e-12f);
        q[0] *= inv_n;
        q[1] *= inv_n;
        q[2] *= inv_n;
        q[3] *= inv_n;
    };

    float q0[4]{};
    float q1[4]{};
    mat_to_quat(a0, a1, a2, q0);
    mat_to_quat(b0, b1, b2, q1);

    float dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
    if (dot < 0.0f) {
        q0[0] = -q0[0];
        q0[1] = -q0[1];
        q0[2] = -q0[2];
        q0[3] = -q0[3];
        dot = -dot;
    }

    float q[4]{};
    if (dot > 0.9995f) {
        const float w0 = 1.0f - beta;
        const float w1 = beta;
        for (int i = 0; i < 4; ++i) {
            q[i] = w0 * q0[i] + w1 * q1[i];
        }
    } else {
        const float clamped_dot = std::clamp(dot, -1.0f, 1.0f);
        const float theta = std::acos(clamped_dot);
        const float sin_theta = std::sin(theta);
        if (sin_theta > 1e-6f) {
            const float w0 = std::sin((1.0f - beta) * theta) / sin_theta;
            const float w1 = std::sin(beta * theta) / sin_theta;
            for (int i = 0; i < 4; ++i) {
                q[i] = w0 * q0[i] + w1 * q1[i];
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                q[i] = (1.0f - beta) * q0[i] + beta * q1[i];
            }
        }
    }
    const float inv_qn =
        1.0f / std::max(std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]), 1e-12f);
    const float x = q[0] * inv_qn;
    const float y = q[1] * inv_qn;
    const float z = q[2] * inv_qn;
    const float w = q[3] * inv_qn;

    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;

    const float s0 = s0_a + (s0_b - s0_a) * beta;
    const float s1 = s1_a + (s1_b - s1_a) * beta;
    const float s2 = (flip_sign ? -1.0f : 1.0f) * (s2_a + (s2_b - s2_a) * beta);

    out[0] = (1.0f - 2.0f * (yy + zz)) * s0;
    out[1] = (2.0f * (xy + wz)) * s0;
    out[2] = (2.0f * (xz - wy)) * s0;
    out[3] = m1[3];

    out[4] = (2.0f * (xy - wz)) * s1;
    out[5] = (1.0f - 2.0f * (xx + zz)) * s1;
    out[6] = (2.0f * (yz + wx)) * s1;
    out[7] = m1[7];

    out[8] = (2.0f * (xz + wy)) * s2;
    out[9] = (2.0f * (yz - wx)) * s2;
    out[10] = (1.0f - 2.0f * (xx + yy)) * s2;
    out[11] = m1[11];

    out[12] = m0[12] + dx * beta;
    out[13] = m0[13] + dy * beta;
    out[14] = m0[14] + dz * beta;
    out[15] = m1[15];
}

/// Computes the exact inverse `out_w2l` of an orthogonal-column 4x4 affine matrix `l2w`.
/// Returns `false` if any column length is outside `[sqrt(0.1), sqrt(100)]` or non-finite.
inline bool InvertRigidMatrix4x4(const float* l2w, float* out_w2l) noexcept {
    if (!l2w || !out_w2l) {
        return false;
    }
    for (int k = 0; k < 16; ++k) {
        if (!std::isfinite(l2w[k])) {
            return false;
        }
    }
    float inv_n2[3]{};
    for (int col = 0; col < 3; ++col) {
        const float* c = l2w + col * 4;
        const float n2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
        if (n2 < 0.1f || n2 > 100.0f) {
            return false;
        }
        inv_n2[col] = 1.0f / n2;
    }

    for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < 3; ++row) {
            out_w2l[col * 4 + row] = l2w[row * 4 + col] * inv_n2[row];
        }
    }
    out_w2l[3] = 0.0f;
    out_w2l[7] = 0.0f;
    out_w2l[11] = 0.0f;

    const float tx = l2w[12], ty = l2w[13], tz = l2w[14];
    out_w2l[12] = -(out_w2l[0] * tx + out_w2l[4] * ty + out_w2l[8] * tz);
    out_w2l[13] = -(out_w2l[1] * tx + out_w2l[5] * ty + out_w2l[9] * tz);
    out_w2l[14] = -(out_w2l[2] * tx + out_w2l[6] * ty + out_w2l[10] * tz);
    out_w2l[15] = 1.0f;
    return true;
}

/// Verified Havok Cloth `hclConstraintSet::m_type` (`[set + 0x18]`) values in `eboot.bin`.
enum class HclConstraintSetType : std::uint32_t {
    kStandardLink = 1u,      ///< hclStandardLinkConstraintSet (finish 0x705fbe, vtable 0x526aa10)
    kStretchLink = 2u,       ///< hclStretchLinkConstraintSet (finish 0x6f9fee, vtable 0x5266aa0 — SIGGRAPH 2012 LRA 1.0f tether)
    kLocalRange = 5u,        ///< hclLocalRangeConstraintSet (finish 0x6fad1e, apply 0x6fb220 -> 0x2553210; m_stiffness at +0x34)
    kVolume = 6u,            ///< hclVolumeConstraint (finish 0x70ea7e)
    kTransition = 8u,        ///< hclTransitionConstraintSet (finish 0x7064be)
    kVolumeAlt = 10u,        ///< hclVolumeConstraintMx / alt (finish 0x70697e)
    kBendStiffnessMx = 16u,  ///< hclBendStiffnessConstraintSetMx (finish 0x705cee, vtable 0x526a930)
    kCompressibleLink = 20u, ///< hclCompressibleLinkConstraintSet (finish 0x6fa43e, vtable 0x5269ca0)
};

/// Computes continuous-time stiffness scale for elastic Verlet PBD link/bend constraints
/// (`frames = clamp(dt * 30, 1/300, 1)`). Matches Havok's substep law `(substeps / frames)^-1.725`.
inline float ComputeClothElasticStiffnessScale(float frames, float exp = 1.725f) noexcept {
    const double f = std::clamp(double(frames), 1.0 / 300.0, 1.0);
    return static_cast<float>(std::pow(f, double(exp)));
}

/// Computes exact continuous-time exponential relaxation multiplier for `hclLocalRangeConstraintSet`
/// (`type == 5`, `m_stiffness` at `[set + 0x34]`) so `(1 - s * factor)^(1 / frames) == (1 - s)`.
/// Hard cages (`m_stiffness >= 0.999f`) stay unscaled (`1.0f`); normal pushout in `0x2553210` is
/// always unscaled (`no xmm9 multiply`).
inline float ComputeClothExponentialRelaxationFactor(float set_stiffness, float frames) noexcept {
    if (!(set_stiffness > 1e-5f) || set_stiffness >= 0.999f || frames >= 0.9999f) {
        return 1.0f;
    }
    const double f = std::clamp(double(frames), 1.0 / 300.0, 1.0);
    const double s = std::clamp(double(set_stiffness), 1e-5, 0.999);
    return static_cast<float>((-std::expm1(f * std::log(1.0 - s))) / s);
}

/// Exact continuous-time replacement for Havok Cloth `ClothStiffnessFactor` (`eboot.bin+0x1167a80`).
inline float ComputeNativeClothConstraintFactor(std::uint32_t type, float set_stiffness_at_0x34,
                                                int mode, int iteration, int substeps, float a,
                                                float b, float cloth_exponent, float frames,
                                                float elastic_stiffness_scale, int t5_mode = 3,
                                                bool t5_pulse_active = true,
                                                bool cloth_fix_enabled = true) noexcept {
    const bool adapts = (mode == 2) && !(substeps == 1 && a == 1.0f && b == 1.0f);
    if (adapts && type == static_cast<std::uint32_t>(HclConstraintSetType::kTransition)) {
        return float(iteration + 1) / float(substeps);
    }
    if (adapts && (type == static_cast<std::uint32_t>(HclConstraintSetType::kLocalRange) ||
                   type == static_cast<std::uint32_t>(HclConstraintSetType::kVolumeAlt))) {
        if (iteration != substeps - 1) {
            return 0.0f;
        }
    }
    float game = 1.0f;
    if (adapts && type != static_cast<std::uint32_t>(HclConstraintSetType::kLocalRange) &&
        type != static_cast<std::uint32_t>(HclConstraintSetType::kVolumeAlt)) {
        const float x = float(substeps) * a * b;
        if (x != 1.0f && x > 0.0f) {
            game = std::exp2f(cloth_exponent * std::log2f(x));
        }
    }
    if (!cloth_fix_enabled) {
        return game;
    }
    switch (static_cast<HclConstraintSetType>(type)) {
    case HclConstraintSetType::kStretchLink:
        // SIGGRAPH 2012 LRA 1.0f collar/waist tether must remain unscaled to prevent high-FPS stretching.
        return game;
    case HclConstraintSetType::kLocalRange:
        if (t5_mode == 0) {
            return game;
        }
        if (t5_mode == 2) {
            return t5_pulse_active ? game : 0.0f;
        }
        if (t5_mode == 1) {
            return game * elastic_stiffness_scale;
        }
        return game * ComputeClothExponentialRelaxationFactor(set_stiffness_at_0x34, frames);
    case HclConstraintSetType::kVolume:
    case HclConstraintSetType::kTransition:
    case HclConstraintSetType::kVolumeAlt:
        return game;
    default:
        return game * elastic_stiffness_scale;
    }
}

/// Bit-exact FromSoftware `XorShift128` PRNG (`eboot.bin+0x20f81c0` + `0x1053f14..0x1053f67`).
/// Seeded via the 48-bit LCG `0x5deece66d * s + 0xb`, advancing a 4-word state and mapping `s[3] % 0x800001`
/// directly into `[1.0f, 2.0f] - 1.0f` mantissa bits (`[0.0f, 1.0f]`).
struct ClothXorShift128 {
    std::uint32_t s[4]{};

    constexpr void Seed(std::uint32_t seed) noexcept {
        // Match `vcvttss2si rsi, xmm0` + `0x20f81c0`: sign-extend 32-bit `esi` to 64-bit `rsi`.
        std::uint64_t lcg =
            0x5deece66dULL *
                static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(seed))) +
            0xbULL;
        for (int i = 0; i < 4; ++i) {
            s[i] = static_cast<std::uint32_t>(lcg >> 32);
            lcg = 0x5deece66dULL * lcg + 0xbULL;
        }
    }

    inline float NextFloat01() noexcept {
        const std::uint32_t t = s[0] ^ (s[0] << 11);
        s[0] = s[1];
        s[1] = s[2];
        s[2] = s[3];
        s[3] = (s[3] ^ (s[3] >> 19)) ^ (t ^ (t >> 8));
        const std::uint32_t bits = 0x3f800000u + (s[3] % 0x800001u);
        float f = 1.0f;
        std::memcpy(&f, &bits, sizeof(f));
        return f - 1.0f;
    }
};

struct ClothWindSmoothState {
    std::uint32_t seed0 = 0u;
    std::uint32_t seed1 = 10000u;
    float smooth = 0.0f;
};

/// Evaluates the 30 Hz quantized seed pair `(seed0, seed1)` and cubic Hermite smoothstep blend weight
/// (`0.0` at integer 30 Hz boundaries for exact 30 FPS parity, $C^1$-smooth across high-FPS frames).
/// NOTE: `smooth` MUST blend the OUTPUT 3D wind vector/speed of `sub_10522f0` and `SprjPointWind`
/// evaluated at discrete `(seed0, seed1)` — NEVER blend the input `gust01` coordinate before
/// `PerlinNoise_fBm_4` (`0x2685e30`), which would sweep `6*p + 400` across 20..240 Perlin lattice
/// cells every 1/30 s (`~17x` wind oscillation speedhack at high FPS).
inline ClothWindSmoothState ComputeClothWindSmoothState(double sim_time_s) noexcept {
    const double tau = std::max(0.0, sim_time_s) * 30.0;
    const double tick0_d = std::floor(tau);
    const std::uint64_t tick0 = static_cast<std::uint64_t>(tick0_d);
    const float frac = std::clamp(static_cast<float>(tau - tick0_d), 0.0f, 1.0f);
    const float smooth = frac * frac * (3.0f - 2.0f * frac);
    const std::uint32_t seed0 = static_cast<std::uint32_t>((tick0 * 10000ull) % 1000000ull);
    const std::uint32_t seed1 = static_cast<std::uint32_t>(((tick0 + 1ull) * 10000ull) % 1000000ull);

    ClothWindSmoothState out{};
    out.seed0 = seed0;
    out.seed1 = seed1;
    out.smooth = smooth;
    return out;
}

/// Smoothly blends two discrete 30 Hz world wind vectors `v0` (`seed0`) and `v1` (`seed1`) returned
/// by `sub_10522f0` (`0x10522f0`) at weight `smooth` in `[0, 1]`, rotating direction smoothly while
/// preserving interpolated wind speed `(1 - smooth)*|v0| + smooth*|v1|` (preventing opposing gust
/// angles from dipping wind speed toward zero at `smooth = 0.5`).
inline void BlendClothWindVectors3D(const float v0[4], const float v1[4], float smooth,
                                    float out[4]) noexcept {
    if (!v0 || !v1 || !out) {
        return;
    }
    const float t = std::clamp(smooth, 0.0f, 1.0f);
    if (t <= 1e-6f) {
        std::memcpy(out, v0, 16);
        return;
    }
    if (t >= 0.999999f) {
        std::memcpy(out, v1, 16);
        return;
    }
    const float x0 = v0[0], y0 = v0[1], z0 = v0[2];
    const float x1 = v1[0], y1 = v1[1], z1 = v1[2];
    if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(z0) || !std::isfinite(x1) ||
        !std::isfinite(y1) || !std::isfinite(z1)) {
        std::memcpy(out, v0, 16);
        return;
    }
    const float len0 = std::sqrt(x0 * x0 + y0 * y0 + z0 * z0);
    const float len1 = std::sqrt(x1 * x1 + y1 * y1 + z1 * z1);
    const float target_spd = len0 + t * (len1 - len0);

    float bx = x0 + t * (x1 - x0);
    float by = y0 + t * (y1 - y0);
    float bz = z0 + t * (z1 - z0);
    float blen = std::sqrt(bx * bx + by * by + bz * bz);

    if (len0 > 1e-5f && len1 > 1e-5f) {
        if (blen <= 1e-5f) {
            // Nearly 180-deg opposite gusts: nudge along a deterministic axis orthogonal to v0.
            float ox = -z0, oy = 0.0f, oz = x0;
            if (ox * ox + oz * oz < 1e-6f) {
                ox = 0.0f;
                oy = z0;
                oz = -y0;
            }
            const float olen = std::max(std::sqrt(ox * ox + oy * oy + oz * oz), 1e-6f);
            const float sin_arc = 4.0f * t * (1.0f - t);
            bx += (ox / olen) * target_spd * sin_arc;
            by += (oy / olen) * target_spd * sin_arc;
            bz += (oz / olen) * target_spd * sin_arc;
            blen = std::sqrt(bx * bx + by * by + bz * bz);
        }
        if (blen > 1e-6f) {
            const float s = target_spd / blen;
            bx *= s;
            by *= s;
            bz *= s;
        }
    }

    out[0] = bx;
    out[1] = by;
    out[2] = bz;
    out[3] = v0[3];
}

/// Blends `SprjPointWind` (`0x1052af0`) scalar speed (`[rbp - 0x150]`) and 3D direction (`[rbp - 0x120]`)
/// evaluated at discrete 30 Hz seeds `(seed0, seed1)`.
inline void BlendClothPointWindSpeedAndDir(float spd0, const float dir0[4], float spd1,
                                           const float dir1[4], float smooth, float* out_spd,
                                           float out_dir[4]) noexcept {
    if (!dir0 || !dir1 || !out_spd || !out_dir) {
        return;
    }
    const float t = std::clamp(smooth, 0.0f, 1.0f);
    *out_spd = spd0 + t * (spd1 - spd0);
    BlendClothWindVectors3D(dir0, dir1, t, out_dir);
}

/// Computes an extra continuous-time negative damping rate (`-s^-1`) for `hclSimulateOperator::preStep`
/// (`0x76fb90`) as `frames -> 0`. Restores the subtle ~6% per 1/30 s algorithmic damping inherent in
/// 30 FPS PBD projection so high-FPS cloth stays calm while standing still (`0.0f` at 30 FPS).
inline float ComputeHighFpsClothDampingNegRate(float frames, float retention30 = 0.94f) noexcept {
    if (!(retention30 > 0.5f && retention30 < 0.9999f) || frames >= 0.999f) {
        return 0.0f;
    }
    const float clamped_f = std::clamp(frames, 1.0f / 300.0f, 1.0f);
    const float rate = (1.0f - clamped_f) * (-30.0f * std::log(retention30));
    return -std::clamp(rate, 0.0f, 6.0f);
}

/// Computes exact `hclSimClothInstance::m_effectiveDamping` (`[sim + 0x10c]`) for `preStep` (`0x76fb90`)
/// across all branches (`prev_sdt == 0`, `prev_sdt == sdt`, `prev_sdt != sdt`, `global_damp == 0`),
/// matching PS4 `0x76feb3..0x76ff36` (`(1 - gd)^sdt`) multiplied by continuous-time PBD damping parity.
inline float ComputeNativeClothEffectiveDamping(float global_damping_per_sec, float sdt,
                                                float pbd_neg_rate) noexcept {
    if (!(sdt > 0.0f && std::isfinite(sdt))) {
        return 1.0f;
    }
    if (!std::isfinite(global_damping_per_sec)) {
        return 1.0f;
    }
    if (global_damping_per_sec >= 1.0f) {
        return 0.0f;
    }
    const float base =
        (global_damping_per_sec > 0.0f) ? std::pow(1.0f - global_damping_per_sec, sdt) : 1.0f;
    const float extra = std::clamp(1.0f + sdt * pbd_neg_rate, 0.0f, 1.0f);
    return base * extra;
}

/// Native C++20 replacement for FromSoftware's `SprjDirWind` (`eboot.bin+0x1053b90..0x105478f`).
/// Eliminates high-FPS cloth tremor while preserving 100% 30-FPS physical wind force & wave behavior:
///   - Blends dual `ClothXorShift128(seed0, seed1)` per-particle turbulence via `smooth_t` ($C^1$).
///   - Computes relative wind drag velocity over the actual Verlet integration interval
///     `prev_sdt = [sim + 0x108]` (`TtActions` runs before `preStep` rescales `prev_pos`).
///   - Scales `(cur - prev)` inside turbulence to 30-FPS substep displacement parity (`sdt / frames`).
inline bool SprjDirWindApplyNative(const std::uint8_t* self, const std::uint8_t* sim, float* forces,
                                   const float* dir_vec, const float* params, float sdt,
                                   std::uint32_t seed0, std::uint32_t seed1, float smooth_t,
                                   float frames, bool smooth_turbulence = true) noexcept {
    if (!self || !sim || !forces || !dir_vec || !params) {
        return true;
    }

    const float wx = dir_vec[0];
    const float wy = dir_vec[1];
    const float wz = dir_vec[2];
    const float w2 = wx * wx + wy * wy + wz * wz;
    const float wind_speed = (w2 > 0.0f && std::isfinite(w2)) ? std::sqrt(w2) : 0.0f;
    const float inv_ws = (wind_speed > 1e-12f) ? (1.0f / wind_speed) : 0.0f;
    const float dir[3] = {wx * inv_ws, wy * inv_ws, wz * inv_ws};

    const float min_mul = params[0];
    const float max_mul = params[1];
    const float wind_phase_deg = params[2];
    const float spatial_freq = params[3];

    const std::int32_t num_particles =
        *reinterpret_cast<const std::int32_t*>(sim + HclSimClothInstanceOffsets::kNumParticles);
    if (num_particles <= 0) {
        return true;
    }
    const std::uint8_t* sim_data =
        *reinterpret_cast<const std::uint8_t* const*>(sim + HclSimClothInstanceOffsets::kSimClothData);
    if (!sim_data) {
        return true;
    }
    const float total_mass =
        *reinterpret_cast<const float*>(sim_data + HclSimClothDataOffsets::kTotalMass);
    if (!(total_mass > 1e-12f && std::isfinite(total_mass))) {
        return true;
    }
    const float inv_total_mass = 1.0f / total_mass;

    const float* particle_datas =
        *reinterpret_cast<const float* const*>(sim_data + HclSimClothDataOffsets::kParticleDatasData);
    const float* cur_pos =
        *reinterpret_cast<const float* const*>(sim + HclSimClothInstanceOffsets::kCurrentPositions);
    const float* prev_pos =
        *reinterpret_cast<const float* const*>(sim + HclSimClothInstanceOffsets::kPreviousPositions);
    if (!particle_datas || !cur_pos || !prev_pos) {
        return true;
    }

    const std::int32_t num_normals =
        *reinterpret_cast<const std::int32_t*>(sim + HclSimClothInstanceOffsets::kParticleNormalsSize);
    const float* normals =
        *reinterpret_cast<const float* const*>(sim + HclSimClothInstanceOffsets::kParticleNormalsData);
    const bool has_normals = (num_normals == num_particles && normals != nullptr);

    const float drag_coeff = *reinterpret_cast<const float*>(self + SprjWindActionOffsets::kDragCoeff);
    const float cull_step = (self[SprjWindActionOffsets::kDirCullEnabled] != 0)
                                ? *reinterpret_cast<const float*>(self + SprjWindActionOffsets::kDirCullStep)
                                : 0.0f;

    const float safe_sdt = (sdt > 1e-5f && std::isfinite(sdt)) ? sdt : (1.0f / 60.0f);
    float inv_vel_dt = 1.0f / safe_sdt;
    float disp_30_scale = 1.0f;
    float st = std::clamp(smooth_t, 0.0f, 1.0f);
    if (smooth_turbulence) {
        const float raw_prev_sdt =
            *reinterpret_cast<const float*>(sim + HclSimClothInstanceOffsets::kPrevSubstepDt);
        const float vel_dt =
            (raw_prev_sdt > 1e-5f && raw_prev_sdt < 0.2f && std::isfinite(raw_prev_sdt))
                ? raw_prev_sdt
                : safe_sdt;
        inv_vel_dt = 1.0f / vel_dt;
        const float clamped_frames = std::clamp(frames, 1.0f / 300.0f, 1.0f);
        const float sdt30_equiv = safe_sdt / clamped_frames;
        disp_30_scale = std::clamp(sdt30_equiv * inv_vel_dt, 0.0f, 16.0f);
    } else {
        // Legacy un-hooked 1000 Hz behavior for A/B comparison (`BB_CLOTH_SEED=0`).
        const float rem = std::fmod(wind_phase_deg, 1000.0f);
        const std::int32_t s_i = static_cast<std::int32_t>(rem * 1000.0f);
        seed0 = seed1 = static_cast<std::uint32_t>(s_i);
        st = 0.0f;
    }

    ClothXorShift128 rng0{};
    ClothXorShift128 rng1{};
    rng0.Seed(seed0);
    rng1.Seed(seed1);

    constexpr float kNegPiF = -3.14159274f; // eboot.bin+0x48cdf90 (0xc0490fdb)
    float cached_f[3] = {0.0f, 0.0f, 0.0f};

    for (std::int32_t i = 0; i < num_particles; ++i) {
        bool eval_particle = (i == 0) || (i == num_particles - 1);
        if (!eval_particle) {
            float int_part = 0.0f;
            const float frac_part = std::modf(cull_step * static_cast<float>(i), &int_part);
            eval_particle = (1.0f > (cull_step + frac_part));
        }

        if (eval_particle) {
            const float* cp = cur_pos + std::size_t(i) * 4;
            const float* pp = prev_pos + std::size_t(i) * 4;
            const float dx = cp[0] - pp[0];
            const float dy = cp[1] - pp[1];
            const float dz = cp[2] - pp[2];

            float w[6]{};
            if (st <= 1e-6f) {
                for (int k = 0; k < 6; ++k) {
                    w[k] = rng0.NextFloat01();
                }
            } else {
                for (int k = 0; k < 6; ++k) {
                    const float a = rng0.NextFloat01();
                    const float b = rng1.NextFloat01();
                    w[k] = a + st * (b - a);
                }
            }

            const float turb_x = (dx * disp_30_scale + (w[0] * 0.2f + 0.05f)) * (w[1] * 0.6f - 0.1f);
            const float turb_y = (dy * disp_30_scale + (w[2] * 0.2f + 0.05f)) * (w[3] * 0.6f - 0.1f);
            const float turb_z = (dz * disp_30_scale + (w[4] * 0.2f + 0.05f)) * (w[5] * 0.6f - 0.1f);

            const float dist = std::sqrt(cp[0] * cp[0] + cp[1] * cp[1] + cp[2] * cp[2]);
            const float angle = (wind_phase_deg + spatial_freq * dist) / 180.0f + kNegPiF;
            const float wave_w = std::clamp(0.5f * std::sin(angle) + 0.5f, 0.0f, 1.0f);
            const float speed_i = ((1.0f - wave_w) * min_mul + wave_w * max_mul) * wind_speed;

            const float vx_rel = dir[0] * speed_i + turb_x - dx * inv_vel_dt;
            const float vy_rel = dir[1] * speed_i + turb_y - dy * inv_vel_dt;
            const float vz_rel = dir[2] * speed_i + turb_z - dz * inv_vel_dt;

            float normal_fac = 0.7f;
            if (has_normals) {
                const float* np = normals + std::size_t(i) * 4;
                normal_fac = std::fabs(dir[0] * np[0] + dir[1] * np[1] + dir[2] * np[2]);
            }

            const float mass_ratio = particle_datas[std::size_t(i) * 4] * inv_total_mass;
            const float scale = (drag_coeff * normal_fac) * mass_ratio;
            cached_f[0] = scale * vx_rel;
            cached_f[1] = scale * vy_rel;
            cached_f[2] = scale * vz_rel;
        }

        float* fp = forces + std::size_t(i) * 4;
        fp[0] += cached_f[0];
        fp[1] += cached_f[1];
        fp[2] += cached_f[2];
    }
    return true;
}

/// Multiplies two column-major 4x4 affine `hkTransform` matrices (`out = parent * local`),
/// matching `hclSimulateOperator::simulate` (`0x76e533..0x76e603` in `eboot.bin`) bit-for-bit.
inline void MulAffineTransform4x4(const float* parent, const float* local, float* out) noexcept {
    for (int col = 0; col < 3; ++col) {
        const float lx = local[col * 4 + 0];
        const float ly = local[col * 4 + 1];
        const float lz = local[col * 4 + 2];
        for (int row = 0; row < 4; ++row) {
            out[col * 4 + row] =
                parent[0 * 4 + row] * lx + parent[1 * 4 + row] * ly + parent[2 * 4 + row] * lz;
        }
    }
    const float tx = local[12];
    const float ty = local[13];
    const float tz = local[14];
    for (int row = 0; row < 4; ++row) {
        out[12 + row] = parent[0 * 4 + row] * tx + parent[1 * 4 + row] * ty +
                        parent[2 * 4 + row] * tz + parent[12 + row];
    }
}

} // namespace BbClothInterp
