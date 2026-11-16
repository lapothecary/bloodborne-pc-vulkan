// SPDX-License-Identifier: GPL-2.0-or-later
// Sealed C++20 ChrExFollowCam layout & 3-mode camera follow physics:
//   0 = Original (PS4): authentic Bloodborne 30-FPS camera inertia scaled continuously by dt.
//   1 = Balanced: Lies of P follow physics calibrated for Bloodborne (LockCamParam arm = 4.00..4.50m,
//       fovY = 43.0 deg; c0000 hkx speeds: jog = 5.000 m/s, sprint = 6.478 m/s):
//       - Direct inertial world-space XZ position spring (ULSpringArmComponent VInterpTo kMove = 7.0 s^-1,
//         kReturn = 4.4 s^-1, leash = 0.85m -> 0.00s glued jerk on A/D tap, camera stays calm in space
//         from frame 1 while hunter smoothly steps off-center up to ~0.71..0.85m = ~18.7% NDC)
//         + Y stair/cobblestone position low-pass (10.0 s^-1).
//       - Critically-damped 2-pole (parabolic y ~ t^2) radial arm boost (wArmIn = 5.4 s^-1 -> silky
//         kickoff, ~1.67m jog / ~2.34m sprint total pull-back) + crisp ~0.55s return on stop
//         (kArmRet = 4.2 s^-1, no multi-second tail), 100% mouse-decoupled.
//   2 = Locked (Rigid 1:1): zero-lag 1:1 character pivot attachment (PoE / rigid style).
//
// Verified against original eboot.bin (file_offset = vma + 0x4000):
//   ChrExFollowCam::Reset  @ 0x143a3e0..0x143ac5f
//   ChrExFollowCam::Update @ 0x143ac60..0x143fad0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace BbCameraFollow {

enum Mode : int {
    ModeOriginal = 0,
    ModeBalanced = 1,
    ModeLocked = 2,
    ModeCount = 3,
};

inline constexpr std::uint64_t kUpdateEntryVma = 0x143ac60ull;
inline constexpr std::uint64_t kPivotSiteVma = 0x143ba73ull;
inline constexpr std::uint64_t kPivotResumeVma = 0x143bab1ull; // kPivotSiteVma + 62

// Sealed layout of Bloodborne's ChrExFollowCam (verified from 0x143a3e0..0x143f0a0).
struct ChrExFollowCamLayout {
    std::uint8_t pad_000[0x50];
    float fov_rad;                    // +0x050: vertical FOV in radians (blended at 0x143af5e..0x143af70)
    std::uint8_t pad_054[0x0c];
    float basis_right[4];             // +0x060: camera right basis (written at 0x143ba5e)
    float basis_up[4];                // +0x070: camera up basis (written at 0x143ba64)
    float basis_back[4];              // +0x080: camera back basis (written at 0x143ba6a)
    float pivot_xyz[4];               // +0x090: smoothed 3D follow pivot (written at 0x143ba73 / 0x143baa8)
    float raw_target_with_ofs[4];     // +0x0a0: raw player + socket offset (written at 0x143bae1)
    float smooth_pivot_with_ofs[4];   // +0x0b0: smoothed pivot + socket offset (written at 0x143baf6)
    float lookat_pivot[4];            // +0x0c0: active look-at pivot (0x143e88f..0x143e980)
    float orbit_focus[4];             // +0x0d0: orbit focus point (0x143c636)
    float eye_shift_origin[4];        // +0x0e0: shifted previous eye origin (0x143bcac..0x143bcb5)
    float target_eye_xyz[4];          // +0x0f0: unblended target eye position (0x143e7da)
    float final_eye_xyz[4];           // +0x100: final world-space camera eye position (0x143e978)
    std::uint8_t pad_110[0x20];
    float rigidity_blend;             // +0x130: eye-lerp bypass (1.0f = immediate target_eye, 0x143e7e6)
    float rigidity_hold_sec;          // +0x134: manual-look rigidity hold timer (0x143c949)
    std::uint8_t pad_138[0x04];
    std::uint8_t reset_cut_flag;      // +0x13c: non-zero on camera cut / warp / init (0x143ba7c)
    std::uint8_t pad_13d[0x03];
    float angle_pitch;                // +0x140: camera pitch (radians)
    float angle_yaw;                  // +0x144: camera yaw (radians)
    std::uint8_t pad_148[0x14];
    std::uint8_t has_manual_look;     // +0x15c: 1 when stick/mouse turned camera this frame (0x143b8cc)
    std::uint8_t pad_15d[0x03];
    float local_socket_offset[4];     // +0x160: LockCamParam local offset (Y blended at 0x143aee1)
    std::uint8_t pad_170[0x14];
    float arm_length_cur;             // +0x184: current arm distance (default 4.0f, 0x143af76 / 0x143d545)
    float arm_length_target;          // +0x188: target arm distance from LockCamParam (default 4.0f, 0x143ae6c)
    std::uint8_t pad_18c[0x0c];
    float pivot_lerp_cur;             // +0x198: active 3D pivot lerp factor (PS4 default 0.10f @ 30 FPS)
    float pivot_lerp_target;          // +0x19c: normal 3D pivot lerp target (PS4 default 0.10f @ 30 FPS)
    float pivot_lerp_collision;       // +0x1a0: obstacle pull-in pivot lerp (PS4 default 0.25f @ 30 FPS)
    float eye_shift_follow_ratio;     // +0x1a4: +0xe0 translation follow ratio (PS4 default 0.30f, 0x143bc79)
    float pitch_restore_lerp;         // +0x1a8: passive pitch restore lerp (PS4 default 0.10f @ 30 FPS, 0x143cef0)
    float lock_lerp_cur;              // +0x1ac: secondary lock lerp current (PS4 default 0.10f @ 30 FPS)
    float lock_lerp_target;           // +0x1b0: secondary lock lerp target (PS4 default 0.10f @ 30 FPS)
    float lock_lerp_collision;        // +0x1b4: secondary lock collision lerp (PS4 default 0.25f @ 30 FPS)
    std::uint8_t pad_1b8[0x0c];
    float tether_dist_lerp;           // +0x1c4: radial arm-distance lerp (1.0f = exact desired_dist, 0x143e964)
    std::uint8_t pad_1c8[0x6c];
    float screen_offset_lerp_sqrt;    // +0x234: squared at 0x143afaa (PS4 default 0.50f -> 0.25f @ 30 FPS)
    std::uint8_t pad_238[0x29];
    std::uint8_t is_locked_on;        // +0x261: 0 = free follow, != 0 = locked onto target (0x143acb1)
    std::uint8_t pad_262[0xce];
    float param_blend_lerp;           // +0x330: LockCamParam transition lerp (PS4 default 0.05f @ 30 FPS)
};

static_assert(offsetof(ChrExFollowCamLayout, fov_rad) == 0x50);
static_assert(offsetof(ChrExFollowCamLayout, pivot_xyz) == 0x90);
static_assert(offsetof(ChrExFollowCamLayout, lookat_pivot) == 0xc0);
static_assert(offsetof(ChrExFollowCamLayout, orbit_focus) == 0xd0);
static_assert(offsetof(ChrExFollowCamLayout, eye_shift_origin) == 0xe0);
static_assert(offsetof(ChrExFollowCamLayout, target_eye_xyz) == 0xf0);
static_assert(offsetof(ChrExFollowCamLayout, final_eye_xyz) == 0x100);
static_assert(offsetof(ChrExFollowCamLayout, rigidity_blend) == 0x130);
static_assert(offsetof(ChrExFollowCamLayout, rigidity_hold_sec) == 0x134);
static_assert(offsetof(ChrExFollowCamLayout, reset_cut_flag) == 0x13c);
static_assert(offsetof(ChrExFollowCamLayout, angle_pitch) == 0x140);
static_assert(offsetof(ChrExFollowCamLayout, has_manual_look) == 0x15c);
static_assert(offsetof(ChrExFollowCamLayout, arm_length_cur) == 0x184);
static_assert(offsetof(ChrExFollowCamLayout, arm_length_target) == 0x188);
static_assert(offsetof(ChrExFollowCamLayout, pivot_lerp_cur) == 0x198);
static_assert(offsetof(ChrExFollowCamLayout, pivot_lerp_target) == 0x19c);
static_assert(offsetof(ChrExFollowCamLayout, pivot_lerp_collision) == 0x1a0);
static_assert(offsetof(ChrExFollowCamLayout, eye_shift_follow_ratio) == 0x1a4);
static_assert(offsetof(ChrExFollowCamLayout, pitch_restore_lerp) == 0x1a8);
static_assert(offsetof(ChrExFollowCamLayout, lock_lerp_cur) == 0x1ac);
static_assert(offsetof(ChrExFollowCamLayout, lock_lerp_target) == 0x1b0);
static_assert(offsetof(ChrExFollowCamLayout, lock_lerp_collision) == 0x1b4);
static_assert(offsetof(ChrExFollowCamLayout, tether_dist_lerp) == 0x1c4);
static_assert(offsetof(ChrExFollowCamLayout, screen_offset_lerp_sqrt) == 0x234);
static_assert(offsetof(ChrExFollowCamLayout, is_locked_on) == 0x261);
static_assert(offsetof(ChrExFollowCamLayout, param_blend_lerp) == 0x330);

/// Exact continuous equivalent of a per-30-FPS-frame lerp factor `alpha30` over `dt` seconds.
inline float DtLerp30(float alpha30, float dt) noexcept {
    const float clamped = std::clamp(alpha30, 0.0f, 0.9999f);
    return 1.0f - std::pow(1.0f - clamped, dt * 30.0f);
}

/// FMath::VInterpTo (RVA 0x021dd370) in Bloodborne meter units (1e-10 m^2 = 0.01 mm threshold).
inline void VInterpTo3D(float (&cur)[3], const float (&target)[3], float dt, float speed) noexcept {
    if (speed <= 0.0f) {
        cur[0] = target[0];
        cur[1] = target[1];
        cur[2] = target[2];
        return;
    }
    const float dx = target[0] - cur[0];
    const float dy = target[1] - cur[1];
    const float dz = target[2] - cur[2];
    if (dx * dx + dy * dy + dz * dz < 1e-10f) {
        cur[0] = target[0];
        cur[1] = target[1];
        cur[2] = target[2];
        return;
    }
    const float alpha = std::clamp(dt * speed, 0.0f, 1.0f);
    cur[0] += dx * alpha;
    cur[1] += dy * alpha;
    cur[2] += dz * alpha;
}

/// Spherical 3D leash clamp (ULSpringArmComponent @ RVA 0x01b67954: CameraLagMaxDistance = 1.20m).
inline void ClampSphericalLeash(float (&cur)[3], const float (&target)[4], float max_dist) noexcept {
    const float fx = cur[0] - target[0];
    const float fy = cur[1] - target[1];
    const float fz = cur[2] - target[2];
    const float f_sq = fx * fx + fy * fy + fz * fz;
    if (f_sq > max_dist * max_dist) {
        const float scale = max_dist / std::sqrt(f_sq);
        cur[0] = target[0] + fx * scale;
        cur[1] = target[1] + fy * scale;
        cur[2] = target[2] + fz * scale;
    }
}

struct RuntimeState {
    ChrExFollowCamLayout* last_cam = nullptr;
    float current_dt = 1.0f / 60.0f;
    int last_mode = -1;

    // Relative offset (PreviousArmOrigin - PreviousDesiredLoc) from the previous frame,
    // invariant to Bloodborne's map-sector world-origin rebases ([rbp - 0x260] @ 0x143b654)
    // and moving platforms / elevators ([rbp - 0x270] @ 0x143b633).
    bool has_prev_origin_offset = false;
    float prev_origin_minus_pivot[3]{0.0f, 0.0f, 0.0f};
    float filtered_origin_vel[3]{0.0f, 0.0f, 0.0f};

    // 2-pole critically-damped (y ~ t^2) radial arm-boost integrators:
    float arm_boost_s1 = 0.0f;
    float arm_boost_state = 0.0f;

    // Arm extension applied in ComputePivot (after 0x143ae8a, before 0x143bc67) and restored
    // bit-exactly in PreUpdate (before 0x143ae6c) so LockCamParam transitions never drift
    // while Havok wall collision raycasts the full extended arm.
    float applied_arm_boost = 0.0f;
    float last_boosted_arm_target = 0.0f;
    float log_accum_sec = 0.0f;

    void ResetSpringState() noexcept {
        filtered_origin_vel[0] = 0.0f;
        filtered_origin_vel[1] = 0.0f;
        filtered_origin_vel[2] = 0.0f;
        arm_boost_s1 = 0.0f;
        arm_boost_state = 0.0f;
    }

    void PreUpdate(std::uint8_t* raw_cam, float raw_dt, bool mouse_look_enabled, int mode) noexcept {
        if (!raw_cam) {
            return;
        }
        auto* cam = reinterpret_cast<ChrExFollowCamLayout*>(raw_cam);
        const float dt = (std::isfinite(raw_dt) && raw_dt > 0.0f)
                             ? std::clamp(raw_dt, 1.0f / 2000.0f, 0.10f)
                             : (1.0f / 60.0f);
        current_dt = dt;

        // Restore pure LockCamParam arm_length before 0x143ae6c / 0x143af76 blend it.
        if (applied_arm_boost > 0.0f) {
            if (cam == last_cam && cam->reset_cut_flag == 0 &&
                std::isfinite(cam->arm_length_target) &&
                std::fabs(cam->arm_length_target - last_boosted_arm_target) < 1e-3f) {
                cam->arm_length_target = std::max(0.5f, cam->arm_length_target - applied_arm_boost);
                cam->arm_length_cur = std::max(0.5f, cam->arm_length_cur - applied_arm_boost);
            }
            applied_arm_boost = 0.0f;
            last_boosted_arm_target = 0.0f;
        }

        if (cam != last_cam) {
            last_cam = cam;
            has_prev_origin_offset = false;
            ResetSpringState();
        }

        const float lerp_05 = DtLerp30(0.05f, dt);
        const float lerp_10 = DtLerp30(0.10f, dt);
        const float lerp_25 = DtLerp30(0.25f, dt);

        // LockCamParam & screen-space framing transitions run at exact PS4 real-time speed.
        cam->param_blend_lerp = lerp_05;
        cam->screen_offset_lerp_sqrt = std::sqrt(lerp_25);

        cam->pivot_lerp_target = lerp_10;
        cam->pivot_lerp_collision = lerp_25;
        cam->lock_lerp_target = lerp_10;
        cam->lock_lerp_collision = lerp_25;
        cam->pitch_restore_lerp = lerp_10;
        if (last_mode != mode || cam->reset_cut_flag != 0 || cam->pivot_lerp_cur > lerp_25 * 1.2f) {
            cam->pivot_lerp_cur = lerp_10;
            cam->lock_lerp_cur = lerp_10;
            has_prev_origin_offset = false;
            if (cam->reset_cut_flag != 0) {
                ResetSpringState();
            } else {
                // Smooth mid-motion F4 mode switch: clear velocity spike, seed XZ lag in ComputePivot.
                filtered_origin_vel[0] = 0.0f;
                filtered_origin_vel[1] = 0.0f;
                filtered_origin_vel[2] = 0.0f;
            }
        }
        last_mode = mode;

        // Prevent passive pitch bobbing on frames between mouse reports:
        // At 0x143bc79..0x143bcb5, eye_shift_origin (+0xe0) translates by
        // (eye_shift_follow_ratio + (1 - eye_shift_follow_ratio) * [r13 + 0x154]) * delta_pivot.
        // With mouse_look_enabled == true, setting +0x1a4 = 1.0f translates +0xe0 1:1 with +0xd0
        // on every frame, so 0x143c6af never tilts camera pitch during movement.
        cam->eye_shift_follow_ratio = mouse_look_enabled ? 1.0f : 0.30f;

        // Keep radial arm-distance lerp (+0x1c4 @ 0x143e964) and eye-rigidity (+0x130 @ 0x143e7e6)
        // permanently locked at 1.0f across all modes and mouse states:
        //   - Mouse rotation has 0.00% effect on camera distance or spring inertia.
        //   - No radial carousel or snap-back can ever occur when turning the mouse in motion.
        cam->rigidity_blend = 1.0f;
        cam->rigidity_hold_sec = 2.0f;
        cam->tether_dist_lerp = 1.0f;
    }

    void ComputePivot(std::uint8_t* raw_cam, const float* prev_pivot_rebased,
                      float* xmm_save_area, int mode) noexcept {
        if (!raw_cam || !prev_pivot_rebased || !xmm_save_area) {
            return;
        }
        auto* cam = reinterpret_cast<ChrExFollowCamLayout*>(raw_cam);
        const float dt = current_dt;

        // xmm0 in SaveXmm area (floats 0..3) = raw ArmOrigin (CSChrPhysicsModule + 0x1e0, w = 1.0f).
        const float target[4] = {
            xmm_save_area[0],
            xmm_save_area[1],
            xmm_save_area[2],
            xmm_save_area[3],
        };
        // PreviousDesiredLoc rebased to the current map sector ([rbp - 0x260] @ 0x143b654).
        const float prev[3] = {
            prev_pivot_rebased[0],
            prev_pivot_rebased[1],
            prev_pivot_rebased[2],
        };

        float out[4] = {target[0], target[1], target[2], target[3]};
        const float lag_dx = target[0] - prev[0];
        const float lag_dy = target[1] - prev[1];
        const float lag_dz = target[2] - prev[2];
        const float lag_dist_sq = lag_dx * lag_dx + lag_dy * lag_dy + lag_dz * lag_dz;
        float diag_chr_speed = 0.0f;
        const float base_arm_before_boost = cam->arm_length_target;

        if (mode == ModeLocked || cam->reset_cut_flag != 0) {
            // Mode 2: 1:1 rigid character lock (0 pixel lag) or camera cut reset.
            out[0] = target[0];
            out[1] = target[1];
            out[2] = target[2];
            ResetSpringState();
        } else if (mode == ModeOriginal) {
            // Mode 0: Authentic PS4 Bloodborne 30-FPS camera inertia scaled continuously by dt.
            if (lag_dist_sq > 25.0f) {
                out[0] = target[0];
                out[1] = target[1];
                out[2] = target[2];
                ResetSpringState();
            } else {
                const float alpha_xz = DtLerp30(0.10f, dt);
                const float alpha_y = DtLerp30(0.08f, dt);
                float cur[3] = {
                    prev[0] + lag_dx * alpha_xz,
                    prev[1] + lag_dy * alpha_y,
                    prev[2] + lag_dz * alpha_xz,
                };
                ClampSphericalLeash(cur, target, 1.10f);
                out[0] = cur[0];
                out[1] = cur[1];
                out[2] = cur[2];
                arm_boost_s1 = 0.0f;
                arm_boost_state = 0.0f;
            }
        } else {
            // Mode 1 (Default): Balanced — Lies of P camera feel calibrated for Bloodborne:
            //   1. Direct physical world-space inertia on XZ (kPivotMoveXZ = 7.0 s^-1,
            //      kPivotReturnXZ = 4.4 s^-1, kPivotLeashXZ = 0.85m):
            //      On A/D or W+A/W+D press, the camera has real inertial mass from t = 0.000s
            //      (0% glued camera jerk; hunter smoothly steps off-center right away up to
            //      ~0.71..0.85m = ~18.7% NDC on A/D, ~0.50..0.60m = ~12.1% NDC on W+D).
            //   2. Silky quadratic (y ~ t^2) radial arm boost (wArmIn = 5.4 s^-1):
            //      2-pole critically-damped longitudinal zoom-out (~1.67m jog / ~2.34m sprint
            //      total pull-back) with zero harsh kickoff.
            //   3. Crisp ~0.55s velvet return on stop (kArmRet = 4.2 s^-1, kPivotRet = 4.4 s^-1):
            //      no 3-second lingering tail after releasing W + Sprint.
            //   4. Cobblestone / stair smoothing on vertical Y (kPivotY = 10.0 s^-1) and 0.00%
            //      mouse yaw/pitch coupling.
            if (lag_dist_sq > 25.0f) {
                out[0] = target[0];
                out[1] = target[1];
                out[2] = target[2];
                ResetSpringState();
            } else {
                constexpr float kPivotMoveXZ = 7.0f;
                constexpr float kPivotReturnXZ = 4.4f;
                constexpr float kPivotLeashXZ = 0.85f;
                constexpr float kPivotY = 10.0f;
                constexpr float kPivotLeashY = 0.65f;

                constexpr float kArmOmegaIn = 5.4f;
                constexpr float kStage1Return = 12.0f;
                constexpr float kArmReturn = 4.2f;
                constexpr float kMaxArmBoost = 1.65f;
                constexpr float kVelFilterTau = 0.035f;

                const float l0[3] = {
                    has_prev_origin_offset ? prev_origin_minus_pivot[0] : lag_dx,
                    has_prev_origin_offset ? prev_origin_minus_pivot[1] : lag_dy,
                    has_prev_origin_offset ? prev_origin_minus_pivot[2] : lag_dz,
                };
                const float delta_o[3] = {
                    has_prev_origin_offset ? (lag_dx - l0[0]) : 0.0f,
                    has_prev_origin_offset ? (lag_dy - l0[1]) : 0.0f,
                    has_prev_origin_offset ? (lag_dz - l0[2]) : 0.0f,
                };
                const float delta_o_sq =
                    delta_o[0] * delta_o[0] + delta_o[1] * delta_o[1] + delta_o[2] * delta_o[2];

                if (!std::isfinite(delta_o_sq) || delta_o_sq > 4.0f) {
                    out[0] = target[0];
                    out[1] = target[1];
                    out[2] = target[2];
                    ResetSpringState();
                } else {
                    const float inv_dt = 1.0f / dt;
                    const float inst_vel[3] = {
                        delta_o[0] * inv_dt,
                        delta_o[1] * inv_dt,
                        delta_o[2] * inv_dt,
                    };
                    const float alpha_v = 1.0f - std::exp(-dt / kVelFilterTau);
                    filtered_origin_vel[0] += (inst_vel[0] - filtered_origin_vel[0]) * alpha_v;
                    filtered_origin_vel[1] += (inst_vel[1] - filtered_origin_vel[1]) * alpha_v;
                    filtered_origin_vel[2] += (inst_vel[2] - filtered_origin_vel[2]) * alpha_v;

                    const float vx = filtered_origin_vel[0];
                    const float vz = filtered_origin_vel[2];
                    const float v_horiz = std::sqrt(vx * vx + vz * vz);
                    diag_chr_speed = v_horiz;

                    // Vertical Y: position low-pass so stairs & cobblestones stay silky smooth.
                    const float uy = kPivotY * dt;
                    const float dec_y = std::exp(-uy);
                    const float phi1_y =
                        (uy < 1e-4f) ? (1.0f - 0.5f * uy + (uy * uy) * (1.0f / 6.0f))
                                     : ((1.0f - dec_y) / uy);
                    const float raw_cur_y = target[1] - (l0[1] * dec_y + delta_o[1] * phi1_y);
                    const float cur_y =
                        std::clamp(raw_cur_y, target[1] - kPivotLeashY, target[1] + kPivotLeashY);

                    // Horizontal XZ: true inertial world-space position spring (ULSpringArmComponent
                    // VInterpTo). Camera stays calm in space at t = 0+ (zero 0.5s glued jerk on A/D!)
                    // while hunter smoothly moves off-center up to 0.71..0.85m, and glides back at
                    // kPivotReturnXZ when decelerating/stopping.
                    float k_xz = kPivotMoveXZ;
                    const float l0_xz_sq = l0[0] * l0[0] + l0[2] * l0[2];
                    if (l0_xz_sq > 1e-8f) {
                        const float inv_l0_xz = 1.0f / std::sqrt(l0_xz_sq);
                        const float l0_xz_len = l0_xz_sq * inv_l0_xz;
                        const float filt_v_along = (vx * l0[0] + vz * l0[2]) * inv_l0_xz;
                        const float inst_v_along =
                            (inst_vel[0] * l0[0] + inst_vel[2] * l0[2]) * inv_l0_xz;
                        const float active_v_along =
                            (inst_v_along * filt_v_along > 0.0f &&
                             std::fabs(inst_v_along) > std::fabs(filt_v_along))
                                ? inst_v_along
                                : filt_v_along;
                        const float drive =
                            std::fabs(active_v_along) / (l0_xz_len * kPivotMoveXZ);
                        const float blend = std::clamp(drive, 0.0f, 1.0f);
                        k_xz = kPivotReturnXZ + (kPivotMoveXZ - kPivotReturnXZ) * blend;
                    }

                    const float u_xz = k_xz * dt;
                    const float dec_xz = std::exp(-u_xz);
                    const float phi1_xz =
                        (u_xz < 1e-4f) ? (1.0f - 0.5f * u_xz + (u_xz * u_xz) * (1.0f / 6.0f))
                                       : ((1.0f - dec_xz) / u_xz);
                    float new_lx = l0[0] * dec_xz + delta_o[0] * phi1_xz;
                    float new_lz = l0[2] * dec_xz + delta_o[2] * phi1_xz;
                    const float new_l_sq = new_lx * new_lx + new_lz * new_lz;
                    if (new_l_sq > kPivotLeashXZ * kPivotLeashXZ) {
                        const float sc = kPivotLeashXZ / std::sqrt(new_l_sq);
                        new_lx *= sc;
                        new_lz *= sc;
                    }

                    out[0] = target[0] - new_lx;
                    out[1] = cur_y;
                    out[2] = target[2] - new_lz;

                    // Radial arm boost: 2-pole critically-damped (y ~ t^2) on expansion,
                    // crisp ~0.55s 1-pole exponential return on stop.
                    const float target_boost =
                        std::min(kMaxArmBoost, 0.064f * v_horiz + 0.0257f * v_horiz * v_horiz);
                    const float ka1 = (target_boost >= arm_boost_s1) ? kArmOmegaIn : kStage1Return;
                    const float ka2 =
                        (arm_boost_s1 >= arm_boost_state) ? kArmOmegaIn : kArmReturn;
                    arm_boost_s1 += (target_boost - arm_boost_s1) * (1.0f - std::exp(-ka1 * dt));
                    arm_boost_state +=
                        (arm_boost_s1 - arm_boost_state) * (1.0f - std::exp(-ka2 * dt));

                    if (std::isfinite(arm_boost_state) && arm_boost_state > 1e-5f &&
                        std::isfinite(cam->arm_length_target) && cam->arm_length_target > 0.5f) {
                        cam->arm_length_target += arm_boost_state;
                        applied_arm_boost = arm_boost_state;
                        last_boosted_arm_target = cam->arm_length_target;
                    }
                }
            }
        }

        prev_origin_minus_pivot[0] = target[0] - out[0];
        prev_origin_minus_pivot[1] = target[1] - out[1];
        prev_origin_minus_pivot[2] = target[2] - out[2];
        has_prev_origin_offset = true;

        static const bool s_log_cam = [] {
            const char* env = std::getenv("BB_LOG_CAMERA_FOLLOW");
            return env && env[0] == '1';
        }();
        if (s_log_cam) {
            log_accum_sec += dt;
            if (log_accum_sec >= 0.25f) {
                log_accum_sec = 0.0f;
                const float lx = prev_origin_minus_pivot[0];
                const float ly = prev_origin_minus_pivot[1];
                const float lz = prev_origin_minus_pivot[2];
                const float l_norm = std::sqrt(lx * lx + ly * ly + lz * lz);
                std::printf(
                    "[BB-CAM] mode=%d dt=%.4fs chr_v=%.2fm/s ||L||=%.3fm arm_boost=%.3fm "
                    "total_pull=%.3fm base_arm=%.2fm\n",
                    mode, dt, diag_chr_speed, l_norm, applied_arm_boost,
                    l_norm + applied_arm_boost, base_arm_before_boost);
            }
        }

        // Store smoothed pivot to [r13 + 0x90] AND to xmm4 (offset 4 * 4 = 16 floats in SaveXmm).
        std::memcpy(cam->pivot_xyz, out, sizeof(out));
        std::memcpy(xmm_save_area + 16, out, sizeof(out));
    }
};

} // namespace BbCameraFollow
