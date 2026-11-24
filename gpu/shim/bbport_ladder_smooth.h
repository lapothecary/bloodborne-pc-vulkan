// SPDX-License-Identifier: GPL-2.0-or-later
// Sealed C++20 visual Y-smoother for Bloodborne ladder traversal (sliding down & climbing).
//
// Root cause (verified from eboot.bin + per-frame runtime telemetry):
//   Bloodborne implements ladder climbing (states 2/3, 0.30s, +0.50m/rung) and fast ladder
//   sliding (states 4/5 entered from 14/17, 0.10s, -0.50m/rung) as non-looping Havok clips
//   chained via HKS state transitions (`4 -> 5 -> 4 -> 5`). Because HKS detects clip completion
//   one frame after `hkbClipGenerator` reaches `localTime == duration`, every single rung ends
//   with a full 1-frame dead stop (`rmY == 0.000 m/s`: 33.3 ms @ 30 FPS, 16.7 ms @ 60 FPS,
//   5.9 ms @ 170 FPS), plus a fractional last-frame velocity drop whenever `0.100s / dt` is not
//   an exact integer (e.g. 144 FPS).
//
// Solution (zero physics risk, exact -0.50000m rung grid preserved):
//   Never modifies `CSChrPhysicsModule + 0x1e0` (Havok proxy, collision, rung grid, and top/bottom
//   exits remain 100% untouched). Instead, hooks `ChrCtrl::UpdateWorldMatrix` (`sub_15132e0` at
//   VMA `0x151336c`: `vmovaps [rbp - 0xb0], xmm1`, 8 bytes) strictly for the local player and adds
//   a causal visual Y offset to the render transform stack copy (`[rbp - 0xb0]`) AND to
//   `ChrExFollowCam::ComputePivot` (`target[1]`) so hero, skeleton, cloth, weapon, shadow, and
//   camera move together as one smooth unit at any frame rate (30..500 FPS):
//     - Fast slide down (`st 4/5` at >2.8 m/s or entered from `14/17`, plus `15/18`):
//       Rung-boundary Phase-Locked Loop (PLL) that advances `vis_y` at constant velocity across
//       each rung, locking phase strictly on `st` flips (zero sawtooth noise) and landing within
//       ~1..3 mm of `pos_y` when stopping or reaching the bottom.
//     - Climbing up / slow climb down (`st 2/3` or slow `st 4/5`):
//       Gentle causal 1-frame dead-stop bridge preserving natural hand-over-hand cadence while
//       eliminating the 1-frame `0.000 m/s` pause at each rung transition.
//
// Verified against original eboot.bin (file_offset = vma + 0x4000):
//   ChrCtrl::UpdateWorldMatrix     @ 0x15132e0
//   Visual pos stack store site    @ 0x151336c (c5 f8 29 8d 50 ff ff ff -> vmovaps [rbp-0xb0], xmm1)
//   ChrCtrl + 0x38 -> ActionFlags, ActionFlags + 0x60 -> int8_t ladder_state (-1 = off ladder)
//   LadderState enum (0x5335370..0x5335438):
//     -1 = Invalid/Off, 0 = GrabBottom, 1 = GrabTop,
//      2 = ClimbUpRight, 3 = ClimbUpLeft, 4 = ClimbOrSlideDownRight, 5 = ClimbOrSlideDownLeft,
//      6 = ExitTop, 7 = ExitBottom, 8 = IdleRight, 9 = IdleLeft,
//     14 = SlideStartRight, 15 = SlidingRight, 16 = SlideStopRight,
//     17 = SlideStartLeft,  18 = SlidingLeft,  19 = SlideStopLeft, 20 = SlideLand

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace BbLadderSmooth {

inline constexpr std::uint64_t kWorldMatPosSiteVma = 0x151336cull;
inline constexpr std::uint8_t kWorldMatPosCode[8] = {
    0xc5, 0xf8, 0x29, 0x8d, 0x50, 0xff, 0xff, 0xff, // vmovaps [rbp - 0xb0], xmm1
};

enum LadderState : int {
    StateOff = -1,
    StateGrabBottom = 0,
    StateGrabTop = 1,
    StateClimbUpRight = 2,
    StateClimbUpLeft = 3,
    StateDownRight = 4,
    StateDownLeft = 5,
    StateExitTop = 6,
    StateExitBottom = 7,
    StateIdleRight = 8,
    StateIdleLeft = 9,
    StateSlideStartRight = 14,
    StateSlidingRight = 15,
    StateSlideStopRight = 16,
    StateSlideStartLeft = 17,
    StateSlidingLeft = 18,
    StateSlideStopLeft = 19,
    StateSlideLand = 20,
};

inline bool Enabled() noexcept {
    static const bool s_enabled = [] {
        const char* env = std::getenv("BB_LADDER_SMOOTH");
        return !(env && env[0] == '0' && env[1] == '\0');
    }();
    return s_enabled;
}

inline bool TraceEnabled() noexcept {
    static const bool s_trace = [] {
        const char* env = std::getenv("BB_LADDER_TRACE");
        return env && env[0] == '1';
    }();
    return s_trace;
}

struct RuntimeState {
    std::atomic<std::uint8_t*> player_chr_ctrl{nullptr};
    std::atomic<float> cached_offset_y{0.0f};
    std::atomic<std::uint64_t> fallback_seq{1};

    std::uint8_t* last_ctrl = nullptr;
    std::uint64_t last_frame_seq = 0;

    bool have_prev_pos = false;
    float prev_pos_y = 0.0f;
    float vis_y = 0.0f;
    float prev_vis_y = 0.0f;

    // Fast slide down PLL state
    bool slide_armed = false; // entered from SlideStart (14/17) or high speed (>2.8 m/s)
    bool in_fast_slide = false;
    int prev_state = StateOff;
    float v_smooth = 0.0f;
    float rung_start_y = 0.0f;
    float rung_elapsed = 0.0f;
    int rung_count = 0;
    int zero_streak = 0;

    // Climb (up / slow down) 1-frame dead-stop bridge state
    float climb_v_ema = 0.0f;

    // Optional trace counter
    std::uint32_t trace_frame = 0;
    int trace_budget = 0;

    void Reset(float pos_y) noexcept {
        have_prev_pos = true;
        prev_pos_y = pos_y;
        vis_y = pos_y;
        prev_vis_y = pos_y;
        slide_armed = false;
        in_fast_slide = false;
        prev_state = StateOff;
        v_smooth = 0.0f;
        rung_start_y = pos_y;
        rung_elapsed = 0.0f;
        rung_count = 0;
        zero_streak = 0;
        climb_v_ema = 0.0f;
        cached_offset_y.store(0.0f, std::memory_order_relaxed);
    }

    void NotePlayerChr(std::uint8_t* chr) noexcept {
        std::uint8_t* ctrl = chr ? *reinterpret_cast<std::uint8_t**>(chr + 0x58) : nullptr;
        player_chr_ctrl.store(ctrl, std::memory_order_release);
        fallback_seq.fetch_add(1, std::memory_order_relaxed);
    }

    float CurrentOffsetY() const noexcept {
        return cached_offset_y.load(std::memory_order_relaxed);
    }

    float UpdateAndGetOffset(std::uint8_t* chr_ctrl, float pos_y, float raw_dt,
                             std::uint64_t frame_seq) noexcept {
        if (!Enabled() || !chr_ctrl || !std::isfinite(pos_y)) {
            return 0.0f;
        }
        if (chr_ctrl != player_chr_ctrl.load(std::memory_order_acquire)) {
            return 0.0f;
        }

        // Multiple ChrCtrl::UpdateWorldMatrix calls in the same simulation frame share one result.
        const std::uint64_t seq =
            (frame_seq != 0) ? frame_seq : fallback_seq.load(std::memory_order_relaxed);
        if (chr_ctrl == last_ctrl && seq == last_frame_seq) {
            return cached_offset_y.load(std::memory_order_relaxed);
        }

        const float dt = (std::isfinite(raw_dt) && raw_dt > 1e-4f)
                             ? std::clamp(raw_dt, 1.0f / 1000.0f, 0.10f)
                             : (1.0f / 30.0f);

        if (chr_ctrl != last_ctrl || !have_prev_pos || std::fabs(pos_y - prev_pos_y) > 1.5f) {
            last_ctrl = chr_ctrl;
            last_frame_seq = seq;
            Reset(pos_y);
            return 0.0f;
        }

        auto* action_flags = *reinterpret_cast<std::uint8_t**>(chr_ctrl + 0x38);
        const int st = action_flags
                           ? static_cast<int>(*reinterpret_cast<const std::int8_t*>(action_flags + 0x60))
                           : StateOff;

        const float dy = pos_y - prev_pos_y;
        const float vy_raw = dy / dt;

        // Track slide entry from SlideStartRight/Left (14/17) or Sliding (15/18).
        if (st == StateSlideStartRight || st == StateSlideStartLeft ||
            st == StateSlidingRight || st == StateSlidingLeft) {
            slide_armed = true;
        } else if (st != StateDownRight && st != StateDownLeft) {
            slide_armed = false;
        }

        const bool is_down_state = (st == StateDownRight || st == StateDownLeft ||
                                    st == StateSlidingRight || st == StateSlidingLeft);
        const bool is_up_state = (st == StateClimbUpRight || st == StateClimbUpLeft);
        const bool want_fast_slide =
            is_down_state && (in_fast_slide || slide_armed || vy_raw < -2.8f);

        if (want_fast_slide) {
            zero_streak = (std::fabs(dy) < 0.002f) ? (zero_streak + 1) : 0;

            if (!in_fast_slide) {
                // First moving frame of fast slide: seed exact analytical rung-cycle velocity
                // for Bloodborne's 0.100s clip + 1-frame HKS transition pause.
                in_fast_slide = true;
                const float t_rung = std::ceil(0.100f / dt - 1e-3f) * dt + dt;
                v_smooth = -0.500f / std::max(t_rung, 0.05f);
                rung_start_y = prev_pos_y;
                vis_y = rung_start_y + v_smooth * dt;
                rung_elapsed = dt;
                rung_count = 0;
                prev_state = st;
            } else if (zero_streak >= 2) {
                // Player released slide / stopping while still in state 4/5: decay smoothly to pos_y.
                float off = (vis_y - pos_y) * std::exp(-35.0f * dt);
                if (std::fabs(off) < 1e-4f) off = 0.0f;
                vis_y = pos_y + off;
            } else {
                if (st != prev_state) {
                    // Exact rung boundary: prev_pos_y is the exact rung coordinate y_k!
                    if (rung_elapsed > 0.06f) {
                        const float v_meas = (prev_pos_y - rung_start_y) / rung_elapsed;
                        if (v_meas > -7.0f && v_meas < -1.5f) {
                            const float phase_err = prev_pos_y - vis_y;
                            const float v_base = 0.75f * v_smooth + 0.25f * v_meas;
                            v_smooth = std::clamp(v_base + 0.35f * phase_err / rung_elapsed, -6.5f, -1.5f);
                        }
                    }
                    rung_start_y = prev_pos_y;
                    rung_elapsed = dt;
                    ++rung_count;
                    prev_state = st;
                } else {
                    rung_elapsed += dt;
                }
                vis_y += v_smooth * dt;
                // Causal bounds: tiny 1.5 cm floor tolerance avoids sudden clip on 1-frame dt spikes;
                // max 20 cm upward lag keeps character locked onto the ladder rails.
                vis_y = std::clamp(vis_y, pos_y - 0.015f, pos_y + 0.20f);
            }
            climb_v_ema = 0.0f;
        } else if (is_up_state || is_down_state) {
            // Normal hand-over-hand climbing (up or slow down): preserve natural cadence while
            // bridging the 1-frame `0.000 m/s` dead stop at every rung transition.
            in_fast_slide = false;
            const float dir = is_up_state ? 1.0f : -1.0f;
            zero_streak = (std::fabs(dy) < 0.002f) ? (zero_streak + 1) : 0;

            if (zero_streak == 0) {
                const bool same_dir = (vy_raw * dir) > 0.05f;
                if (same_dir) {
                    climb_v_ema = (climb_v_ema == 0.0f) ? vy_raw : (0.7f * climb_v_ema + 0.3f * vy_raw);
                }
                // Fixed small causal headroom (~0.55 frames at 1.5 m/s) so the dead frame bridges cleanly
                // without dipping when velocity increases at the start of the next rung.
                const float headroom = 0.55f * 1.5f * dt * dir;
                const float target = pos_y - headroom;
                vis_y += (target - vis_y) * (1.0f - std::exp(-22.0f * dt));
            } else if (zero_streak == 1 && (climb_v_ema * dir) > 0.1f) {
                // Single transition dead frame: glide smoothly toward pos_y without overshooting.
                const float next_y = vis_y + climb_v_ema * dt;
                vis_y = is_up_state ? std::min(pos_y, next_y) : std::max(pos_y, next_y);
            } else {
                float off = (vis_y - pos_y) * std::exp(-35.0f * dt);
                if (std::fabs(off) < 1e-4f) off = 0.0f;
                vis_y = pos_y + off;
            }
            vis_y = std::clamp(vis_y, pos_y - 0.10f, pos_y + 0.10f);
            prev_state = st;
        } else {
            // Idle, grabbing, exiting top/bottom, or off ladder: smoothly decay any residual mm to 0.
            in_fast_slide = false;
            zero_streak = 0;
            climb_v_ema = 0.0f;
            float off = (vis_y - pos_y) * std::exp(-35.0f * dt);
            if (std::fabs(off) < 1e-4f) off = 0.0f;
            vis_y = pos_y + off;
            prev_state = st;
        }

        const float offset_y = vis_y - pos_y;

        if (TraceEnabled()) {
            if (st != StateOff && (is_down_state || is_up_state)) {
                if (trace_budget <= 0) trace_budget = 160;
            }
            if (st == StateOff) {
                trace_budget = 0;
            } else if (trace_budget > 0) {
                --trace_budget;
                const float vy_vis = (vis_y - prev_vis_y) / dt;
                std::printf(
                    "[LADDER-SMOOTH] f=%u st=%d dt=%.4f rawY=%.5f rawVy=%+.3f visVy=%+.3f off=%+.4f slide=%d\n",
                    ++trace_frame, st, dt, pos_y, vy_raw, vy_vis, offset_y, in_fast_slide ? 1 : 0);
            }
        }

        prev_pos_y = pos_y;
        prev_vis_y = vis_y;
        last_frame_seq = seq;
        cached_offset_y.store(offset_y, std::memory_order_relaxed);
        return offset_y;
    }
};

} // namespace BbLadderSmooth
