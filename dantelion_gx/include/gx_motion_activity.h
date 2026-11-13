#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>

#include "gx_runtime_vma_and_tls.h"

extern "C" {
GX_HOST_IMPORT void bb_direct_vk_note_vel_silhouette_mask() noexcept;
}

namespace gx_direct_vk {

inline void NoteVelocitySilhouetteMask() noexcept {
    if (bb_direct_vk_note_vel_silhouette_mask) {
        bb_direct_vk_note_vel_silhouette_mask();
    }
}

// Minimum single-frame 3x4 bone matrix delta (~1.2 cm world translation or ~0.7 deg rotation)
// required to produce >= 1 pixel of visible Yebis2 object motion blur. Idle character breathing
// and gentle wind sway stay 5x-50x below this threshold, letting Yebis2 skip 6 fullscreen passes.
inline constexpr float kObjectVelocityBoneDeltaThreshold = 0.012f;

// Hold counter across 3 frames so pipelined GXSkinningUpdateJob workers and trailing blur frames
// never miss active character/enemy/weapon motion while the camera stands still.
inline std::atomic<uint8_t> g_obj_velocity_hold_frames{0};

inline bool KeepIdleMotionBlur() noexcept {
    static const bool keep = []() noexcept {
        const char* env = std::getenv("BB_KEEP_IDLE_MOTION_BLUR");
        return env && env[0] == '1';
    }();
    return keep;
}

inline void NoteActiveObjectVelocity() noexcept {
    if (g_obj_velocity_hold_frames.load(std::memory_order_relaxed) < 3u) {
        g_obj_velocity_hold_frames.store(3u, std::memory_order_relaxed);
    }
}

inline bool ConsumeActiveObjectVelocity() noexcept {
    if (KeepIdleMotionBlur()) [[unlikely]] {
        return true;
    }
    uint8_t cur = g_obj_velocity_hold_frames.load(std::memory_order_relaxed);
    while (cur > 0u) {
        if (g_obj_velocity_hold_frames.compare_exchange_weak(
                cur, static_cast<uint8_t>(cur - 1u),
                std::memory_order_relaxed, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

} // namespace gx_direct_vk
