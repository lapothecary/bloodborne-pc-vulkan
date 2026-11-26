// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the game's frame rate set free by hooks in its code (BB_FPS=uncap, the default), not by
// patches: its frame pacer no longer waits for a 30 FPS frame, its step gets the frame's real time,
// and what it ties to a frame (a 30 FPS frame's time, a per-frame step, smoothing and damping) reads
// values the hooks recompute from that time every frame (bbport_frame_rate.cpp).
#pragma once

#include <cstdint>

namespace BbFrameRate {
/// Installs the hooks (at load, before the game runs: bbgpu_patch_image).
void PatchImage(unsigned char* image, std::uint64_t size);
/// Whether the hooks run the game's frame rate: the video out then shows a frame as soon as the
/// game submits it (the game's own flip rate asks for 30 FPS).
bool Unlocked();
/// Controlled by BB_CLEAN_PACER (default 1): disables PS4's 5 ms Main busy-spin and avoids
/// redundant VideoOut double-limiting when the unlocked guest pacer already enforces BB_FPS_LIMIT.
bool CleanPacerEnabled() noexcept;
/// A frame rate limit while the game runs (30..480; 0: the launched mode and limit). The game's
/// frame pacer waits for it itself. Lowers a running game to 30 FPS for what the unlocked rate
/// breaks (the cinematic of Laurence, the optional boss: a hang above 30 FPS, as with the
/// community 60 FPS patch).
void SetLimit(int fps);
int Limit();
/// Monotonic clock in nanoseconds (CLOCK_MONOTONIC on Linux, QPC steady_clock on Win32).
std::uint64_t NowMonotonicNs() noexcept;
/// High-precision deadline wait (TIMER_ABSTIME / high-res waitable timer + short _mm_pause tail).
/// `timer_slot`: 0 for Main pacer, 1 for Present pacer (zero thread_local / PT_TLS cost).
void SleepUntilMonotonicNs(std::uint64_t deadline_ns, int timer_slot = 0) noexcept;
/// Active target frame period in nanoseconds when unlocked and capped, or 0 when uncapped.
std::uint64_t TargetPeriodNs() noexcept;
/// Increments whenever SetLimit() changes the frame rate target.
std::uint64_t PacerEpoch() noexcept;
/// Controlled by BB_PRESENT_PACE (default 1): paces vkQueuePresentKHR delivery on bb:Present
/// when capped in Mailbox/Immediate so variable render durations (2.4..4.8 ms) do not cause
/// early burst presents (172-175 FPS) or 2.5 ms phase jitter.
bool PresentPacingEnabled() noexcept;
/// Latest simulation frame delta in 30 FPS units (`dt * 30.0f`: 1.0 at 30 FPS, ~0.176 at 170 FPS).
float CurrentFramesPer30() noexcept;
/// Monotonic simulation frame sequence counter (increments once per FrameStep).
std::uint64_t FrameSeq() noexcept;
/// Synchronizes cloth world timing & 30 Hz smooth wind seeds when `ClothExp_FD4HavokCloth_step` runs.
void NotifyClothWorldStep(float dt) noexcept;
/// Queries current 30 Hz smooth wind seeds, cubic blend `[0..1]`, and `cloth_frames` for `ClothExp`.
void GetClothWindSmoothParams(std::uint32_t& out_seed0, std::uint32_t& out_seed1,
                              float& out_smooth, float& out_frames) noexcept;
} // namespace BbFrameRate
