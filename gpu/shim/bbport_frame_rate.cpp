// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the game's frame rate set free by hooks in its code (bbport_frame_rate.h).
//
// The game's main loop (0x2018d20) runs its frame pacer (0x2034770: it waits for the frame time
// its mode asks, 1/30 s, and measures the frame at [pacer + 0x264]) and then steps the game by a
// fixed 1/30 s (0x2018e36). The hooks keep the pacer's mode at one vblank and no frame time to
// wait for beyond the shortest frame, and step the game by the measured frame time (1/480 to
// 1/30 s: below 30 FPS the game slows down, as it always did).
//
// What the game ties to a 30 FPS frame besides (found by the frame time each function is called
// with, BB_RE_ENTRY, and by what its code does with the constant): subsystems stepped by a fixed
// 1/30 s, per-frame steps and approaches, per-frame damping. Their instructions read the constant
// from a page next to the image instead (the instruction's own address operand moved; the game's
// code is otherwise as it was), where FrameStep writes the value for this frame's time: a step of
// 1/30 s becomes the frame's time, a per-frame factor f becomes f^(frames) for the frame's length
// in 30 FPS frames. Durations the game converts from 30 FPS frames into seconds stay as they are:
// what counts them down uses the real time.
#include "common/types.h"
#include "bbport_cloth_interp.h"
#include "bbport_fast_load.h"
#include "bbport_frame_rate.h"
#include "bbport_guest_code.h"
#include "bbport_sections.h"
#include "bbport_settings.h"
#include "bbport_toggles.h"
#include "cloth_exp/cloth_exp_registry.h"
#include "vulkan/low_latency.h"

#include <array>
#include <cerrno>
#include <chrono>
#include <ctime>
#include <sys/mman.h>
#include <x86intrin.h>
#ifdef __linux__
#include <sys/prctl.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#endif
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace BbFrameRate {
namespace {
using BbGuestCode::Emitter;
using u64 = std::uint64_t;
using u8 = std::uint8_t;

std::atomic<bool> unlocked{false};
u64 image_base = 0;

constexpr float Frame30 = 1.0f / 30.0f;
/// With FixedBlockPool (0x0263b8b0 / 0x0263b9b0) now thread-safe and quarantine-protected in C++20,
/// the pacer supports true uncapped frame rate (`ShortestFrame = 0.0f` when `fps_limit == 0`) or
/// explicit targets up to `MaxRate = 5000` FPS, while clamping simulation `dt >= MinSimDt` so
/// Havok physics never divides by zero on sub-millisecond frames.
constexpr int MaxRate = 5000;
constexpr float ShortestFrame = 0.0f;
constexpr float MinSimDt = 1.0f / float(MaxRate);

/// What the redirected instructions read, rewritten by FrameStep for each frame.
struct alignas(64) Live {
    float dt = Frame30;               ///< the frame's time: the game steps by it
    float shortest = ShortestFrame;   ///< the pacer's frame time target (0.0f = truly uncapped)
    float spin_margin = 0.00020f;     ///< pacer usleep spin margin (200 us vs PS4's 5.0 ms)
    float neg_spin_margin = -0.00020f;
    float decay95[4] = {0.95f, 0.95f, 0.95f, 0.95f}; ///< 0.95 per 30 FPS frame (read as four)
    float approach30 = Frame30;       ///< a 1/30 approach per 30 FPS frame
    float approach10 = 0.1f;          ///< a 0.1 approach per 30 FPS frame
    float step_down = -0.1f;          ///< -0.1 per 30 FPS frame
    float step_up = 0.1f;             ///< +0.1 per 30 FPS frame
    float decay08 = 0.8f;             ///< 0.8 per 30 FPS frame
    float grow12 = 1.2f;              ///< 1.2 per 30 FPS frame (run wall-check speed recovery)
    float decay65 = 0.65f;            ///< 0.65 per 30 FPS frame (slope-slide damping, 0x1a46bd6)
    float slope_accel = 1.0f;         ///< 1.0 per 30 FPS frame (slope-slide acceleration, 0x1a46c52)
    float decay60 = 0.6f;             ///< 0.6 per 30 FPS frame (foot-IK pelvis Y-offset decay, 0x1515195)
    float cloth = 1.0f;               ///< cloth constraints' stiffness
    float cloth_stiffness_scale = 1.0f; ///< frames^(-cloth_exponent) = frames^1.725 (Havok substep-equivalent)
    float model_turn = 0.0f;          ///< the model preview's turn per 30 FPS frame (radians per stick unit)
    float dt_half = Frame30 * 0.5f;   ///< 1/60 s at 30 FPS (0.5 * dt), used by foliage wave timers (0x2313870)
    alignas(16) float cloth_friction_scale[4] = {1.0f, 1.0f, 1.0f, 1.0f}; ///< dt-adapted cloth collision friction scale
    std::uint32_t cloth_wind_seed30 = 0; ///< 30 Hz quantized XorShift seed (t0) for smooth wind turbulence
    std::uint32_t cloth_wind_seed1 = 10000; ///< next 30 Hz quantized XorShift seed (t1)
    float cloth_wind_smooth = 0.0f; ///< cubic smoothstep blend [0..1] for output wind vectors & turbulence
    float cloth_pbd_neg_rate = 0.0f; ///< high-FPS PBD micro-jitter damping rate (-s^-1; 0 at 30 FPS)
    float cloth_frames = 1.0f; ///< clamped cdt * 30.0f synchronized with hclWorld::step
};
static_assert(offsetof(Live, cloth_friction_scale) % 16 == 0,
              "cloth_friction_scale must be 16-byte aligned for vmulps");
Live* live = nullptr;
bool cloth_world_synced = false;
bool cloth_fixed30_hooked = false;
bool cloth_dir_wind_hooked = false;
double cloth_sim_time_s = 0.0;
double cloth_fixed_accum = 0.0;
std::uint32_t cloth_steady_30_streak = 0;
std::uint64_t cloth_fixed_step_seq = 0;
std::atomic<bool> cloth_step_this_frame{true};
std::atomic<float> cloth_interp_alpha{1.0f};
std::atomic<float> cloth_step_beta{1.0f};
std::atomic<float> cloth_step_dt{Frame30};
std::atomic<std::uint64_t> cloth_world_frame_seq{0};
std::atomic<float> current_frames_per_30{1.0f};
std::atomic<u64> sim_frame_seq{0};
float model_turn_base = 0.0f;         ///< the game's own model_turn (its constant, read at start)
/// The game's exponent of Havok's stiffness adaptation (0x48d7bbc: -1.725), read at start.
float cloth_exponent = -1.725f;

bool ClothFixEnabled() noexcept {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_CLOTH_FIX");
        return !(env && env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

bool ClothWindSeedEnabled() noexcept {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_CLOTH_SEED");
        return !(env && env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

float ClothExpOverride() noexcept {
    static const float v = [] {
        const char* env = std::getenv("BB_CLOTH_STIFF_EXP");
        if (!env || !*env) {
            env = std::getenv("BB_CLOTH_EXP");
            // Ignore boolean "0" / "1" used by BbClothExp::PatchImage startup toggle.
            if (env && (env[0] == '0' || env[0] == '1') && env[1] == '\0') {
                return -1.0f;
            }
        }
        const float parsed = (env && *env) ? std::strtof(env, nullptr) : -1.0f;
        return (parsed > 0.0f && parsed <= 3.0f) ? parsed : -1.0f;
    }();
    return v;
}

float ClothDampRetention30() noexcept {
    static const float v = [] {
        const char* env = std::getenv("BB_CLOTH_DAMP");
        const float parsed = (env && *env) ? std::strtof(env, nullptr) : 0.94f;
        return (parsed >= 0.5f && parsed <= 1.0f) ? parsed : 0.94f;
    }();
    return v;
}

bool ClothWindOffEnabled() noexcept {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_CLOTH_WIND_OFF");
        return env && env[0] == '1' && env[1] == '\0';
    }();
    return enabled;
}

std::uint32_t ClothSkipMask() noexcept {
    static const std::uint32_t mask = [] {
        const char* env = std::getenv("BB_CLOTH_SKIP_T");
        if (!env || !*env) return 0u;
        return static_cast<std::uint32_t>(std::strtoul(env, nullptr, 0));
    }();
    return mask;
}

void UpdateClothNativeTiming(float cdt, bool update_t5_pulse) noexcept;
int base_rate = 0;               /// the launched mode's rate (0 = truly uncapped)
std::atomic<int> live_limit{0};
std::atomic<u64> pacer_epoch{1};
bool pacer_wait_hooked = false;
u64 pacer_prev_exit_ns = 0;
u64 pacer_prev_exit_cpu_ns = 0;
u64 pacer_prev_exit_tsc = 0;
u64 pacer_next_deadline_ns = 0;
u64 pacer_last_period_ns = 0;
u64 pacer_seen_epoch = 0;
std::uint32_t pacer_miss_streak = 0;

u64 MonotonicNs() noexcept {
#ifdef __linux__
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return u64(ts.tv_sec) * 1'000'000'000ull + u64(ts.tv_nsec);
#else
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
#endif
}

u64 ThreadCpuNs(bool& ok) noexcept {
#if defined(_WIN32)
    ULONG64 thread_cycles = 0;
    if (QueryThreadCycleTime(GetCurrentThread(), &thread_cycles)) {
        static const u64 base_tsc = __rdtsc();
        static const auto base_time = std::chrono::steady_clock::now();
        const u64 dtsc = __rdtsc() - base_tsc;
        const u64 dns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - base_time)
                                .count());
        const double ns_per_cyc = (dtsc > 100'000ull && dns > 100'000ull)
                                      ? (double(dns) / double(dtsc))
                                      : (1.0 / 3.6);
        ok = true;
        return u64(double(thread_cycles) * ns_per_cyc);
    }
    ok = false;
    return 0;
#elif defined(CLOCK_THREAD_CPUTIME_ID)
    struct timespec ts_cpu{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts_cpu) == 0) {
        ok = true;
        return u64(ts_cpu.tv_sec) * 1'000'000'000ull + u64(ts_cpu.tv_nsec);
    }
    ok = false;
    return 0;
#else
    ok = false;
    return 0;
#endif
}

/// Phase-locked monotonic deadline pacer replacing SprjFlipper's relative gettimeofday + usleep
/// block (0x2034964..0x2034c55). Eliminates relative sleep wakeup drift and zero-catchup ratcheting
/// while measuring exact active CPU work, off-CPU thread sync, and pacer sleep on MOMainThread.
void SleepUntilDeadlineImpl(u64 entry_ns, u64 deadline_ns, int timer_slot) noexcept {
    if (entry_ns >= deadline_ns) {
        return;
    }
#if defined(__linux__)
    (void)timer_slot;
    constexpr u64 SpinMarginNs = 220'000ull; // 220 us precision spin after hrtimer wake
    if (deadline_ns > entry_ns + SpinMarginNs) {
        const u64 wake_ns = deadline_ns - SpinMarginNs;
        struct timespec ts_wake{
            .tv_sec = static_cast<time_t>(wake_ns / 1'000'000'000ull),
            .tv_nsec = static_cast<long>(wake_ns % 1'000'000'000ull),
        };
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts_wake, nullptr) == EINTR) {
        }
    }
#elif defined(_WIN32)
    constexpr u64 SpinMarginNs = 450'000ull;
    if (deadline_ns > entry_ns + SpinMarginNs + 100'000ull) {
        static HANDLE hi_timers[2] = {nullptr, nullptr};
        static bool hi_init[2] = {false, false};
        const int idx = (timer_slot == 1) ? 1 : 0;
        if (!hi_init[idx]) {
            hi_init[idx] = true;
            hi_timers[idx] = CreateWaitableTimerExW(
                nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        }
        if (HANDLE ht = hi_timers[idx]) {
            const u64 cur_ns = MonotonicNs();
            if (deadline_ns > cur_ns + SpinMarginNs) {
                const u64 sleep_100ns = (deadline_ns - cur_ns - SpinMarginNs) / 100ull;
                if (sleep_100ns > 0) {
                    LARGE_INTEGER due{};
                    due.QuadPart = -static_cast<LONGLONG>(sleep_100ns);
                    if (SetWaitableTimer(ht, &due, 0, nullptr, nullptr, FALSE)) {
                        WaitForSingleObject(ht, 50);
                    }
                }
            }
        }
    }
#else
    (void)timer_slot;
#endif
    while (MonotonicNs() < deadline_ns) {
        _mm_pause();
    }
}

u64 ComputeFgSimPeriodNs(u64 raw_period_ns) noexcept {
    if (raw_period_ns == 0ull) {
        return 0ull;
    }
    const auto& s = BbSettings::Get();
    const bool is_auto = s.fps_limit_auto.load(std::memory_order_relaxed);
    const int mon_hz = BbSettings::MonitorRefreshHz();
    const int cur_limit = s.fps_limit.load(std::memory_order_relaxed);
    // When FPS Limit is Auto (e.g. Auto 170 Hz) or set at/above monitor Hz, halve base simulation
    // so 2x FG outputs exact monitor Hz (85 -> 170 FPS) with zero wasted GPU frames.
    // When the user explicitly selects a lower limit (e.g. 60 or 85 on a 170 Hz monitor), treat
    // it as base FPS so 60 -> 120 FPS FG instead of unexpectedly dropping simulation to 30 FPS.
    if (is_auto || (mon_hz >= 60 && cur_limit >= mon_hz)) {
        constexpr u64 kMinSimRatePeriodNs = 33'333'333ull;
        const u64 mult = static_cast<u64>(
            std::clamp(BbStats::fg_multiplier.load(std::memory_order_relaxed), 1, 4));
        if (raw_period_ns < kMinSimRatePeriodNs) {
            return std::min<u64>(raw_period_ns * mult, kMinSimRatePeriodNs);
        }
    }
    return raw_period_ns;
}

u64 PS4_SYSV_ABI PacerWait(u64 pacer, std::uint32_t reset_flag) {
#ifdef __linux__
    static bool timerslack_set = false;
    if (!timerslack_set) {
        timerslack_set = true;
        prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
    }
#endif
    const u64 entry_ns = MonotonicNs();
    const u64 entry_tsc = __rdtsc();
    bool have_entry_cpu = false;
    const u64 entry_cpu_ns = ThreadCpuNs(have_entry_cpu);

    if (pacer_prev_exit_ns != 0 && entry_ns >= pacer_prev_exit_ns) {
        const u64 active_wall_ns = entry_ns - pacer_prev_exit_ns;
        if (have_entry_cpu && pacer_prev_exit_cpu_ns != 0 &&
            entry_cpu_ns >= pacer_prev_exit_cpu_ns) {
            const u64 active_cpu_ns = entry_cpu_ns - pacer_prev_exit_cpu_ns;
            if (active_cpu_ns < 500'000'000ull) {
                BbStats::main_cpu_busy_ns.fetch_add(active_cpu_ns, std::memory_order_relaxed);
                BbStats::main_cpu_frames.fetch_add(1, std::memory_order_relaxed);
                const u64 sync_wait_ns =
                    (active_wall_ns > active_cpu_ns && active_wall_ns < 500'000'000ull)
                        ? (active_wall_ns - active_cpu_ns)
                        : 0ull;
                BbStats::main_sync_wait_ns.fetch_add(sync_wait_ns, std::memory_order_relaxed);
            }
        }
    }

    static u64 prev_pacer_draws = 0;
    static bool was_loading_burst = false;
    static u64 prev_hitch_spv_ns = 0, prev_hitch_spv_cnt = 0;
    static u64 prev_hitch_pso_ns = 0, prev_hitch_pso_cnt = 0, prev_hitch_pso_bg = 0;
    static u64 prev_hitch_imgs = 0, prev_hitch_img_bytes = 0, prev_hitch_buf_bytes = 0;
    static u64 prev_hitch_rdbk = 0, prev_hitch_faults = 0, prev_hitch_gpu_wait_ns = 0;
    static u64 prev_hitch_gnm_wait_ns = 0, prev_hitch_acq_ns = 0, prev_hitch_pres_ns = 0;
    static std::array<u64, BbSections::MainXRay::MaxGroups> prev_hitch_mx_t0{};
    const u64 cur_draws = BbStats::draws.load(std::memory_order_relaxed);
    const u64 frame_draws = (cur_draws >= prev_pacer_draws) ? (cur_draws - prev_pacer_draws) : 0ull;
    prev_pacer_draws = cur_draws;
    const bool loading_burst = BbFastLoad::UpdateFrameAndCheckBurst(entry_ns, frame_draws);
    if (was_loading_burst && !loading_burst) {
        pacer_epoch.fetch_add(1, std::memory_order_relaxed);
    }
    was_loading_burst = loading_burst;

    {
        const u64 cur_spv_ns = BbStats::hitch_spv_sync_ns.load(std::memory_order_relaxed);
        const u64 cur_spv_cnt = BbStats::hitch_spv_sync_count.load(std::memory_order_relaxed);
        const u64 cur_pso_ns = BbStats::hitch_pso_sync_ns.load(std::memory_order_relaxed);
        const u64 cur_pso_cnt = BbStats::hitch_pso_sync_count.load(std::memory_order_relaxed);
        const u64 cur_pso_bg = BbStats::hitch_pso_async_count.load(std::memory_order_relaxed);
        const u64 cur_imgs = BbStats::images_registered.load(std::memory_order_relaxed);
        const u64 cur_img_b = BbStats::image_upload_bytes.load(std::memory_order_relaxed);
        const u64 cur_buf_b = BbStats::buffer_upload_bytes.load(std::memory_order_relaxed);
        const u64 cur_rdbk = BbStats::readbacks.load(std::memory_order_relaxed);
        const u64 cur_faults = BbStats::tracker_faults_total.load(std::memory_order_relaxed);
        const u64 cur_gpu_w = BbStats::sync_recording_ns.load(std::memory_order_relaxed) +
                              BbStats::tick_wait_ns.load(std::memory_order_relaxed) +
                              BbStats::present_wait_ns.load(std::memory_order_relaxed);
        const u64 cur_gnm_w = BbStats::gnm_frame_wait_ns.load(std::memory_order_relaxed);
        const u64 cur_acq_w = BbStats::swapchain_acquire_wait_ns.load(std::memory_order_relaxed);
        const u64 cur_pres_w = BbStats::swapchain_present_wait_ns.load(std::memory_order_relaxed);

        const float cap_s = live ? live->shortest : 0.0f;
        const u64 raw_cap_period_ns =
            (cap_s > 0.0f && std::isfinite(cap_s))
                ? static_cast<u64>(std::llround(double(cap_s) * 1'000'000'000.0))
                : 0ull;
        const u64 cap_period_ns =
            (raw_cap_period_ns > 0ull && BbStats::fg_active.load(std::memory_order_relaxed))
                ? ComputeFgSimPeriodNs(raw_cap_period_ns)
                : raw_cap_period_ns;
        const u64 min_hitch_ns =
            (cap_period_ns > 0 && cap_period_ns <= 8'500'000ull)
                ? std::max<u64>(9'000'000ull, (cap_period_ns * 3ull) / 2ull)
                : ((cap_period_ns > 0 && cap_period_ns <= 10'000'000ull)
                       ? std::max<u64>(12'000'000ull, cap_period_ns * 2ull)
                       : 20'000'000ull);

        static double hitch_ema_wall_ns = 0.0;
        const bool mx_on = BbSections::main_xray.installed.load(std::memory_order_relaxed);
        const std::size_t mx_cnt =
            mx_on ? std::min<std::size_t>(
                        BbSections::main_xray.group_count.load(std::memory_order_acquire),
                        BbSections::MainXRay::MaxGroups)
                  : 0u;
        const u64 slow_t0_cyc =
            mx_on ? BbSections::main_xray.frame_max_t0_cycles.exchange(0ull, std::memory_order_relaxed)
                  : 0ull;
        const u64 slow_t0_fn =
            mx_on ? BbSections::main_xray.frame_max_t0_fn_off.exchange(0ull, std::memory_order_relaxed)
                  : 0ull;
        const std::int32_t slow_t0_mm =
            mx_on ? BbSections::main_xray.frame_max_t0_movemap_state.exchange(
                        -1, std::memory_order_relaxed)
                  : -1;
        if (!loading_burst && frame_draws >= 100 && pacer_prev_exit_ns != 0 &&
            entry_ns >= pacer_prev_exit_ns) {
            const u64 wall_ns = entry_ns - pacer_prev_exit_ns;
            const u64 delta_spv_cnt = cur_spv_cnt - prev_hitch_spv_cnt;
            const u64 delta_pso_cnt = cur_pso_cnt - prev_hitch_pso_cnt;
            const u64 adaptive_thresh_ns =
                std::max<u64>(min_hitch_ns, static_cast<u64>(hitch_ema_wall_ns * 2.25));
            const bool is_real_hitch =
                wall_ns < 5'000'000'000ull &&
                (wall_ns >= adaptive_thresh_ns ||
                 ((delta_spv_cnt > 0 || delta_pso_cnt > 0) && wall_ns >= 12'000'000ull));
            if (is_real_hitch) [[unlikely]] {
                const u64 cpu_ns = (have_entry_cpu && pacer_prev_exit_cpu_ns != 0 &&
                                    entry_cpu_ns >= pacer_prev_exit_cpu_ns)
                                       ? (entry_cpu_ns - pacer_prev_exit_cpu_ns)
                                       : 0ull;
                const u64 wait_ns = (wall_ns > cpu_ns) ? (wall_ns - cpu_ns) : 0ull;
                char top_buf[288];
                top_buf[0] = '\0';
                if (mx_cnt > 0 && pacer_prev_exit_tsc != 0 && entry_tsc > pacer_prev_exit_tsc) {
                    const double ms_per_cyc =
                        (double(wall_ns) / 1e6) / double(entry_tsc - pacer_prev_exit_tsc);
                    struct TopG {
                        std::size_t idx = 0;
                        u64 cyc = 0;
                    };
                    TopG top[3]{};
                    u64 sum_cyc = 0;
                    for (std::size_t g = 0; g < mx_cnt; ++g) {
                        const u64 c = BbSections::main_xray.cycles_t0[g].load(std::memory_order_relaxed);
                        const u64 dc = c >= prev_hitch_mx_t0[g] ? (c - prev_hitch_mx_t0[g]) : 0ull;
                        sum_cyc += dc;
                        if (dc > top[2].cyc) {
                            top[2] = {g, dc};
                            if (top[2].cyc > top[1].cyc) std::swap(top[2], top[1]);
                            if (top[1].cyc > top[0].cyc) std::swap(top[1], top[0]);
                        }
                    }
                    int pos = std::snprintf(top_buf, sizeof(top_buf), " | top=[");
                    bool any = false;
                    for (int k = 0; k < 3; ++k) {
                        if (top[k].cyc == 0 || pos >= int(sizeof(top_buf)) - 80) break;
                        const double gms = double(top[k].cyc) * ms_per_cyc;
                        if (gms < 0.25 && any) break;
                        const char* nm =
                            BbSections::main_xray.name_ready[top[k].idx].load(std::memory_order_acquire)
                                ? BbSections::main_xray.names[top[k].idx]
                                : "?";
                        pos += std::snprintf(top_buf + pos, sizeof(top_buf) - std::size_t(pos),
                                             "%s%s=%.2fms", any ? ", " : "", nm, gms);
                        any = true;
                    }
                    const double untracked_ms =
                        std::max(0.0, (double(wall_ns) / 1e6) - double(sum_cyc) * ms_per_cyc);
                    if (untracked_ms >= 1.5 && pos < int(sizeof(top_buf)) - 64) {
                        pos += std::snprintf(top_buf + pos, sizeof(top_buf) - std::size_t(pos),
                                             "%sother=%.2fms", any ? ", " : "", untracked_ms);
                    }
                    pos += std::snprintf(top_buf + pos, sizeof(top_buf) - std::size_t(pos), "]");
                    if (slow_t0_cyc > 0 && slow_t0_fn != 0 && pos < int(sizeof(top_buf)) - 48) {
                        const double sms = double(slow_t0_cyc) * ms_per_cyc;
                        if (slow_t0_mm >= 0) {
                            std::snprintf(top_buf + pos, sizeof(top_buf) - std::size_t(pos),
                                          " slow=0x%llx(st=%d,%.2fms)",
                                          static_cast<unsigned long long>(slow_t0_fn), slow_t0_mm,
                                          sms);
                        } else {
                            std::snprintf(top_buf + pos, sizeof(top_buf) - std::size_t(pos),
                                          " slow=0x%llx(%.2fms)",
                                          static_cast<unsigned long long>(slow_t0_fn), sms);
                        }
                    }
                }
                BbStats::LogHitch(
                    "[BB-HITCH][FRAME] frame=%u wall=%.2f ms (main_cpu=%.2f ms, main_wait=%.2f ms, "
                    "gnm_wait=%.2f ms | spv=%.2f ms [%llu], pso=%.2f ms [%llu], pso_bg=%llu | "
                    "tex=%llu [%.2f MiB], buf=%.2f MiB, rdbk=%llu, faults=%llu, gpu_wait=%.2f ms, "
                    "acq=%.2f ms, pres=%.2f ms | draws=%llu%s)\n",
                    BbStats::frame_number.load(std::memory_order_relaxed), double(wall_ns) / 1e6,
                    double(cpu_ns) / 1e6, double(wait_ns) / 1e6,
                    double(cur_gnm_w - prev_hitch_gnm_wait_ns) / 1e6,
                    double(cur_spv_ns - prev_hitch_spv_ns) / 1e6,
                    static_cast<unsigned long long>(delta_spv_cnt),
                    double(cur_pso_ns - prev_hitch_pso_ns) / 1e6,
                    static_cast<unsigned long long>(delta_pso_cnt),
                    static_cast<unsigned long long>(cur_pso_bg - prev_hitch_pso_bg),
                    static_cast<unsigned long long>(cur_imgs - prev_hitch_imgs),
                    double(cur_img_b - prev_hitch_img_bytes) / (1024.0 * 1024.0),
                    double(cur_buf_b - prev_hitch_buf_bytes) / (1024.0 * 1024.0),
                    static_cast<unsigned long long>(cur_rdbk - prev_hitch_rdbk),
                    static_cast<unsigned long long>(cur_faults - prev_hitch_faults),
                    double(cur_gpu_w - prev_hitch_gpu_wait_ns) / 1e6,
                    double(cur_acq_w - prev_hitch_acq_ns) / 1e6,
                    double(cur_pres_w - prev_hitch_pres_ns) / 1e6,
                    static_cast<unsigned long long>(frame_draws), top_buf);
            }
            if (wall_ns < 200'000'000ull) {
                hitch_ema_wall_ns = (hitch_ema_wall_ns == 0.0)
                                        ? double(wall_ns)
                                        : (hitch_ema_wall_ns * 0.96 + double(wall_ns) * 0.04);
            }
        }
        for (std::size_t g = 0; g < mx_cnt; ++g) {
            prev_hitch_mx_t0[g] = BbSections::main_xray.cycles_t0[g].load(std::memory_order_relaxed);
        }
        prev_hitch_spv_ns = cur_spv_ns;
        prev_hitch_spv_cnt = cur_spv_cnt;
        prev_hitch_pso_ns = cur_pso_ns;
        prev_hitch_pso_cnt = cur_pso_cnt;
        prev_hitch_pso_bg = cur_pso_bg;
        prev_hitch_imgs = cur_imgs;
        prev_hitch_img_bytes = cur_img_b;
        prev_hitch_buf_bytes = cur_buf_b;
        prev_hitch_rdbk = cur_rdbk;
        prev_hitch_faults = cur_faults;
        prev_hitch_gpu_wait_ns = cur_gpu_w;
        prev_hitch_gnm_wait_ns = cur_gnm_w;
        prev_hitch_acq_ns = cur_acq_w;
        prev_hitch_pres_ns = cur_pres_w;
    }

    const float target_s = live ? live->shortest : 0.0f;
    u64 period_ns =
        (target_s > 0.0f && std::isfinite(target_s))
            ? static_cast<u64>(std::llround(double(target_s) * 1'000'000'000.0))
            : 0ull;
    if (loading_burst) {
        // Advance the game's resource / Param / ChrAsm state machine at 500 Hz (2.0 ms/step)
        // during active .dcx loading screens instead of sleeping 11.7..16.6 ms per step.
        constexpr u64 kFastLoadPeriodNs = 2'000'000ull;
        period_ns = (period_ns == 0ull) ? kFastLoadPeriodNs : std::min<u64>(period_ns, kFastLoadPeriodNs);
    } else if (period_ns > 0ull && BbStats::fg_active.load(std::memory_order_relaxed)) {
        period_ns = ComputeFgSimPeriodNs(period_ns);
    }
    const u8 reset_273 = *reinterpret_cast<const u8*>(pacer + 0x273);
    const u64 epoch = pacer_epoch.load(std::memory_order_relaxed);
    bool did_wait = false;

    // Close the previous simulation frame and perform driver-paced Low Latency sleep (NVIDIA
    // Reflex / AMD Anti-Lag 2) before our monotonic FPS-cap pacer. When GPU-bound or VSync-bound,
    // the driver sleeps just-in-time before GPU completion (cutting 1-2 queued frames); when
    // under-loaded at a cap, Reflex returns immediately (~10 us) and SleepUntilDeadlineImpl keeps
    // sub-0.15 ms jitter untouched. Skipped during 500 Hz Fast Load bursts.
    Vulkan::LowLatency::EndSimulationFrame();
    const bool ll_slept = Vulkan::LowLatency::Sleep(loading_burst);
    const u64 pace_now_ns = ll_slept ? MonotonicNs() : entry_ns;

    if (period_ns == 0 || reset_flag != 0 || reset_273 != 0 || pacer_prev_exit_ns == 0) {
        pacer_seen_epoch = epoch;
        pacer_last_period_ns = period_ns;
        pacer_next_deadline_ns = pace_now_ns;
        pacer_miss_streak = 0;
    } else {
        if (epoch != pacer_seen_epoch || period_ns != pacer_last_period_ns) {
            pacer_seen_epoch = epoch;
            pacer_last_period_ns = period_ns;
            pacer_next_deadline_ns = pacer_prev_exit_ns + period_ns;
        } else {
            pacer_next_deadline_ns += period_ns;
        }
        // Allow only subtle sub-ms phase-locked catch-up debt (<= 0.5 ms or 1/8 period) so tiny
        // timer jitter recovers smoothly without creating 2.2 ms rubberband micro-frames after a
        // 9..10 ms hitch.
        const u64 max_debt_ns = std::min<u64>(period_ns / 8ull, 500'000ull);
        if (pace_now_ns > pacer_next_deadline_ns + max_debt_ns) {
            pacer_next_deadline_ns = pace_now_ns;
        }
        if (pace_now_ns < pacer_next_deadline_ns) {
            did_wait = true;
            pacer_miss_streak = 0;
            SleepUntilDeadlineImpl(pace_now_ns, pacer_next_deadline_ns, 0);
        } else if (pacer_miss_streak < 100u) {
            ++pacer_miss_streak;
        }
    }

    const u64 exit_ns = did_wait ? MonotonicNs() : pace_now_ns;
    if (pacer_prev_exit_ns != 0 && exit_ns >= pacer_prev_exit_ns) {
        BbStats::last_main_frame_dt_ns.store(exit_ns - pacer_prev_exit_ns,
                                             std::memory_order_relaxed);
    }
    const u64 exit_tsc = (did_wait || ll_slept) ? __rdtsc() : entry_tsc;
    bool have_exit_cpu = have_entry_cpu;
    const u64 exit_cpu_ns = (did_wait || ll_slept) ? ThreadCpuNs(have_exit_cpu) : entry_cpu_ns;
    const u64 sleep_ns = exit_ns > entry_ns ? (exit_ns - entry_ns) : 0ull;
    if (did_wait && exit_ns > pacer_next_deadline_ns + 500'000ull) {
        const u64 over_ns = exit_ns - pacer_next_deadline_ns;
        if (!loading_burst && frame_draws >= 100 && over_ns >= 2'500'000ull) [[unlikely]] {
            const u64 target_sleep_ns =
                pacer_next_deadline_ns > pace_now_ns ? (pacer_next_deadline_ns - pace_now_ns) : 0ull;
            BbStats::LogHitch(
                "[BB-HITCH][PACER_SLEEP] frame=%u target=%.2f ms actual=%.2f ms over=+%.2f ms "
                "(draws=%llu)\n",
                BbStats::frame_number.load(std::memory_order_relaxed),
                double(target_sleep_ns) / 1e6, double(sleep_ns) / 1e6, double(over_ns) / 1e6,
                static_cast<unsigned long long>(frame_draws));
        }
        pacer_next_deadline_ns = exit_ns;
    }
    if (sleep_ns < 500'000'000ull) {
        BbStats::pacer_sleep_ns.fetch_add(sleep_ns, std::memory_order_relaxed);
        BbStats::pacer_frames.fetch_add(1, std::memory_order_relaxed);
    }

    const u64 cur_us = exit_ns / u64(1000);
    const u64 prev_us =
        pacer_prev_exit_ns != 0 ? std::min<u64>(cur_us, pacer_prev_exit_ns / u64(1000)) : cur_us;
    *reinterpret_cast<u64*>(pacer + 0x20) = prev_us;
    *reinterpret_cast<u64*>(pacer + 0x28) = cur_us;
    pacer_prev_exit_ns = exit_ns;
    pacer_prev_exit_cpu_ns = have_exit_cpu ? exit_cpu_ns : 0ull;
    pacer_prev_exit_tsc = exit_tsc;
    Vulkan::LowLatency::BeginSimulationFrame();
    return cur_us;
}

void MaybePrintClothTrace(float frame_dt) noexcept;

/// The pacer at the main loop's step (rdi: it): this frame's values.
void PS4_SYSV_ABI FrameStep(u64 pacer) {
#ifdef __linux__
    if (CleanPacerEnabled()) {
        static bool timerslack_set = false;
        if (!timerslack_set) {
            timerslack_set = true;
            prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
        }
    }
#endif
    const float measured = *reinterpret_cast<const float*>(pacer + 0x264);
    const float min_dt = pacer_wait_hooked ? MinSimDt : std::max(live->shortest, MinSimDt);
    float dt = std::clamp(std::isfinite(measured) ? measured : Frame30, min_dt, Frame30);
    if (!pacer_wait_hooked) {
        Vulkan::LowLatency::EndSimulationFrame();
        Vulkan::LowLatency::Sleep(false);
        Vulkan::LowLatency::BeginSimulationFrame();
        bool have_cpu_ns = false;
        const u64 cur_cpu_ns = ThreadCpuNs(have_cpu_ns);
        if (have_cpu_ns) {
            static u64 fallback_prev_cpu_ns = 0;
            if (fallback_prev_cpu_ns != 0 && cur_cpu_ns > fallback_prev_cpu_ns) {
                u64 delta_ns = cur_cpu_ns - fallback_prev_cpu_ns;
                if (delta_ns < 500'000'000ull) {
                    if (live->shortest > 0.0f && std::isfinite(measured) &&
                        measured <= live->shortest * 1.05f) {
                        const u64 spin_ns = 200'000ull;
                        delta_ns = delta_ns > spin_ns ? (delta_ns - spin_ns) : (delta_ns / 2);
                    }
                    BbStats::main_cpu_busy_ns.fetch_add(delta_ns, std::memory_order_relaxed);
                    BbStats::main_cpu_frames.fetch_add(1, std::memory_order_relaxed);
                }
            }
            fallback_prev_cpu_ns = cur_cpu_ns;
        }
    }
    // BB_FRAME_DT_SMOOTH=a (diagnostics, 0 < a < 1): the step follows the measured frame time by
    // that fraction a frame, so frame-to-frame jitter of the measurement does not reach the
    // simulation (time stays right on average).
    static const float smooth = [] {
        const char* env = std::getenv("BB_FRAME_DT_SMOOTH");
        const float a = env && *env ? std::strtof(env, nullptr) : 0.0f;
        return a > 0.0f && a < 1.0f ? a : 0.0f;
    }();
    if (smooth > 0.0f) {
        static float smoothed = 0.0f;
        smoothed = smoothed > 0.0f ? smoothed + smooth * (dt - smoothed) : dt;
        dt = std::clamp(smoothed, min_dt, Frame30);
    } else {
        static const bool cap_dt_stabilize = [] {
            const char* env = std::getenv("BB_CAP_DT_STABILIZE");
            return !env || !*env || !(env[0] == '0' && env[1] == '\0');
        }();
        if (cap_dt_stabilize && pacer_wait_hooked && live->shortest > 0.0f) {
            static double dt_debt_s = 0.0;
            static u64 dt_stab_epoch = 0;
            static float dt_stab_target = 0.0f;
            const u64 cur_ep = pacer_epoch.load(std::memory_order_relaxed);
            const u64 raw_ns = static_cast<u64>(
                std::llround(double(live->shortest) * 1'000'000'000.0));
            const u64 sim_ns =
                (raw_ns > 0ull && BbStats::fg_active.load(std::memory_order_relaxed))
                    ? ComputeFgSimPeriodNs(raw_ns)
                    : raw_ns;
            const float target_f =
                sim_ns > 0ull ? static_cast<float>(double(sim_ns) * 1e-9) : live->shortest;
            const double target_d = double(target_f);
            if (cur_ep == dt_stab_epoch && target_f == dt_stab_target &&
                pacer_miss_streak < 2u &&
                std::isfinite(measured) && measured > 0.0f &&
                double(dt) >= target_d * 0.35 && double(dt) <= target_d * 1.85) {
                const double want = double(dt) + dt_debt_s;
                double step = std::clamp(
                    target_d + (want - target_d) * 0.12, target_d * 0.88, target_d * 1.18);
                // Strictly conserve real time: never discard excess positive debt (prevents
                // slow-motion when GPU/CPU/VSync runs below the configured FPS cap).
                const double max_debt = target_d * 0.5;
                if (want - step > max_debt) {
                    step = want - max_debt;
                } else if (want - step < -max_debt) {
                    step = want + max_debt;
                }
                dt_debt_s = want - step;
                dt = std::clamp(static_cast<float>(step), min_dt, Frame30);
            } else {
                dt_debt_s = 0.0;
                dt_stab_epoch = cur_ep;
                dt_stab_target = target_f;
            }
        }
    }
    {
        const std::uint32_t dt_us = static_cast<std::uint32_t>(dt * 1'000'000.0f + 0.5f);
        std::uint32_t cur_min = BbStats::sim_dt_min_us.load(std::memory_order_relaxed);
        while (dt_us < cur_min && !BbStats::sim_dt_min_us.compare_exchange_weak(
                                      cur_min, dt_us, std::memory_order_relaxed)) {
        }
        std::uint32_t cur_max = BbStats::sim_dt_max_us.load(std::memory_order_relaxed);
        while (dt_us > cur_max && !BbStats::sim_dt_max_us.compare_exchange_weak(
                                      cur_max, dt_us, std::memory_order_relaxed)) {
        }
    }
    const float frames = dt * 30.0f; // the frame's length in 30 FPS frames
    const double frames_d = double(dt) * 30.0;
    live->dt = dt;
    const float decay95 = static_cast<float>(std::pow(0.95, frames_d));
    for (float& d : live->decay95) d = decay95;
    live->approach30 = static_cast<float>(-std::expm1(frames_d * std::log(1.0 - (1.0 / 30.0))));
    live->approach10 = static_cast<float>(-std::expm1(frames_d * std::log(0.9)));
    live->step_down = -0.1f * frames;
    live->step_up = 0.1f * frames;
    live->decay08 = static_cast<float>(std::pow(0.8, frames_d));
    live->grow12 = static_cast<float>(std::pow(1.2, frames_d));
    live->decay65 = static_cast<float>(std::pow(0.65, frames_d));
    live->slope_accel = frames;
    current_frames_per_30.store(frames, std::memory_order_relaxed);
    sim_frame_seq.fetch_add(1, std::memory_order_relaxed);
    live->decay60 = static_cast<float>(std::pow(0.6, frames_d));
    live->model_turn = model_turn_base * frames;
    live->dt_half = dt * 0.5f;
    // The cloth setups' range checks of Stiffness, Rigid Factor and Allowed Compression (0x80d800
    // only warns) and a few uses of 1.0 next to them: the game's 1.0. The cloth follows the frame
    // rate by its constraints' stiffness factor (ClothStiffnessFactor) and collision friction hooks.
    // BB_CLOTH_STIFFNESS=value (diagnostics): another value there.
    static const float cloth_override = [] {
        const char* env = std::getenv("BB_CLOTH_STIFFNESS");
        return env && *env ? std::strtof(env, nullptr) : -1.0f;
    }();
    live->cloth = cloth_override > 0.0f ? cloth_override : 1.0f;
    // Update cloth stiffness & friction from FrameStep only as a fallback when OnClothWorldStep
    // (0xfd54a2) is not hooked. When hooked, OnClothWorldStep updates them from hclWorld's exact
    // dt right after waitForStepComplete() (0x76a030), preventing worker-thread races and up to
    // 9x out-of-phase stiffness pumping on variable high-FPS frames.
    if (!cloth_world_synced) {
        UpdateClothNativeTiming(dt, false);
    }
    MaybePrintClothTrace(dt);
}

// ---- Cloth + Wind diagnostic telemetry (BB_CLOTH_TRACE=1; zero overhead when off) ----
bool ClothTraceEnabled() noexcept {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_CLOTH_TRACE");
        return env && env[0] == '1' && env[1] == '\0';
    }();
    return enabled;
}

struct ClothLayerTrace {
    std::atomic<const u8*> sim{nullptr};
    std::atomic<int> n{0};
    std::atomic<int> sub{0};
    std::atomic<float> gdamp{0.0f};
    std::atomic<float> edamp{0.0f};
    std::atomic<float> fvel{0.0f};
    std::atomic<int> ovr{0};
};

struct ClothTraceState {
    static constexpr int kMaxLayers = 6;
    static constexpr int kNumStages = 8; // 0=Int, 1..5=T1..T5, 6=Col, 7=Frc(pure ext force)
    std::atomic<u64> last_print_ns{0};
    std::atomic<std::uint32_t> sim_frames{0};
    std::atomic<std::uint32_t> world_step_calls{0};
    std::atomic<float> world_step_dt{0.0f};
    std::atomic<std::uint32_t> sim_op_calls{0};
    std::atomic<float> sim_op_dt{0.0f};
    std::atomic<float> sim_op_slowmo{0.0f};
    std::atomic<int> sim_op_substeps{0};
    std::atomic<int> sim_op_actions{0};
    std::atomic<float> sim_op_eff_damp{0.0f};
    std::atomic<std::uint32_t> fd4_wind_upd_calls{0};
    std::atomic<float> fd4_wind_dt{0.0f};
    std::atomic<float> fd4_wind_time{0.0f};
    std::atomic<float> fd4_wind_phase{0.0f};
    std::atomic<float> fd4_wind_speed{0.0f};
    std::atomic<std::uint32_t> area_wind_upd_calls{0};
    std::atomic<std::uint32_t> area_wind_apply_calls{0};
    std::atomic<float> area_wind_dt{0.0f};
    std::atomic<float> area_wind_period{0.0f};
    std::atomic<float> area_wind_timer{0.0f};
    std::atomic<float> area_wind_pitch{0.0f};
    std::atomic<float> area_wind_yaw{0.0f};
    std::atomic<std::uint32_t> sprj_wind_calls{0};
    std::atomic<std::uint32_t> point_wind_calls{0};
    std::atomic<std::uint32_t> dir_wind_calls{0};
    std::atomic<float> dir_wind_dt{0.0f};
    std::atomic<float> dir_wind_min{0.0f};
    std::atomic<float> dir_wind_max{0.0f};
    std::atomic<float> dir_wind_phase{0.0f};
    std::atomic<float> dir_wind_sfreq{0.0f};
    std::atomic<float> dir_wind_drag{0.0f};
    std::atomic<std::uint64_t> wind_spd_sum_mm_s{0};
    std::atomic<std::uint64_t> wind_spd_samples{0};
    std::atomic<std::uint64_t> wind_acc_sum_mm_s2{0};
    std::atomic<std::uint64_t> wind_dv01_sum_mm_s{0};
    std::atomic<std::uint64_t> wind_dv01_samples{0};
    std::atomic<std::uint32_t> simple_wind_calls{0};
    std::atomic<std::uint32_t> stiff_calls{0};
    std::atomic<std::uint32_t> stiff_type_mask{0};
    std::atomic<std::uint32_t> stiff_type_counts[24]{};
    std::atomic<int> stiff_last_mode{0};
    std::atomic<int> stiff_last_substeps{0};
    std::atomic<float> t5_stiff_min{999.0f};
    std::atomic<float> t5_stiff_max{-1.0f};
    std::atomic<std::uint32_t> t5_shape_calls{0};
    std::atomic<std::uint32_t> t5_sphere_calls{0};
    std::atomic<std::uint64_t> vel_samples{0};
    std::atomic<std::uint64_t> vel_sum_mm_s{0};
    std::atomic<std::uint32_t> vel_max_mm_s{0};
    std::atomic<std::uint64_t> pin_vel_samples{0};
    std::atomic<std::uint64_t> pin_vel_sum_mm_s{0};
    std::atomic<std::uint32_t> pin_step_max_um{0};
    std::atomic<std::uint32_t> pin_reversals{0};
    std::atomic<std::uint32_t> pin_rev_frames{0};
    std::atomic<std::uint32_t> xfer_calls{0};
    std::atomic<std::uint64_t> xfer_um_sum{0};
    std::atomic<std::uint32_t> xfer_max_um{0};
    std::atomic<std::uint32_t> vel_max_particles{0};
    std::atomic<std::uint32_t> vel_reversals{0};
    std::atomic<std::uint32_t> vel_rev_frames{0};
    std::atomic<std::uint32_t> prestep_override_calls{0};
    std::atomic<std::uint32_t> prestep_rescale_calls{0};
    std::atomic<float> prestep_ratio_min{999.0f};
    std::atomic<float> prestep_ratio_max{0.0f};
    std::atomic<std::uint32_t> stage_calls[kNumStages]{};
    std::atomic<std::uint64_t> stage_um_sum[kNumStages]{};
    std::atomic<std::uint64_t> stage_vel_mm_s_sum[kNumStages]{};
    std::atomic<std::uint32_t> stage_max_um[kNumStages]{};
    ClothLayerTrace layers[kMaxLayers]{};
    std::atomic<std::uint32_t> f30_steps{0};
    std::atomic<std::uint32_t> f30_skips{0};
    std::atomic<std::uint32_t> f30_resets{0};
    std::atomic<std::uint32_t> f30_rigid_ok{0};
    std::atomic<std::uint32_t> f30_prestep_calls{0};
};
ClothTraceState cloth_trace;
std::atomic<std::uint32_t> cloth_tick30_seq{0};
std::atomic<std::uint32_t> cloth_t5_last_tick30{0};
std::atomic<bool> cloth_t5_pulse_active{true};

void TraceUpdateAtomicMaxU32(std::atomic<std::uint32_t>& dst, std::uint32_t val) noexcept {
    std::uint32_t cur = dst.load(std::memory_order_relaxed);
    while (val > cur && !dst.compare_exchange_weak(cur, val, std::memory_order_relaxed)) {
    }
}

int TraceSnapshotFreeParticles(const u8* sim, float* pre_xyz, int max_n) noexcept {
    if (!sim || !pre_xyz || max_n <= 0) return 0;
    const float* cur = *reinterpret_cast<const float* const*>(sim + 0x18);
    const std::int32_t n = *reinterpret_cast<const std::int32_t*>(sim + 0x20);
    if (!cur || n <= 0 || n > 4096) return 0;
    const int m = std::min<int>(n, max_n);
    for (int i = 0; i < m; ++i) {
        pre_xyz[i * 3 + 0] = cur[i * 4 + 0];
        pre_xyz[i * 3 + 1] = cur[i * 4 + 1];
        pre_xyz[i * 3 + 2] = cur[i * 4 + 2];
    }
    return m;
}

void TraceRecordStageDelta(int stage, const u8* sim, const float* pre_xyz, int snap_n, float sdt) noexcept {
    if (stage < 0 || stage >= ClothTraceState::kNumStages || !sim || !pre_xyz || snap_n <= 0) return;
    const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
    const float* cur = *reinterpret_cast<const float* const*>(sim + 0x18);
    if (!sim_data || !cur) return;
    const float* pdatas = *reinterpret_cast<const float* const*>(sim_data + 0x38);
    const std::int32_t n_pdatas = *reinterpret_cast<const std::int32_t*>(sim_data + 0x40);
    if (!pdatas || n_pdatas < snap_n) return;
    const float safe_sdt = (sdt > 1e-5f && sdt < 0.2f && std::isfinite(sdt)) ? sdt : (1.0f / 30.0f);
    double sum_m = 0.0;
    double max_m = 0.0;
    int free_cnt = 0;
    for (int i = 0; i < snap_n; ++i) {
        if (!(pdatas[i * 4 + 1] > 0.0f)) continue;
        const float dx = cur[i * 4 + 0] - pre_xyz[i * 3 + 0];
        const float dy = cur[i * 4 + 1] - pre_xyz[i * 3 + 1];
        const float dz = cur[i * 4 + 2] - pre_xyz[i * 3 + 2];
        const double d = std::sqrt(double(dx) * double(dx) + double(dy) * double(dy) + double(dz) * double(dz));
        if (std::isfinite(d) && d < 2.0) {
            sum_m += d;
            if (d > max_m) max_m = d;
            ++free_cnt;
        }
    }
    if (free_cnt > 0) {
        const double avg_m = sum_m / double(free_cnt);
        cloth_trace.stage_calls[stage].fetch_add(1u, std::memory_order_relaxed);
        cloth_trace.stage_um_sum[stage].fetch_add(
            static_cast<std::uint64_t>(avg_m * 1'000'000.0), std::memory_order_relaxed);
        cloth_trace.stage_vel_mm_s_sum[stage].fetch_add(
            static_cast<std::uint64_t>((avg_m / double(safe_sdt)) * 1000.0), std::memory_order_relaxed);
        TraceUpdateAtomicMaxU32(cloth_trace.stage_max_um[stage],
                                static_cast<std::uint32_t>(std::min(max_m * 1'000'000.0, 999999.0)));
    }
}

void PS4_SYSV_ABI ProbeWorldStep(const void* /*self*/, float dt) noexcept {
    cloth_trace.world_step_calls.fetch_add(1u, std::memory_order_relaxed);
    cloth_trace.world_step_dt.store(dt, std::memory_order_relaxed);
}

void PS4_SYSV_ABI ProbeSimOp(const u8* op, const u8* ctx, float dt) noexcept {
    cloth_trace.sim_op_calls.fetch_add(1u, std::memory_order_relaxed);
    cloth_trace.sim_op_dt.store(dt, std::memory_order_relaxed);
    if (op && ctx) {
        const std::int32_t idx = *reinterpret_cast<const std::int32_t*>(op + 0x1c);
        const u8* const* arr = *reinterpret_cast<const u8* const* const*>(ctx + 0x20);
        if (arr && idx >= 0 && idx < 64) {
            const u8* sim = arr[idx];
            if (sim) {
                const float eff_damp = *reinterpret_cast<const float*>(sim + 0x10c);
                using GetSimCfgFn = const u8*(PS4_SYSV_ABI*)(const u8*);
                const auto get_sim_cfg = reinterpret_cast<GetSimCfgFn>(image_base + 0x748c50);
                const u8* sim_cfg = get_sim_cfg(sim);
                std::int32_t substeps = sim_cfg ? *reinterpret_cast<const std::int32_t*>(sim_cfg + 0x18) : 0;
                if (substeps <= 0) {
                    substeps = *reinterpret_cast<const std::int32_t*>(op + 0x20);
                }
                const float gdamp = sim_cfg ? *reinterpret_cast<const float*>(sim_cfg + 0x10) : 0.0f;
                const int has_ovr = (*reinterpret_cast<const void* const*>(sim + 0x1d8) != nullptr) ? 1 : 0;

                cloth_trace.sim_op_slowmo.store(
                    *reinterpret_cast<const float*>(sim + 0x1d0), std::memory_order_relaxed);
                cloth_trace.sim_op_actions.store(
                    *reinterpret_cast<const std::int32_t*>(sim + 0x170), std::memory_order_relaxed);
                cloth_trace.sim_op_eff_damp.store(eff_damp, std::memory_order_relaxed);
                cloth_trace.sim_op_substeps.store(substeps, std::memory_order_relaxed);

                // Measure FREE-particle-only (invMass > 0) vs PINNED velocity (mm/s) and reversals
                const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
                const float* pdatas =
                    sim_data ? *reinterpret_cast<const float* const*>(sim_data + 0x38) : nullptr;
                const std::int32_t n_pdatas =
                    sim_data ? *reinterpret_cast<const std::int32_t*>(sim_data + 0x40) : 0;
                const float* cur = *reinterpret_cast<const float* const*>(sim + 0x18);
                const float* prv = *reinterpret_cast<const float* const*>(sim + 0x28);
                const std::int32_t n = *reinterpret_cast<const std::int32_t*>(sim + 0x20);
                const float prev_sdt = *reinterpret_cast<const float*>(sim + 0x108);
                if (cur && prv && pdatas && n_pdatas >= n && n >= 8 && n <= 4096 && prev_sdt > 1e-5f &&
                    prev_sdt < 0.1f) {
                    const float inv_dt = 1.0f / prev_sdt;
                    double free_speed_sum = 0.0;
                    double free_speed_max = 0.0;
                    int free_n = 0;
                    int probe_free_idx = -1;
                    for (int i = 0; i < n; ++i) {
                        const float dx = cur[i * 4 + 0] - prv[i * 4 + 0];
                        const float dy = cur[i * 4 + 1] - prv[i * 4 + 1];
                        const float dz = cur[i * 4 + 2] - prv[i * 4 + 2];
                        const float d2 = dx * dx + dy * dy + dz * dz;
                        if (pdatas[i * 4 + 1] > 0.0f) {
                            probe_free_idx = i;
                            if (d2 >= 0.0f && d2 < 4.0f) {
                                const double spd = std::sqrt(double(d2)) * double(inv_dt);
                                free_speed_sum += spd;
                                if (spd > free_speed_max) free_speed_max = spd;
                                ++free_n;
                            }
                        }
                    }
                    if (free_n > 0) {
                        const double avg_m_s = free_speed_sum / double(free_n);
                        cloth_trace.vel_sum_mm_s.fetch_add(
                            static_cast<std::uint64_t>(avg_m_s * 1000.0), std::memory_order_relaxed);
                        cloth_trace.vel_samples.fetch_add(1u, std::memory_order_relaxed);
                        TraceUpdateAtomicMaxU32(
                            cloth_trace.vel_max_mm_s,
                            static_cast<std::uint32_t>(std::min(free_speed_max * 1000.0, 999999.0)));
                        TraceUpdateAtomicMaxU32(cloth_trace.vel_max_particles,
                                                static_cast<std::uint32_t>(n));

                        // Per-layer lock-free table (up to 6 active cloths)
                        for (int li = 0; li < ClothTraceState::kMaxLayers; ++li) {
                            const u8* slot = cloth_trace.layers[li].sim.load(std::memory_order_relaxed);
                            if (slot == sim) {
                                cloth_trace.layers[li].n.store(n, std::memory_order_relaxed);
                                cloth_trace.layers[li].sub.store(substeps, std::memory_order_relaxed);
                                cloth_trace.layers[li].gdamp.store(gdamp, std::memory_order_relaxed);
                                cloth_trace.layers[li].edamp.store(eff_damp, std::memory_order_relaxed);
                                cloth_trace.layers[li].fvel.store(static_cast<float>(avg_m_s),
                                                                  std::memory_order_relaxed);
                                cloth_trace.layers[li].ovr.store(has_ovr, std::memory_order_relaxed);
                                break;
                            }
                            if (slot == nullptr) {
                                const u8* expected = nullptr;
                                if (cloth_trace.layers[li].sim.compare_exchange_strong(
                                        expected, sim, std::memory_order_relaxed)) {
                                    cloth_trace.layers[li].n.store(n, std::memory_order_relaxed);
                                    cloth_trace.layers[li].sub.store(substeps, std::memory_order_relaxed);
                                    cloth_trace.layers[li].gdamp.store(gdamp, std::memory_order_relaxed);
                                    cloth_trace.layers[li].edamp.store(eff_damp, std::memory_order_relaxed);
                                    cloth_trace.layers[li].fvel.store(static_cast<float>(avg_m_s),
                                                                      std::memory_order_relaxed);
                                    cloth_trace.layers[li].ovr.store(has_ovr, std::memory_order_relaxed);
                                    break;
                                }
                            }
                        }

                        // Track velocity reversals on the highest-index FREE particle AND real frame-to-frame
                        // motion of PINNED (skeleton-driven!) particles on the largest cloth (Hunter coat/cape)
                        static std::atomic<const u8*> tracked_sim{nullptr};
                        static std::atomic<int> tracked_n{0};
                        static std::atomic<float> prev_vx{0.0f}, prev_vy{0.0f}, prev_vz{0.0f};
                        static std::mutex pin_trace_mtx;
                        static const u8* pin_prev_sim = nullptr;
                        static int pin_prev_n = 0;
                        static float pin_prev_xyz[512 * 3]{};
                        static float pin_prev_v[3] = {0.0f, 0.0f, 0.0f};
                        const int cur_tn = tracked_n.load(std::memory_order_relaxed);
                        if (n > cur_tn) {
                            tracked_n.store(n, std::memory_order_relaxed);
                            tracked_sim.store(sim, std::memory_order_relaxed);
                        }
                        if (sim == tracked_sim.load(std::memory_order_relaxed)) {
                            if (probe_free_idx >= 0) {
                                const int pi = probe_free_idx;
                                const float vx = (cur[pi * 4 + 0] - prv[pi * 4 + 0]) * inv_dt;
                                const float vy = (cur[pi * 4 + 1] - prv[pi * 4 + 1]) * inv_dt;
                                const float vz = (cur[pi * 4 + 2] - prv[pi * 4 + 2]) * inv_dt;
                                const float pvx = prev_vx.exchange(vx, std::memory_order_relaxed);
                                const float pvy = prev_vy.exchange(vy, std::memory_order_relaxed);
                                const float pvz = prev_vz.exchange(vz, std::memory_order_relaxed);
                                const float dot = vx * pvx + vy * pvy + vz * pvz;
                                if (dot < -1e-4f) {
                                    cloth_trace.vel_reversals.fetch_add(1u, std::memory_order_relaxed);
                                }
                                cloth_trace.vel_rev_frames.fetch_add(1u, std::memory_order_relaxed);
                            }
                            const int m = std::min<int>(n, 512);
                            const float fdt = (dt > 1e-5f && dt < 0.2f && std::isfinite(dt)) ? dt : prev_sdt;
                            std::lock_guard<std::mutex> lk(pin_trace_mtx);
                            if (pin_prev_sim == sim && pin_prev_n == m) {
                                double p_spd_sum = 0.0;
                                double p_max_d = 0.0;
                                double sum_dx = 0.0, sum_dy = 0.0, sum_dz = 0.0;
                                int p_cnt = 0;
                                for (int i = 0; i < m; ++i) {
                                    if (pdatas[i * 4 + 1] > 0.0f) continue;
                                    const float dx = cur[i * 4 + 0] - pin_prev_xyz[i * 3 + 0];
                                    const float dy = cur[i * 4 + 1] - pin_prev_xyz[i * 3 + 1];
                                    const float dz = cur[i * 4 + 2] - pin_prev_xyz[i * 3 + 2];
                                    const double d = std::sqrt(double(dx) * double(dx) +
                                                               double(dy) * double(dy) +
                                                               double(dz) * double(dz));
                                    if (std::isfinite(d) && d < 2.0) {
                                        p_spd_sum += d / double(fdt);
                                        if (d > p_max_d) p_max_d = d;
                                        sum_dx += dx;
                                        sum_dy += dy;
                                        sum_dz += dz;
                                        ++p_cnt;
                                    }
                                }
                                if (p_cnt > 0) {
                                    cloth_trace.pin_vel_sum_mm_s.fetch_add(
                                        static_cast<std::uint64_t>((p_spd_sum / double(p_cnt)) * 1000.0),
                                        std::memory_order_relaxed);
                                    cloth_trace.pin_vel_samples.fetch_add(1u, std::memory_order_relaxed);
                                    TraceUpdateAtomicMaxU32(
                                        cloth_trace.pin_step_max_um,
                                        static_cast<std::uint32_t>(std::min(p_max_d * 1'000'000.0, 999999.0)));
                                    const float pvx = static_cast<float>((sum_dx / double(p_cnt)) / double(fdt));
                                    const float pvy = static_cast<float>((sum_dy / double(p_cnt)) / double(fdt));
                                    const float pvz = static_cast<float>((sum_dz / double(p_cnt)) / double(fdt));
                                    const float dot =
                                        pvx * pin_prev_v[0] + pvy * pin_prev_v[1] + pvz * pin_prev_v[2];
                                    if (dot < -1e-6f) {
                                        cloth_trace.pin_reversals.fetch_add(1u, std::memory_order_relaxed);
                                    }
                                    cloth_trace.pin_rev_frames.fetch_add(1u, std::memory_order_relaxed);
                                    pin_prev_v[0] = pvx;
                                    pin_prev_v[1] = pvy;
                                    pin_prev_v[2] = pvz;
                                }
                            }
                            pin_prev_sim = sim;
                            pin_prev_n = m;
                            for (int i = 0; i < m; ++i) {
                                pin_prev_xyz[i * 3 + 0] = cur[i * 4 + 0];
                                pin_prev_xyz[i * 3 + 1] = cur[i * 4 + 1];
                                pin_prev_xyz[i * 3 + 2] = cur[i * 4 + 2];
                            }
                        }
                    }
                }
            }
        }
    }
}

void PS4_SYSV_ABI ProbeFd4WindUpd(const u8* wind, float dt) noexcept {
    cloth_trace.fd4_wind_upd_calls.fetch_add(1u, std::memory_order_relaxed);
    cloth_trace.fd4_wind_dt.store(dt, std::memory_order_relaxed);
    if (wind) {
        cloth_trace.fd4_wind_time.store(
            *reinterpret_cast<const float*>(wind + 0x1f4), std::memory_order_relaxed);
        cloth_trace.fd4_wind_phase.store(
            *reinterpret_cast<const float*>(wind + 0x1b4), std::memory_order_relaxed);
        cloth_trace.fd4_wind_speed.store(
            *reinterpret_cast<const float*>(wind + 0x1a8), std::memory_order_relaxed);
    }
}

void PS4_SYSV_ABI ProbeAreaWindUpd(const u8* wind, const u8* step_info) noexcept {
    cloth_trace.area_wind_upd_calls.fetch_add(1u, std::memory_order_relaxed);
    if (step_info) {
        cloth_trace.area_wind_dt.store(
            *reinterpret_cast<const float*>(step_info + 0x8), std::memory_order_relaxed);
    }
    if (wind) {
        cloth_trace.area_wind_period.store(
            *reinterpret_cast<const float*>(wind + 0x1bc), std::memory_order_relaxed);
        cloth_trace.area_wind_timer.store(
            *reinterpret_cast<const float*>(wind + 0x1c0), std::memory_order_relaxed);
        cloth_trace.area_wind_pitch.store(
            *reinterpret_cast<const float*>(wind + 0x1b4), std::memory_order_relaxed);
        cloth_trace.area_wind_yaw.store(
            *reinterpret_cast<const float*>(wind + 0x1b8), std::memory_order_relaxed);
    }
}

void PS4_SYSV_ABI ProbeAreaWindApply() noexcept {
    cloth_trace.area_wind_apply_calls.fetch_add(1u, std::memory_order_relaxed);
}

void PS4_SYSV_ABI ProbeSprjWindApply() noexcept {
    cloth_trace.sprj_wind_calls.fetch_add(1u, std::memory_order_relaxed);
}

void PS4_SYSV_ABI ProbePointWindApply() noexcept {
    cloth_trace.point_wind_calls.fetch_add(1u, std::memory_order_relaxed);
}

void PS4_SYSV_ABI ProbeDirWindApply(const u8* self, const u8* /*sim*/, const u8* /*forces*/,
                                    const u8* dir, const float* params, float dt) noexcept {
    cloth_trace.dir_wind_calls.fetch_add(1u, std::memory_order_relaxed);
    cloth_trace.dir_wind_dt.store(dt, std::memory_order_relaxed);
    if (self) {
        cloth_trace.dir_wind_drag.store(
            *reinterpret_cast<const float*>(self + BbClothInterp::SprjWindActionOffsets::kDragCoeff),
            std::memory_order_relaxed);
    }
    if (dir) {
        const float* dv = reinterpret_cast<const float*>(dir);
        const float w2 = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
        if (w2 >= 0.0f && std::isfinite(w2)) {
            cloth_trace.wind_spd_sum_mm_s.fetch_add(
                static_cast<std::uint64_t>(std::sqrt(double(w2)) * 1000.0),
                std::memory_order_relaxed);
            cloth_trace.wind_spd_samples.fetch_add(1u, std::memory_order_relaxed);
        }
    }
    if (params) {
        cloth_trace.dir_wind_min.store(params[0], std::memory_order_relaxed);
        cloth_trace.dir_wind_max.store(params[1], std::memory_order_relaxed);
        cloth_trace.dir_wind_phase.store(params[2], std::memory_order_relaxed);
        cloth_trace.dir_wind_sfreq.store(params[3], std::memory_order_relaxed);
    }
}

void PS4_SYSV_ABI ProbeSimpleWindApply() noexcept {
    cloth_trace.simple_wind_calls.fetch_add(1u, std::memory_order_relaxed);
}

void MaybePrintClothTrace(float frame_dt) noexcept {
    if (!ClothTraceEnabled()) return;
    cloth_trace.sim_frames.fetch_add(1u, std::memory_order_relaxed);
    const u64 now = MonotonicNs();
    const u64 prev = cloth_trace.last_print_ns.load(std::memory_order_relaxed);
    if (prev == 0) {
        cloth_trace.last_print_ns.store(now, std::memory_order_relaxed);
        return;
    }
    const double elapsed = double(now - prev) * 1e-9;
    if (elapsed < 1.0) return;
    cloth_trace.last_print_ns.store(now, std::memory_order_relaxed);

    const std::uint32_t nf = cloth_trace.sim_frames.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t nw = cloth_trace.world_step_calls.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t ns = cloth_trace.sim_op_calls.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t nfw = cloth_trace.fd4_wind_upd_calls.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t nau = cloth_trace.area_wind_upd_calls.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t naa = cloth_trace.area_wind_apply_calls.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t nsw = cloth_trace.sprj_wind_calls.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t npw = cloth_trace.point_wind_calls.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t ndw = cloth_trace.dir_wind_calls.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t nsm = cloth_trace.simple_wind_calls.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t nst = cloth_trace.stiff_calls.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t mask = cloth_trace.stiff_type_mask.exchange(0u, std::memory_order_relaxed);
    const float t5_min = cloth_trace.t5_stiff_min.exchange(999.0f, std::memory_order_relaxed);
    const float t5_max = cloth_trace.t5_stiff_max.exchange(-1.0f, std::memory_order_relaxed);
    const std::uint32_t t5_shp = cloth_trace.t5_shape_calls.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t t5_sph = cloth_trace.t5_sphere_calls.exchange(0u, std::memory_order_relaxed);
    const std::uint64_t vsamp = cloth_trace.vel_samples.exchange(0u, std::memory_order_relaxed);
    const std::uint64_t vsum = cloth_trace.vel_sum_mm_s.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t vmax_mm = cloth_trace.vel_max_mm_s.exchange(0u, std::memory_order_relaxed);
    const std::uint64_t psamp = cloth_trace.pin_vel_samples.exchange(0u, std::memory_order_relaxed);
    const std::uint64_t psum = cloth_trace.pin_vel_sum_mm_s.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t pmax_um = cloth_trace.pin_step_max_um.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t prev_cnt = cloth_trace.pin_reversals.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t prev_f = cloth_trace.pin_rev_frames.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t xfer_c = cloth_trace.xfer_calls.exchange(0u, std::memory_order_relaxed);
    const std::uint64_t xfer_um = cloth_trace.xfer_um_sum.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t xfer_mx = cloth_trace.xfer_max_um.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t vmaxn = cloth_trace.vel_max_particles.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t vrev = cloth_trace.vel_reversals.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t vrf = cloth_trace.vel_rev_frames.exchange(0u, std::memory_order_relaxed);
    const std::uint64_t wspd_n = cloth_trace.wind_spd_samples.exchange(0u, std::memory_order_relaxed);
    const std::uint64_t wspd_sum = cloth_trace.wind_spd_sum_mm_s.exchange(0u, std::memory_order_relaxed);
    const std::uint64_t wacc_sum = cloth_trace.wind_acc_sum_mm_s2.exchange(0u, std::memory_order_relaxed);
    const std::uint64_t wdv_n = cloth_trace.wind_dv01_samples.exchange(0u, std::memory_order_relaxed);
    const std::uint64_t wdv_sum = cloth_trace.wind_dv01_sum_mm_s.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t pre_ovr = cloth_trace.prestep_override_calls.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t pre_rsc = cloth_trace.prestep_rescale_calls.exchange(0u, std::memory_order_relaxed);
    const float pre_rmin = cloth_trace.prestep_ratio_min.exchange(999.0f, std::memory_order_relaxed);
    const float pre_rmax = cloth_trace.prestep_ratio_max.exchange(0.0f, std::memory_order_relaxed);
    const std::uint32_t f30_st = cloth_trace.f30_steps.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t f30_sk = cloth_trace.f30_skips.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t f30_rs = cloth_trace.f30_resets.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t f30_rg = cloth_trace.f30_rigid_ok.exchange(0u, std::memory_order_relaxed);
    const std::uint32_t f30_pr = cloth_trace.f30_prestep_calls.exchange(0u, std::memory_order_relaxed);
    const double avg_vel_m_s = vsamp > 0 ? (double(vsum) / double(vsamp)) * 0.001 : 0.0;
    const double avg_pin_m_s = psamp > 0 ? (double(psum) / double(psamp)) * 0.001 : 0.0;
    const double avg_wspd = wspd_n > 0 ? (double(wspd_sum) / double(wspd_n)) * 0.001 : 0.0;
    const double avg_wacc = wspd_n > 0 ? (double(wacc_sum) / double(wspd_n)) * 0.001 : 0.0;
    const double avg_wdv = wdv_n > 0 ? (double(wdv_sum) / double(wdv_n)) * 0.001 : 0.0;
    const double rev_hz = double(vrev) / elapsed;
    char tbuf[192];
    int tpos = 0;
    tbuf[0] = '\0';
    for (int t = 0; t < 24; ++t) {
        const std::uint32_t c = cloth_trace.stiff_type_counts[t].exchange(0u, std::memory_order_relaxed);
        if (c > 0 && tpos < int(sizeof(tbuf)) - 24) {
            tpos += std::snprintf(tbuf + tpos, sizeof(tbuf) - std::size_t(tpos), "%s%d:%u",
                                  tpos ? "," : "", t, c);
        }
    }
    std::printf(
        "[CLOTH-TRACE] fps=%.1f live_dt=%.2fms k_scale=%.4f | "
        "hclWorld=%u(dt=%.2fms) simOp=%u(dt=%.2fms sub=%d slow=%.2f act=%d damp=%.4f) | "
        "F30(pre=%u step=%u skip=%u rst=%u rigid=%u) | "
        "FD4Wind=%u(dt=%.2fms t=%.2f ph=%.2f spd=%.2f) | "
        "AreaW=%u/%u(dt=%.2fms per=%.2f tmr=%.2f p=%.2f y=%.2f) | "
        "SprjW=%u PtW=%u DirW=%u(dt=%.2fms |v|=%.2f dv01=%.2f drag=%.2f acc=%.2f ph=%.1f) SimpW=%u | "
        "Stiff=%u(mode=%d sub=%d mask=0x%x [%s]) | "
        "T5(k=%.2f..%.2f shp=%u sph=%u) FreeVel=%.3f(max=%.3f) PinVel=%.3f Rev=%.1fHz(%u/%u n=%u)\n",
        double(nf) / elapsed, frame_dt * 1000.0f, live ? live->cloth_stiffness_scale : 1.0f,
        nw, cloth_trace.world_step_dt.load(std::memory_order_relaxed) * 1000.0f,
        ns, cloth_trace.sim_op_dt.load(std::memory_order_relaxed) * 1000.0f,
        cloth_trace.sim_op_substeps.load(std::memory_order_relaxed),
        cloth_trace.sim_op_slowmo.load(std::memory_order_relaxed),
        cloth_trace.sim_op_actions.load(std::memory_order_relaxed),
        cloth_trace.sim_op_eff_damp.load(std::memory_order_relaxed),
        f30_pr, f30_st, f30_sk, f30_rs, f30_rg,
        nfw, cloth_trace.fd4_wind_dt.load(std::memory_order_relaxed) * 1000.0f,
        cloth_trace.fd4_wind_time.load(std::memory_order_relaxed),
        cloth_trace.fd4_wind_phase.load(std::memory_order_relaxed),
        cloth_trace.fd4_wind_speed.load(std::memory_order_relaxed),
        nau, naa, cloth_trace.area_wind_dt.load(std::memory_order_relaxed) * 1000.0f,
        cloth_trace.area_wind_period.load(std::memory_order_relaxed),
        cloth_trace.area_wind_timer.load(std::memory_order_relaxed),
        cloth_trace.area_wind_pitch.load(std::memory_order_relaxed),
        cloth_trace.area_wind_yaw.load(std::memory_order_relaxed),
        nsw, npw, ndw, cloth_trace.dir_wind_dt.load(std::memory_order_relaxed) * 1000.0f,
        avg_wspd, avg_wdv, cloth_trace.dir_wind_drag.load(std::memory_order_relaxed), avg_wacc,
        cloth_trace.dir_wind_phase.load(std::memory_order_relaxed),
        nsm, nst, cloth_trace.stiff_last_mode.load(std::memory_order_relaxed),
        cloth_trace.stiff_last_substeps.load(std::memory_order_relaxed), mask, tbuf,
        t5_max >= 0.0f ? t5_min : 0.0f, t5_max >= 0.0f ? t5_max : 0.0f, t5_shp, t5_sph,
        avg_vel_m_s, double(vmax_mm) * 0.001, avg_pin_m_s, rev_hz, vrev, vrf, vmaxn);

    if (ns > 0) {
        static constexpr const char* kStageNames[ClothTraceState::kNumStages] = {
            "Int", "T1", "T2", "T3", "T4", "T5", "Col", "Frc"};
        char sbuf[512];
        int spos = 0;
        sbuf[0] = '\0';
        for (int s = 0; s < ClothTraceState::kNumStages; ++s) {
            const std::uint32_t c = cloth_trace.stage_calls[s].exchange(0u, std::memory_order_relaxed);
            const std::uint64_t um = cloth_trace.stage_um_sum[s].exchange(0u, std::memory_order_relaxed);
            const std::uint64_t vm =
                cloth_trace.stage_vel_mm_s_sum[s].exchange(0u, std::memory_order_relaxed);
            const std::uint32_t mx = cloth_trace.stage_max_um[s].exchange(0u, std::memory_order_relaxed);
            if (c > 0 && spos < int(sizeof(sbuf)) - 64) {
                const double avg_mm = (double(um) / double(c)) * 0.001;
                const double avg_v = (double(vm) / double(c)) * 0.001;
                const double max_mm = double(mx) * 0.001;
                spos += std::snprintf(sbuf + spos, sizeof(sbuf) - std::size_t(spos),
                                      "%s%s:%.3fmm(mx=%.2f,v=%.3f)", spos ? " " : "", kStageNames[s],
                                      avg_mm, max_mm, avg_v);
            }
        }
        char lbuf[320];
        int lpos = 0;
        lbuf[0] = '\0';
        for (int li = 0; li < ClothTraceState::kMaxLayers; ++li) {
            if (cloth_trace.layers[li].sim.load(std::memory_order_relaxed) != nullptr &&
                lpos < int(sizeof(lbuf)) - 56) {
                lpos += std::snprintf(
                    lbuf + lpos, sizeof(lbuf) - std::size_t(lpos), "%s[n=%d s=%d gd=%.2f ed=%.4f fv=%.3f%s]",
                    lpos ? " " : "", cloth_trace.layers[li].n.load(std::memory_order_relaxed),
                    cloth_trace.layers[li].sub.load(std::memory_order_relaxed),
                    cloth_trace.layers[li].gdamp.load(std::memory_order_relaxed),
                    cloth_trace.layers[li].edamp.load(std::memory_order_relaxed),
                    cloth_trace.layers[li].fvel.load(std::memory_order_relaxed),
                    cloth_trace.layers[li].ovr.load(std::memory_order_relaxed) ? " OVR" : "");
            }
        }
        const double xfer_avg_mm = xfer_c > 0 ? (double(xfer_um) / double(xfer_c)) * 0.001 : 0.0;
        std::printf(
            "[CLOTH-STAGES] pre(ovr=%u rsc=%u r=%.2f..%.2f) Pin(v=%.3f mx=%.2fmm rev=%.1fHz %u/%u) "
            "Xfer(c=%u %.3fmm mx=%.2f) | %s | %s\n",
            pre_ovr, pre_rsc, pre_rmax > 0.0f ? pre_rmin : 1.0f, pre_rmax > 0.0f ? pre_rmax : 1.0f,
            avg_pin_m_s, double(pmax_um) * 0.001, double(prev_cnt) / elapsed, prev_cnt, prev_f, xfer_c,
            xfer_avg_mm, double(xfer_mx) * 0.001, sbuf, lbuf);
    }
    std::fflush(stdout);
}

// ---- Cloth: the constraints' stiffness factor per substep ----
/// Havok's stiffness factor of a constraint set for one substep (0x1167a80, replaced): `mode` 2
/// adapts stiffness to solves per step `(substeps x a x b)^-1.725`, type 8 uses iteration share,
/// and types 5 (`hclLocalRangeConstraintSet`, `0x6fb220`) / 10 only run on the last substep.
/// Verified `HclConstraintSetType` `[set + 0x18]` classes in `eboot.bin` + `bd_*.partsbnd.dcx`:
///   - `type = 2` (`hclStretchLinkConstraintSet`): SIGGRAPH 2012 LRA `1.0f` max-length tether — MUST
///     stay unscaled (`game`) across all FPS or coats stretch excessively at high FPS.
///   - `type = 5` (`hclLocalRangeConstraintSet`, solver `0x2553210`): `m_minNormalDistance` /
///     `m_maxNormalDistance` pushout is unscaled on every solve (zero torso/back clipping when
///     stepped natively every frame), while `m_stiffness` (`[set + 0x34]`) governs `m_maximumDistance`:
///     hard `1.0f` cages stay unscaled (`1.0f`) and soft `< 0.999f` ranges (`Local Range outer` `0.5f`)
///     use exact continuous-time exponential relaxation `(-expm1(frames * log(1 - s))) / s`.
///   - `types = 1, 16, 20` (`StandardLink`, `BendStiffnessMx`, `CompressibleLink`): elastic springs
///     scale by `frames^(-cloth_exponent) = frames^1.725` (`0x48d7bbc`).
float PS4_SYSV_ABI ClothStiffnessFactor(const u8* set, int mode, int iteration, int substeps, float a,
                                        float b) {
    const std::uint32_t type = *reinterpret_cast<const std::uint32_t*>(set + 0x18);
    const float set_stiff =
        (type == static_cast<std::uint32_t>(BbClothInterp::HclConstraintSetType::kLocalRange))
            ? *reinterpret_cast<const float*>(set + 0x34)
            : 1.0f;
    if (ClothTraceEnabled()) {
        cloth_trace.stiff_calls.fetch_add(1u, std::memory_order_relaxed);
        cloth_trace.stiff_last_mode.store(mode, std::memory_order_relaxed);
        cloth_trace.stiff_last_substeps.store(substeps, std::memory_order_relaxed);
        if (type < 24u) {
            cloth_trace.stiff_type_mask.fetch_or(1u << type, std::memory_order_relaxed);
            cloth_trace.stiff_type_counts[type].fetch_add(1u, std::memory_order_relaxed);
        }
        if (type == 5u) {
            const float st = set_stiff;
            float cur_min = cloth_trace.t5_stiff_min.load(std::memory_order_relaxed);
            while (st < cur_min && !cloth_trace.t5_stiff_min.compare_exchange_weak(
                                       cur_min, st, std::memory_order_relaxed)) {
            }
            float cur_max = cloth_trace.t5_stiff_max.load(std::memory_order_relaxed);
            while (st > cur_max && !cloth_trace.t5_stiff_max.compare_exchange_weak(
                                       cur_max, st, std::memory_order_relaxed)) {
            }
            if (set[0x3c] != 0) {
                cloth_trace.t5_shape_calls.fetch_add(1u, std::memory_order_relaxed);
            } else {
                cloth_trace.t5_sphere_calls.fetch_add(1u, std::memory_order_relaxed);
            }
        }
    }
    static const bool fix_on = [] {
        const char* env = std::getenv("BB_CLOTH_FIX");
        return !(env && env[0] == '0' && env[1] == '\0');
    }();
    static const int t5_mode = [] {
        const char* env = std::getenv("BB_CLOTH_TYPE5");
        if (!env || !*env) return 3;
        return std::atoi(env);
    }();
    const float frames = live ? live->cloth_frames : 1.0f;
    const float elastic_scale = live ? live->cloth_stiffness_scale : 1.0f;
    return BbClothInterp::ComputeNativeClothConstraintFactor(
        type, set_stiff, mode, iteration, substeps, a, b, cloth_exponent, frames, elastic_scale,
        t5_mode, cloth_t5_pulse_active.load(std::memory_order_relaxed),
        fix_on && !cloth_fixed30_hooked);
}

// ---- Redirected constants: instruction (its RIP-relative operand), the constant, the value ----
struct Redirect {
    u64 site;
    u64 constant;
    std::size_t value; ///< offsetof(Live, ..)
};
#define LIVE(field) offsetof(Live, field)
constexpr Redirect Redirects[] = {
    // A fixed 1/30 s step or "one frame" (the patches' lists, checked one by one):
    {0xab9f0d, 0x47efdcc, LIVE(dt)},   // a time step, one frame when unset
    {0xdca3ed, 0x4888688, LIVE(dt)},   // a per-frame budget
    {0xee596a, 0x48ad65c, LIVE(dt)},   // effects: a fade of at least one frame (three releases)
    {0xee5a8a, 0x48ad660, LIVE(dt)},
    {0xee5baa, 0x48ad664, LIVE(dt)},
    {0xef06f1, 0x48adc08, LIVE(dt)},   // at least one frame
    {0xef6992, 0x48add28, LIVE(dt)},
    {0x1044943, 0x48cc3f4, LIVE(dt)},  // an update by one frame
    {0x12a4416, 0x492577c, LIVE(dt)},  // a timer advanced by one frame each frame
    {0x15dc1f7, 0x4926c9c, LIVE(dt)},
    {0x173456e, 0x49270e4, LIVE(dt)},  // an update by one frame
    {0x17f9d26, 0x4927554, LIVE(dt)},
    {0x180f647, 0x49275f4, LIVE(dt)},  // a timer advanced by one frame each frame
    {0x18bc4f7, 0x4927c4c, LIVE(dt)},  // an update by one frame (twice)
    {0x18bc50d, 0x4927c4c, LIVE(dt)},
    {0x1dbc308, 0x4928e98, LIVE(dt)},  // frames counted at one frame's time
    {0x1f77d4a, 0x4929170, LIVE(dt)},  // an update by one frame (0 after a reset)
    {0x227784b, 0x4929b80, LIVE(dt)},  // timers advanced by one frame (0xebbda0)
    {0x2277892, 0x4929b80, LIVE(dt)},
    {0x2313d8d, 0x4929e70, LIVE(dt_half)}, // foliage wave/timer step: 1/60 at 30 FPS (0.5 * dt),
    {0x2313dad, 0x4929e70, LIVE(dt_half)},
    {0x2313db7, 0x4929e74, LIVE(dt)},      // 1/30 otherwise
    {0x2315d71, 0x4929e74, LIVE(dt)},
    // Body parts the physics moves (0x1515ec0, each part every frame): their timers advanced by the
    // step they are given (0x180df90 at +0xf0, 0x1811650 at +0xd0), 1/30 s; not in the patches.
    {0x1515fab, 0x4926364, LIVE(dt)},
    {0x1516030, 0x4926364, LIVE(dt)},
    // Steps and approaches per frame:
    {0x19084f5, 0x4927fd4, LIVE(step_down)}, // a value ramped by 0.1 a frame
    {0x1908522, 0x4927fd4, LIVE(step_down)},
    {0x1908518, 0x4927fe0, LIVE(step_up)},
    {0x190854c, 0x4927fe0, LIVE(step_up)},
    {0x143e2b2, 0x4925e94, LIVE(approach30)}, // the follow camera's approaches
    {0x143af8c, 0x4925e68, LIVE(approach10)},
    // Damping & acceleration per frame:
    {0x1a4abca, 0x490b5b0, LIVE(decay95)},     // a character's motion (0x1a4a730, x0.95 a frame)
    {0x1514c4d, 0x4926334, LIVE(decay08)},     // the run's wall check (x0.8 a frame while blocked)
    {0x1514c77, 0x4926338, LIVE(grow12)},      // the run's wall check recovery (x1.2 a frame)
    // Character slope sliding (sub_1a46bc0, called from sub_1a4a730 at 0x1a4aa13):
    // Unscaled 1.0 m/s per frame pushback + x0.65 decay caused characters to stall and twitch on
    // small beveled curbs/bumps (>35 deg normal) at >=90 FPS because slide_vel exceeded walk speed
    // before crossing the bevel.
    {0x1a46bd6, 0x5128374, LIVE(decay65)},     // slope-slide damping (x0.65 per 30 FPS frame)
    {0x1a46c52, 0x5128370, LIVE(slope_accel)}, // slope-slide acceleration (1.0 * frames)
    // Foot-IK ground height pelvis Y-offset smoothing (sub_1514500, 0x1515182..0x15151a6):
    {0x1515182, 0x5127b28, LIVE(approach10)},  // pelvis Y-offset gain (+0.1 * delta per 30 FPS frame)
    {0x1515195, 0x5127b2c, LIVE(decay60)},     // pelvis Y-offset decay (x0.6 per 30 FPS frame)
    // Cloth constraints' stiffness:
    {0x47c6dc, 0x47693f0, LIVE(cloth)},
    {0x7e45be, 0x47b63e4, LIVE(cloth)},
    {0x7e4642, 0x47b63e4, LIVE(cloth)},
    {0x7e6239, 0x47b63e4, LIVE(cloth)},
    {0x7f706d, 0x47b8474, LIVE(cloth)},
    {0x7f70ad, 0x47b8474, LIVE(cloth)},
    {0x7f79e5, 0x47b8474, LIVE(cloth)},
    {0x7f7be8, 0x47b8474, LIVE(cloth)},
    {0x7f7dbd, 0x47b8474, LIVE(cloth)},
    {0x805898, 0x47b91c0, LIVE(cloth)},
    {0x805d18, 0x47b91c0, LIVE(cloth)},
    {0x806269, 0x47b91c0, LIVE(cloth)},
    {0x8062f0, 0x47b91c0, LIVE(cloth)},
    {0x809d62, 0x47b98b8, LIVE(cloth)},
    {0x80a4c9, 0x47b98b8, LIVE(cloth)},
    {0x80a876, 0x47b98b8, LIVE(cloth)},
    // The model preview (0x1db2780: the character creator, the stick turns the model; xmm0/xmm1 the
    // stick): yaw and pitch += stick x the constant, per frame. 7x as fast at 200 FPS.
    {0x1db27b0, 0x4928dfc, LIVE(model_turn)},
};
#undef LIVE

/// [address, address + size) writable (the loader sets the image's protection after
/// bbgpu_patch_image; hooks installed earlier, BbGuestHooks, may have set a page's already).
void Writable(u64 address, std::size_t size) {
    const u64 page = u64(sysconf(_SC_PAGESIZE));
    const u64 start = address & ~(page - 1);
    mprotect(reinterpret_cast<void*>(start), address + size - start, PROT_READ | PROT_WRITE | PROT_EXEC);
}

/// The instruction at `r.site` reads `r.constant` RIP-relative (a VEX instruction, no immediate
/// after the operand): it reads the live value instead.
bool RedirectOperand(const Redirect& r) {
    u8* code = reinterpret_cast<u8*>(image_base + r.site);
    std::size_t disp_at = 0, length = 0;
    if (code[0] == 0xc5 && (code[3] & 0xc7) == 0x05) {
        disp_at = 4, length = 8;
    } else if (code[0] == 0xc4 && (code[4] & 0xc7) == 0x05) {
        disp_at = 5, length = 9;
    } else {
        return false;
    }
    std::int32_t disp = 0;
    std::memcpy(&disp, code + disp_at, 4);
    const u64 next = image_base + r.site + length;
    if (next + std::int64_t(disp) != image_base + r.constant) return false;
    const std::int64_t to = std::int64_t(reinterpret_cast<u64>(live) + r.value) - std::int64_t(next);
    if (to != std::int32_t(to)) return false;
    const std::int32_t to32 = std::int32_t(to);
    Writable(reinterpret_cast<u64>(code + disp_at), 4);
    std::memcpy(code + disp_at, &to32, 4);
    return true;
}

// ---- Detours: the instructions at a site replaced by a jump to a stub ----
u8* stub_page = nullptr;
std::size_t stub_used = 0;
constexpr std::size_t StubPageSize = 16384;

/// `e` into the stub page; a jump to it at `site` over `length` bytes (the rest int3) if the site
/// holds `expected`.
bool Detour(u64 site, const u8* expected, std::size_t length, const Emitter& e) {
    u8* code = reinterpret_cast<u8*>(image_base + site);
    if (length < 5 || std::memcmp(code, expected, length) != 0 || e.overflow ||
        stub_used + e.size > StubPageSize) {
        return false;
    }
    u8* stub = stub_page + stub_used;
    const std::int64_t rel = std::int64_t(reinterpret_cast<u64>(stub)) - std::int64_t(image_base + site + 5);
    if (rel != std::int32_t(rel)) return false;
    std::memcpy(stub, e.code.data(), e.size);
    stub_used += (e.size + 15) & ~std::size_t(15);
    const std::int32_t rel32 = std::int32_t(rel);
    Writable(reinterpret_cast<u64>(code), length);
    code[0] = 0xe9;
    std::memcpy(code + 1, &rel32, 4);
    for (std::size_t i = 5; i < length; ++i) code[i] = 0xcc;
    return true;
}

/// The pacer's mode, after the game chose it (0x203493d, `lea r15, [r12 + 0x268]`; r12 the
/// pacer): one vblank (+0x10), no flip skipping (+0x14), the shortest frame as the frame time
/// target (+0x18), no vblank counts to wait for (+0x268, +0x26c).
bool HookPacerMode() {
    constexpr u8 code[] = {0x4d, 0x8d, 0xbc, 0x24, 0x68, 0x02, 0x00, 0x00};
    Emitter e;
    e.Bytes({0x41, 0xc7, 0x44, 0x24, 0x10, 0x01, 0x00, 0x00, 0x00}); // mov dword [r12 + 0x10], 1
    e.Bytes({0x41, 0xc6, 0x44, 0x24, 0x14, 0x00});                   // mov byte [r12 + 0x14], 0
    e.Bytes({0x50, 0x48, 0xb8});                                       // push rax; movabs rax,
    e.U64(reinterpret_cast<u64>(&live->shortest));                     //   &shortest
    e.Bytes({0x8b, 0x00});                                             // mov eax, [rax]
    e.Bytes({0x41, 0x89, 0x44, 0x24, 0x18, 0x58});                     // mov [r12 + 0x18], eax; pop rax
    e.Bytes({0x49, 0xc7, 0x84, 0x24, 0x68, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}); // mov qword [r12 + 0x268], 0
    if (CleanPacerEnabled()) {
        // Clear PS4's 30-FPS pure-busy-spin flag (r13b -> r8b at 0x20349cb) so 0x2034ada takes
        // the sceKernelUsleep + 200 us precision-spin loop instead of burning Main CPU.
        e.Bytes({0x45, 0x31, 0xed});                                   // xor r13d, r13d
    }
    for (const u8 b : code) e.Bytes({b});
    e.JmpAbs(image_base + 0x203493d + sizeof(code));
    return Detour(0x203493d, code, sizeof(code), e);
}

/// Monotonic phase-locked deadline detour at 0x2034964 (`mov rax, [r12 + 0x28]; mov [r12 + 0x20],
/// rax`) jumping directly to SprjFlipper's post-wait dt/ring update at 0x2034c55.
bool HookPacerWait() {
    constexpr u8 code[] = {0x49, 0x8b, 0x44, 0x24, 0x28, 0x49, 0x89, 0x44, 0x24, 0x20};
    Emitter e;
    e.Bytes({0x4c, 0x89, 0xe7});       // mov rdi, r12 (pacer; rsp is already 16-byte aligned at 0x2034964)
    e.Bytes({0x8b, 0x75, 0xbc});       // mov esi, [rbp - 0x44] (reset flag from 0x20348fa)
    e.CallAbs(reinterpret_cast<const void*>(&PacerWait));
    e.Bytes({0x48, 0x89, 0xc6});       // mov rsi, rax (cur_exit_us expected at 0x2034c87)
    e.Bytes({0x31, 0xdb});             // xor ebx, ebx
    e.Bytes({0x45, 0x31, 0xed});       // xor r13d, r13d
    e.Bytes({0x45, 0x31, 0xff});       // xor r15d, r15d
    e.Bytes({0xc5, 0xf8, 0x57, 0xc0}); // vxorps xmm0, xmm0, xmm0
    e.JmpAbs(image_base + 0x2034c55);
    return Detour(0x2034964, code, sizeof(code), e);
}

/// The pacer keeps its one vblank when a frame came late (0x2034d96: `test dl, dl; je; mov eax,
/// [r12 + 0x10]`, then eax is stored back: 0 when late).
bool HookPacerLate() {
    constexpr u8 code[] = {0x84, 0xd2, 0x74, 0x05, 0x41, 0x8b, 0x44, 0x24, 0x10};
    Emitter e;
    e.Bytes({0x84, 0xd2});                   // test dl, dl
    e.Bytes({0x41, 0x8b, 0x44, 0x24, 0x10}); // mov eax, [r12 + 0x10]
    e.JmpAbs(image_base + 0x2034d9f);
    return Detour(0x2034d96, code, sizeof(code), e);
}

/// The main loop's step (0x2018e36: `mov dword [rbp - 0x38], 1/30`, its time; rbx the pacer):
/// FrameStep, then the frame's time.
bool HookStep() {
    constexpr u8 code[] = {0xc7, 0x45, 0xc8, 0x89, 0x88, 0x08, 0x3d};
    Emitter e;
    e.SaveVolatile();
    e.Bytes({0x48, 0x89, 0xdf}); // mov rdi, rbx
    e.CallAbs(reinterpret_cast<const void*>(&FrameStep));
    e.RestoreVolatile();
    e.StoreToFrame(&live->dt, -0x38);
    e.JmpAbs(image_base + 0x2018e36 + sizeof(code));
    return Detour(0x2018e36, code, sizeof(code), e);
}

/// A subsystem stepped by a fixed 1/30 s (`mov dword [rbp + offset], 1/30` at `site`): by the
/// frame's time.
bool HookFixedStep(u64 site, std::int8_t offset) {
    const u8 code[] = {0xc7, 0x45, u8(offset), 0x89, 0x88, 0x08, 0x3d};
    Emitter e;
    e.StoreToFrame(&live->dt, offset);
    e.JmpAbs(image_base + site + sizeof(code));
    return Detour(site, code, sizeof(code), e);
}

/// The run's wall check (0x1514c24: the distance moved this frame x 30 = a speed at 30 FPS): the
/// distance divided by the frame's time ([rbp - 0x74], the character update's), a speed at any
/// frame rate. Above ~90 FPS the per-frame distance fell under the check's threshold and the run
/// dropped to half speed.
/// 1: done by the hook, 2: the community's Sprint Fix patch already did it, 0: other code.
int HookRunSpeed() {
    constexpr u64 site = 0x1514c24;
    const u8* code = reinterpret_cast<const u8*>(image_base + site);
    constexpr u8 patched[] = {0xc5, 0xfa, 0x5e, 0x45, 0x8c, 0x0f, 0x1f, 0x00}; // vdivss; nop
    if (std::memcmp(code, patched, sizeof(patched)) == 0) {
        return 2;
    }
    std::int32_t disp = 0;
    std::memcpy(&disp, code + 4, 4);
    if (code[0] != 0xc5 || code[1] != 0xfa || code[2] != 0x59 || code[3] != 0x05 ||
        image_base + site + 8 + std::int64_t(disp) != image_base + 0x492632c) {
        return 0;
    }
    u8 expected[8];
    std::memcpy(expected, code, 8);
    Emitter e;
    e.Bytes({0xc5, 0xfa, 0x5e, 0x45, 0x8c}); // vdivss xmm0, xmm0, [rbp - 0x74]
    e.JmpAbs(image_base + site + 8);
    return Detour(site, expected, 8, e) ? 1 : 0;
}

/// Havok's stiffness factor (0x1167a80: `push rbp; mov rbp, rsp; push r15`) by
/// ClothStiffnessFactor (its arguments as they are, a tail call).
bool HookClothStiffness() {
    constexpr u64 site = 0x1167a80;
    constexpr u8 code[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};
    std::memcpy(&cloth_exponent, reinterpret_cast<const void*>(image_base + 0x48d7bbc), sizeof(float));
    if (!(cloth_exponent < 0.0f && cloth_exponent > -4.0f)) return false;
    Emitter e;
    e.JmpAbs(reinterpret_cast<u64>(&ClothStiffnessFactor));
    return Detour(site, code, sizeof(code), e);
}

/// Havok Cloth collision friction is applied per substep as `v_t *= (1 - particle.friction)`
/// with no dt scaling, causing capes/mantles to stick to shoulder/back colliders like velcro at
/// 90..300+ FPS. Multiply each particle's loaded friction in registers by `live->cloth_friction_scale`
/// (thread-safe, zero shared asset mutation, zero register clobber) across the landscape/pinch
/// solver (0x7745e2) and all 8 collidable shape solvers (16 sites in 0x257c14d..0x2580bee).
int HookClothFriction() {
    const char* env = std::getenv("BB_CLOTH_FRICTION");
    if (env && env[0] == '0' && env[1] == '\0') {
        return 0;
    }
    const u64 scale_addr = reinterpret_cast<u64>(&live->cloth_friction_scale[0]);
    int hooked = 0;

    // 1. Landscape / pinch collision solver (0x7745e2: vbroadcastss xmm1, [rcx + rdi + 0xc], 7 B)
    {
        constexpr u64 site = 0x7745e2;
        constexpr u8 code[7] = {0xc4, 0xe2, 0x79, 0x18, 0x4c, 0x39, 0x0c};
        const u64 stub_addr = reinterpret_cast<u64>(stub_page + stub_used);
        const std::int64_t disp = std::int64_t(scale_addr) - std::int64_t(stub_addr + sizeof(code) + 8);
        if (disp == std::int32_t(disp)) {
            Emitter e;
            for (const u8 b : code) e.Bytes({b});
            e.Bytes({0xc5, 0xf0, 0x59, 0x0d}); // vmulps xmm1, xmm1, [rip + disp32]
            e.U32(static_cast<std::uint32_t>(std::int32_t(disp)));
            e.JmpAbs(image_base + site + sizeof(code));
            if (Detour(site, code, sizeof(code), e)) {
                ++hooked;
            }
        }
    }

    // 2. All 8 Havok Cloth collidable shape solvers (2 paths each, 6 B vmovss xmm0, [reg1+reg2+0xc])
    struct FrictionSite {
        u64 site;
        u8 sib; // 0x37 = [rdi + rsi + 0xc], 0x16 = [rsi + rdx + 0xc]
    };
    constexpr FrictionSite sites[] = {
        {0x257c14d, 0x37}, {0x257c43e, 0x16},
        {0x257cc6e, 0x37}, {0x257cf8e, 0x37},
        {0x257d78f, 0x37}, {0x257da8e, 0x16},
        {0x257e16d, 0x37}, {0x257e45e, 0x16},
        {0x257eb4d, 0x37}, {0x257ee3e, 0x16},
        {0x257f52d, 0x37}, {0x257f81e, 0x16},
        {0x257ff0d, 0x37}, {0x258020e, 0x16},
        {0x25808fd, 0x37}, {0x2580bee, 0x16},
    };
    for (const FrictionSite& s : sites) {
        const u8 code[6] = {0xc5, 0xfa, 0x10, 0x44, s.sib, 0x0c};
        const u64 stub_addr = reinterpret_cast<u64>(stub_page + stub_used);
        const std::int64_t disp = std::int64_t(scale_addr) - std::int64_t(stub_addr + sizeof(code) + 8);
        if (disp != std::int32_t(disp)) continue;
        Emitter e;
        for (const u8 b : code) e.Bytes({b});
        e.Bytes({0xc5, 0xfa, 0x59, 0x05}); // vmulss xmm0, xmm0, [rip + disp32]
        e.U32(static_cast<std::uint32_t>(std::int32_t(disp)));
        e.JmpAbs(image_base + s.site + sizeof(code));
        if (Detour(s.site, code, sizeof(code), e)) {
            ++hooked;
        }
    }
    return hooked;
}

/// Foliage wind/interaction sway in 0x2313870:
/// - At 0x23149ea (`vmovdqa xmm14, [rbp - 0x170]`), `xmm14` multiplies the already-integrated
///   wind phase `[rbp - 0x190]` (`[rdi + 0x1b4]`, advanced by `live->dt` in 0x29552d0) and sway
///   vector (`0x2314a66: vmulps xmm12, xmm4, xmm2`, `0x2314ae7: vmulss xmm9, xmm14, xmm1`).
///   On native PS4 (30 FPS, `[SprjFlipper + 0x10] == 1`), `xmm14` is the constant `0x4929e70`
///   (`1/60f = 0.01666667f`). Keeping `xmm14` fixed at `0x4929e70` prevents:
///   (a) 2x speed/amplitude exaggeration at 30 FPS when `live->dt = 1/30f`,
///   (b) amplitude/speed shrinkage to `60 / FPS` (~16% at 360 FPS), and
///   (c) frame-to-frame phase jumps (jarring cadence hitches) from `live->dt * accumulated_time` jitter.
/// - At 0x2314b35..0x2314b43, `xmm10` (the per-frame timer step added to `[r14 + 0x610/0x620]`
///   at 0x2314bdb, 0x231581a, 0x2315aa5) loads `[rbp - 0x170]` (`live->dt_half = 0.5f * dt`).
bool HookFoliageSway() {
    constexpr u64 site_amp = 0x23149ea;
    constexpr u8 code_amp[] = {0xc5, 0x79, 0x6f, 0xb5, 0x90, 0xfe, 0xff, 0xff};
    Emitter e_amp;
    e_amp.Bytes({0x48, 0xb8}); // movabs rax, image_base + 0x4929e70 (rax is overwritten at 0x2314aef)
    e_amp.U64(image_base + 0x4929e70);
    e_amp.Bytes({0xc5, 0x7a, 0x10, 0x30}); // vmovss xmm14, [rax] (1/60f = 0.01666667f)
    e_amp.JmpAbs(image_base + site_amp + sizeof(code_amp));

    constexpr u64 site_step = 0x2314b35;
    constexpr u8 code_step[] = {
        0x44, 0x8b, 0x85, 0x48, 0xfe, 0xff, 0xff, // mov r8d, [rbp - 0x1b8]
        0x75, 0x05,                               // jne +5
        0xc4, 0x41, 0x79, 0x6f, 0xd6,             // vmovdqa xmm10, xmm14
    };
    Emitter e_step;
    e_step.Bytes({0x44, 0x8b, 0x85, 0x48, 0xfe, 0xff, 0xff});       // mov r8d, [rbp - 0x1b8]
    e_step.Bytes({0x75, 0x08});                                     // jne +8 (preserve pause check)
    e_step.Bytes({0xc5, 0x79, 0x6f, 0x95, 0x90, 0xfe, 0xff, 0xff}); // vmovdqa xmm10, [rbp - 0x170]
    e_step.JmpAbs(image_base + site_step + sizeof(code_step));

    return Detour(site_amp, code_amp, sizeof(code_amp), e_amp) &&
           Detour(site_step, code_step, sizeof(code_step), e_step);
}

bool ClothFixed30Enabled() noexcept {
    static const bool enabled = [] {
        const char* fix = std::getenv("BB_CLOTH_FIX");
        if (fix && fix[0] == '0' && fix[1] == '\0') return false;
        const char* sync = std::getenv("BB_CLOTH_SYNC");
        if (sync && sync[0] == '0' && sync[1] == '\0') return false;
        const char* env = std::getenv("BB_CLOTH_FIXED30");
        return !(env && env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

int ClothPostCollideMode() noexcept {
    static const int mode = [] {
        const char* env = std::getenv("BB_CLOTH_POST_COLLIDE");
        return (env && *env) ? std::clamp(std::atoi(env), 0, 2) : 0;
    }();
    return mode;
}

struct ClothFixedState {
    const u8* sim_data = nullptr;
    int n = 0;
    std::int32_t move_buf_idx = -1; ///< hclMoveParticlesOperator (+0x34) buffer index for rigid L2W/W2L
    std::uint64_t last_frame_seq = 0;
    std::uint64_t prev_pose_fseq = 0;
    bool initialized = false;
    bool has_origin = false;
    bool use_rigid_frame = false;
    bool has_prev_pose = false;
    bool prev_use_rigid = false;
    float origin_curr[3] = {0.0f, 0.0f, 0.0f};
    float prev_l2w[16]{};
    float prev_anchor[3]{};
    std::vector<float> raw_curr;          ///< un-interpolated m_currentPositions (n * 4)
    std::vector<float> raw_prev;          ///< un-interpolated m_previousPositions (n * 4)
    std::vector<float> local_prev;        ///< previous 30 Hz step local-frame positions (n * 4)
    std::vector<float> local_curr;        ///< current 30 Hz step local-frame positions (n * 4)
    std::vector<float> curr_pinned_saved; ///< live t_f pinned positions preserved across do_step (n * 4)
    std::vector<float> prev_pinned;       ///< previous base frame t_{f-1} pinned positions (n * 4)
    /// Per-particle pin mask:
    ///   0 = free particle (invMass > 0, not in m_fixedParticles)
    ///   bit 0 (1) = invMass == 0 (hclMoveParticlesOperator sets prev = cur when m_fixedParticles empty)
    ///   bit 1 (2) = listed in sim_data->m_fixedParticles (+0x48/+0x50; simulate lerps prev -> cur)
    std::vector<u8> fixed_mask;
    struct TSetSnap {
        const u8* orig_tset = nullptr;
        std::int32_t n_mats = 0;
        std::size_t mat_offset = 0;
    };
    std::vector<TSetSnap> prev_tset_meta;
    std::vector<BbClothInterp::ClothMat4> prev_tset_mats;
    std::vector<BbClothInterp::ClothMat4> interp_tset_mats;
    std::vector<BbClothInterp::ClonedHclTransformSet> cloned_tsets;
    std::vector<const u8*> cloned_tset_ptrs;
};
std::mutex cloth_fixed_mutex;
std::unordered_map<const u8*, std::unique_ptr<ClothFixedState>> cloth_fixed_states;
using ClothPreStepFn = void(PS4_SYSV_ABI*)(u8*, u8*);
ClothPreStepFn orig_cloth_prestep = nullptr;

inline void TransformPoint4x4(const float* m, float x, float y, float z, float out[3]) noexcept {
    out[0] = m[0] * x + m[4] * y + m[8] * z + m[12];
    out[1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    out[2] = m[2] * x + m[6] * y + m[10] * z + m[14];
}

inline bool IsValidRigidFrame4x4(const float* l2w, const float* w2l,
                                 const float probe[3], bool has_anchor) noexcept {
    for (int k = 0; k < 16; ++k) {
        if (!std::isfinite(l2w[k]) || !std::isfinite(w2l[k])) return false;
    }
    for (int col = 0; col < 3; ++col) {
        const float* c = l2w + col * 4;
        const float n2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
        if (n2 < 0.1f || n2 > 100.0f) return false;
    }
    // Reject an unused identity matrix when the character itself is far from world origin.
    if (has_anchor) {
        const float t2 = l2w[12] * l2w[12] + l2w[13] * l2w[13] + l2w[14] * l2w[14];
        const float p2 = probe[0] * probe[0] + probe[1] * probe[1] + probe[2] * probe[2];
        if (t2 < 1e-8f && p2 > 16.0f &&
            std::fabs(l2w[0] - 1.0f) < 1e-4f && std::fabs(l2w[5] - 1.0f) < 1e-4f &&
            std::fabs(l2w[10] - 1.0f) < 1e-4f) {
            return false;
        }
    }
    float loc[3], back[3];
    TransformPoint4x4(w2l, probe[0], probe[1], probe[2], loc);
    TransformPoint4x4(l2w, loc[0], loc[1], loc[2], back);
    const float dx = back[0] - probe[0];
    const float dy = back[1] - probe[1];
    const float dz = back[2] - probe[2];
    return (dx * dx + dy * dy + dz * dz) < 1e-3f;
}

/// Wrapper around `hclSimulateOperator::preStep` (`0x76fb90`, vtable `0x55f2a48`):
///   - In default native per-frame mode (`!cloth_fixed30_hooked`), runs Havok's `preStep` and then
///     deterministically computes `[sim + 0x10c] = ComputeNativeClothEffectiveDamping(...)` on every
///     frame across all branches (`prev_sdt == 0`, `prev_sdt == new_sdt`, `global_damp == 0`).
///   - In optional `BB_CLOTH_FIXED30=1` mode (`cloth_fixed30_hooked`), feeds fixed `1/30 s` `step_dt`.
void PS4_SYSV_ABI WrappedClothPreStep(u8* op, u8* step_info) noexcept {
    if (!orig_cloth_prestep) {
        return;
    }
    const auto call_prestep = [](u8* p_op, u8* p_si) noexcept {
        if (!BbClothExp::TryDispatchPreStep(p_op, p_si)) {
            orig_cloth_prestep(p_op, p_si);
        }
    };
    if (!op || !step_info) {
        call_prestep(op, step_info);
        return;
    }
    if (!cloth_fixed30_hooked) {
        u8* sim_for_trace = nullptr;
        float old_sdt_for_trace = 0.0f;
        if (ClothTraceEnabled()) {
            u8* ctx0 = *reinterpret_cast<u8**>(step_info + 0x10);
            if (ctx0) {
                const std::int32_t idx0 = *reinterpret_cast<const std::int32_t*>(op + 0x1c);
                u8* const* arr0 = *reinterpret_cast<u8* const* const*>(ctx0 + 0x20);
                const std::int32_t cnt0 = *reinterpret_cast<const std::int32_t*>(ctx0 + 0x28);
                if (arr0 && idx0 >= 0 && idx0 < cnt0 && idx0 < 64) {
                    sim_for_trace = arr0[idx0];
                    if (sim_for_trace) {
                        old_sdt_for_trace = *reinterpret_cast<const float*>(
                            sim_for_trace + BbClothInterp::HclSimClothInstanceOffsets::kPrevSubstepDt);
                    }
                }
            }
        }
        call_prestep(op, step_info);
        if (!ClothFixEnabled() || !live) {
            return;
        }
        u8* ctx = *reinterpret_cast<u8**>(step_info + 0x10);
        if (!ctx) {
            return;
        }
        const std::int32_t idx = *reinterpret_cast<const std::int32_t*>(op + 0x1c);
        u8* const* sim_arr = *reinterpret_cast<u8* const* const*>(ctx + 0x20);
        const std::int32_t sim_cnt = *reinterpret_cast<const std::int32_t*>(ctx + 0x28);
        if (!sim_arr || idx < 0 || idx >= sim_cnt || idx >= 64) {
            return;
        }
        u8* sim = sim_arr[idx];
        if (!sim) {
            return;
        }
        using GetSimCfgFn = const u8*(PS4_SYSV_ABI*)(const u8*);
        const auto get_sim_cfg = reinterpret_cast<GetSimCfgFn>(image_base + 0x748c50);
        const u8* sim_cfg = get_sim_cfg(sim);
        if (!sim_cfg) {
            return;
        }
        const float new_sdt =
            *reinterpret_cast<const float*>(sim + BbClothInterp::HclSimClothInstanceOffsets::kPrevSubstepDt);
        if (ClothTraceEnabled()) {
            if (*reinterpret_cast<const void* const*>(sim + 0x1d8) != nullptr) {
                cloth_trace.prestep_override_calls.fetch_add(1u, std::memory_order_relaxed);
            }
            if (sim == sim_for_trace && old_sdt_for_trace > 0.0f && old_sdt_for_trace != new_sdt) {
                cloth_trace.prestep_rescale_calls.fetch_add(1u, std::memory_order_relaxed);
                const float ratio = new_sdt / old_sdt_for_trace;
                float rmin = cloth_trace.prestep_ratio_min.load(std::memory_order_relaxed);
                while (ratio < rmin && !cloth_trace.prestep_ratio_min.compare_exchange_weak(
                                           rmin, ratio, std::memory_order_relaxed)) {
                }
                float rmax = cloth_trace.prestep_ratio_max.load(std::memory_order_relaxed);
                while (ratio > rmax && !cloth_trace.prestep_ratio_max.compare_exchange_weak(
                                           rmax, ratio, std::memory_order_relaxed)) {
                }
            }
        }
        const float global_damp = *reinterpret_cast<const float*>(sim_cfg + 0x10);
        *reinterpret_cast<float*>(sim + BbClothInterp::HclSimClothInstanceOffsets::kEffectiveDamping) =
            BbClothInterp::ComputeNativeClothEffectiveDamping(
                global_damp, new_sdt, live->cloth_pbd_neg_rate);
        return;
    }
    if (ClothTraceEnabled()) {
        cloth_trace.f30_prestep_calls.fetch_add(1u, std::memory_order_relaxed);
    }
    u8* ctx = *reinterpret_cast<u8**>(step_info + 0x10);
    if (!ctx) {
        call_prestep(op, step_info);
        return;
    }
    const std::int32_t idx = *reinterpret_cast<const std::int32_t*>(op + 0x1c);
    u8* const* sim_arr = *reinterpret_cast<u8* const* const*>(ctx + 0x20);
    const std::int32_t sim_cnt = *reinterpret_cast<const std::int32_t*>(ctx + 0x28);
    if (!sim_arr || idx < 0 || idx >= sim_cnt || idx >= 64) {
        call_prestep(op, step_info);
        return;
    }
    u8* sim = sim_arr[idx];
    if (!sim) {
        call_prestep(op, step_info);
        return;
    }

    const float step_dt = cloth_step_dt.load(std::memory_order_relaxed);
    const float prev_sdt = *reinterpret_cast<const float*>(sim + 0x108);
    const float slowmo = (*reinterpret_cast<const std::int32_t*>(sim + 0x1cc) == 1)
                             ? 1.0f
                             : *reinterpret_cast<const float*>(sim + 0x1d0);
    using GetSimCfgFn = const u8*(PS4_SYSV_ABI*)(const u8*);
    const auto get_sim_cfg = reinterpret_cast<GetSimCfgFn>(image_base + 0x748c50);
    const u8* sim_cfg = get_sim_cfg(sim);
    std::int32_t substeps = sim_cfg ? *reinterpret_cast<const std::int32_t*>(sim_cfg + 0x18) : 0;
    if (substeps == 0) {
        substeps = *reinterpret_cast<const std::int32_t*>(op + 0x20);
    }
    const float new_sdt =
        (slowmo > 0.0f && substeps > 0) ? (step_dt / (slowmo * float(substeps))) : 0.0f;

    alignas(16) u8 step_copy[0x28];
    std::memcpy(step_copy, step_info, sizeof(step_copy));
    std::memcpy(step_copy, &step_dt, sizeof(float));

    // Steady-state fast path (99.99% of frames at >30 FPS and first spawn):
    // When prev_sdt == 0 (first frame) or prev_sdt == new_sdt (constant 1/30 s), 0x76fb90 never
    // enters the m_previousPositions rescaling loop (0x76fdf0) and keeps exact PS4 effectiveDamping.
    if (prev_sdt == 0.0f || prev_sdt == new_sdt) {
        call_prestep(op, step_copy);
        return;
    }

    // Rare dt/slowmo transition path (e.g. FPS dip below 30 FPS or dynamic slow-motion change).
    const bool step_now = cloth_step_this_frame.load(std::memory_order_relaxed);
    if (!step_now) {
        return;
    }

    ClothFixedState* st = nullptr;
    {
        std::lock_guard<std::mutex> lock(cloth_fixed_mutex);
        auto it = cloth_fixed_states.find(sim);
        if (it != cloth_fixed_states.end()) {
            st = it->second.get();
        }
    }
    float* cur = *reinterpret_cast<float* const*>(sim + 0x18);
    const std::int32_t n = *reinterpret_cast<const std::int32_t*>(sim + 0x20);
    float* prev = *reinterpret_cast<float* const*>(sim + 0x28);
    const bool have_raw = (st && st->initialized && st->n == n && cur && prev);
    if (have_raw) {
        for (int i = 0; i < n; ++i) {
            if (st->fixed_mask[std::size_t(i)] == 0u) {
                std::memcpy(&cur[i * 4], &st->raw_curr[std::size_t(i) * 4u], 16);
                std::memcpy(&prev[i * 4], &st->raw_prev[std::size_t(i) * 4u], 16);
            }
        }
    }
    call_prestep(op, step_copy);
    if (have_raw) {
        for (int i = 0; i < n; ++i) {
            if (st->fixed_mask[std::size_t(i)] == 0u) {
                std::memcpy(&st->raw_prev[std::size_t(i) * 4u], &prev[i * 4], 16);
            }
        }
    }
}

/// True fixed-30 Hz wrapper around `hclSimulateOperator::simulate` (`0x76e370`, sole caller
/// `0x76ab3c`). Paired with `WrappedClothPreStep` (`0x76fb90`), steps Havok Cloth physics strictly
/// at PS4's exact `1/30 s` cadence (`k_scale = 1.0`, `friction = 1.0`, exact PS4 `effectiveDamping`)
/// for 100% identical gravity, wind, movement inertia, and skirt containment at any FPS (30..500+),
/// while smoothly interpolating free particles every render frame in the character's live rigid
/// object frame (`hclBuffer::m_bufferToWorldTransform` / `m_worldToBufferTransform`) so rotation
/// and translation track the skeleton with 0 ms lag and -92% cloth CPU cost.
int PS4_SYSV_ABI WrappedClothSimulate(u8* op, u8* ctx, float dt) noexcept {
    using SimFn = int(PS4_SYSV_ABI*)(u8*, u8*, float);
    using NormalsFn = void(PS4_SYSV_ABI*)(void*, u8*);
    using AabbFn = void(PS4_SYSV_ABI*)(u8*);
    const auto orig_sim = reinterpret_cast<SimFn>(image_base + 0x76e370);
    if (!cloth_fixed30_hooked || !op || !ctx) {
        return orig_sim(op, ctx, dt);
    }
    const std::int32_t idx = *reinterpret_cast<const std::int32_t*>(op + 0x1c);
    u8* const* sim_arr = *reinterpret_cast<u8* const* const*>(ctx + 0x20);
    const std::int32_t sim_cnt = *reinterpret_cast<const std::int32_t*>(ctx + 0x28);
    if (!sim_arr || idx < 0 || idx >= sim_cnt || idx >= 64) {
        return orig_sim(op, ctx, dt);
    }
    u8* sim = sim_arr[idx];
    if (!sim) {
        return orig_sim(op, ctx, dt);
    }
    const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
    float* cur = *reinterpret_cast<float* const*>(sim + 0x18);
    const std::int32_t n = *reinterpret_cast<const std::int32_t*>(sim + 0x20);
    float* prev = *reinterpret_cast<float* const*>(sim + 0x28);
    if (!sim_data || !cur || !prev || n <= 0 || n > 16384) {
        return orig_sim(op, ctx, dt);
    }

    ClothFixedState* st = nullptr;
    {
        std::lock_guard<std::mutex> lock(cloth_fixed_mutex);
        auto& ptr = cloth_fixed_states[sim];
        if (!ptr) {
            ptr = std::make_unique<ClothFixedState>();
        }
        st = ptr.get();
    }

    const std::uint64_t fseq = cloth_world_frame_seq.load(std::memory_order_relaxed);
    st->last_frame_seq = fseq;

    bool reset = (!st->initialized || st->sim_data != sim_data || st->n != n);
    if (reset) {
        st->sim_data = sim_data;
        st->n = n;
        st->initialized = false;
        st->has_origin = false;
        st->use_rigid_frame = false;
        st->has_prev_pose = false;
        st->prev_use_rigid = false;
        st->prev_pose_fseq = 0;
        st->move_buf_idx = -1;
        const std::size_t n4 = std::size_t(n) * 4u;
        st->raw_curr.resize(n4);
        st->raw_prev.resize(n4);
        st->local_prev.resize(n4);
        st->local_curr.resize(n4);
        st->curr_pinned_saved.resize(n4);
        st->prev_pinned.resize(n4);
        st->fixed_mask.assign(std::size_t(n), 0u);
        // Mark pinned particles separately for invMass == 0 (bit 0) vs m_fixedParticles (bit 1):
        // When m_fixedParticles is empty (all Bloodborne Hunter Garb meshes), hclMoveParticlesOperator
        // sets BOTH prev[p] = cur[p] = live_world_pos so Verlet (0x7740bf) computes cur - prev == 0.
        const u8* pdatas = *reinterpret_cast<const u8* const*>(sim_data + 0x38);
        const std::int32_t n_pdatas = *reinterpret_cast<const std::int32_t*>(sim_data + 0x40);
        if (pdatas && n_pdatas >= n) {
            for (int i = 0; i < n; ++i) {
                const float inv_m = *reinterpret_cast<const float*>(pdatas + std::size_t(i) * 16u + 4u);
                if (inv_m == 0.0f) {
                    st->fixed_mask[std::size_t(i)] |= 1u;
                }
            }
        }
        const std::uint16_t* fix_idx = *reinterpret_cast<const std::uint16_t* const*>(sim_data + 0x48);
        const std::int32_t n_fix = *reinterpret_cast<const std::int32_t*>(sim_data + 0x50);
        if (fix_idx && n_fix > 0 && n_fix <= n) {
            for (int k = 0; k < n_fix; ++k) {
                const std::uint16_t pi = fix_idx[k];
                if (int(pi) < n) {
                    st->fixed_mask[pi] |= 2u;
                }
            }
        }
        // Locate this simCloth's hclMoveParticlesOperator (type 11) in the active hclClothState
        // so we can read the character's exact live rigid L2W (+0x80) and W2L (+0xc0) transforms.
        const u8* cloth_data = *reinterpret_cast<const u8* const*>(ctx + 0x10);
        const std::int32_t state_idx = *reinterpret_cast<const std::int32_t*>(ctx + 0x18);
        if (cloth_data && state_idx >= 0) {
            const std::int32_t n_states = *reinterpret_cast<const std::int32_t*>(cloth_data + 0x60);
            const u8* const* states = *reinterpret_cast<const u8* const* const*>(cloth_data + 0x58);
            const u8* const* all_ops = *reinterpret_cast<const u8* const* const*>(cloth_data + 0x48);
            const std::int32_t n_all_ops = *reinterpret_cast<const std::int32_t*>(cloth_data + 0x50);
            if (states && all_ops && state_idx < n_states && n_all_ops > 0) {
                const u8* state = states[state_idx];
                if (state) {
                    const std::int32_t* op_idxs = *reinterpret_cast<const std::int32_t* const*>(state + 0x18);
                    const std::int32_t n_ops = *reinterpret_cast<const std::int32_t*>(state + 0x20);
                    if (op_idxs && n_ops > 0 && n_ops < 256) {
                        for (int k = 0; k < n_ops; ++k) {
                            const std::int32_t oi = op_idxs[k];
                            if (oi < 0 || oi >= n_all_ops) continue;
                            const u8* o = all_ops[oi];
                            if (!o) continue;
                            if (*reinterpret_cast<const std::int32_t*>(o + 0x18) == 11 &&
                                *reinterpret_cast<const std::int32_t*>(o + 0x30) == idx) {
                                st->move_buf_idx = *reinterpret_cast<const std::int32_t*>(o + 0x34);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Compute live centroid of pinned vertices (fallback anchor + round-trip probe point).
    double ax = 0.0, ay = 0.0, az = 0.0;
    int anchor_cnt = 0;
    for (int i = 0; i < n; ++i) {
        if (st->fixed_mask[std::size_t(i)] != 0u) {
            const float x = cur[i * 4 + 0];
            const float y = cur[i * 4 + 1];
            const float z = cur[i * 4 + 2];
            if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
                ax += double(x);
                ay += double(y);
                az += double(z);
                ++anchor_cnt;
            }
        }
    }
    const bool has_anchor = (anchor_cnt > 0);
    const float anchor_now[3] = {
        has_anchor ? static_cast<float>(ax / double(anchor_cnt)) : 0.0f,
        has_anchor ? static_cast<float>(ay / double(anchor_cnt)) : 0.0f,
        has_anchor ? static_cast<float>(az / double(anchor_cnt)) : 0.0f,
    };

    // Resolve live rigid 4x4 character frame (rotation + translation) from hclMoveParticlesOperator's
    // active hclBuffer (+0x80 m_bufferToWorldTransform, +0xc0 m_worldToBufferTransform), falling back
    // to ctx->m_transformSets root bone when Skin_outer/MoveParticles keep hclBuffer at Identity.
    const float probe[3] = {
        has_anchor ? anchor_now[0] : cur[0],
        has_anchor ? anchor_now[1] : cur[1],
        has_anchor ? anchor_now[2] : cur[2],
    };
    bool use_rigid = false;
    float l2w[16];
    float w2l[16];
    if (st->move_buf_idx >= 0) {
        u8* const* bufs = *reinterpret_cast<u8* const* const*>(ctx + 0x30);
        const std::int32_t n_bufs = *reinterpret_cast<const std::int32_t*>(ctx + 0x38);
        if (bufs && st->move_buf_idx < n_bufs) {
            const u8* raw_buf = bufs[st->move_buf_idx];
            if (raw_buf) {
                const std::int32_t sh_idx = *reinterpret_cast<const std::int32_t*>(raw_buf + 0x100);
                const u8* active_buf = (sh_idx >= 0 && sh_idx < n_bufs && bufs[sh_idx]) ? bufs[sh_idx] : raw_buf;
                std::memcpy(l2w, active_buf + 0x80, sizeof(l2w));
                std::memcpy(w2l, active_buf + 0xc0, sizeof(w2l));
                use_rigid = IsValidRigidFrame4x4(l2w, w2l, probe, has_anchor);
            }
        }
    }
    if (!use_rigid) {
        const auto* tsets = *reinterpret_cast<const u8* const* const*>(
            ctx + BbClothInterp::HclClothInstanceOffsets::kTransformSetsData);
        const std::int32_t n_tsets = *reinterpret_cast<const std::int32_t*>(
            ctx + BbClothInterp::HclClothInstanceOffsets::kTransformSetsSize);
        if (tsets && n_tsets > 0) {
            const std::int32_t xfer_ts = *reinterpret_cast<const std::int32_t*>(
                sim_data + BbClothInterp::HclSimClothDataOffsets::kTransferTransformSetIndex);
            const std::int32_t xfer_bi = *reinterpret_cast<const std::int32_t*>(
                sim_data + BbClothInterp::HclSimClothDataOffsets::kTransferTransformIndex);
            const std::int32_t col_ts = *reinterpret_cast<const std::int32_t*>(
                sim_data + BbClothInterp::HclSimClothDataOffsets::kCollidableTransformSetIndex);
            const std::int32_t try_ts = (xfer_ts >= 0 && xfer_ts < n_tsets) ? xfer_ts : col_ts;
            const std::int32_t try_bi = (xfer_ts >= 0 && xfer_ts < n_tsets && xfer_bi >= 0) ? xfer_bi : 0;
            if (try_ts >= 0 && try_ts < n_tsets && tsets[try_ts]) {
                const u8* tset = tsets[try_ts];
                const auto* mats = *reinterpret_cast<const BbClothInterp::ClothMat4* const*>(
                    tset + BbClothInterp::HclTransformSetOffsets::kMatricesData);
                const std::int32_t n_mats = *reinterpret_cast<const std::int32_t*>(
                    tset + BbClothInterp::HclTransformSetOffsets::kMatricesSize);
                if (mats && try_bi >= 0 && try_bi < n_mats &&
                    (reinterpret_cast<std::uintptr_t>(mats) & 0xFu) == 0u) {
                    std::memcpy(l2w, mats[try_bi].m, sizeof(l2w));
                    if (BbClothInterp::InvertRigidMatrix4x4(l2w, w2l)) {
                        use_rigid = IsValidRigidFrame4x4(l2w, w2l, probe, has_anchor);
                    }
                }
            }
        }
    }

    const bool has_origin = has_anchor || use_rigid;
    const float origin_now[3] = {
        has_anchor ? anchor_now[0] : (use_rigid ? l2w[12] : 0.0f),
        has_anchor ? anchor_now[1] : (use_rigid ? l2w[13] : 0.0f),
        has_anchor ? anchor_now[2] : (use_rigid ? l2w[14] : 0.0f),
    };

    if (!reset && st->initialized) {
        if (st->use_rigid_frame != use_rigid) {
            reset = true;
        } else if (has_origin && st->has_origin) {
            const float dx = origin_now[0] - st->origin_curr[0];
            const float dy = origin_now[1] - st->origin_curr[1];
            const float dz = origin_now[2] - st->origin_curr[2];
            if (dx * dx + dy * dy + dz * dz > 2.25f) { // > 1.5 m teleport/respawn
                reset = true;
            }
        }
    }

    using GetSimCfgFn = const u8*(PS4_SYSV_ABI*)(const u8*);
    const auto get_sim_cfg = reinterpret_cast<GetSimCfgFn>(image_base + 0x748c50);
    const u8* sim_cfg = get_sim_cfg(sim);
    const bool xfer_enabled = sim_cfg && (sim_cfg[0x1e] != 0u);
    bool has_local_range = false;
    u8* const* static_csets = *reinterpret_cast<u8* const* const*>(
        sim_data + BbClothInterp::HclSimClothDataOffsets::kStaticConstraintSetsData);
    const std::int32_t n_static_csets = *reinterpret_cast<const std::int32_t*>(
        sim_data + BbClothInterp::HclSimClothDataOffsets::kStaticConstraintSetsSize);
    if (static_csets && n_static_csets > 0 && n_static_csets <= 64) {
        for (std::int32_t ci = 0; ci < n_static_csets; ++ci) {
            const u8* cs = static_csets[ci];
            if (cs && *reinterpret_cast<const std::uint32_t*>(cs + 0x18) == 5u) {
                has_local_range = true;
                break;
            }
        }
    }

    const bool do_step = reset || cloth_step_this_frame.load(std::memory_order_relaxed);
    const float alpha = cloth_interp_alpha.load(std::memory_order_relaxed);
    const float step_beta = cloth_step_beta.load(std::memory_order_relaxed);
    const float step_dt = cloth_step_dt.load(std::memory_order_relaxed);
    // Keep cur[pinned] and m_transformSets locked to live t_f whenever Skin_outer has already
    // deformed a LocalRange reference mesh (SimBuffer_2) to live t_f or transferMotion is off,
    // preventing step_beta rewind from pulling collar pins inside the live t_f back mesh.
    const bool can_interp_pose = xfer_enabled && !has_local_range && !reset && st->initialized &&
                                 st->has_prev_pose && (fseq == st->prev_pose_fseq + 1u) &&
                                 (step_beta < 0.9999f);
    const auto recalc_normals = reinterpret_cast<NormalsFn>(image_base + 0x76f040);
    const auto compute_aabb_full = reinterpret_cast<AabbFn>(image_base + 0x748ab0);
    const auto compute_aabb_basic = reinterpret_cast<AabbFn>(image_base + 0x748a00);

    if (ClothTraceEnabled()) {
        if (use_rigid) {
            cloth_trace.f30_rigid_ok.fetch_add(1u, std::memory_order_relaxed);
        }
        if (reset) {
            cloth_trace.f30_resets.fetch_add(1u, std::memory_order_relaxed);
        }
    }

    auto project_against_live_body = [&]() noexcept {
        const int post_mode = ClothPostCollideMode();
        if (post_mode <= 0) {
            return;
        }
        using CollideFn = void(PS4_SYSV_ABI*)(u8*, u8*, u8*, float);
        using ApplyFn = void(PS4_SYSV_ABI*)(u8*, u8*, u8*, int, int, float);
        const auto collide_particles = reinterpret_cast<CollideFn>(image_base + 0x774260);

        // 1. Update sim->m_activeCollidables (+0x1c0) in-place to live t_f bone transforms without
        //    mutating sim->m_colliders (+0xd0), keeping 30 Hz collider velocities untouched.
        const std::int32_t n_char_cols = *reinterpret_cast<const std::int32_t*>(
            sim + BbClothInterp::HclSimClothInstanceOffsets::kCollidablesSize);
        const std::int32_t n_active_cols = *reinterpret_cast<const std::int32_t*>(
            sim + BbClothInterp::HclSimClothInstanceOffsets::kActiveCollidablesSize);
        u8* active_cols = *reinterpret_cast<u8* const*>(
            sim + BbClothInterp::HclSimClothInstanceOffsets::kActiveCollidablesData);
        const std::int32_t col_tset_idx = *reinterpret_cast<const std::int32_t*>(
            sim_data + BbClothInterp::HclSimClothDataOffsets::kCollidableTransformSetIndex);
        const auto* tsets = *reinterpret_cast<const u8* const* const*>(
            ctx + BbClothInterp::HclClothInstanceOffsets::kTransformSetsData);
        const std::int32_t n_tsets = *reinterpret_cast<const std::int32_t*>(
            ctx + BbClothInterp::HclClothInstanceOffsets::kTransformSetsSize);
        if (active_cols && n_char_cols > 0 && n_active_cols >= n_char_cols &&
            tsets && col_tset_idx >= 0 && col_tset_idx < n_tsets && tsets[col_tset_idx]) {
            const u8* tset = tsets[col_tset_idx];
            const auto* bone_mats = *reinterpret_cast<const BbClothInterp::ClothMat4* const*>(
                tset + BbClothInterp::HclTransformSetOffsets::kMatricesData);
            const std::int32_t n_bones = *reinterpret_cast<const std::int32_t*>(
                tset + BbClothInterp::HclTransformSetOffsets::kMatricesSize);
            const std::int32_t* bone_map = *reinterpret_cast<const std::int32_t* const*>(
                sim_data + BbClothInterp::HclSimClothDataOffsets::kCollidableTransformMapData);
            const std::int32_t n_map = *reinterpret_cast<const std::int32_t*>(
                sim_data + BbClothInterp::HclSimClothDataOffsets::kCollidableTransformMapSize);
            const auto* local_offs = *reinterpret_cast<const BbClothInterp::ClothMat4* const*>(
                sim_data + BbClothInterp::HclSimClothDataOffsets::kCollidableTransformOffsetsData);
            if (bone_mats && n_bones > 0 && bone_map && n_map >= n_char_cols && local_offs &&
                (reinterpret_cast<std::uintptr_t>(bone_mats) & 0xFu) == 0u &&
                (reinterpret_cast<std::uintptr_t>(local_offs) & 0xFu) == 0u) {
                for (std::int32_t c = 0; c < n_char_cols; ++c) {
                    const std::int32_t bi = bone_map[c];
                    if (bi < 0 || bi >= n_bones) continue;
                    u8* col_dst = active_cols + std::size_t(c) * BbClothInterp::HclCollidableOffsets::kStride;
                    float world_mat[16];
                    BbClothInterp::MulAffineTransform4x4(bone_mats[bi].m, local_offs[c].m, world_mat);
                    std::memcpy(col_dst + BbClothInterp::HclCollidableOffsets::kTransform, world_mat,
                                sizeof(world_mat));
                    std::memset(col_dst + BbClothInterp::HclCollidableOffsets::kLinearVelocity, 0, 32);
                }
            }
        }

        // 2. Project interpolated display positions `cur` outside live t_f capsules/shapes and
        //    live t_f-skinned LocalRange back/shoulder surfaces in exact .hkx execution order.
        //    `st->raw_curr` and `st->raw_prev` stay untouched so zero fake Verlet velocity is injected.
        auto run_cset_if_needed = [&](std::int32_t ci) noexcept {
            if (!static_csets || ci < 0 || ci >= n_static_csets) return;
            u8* cs = static_csets[ci];
            if (!cs) return;
            const std::uint32_t type = *reinterpret_cast<const std::uint32_t*>(cs + 0x18);
            if (type == 5u || (post_mode >= 2 && type == 2u)) {
                const auto* vtbl = *reinterpret_cast<void* const* const*>(cs);
                if (vtbl && vtbl[0x50 / sizeof(void*)]) {
                    const auto apply_fn = reinterpret_cast<ApplyFn>(vtbl[0x50 / sizeof(void*)]);
                    apply_fn(cs, sim, ctx, static_cast<int>(ci), 0, 1.0f);
                }
            }
        };

        const std::int32_t n_exec = *reinterpret_cast<const std::int32_t*>(
            op + BbClothInterp::HclSimulateOperatorOffsets::kConstraintExecutionsSize);
        const std::int32_t* exec_ids = *reinterpret_cast<const std::int32_t* const*>(
            op + BbClothInterp::HclSimulateOperatorOffsets::kConstraintExecutionsData);
        if (exec_ids && n_exec > 0 && n_exec <= 128) {
            for (std::int32_t ei = 0; ei < n_exec; ++ei) {
                const std::int32_t id = exec_ids[ei];
                if (id == -1) {
                    if (active_cols && n_active_cols > 0) {
                        collide_particles(op, ctx, sim, Frame30);
                    }
                } else {
                    run_cset_if_needed(id);
                }
            }
        } else {
            for (std::int32_t ci = 0; ci < n_static_csets; ++ci) {
                run_cset_if_needed(ci);
            }
            if (active_cols && n_active_cols > 0) {
                collide_particles(op, ctx, sim, Frame30);
            }
        }
    };

    auto write_interp_free = [&]() noexcept {
        for (int i = 0; i < n; ++i) {
            if (st->fixed_mask[std::size_t(i)] != 0u) continue;
            const std::size_t k = std::size_t(i) * 4u;
            const float lx = st->local_prev[k + 0] + (st->local_curr[k + 0] - st->local_prev[k + 0]) * alpha;
            const float ly = st->local_prev[k + 1] + (st->local_curr[k + 1] - st->local_prev[k + 1]) * alpha;
            const float lz = st->local_prev[k + 2] + (st->local_curr[k + 2] - st->local_prev[k + 2]) * alpha;
            if (use_rigid) {
                float wp[3];
                TransformPoint4x4(l2w, lx, ly, lz, wp);
                cur[i * 4 + 0] = wp[0];
                cur[i * 4 + 1] = wp[1];
                cur[i * 4 + 2] = wp[2];
            } else {
                cur[i * 4 + 0] = anchor_now[0] + lx;
                cur[i * 4 + 1] = anchor_now[1] + ly;
                cur[i * 4 + 2] = anchor_now[2] + lz;
            }
            cur[i * 4 + 3] = st->raw_curr[k + 3];
        }
        project_against_live_body();
        if (sim_data[0x14c] != 0) {
            recalc_normals(nullptr, sim);
        }
        if (sim[0x1c8] != 0) {
            compute_aabb_full(sim);
        } else {
            compute_aabb_basic(sim);
        }
    };

    auto save_curr_pose_as_prev = [&]() noexcept {
        for (int i = 0; i < n; ++i) {
            if (st->fixed_mask[std::size_t(i)] == 0u) continue;
            std::memcpy(&st->prev_pinned[std::size_t(i) * 4u], &cur[i * 4], 16);
        }
        if (use_rigid) {
            std::memcpy(st->prev_l2w, l2w, sizeof(st->prev_l2w));
        }
        std::memcpy(st->prev_anchor, anchor_now, sizeof(st->prev_anchor));
        st->prev_use_rigid = use_rigid;
        st->prev_tset_meta.clear();
        st->prev_tset_mats.clear();
        const auto* orig_tsets = *reinterpret_cast<const u8* const* const*>(
            ctx + BbClothInterp::HclClothInstanceOffsets::kTransformSetsData);
        const std::int32_t n_tsets = *reinterpret_cast<const std::int32_t*>(
            ctx + BbClothInterp::HclClothInstanceOffsets::kTransformSetsSize);
        if (orig_tsets && n_tsets > 0 && n_tsets <= 8) {
            for (std::int32_t ts_i = 0; ts_i < n_tsets; ++ts_i) {
                const u8* tset = orig_tsets[ts_i];
                const BbClothInterp::ClothMat4* mats = nullptr;
                std::int32_t n_mats = 0;
                if (tset) {
                    mats = *reinterpret_cast<const BbClothInterp::ClothMat4* const*>(
                        tset + BbClothInterp::HclTransformSetOffsets::kMatricesData);
                    n_mats = *reinterpret_cast<const std::int32_t*>(
                        tset + BbClothInterp::HclTransformSetOffsets::kMatricesSize);
                    if (!mats || n_mats <= 0 || n_mats > 512 ||
                        (reinterpret_cast<std::uintptr_t>(mats) & 0xFu) != 0u) {
                        mats = nullptr;
                        n_mats = 0;
                    }
                }
                const std::size_t off = st->prev_tset_mats.size();
                st->prev_tset_meta.push_back({tset, n_mats, off});
                if (mats && n_mats > 0) {
                    st->prev_tset_mats.insert(st->prev_tset_mats.end(), mats, mats + n_mats);
                }
            }
        }
        st->prev_pose_fseq = fseq;
        st->has_prev_pose = true;
    };

    if (do_step) {
        if (ClothTraceEnabled()) {
            cloth_trace.f30_steps.fetch_add(1u, std::memory_order_relaxed);
        }
        if (can_interp_pose) {
            for (int i = 0; i < n; ++i) {
                if (st->fixed_mask[std::size_t(i)] == 0u) continue;
                std::memcpy(&st->curr_pinned_saved[std::size_t(i) * 4u], &cur[i * 4], 16);
            }
        }
        if (!reset && st->initialized) {
            // Restore un-interpolated physics positions ONLY for free particles, and feed exact
            // 30 Hz sub-frame interpolated pinned anchors + collider transforms so Havok
            // transferMotion (`0x771290`) and Verlet constraints never see 2:1 beat aliasing.
            // CRITICAL: For invMass==0 particles not in m_fixedParticles (100% of Hunter Garb),
            // Verlet (`0x7740bf`) computes cur + damp*(cur - prev) WITHOUT multiplying by invMass,
            // so prev[pinned] MUST stay equal to cur[pinned] (matching hclMoveParticlesOperator).
            for (int i = 0; i < n; ++i) {
                const u8 fm = st->fixed_mask[std::size_t(i)];
                const std::size_t k = std::size_t(i) * 4u;
                if (fm == 0u) {
                    std::memcpy(&cur[i * 4], &st->raw_curr[k], 16);
                    std::memcpy(&prev[i * 4], &st->raw_prev[k], 16);
                } else {
                    if (can_interp_pose) {
                        const float* p0 = &st->prev_pinned[k];
                        const float* p1 = &st->curr_pinned_saved[k];
                        if (std::isfinite(p0[0]) && std::isfinite(p0[1]) && std::isfinite(p0[2]) &&
                            std::isfinite(p1[0]) && std::isfinite(p1[1]) && std::isfinite(p1[2])) {
                            const float pdx = p1[0] - p0[0];
                            const float pdy = p1[1] - p0[1];
                            const float pdz = p1[2] - p0[2];
                            if (pdx * pdx + pdy * pdy + pdz * pdz <= 2.25f) {
                                cur[i * 4 + 0] = p0[0] + pdx * step_beta;
                                cur[i * 4 + 1] = p0[1] + pdy * step_beta;
                                cur[i * 4 + 2] = p0[2] + pdz * step_beta;
                                cur[i * 4 + 3] = p1[3];
                            }
                        }
                    }
                    if ((fm & 2u) != 0u) {
                        std::memcpy(&prev[i * 4], &st->raw_curr[k], 16);
                    } else {
                        std::memcpy(&prev[i * 4], &cur[i * 4], 16);
                    }
                }
            }
            st->local_prev.swap(st->local_curr);
        }

        struct TransformSetSwapGuard {
            const u8* const** slot = nullptr;
            const u8* const* orig = nullptr;
            ~TransformSetSwapGuard() noexcept {
                if (slot) {
                    *slot = orig;
                }
            }
        };
        int ret = 0;
        {
            TransformSetSwapGuard tset_guard;
            if (can_interp_pose && !st->prev_tset_meta.empty()) {
                auto* tset_slot = reinterpret_cast<const u8* const**>(
                    ctx + BbClothInterp::HclClothInstanceOffsets::kTransformSetsData);
                const u8* const* orig_tsets = *tset_slot;
                const std::int32_t n_tsets = *reinterpret_cast<const std::int32_t*>(
                    ctx + BbClothInterp::HclClothInstanceOffsets::kTransformSetsSize);
                if (orig_tsets && n_tsets == static_cast<std::int32_t>(st->prev_tset_meta.size())) {
                    bool match = true;
                    for (std::int32_t ts_i = 0; ts_i < n_tsets; ++ts_i) {
                        const auto& pm = st->prev_tset_meta[std::size_t(ts_i)];
                        const u8* tset = orig_tsets[ts_i];
                        if (pm.n_mats == 0) {
                            if (pm.orig_tset != tset) {
                                match = false;
                                break;
                            }
                            continue;
                        }
                        if (!tset || tset != pm.orig_tset) {
                            match = false;
                            break;
                        }
                        const auto* mats = *reinterpret_cast<const BbClothInterp::ClothMat4* const*>(
                            tset + BbClothInterp::HclTransformSetOffsets::kMatricesData);
                        const std::int32_t n_mats = *reinterpret_cast<const std::int32_t*>(
                            tset + BbClothInterp::HclTransformSetOffsets::kMatricesSize);
                        if (!mats || n_mats != pm.n_mats ||
                            (reinterpret_cast<std::uintptr_t>(mats) & 0xFu) != 0u) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        st->interp_tset_mats.resize(st->prev_tset_mats.size());
                        st->cloned_tsets.resize(std::size_t(n_tsets));
                        st->cloned_tset_ptrs.resize(std::size_t(n_tsets));
                        for (std::int32_t ts_i = 0; ts_i < n_tsets; ++ts_i) {
                            const auto& pm = st->prev_tset_meta[std::size_t(ts_i)];
                            const u8* tset = orig_tsets[ts_i];
                            if (pm.n_mats <= 0) {
                                st->cloned_tset_ptrs[std::size_t(ts_i)] = tset;
                                continue;
                            }
                            const auto* mats = *reinterpret_cast<const BbClothInterp::ClothMat4* const*>(
                                tset + BbClothInterp::HclTransformSetOffsets::kMatricesData);
                            BbClothInterp::ClothMat4* dst_mats =
                                st->interp_tset_mats.data() + pm.mat_offset;
                            const BbClothInterp::ClothMat4* src0_mats =
                                st->prev_tset_mats.data() + pm.mat_offset;
                            for (std::int32_t m = 0; m < pm.n_mats; ++m) {
                                BbClothInterp::LerpTransformMatrix4x4(
                                    src0_mats[m].m, mats[m].m, step_beta, dst_mats[m].m);
                            }
                            BbClothInterp::InitClonedHclTransformSet(
                                st->cloned_tsets[std::size_t(ts_i)], tset, dst_mats);
                            st->cloned_tset_ptrs[std::size_t(ts_i)] =
                                st->cloned_tsets[std::size_t(ts_i)].bytes;
                        }
                        tset_guard.slot = tset_slot;
                        tset_guard.orig = orig_tsets;
                        *tset_slot = st->cloned_tset_ptrs.data();
                    }
                }
            }
            ret = orig_sim(op, ctx, step_dt);
        }

        std::memcpy(st->raw_curr.data(), cur, std::size_t(n) * 16u);
        std::memcpy(st->raw_prev.data(), prev, std::size_t(n) * 16u);

        float w2l_sim[16]{};
        bool use_sim_w2l = false;
        if (can_interp_pose && use_rigid && st->prev_use_rigid) {
            float l2w_sim[16]{};
            BbClothInterp::LerpTransformMatrix4x4(st->prev_l2w, l2w, step_beta, l2w_sim);
            const float sim_probe[3] = {
                has_anchor ? (st->prev_anchor[0] + (anchor_now[0] - st->prev_anchor[0]) * step_beta)
                           : l2w_sim[12],
                has_anchor ? (st->prev_anchor[1] + (anchor_now[1] - st->prev_anchor[1]) * step_beta)
                           : l2w_sim[13],
                has_anchor ? (st->prev_anchor[2] + (anchor_now[2] - st->prev_anchor[2]) * step_beta)
                           : l2w_sim[14],
            };
            if (BbClothInterp::InvertRigidMatrix4x4(l2w_sim, w2l_sim) &&
                IsValidRigidFrame4x4(l2w_sim, w2l_sim, sim_probe, has_anchor)) {
                use_sim_w2l = true;
            }
        }
        const float* w2l_for_step = use_sim_w2l ? w2l_sim : w2l;
        const float anchor_for_step[3] = {
            can_interp_pose
                ? (st->prev_anchor[0] + (anchor_now[0] - st->prev_anchor[0]) * step_beta)
                : anchor_now[0],
            can_interp_pose
                ? (st->prev_anchor[1] + (anchor_now[1] - st->prev_anchor[1]) * step_beta)
                : anchor_now[1],
            can_interp_pose
                ? (st->prev_anchor[2] + (anchor_now[2] - st->prev_anchor[2]) * step_beta)
                : anchor_now[2],
        };

        for (int i = 0; i < n; ++i) {
            const std::size_t k = std::size_t(i) * 4u;
            if (use_rigid) {
                float lp[3];
                TransformPoint4x4(w2l_for_step, cur[i * 4 + 0], cur[i * 4 + 1], cur[i * 4 + 2], lp);
                st->local_curr[k + 0u] = lp[0];
                st->local_curr[k + 1u] = lp[1];
                st->local_curr[k + 2u] = lp[2];
            } else {
                st->local_curr[k + 0u] = cur[i * 4 + 0] - anchor_for_step[0];
                st->local_curr[k + 1u] = cur[i * 4 + 1] - anchor_for_step[1];
                st->local_curr[k + 2u] = cur[i * 4 + 2] - anchor_for_step[2];
            }
            st->local_curr[k + 3u] = cur[i * 4 + 3];
        }
        if (can_interp_pose) {
            // Restore live t_f pinned seam vertices so rendered shoulder/collar stitches never lag.
            for (int i = 0; i < n; ++i) {
                if (st->fixed_mask[std::size_t(i)] == 0u) continue;
                std::memcpy(&cur[i * 4], &st->curr_pinned_saved[std::size_t(i) * 4u], 16);
            }
        }
        st->has_origin = has_origin;
        st->use_rigid_frame = use_rigid;
        st->origin_curr[0] = origin_now[0];
        st->origin_curr[1] = origin_now[1];
        st->origin_curr[2] = origin_now[2];

        bool snap_prev = (reset || !st->initialized);
        if (!snap_prev) {
            float max_d2 = 0.0f;
            for (int i = 0; i < n; ++i) {
                if (st->fixed_mask[std::size_t(i)] != 0u) continue;
                const std::size_t k = std::size_t(i) * 4u;
                const float dx = st->local_curr[k + 0u] - st->local_prev[k + 0u];
                const float dy = st->local_curr[k + 1u] - st->local_prev[k + 1u];
                const float dz = st->local_curr[k + 2u] - st->local_prev[k + 2u];
                const float d2 = dx * dx + dy * dy + dz * dz;
                if (!(d2 >= 0.0f) || d2 > max_d2) max_d2 = d2;
            }
            if (!(max_d2 >= 0.0f) || max_d2 > 2.25f) {
                snap_prev = true;
            }
        }
        if (snap_prev) {
            st->local_prev = st->local_curr;
            st->initialized = true;
            if (can_interp_pose) {
                write_interp_free();
            }
        } else if (alpha < 0.999f || can_interp_pose) {
            write_interp_free();
        }
        save_curr_pose_as_prev();
        return ret;
    }

    if (ClothTraceEnabled()) {
        cloth_trace.f30_skips.fetch_add(1u, std::memory_order_relaxed);
    }
    // Non-step high-FPS frame: smoothly interpolate local-frame free particles onto this frame's
    // live rigid character transform without stepping Havok Verlet (-92% cloth CPU at 360 FPS).
    write_interp_free();
    save_curr_pose_as_prev();
    return 0;
}

bool HookClothPreStep() {
    if (!ClothFixEnabled()) {
        return false;
    }
    constexpr u64 pre_site = 0x76fb90;
    constexpr u8 pre_expected[6] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57}; // push rbp; mov rbp,rsp; push r15
    if (std::memcmp(reinterpret_cast<const u8*>(image_base + pre_site), pre_expected, sizeof(pre_expected)) != 0) {
        return false;
    }
    Emitter tramp;
    for (const u8 b : pre_expected) tramp.Bytes({b});
    tramp.JmpAbs(image_base + pre_site + sizeof(pre_expected));
    if (tramp.overflow || stub_used + tramp.size + 64 > StubPageSize) {
        return false;
    }
    u8* tramp_addr = stub_page + stub_used;
    std::memcpy(tramp_addr, tramp.code.data(), tramp.size);
    stub_used += (tramp.size + 15) & ~std::size_t(15);
    orig_cloth_prestep = reinterpret_cast<ClothPreStepFn>(tramp_addr);

    Emitter e_pre;
    e_pre.JmpAbs(reinterpret_cast<u64>(&WrappedClothPreStep));
    if (!Detour(pre_site, pre_expected, sizeof(pre_expected), e_pre)) {
        orig_cloth_prestep = nullptr;
        return false;
    }
    return true;
}

bool HookClothFixed30() {
    if (!ClothFixed30Enabled() || !orig_cloth_prestep) {
        return false;
    }
    // Patch the sole call site of hclSimulateOperator::simulate (0x76ab3c: call 0x76e370).
    constexpr u64 site = 0x76ab3c;
    constexpr u8 expected[5] = {0xe8, 0x2f, 0x38, 0x00, 0x00}; // call 0x76e370
    u8* code = reinterpret_cast<u8*>(image_base + site);
    if (std::memcmp(code, expected, sizeof(expected)) != 0) {
        return false;
    }
    Emitter e;
    e.JmpAbs(reinterpret_cast<u64>(&WrappedClothSimulate));
    if (e.overflow || stub_used + e.size > StubPageSize) {
        return false;
    }
    u8* stub = stub_page + stub_used;
    const std::int64_t rel = std::int64_t(reinterpret_cast<u64>(stub)) - std::int64_t(image_base + site + 5);
    if (rel != std::int32_t(rel)) {
        return false;
    }
    std::memcpy(stub, e.code.data(), e.size);
    stub_used += (e.size + 15) & ~std::size_t(15);
    const std::int32_t rel32 = std::int32_t(rel);
    Writable(reinterpret_cast<u64>(code), sizeof(expected));
    code[0] = 0xe8; // call stub -> tail-jmp WrappedClothSimulate -> returns to 0x76ab41
    std::memcpy(code + 1, &rel32, 4);
    return true;
}

void PS4_SYSV_ABI TraceConstraintApply(u8* set, u8* sim, u8* ctx, int idx, int use_factor,
                                       float factor) noexcept {
    if (!set) {
        return;
    }
    const std::uint32_t type = *reinterpret_cast<const std::uint32_t*>(set + 0x18);
    if (type < 24u && (ClothSkipMask() & (1u << type)) != 0u) {
        return;
    }
    using ApplyFn = void(PS4_SYSV_ABI*)(u8*, u8*, u8*, int, int, float);
    const auto* vtbl = *reinterpret_cast<void* const* const*>(set);
    const auto apply_fn = reinterpret_cast<ApplyFn>(vtbl[0x50 / sizeof(void*)]);
    if (!ClothTraceEnabled() || !sim) {
        apply_fn(set, sim, ctx, idx, use_factor, factor);
        return;
    }
    float pre_xyz[512 * 3];
    const int snap_n = TraceSnapshotFreeParticles(sim, pre_xyz, 512);
    apply_fn(set, sim, ctx, idx, use_factor, factor);
    if (snap_n > 0 && type >= 1u && type <= 5u) {
        const float sdt = *reinterpret_cast<const float*>(sim + 0x108);
        TraceRecordStageDelta(static_cast<int>(type), sim, pre_xyz, snap_n, sdt);
    }
}

/// Forces `r8d = 1` (`useStiffnessFactor = true`) across both `solveConstraints` paths (`0x771f60`
/// at `0x7720f9` and `0x772180` at `0x77230b`) in native mode so `hclLocalRangeConstraintSet::apply`
/// (`0x6fb220`) always applies `ComputeClothExponentialRelaxationFactor` even on garments authored
/// with `m_adaptConstraintStiffness == 0` or `m_subSteps == 1` (e.g. all 5 layers of Hunter Garb).
/// When `BB_CLOTH_TRACE=1` or `BB_CLOTH_SKIP_T != 0`, wraps the virtual `apply` call with
/// `TraceConstraintApply` (zero overhead in normal runs).
int HookClothSolveConstraintsUseFactor() {
    if (!ClothFixEnabled() || cloth_fixed30_hooked) {
        return 0;
    }
    const bool wrap_apply = ClothTraceEnabled() || (ClothSkipMask() != 0u);
    int hooked = 0;
    if (wrap_apply) {
        constexpr u64 site1 = 0x7720f9;
        constexpr u8 expected1[10] = {0x44, 0x89, 0xf9, 0x44, 0x8b, 0x45, 0xc0, 0xff, 0x50, 0x50};
        Emitter e1;
        e1.Bytes({0x44, 0x89, 0xf9});                   // mov ecx, r15d
        e1.Bytes({0x41, 0xb8, 0x01, 0x00, 0x00, 0x00}); // mov r8d, 1
        e1.CallAbs(reinterpret_cast<const void*>(&TraceConstraintApply));
        e1.JmpAbs(image_base + 0x772103);
        if (Detour(site1, expected1, sizeof(expected1), e1)) {
            ++hooked;
        }
        constexpr u64 site2 = 0x77230b;
        constexpr u8 expected2[10] = {0x4c, 0x89, 0xe2, 0x44, 0x8b, 0x45, 0xb4, 0xff, 0x50, 0x50};
        Emitter e2;
        e2.Bytes({0x4c, 0x89, 0xe2});                   // mov rdx, r12
        e2.Bytes({0x41, 0xb8, 0x01, 0x00, 0x00, 0x00}); // mov r8d, 1
        e2.CallAbs(reinterpret_cast<const void*>(&TraceConstraintApply));
        e2.JmpAbs(image_base + 0x772315);
        if (Detour(site2, expected2, sizeof(expected2), e2)) {
            ++hooked;
        }
        return hooked;
    }
    {
        constexpr u64 site1 = 0x7720f9;
        constexpr u8 expected1[7] = {0x44, 0x89, 0xf9, 0x44, 0x8b, 0x45, 0xc0}; // mov ecx,r15d; mov r8d,[rbp-0x40]
        Emitter e1;
        e1.Bytes({0x44, 0x89, 0xf9});                   // mov ecx, r15d
        e1.Bytes({0x41, 0xb8, 0x01, 0x00, 0x00, 0x00}); // mov r8d, 1
        e1.JmpAbs(image_base + 0x772100);
        if (Detour(site1, expected1, sizeof(expected1), e1)) {
            ++hooked;
        }
    }
    {
        constexpr u64 site2 = 0x77230b;
        constexpr u8 expected2[7] = {0x4c, 0x89, 0xe2, 0x44, 0x8b, 0x45, 0xb4}; // mov rdx,r12; mov r8d,[rbp-0x4c]
        Emitter e2;
        e2.Bytes({0x4c, 0x89, 0xe2});                   // mov rdx, r12
        e2.Bytes({0x41, 0xb8, 0x01, 0x00, 0x00, 0x00}); // mov r8d, 1
        e2.JmpAbs(image_base + 0x772312);
        if (Detour(site2, expected2, sizeof(expected2), e2)) {
            ++hooked;
        }
    }
    return hooked;
}

void UpdateClothNativeTiming(float cdt, bool update_t5_pulse) noexcept {
    const double frames_d = std::clamp(double(cdt) * 30.0, 1.0 / 300.0, 1.0);
    const float clamped_frames = static_cast<float>(frames_d);
    live->cloth_frames = clamped_frames;
    const float exp_ov = ClothExpOverride();
    const float cloth_exp = exp_ov > 0.0f ? exp_ov : -cloth_exponent;
    const bool fix_on = ClothFixEnabled();
    live->cloth_stiffness_scale =
        fix_on ? BbClothInterp::ComputeClothElasticStiffnessScale(clamped_frames, cloth_exp) : 1.0f;
    constexpr double kClothFrictionRef = 0.35;
    const float friction_scale = static_cast<float>(
        (-std::expm1(frames_d * std::log(1.0 - kClothFrictionRef))) / kClothFrictionRef);
    for (float& s : live->cloth_friction_scale) s = friction_scale;

    cloth_sim_time_s += double(cdt);
    const auto ws = BbClothInterp::ComputeClothWindSmoothState(cloth_sim_time_s);
    live->cloth_wind_seed30 = ws.seed0;
    live->cloth_wind_seed1 = ws.seed1;
    live->cloth_wind_smooth = ClothWindSeedEnabled() ? ws.smooth : 0.0f;
    live->cloth_pbd_neg_rate =
        fix_on ? BbClothInterp::ComputeHighFpsClothDampingNegRate(clamped_frames, ClothDampRetention30())
               : 0.0f;

    if (update_t5_pulse) {
        const std::uint64_t tick30 = static_cast<std::uint64_t>(std::floor(cloth_sim_time_s * 30.0));
        const std::uint32_t t30 = static_cast<std::uint32_t>(tick30);
        const std::uint32_t prev_t30 = cloth_t5_last_tick30.exchange(t30, std::memory_order_relaxed);
        cloth_t5_pulse_active.store(t30 != prev_t30, std::memory_order_relaxed);
        cloth_tick30_seq.store(t30, std::memory_order_relaxed);
    }
}

void PS4_SYSV_ABI OnClothWorldStep(float* dt_ptr) noexcept {
    if (!dt_ptr) {
        return;
    }
    const float dt = *dt_ptr;
    if (!(dt > 0.0f && std::isfinite(dt))) {
        return;
    }
    const std::uint64_t fseq = cloth_world_frame_seq.fetch_add(1u, std::memory_order_relaxed) + 1u;
    const float cdt = std::clamp(dt, MinSimDt, Frame30);

    if (cloth_fixed30_hooked) {
        // Periodically prune stale cloth entries while all Havok Cloth workers are idle (right
        // after waitForStepComplete() 0x76a030).
        if ((fseq & 63u) == 0u && !cloth_fixed_states.empty()) {
            for (auto it = cloth_fixed_states.begin(); it != cloth_fixed_states.end();) {
                if (fseq - it->second->last_frame_seq > 180u) {
                    it = cloth_fixed_states.erase(it);
                } else {
                    ++it;
                }
            }
        }
        live->cloth_stiffness_scale = 1.0f;
        live->cloth_frames = 1.0f;
        for (float& s : live->cloth_friction_scale) s = 1.0f;
        const auto tick = BbClothInterp::ComputeClothFixed30Tick(
            cloth_fixed_accum, cloth_steady_30_streak, cdt, Frame30);
        cloth_step_this_frame.store(tick.do_step, std::memory_order_relaxed);
        cloth_interp_alpha.store(tick.alpha, std::memory_order_relaxed);
        cloth_step_beta.store(tick.step_beta, std::memory_order_relaxed);
        cloth_step_dt.store(tick.step_dt, std::memory_order_relaxed);
        if (tick.do_step) {
            ++cloth_fixed_step_seq;
        }
        const auto ws = BbClothInterp::ComputeClothWindSmoothState(double(cloth_fixed_step_seq) / 30.0);
        live->cloth_wind_seed30 = ws.seed0;
        live->cloth_wind_seed1 = ws.seed1;
        live->cloth_wind_smooth = 0.0f;
        live->cloth_pbd_neg_rate = 0.0f;
        return;
    }

    UpdateClothNativeTiming(cdt, true);
}

/// Synchronizes Havok Cloth fixed-30 Hz cadence (or fallback dt-adapted scales) and 30 Hz wind
/// turbulence PRNG directly to `hclWorld`'s own `dt` at `0xfd54a2` — immediately after
/// `waitForStepComplete()` (`0x76a030`) and before worker jobs dispatch (`0x769450`/`0x769190`).
bool HookClothWorldSync() {
    if (!ClothFixEnabled()) {
        return false;
    }
    const char* sync_env = std::getenv("BB_CLOTH_SYNC");
    if (sync_env && sync_env[0] == '0' && sync_env[1] == '\0') {
        return false;
    }
    constexpr u64 site = 0xfd54a2;
    constexpr u8 code[10] = {
        0x80, 0x7b, 0x30, 0x00,             // cmp byte ptr [rbx + 0x30], 0
        0x0f, 0x84, 0x8e, 0x00, 0x00, 0x00, // je 0xfd553a
    };
    Emitter e;
    e.SaveVolatile();
    e.SaveXmm();
    e.Bytes({0x48, 0x8d, 0x7d, 0xe4}); // lea rdi, [rbp - 0x1c] (&dt)
    e.CallAbs(reinterpret_cast<const void*>(&OnClothWorldStep));
    e.RestoreXmm();
    e.RestoreVolatile();
    e.Bytes({0x80, 0x7b, 0x30, 0x00}); // cmp byte ptr [rbx + 0x30], 0
    e.Bytes({0x75, 0x0e});             // jne +14 (skip single-threaded JmpAbs)
    e.JmpAbs(image_base + 0xfd553a);
    e.JmpAbs(image_base + 0xfd54ac);
    return Detour(site, code, sizeof(code), e);
}

bool PS4_SYSV_ABI SprjDirWindHook(const u8* self, const u8* sim, float* forces, const float* dir_vec,
                                  const float* params, float sdt) noexcept {
    if (ClothTraceEnabled()) {
        ProbeDirWindApply(self, sim, reinterpret_cast<const u8*>(forces),
                          reinterpret_cast<const u8*>(dir_vec), params, sdt);
    }
    if (!live || ClothWindOffEnabled()) {
        return true;
    }
    const bool ok = BbClothInterp::SprjDirWindApplyNative(
        self, sim, forces, dir_vec, params, sdt, live->cloth_wind_seed30, live->cloth_wind_seed1,
        live->cloth_wind_smooth, live->cloth_frames, ClothWindSeedEnabled());
    if (ClothTraceEnabled() && sim && forces) {
        const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
        const std::int32_t n = *reinterpret_cast<const std::int32_t*>(sim + 0x20);
        if (sim_data && n > 0 && n <= 4096) {
            const float* pdatas = *reinterpret_cast<const float* const*>(sim_data + 0x38);
            const std::int32_t np = *reinterpret_cast<const std::int32_t*>(sim_data + 0x40);
            if (pdatas && np >= n) {
                double acc_sum = 0.0;
                int free_cnt = 0;
                for (int i = 0; i < n; ++i) {
                    const float inv_m = pdatas[i * 4 + 1];
                    if (!(inv_m > 0.0f)) continue;
                    const float fx = forces[i * 4 + 0];
                    const float fy = forces[i * 4 + 1];
                    const float fz = forces[i * 4 + 2];
                    const double f = std::sqrt(double(fx) * double(fx) + double(fy) * double(fy) +
                                               double(fz) * double(fz));
                    if (std::isfinite(f)) {
                        acc_sum += f * double(inv_m);
                        ++free_cnt;
                    }
                }
                if (free_cnt > 0) {
                    cloth_trace.wind_acc_sum_mm_s2.fetch_add(
                        static_cast<std::uint64_t>((acc_sum / double(free_cnt)) * 1000.0),
                        std::memory_order_relaxed);
                }
            }
        }
    }
    return ok;
}

/// Replaces FromSoftware's `SprjDirWind` (`0x1053b90`) with `SprjDirWindHook` (`BbClothInterp::SprjDirWindApplyNative`):
/// $C^1$-smooth dual-seed 30 Hz turbulence, true `prev_sdt` (`[sim + 0x108]`) relative drag velocity,
/// and 30-FPS substep displacement parity inside turbulence.
bool HookClothDirWind() {
    if (!ClothFixEnabled() && !ClothWindOffEnabled()) {
        return false;
    }
    constexpr u8 pro6[6] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};
    Emitter e;
    e.JmpAbs(reinterpret_cast<u64>(&SprjDirWindHook));
    return Detour(0x1053b90, pro6, sizeof(pro6), e);
}

constexpr u64 kWindSeedOverrideTag = 0x5eed000100000000ULL;

/// Evaluates `sub_10522f0` (`0x10522f0`) at discrete 30 Hz seeds `(seed0, seed1)` (thread-safely
/// passing the tagged seed via `rdx -> r13` with +0 B TLS) and smoothstep-blends the OUTPUT 3D
/// wind vector (`BlendClothWindVectors3D`), keeping exact 30-FPS wind frequency at any frame rate.
__m128 PS4_SYSV_ABI SprjWindEvalWorldDirHook(const u8* self, const u8* wind_handle) noexcept {
    using EvalFn = __m128(PS4_SYSV_ABI*)(const u8*, const u8*, u64);
    const auto orig_eval = reinterpret_cast<EvalFn>(image_base + 0x10522f0);
    if (!live || !ClothWindSeedEnabled()) {
        return orig_eval(self, wind_handle, 0ULL);
    }
    const std::uint32_t seed0 = live->cloth_wind_seed30;
    const std::uint32_t seed1 = live->cloth_wind_seed1;
    const float smooth = live->cloth_wind_smooth;
    const __m128 m0 = orig_eval(self, wind_handle, kWindSeedOverrideTag | u64(seed0));
    if (!(smooth > 1e-6f && smooth < 1.0f)) {
        return m0;
    }
    const __m128 m1 = orig_eval(self, wind_handle, kWindSeedOverrideTag | u64(seed1));
    alignas(16) float v0[4]{};
    alignas(16) float v1[4]{};
    alignas(16) float out[4]{};
    _mm_store_ps(v0, m0);
    _mm_store_ps(v1, m1);
    if (ClothTraceEnabled()) {
        const float dx = v1[0] - v0[0];
        const float dy = v1[1] - v0[1];
        const float dz = v1[2] - v0[2];
        const double dv = std::sqrt(double(dx) * double(dx) + double(dy) * double(dy) + double(dz) * double(dz));
        if (std::isfinite(dv)) {
            cloth_trace.wind_dv01_sum_mm_s.fetch_add(
                static_cast<std::uint64_t>(dv * 1000.0), std::memory_order_relaxed);
            cloth_trace.wind_dv01_samples.fetch_add(1u, std::memory_order_relaxed);
        }
    }
    BbClothInterp::BlendClothWindVectors3D(v0, v1, smooth, out);
    return _mm_load_ps(out);
}

void PS4_SYSV_ABI OnPointWindPrepareSeed(u8* frame, float raw_seed_f) noexcept {
    const std::int32_t pass = *reinterpret_cast<const std::int32_t*>(frame - 0x1dc);
    if (!live || !ClothWindSeedEnabled()) {
        *reinterpret_cast<std::int32_t*>(frame - 0xd8) = static_cast<std::int32_t>(raw_seed_f);
        *reinterpret_cast<float*>(frame - 0x1d8) = 0.0f;
        return;
    }
    if (pass == 0) {
        *reinterpret_cast<std::int32_t*>(frame - 0xd8) =
            static_cast<std::int32_t>(live->cloth_wind_seed30);
        *reinterpret_cast<std::int32_t*>(frame - 0x1d4) =
            static_cast<std::int32_t>(live->cloth_wind_seed1);
        *reinterpret_cast<float*>(frame - 0x1d8) = live->cloth_wind_smooth;
    } else {
        *reinterpret_cast<std::int32_t*>(frame - 0xd8) =
            *reinterpret_cast<const std::int32_t*>(frame - 0x1d4);
    }
}

void PS4_SYSV_ABI OnPointWindAfterEval(u8* frame) noexcept {
    auto* pass = reinterpret_cast<std::int32_t*>(frame - 0x1dc);
    const float smooth = *reinterpret_cast<const float*>(frame - 0x1d8);
    float* dir_cur = reinterpret_cast<float*>(frame - 0x120);
    float* spd_cur = reinterpret_cast<float*>(frame - 0x150);
    if (*pass == 0) {
        if (!(smooth > 1e-6f && smooth < 1.0f)) {
            *pass = 0;
            return;
        }
        std::memcpy(frame - 0x1d0, dir_cur, 16);
        *reinterpret_cast<float*>(frame - 0x1e0) = *spd_cur;
        *pass = 1;
        return;
    }
    const float* dir0 = reinterpret_cast<const float*>(frame - 0x1d0);
    const float spd0 = *reinterpret_cast<const float*>(frame - 0x1e0);
    float out_spd = *spd_cur;
    alignas(16) float out_dir[4]{};
    BbClothInterp::BlendClothPointWindSpeedAndDir(
        spd0, dir0, *spd_cur, dir_cur, smooth, &out_spd, out_dir);
    *spd_cur = out_spd;
    std::memcpy(dir_cur, out_dir, 16);
    *pass = 0;
}

/// Replaces both 1000 Hz wind evaluators (`sub_10522f0` called at `0x1052200` and `SprjPointWind`
/// `0x1052af0`) with discrete 30 Hz `(seed0, seed1)` OUTPUT vector/speed smoothstep blending, plus
/// `prev_sdt = [sim + 0x108]` particle velocity guard in `SprjPointWind` (`0x10532f2`).
int HookClothWindEval() {
    if (!ClothFixEnabled()) {
        return 0;
    }
    int hooked = 0;

    if (ClothWindSeedEnabled()) {
        // 1. sub_10522f0: save tagged 3rd arg rdx -> r13 (0x1052304), apply at seed site (0x1052363),
        //    and redirect sole caller (0x1052200: call 0x10522f0) to SprjWindEvalWorldDirHook.
        constexpr u8 exp_save6[6] = {0x49, 0x89, 0xf7, 0x48, 0x89, 0xfb}; // mov r15,rsi; mov rbx,rdi
        constexpr u8 exp_seed11[11] = {
            0xc4, 0xe1, 0xfa, 0x2c, 0xf0,             // vcvttss2si rsi, xmm0
            0x89, 0xb5, 0x28, 0xff, 0xff, 0xff,       // mov [rbp - 0xd8], esi
        };
        constexpr u8 exp_call5[5] = {0xe8, 0xeb, 0x00, 0x00, 0x00}; // call 0x10522f0
        u8* call_code = reinterpret_cast<u8*>(image_base + 0x1052200);
        if (std::memcmp(reinterpret_cast<const u8*>(image_base + 0x1052304), exp_save6, 6) == 0 &&
            std::memcmp(reinterpret_cast<const u8*>(image_base + 0x1052363), exp_seed11, 11) == 0 &&
            std::memcmp(call_code, exp_call5, 5) == 0) {
            Emitter e_save;
            e_save.Bytes({0x49, 0x89, 0xf7, 0x48, 0x89, 0xfb}); // mov r15, rsi; mov rbx, rdi
            e_save.Bytes({0x49, 0x89, 0xd5});                   // mov r13, rdx
            e_save.JmpAbs(image_base + 0x105230a);

            Emitter e_seed;
            e_seed.Bytes({0x4c, 0x89, 0xe8});                   // mov rax, r13
            e_seed.Bytes({0x48, 0xc1, 0xe8, 0x20});             // shr rax, 32
            e_seed.Bytes({0x3d, 0x01, 0x00, 0xed, 0x5e});       // cmp eax, 0x5eed0001
            e_seed.Bytes({0x75, 0x05});                         // jne fallback (+5)
            e_seed.Bytes({0x49, 0x63, 0xf5});                   // movsxd rsi, r13d
            e_seed.Bytes({0xeb, 0x05});                         // jmp store (+5)
            e_seed.Bytes({0xc4, 0xe1, 0xfa, 0x2c, 0xf0});       // fallback: vcvttss2si rsi, xmm0
            e_seed.Bytes({0x89, 0xb5, 0x28, 0xff, 0xff, 0xff}); // store: mov [rbp - 0xd8], esi
            e_seed.JmpAbs(image_base + 0x105236e);

            Emitter e_call;
            e_call.JmpAbs(reinterpret_cast<u64>(&SprjWindEvalWorldDirHook));
            if (Detour(0x1052304, exp_save6, 6, e_save) &&
                Detour(0x1052363, exp_seed11, 11, e_seed) &&
                !e_call.overflow && stub_used + e_call.size <= StubPageSize) {
                u8* stub = stub_page + stub_used;
                const std::int64_t rel =
                    std::int64_t(reinterpret_cast<u64>(stub)) - std::int64_t(image_base + 0x1052205);
                if (rel == std::int32_t(rel)) {
                    std::memcpy(stub, e_call.code.data(), e_call.size);
                    stub_used += (e_call.size + 15) & ~std::size_t(15);
                    const std::int32_t rel32 = std::int32_t(rel);
                    Writable(reinterpret_cast<u64>(call_code), 5);
                    call_code[0] = 0xe8;
                    std::memcpy(call_code + 1, &rel32, 4);
                    ++hooked;
                }
            }
        }

        // 2. SprjPointWind (0x1052af0): 2-pass discrete (seed0, seed1) output speed+dir blend
        //    using untouched stack slots [rbp - 0x1d0 .. rbp - 0x1e0] prior to 0x10533d2.
        constexpr u8 exp_pw_init10[10] = {
            0xc7, 0x85, 0x28, 0xff, 0xff, 0xff, 0x15, 0xcd, 0x5b, 0x07, // mov dword [rbp-0xd8], 0x75bcd15
        };
        constexpr u8 exp_pw_seed11[11] = {
            0xc4, 0xe1, 0xfa, 0x2c, 0xf0,             // vcvttss2si rsi, xmm0
            0x89, 0xb5, 0x28, 0xff, 0xff, 0xff,       // mov [rbp - 0xd8], esi
        };
        constexpr u8 exp_pw_post11[11] = {
            0xc5, 0xf8, 0x29, 0x85, 0xe0, 0xfe, 0xff, 0xff, // vmovaps [rbp - 0x120], xmm0
            0x4c, 0x89, 0xf7,                               // mov rdi, r14
        };
        if (std::memcmp(reinterpret_cast<const u8*>(image_base + 0x1052b49), exp_pw_init10, 10) == 0 &&
            std::memcmp(reinterpret_cast<const u8*>(image_base + 0x1052b84), exp_pw_seed11, 11) == 0 &&
            std::memcmp(reinterpret_cast<const u8*>(image_base + 0x1053200), exp_pw_post11, 11) == 0) {
            Emitter e_init;
            e_init.Bytes({0xc7, 0x85, 0x24, 0xfe, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00}); // mov dword [rbp-0x1dc], 0
            for (const u8 b : exp_pw_init10) e_init.Bytes({b});
            e_init.JmpAbs(image_base + 0x1052b53);

            Emitter e_pw_seed;
            e_pw_seed.SaveVolatile();
            e_pw_seed.SaveXmm();
            e_pw_seed.Bytes({0x48, 0x89, 0xef}); // mov rdi, rbp (xmm0 holds raw_seed_f)
            e_pw_seed.CallAbs(reinterpret_cast<const void*>(&OnPointWindPrepareSeed));
            e_pw_seed.RestoreXmm();
            e_pw_seed.RestoreVolatile();
            e_pw_seed.Bytes({0x48, 0x63, 0xb5, 0x28, 0xff, 0xff, 0xff}); // movsxd rsi, [rbp - 0xd8]
            e_pw_seed.JmpAbs(image_base + 0x1052b8f);

            Emitter e_pw_post;
            e_pw_post.Bytes({0xc5, 0xf8, 0x29, 0x85, 0xe0, 0xfe, 0xff, 0xff}); // vmovaps [rbp-0x120], xmm0
            e_pw_post.SaveVolatile();
            e_pw_post.SaveXmm();
            e_pw_post.Bytes({0x48, 0x89, 0xef}); // mov rdi, rbp
            e_pw_post.CallAbs(reinterpret_cast<const void*>(&OnPointWindAfterEval));
            e_pw_post.RestoreXmm();
            e_pw_post.RestoreVolatile();
            e_pw_post.Bytes({0x83, 0xbd, 0x24, 0xfe, 0xff, 0xff, 0x01}); // cmp dword [rbp-0x1dc], 1
            e_pw_post.Bytes({0x74, 0x11});                               // je loop_pass1 (+17 B)
            e_pw_post.Bytes({0x4c, 0x89, 0xf7});                         // mov rdi, r14 (3 B)
            e_pw_post.JmpAbs(image_base + 0x105320b);                    // 14 B
            e_pw_post.JmpAbs(image_base + 0x1052b53);                    // loop_pass1
            if (Detour(0x1052b49, exp_pw_init10, 10, e_init) &&
                Detour(0x1052b84, exp_pw_seed11, 11, e_pw_seed) &&
                Detour(0x1053200, exp_pw_post11, 11, e_pw_post)) {
                ++hooked;
            }
        }
    }

    // 3. Fix SprjPointWind's particle velocity interval (0x10532f2: vdivss xmm10, xmm1, [rbp - 0x160])
    //    to use `[sim + 0x108]` (prev_sdt) when valid.
    constexpr u8 expected8[8] = {0xc5, 0x72, 0x5e, 0x95, 0xa0, 0xfe, 0xff, 0xff};
    Emitter e_vel;
    e_vel.Bytes({0x8b, 0x81, 0x08, 0x01, 0x00, 0x00}); // mov eax, dword ptr [rcx + 0x108]
    e_vel.Bytes({0x3d, 0xac, 0xc5, 0x27, 0x37});       // cmp eax, 0x3727c5ac (1e-5f)
    const std::size_t jbe_disp = e_vel.size + 1;
    e_vel.Bytes({0x76, 0x00});                         // jbe fallback
    e_vel.Bytes({0x3d, 0xcd, 0xcc, 0x4c, 0x3e});       // cmp eax, 0x3e4ccccd (0.2f)
    const std::size_t jae_disp = e_vel.size + 1;
    e_vel.Bytes({0x73, 0x00});                         // jae fallback
    e_vel.Bytes({0xc5, 0x79, 0x6e, 0xd0});             // vmovd xmm10, eax
    const std::size_t jmp_disp = e_vel.size + 1;
    e_vel.Bytes({0xeb, 0x00});                         // jmp divide
    const std::size_t fallback_off = e_vel.size;
    e_vel.code[jbe_disp] = static_cast<u8>(fallback_off - (jbe_disp + 1));
    e_vel.code[jae_disp] = static_cast<u8>(fallback_off - (jae_disp + 1));
    e_vel.Bytes({0xc5, 0x7a, 0x10, 0x95, 0xa0, 0xfe, 0xff, 0xff}); // vmovss xmm10, dword ptr [rbp - 0x160]
    const std::size_t divide_off = e_vel.size;
    e_vel.code[jmp_disp] = static_cast<u8>(divide_off - (jmp_disp + 1));
    e_vel.Bytes({0xc4, 0x41, 0x72, 0x5e, 0xd2});       // vdivss xmm10, xmm1, xmm10
    e_vel.JmpAbs(image_base + 0x10532fa);
    if (Detour(0x10532f2, expected8, sizeof(expected8), e_vel)) {
        ++hooked;
    }
    return hooked;
}

using ClothStageFn = void(PS4_SYSV_ABI*)(u8*, u8*, u8*, float);
using ClothXferFn = void(PS4_SYSV_ABI*)(u8*, u8*, const void*, const void*, float);
ClothStageFn orig_cloth_integrate = nullptr;
ClothStageFn orig_cloth_collide = nullptr;
ClothXferFn orig_cloth_xfer = nullptr;

void PS4_SYSV_ABI TraceTransferMotion(u8* op, u8* sim, const void* prev_m, const void* cur_m,
                                      float dt) noexcept {
    if (!orig_cloth_xfer) {
        return;
    }
    if (!ClothTraceEnabled() || !sim) {
        orig_cloth_xfer(op, sim, prev_m, cur_m, dt);
        return;
    }
    float pre_xyz[512 * 3];
    const int snap_n = TraceSnapshotFreeParticles(sim, pre_xyz, 512);
    orig_cloth_xfer(op, sim, prev_m, cur_m, dt);
    cloth_trace.xfer_calls.fetch_add(1u, std::memory_order_relaxed);
    if (snap_n > 0) {
        const u8* sim_data = *reinterpret_cast<const u8* const*>(sim + 0x10);
        const float* cur = *reinterpret_cast<const float* const*>(sim + 0x18);
        if (sim_data && cur) {
            const float* pdatas = *reinterpret_cast<const float* const*>(sim_data + 0x38);
            const std::int32_t np = *reinterpret_cast<const std::int32_t*>(sim_data + 0x40);
            if (pdatas && np >= snap_n) {
                double sum_m = 0.0, max_m = 0.0;
                int free_cnt = 0;
                for (int i = 0; i < snap_n; ++i) {
                    if (!(pdatas[i * 4 + 1] > 0.0f)) continue;
                    const float dx = cur[i * 4 + 0] - pre_xyz[i * 3 + 0];
                    const float dy = cur[i * 4 + 1] - pre_xyz[i * 3 + 1];
                    const float dz = cur[i * 4 + 2] - pre_xyz[i * 3 + 2];
                    const double d = std::sqrt(double(dx) * double(dx) + double(dy) * double(dy) +
                                               double(dz) * double(dz));
                    if (std::isfinite(d) && d < 2.0) {
                        sum_m += d;
                        if (d > max_m) max_m = d;
                        ++free_cnt;
                    }
                }
                if (free_cnt > 0) {
                    cloth_trace.xfer_um_sum.fetch_add(
                        static_cast<std::uint64_t>((sum_m / double(free_cnt)) * 1'000'000.0),
                        std::memory_order_relaxed);
                    TraceUpdateAtomicMaxU32(
                        cloth_trace.xfer_max_um,
                        static_cast<std::uint32_t>(std::min(max_m * 1'000'000.0, 999999.0)));
                }
            }
        }
    }
}

void PS4_SYSV_ABI TraceIntegrate(u8* op, u8* ctx, u8* sim, float sdt) noexcept {
    if (!orig_cloth_integrate) {
        return;
    }
    if (!ClothTraceEnabled() || !sim) {
        orig_cloth_integrate(op, ctx, sim, sdt);
        return;
    }
    float pre_xyz[512 * 3];
    float inertial_xyz[512 * 3];
    const int snap_n = TraceSnapshotFreeParticles(sim, pre_xyz, 512);
    if (snap_n > 0) {
        const float* prv = *reinterpret_cast<const float* const*>(sim + 0x28);
        const float eff_damp = *reinterpret_cast<const float*>(sim + 0x10c);
        if (prv) {
            for (int i = 0; i < snap_n; ++i) {
                inertial_xyz[i * 3 + 0] = pre_xyz[i * 3 + 0] + eff_damp * (pre_xyz[i * 3 + 0] - prv[i * 4 + 0]);
                inertial_xyz[i * 3 + 1] = pre_xyz[i * 3 + 1] + eff_damp * (pre_xyz[i * 3 + 1] - prv[i * 4 + 1]);
                inertial_xyz[i * 3 + 2] = pre_xyz[i * 3 + 2] + eff_damp * (pre_xyz[i * 3 + 2] - prv[i * 4 + 2]);
            }
        }
    }
    orig_cloth_integrate(op, ctx, sim, sdt);
    if (snap_n > 0) {
        TraceRecordStageDelta(0, sim, pre_xyz, snap_n, sdt);
        TraceRecordStageDelta(7, sim, inertial_xyz, snap_n, sdt);
    }
}

void PS4_SYSV_ABI TraceCollide(u8* op, u8* ctx, u8* sim, float sdt) noexcept {
    if (!orig_cloth_collide) {
        return;
    }
    if ((ClothSkipMask() & (1u << 6)) != 0u) {
        return;
    }
    if (!ClothTraceEnabled() || !sim) {
        orig_cloth_collide(op, ctx, sim, sdt);
        return;
    }
    float pre_xyz[512 * 3];
    const int snap_n = TraceSnapshotFreeParticles(sim, pre_xyz, 512);
    orig_cloth_collide(op, ctx, sim, sdt);
    if (snap_n > 0) {
        TraceRecordStageDelta(6, sim, pre_xyz, snap_n, sdt);
    }
}

bool HookClothStageTrampoline(u64 site, const u8* expected6, const void* wrapper,
                              ClothStageFn* orig_out) {
    if (std::memcmp(reinterpret_cast<const u8*>(image_base + site), expected6, 6) != 0) {
        return false;
    }
    Emitter tramp;
    for (int i = 0; i < 6; ++i) tramp.Bytes({expected6[i]});
    tramp.JmpAbs(image_base + site + 6);
    if (tramp.overflow || stub_used + tramp.size + 64 > StubPageSize) {
        return false;
    }
    u8* tramp_addr = stub_page + stub_used;
    std::memcpy(tramp_addr, tramp.code.data(), tramp.size);
    stub_used += (tramp.size + 15) & ~std::size_t(15);
    *orig_out = reinterpret_cast<ClothStageFn>(tramp_addr);
    Emitter e;
    e.JmpAbs(reinterpret_cast<u64>(wrapper));
    if (!Detour(site, expected6, 6, e)) {
        *orig_out = nullptr;
        return false;
    }
    return true;
}

int InstallClothTraceHooks() {
    constexpr u8 std_pro6[6] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};
    const bool skip_col = (ClothSkipMask() & (1u << 6)) != 0u;
    if (!ClothTraceEnabled()) {
        if (skip_col) {
            (void)HookClothStageTrampoline(0x774260, std_pro6, reinterpret_cast<const void*>(&TraceCollide),
                                           &orig_cloth_collide);
        }
        return 0;
    }
    auto hook_entry = [](u64 site, const u8* expected, std::size_t len, const void* fn) -> bool {
        Emitter e;
        e.SaveVolatile();
        e.SaveXmm();
        e.CallAbs(fn);
        e.RestoreXmm();
        e.RestoreVolatile();
        for (std::size_t i = 0; i < len; ++i) e.Bytes({expected[i]});
        e.JmpAbs(image_base + site + len);
        return Detour(site, expected, len, e);
    };
    constexpr u8 fd4_wind5[5] = {0xc5, 0xfa, 0x11, 0x47, 0x38};
    constexpr u8 simp_wind9[9] = {0x48, 0x8b, 0x46, 0x10, 0xc5, 0xfa, 0x10, 0x48, 0x78};
    int n = 0;
    if (hook_entry(0xfd5440, std_pro6, 6, reinterpret_cast<const void*>(&ProbeWorldStep))) ++n;
    if (hook_entry(0x76e370, std_pro6, 6, reinterpret_cast<const void*>(&ProbeSimOp))) ++n;
    if (hook_entry(0x29552d0, fd4_wind5, 5, reinterpret_cast<const void*>(&ProbeFd4WindUpd))) ++n;
    if (hook_entry(0xfd8120, std_pro6, 6, reinterpret_cast<const void*>(&ProbeAreaWindUpd))) ++n;
    if (hook_entry(0xfd8350, std_pro6, 6, reinterpret_cast<const void*>(&ProbeAreaWindApply))) ++n;
    if (hook_entry(0x1051fa0, std_pro6, 6, reinterpret_cast<const void*>(&ProbeSprjWindApply))) ++n;
    if (hook_entry(0x1052af0, std_pro6, 6, reinterpret_cast<const void*>(&ProbePointWindApply))) ++n;
    if (cloth_dir_wind_hooked ||
        hook_entry(0x1053b90, std_pro6, 6, reinterpret_cast<const void*>(&ProbeDirWindApply))) {
        ++n;
    }
    if (hook_entry(0x71fb60, simp_wind9, 9, reinterpret_cast<const void*>(&ProbeSimpleWindApply))) ++n;
    if (HookClothStageTrampoline(0x773df0, std_pro6, reinterpret_cast<const void*>(&TraceIntegrate),
                                 &orig_cloth_integrate)) {
        ++n;
    }
    if (HookClothStageTrampoline(0x774260, std_pro6, reinterpret_cast<const void*>(&TraceCollide),
                                 &orig_cloth_collide)) {
        ++n;
    }
    if (HookClothStageTrampoline(0x771290, std_pro6, reinterpret_cast<const void*>(&TraceTransferMotion),
                                 reinterpret_cast<ClothStageFn*>(&orig_cloth_xfer))) {
        ++n;
    }
    std::printf("Frame rate: BB_CLOTH_TRACE=1 armed (%d/12 cloth+wind+stage probes, wind_off=%d, skip_mask=0x%x)\n",
                n, ClothWindOffEnabled() ? 1 : 0, ClothSkipMask());
    return n;
}

/// A mapping within +-2 GiB of the image (below or above it), for rel32 jumps and RIP-relative operands.
u8* MapNear(int prot, u64 image_size) {
    const u64 above_base = (image_base + image_size + 0xFFFFull) & ~0xFFFFull;
    for (u64 k = 1; k <= 64; ++k) {
        for (const u64 hint : {image_base - k * (8ull << 20), above_base + k * (8ull << 20)}) {
            void* p = mmap(reinterpret_cast<void*>(hint & ~0xFFFFull), StubPageSize, prot,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
            if (p != MAP_FAILED) return static_cast<u8*>(p);
        }
    }
    return nullptr;
}
} // namespace

void PatchImage(unsigned char* image, std::uint64_t image_size) {
    // BB_FPS (run.sh, the launcher's frame rate): uncap (default), 60 or 90 (the same, at most that
    // many frames a second), 30 (the game's own). BB_FRAME_HOOKS=0: the community's frame rate
    // patches instead (scripts/patches.py; comparisons).
    const char* fps = std::getenv("BB_FPS");
    const char* hooks = std::getenv("BB_FRAME_HOOKS");
    const bool is_uncapped = !fps || std::strcmp(fps, "uncap") == 0;
    const int mode = is_uncapped ? MaxRate : std::atoi(fps);
    if (mode <= 30 || (hooks && hooks[0] == '0')) {
        return;
    }
    image_base = reinterpret_cast<u64>(image);
    u8* data = MapNear(PROT_READ | PROT_WRITE, image_size);
    stub_page = MapNear(PROT_READ | PROT_WRITE | PROT_EXEC, image_size);
    if (!data || !stub_page) {
        std::printf("Frame rate: no memory next to the game's code: 30 FPS\n");
        return;
    }
    live = new (data) Live{};
    std::memcpy(&model_turn_base, image + 0x4928dfc, sizeof(model_turn_base));
    live->model_turn = model_turn_base;
    // The mode's rate and BB_FPS_LIMIT (the launcher's frame rate limit): the game's pacer waits for
    // it itself, before the frame's simulation (lower latency than holding finished frames back at
    // the present). When uncapped (rate == 0), live->shortest is 0.0f (truly unlimited).
    int rate = is_uncapped ? 0 : std::clamp(mode, 30, MaxRate);
    if (const char* limit = std::getenv("BB_FPS_LIMIT"); limit && std::atoi(limit) > 0) {
        const int lim = std::clamp(std::atoi(limit), 30, MaxRate);
        rate = rate > 0 ? std::min(rate, lim) : lim;
    }
    live->shortest = rate > 0 ? (1.0f / float(rate)) : ShortestFrame;
    base_rate = rate;
    if (const int saved = live_limit.load(std::memory_order_relaxed); saved > 0) {
        live->shortest = 1.0f / float(std::clamp(saved, 30, MaxRate));
    }
    // The core first: without all three the game keeps its own frame rate.
    if (!(HookPacerMode() && HookPacerLate() && HookStep())) {
        std::printf("Frame rate: the game's frame pacer is not the expected code: 30 FPS\n");
        mprotect(stub_page, StubPageSize, PROT_READ | PROT_EXEC);
        return;
    }
    unlocked = true;
    int steps = HookFixedStep(0x17f9b71, -0x48) + HookFixedStep(0x1572a7d, -0x30);
    const bool foliage = HookFoliageSway();
    const int run = HookRunSpeed();
    const bool cloth = HookClothStiffness();
    const int cloth_friction = HookClothFriction();
    cloth_world_synced = HookClothWorldSync();
    const bool cloth_prestep_hooked = HookClothPreStep();
    cloth_fixed30_hooked = cloth_world_synced && cloth_prestep_hooked && HookClothFixed30();
    const int cloth_r8_sites = HookClothSolveConstraintsUseFactor();
    cloth_dir_wind_hooked = HookClothDirWind();
    const int cloth_wind_eval = HookClothWindEval();
    (void)InstallClothTraceHooks();
    int redirected = 0;
    for (const Redirect& r : Redirects) {
        if (RedirectOperand(r)) {
            ++redirected;
        } else {
            std::printf("Frame rate: %#llx is not the expected code, left as it is\n", (unsigned long long)r.site);
        }
    }
    int spin_redirected = 0;
    if (CleanPacerEnabled()) {
        pacer_wait_hooked = HookPacerWait();
        constexpr Redirect PacerSpinRedirects[] = {
            {0x2034b9e, 0x4929390, offsetof(Live, spin_margin)},     // 1.0 ms threshold -> 200 us
            {0x2034ba8, 0x4929394, offsetof(Live, spin_margin)},     // 5.0 ms threshold -> 200 us
            {0x2034bb2, 0x4929398, offsetof(Live, neg_spin_margin)}, // -5.0 ms subtract -> -200 us
        };
        for (const Redirect& r : PacerSpinRedirects) {
            if (RedirectOperand(r)) {
                ++spin_redirected;
            }
        }
    }
    mprotect(stub_page, StubPageSize, PROT_READ | PROT_EXEC);
    std::printf("Frame rate: the model preview turns %.5f rad per stick unit per 30 FPS frame\n",
                model_turn_base);
    const bool cloth_fix_on = ClothFixEnabled();
    const float logged_cloth_exp =
        ClothExpOverride() > 0.0f ? ClothExpOverride() : -cloth_exponent;
    const int logged_t5_mode = [&] {
        if (!cloth_fix_on) return 0;
        const char* env = std::getenv("BB_CLOTH_TYPE5");
        return (env && *env) ? std::atoi(env) : 3;
    }();
    std::printf("Frame rate: free (hooks, no patches): the game steps by each frame's time "
                "(pacer=%s%s, sim=1/%d..1/30 s); %d fixed steps, %d of %zu per-frame values follow the frame's "
                "time, foliage sway=%s, pacer usleep spin=%d/3, the run's wall check %s, "
                "cloth=%s (stiffness %s, exp=%.3f, type5=%d [r8=%d/2], friction=%d/17, world_dt_sync=%s, "
                "dir_wind=%s, wind_eval=%d/3, pbd_damp=%.2f, post_col=%d)\n",
                live->shortest > 0.0f ? "capped" : "truly uncapped (0.0 ms)",
                pacer_wait_hooked ? " + monotonic-abstime" : "", MaxRate,
                steps, redirected, std::size(Redirects),
                foliage ? "fixed-amplitude + dt-phase" : "unchanged", spin_redirected,
                run == 1 ? "by speed" : run == 2 ? "by speed (a Sprint Fix patch did it already)"
                                                 : "unchanged (not found)",
                cloth_fixed30_hooked ? "fixed30+interp" : "native",
                cloth ? "by solves per second" : "per frame (not found)",
                logged_cloth_exp, logged_t5_mode, cloth_r8_sites, cloth_friction,
                cloth_world_synced ? "on" : "off", cloth_dir_wind_hooked ? "native" : "orig",
                cloth_wind_eval, (!cloth_fixed30_hooked && cloth_prestep_hooked) ? ClothDampRetention30() : 1.0f,
                cloth_fixed30_hooked ? ClothPostCollideMode() : 0);
}

bool CleanPacerEnabled() noexcept {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_CLEAN_PACER");
        return !env || !*env || !(env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

bool Unlocked() {
    return unlocked.load(std::memory_order_relaxed);
}
void SetLimit(int fps) {
    live_limit.store(fps, std::memory_order_relaxed);
    pacer_epoch.fetch_add(1, std::memory_order_relaxed);
    if (!unlocked.load(std::memory_order_relaxed) || !live) {
        return;
    }
    const int rate = fps > 0 ? std::clamp(fps, 30, MaxRate) : base_rate;
    live->shortest = rate > 0 ? (1.0f / float(rate)) : ShortestFrame;
    if (rate > 0) {
        std::printf("Frame rate: limit %d FPS (live)\n", rate);
    } else {
        std::printf("Frame rate: truly uncapped (live, 0.0 ms pacer target)\n");
    }
}
int Limit() {
    return live_limit.load(std::memory_order_relaxed);
}

std::uint64_t NowMonotonicNs() noexcept {
    return MonotonicNs();
}

void SleepUntilMonotonicNs(std::uint64_t deadline_ns, int timer_slot) noexcept {
#ifdef __linux__
    static bool slot_timerslack_set[2] = {false, false};
    const int idx = (timer_slot == 1) ? 1 : 0;
    if (!slot_timerslack_set[idx]) {
        slot_timerslack_set[idx] = true;
        prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
    }
#endif
    const u64 now = MonotonicNs();
    if (now < deadline_ns) {
        SleepUntilDeadlineImpl(now, deadline_ns, timer_slot);
    }
}

std::uint64_t TargetPeriodNs() noexcept {
    if (!unlocked.load(std::memory_order_relaxed) || !live) {
        return 0ull;
    }
    const float t = live->shortest;
    const u64 raw_ns = (t > 0.0f && std::isfinite(t))
                           ? static_cast<u64>(std::llround(double(t) * 1'000'000'000.0))
                           : 0ull;
    if (raw_ns > 0ull && BbStats::fg_active.load(std::memory_order_relaxed)) {
        const u64 mult = static_cast<u64>(
            std::clamp(BbStats::fg_multiplier.load(std::memory_order_relaxed), 1, 4));
        return ComputeFgSimPeriodNs(raw_ns) / mult;
    }
    return raw_ns;
}

std::uint64_t PacerEpoch() noexcept {
    return pacer_epoch.load(std::memory_order_relaxed);
}

bool PresentPacingEnabled() noexcept {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_PRESENT_PACE");
        return !env || !*env || !(env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

float CurrentFramesPer30() noexcept {
    const float f = current_frames_per_30.load(std::memory_order_relaxed);
    return (f > 0.0f && std::isfinite(f)) ? std::clamp(f, 0.05f, 1.5f) : 1.0f;
}

std::uint64_t FrameSeq() noexcept {
    return sim_frame_seq.load(std::memory_order_relaxed);
}
} // namespace BbFrameRate
