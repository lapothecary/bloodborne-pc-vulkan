// SPDX-License-Identifier: GPL-2.0-or-later
#include "vulkan/low_latency.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <shared_mutex>

#include "bbport_settings.h"
#include "bbport_toggles.h"
#include "common/logging/log.h"
#include "vulkan/instance.h"

namespace Vulkan::LowLatency {

namespace {

std::shared_mutex g_swapchain_mutex;
vk::Device g_device{};
vk::SwapchainKHR g_swapchain{};
vk::Semaphore g_sleep_semaphore{};

std::atomic<int> g_backend{BbSettings::LowLatencyBackendNone};
std::atomic<int> g_applied_mode{-1};

std::atomic<u64> g_sleep_sema_value{0};
std::atomic<u64> g_next_sim_id{0};
std::atomic<u64> g_active_sim_id{0};
std::atomic<u64> g_last_render_id{0};
std::atomic<u64> g_amd_input_sent_id{0};
std::atomic<u64> g_last_timings_poll_ns{0};

u64 NowNs() noexcept {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count());
}

void ApplyModeLocked(int mode) {
    const int backend = g_backend.load(std::memory_order_acquire);
    if (!g_device || backend == BbSettings::LowLatencyBackendNone) {
        g_applied_mode.store(BbSettings::LowLatencyOff, std::memory_order_release);
        return;
    }

#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
    if (backend == BbSettings::LowLatencyBackendNvReflex) {
        if (!g_swapchain) {
            return;
        }
        const bool enable = (mode != BbSettings::LowLatencyOff);
        const bool boost = (mode == BbSettings::LowLatencyBoost);
        // minimumIntervalUs = 0: BbFrameRate + Presenter enforce the exact FPS cap with
        // sub-0.15 ms jitter; Reflex sleeps strictly when the GPU/VSync queue backs up.
        const vk::LatencySleepModeInfoNV sleep_mode{
            .lowLatencyMode = enable ? vk::True : vk::False,
            .lowLatencyBoost = (enable && boost) ? vk::True : vk::False,
            .minimumIntervalUs = 0u,
        };
        const vk::Result res = g_device.setLatencySleepModeNV(g_swapchain, sleep_mode);
        if (res == vk::Result::eSuccess) {
            const int prev = g_applied_mode.exchange(mode, std::memory_order_acq_rel);
            if (prev != mode) {
                std::printf("Low Latency: NVIDIA Reflex -> %s\n",
                            mode == BbSettings::LowLatencyBoost
                                ? "On + Boost"
                                : (mode == BbSettings::LowLatencyOn ? "On" : "Off"));
                std::fflush(stdout);
            }
        } else {
            std::printf("Low Latency: vkSetLatencySleepModeNV failed: %s\n",
                        vk::to_string(res).c_str());
            std::fflush(stdout);
            LOG_WARNING(Render_Vulkan, "vkSetLatencySleepModeNV failed: {}", vk::to_string(res));
        }
        return;
    }
#endif

#ifdef VK_AMD_ANTI_LAG_EXTENSION_NAME
    if (backend == BbSettings::LowLatencyBackendAmdAntiLag) {
        const bool enable = (mode != BbSettings::LowLatencyOff);
        if (!enable) {
            const vk::AntiLagDataAMD data{
                .mode = vk::AntiLagModeAMD::eOff,
                .maxFPS = 0u,
                .pPresentationInfo = nullptr,
            };
            g_device.antiLagUpdateAMD(data);
        }
        const int prev = g_applied_mode.exchange(mode, std::memory_order_acq_rel);
        if (prev != mode) {
            std::printf("Low Latency: AMD Anti-Lag 2 -> %s\n", enable ? "On" : "Off");
            std::fflush(stdout);
        }
        return;
    }
#endif
}

#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
void EmitNvMarker(u64 present_id, vk::LatencyMarkerNV marker) {
    if (present_id == 0 ||
        g_backend.load(std::memory_order_relaxed) != BbSettings::LowLatencyBackendNvReflex) {
        return;
    }
    const int mode = EffectiveMode(BbSettings::LowLatencyBackendNvReflex,
                                   BbSettings::Get().low_latency.load(std::memory_order_relaxed));
    if (mode == BbSettings::LowLatencyOff) {
        return;
    }
    std::shared_lock lock{g_swapchain_mutex};
    if (!g_device || !g_swapchain) {
        return;
    }
    const vk::SetLatencyMarkerInfoNV info{
        .presentID = present_id,
        .marker = marker,
    };
    g_device.setLatencyMarkerNV(g_swapchain, info);
}
#endif

} // namespace

int EffectiveMode(int backend, int raw_mode) noexcept {
    const int clamped = std::clamp(raw_mode, int(BbSettings::LowLatencyOff),
                                   int(BbSettings::LowLatencyBoost));
    if (backend == BbSettings::LowLatencyBackendNone) {
        return BbSettings::LowLatencyOff;
    }
    if (backend == BbSettings::LowLatencyBackendNvReflex &&
        clamped == BbSettings::LowLatencyOff &&
        BbSettings::IsDlssFrameGen(BbSettings::EffectiveFrameGen())) {
        return BbSettings::LowLatencyOn;
    }
    if (backend == BbSettings::LowLatencyBackendAmdAntiLag &&
        clamped == BbSettings::LowLatencyBoost) {
        return BbSettings::LowLatencyOn;
    }
    return clamped;
}

void Init(const Instance& instance) {
    std::unique_lock lock{g_swapchain_mutex};
    g_device = instance.GetDevice();
    g_swapchain = vk::SwapchainKHR{};
    g_sleep_semaphore = vk::Semaphore{};
    g_sleep_sema_value.store(0, std::memory_order_relaxed);
    g_next_sim_id.store(0, std::memory_order_relaxed);
    g_active_sim_id.store(0, std::memory_order_relaxed);
    g_last_render_id.store(0, std::memory_order_relaxed);
    g_amd_input_sent_id.store(0, std::memory_order_relaxed);
    g_last_timings_poll_ns.store(0, std::memory_order_relaxed);
    g_applied_mode.store(-1, std::memory_order_relaxed);
    BbStats::reflex_pc_latency_us.store(0, std::memory_order_relaxed);
    BbStats::low_latency_sleep_ns.store(0, std::memory_order_relaxed);

    int backend = BbSettings::LowLatencyBackendNone;
#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
    if (instance.IsLowLatency2Supported()) {
        vk::SemaphoreTypeCreateInfo type_info{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        };
        vk::SemaphoreCreateInfo sema_info{
            .pNext = &type_info,
        };
        auto [res, sema] = g_device.createSemaphore(sema_info);
        if (res == vk::Result::eSuccess) {
            g_sleep_semaphore = sema;
            backend = BbSettings::LowLatencyBackendNvReflex;
        } else {
            LOG_WARNING(Render_Vulkan, "Low Latency: failed to create timeline semaphore ({})",
                        vk::to_string(res));
        }
    }
#endif
#ifdef VK_AMD_ANTI_LAG_EXTENSION_NAME
    if (backend == BbSettings::LowLatencyBackendNone && instance.IsAmdAntiLagSupported()) {
        backend = BbSettings::LowLatencyBackendAmdAntiLag;
    }
#endif

    g_backend.store(backend, std::memory_order_release);
    BbSettings::ConfigureLowLatencySupport(backend);

    if (backend == BbSettings::LowLatencyBackendNvReflex) {
        std::printf("Low Latency: NVIDIA Reflex (VK_NV_low_latency2) ready\n");
    } else if (backend == BbSettings::LowLatencyBackendAmdAntiLag) {
        std::printf("Low Latency: AMD Anti-Lag 2 (VK_AMD_anti_lag) ready\n");
    } else {
        std::printf("Low Latency: unavailable on this GPU/driver\n");
    }
}

void Shutdown() {
    std::unique_lock lock{g_swapchain_mutex};
#ifdef VK_AMD_ANTI_LAG_EXTENSION_NAME
    if (g_device &&
        g_backend.load(std::memory_order_relaxed) == BbSettings::LowLatencyBackendAmdAntiLag &&
        g_applied_mode.load(std::memory_order_relaxed) != BbSettings::LowLatencyOff) {
        const vk::AntiLagDataAMD data{
            .mode = vk::AntiLagModeAMD::eOff,
            .maxFPS = 0u,
            .pPresentationInfo = nullptr,
        };
        g_device.antiLagUpdateAMD(data);
    }
#endif
    if (g_device && g_sleep_semaphore) {
        g_device.destroySemaphore(g_sleep_semaphore);
        g_sleep_semaphore = vk::Semaphore{};
    }
    g_swapchain = vk::SwapchainKHR{};
    g_device = vk::Device{};
    g_backend.store(BbSettings::LowLatencyBackendNone, std::memory_order_release);
    g_applied_mode.store(-1, std::memory_order_release);
    BbStats::reflex_pc_latency_us.store(0, std::memory_order_relaxed);
}

void AttachSwapchain(vk::SwapchainKHR swapchain) {
    std::unique_lock lock{g_swapchain_mutex};
    g_swapchain = swapchain;
    g_applied_mode.store(-1, std::memory_order_relaxed);
    const int backend = g_backend.load(std::memory_order_acquire);
    if (backend != BbSettings::LowLatencyBackendNone && g_swapchain) {
        const int want = EffectiveMode(
            backend, BbSettings::Get().low_latency.load(std::memory_order_relaxed));
        ApplyModeLocked(want);
    }
}

void DetachSwapchain() {
    std::unique_lock lock{g_swapchain_mutex};
    g_swapchain = vk::SwapchainKHR{};
}

void SyncModeIfChanged() {
    const int backend = g_backend.load(std::memory_order_acquire);
    if (backend == BbSettings::LowLatencyBackendNone) {
        return;
    }
    const int want =
        EffectiveMode(backend, BbSettings::Get().low_latency.load(std::memory_order_relaxed));
    if (want == g_applied_mode.load(std::memory_order_acquire)) {
        return;
    }
    std::unique_lock lock{g_swapchain_mutex};
    ApplyModeLocked(want);
    if (want == BbSettings::LowLatencyOff) {
        BbStats::reflex_pc_latency_us.store(0, std::memory_order_relaxed);
    }
}

void EndSimulationFrame() {
#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
    const u64 sim_id = g_active_sim_id.load(std::memory_order_relaxed);
    if (sim_id != 0) {
        EmitNvMarker(sim_id, vk::LatencyMarkerNV::eSimulationEnd);
    }
#endif
}

bool Sleep(bool loading_burst) {
    SyncModeIfChanged();
    const int backend = g_backend.load(std::memory_order_acquire);
    if (loading_burst || backend == BbSettings::LowLatencyBackendNone) {
        return false;
    }
    const int mode =
        EffectiveMode(backend, BbSettings::Get().low_latency.load(std::memory_order_relaxed));
    if (mode == BbSettings::LowLatencyOff) {
        return false;
    }

#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
    if (backend == BbSettings::LowLatencyBackendNvReflex) {
        vk::Device dev{};
        vk::Semaphore sema{};
        u64 wait_val = 0;
        {
            std::shared_lock lock{g_swapchain_mutex};
            if (!g_device || !g_swapchain || !g_sleep_semaphore) {
                return false;
            }
            wait_val = g_sleep_sema_value.fetch_add(1, std::memory_order_acq_rel) + 1;
            const vk::LatencySleepInfoNV sleep_info{
                .signalSemaphore = g_sleep_semaphore,
                .value = wait_val,
            };
            const vk::Result res = g_device.latencySleepNV(g_swapchain, sleep_info);
            if (res != vk::Result::eSuccess) {
                return false;
            }
            dev = g_device;
            sema = g_sleep_semaphore;
        }
        // Wait outside g_swapchain_mutex so a concurrent swapchain resize can never deadlock.
        const u64 t0 = NowNs();
        const vk::SemaphoreWaitInfo wait_info{
            .semaphoreCount = 1,
            .pSemaphores = &sema,
            .pValues = &wait_val,
        };
        (void)dev.waitSemaphores(wait_info, 100'000'000ull);
        const u64 t1 = NowNs();
        if (t1 > t0) {
            BbStats::low_latency_sleep_ns.fetch_add(t1 - t0, std::memory_order_relaxed);
        }
        return true;
    }
#endif

#ifdef VK_AMD_ANTI_LAG_EXTENSION_NAME
    if (backend == BbSettings::LowLatencyBackendAmdAntiLag) {
        const u64 next_id = g_next_sim_id.load(std::memory_order_relaxed) + 1;
        const u64 t0 = NowNs();
        {
            std::shared_lock lock{g_swapchain_mutex};
            if (!g_device) {
                return false;
            }
            const vk::AntiLagPresentationInfoAMD pres_info{
                .stage = vk::AntiLagStageAMD::eInput,
                .frameIndex = next_id,
            };
            const vk::AntiLagDataAMD data{
                .mode = vk::AntiLagModeAMD::eOn,
                .maxFPS = 0u,
                .pPresentationInfo = &pres_info,
            };
            g_device.antiLagUpdateAMD(data);
        }
        g_amd_input_sent_id.store(next_id, std::memory_order_relaxed);
        const u64 t1 = NowNs();
        if (t1 > t0) {
            BbStats::low_latency_sleep_ns.fetch_add(t1 - t0, std::memory_order_relaxed);
        }
        return true;
    }
#endif

    return false;
}

u64 BeginSimulationFrame() {
    const u64 id = g_next_sim_id.fetch_add(1, std::memory_order_acq_rel) + 1;
    g_active_sim_id.store(id, std::memory_order_release);

    const int backend = g_backend.load(std::memory_order_acquire);
    if (backend == BbSettings::LowLatencyBackendNone) {
        return id;
    }

#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
    if (backend == BbSettings::LowLatencyBackendNvReflex) {
        EmitNvMarker(id, vk::LatencyMarkerNV::eSimulationStart);
        EmitNvMarker(id, vk::LatencyMarkerNV::eInputSample);
        return id;
    }
#endif

#ifdef VK_AMD_ANTI_LAG_EXTENSION_NAME
    if (backend == BbSettings::LowLatencyBackendAmdAntiLag) {
        const int mode = EffectiveMode(
            backend, BbSettings::Get().low_latency.load(std::memory_order_relaxed));
        if (mode != BbSettings::LowLatencyOff &&
            g_amd_input_sent_id.load(std::memory_order_relaxed) != id) {
            std::shared_lock lock{g_swapchain_mutex};
            if (g_device) {
                const vk::AntiLagPresentationInfoAMD pres_info{
                    .stage = vk::AntiLagStageAMD::eInput,
                    .frameIndex = id,
                };
                const vk::AntiLagDataAMD data{
                    .mode = vk::AntiLagModeAMD::eOn,
                    .maxFPS = 0u,
                    .pPresentationInfo = &pres_info,
                };
                g_device.antiLagUpdateAMD(data);
                g_amd_input_sent_id.store(id, std::memory_order_relaxed);
            }
        }
    }
#endif

    return id;
}

u64 AssignRenderPresentId() {
    const u64 sim_id = g_active_sim_id.load(std::memory_order_acquire);
    u64 prev = g_last_render_id.load(std::memory_order_relaxed);
    while (true) {
        const u64 next = std::max(sim_id, prev + 1);
        if (g_last_render_id.compare_exchange_weak(prev, next, std::memory_order_acq_rel)) {
            return next;
        }
    }
}

void MarkRenderSubmitStart(u64 present_id) {
#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
    EmitNvMarker(present_id, vk::LatencyMarkerNV::eRendersubmitStart);
#else
    (void)present_id;
#endif
}

void MarkRenderSubmitEnd(u64 present_id) {
#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
    EmitNvMarker(present_id, vk::LatencyMarkerNV::eRendersubmitEnd);
#else
    (void)present_id;
#endif
}

void MarkPresentStart(u64 present_id) {
    if (present_id == 0) {
        return;
    }
    SyncModeIfChanged();
    const int backend = g_backend.load(std::memory_order_acquire);
    if (backend == BbSettings::LowLatencyBackendNone) {
        return;
    }
#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
    if (backend == BbSettings::LowLatencyBackendNvReflex) {
        EmitNvMarker(present_id, vk::LatencyMarkerNV::ePresentStart);
        return;
    }
#endif
#ifdef VK_AMD_ANTI_LAG_EXTENSION_NAME
    if (backend == BbSettings::LowLatencyBackendAmdAntiLag) {
        const int mode = EffectiveMode(
            backend, BbSettings::Get().low_latency.load(std::memory_order_relaxed));
        if (mode != BbSettings::LowLatencyOff) {
            std::shared_lock lock{g_swapchain_mutex};
            if (g_device) {
                const vk::AntiLagPresentationInfoAMD pres_info{
                    .stage = vk::AntiLagStageAMD::ePresent,
                    .frameIndex = present_id,
                };
                const vk::AntiLagDataAMD data{
                    .mode = vk::AntiLagModeAMD::eOn,
                    .maxFPS = 0u,
                    .pPresentationInfo = &pres_info,
                };
                g_device.antiLagUpdateAMD(data);
            }
        }
    }
#endif
}

void MarkPresentEnd(u64 present_id) {
#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
    EmitNvMarker(present_id, vk::LatencyMarkerNV::ePresentEnd);
#else
    (void)present_id;
#endif
}

void PollTimingsIfDue() {
#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
    if (g_backend.load(std::memory_order_relaxed) != BbSettings::LowLatencyBackendNvReflex) {
        return;
    }
    const int mode = EffectiveMode(BbSettings::LowLatencyBackendNvReflex,
                                   BbSettings::Get().low_latency.load(std::memory_order_relaxed));
    if (mode == BbSettings::LowLatencyOff) {
        BbStats::reflex_pc_latency_us.store(0, std::memory_order_relaxed);
        return;
    }
    const u64 now_ns = NowNs();
    const u64 prev_ns = g_last_timings_poll_ns.load(std::memory_order_relaxed);
    if (prev_ns != 0 && now_ns < prev_ns + 1'000'000'000ull) {
        return;
    }
    g_last_timings_poll_ns.store(now_ns, std::memory_order_relaxed);

    if (!VULKAN_HPP_DEFAULT_DISPATCHER.vkGetLatencyTimingsNV) {
        return;
    }

    std::shared_lock lock{g_swapchain_mutex};
    if (!g_device || !g_swapchain) {
        return;
    }
    std::array<vk::LatencyTimingsFrameReportNV, 64> reports{};
    vk::GetLatencyMarkerInfoNV info{
        .timingCount = u32(reports.size()),
        .pTimings = reports.data(),
    };
    g_device.getLatencyTimingsNV(g_swapchain, &info);
    if (info.timingCount == 0) {
        return;
    }
    // Average the most recent valid completed frames (up to 16) for stable telemetry.
    u64 sum_us = 0;
    u32 count = 0;
    const u32 n = std::min<u32>(info.timingCount, u32(reports.size()));
    for (u32 i = n; i > 0 && count < 16u; --i) {
        const auto& r = reports[i - 1];
        const u64 start_us = r.simStartTimeUs != 0 ? r.simStartTimeUs : r.inputSampleTimeUs;
        const u64 end_us = r.gpuRenderEndTimeUs != 0 ? r.gpuRenderEndTimeUs : r.presentEndTimeUs;
        if (start_us != 0 && end_us > start_us && (end_us - start_us) < 250'000ull) {
            sum_us += (end_us - start_us);
            ++count;
        }
    }
    if (count > 0) {
        BbStats::reflex_pc_latency_us.store(u32(sum_us / count), std::memory_order_relaxed);
    }
#endif
}

bool IsNvReflexActive() noexcept {
    if (g_backend.load(std::memory_order_relaxed) != BbSettings::LowLatencyBackendNvReflex) {
        return false;
    }
    return EffectiveMode(BbSettings::LowLatencyBackendNvReflex,
                         BbSettings::Get().low_latency.load(std::memory_order_relaxed)) !=
           BbSettings::LowLatencyOff;
}

bool IsSupported() noexcept {
    return g_backend.load(std::memory_order_relaxed) != BbSettings::LowLatencyBackendNone;
}

const char* StatusLabel() noexcept {
    const int backend = g_backend.load(std::memory_order_relaxed);
    if (backend == BbSettings::LowLatencyBackendNone) {
        return "Unavailable";
    }
    const int mode =
        EffectiveMode(backend, BbSettings::Get().low_latency.load(std::memory_order_relaxed));
    if (backend == BbSettings::LowLatencyBackendNvReflex) {
        if (mode == BbSettings::LowLatencyBoost) return "NVIDIA Reflex (On + Boost)";
        if (mode == BbSettings::LowLatencyOn) return "NVIDIA Reflex (On)";
        return "NVIDIA Reflex (Off)";
    }
    return mode != BbSettings::LowLatencyOff ? "AMD Anti-Lag 2 (On)" : "AMD Anti-Lag 2 (Off)";
}

} // namespace Vulkan::LowLatency
