// SPDX-License-Identifier: GPL-2.0-or-later
// Native Vulkan Low Latency manager: NVIDIA Reflex (VK_NV_low_latency2) and AMD Anti-Lag 2
// (VK_AMD_anti_lag). Thread-safe across MOMainThread (simulation/pacer), bb:GpuComm (render
// submit), and bb:Present (swapchain recreate & present) with zero static PT_TLS overhead.
#pragma once

#include <cstdint>

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

class Instance;

namespace LowLatency {

/// Initializes the low-latency backend from the active Vulkan device (called by Instance).
void Init(const Instance& instance);

/// Tears down low-latency resources before the Vulkan device is destroyed (called by Instance).
void Shutdown();

/// Registers the newly created swapchain and applies the active sleep mode (called by Swapchain).
void AttachSwapchain(vk::SwapchainKHR swapchain);

/// Unregisters the active swapchain before it is destroyed or recreated (called by Swapchain).
void DetachSwapchain();

/// Applies live mode changes from BbSettings::Get().low_latency (fast atomic check).
void SyncModeIfChanged();

/// Marks the end of the previous simulation frame on MOMainThread (before pacer/latency sleep).
void EndSimulationFrame();

/// Performs driver-paced low-latency sleep before simulation/input sampling.
/// Returns true if a driver low-latency call was executed (caller should refresh monotonic clock).
bool Sleep(bool loading_burst);

/// Starts a new simulation frame and emits SIMULATION_START + INPUT_SAMPLE markers.
/// Returns the monotonically increasing present ID (>= 1) for this simulation frame.
u64 BeginSimulationFrame();

/// Assigns a strictly monotonically increasing present ID (>= 1) for a frame being prepared on
/// bb:GpuComm (Presenter::PrepareFrame).
u64 AssignRenderPresentId();

/// Emits RENDERSUBMIT_START / RENDERSUBMIT_END markers on bb:GpuComm.
void MarkRenderSubmitStart(u64 present_id);
void MarkRenderSubmitEnd(u64 present_id);

/// Emits PRESENT_START / PRESENT_END markers (and AMD Anti-Lag 2 present stage) on bb:Present.
void MarkPresentStart(u64 present_id);
void MarkPresentEnd(u64 present_id);

/// Polls NVIDIA Reflex frame timings at 1 Hz (non-blocking) to populate BbStats telemetry.
void PollTimingsIfDue();

/// True when VK_NV_low_latency2 is active and enabled (Off returns false).
bool IsNvReflexActive() noexcept;

/// True when either NVIDIA Reflex or AMD Anti-Lag 2 is supported on the active device.
bool IsSupported() noexcept;

/// Resolves the effective low-latency mode (enforces NVIDIA Reflex On when DLSS-G is active).
int EffectiveMode(int backend, int raw_mode) noexcept;

/// Human-readable active status string for logs/HUD (e.g. "NVIDIA Reflex (On + Boost)").
const char* StatusLabel() noexcept;

} // namespace LowLatency

} // namespace Vulkan
