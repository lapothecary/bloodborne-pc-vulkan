// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// File: gpu/cloth_exp/cloth_exp_registry.h
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

namespace BbClothExp {

/// Live-switchable cloth engine execution mode (`Insert -> Developer / Legacy A/B Debug`
/// or `BB_CLOTH_EXP=0 / 1 / 1.1 / 1.2` at startup).
enum class ClothExpMode : int {
    Off = 0,          ///< Baseline production Fixed-30 Hz + rigid-frame interpolation
    Havok1to1 = 1,    ///< Mode 1: 1:1 decompiled C++20 Havok + Wind (40/40 functions, Fixed-30)
    LocalFp64 = 2,    ///< Mode 1.1: Canonical Havok in Character-Local FP64 + Explicit Velocity (Native Any-FPS)
    XpbdEnhanced = 3, ///< Mode 1.2: Next-Gen XPBD Solver (Native Any-FPS)
};

struct ClothExpEntry {
    std::uint64_t vma;
    std::uint32_t ps4_size;
    void* func_ptr;
    const char* name;
    const char* module_name;
    bool detour_guest_entry; ///< false for tiny intra-block dispatchers called via C++20
};

/// Installs non-destructive Zydis trampolines + live-switchable dispatch stubs across all
/// registered experimental cloth/wind functions. Default state is OFF unless `BB_CLOTH_EXP` or
/// `bbport.ini` (`cloth_exp_mode`) selects a mode.
void PatchImage(unsigned char* image, std::uint64_t size) noexcept;

/// Re-applies persisted `bbport.ini` cloth mode, range, and XPBD tuning after `BbSettings::Load()`
/// (environment overrides `BB_CLOTH_EXP*` always take priority when explicitly set).
void SyncFromSettings() noexcept;

/// Live 4-mode selector (safe to switch on the fly at any FPS or with Frame Generation active).
void SetMode(ClothExpMode mode) noexcept;
ClothExpMode GetMode() noexcept;
bool IsLocalFp64Mode() noexcept;
bool IsXpbdMode() noexcept;
bool IsNativeFpsMode() noexcept;

/// Legacy bool helpers (`Enabled()` is true whenever `GetMode() != ClothExpMode::Off`).
void SetEnabled(bool enabled) noexcept;
bool Enabled() noexcept;

/// Live index-range bisection (`[min_idx .. max_idx]` inclusive among registered entries).
void SetActiveRange(int min_idx, int max_idx) noexcept;
void GetActiveRange(int& out_min_idx, int& out_max_idx) noexcept;

/// Introspection for the `Insert` overlay & `verify_cloth_exp_asm.py`.
std::size_t EntryCount() noexcept;
std::size_t InstalledHookCount() noexcept;
const ClothExpEntry* GetEntries() noexcept;
bool IsEntryActive(std::size_t idx) noexcept;
bool TryDispatchPreStep(unsigned char* op, unsigned char* step_info) noexcept;
std::uint64_t TotalDispatchCount() noexcept;
void ResetDispatchCount() noexcept;

} // namespace BbClothExp
