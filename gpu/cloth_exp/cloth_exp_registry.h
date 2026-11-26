// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// File: gpu/cloth_exp/cloth_exp_registry.h
// ============================================================================
#pragma once

#include <cstddef>
#include <cstdint>

namespace BbClothExp {

struct ClothExpEntry {
    std::uint64_t vma;
    std::uint32_t ps4_size;
    void* func_ptr;
    const char* name;
    const char* module_name;
    bool detour_guest_entry; ///< false for tiny intra-block dispatchers called via C++20
};

/// Installs non-destructive Zydis trampolines + live-switchable dispatch stubs across all
/// registered experimental cloth/wind functions. Default state is OFF unless `BB_CLOTH_EXP=1`.
void PatchImage(unsigned char* image, std::uint64_t size) noexcept;

/// Live runtime A/B switch (safe to flip at any moment from the ImGui `Insert` overlay!).
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
