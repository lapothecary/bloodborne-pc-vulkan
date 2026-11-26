// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// File: gpu/cloth_exp/cloth_exp_registry.cpp
//
// Non-destructive Zydis trampoline & W^X live-dispatch harness.
// Allows instant real-time A/B toggling & per-function range bisection from
// the `Insert` overlay (`Developer / Legacy A/B Debug`) without restarting.
// ============================================================================

#include "cloth_exp/cloth_exp_registry.h"

#include <Zydis/Zydis.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sys/mman.h>
#include <unistd.h>

#include "bbport_guest_code_hooks.h"
#include "cloth_exp/cloth_exp_constraints_and_col.h"
#include "cloth_exp/cloth_exp_runtime.h"
#include "cloth_exp/cloth_exp_sim_and_integrate.h"
#include "cloth_exp/cloth_exp_sprj_wind.h"
#include "cloth_exp/cloth_exp_world_and_ops.h"

#if defined(_WIN32)
extern "C" int bb_guest_tls_slot(void) noexcept;
#endif

namespace BbClothExp {

std::uintptr_t g_cloth_image_base = 0;
#if defined(_WIN32)
std::uint32_t g_cloth_win32_tls_teb_offset = 0x1480u;
#endif

std::uint64_t ReadDispatchCounterInternal() noexcept;
void ResetDispatchCounterInternal() noexcept;

namespace {

const ClothExpEntry kEntries[] = {
    // Module 01: World, Step, Operators & AABB (Batch 1 — 14 functions, 5,729 B)
    {0x0748c50ULL,  21u, reinterpret_cast<void*>(&ClothExp_hclSimClothInstance_getActiveConfig),     "hclSimClothInstance_getActiveConfig",     "01_WorldAndOps", true},
    {0x0748a00ULL, 161u, reinterpret_cast<void*>(&ClothExp_hclSimClothInstance_updateParticlesAabb), "hclSimClothInstance_updateParticlesAabb", "01_WorldAndOps", true},
    {0x0748ab0ULL, 410u, reinterpret_cast<void*>(&ClothExp_hclSimClothInstance_updateAabbs),         "hclSimClothInstance_updateAabbs",         "01_WorldAndOps", true},
    {0x1167a80ULL, 166u, reinterpret_cast<void*>(&ClothExp_hclConstraintStiffnessFactor),            "hclConstraintStiffnessFactor",            "01_WorldAndOps", true},
    {0x06fffb0ULL,  16u, reinterpret_cast<void*>(&ClothExp_hclMoveParticlesOperator_process),        "hclMoveParticlesOperator_process",        "01_WorldAndOps", false},
    {0x06fffc0ULL, 439u, reinterpret_cast<void*>(&ClothExp_hclMoveParticlesOperator_moveSomeSIMD),   "hclMoveParticlesOperator_moveSomeSIMD",   "01_WorldAndOps", true},
    {0x0700180ULL, 439u, reinterpret_cast<void*>(&ClothExp_hclMoveParticlesOperator_moveSomeScalar), "hclMoveParticlesOperator_moveSomeScalar", "01_WorldAndOps", true},
    {0x0769450ULL, 120u, reinterpret_cast<void*>(&ClothExp_hclWorld_step),                           "hclWorld_step",                           "01_WorldAndOps", true},
    {0x07694d0ULL, 659u, reinterpret_cast<void*>(&ClothExp_hclWorld_addClothJobs),                   "hclWorld_addClothJobs",                   "01_WorldAndOps", true},
    {0x0769190ULL, 701u, reinterpret_cast<void*>(&ClothExp_hclWorld_stepSingleThread),               "hclWorld_stepSingleThread",               "01_WorldAndOps", true},
    {0x076a960ULL, 1436u, reinterpret_cast<void*>(&ClothExp_hclOperator_execute),                    "hclOperator_execute",                     "01_WorldAndOps", true},
    {0x0fd5440ULL, 346u, reinterpret_cast<void*>(&ClothExp_FD4HavokCloth_step),                      "FD4HavokCloth_step",                      "01_WorldAndOps", true},
    {0x0fd5ab0ULL, 255u, reinterpret_cast<void*>(&ClothExp_FD4HavokCloth_queryWind),                 "FD4HavokCloth_queryWind",                 "01_WorldAndOps", true},
    {0x0fd8120ULL, 560u, reinterpret_cast<void*>(&ClothExp_FD4Wind_step),                            "FD4Wind_step",                            "01_WorldAndOps", true},

    // Module 02: Constraints & Collision Solvers (Batch 2 — 16 functions, 8,668 B)
    {0x0767cf0ULL,  541u, reinterpret_cast<void*>(&ClothExp_hclCollidable_setTransform),                   "hclCollidable_setTransform",                   "02_ConstraintsAndCol", true},
    {0x0771f60ULL,  537u, reinterpret_cast<void*>(&ClothExp_hclSimulateOperator_solveConstraints),         "hclSimulateOperator_solveConstraints",         "02_ConstraintsAndCol", true},
    {0x0772180ULL,  620u, reinterpret_cast<void*>(&ClothExp_hclSimulateOperator_solveConstraintsExec),     "hclSimulateOperator_solveConstraintsExec",     "02_ConstraintsAndCol", true},
    {0x0774260ULL, 1261u, reinterpret_cast<void*>(&ClothExp_hclSimulateOperator_collideParticles),         "hclSimulateOperator_collideParticles",         "02_ConstraintsAndCol", true},
    {0x0774750ULL,  968u, reinterpret_cast<void*>(&ClothExp_hclSimClothInstance_collideParticlesInternal), "hclSimClothInstance_collideParticlesInternal", "02_ConstraintsAndCol", true},
    {0x06fb220ULL, 1547u, reinterpret_cast<void*>(&ClothExp_hclLocalRangeConstraintSet_apply),             "hclLocalRangeConstraintSet_apply",             "02_ConstraintsAndCol", true},
    {0x071ac50ULL,  270u, reinterpret_cast<void*>(&ClothExp_hclStretchLinkConstraintSet_apply),            "hclStretchLinkConstraintSet_apply",            "02_ConstraintsAndCol", true},
    {0x0720dc0ULL,  224u, reinterpret_cast<void*>(&ClothExp_hclStretchLinkConstraintSetMx_apply),          "hclStretchLinkConstraintSetMx_apply",          "02_ConstraintsAndCol", true},
    {0x07422b0ULL,  329u, reinterpret_cast<void*>(&ClothExp_hclStandardLinkConstraintSet_apply),           "hclStandardLinkConstraintSet_apply",           "02_ConstraintsAndCol", true},
    {0x07420c0ULL,  224u, reinterpret_cast<void*>(&ClothExp_hclStandardLinkConstraintSetMx_apply),         "hclStandardLinkConstraintSetMx_apply",         "02_ConstraintsAndCol", true},
    {0x07386e0ULL,  471u, reinterpret_cast<void*>(&ClothExp_hclBendStiffnessConstraintSet_apply),          "hclBendStiffnessConstraintSet_apply",          "02_ConstraintsAndCol", true},
    {0x0741f20ULL,  284u, reinterpret_cast<void*>(&ClothExp_hclBendStiffnessConstraintSetMx_apply),        "hclBendStiffnessConstraintSetMx_apply",        "02_ConstraintsAndCol", true},
    {0x07464d0ULL,  415u, reinterpret_cast<void*>(&ClothExp_hclBendLinkConstraintSet_apply),               "hclBendLinkConstraintSet_apply",               "02_ConstraintsAndCol", true},
    {0x072b2d0ULL,  224u, reinterpret_cast<void*>(&ClothExp_hclBendLinkConstraintSetMx_apply),             "hclBendLinkConstraintSetMx_apply",             "02_ConstraintsAndCol", true},
    {0x07308c0ULL,  364u, reinterpret_cast<void*>(&ClothExp_hclCompressibleLinkConstraintSet_apply),       "hclCompressibleLinkConstraintSet_apply",       "02_ConstraintsAndCol", true},
    {0x076c7e0ULL,  389u, reinterpret_cast<void*>(&ClothExp_hclVolumeConstraint_apply),                    "hclVolumeConstraint_apply",                    "02_ConstraintsAndCol", true},

    // Module 03: Simulate, Motion Transfer, Integrate, Pinch & Normals (Batch 3 — 6 functions, 12,555 B)
    {0x076fb90ULL, 1717u, reinterpret_cast<void*>(&ClothExp_hclSimulateOperator_preStep),                  "hclSimulateOperator_preStep",                  "03_SimulateAndIntegrate", true},
    {0x076e370ULL, 3276u, reinterpret_cast<void*>(&ClothExp_hclSimulateOperator_simulate),                 "hclSimulateOperator_simulate",                 "03_SimulateAndIntegrate", true},
    {0x0771290ULL, 2054u, reinterpret_cast<void*>(&ClothExp_hclSimulateOperator_transferMotion),           "hclSimulateOperator_transferMotion",           "03_SimulateAndIntegrate", true},
    {0x0773df0ULL, 1130u, reinterpret_cast<void*>(&ClothExp_hclSimulateOperator_integrate),                "hclSimulateOperator_integrate",                "03_SimulateAndIntegrate", true},
    {0x076dcc0ULL, 1496u, reinterpret_cast<void*>(&ClothExp_hclSimulateOperator_pinchAndRelease),          "hclSimulateOperator_pinchAndRelease",          "03_SimulateAndIntegrate", true},
    {0x076f040ULL, 2882u, reinterpret_cast<void*>(&ClothExp_hclSimulateOperator_updateParticleNormals),    "hclSimulateOperator_updateParticleNormals",    "03_SimulateAndIntegrate", true},

    // Module 04: FromSoftware SprjClothWind Action, World Wind & Turbulence (Batch 4 — 4 functions, 10,196 B)
    {0x1051fa0ULL,  844u, reinterpret_cast<void*>(&ClothExp_SprjWindAction_apply),      "SprjWindAction_apply",      "04_SprjWind", true},
    {0x10522f0ULL, 2037u, reinterpret_cast<void*>(&ClothExp_SprjWindEvalWorldDir),      "SprjWindEvalWorldDir",      "04_SprjWind", true},
    {0x1052af0ULL, 4244u, reinterpret_cast<void*>(&ClothExp_SprjPointWind_apply),       "SprjPointWind_apply",       "04_SprjWind", true},
    {0x1053b90ULL, 3071u, reinterpret_cast<void*>(&ClothExp_SprjDirWind_apply),         "SprjDirWind_apply",         "04_SprjWind", true},
};

std::size_t g_prestep_entry_idx = ~std::size_t(0);

constexpr std::size_t kEntryCount = sizeof(kEntries) / sizeof(kEntries[0]);
static_assert(kEntryCount <= 256u, "Live flag page holds up to 256 1-byte atomic flags");

std::mutex g_state_mutex;
volatile u8* g_live_flag_page = nullptr; // 4 KiB RW page adjacent to RX trampoline pages
bool g_entry_installed[kEntryCount]{};
std::atomic<bool> g_master_enabled{false};
std::atomic<int> g_range_min{0};
std::atomic<int> g_range_max{static_cast<int>(kEntryCount) - 1};
std::atomic<std::size_t> g_installed_count{0};

bool MatchesModuleFilter(const char* mod_name, const char* env_mods) noexcept {
    if (!env_mods || !env_mods[0] || std::strcmp(env_mods, "all") == 0) {
        return true;
    }
    const char* p = env_mods;
    while (*p) {
        while (*p == ',' || *p == ' ') ++p;
        const char* start = p;
        while (*p && *p != ',' && *p != ' ') ++p;
        const std::size_t len = static_cast<std::size_t>(p - start);
        if (len > 0 && std::strncmp(mod_name, start, len) == 0) {
            return true;
        }
    }
    return false;
}

void RefreshLiveFlagsLocked(const char* env_mods) noexcept {
    if (!g_live_flag_page) {
        return;
    }
    const bool master = g_master_enabled.load(std::memory_order_relaxed);
    const int lo = g_range_min.load(std::memory_order_relaxed);
    const int hi = g_range_max.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < kEntryCount; ++i) {
        const bool in_range = (static_cast<int>(i) >= lo && static_cast<int>(i) <= hi);
        const bool mod_ok = env_mods ? MatchesModuleFilter(kEntries[i].module_name, env_mods) : true;
        const u8 val = (master && g_entry_installed[i] && in_range && mod_ok) ? 1u : 0u;
        __atomic_store_n(&g_live_flag_page[i], val, __ATOMIC_RELEASE);
    }
}

void EmitJmpAbs(u8*& dst, std::uintptr_t target) noexcept {
    dst[0] = 0xff;
    dst[1] = 0x25;
    dst[2] = 0x00;
    dst[3] = 0x00;
    dst[4] = 0x00;
    dst[5] = 0x00;
    const std::uint64_t t64 = static_cast<std::uint64_t>(target);
    std::memcpy(dst + 6, &t64, 8);
    dst += 14;
}

void* MapNearClothArena(std::uintptr_t code_base_addr, std::size_t total_bytes) noexcept {
    constexpr std::int64_t kJumpReach = 0x7000'0000LL;
    const std::size_t granule =
        std::max<std::size_t>(static_cast<std::size_t>(::sysconf(_SC_PAGESIZE)), 65536u);
    for (std::int64_t step = 1; step * (64LL << 20) < kJumpReach; ++step) {
        for (const std::int64_t sign : {-1LL, 1LL}) {
            const std::int64_t at = static_cast<std::int64_t>(code_base_addr) + sign * step * (64LL << 20);
            if (at <= 0) {
                continue;
            }
            auto* hint = reinterpret_cast<void*>(static_cast<std::uintptr_t>(at) & ~(granule - 1ULL));
            void* p = ::mmap(hint, total_bytes, PROT_READ | PROT_WRITE | PROT_EXEC,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
            if (p != MAP_FAILED && p != nullptr) {
                return p;
            }
        }
    }
    return nullptr;
}

} // namespace

void PatchImage(unsigned char* image, std::uint64_t size) noexcept {
    if (!image || size < 0x1200000ULL) {
        return;
    }
    g_cloth_image_base = reinterpret_cast<std::uintptr_t>(image);
#if defined(_WIN32)
    g_cloth_win32_tls_teb_offset =
        static_cast<std::uint32_t>(0x1480 + 8 * bb_guest_tls_slot());
#endif

    const std::size_t page_sz = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const std::size_t granule = std::max<std::size_t>(page_sz, 65536u);
    const std::size_t flag_bytes = granule;
    const std::size_t code_bytes = granule; // 64 KiB RX trampoline & stub arena
    const std::size_t total_bytes = flag_bytes + code_bytes;

    void* arena = MapNearClothArena(reinterpret_cast<std::uintptr_t>(image), total_bytes);
    if (!arena) {
        std::printf("[ClothExp] Failed to map near trampoline arena (%zu bytes)\n", total_bytes);
        return;
    }

    std::memset(arena, 0, flag_bytes);
    std::memset(static_cast<u8*>(arena) + flag_bytes, 0xcc, code_bytes);
    g_live_flag_page = static_cast<volatile u8*>(arena);
    u8* code_base = static_cast<u8*>(arena) + flag_bytes;
    u8* code_cur = code_base;
    u8* const code_end = code_base + code_bytes;

    ZydisDecoder decoder{};
    ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);

    std::size_t hooked = 0;
    for (std::size_t i = 0; i < kEntryCount; ++i) {
        const ClothExpEntry& e = kEntries[i];
        if (e.vma == 0x076fb90ULL) {
            g_prestep_entry_idx = i;
        }
        if (!e.detour_guest_entry) {
            // Pure C++20 intra-module dispatcher (e.g. 0x06fffb0 called by hclOperator_execute)
            g_entry_installed[i] = true;
            ++hooked;
            continue;
        }
        if (e.vma + 16u > size || code_cur + 128u > code_end) {
            continue;
        }

        u8* guest_fn = image + e.vma;
        if (e.vma == 0x076fb90ULL && guest_fn[0] == 0xe9) {
            // `WrappedClothPreStep` is already installed at `0x076fb90` and delegates via
            // `TryDispatchPreStep()` so `FIXED30` 1/30 s step substitution stays synchronized.
            g_entry_installed[i] = true;
            ++hooked;
            continue;
        }
        u8* orig_tramp = code_cur;
        std::size_t stolen = 0;
        bool ok = true;
        bool ended_with_jmp = false;

        while (stolen < 5u) {
            ZydisDecodedInstruction insn{};
            ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT_VISIBLE]{};
            if (!ZYAN_SUCCESS(ZydisDecoderDecodeFull(&decoder, guest_fn + stolen,
                                                     32u - std::min<std::size_t>(stolen, 16u),
                                                     &insn, ops))) {
                ok = false;
                break;
            }
            if (insn.mnemonic == ZYDIS_MNEMONIC_JMP && insn.length == 5u &&
                guest_fn[stolen] == 0xe9) {
                // Existing 5-byte relative detour (e.g. from BbFrameRate): chain cleanly to its target!
                std::int32_t rel32 = 0;
                std::memcpy(&rel32, guest_fn + stolen + 1u, 4);
                const std::uintptr_t target =
                    reinterpret_cast<std::uintptr_t>(guest_fn + stolen + 5u) +
                    static_cast<std::intptr_t>(rel32);
                EmitJmpAbs(code_cur, target);
                stolen += 5u;
                ended_with_jmp = true;
                break;
            }
            if ((insn.attributes & ZYDIS_ATTRIB_IS_RELATIVE) != 0) {
                // Support `mov/add reg, [rip + disp32]` if ever encountered in future batches
                ok = false;
                break;
            }
            std::memcpy(code_cur, guest_fn + stolen, insn.length);
            code_cur += insn.length;
            stolen += insn.length;
        }

        if (!ok || stolen < 5u) {
            code_cur = orig_tramp;
            std::printf("[ClothExp] Skipped 0x%07llx (%s): non-relocatable prologue\n",
                        static_cast<unsigned long long>(e.vma), e.name);
            continue;
        }
        if (!ended_with_jmp) {
            EmitJmpAbs(code_cur, reinterpret_cast<std::uintptr_t>(guest_fn + stolen));
        }

        // Align dispatch stub to 16 bytes
        code_cur = reinterpret_cast<u8*>((reinterpret_cast<std::uintptr_t>(code_cur) + 15ULL) &
                                         ~std::uintptr_t(15ULL));
        u8* dispatch_stub = code_cur;

        // 1. `cmp byte ptr [rip + disp32], 0` -> `80 3d <disp32> 00` (7 bytes, flags dead on entry)
        const std::int64_t flag_disp =
            reinterpret_cast<std::int64_t>(const_cast<u8*>(&g_live_flag_page[i])) -
            reinterpret_cast<std::int64_t>(code_cur + 7);
        if (flag_disp < -(1LL << 31) || flag_disp >= (1LL << 31)) {
            continue;
        }
        const auto flag_disp32 = static_cast<std::int32_t>(flag_disp);
        code_cur[0] = 0x80;
        code_cur[1] = 0x3d;
        std::memcpy(code_cur + 2, &flag_disp32, 4);
        code_cur[6] = 0x00;
        code_cur += 7;

        // 2. `je orig_tramp` -> `0f 84 <rel32>` (6 bytes)
        const std::int64_t je_rel =
            reinterpret_cast<std::int64_t>(orig_tramp) - reinterpret_cast<std::int64_t>(code_cur + 6);
        const auto je_rel32 = static_cast<std::int32_t>(je_rel);
        code_cur[0] = 0x0f;
        code_cur[1] = 0x84;
        std::memcpy(code_cur + 2, &je_rel32, 4);
        code_cur += 6;

        // 3. `jmp [rip + 0]` -> C++20 function (14 bytes)
        EmitJmpAbs(code_cur, reinterpret_cast<std::uintptr_t>(e.func_ptr));
        code_cur = reinterpret_cast<u8*>((reinterpret_cast<std::uintptr_t>(code_cur) + 15ULL) &
                                         ~std::uintptr_t(15ULL));

        // Patch guest function prologue -> dispatch_stub
        const std::int64_t jmp_rel =
            reinterpret_cast<std::int64_t>(dispatch_stub) -
            reinterpret_cast<std::int64_t>(guest_fn + 5);
        if (jmp_rel < -(1LL << 31) || jmp_rel >= (1LL << 31)) {
            continue;
        }

        const std::uintptr_t p_begin =
            reinterpret_cast<std::uintptr_t>(guest_fn) & ~(page_sz - 1ULL);
        const std::uintptr_t p_end =
            (reinterpret_cast<std::uintptr_t>(guest_fn) + stolen + page_sz - 1ULL) &
            ~(page_sz - 1ULL);
        if (::mprotect(reinterpret_cast<void*>(p_begin), p_end - p_begin,
                       PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
            continue;
        }
        const auto jmp_rel32 = static_cast<std::int32_t>(jmp_rel);
        guest_fn[0] = 0xe9;
        std::memcpy(guest_fn + 1, &jmp_rel32, 4);
        for (std::size_t pad = 5u; pad < stolen; ++pad) {
            guest_fn[pad] = 0x90;
        }
        ::mprotect(reinterpret_cast<void*>(p_begin), p_end - p_begin, Core::GuestCodeProtection());

        g_entry_installed[i] = true;
        ++hooked;
    }

    // Seal W^X: Page 0 stays RW (flags), Pages 1..8 become RX (code).
    ::mprotect(arena, flag_bytes, PROT_READ | PROT_WRITE);
    ::mprotect(code_base, code_bytes, PROT_READ | PROT_EXEC);
    g_installed_count.store(hooked, std::memory_order_release);

    const char* env_exp = std::getenv("BB_CLOTH_EXP");
    const char* env_min = std::getenv("BB_CLOTH_EXP_MIN");
    const char* env_max = std::getenv("BB_CLOTH_EXP_MAX");
    const char* env_mods = std::getenv("BB_CLOTH_EXP_MODULES");

    int min_idx = 0;
    int max_idx = static_cast<int>(kEntryCount) - 1;
    if (env_min && *env_min) {
        min_idx = std::clamp(std::atoi(env_min), 0, static_cast<int>(kEntryCount) - 1);
    }
    if (env_max && *env_max) {
        max_idx = std::clamp(std::atoi(env_max), min_idx, static_cast<int>(kEntryCount) - 1);
    }
    g_range_min.store(min_idx, std::memory_order_relaxed);
    g_range_max.store(max_idx, std::memory_order_relaxed);

    const bool start_on = (env_exp && env_exp[0] == '1' && env_exp[1] == '\0');
    g_master_enabled.store(start_on, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(g_state_mutex);
        RefreshLiveFlagsLocked(env_mods);
    }

    std::printf("[ClothExp] Experimental C++20 Cloth Engine ready: %zu / %zu functions hooked "
                "(state=%s, range=[%d..%d], toggle live in Insert -> Developer Debug)\n",
                hooked, kEntryCount, start_on ? "ON" : "OFF (default)", min_idx, max_idx);
}

void SetEnabled(bool enabled) noexcept {
    std::lock_guard<std::mutex> lk(g_state_mutex);
    g_master_enabled.store(enabled, std::memory_order_relaxed);
    RefreshLiveFlagsLocked(std::getenv("BB_CLOTH_EXP_MODULES"));
}

bool Enabled() noexcept {
    return g_master_enabled.load(std::memory_order_relaxed);
}

void SetActiveRange(int min_idx, int max_idx) noexcept {
    const int last = std::max(0, static_cast<int>(kEntryCount) - 1);
    const int lo = std::clamp(min_idx, 0, last);
    const int hi = std::clamp(max_idx, lo, last);
    std::lock_guard<std::mutex> lk(g_state_mutex);
    g_range_min.store(lo, std::memory_order_relaxed);
    g_range_max.store(hi, std::memory_order_relaxed);
    RefreshLiveFlagsLocked(std::getenv("BB_CLOTH_EXP_MODULES"));
}

void GetActiveRange(int& out_min_idx, int& out_max_idx) noexcept {
    out_min_idx = g_range_min.load(std::memory_order_relaxed);
    out_max_idx = g_range_max.load(std::memory_order_relaxed);
}

std::size_t EntryCount() noexcept {
    return kEntryCount;
}

std::size_t InstalledHookCount() noexcept {
    return g_installed_count.load(std::memory_order_acquire);
}

const ClothExpEntry* GetEntries() noexcept {
    return kEntries;
}

bool IsEntryActive(std::size_t idx) noexcept {
    if (!g_live_flag_page || idx >= kEntryCount) {
        return false;
    }
    return __atomic_load_n(&g_live_flag_page[idx], __ATOMIC_ACQUIRE) != 0u;
}

bool TryDispatchPreStep(unsigned char* op, unsigned char* step_info) noexcept {
    if (!IsEntryActive(g_prestep_entry_idx)) {
        return false;
    }
    ClothExp_hclSimulateOperator_preStep(op, step_info);
    return true;
}

std::uint64_t TotalDispatchCount() noexcept {
    return ReadDispatchCounterInternal();
}

void ResetDispatchCount() noexcept {
    ResetDispatchCounterInternal();
}

} // namespace BbClothExp
