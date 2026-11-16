// SPDX-License-Identifier: GPL-2.0-or-later
// Block 5, Step 5.1: In-process SPIRV-Tools optimization pass for translated GCN shaders.
#pragma once

#include <vector>

#include "common/types.h"

namespace Shader::Backend::SPIRV {

struct SpirvOptStats {
    u32 modules_optimized = 0;
    u32 modules_failed = 0;
    u64 words_before = 0;
    u64 words_after = 0;
    u64 total_ns = 0;
    u64 nocontract_stripped = 0;
    u64 legacy_mul_folded = 0;
    u32 inf_nan_preserve_stripped = 0;
    u32 hw_interp_modules = 0;
    u64 hw_interp_varyings = 0;
    u64 hw_interp_channels = 0;
    u64 hw_interp_int_nudges = 0;
    u64 self_minmax_folded = 0;
    u64 safe_rsq_muls_folded = 0;
    u64 safe_rsq_guards_emitted = 0;
    u64 vec4_out_stores_coalesced = 0;
    u32 native_replaced = 0;
};

/// Controlled by BB_SPIRV_OPT (default 1 = enabled, 0 = raw EmitSPIRV output).
[[nodiscard]] bool SpirvOptEnabled() noexcept;

/// Block 5, Step 5.2: Controlled by BB_SHADER_FAST_FP (default 1 = unlock hardware FFMA
/// contraction, fold exact V_MUL_LEGACY_F32 squares/constants, and strip SignedZeroInfNanPreserve
/// on NaN-safe modules; 0 = strict GCN float controls).
[[nodiscard]] bool FastFpEnabled() noexcept;

/// Block 5, Step 5.3: Controlled by BB_HW_INTERP (default 1 = replace software PerVertexKHR /
/// FragmentBarycentricKHR manual barycentric FSUB+FFMA chains with hardware rasterizer IPA
/// interpolation + Flat provoking-vertex loads + 1+2^-16 integer-varying truncation nudge;
/// 0 = software FragmentBarycentricKHR).
[[nodiscard]] bool HwInterpEnabled() noexcept;

/// Ensures any Vertex Shader SPIR-V module declaring `Output vec4 cldist_attr0` writes all 4
/// components (inserting `OpStore %cldist_attr0 vec4(1.0)` on entry when only `.xy` were stored),
/// healing pre-fix cached shaders in `< 2 us` so Fragment `cldist_attr0 < 0` never reads garbage.
bool EnsureClipDistanceInitializedInSpv(std::vector<u32>& spv);

/// Runs SPIRV-Tools RegisterPerformancePasses() (-O) targeting Vulkan 1.3.
/// When `sync_fast_path` is true (synchronous Stage A fallback on large shaders >= 1800 words),
/// runs the O(N) HwInterp / NativeShader / FastFp passes in <0.1 ms and defers the heavy O(N^2)
/// SPIRV-Tools SSA passes to a background worker (`*out_deferred_heavy = true`).
bool OptimizeSPIRV(std::vector<u32>& spv, u64 pgm_hash = 0, bool sync_fast_path = false,
                   bool* out_deferred_heavy = nullptr);

/// Atomically snapshots and resets accumulated optimization counters.
[[nodiscard]] SpirvOptStats GetAndResetSpirvOptStats() noexcept;

} // namespace Shader::Backend::SPIRV
