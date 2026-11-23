// SPDX-License-Identifier: GPL-2.0-or-later
// Shader Permutation & Pipeline Recipe Catalog (Layer 3 Zero-Stutter Pre-Bake).
// Stores 100% IP-free mathematical GPU state recipes (StageSpecialization + GraphicsPipelineKey)
// in a compact binary stream (`data/shader_recipes.bbrc`) and captures newly discovered runtime
// permutations/PSOs into `logs/recipes/unknown_recipes.bbrc` so they can be merged into future
// pre-bake catalogs.
#pragma once

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "common/types.h"
#include "render/pipeline.h"
#include "shader/info.h"
#include "shader/specialization.h"

namespace Vulkan {

class RecipeCatalog {
public:
    static constexpr u32 FormatVersion = 1u;

    enum class RecordKind : u8 {
        Stage = 1,
        GraphicsPso = 2,
        ComputePso = 3,
    };

    static constexpr u8 InfoFlagHasFetchShader = 1u << 0;
    static constexpr u8 InfoFlagHasBitwiseXor  = 1u << 1;
    static constexpr u8 InfoFlagUsesDma        = 1u << 2;

    struct StageRecipe {
        u64 pgm_hash{};
        u64 perm_hash{};
        u64 spec_digest{};
        u32 perm_idx{};
        u32 fetch_shader_sgpr_base{};
        Shader::HwStage hw_stage{};
        Shader::SwStage sw_stage{};
        u8 mrt_mask{};
        u8 info_flags{};
        Shader::StageSpecialization spec{};
        std::vector<u32> flattened_ud_buf;
    };

    struct GraphicsPsoRecipe {
        u64 key_hash{};
        GraphicsPipelineKey key{};
        GraphicsPipeline::SerializationSupport sdata{};
    };

    struct ComputePsoRecipe {
        u64 key_hash{};
        ComputePipelineKey key{};
    };

    struct Stats {
        size_t loaded_stages{};
        size_t loaded_gfx_psos{};
        size_t loaded_cs_psos{};
        size_t captured_stages{};
        size_t captured_gfx_psos{};
        size_t captured_cs_psos{};
        bool catalog_found{};
        bool capture_enabled{};
    };

    static RecipeCatalog& Get() noexcept;

    /// Loads the shipped recipe catalog (`data/shader_recipes.bbrc`) and indexes any previously
    /// captured session recipes (`logs/recipes/unknown_recipes.bbrc`) for deduplication.
    /// Safe to call multiple times (idempotent after first initialization).
    void Initialize(const std::filesystem::path& catalog_path = {},
                    const std::filesystem::path& capture_path = {});

    /// Flushes and closes the capture file and logs a 1-line summary.
    void Shutdown();

    /// Computes a stable 64-bit content digest for a shader stage specialization independent of
    /// its runtime `perm_idx` slot ordering.
    static u64 ComputeStageDigest(u64 pgm_hash, Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                                  const Shader::StageSpecialization& spec);

    /// OS/profile-independent canonical stage digest: ignores host-specific `BufferSpecialization::paged`
    /// (which differs between Linux `BB_GUEST_IN_PLACE=1` and Windows `BB_GUEST_IN_PLACE=0`) and
    /// clears unbound (`!bitset[b]`) resource slots so recipes match before/after
    /// `StageSpecialization::NormalizeUnboundDefaults`.
    static u64 ComputeCanonicalStageDigest(u64 pgm_hash, Shader::HwStage hw_stage,
                                           Shader::SwStage sw_stage,
                                           const Shader::StageSpecialization& spec);

    /// Appends a shader stage permutation to `logs/recipes/unknown_recipes.bbrc` if not yet known.
    /// Thread-safe, zero disk I/O when already known.
    void CaptureStageIfUnknown(const Shader::Info& info, const Shader::StageSpecialization& spec,
                               u64 perm_hash, size_t perm_idx);

    /// Appends a graphics PSO recipe to `logs/recipes/unknown_recipes.bbrc` if not yet known.
    void CaptureGraphicsPsoIfUnknown(const GraphicsPipelineKey& key, u64 key_hash,
                                     const GraphicsPipeline::SerializationSupport& sdata);

    /// Appends a compute PSO recipe to `logs/recipes/unknown_recipes.bbrc` if not yet known.
    void CaptureComputePsoIfUnknown(const ComputePipelineKey& key);

    /// Read-only snapshots of loaded catalog recipes (populated during Initialize()).
    const std::vector<StageRecipe>& GetStageRecipes() const noexcept {
        return stage_recipes_;
    }
    const std::vector<GraphicsPsoRecipe>& GetGraphicsPsoRecipes() const noexcept {
        return gfx_pso_recipes_;
    }
    const std::vector<ComputePsoRecipe>& GetComputePsoRecipes() const noexcept {
        return cs_pso_recipes_;
    }

    Stats GetStats() const;

    const std::filesystem::path& CatalogPath() const noexcept {
        return catalog_path_;
    }
    const std::filesystem::path& CapturePath() const noexcept {
        return capture_path_;
    }

    static std::filesystem::path ResolveDefaultCatalogPath();
    static std::filesystem::path ResolveDefaultCapturePath();

private:
    RecipeCatalog() = default;
    ~RecipeCatalog();

    RecipeCatalog(const RecipeCatalog&) = delete;
    RecipeCatalog& operator=(const RecipeCatalog&) = delete;

    bool LoadStreamFile(const std::filesystem::path& path, bool store_recipes);
    bool ParseStreamBytes(const u8* data, size_t size, bool store_recipes);
    bool EnsureCaptureStreamOpenLocked();
    void AppendRecordLocked(RecordKind kind, const std::vector<u8>& payload);

    mutable std::mutex mutex_;
    std::atomic<bool> initialized_{false};
    bool capture_enabled_{true};
    bool catalog_found_{false};

    std::filesystem::path catalog_path_;
    std::filesystem::path capture_path_;
    std::ofstream capture_stream_;

    std::vector<StageRecipe> stage_recipes_;
    std::vector<GraphicsPsoRecipe> gfx_pso_recipes_;
    std::vector<ComputePsoRecipe> cs_pso_recipes_;

    std::unordered_set<u64> known_stage_digests_;
    std::unordered_set<u64> known_gfx_pso_hashes_;
    std::unordered_set<u64> known_cs_pso_hashes_;

    size_t captured_stages_{0};
    size_t captured_gfx_psos_{0};
    size_t captured_cs_psos_{0};
};

} // namespace Vulkan
