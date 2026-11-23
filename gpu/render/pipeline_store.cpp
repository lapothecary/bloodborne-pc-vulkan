// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: saving to and preloading from the pipeline cache on disk (pipeline_store.h).

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <unordered_set>
#include <vector>

#include "common/hash.h"
#include "common/serdes.h"
#include "core/emulator_settings.h"
#include "render/blob_store.h"
#include "render/direct_shaders.h"
#include "render/pipeline_cache.h"
#include "render/pipeline_store.h"
#include "render/recipe_catalog.h"
#include "shader/backend/spirv/native_shaders.h"
#include "shader/backend/spirv/spirv_opt.h"
#include "shader/info.h"
#include "shader/specialization.h"
#include "vulkan/instance.h"

namespace Vulkan {

namespace {

BbCache::BlobStore& Store() {
    return BbCache::BlobStore::Instance();
}

std::string MetaName(u64 perm_hash) {
    return fmt::format("{:#018x}", perm_hash);
}

std::string BinaryName(u64 pgm_hash, size_t perm_idx) {
    if (Shader::Backend::SPIRV::IsSpvPatchedLightAccPsHash(pgm_hash)) {
        return fmt::format("ltb2_{:#018x}_{}", pgm_hash, perm_idx);
    }
    return fmt::format("{:#018x}_{}", pgm_hash, perm_idx);
}

bool IsOpt33Phase2ShaderHash(u64 pgm_hash) noexcept {
    switch (pgm_hash) {
    // Phase 2.1a: Fullscreen Deferred PBR, Deferred AO & Light Composite
    case 0x00000000d92266b9ULL:
    case 0x0000000031b90cc5ULL:
    case 0x000000003fe9e964ULL:
    case 0x0000000032e47788ULL:
    // Phase 2.2: Screen-Space Shadow Bilateral Blur
    case 0x0000000011ffdfb5ULL:
    // Phase 2.1b: VSM Point-Light (1..4) & Spot-Light (1..4) Deferred PBR
    case 0x00000000e96447c3ULL:
    case 0x00000000a22c7f71ULL:
    case 0x000000006b42fc89ULL:
    case 0x000000000ef50047ULL:
    case 0x00000000bae5454eULL:
    case 0x00000000896e7242ULL:
    case 0x0000000002cd8f9bULL:
    case 0x000000008ea5be99ULL:
    // Phase 2.1b: Point-Light Blinn-H (1..12) & Phong-R (1..12) Deferred PBR
    case 0x00000000370174f7ULL:
    case 0x00000000600f86b0ULL:
    case 0x0000000089f9ee3eULL:
    case 0x00000000149432ffULL:
    case 0x000000005de04266ULL:
    case 0x000000005d7150daULL:
    case 0x000000008d5387d9ULL:
    case 0x00000000f81596cdULL:
    case 0x000000000b2ace4cULL:
    case 0x00000000201cb43dULL:
    case 0x000000005f76eb3dULL:
    case 0x00000000799bfa32ULL:
    case 0x000000007cc2c044ULL:
    case 0x00000000186d69d5ULL:
    case 0x00000000d22cbfd1ULL:
    case 0x000000003105494cULL:
    case 0x00000000d75fe2ffULL:
    case 0x0000000007d07cf3ULL:
    case 0x00000000f2531727ULL:
    case 0x00000000c6edb19aULL:
    case 0x00000000b59b4c4eULL:
    case 0x00000000505ebdbeULL:
    case 0x0000000022b357f3ULL:
    case 0x000000009f9ef267ULL:
    // Phase 2.3: Yebis 2 Post-FX (UAV Image Clear, DoF/Glare Downsample & Bokeh, Tonemap, Gamma)
    case 0x000000009a9cf8a9ULL:
    case 0x00000000265a6d2dULL:
    case 0x00000000a2c4f30aULL:
    case 0x00000000ff902740ULL:
    case 0x00000000e80526ccULL:
    case 0x0000000038d65b32ULL:
    case 0x000000005e67d866ULL:
    case 0x000000005c02eaf8ULL:
    // Phase 2.4: G-Buffer 6-MRT (Skinned 4-bone VS, 6 Alpha-Cutout PS, 9 Opaque PS)
    case 0x000000003deaa573ULL:
    case 0x000000001c3953c6ULL:
    case 0x0000000070bb42afULL:
    case 0x00000000998b035eULL:
    case 0x000000002140da2bULL:
    case 0x00000000a89e01e9ULL:
    case 0x00000000dbeccd9cULL:
    case 0x00000000bb10e440ULL:
    case 0x00000000e21e283aULL:
    case 0x000000008aba0ecdULL:
    case 0x00000000ca7226ecULL:
    case 0x00000000e54449c7ULL:
    case 0x0000000050287a64ULL:
    case 0x0000000052738b8aULL:
    case 0x00000000dbb222ebULL:
    case 0x000000009e3394c8ULL:
        return true;
    default:
        return false;
    }
}

bool IsOpt35Phase5ShaderHash(u64 pgm_hash) noexcept {
    switch (pgm_hash) {
    // Phase 5.3: G-Buffer 2b Foliage/Cutout native VS + uniform detail/probe early-out PS
    case 0x00000000ddbe557eULL:
    case 0x00000000803b9d24ULL:
    case 0x00000000b10c4f55ULL:
    case 0x00000000e859b6d9ULL:
    case 0x00000000e21e283aULL:
    case 0x0000000070bb42afULL:
    case 0x0000000052738b8aULL:
    // Step 2.3 (Sub-batch 1): Bloom/Glare, MotionBlur tile dilate, SSR Hi-Z, ConstFill, Scaleform BGRA
    case 0x0000000057f3b8c2ULL:
    case 0x000000004eb5a42fULL:
    case 0x00000000abf92450ULL:
    case 0x0000000001ae5388ULL:
    case 0x000000009eb30215ULL:
    case 0x000000004baeaa00ULL:
        return true;
    default:
        return false;
    }
}

bool IsOpt36BgraFixShaderHash(u64 pgm_hash) noexcept {
    switch (pgm_hash) {
    // Mixed RGBA/BGRA specialization fix (kBgraExport constant_id = 0)
    case 0x0000000057f3b8c2ULL: // ps_57f3b8c2_bloom_copy
    case 0x00000000f7229091ULL: // ps_f7229091_bgra_blit
    case 0x000000008c08444cULL: // ps_8c08444c_blur13
    case 0x00000000ed4c7668ULL: // ps_ed4c7668_lut3d
        return true;
    default:
        return false;
    }
}

bool IsOpt38StageC2ShaderHash(u64 pgm_hash) noexcept {
    switch (pgm_hash) {
    // Stage C2: GXSssBlurX/Y loc1 fix + 6-MRT G-Buffer cutout 0f54e5f9 + MotBlur combine d3c8bb21
    case 0x000000002eacb983ULL:
    case 0x000000008c13b984ULL:
    case 0x000000000f54e5f9ULL:
    case 0x00000000d3c8bb21ULL:
        return true;
    default:
        return false;
    }
}

bool IsOpt46MotBlurObjFpsShaderHash(u64 pgm_hash) noexcept {
    switch (pgm_hash) {
    // Remaster-v2 Stage 1.3: FPS-normalized 8-bit Object-Velocity encoding + Native PostFX exact scale
    case 0x0000000034bc187cULL:
    case 0x00000000749e4f9eULL:
    case 0x00000000b25e4faeULL:
    case 0x00000000d3c8bb21ULL:
    case 0x00000000e0305cefULL:
        return true;
    default:
        return false;
    }
}

bool IsOpt40RemasterSsgiShaderHash(u64 pgm_hash) noexcept {
    switch (pgm_hash) {
    // Remaster-v2 Performance Audit (opt54): Half-Res SSGI moved out of ps_32e47788 + PCSS umbra early-out in ps_f55643de
    case 0x0000000032e47788ULL:
    case 0x00000000f55643deULL:
        return true;
    default:
        return false;
    }
}

bool IsOpt39RemasterPcssShaderHash(u64 pgm_hash) noexcept {
    switch (pgm_hash) {
    // Remaster-v2 Stage 1.2: PCSS contact-hardening soft shadows (4K CSM + 5 cubemap VSM point-light shaders)
    case 0x00000000f55643deULL:
    case 0x000000006b42fc89ULL:
    case 0x000000000ef50047ULL:
    case 0x00000000a22c7f71ULL:
    case 0x00000000e96447c3ULL:
    case 0x00000000f40da260ULL:
        return true;
    default:
        return false;
    }
}

bool IsOpt37StageC1ShaderHash(u64 pgm_hash) noexcept {
    switch (pgm_hash) {
    // Stage C1: 4K shadow depth, motion blur alpha velocity
    case 0x000000007cde8e5fULL:
    case 0x00000000b25e4faeULL:
        return true;
    default:
        return false;
    }
}

bool IsOpt49RemasterFireShaderHash(u64 pgm_hash) noexcept {
    switch (pgm_hash) {
    // Stage 2.0 Remaster: Native GXFfx Fire & Embers suite (1-MRT + 2-MRT half-res soft particles)
    case 0x00000000832cdf8dULL:
    case 0x000000002e86d610ULL:
    case 0x0000000071c96598ULL:
    case 0x000000005e224cb6ULL:
    case 0x00000000d729c469ULL:
    case 0x0000000020739372ULL:
        return true;
    default:
        return false;
    }
}

bool IsOpt56Pack5ShaderHash(u64 pgm_hash) noexcept {
    switch (pgm_hash) {
    // Pack #5 (v1.2.3): Native Vulkan 1.3 GLSL 460 for top 5 recurring PS4-SPIRV hotspots
    case 0x00000000a61e325dULL: // Sample13WeightforGlowKernel
    case 0x00000000133eb139ULL: // Deferred Decal_DifSpc 5-MRT
    case 0x000000008a6c7657ULL: // Forward PBR + 4-tap CSM + cutout
    case 0x000000006720ae0dULL: // Forward Lightmap + wrap/SSS + 4-tap CSM + cutout
    case 0x00000000c31bd920ULL: // Forward Lightmap + IBL/SSR + wrap/SSS + 4-tap CSM + cutout
    // Phase 2 (v1.2.4): Native Vulkan 1.3 GLSL 460 for SSAO Gen/Reproj & Bilateral Upscale
    case 0x0000000083da8467ULL: // ScreenSpaceAO2_WithNormalReProj
    case 0x00000000c45c51cbULL: // ScreenSpaceAO2_BilateralUpScale
        return true;
    default:
        return false;
    }
}

std::string OptBinaryName(u64 pgm_hash, size_t perm_idx) {
    const bool hw_interp = Shader::Backend::SPIRV::HwInterpEnabled();
    const bool fast_fp = Shader::Backend::SPIRV::FastFpEnabled();
    const bool native_sh = Shader::Backend::SPIRV::NativeShadersEnabled();
    // Only shaders that actually have a native GLSL replacement differ between
    // BB_NATIVE_SHADERS=1 and BB_NATIVE_SHADERS=0. Reusing the standard opt31..39
    // cache for the other ~6,828 non-native shaders avoids re-optimizing the entire
    // 6,969-shader cache at startup whenever BB_NATIVE_SHADERS=0 is toggled.
    const bool use_native_prefix = native_sh || !Shader::Backend::SPIRV::HasNativeShader(pgm_hash);
    const bool ltb_patched = Shader::Backend::SPIRV::IsSpvPatchedLightAccPsHash(pgm_hash);
    const char* prefix = "opt1";
    if (hw_interp) {
        if (fast_fp) {
            prefix = use_native_prefix
                         ? (IsOpt56Pack5ShaderHash(pgm_hash)
                                ? "opt57"
                                : (IsOpt49RemasterFireShaderHash(pgm_hash)
                                ? "opt55"
                                : (IsOpt46MotBlurObjFpsShaderHash(pgm_hash)
                                       ? "opt48"
                                       : (IsOpt40RemasterSsgiShaderHash(pgm_hash)
                                       ? "opt54"
                                       : (IsOpt39RemasterPcssShaderHash(pgm_hash)
                                              ? "opt39"
                                              : (IsOpt38StageC2ShaderHash(pgm_hash)
                                                     ? "opt38"
                                                     : (IsOpt37StageC1ShaderHash(pgm_hash)
                                                            ? "opt37"
                                                            : (IsOpt36BgraFixShaderHash(pgm_hash)
                                                                   ? "opt36"
                                                                   : (IsOpt35Phase5ShaderHash(pgm_hash)
                                                                          ? "opt35"
                                                                          : (IsOpt33Phase2ShaderHash(pgm_hash)
                                                                                 ? "opt33"
                                                                                 : ((pgm_hash ==
                                                                                         0x0000000024042a9bULL ||
                                                                                     pgm_hash ==
                                                                                         0x00000000f55643deULL ||
                                                                                     ltb_patched)
                                                                                        ? "opt32"
                                                                                        : "opt31")))))))))))
                         : (ltb_patched ? "opt6_ltb2" : "opt6_nonat");
        } else {
            prefix = ltb_patched ? "opt3_ltb2" : "opt3_strict";
        }
    } else if (fast_fp) {
        prefix = ltb_patched ? "opt2_ltb2" : "opt2_r2";
    } else if (ltb_patched) {
        prefix = "opt1_ltb2";
    }
    return fmt::format("{}_{:#018x}_{}", prefix, pgm_hash, perm_idx);
}

void OptimizeSpirvInWarmUp(std::vector<u32>& spv, u64 pgm_hash) {
#ifndef _WIN32
    // During live gameplay Linux sets mallopt(M_ARENA_MAX, 1) so guest-visible heap pointers
    // stay below 1 TiB. Running SPIRV-Tools concurrently across 12 WarmUp threads under a
    // single glibc arena causes >100x mutex thrashing; serializing only the SPIRV-Tools call
    // makes cache-miss upgrades ~5x faster while file I/O and vkCreateShaderModule stay parallel.
    static std::mutex s_linux_opt_mutex;
    std::scoped_lock lk{s_linux_opt_mutex};
#endif
    Shader::Backend::SPIRV::OptimizeSPIRV(spv, pgm_hash);
}

const char* DriverCacheBlobName() noexcept {
    if (!Shader::Backend::SPIRV::SpirvOptEnabled()) {
        return "vk_driver_cache";
    }
    const bool hw_interp = Shader::Backend::SPIRV::HwInterpEnabled();
    const bool fast_fp = Shader::Backend::SPIRV::FastFpEnabled();
    const bool native_sh = Shader::Backend::SPIRV::NativeShadersEnabled();
    if (hw_interp) {
        if (fast_fp) {
            return native_sh ? "vk_driver_cache_opt35" : "vk_driver_cache_opt6_nonat";
        }
        return "vk_driver_cache_opt3_strict";
    }
    return fast_fp ? "vk_driver_cache_opt2_r2" : "vk_driver_cache_opt1";
}

/// Pipeline entries: the version, graphics (0) or compute (1), the key, the support data.
template <typename Key, typename Support>
void SavePipeline(const Key& key, const Support& sdata, u32 is_compute, const std::string& name) {
    if (!Store().IsOpened()) {
        return;
    }
    Serialization::Archive ar{};
    Serialization::Writer writer{ar};
    writer.Write(PipelineStore::PipelineKeyVersion);
    writer.Write(is_compute);
    key.Serialize(ar);
    sdata.Serialize(ar);
    Store().Save(BbCache::BlobType::PipelineKey, name, ar.TakeOff());
}

/// A permutation's metadata. False when it is from another build, or a motion vertex shader:
/// those embed session-local buffer device addresses and are translated again.
bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    Shader::StageSpecialization& spec, size_t& perm_idx,
                    const Shader::Profile& profile) {
    Serialization::Reader reader{ar};
    u32 meta_version{}, binary_version{};
    reader.Read(meta_version);
    if (meta_version != PipelineStore::ShaderMetaVersion) {
        return false;
    }
    reader.Read(binary_version);
    if (binary_version != PipelineStore::ShaderBinaryVersion) {
        return false;
    }
    u64 perm_hash{};
    reader.Read(perm_hash);
    reader.Read(perm_idx);
    spec.Deserialize(ar);
    info.Deserialize(ar);
    spec.info = &info;
    spec.NormalizeUnboundDefaults(profile);
    return !(info.hw_stage == Shader::HwStage::Vertex && spec.runtime_info.hw.vs.motion_vectors);
}

} // namespace

namespace PipelineStore {

void SaveCompute(const ComputePipelineKey& key, const ComputePipeline::SerializationSupport& sdata,
                 bool capture_recipe) {
    SavePipeline(key, sdata, 1, fmt::format("c_{:#018x}", key.value));
    if (capture_recipe) {
        RecipeCatalog::Get().CaptureComputePsoIfUnknown(key);
    }
}

void SaveGraphics(const GraphicsPipelineKey& key, u64 hash,
                  const GraphicsPipeline::SerializationSupport& sdata, bool capture_recipe) {
    SavePipeline(key, sdata, 0, fmt::format("g_{:#018x}", hash));
    if (capture_recipe) {
        RecipeCatalog::Get().CaptureGraphicsPsoIfUnknown(key, hash, sdata);
    }
}

void SaveShaderMeta(const Shader::Info& info, const Shader::StageSpecialization& spec,
                    u64 perm_hash, size_t perm_idx, bool capture_recipe) {
    if (!Store().IsOpened()) {
        return;
    }
    Serialization::Archive ar;
    Serialization::Writer writer{ar};
    writer.Write(ShaderMetaVersion);
    writer.Write(ShaderBinaryVersion);
    writer.Write(perm_hash);
    writer.Write(perm_idx);
    spec.Serialize(ar);
    info.Serialize(ar);
    Store().Save(BbCache::BlobType::ShaderMeta, MetaName(perm_hash), ar.TakeOff());
    if (capture_recipe) {
        RecipeCatalog::Get().CaptureStageIfUnknown(info, spec, perm_hash, perm_idx);
    }
}

void SaveShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx) {
    if (!Store().IsOpened()) {
        return;
    }
    Store().Save(BbCache::BlobType::ShaderBinary, BinaryName(pgm_hash, perm_idx), std::move(spv));
}

void SaveOptShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx) {
    if (!Store().IsOpened() ||
        (Shader::Backend::SPIRV::NativeShadersCustomRange() &&
         Shader::Backend::SPIRV::HasNativeShader(pgm_hash))) {
        return;
    }
    Store().Save(BbCache::BlobType::ShaderBinary, OptBinaryName(pgm_hash, perm_idx),
                 std::move(spv));
}

} // namespace PipelineStore

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer{ar}.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader{ar}.Read(value);
    return true;
}

void ComputePipeline::SerializationSupport::Serialize(Serialization::Archive&) const {}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive&) {
    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer{ar}.Write(this, sizeof(*this));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader{ar}.Read(this, sizeof(*this));
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer writer{ar};
    writer.Write(&vertex_attributes, sizeof(vertex_attributes));
    writer.Write(&vertex_bindings, sizeof(vertex_bindings));
    writer.Write(&divisors, sizeof(divisors));
    writer.Write(multisampling);
    writer.Write(tcs);
    writer.Write(tes);
    writer.Write(fragment);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader reader{ar};
    reader.Read(&vertex_attributes, sizeof(vertex_attributes));
    reader.Read(&vertex_bindings, sizeof(vertex_bindings));
    reader.Read(&divisors, sizeof(divisors));
    reader.Read(multisampling);
    reader.Read(tcs);
    reader.Read(tes);
    // The clip distance discard shader is stored since 2026-10-13; older entries end before it
    // (their pipelines are recreated without it, as they were before).
    if (!ar.IsEoS()) {
        reader.Read(fragment);
    }
    return true;
}

bool PipelineCache::InitDriverPipelineCache(bool load_from_store, size_t* loaded_bytes) {
    if (loaded_bytes) {
        *loaded_bytes = 0;
    }
    if (load_from_store && Store().IsOpened()) {
        std::vector<u8> driver_blob;
        Store().Load(BbCache::BlobType::ShaderProfile, DriverCacheBlobName(), driver_blob);
        if (driver_blob.size() >= sizeof(VkPipelineCacheHeaderVersionOne)) {
            VkPipelineCacheHeaderVersionOne hdr{};
            std::memcpy(&hdr, driver_blob.data(), sizeof(hdr));
            const auto props = instance.GetPhysicalDevice().getProperties();
            const bool valid =
                hdr.headerSize >= sizeof(VkPipelineCacheHeaderVersionOne) &&
                hdr.headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
                hdr.vendorID == props.vendorID &&
                hdr.deviceID == props.deviceID &&
                std::memcmp(hdr.pipelineCacheUUID, props.pipelineCacheUUID.data(), VK_UUID_SIZE) ==
                    0;
            if (valid) {
                auto [res, cache] = instance.GetDevice().createPipelineCacheUnique({
                    .initialDataSize = driver_blob.size(),
                    .pInitialData = driver_blob.data(),
                });
                if (res == vk::Result::eSuccess) {
                    pipeline_cache = std::move(cache);
                    if (loaded_bytes) {
                        *loaded_bytes = driver_blob.size();
                    }
                    return true;
                }
                LOG_WARNING(Render_Vulkan,
                            "Driver rejected persisted VkPipelineCache ({}), creating fresh cache",
                            vk::to_string(res));
            } else {
                LOG_INFO(Render_Vulkan,
                         "Persisted VkPipelineCache UUID/device mismatch, recreating driver cache");
            }
        }
    }
    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
    return false;
}

void PipelineCache::SaveDriverPipelineCache() {
    std::scoped_lock lk{driver_cache_save_mutex};
    if (!pipeline_cache || !Store().IsOpened()) {
        return;
    }
    // Clear dirty before reading the driver cache so any new PSO compiled concurrently on
    // another thread re-marks it dirty for the next periodic flush or shutdown Sync().
    driver_cache_dirty.store(false, std::memory_order_relaxed);
    auto [res, bytes] = instance.GetDevice().getPipelineCacheData(*pipeline_cache);
    if (res == vk::Result::eSuccess && bytes.size() >= sizeof(VkPipelineCacheHeaderVersionOne)) {
        Store().Save(BbCache::BlobType::ShaderProfile, DriverCacheBlobName(), std::move(bytes));
        last_driver_cache_save_sec = static_cast<u32>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
    }
}

void PipelineCache::NoteDriverPipelineCacheDirty() {
    driver_cache_dirty.store(true, std::memory_order_relaxed);
    if (!Store().IsOpened()) {
        return;
    }
    const auto now_sec = static_cast<u32>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    if (now_sec - last_driver_cache_save_sec < 30) {
        return;
    }
    if (driver_cache_save_inflight.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    last_driver_cache_save_sec = now_sec;
    if (driver_cache_save_thread.joinable()) {
        driver_cache_save_thread.join();
    }
    driver_cache_save_thread = std::thread([this]() {
        SaveDriverPipelineCache();
        driver_cache_save_inflight.store(false, std::memory_order_release);
    });
}

bool PipelineCache::PreloadStage(Serialization::Archive& ar, const Shader::Info*& out_info,
                                 vk::ShaderModule& out_module, Shader::RuntimeInfo& out_runtime,
                                 std::optional<Shader::Gcn::FetchShaderData>& out_fetch) {
    auto program = std::make_unique<Program>();
    Shader::StageSpecialization spec{};
    spec.info = &program->info;
    size_t perm_idx{};
    if (!LoadShaderMeta(ar, program->info, spec, perm_idx, profile)) {
        return false;
    }
    if (profile.paged_buffers && program->info.pgm_hash == profile.buffer_copy_shader_hash &&
        (!program->info.uses_paged_buffers || perm_idx != 0 || spec.buffers.size() < 3 ||
         !spec.buffers[1].paged || !spec.buffers[2].paged)) {
        return false;
    }
    if (spec.fetch_shader_data) {
        out_fetch = spec.fetch_shader_data; // the vertex stage's (as when selected)
    }
    out_runtime = spec.runtime_info;

    const u64 pgm_hash = program->info.pgm_hash;
    // Load SPIR-V and create VkShaderModule only when this (pgm_hash, perm_idx) is not already
    // resident in program_cache. When BB_SPIRV_OPT=1 (default), loads opt1_<hash>_<perm>.spv or
    // upgrades the existing raw <hash>_<perm>.spv in-place once and caches opt1_ on disk.
    const auto load_and_compile = [&](vk::ShaderModule& out_mod) -> bool {
        std::vector<u32> spv{};
        bool upgraded_from_raw = false;
        const bool bypass_opt_cache =
            Shader::Backend::SPIRV::NativeShadersCustomRange() &&
            Shader::Backend::SPIRV::HasNativeShader(pgm_hash);
        if (Shader::Backend::SPIRV::SpirvOptEnabled()) {
            if (!bypass_opt_cache) {
                Store().Load(BbCache::BlobType::ShaderBinary, OptBinaryName(pgm_hash, perm_idx),
                             spv);
                if (!spv.empty() && (spv.size() < 5 || spv[0] != 0x07230203u)) {
                    spv.clear();
                }
            }
            if (spv.empty()) {
                Store().Load(BbCache::BlobType::ShaderBinary, BinaryName(pgm_hash, perm_idx), spv);
                if (spv.size() < 5 || spv[0] != 0x07230203u) {
                    if (!spv.empty()) {
                        throw Serialization::CorruptData{"damaged SPIR-V in the shader cache"};
                    }
                    return false;
                }
                OptimizeSpirvInWarmUp(spv, pgm_hash);
                upgraded_from_raw = !bypass_opt_cache;
            }
        } else {
            Store().Load(BbCache::BlobType::ShaderBinary, BinaryName(pgm_hash, perm_idx), spv);
            if (spv.size() < 5 || spv[0] != 0x07230203u) {
                if (!spv.empty()) {
                    throw Serialization::CorruptData{"damaged SPIR-V in the shader cache"};
                }
                return false;
            }
        }
        if (Shader::Backend::SPIRV::EnsureClipDistanceInitializedInSpv(spv)) {
            upgraded_from_raw = !bypass_opt_cache;
        }
        // A module the driver rejects is a damaged entry (the cache is rebuilt), not a fatal error.
        auto [result, created] = instance.GetDevice().createShaderModule(
            {.codeSize = spv.size() * sizeof(u32), .pCode = spv.data()});
        if (result != vk::Result::eSuccess) {
            throw Serialization::CorruptData{"cached SPIR-V rejected by the driver"};
        }
        if (upgraded_from_raw) {
            PipelineStore::SaveOptShaderBinary(std::move(spv), pgm_hash, perm_idx);
            driver_cache_dirty.store(true, std::memory_order_relaxed);
        }
        out_mod = created;
        return true;
    };

    vk::ShaderModule module{};
    auto [it_pgm, new_program] = program_cache.try_emplace(pgm_hash);
    if (new_program) {
        if (!load_and_compile(module)) {
            program_cache.erase(it_pgm);
            return false;
        }
        it_pgm.value() = std::move(program);
        spec.info = &it_pgm.value()->info;
        it_pgm.value()->info_template = std::make_unique<Shader::Info>(it_pgm.value()->info);
        DirectShaders::NoteProgramCreated(it_pgm.value()->info.pgm_hash, it_pgm.value().get());
    } else {
        auto& existing = *it_pgm.value();
        // Merge paged-buffer layout when a later permutation upgraded uses_paged_buffers while
        // permutation 0's on-disk .meta predated the upgrade.
        if (program->info.uses_paged_buffers && !existing.info.uses_paged_buffers) {
            existing.info.uses_paged_buffers = true;
            existing.info.buffers = program->info.buffers;
            existing.info_template = std::make_unique<Shader::Info>(existing.info);
        }
        // Always rebind spec.info to the persistent Program in program_cache so it never dangles
        // after the local `program` unique_ptr goes out of scope.
        spec.info = &existing.info;

        auto& modules = existing.modules;
        if (perm_idx < modules.size() && modules[perm_idx].module) {
            module = modules[perm_idx].module;
        } else {
            const auto it = std::ranges::find(modules, spec, &Program::Module::spec);
            if (it == modules.end()) {
                if (!load_and_compile(module)) {
                    return false;
                }
            } else {
                const size_t idx = std::distance(modules.begin(), it);
                if (perm_idx != idx) {
                    LOG_WARNING(Render_Vulkan,
                                "Cached permutation {} of {}_{:x} conflicts with index {}, "
                                "skipping preload",
                                perm_idx, program->info.hw_stage, program->info.pgm_hash, idx);
                    return false;
                }
                module = it->module;
            }
        }
    }
    it_pgm.value()->InsertPermut(module, std::move(spec), perm_idx);
    out_info = &it_pgm.value()->info;
    out_module = module;
    return true;
}

namespace {
/// The cache's header: a tag and the versions; the device profile follows.
constexpr std::array<u32, 4> CacheVersions{0x42425043u, PipelineStore::ShaderBinaryVersion,
                                           PipelineStore::ShaderMetaVersion,
                                           PipelineStore::PipelineKeyVersion};
constexpr size_t HeaderSize = sizeof(CacheVersions);

struct PreloadComputeItem {
    ComputePipelineKey key{};
    ComputePipeline::SerializationSupport sdata{};
    const Shader::Info* info = nullptr;
    vk::ShaderModule module{};
    std::unique_ptr<ComputePipeline> built;
};

struct PreloadGraphicsItem {
    GraphicsPipelineKey key{};
    GraphicsPipeline::SerializationSupport sdata{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::unique_ptr<GraphicsPipeline> built;
};
} // namespace

void PipelineCache::SaveProfile() {
    std::vector<u8> data(HeaderSize + sizeof(profile));
    std::memcpy(data.data(), CacheVersions.data(), HeaderSize);
    std::memcpy(data.data() + HeaderSize, &profile, sizeof(profile));
    Store().Save(BbCache::BlobType::ShaderProfile, "profile", std::move(data));
}

void PipelineCache::DropPreloaded() {
    // Modules of a damaged entry may sit in programs that later lookups would reuse: nothing
    // preloaded is kept.
    InvalidateFastSelect();
    {
        std::unique_lock gr_lk{graphics_pipelines_mutex};
        graphics_pipelines.clear();
    }
    compute_pipelines.clear();
    std::unordered_set<VkShaderModule> modules;
    for (const auto& [_, program] : program_cache) {
        if (!program) {
            continue;
        }
        for (const auto& permutation : program->modules) {
            if (permutation.module) {
                modules.insert(VkShaderModule(permutation.module));
            }
        }
    }
    for (const VkShaderModule module : modules) {
        instance.GetDevice().destroyShaderModule(vk::ShaderModule{module});
    }
    program_cache.clear();
    DirectShaders::ClearCachedPrograms();
}

void PipelineCache::WarmUp() {
    const auto t0 = std::chrono::steady_clock::now();
    DirectShaders::InitializeManifestLibrary();
    RecipeCatalog::Get().Initialize();
    if (!EmulatorSettings.IsPipelineCacheEnabled()) {
        InitDriverPipelineCache(false);
        return;
    }
    Store().Open();
    const auto rebuild = [&] {
        InitDriverPipelineCache(false);
        Store().Clear();
        Store().FinishPreload();
        SaveProfile();
    };

    std::vector<u8> header{};
    Store().Load(BbCache::BlobType::ShaderProfile, "profile", header);
    if (header.empty()) {
        rebuild();
        return;
    }
    bool compatible = header.size() == HeaderSize + sizeof(Shader::Profile) &&
                      std::memcmp(header.data(), CacheVersions.data(), HeaderSize) == 0;
    if (compatible) {
        Shader::Profile cached{};
        std::memcpy(&cached, header.data() + HeaderSize, sizeof(cached));
        compatible = cached == profile;
    }
    if (!compatible) {
        LOG_WARNING(Render,
                    "Pipeline cache isn't compatible with current compiler/system: rebuilding it");
        rebuild();
        return;
    }

    size_t loaded_driver_bytes = 0;
    const bool had_driver_cache = InitDriverPipelineCache(true, &loaded_driver_bytes);

    // Phase 1: Deserialize pipeline keys, deduplicate SPIR-V VkShaderModules into program_cache,
    // and merge all multi-permutation Shader::Info layouts on the main thread so Program::info
    // pointers are stable and immutable during Phase 2 parallel PSO creation.
    std::vector<PreloadComputeItem> compute_items;
    std::vector<PreloadGraphicsItem> graphics_items;
    std::unordered_set<ComputePipelineKey> seen_compute;
    std::unordered_set<GraphicsPipelineKey> seen_graphics;
    u32 num_total = 0, num_damaged = 0;

    Store().ForEachBlob(BbCache::BlobType::PipelineKey, [&](std::vector<u8>&& data) {
        ++num_total;
        try {
            Serialization::Archive ar{std::move(data)};
            Serialization::Reader reader{ar};
            u32 version{}, is_compute{};
            reader.Read(version);
            if (version != PipelineStore::PipelineKeyVersion) {
                return;
            }
            reader.Read(is_compute);
            if (is_compute) {
                PreloadComputeItem item{};
                item.key.Deserialize(ar);
                item.sdata.Deserialize(ar);
                if (seen_compute.contains(item.key)) {
                    return;
                }
                std::vector<u8> meta;
                Store().Load(BbCache::BlobType::ShaderMeta, MetaName(item.key.value), meta);
                if (meta.empty()) {
                    return;
                }
                Serialization::Archive meta_ar{std::move(meta)};
                Shader::RuntimeInfo rt{};
                std::optional<Shader::Gcn::FetchShaderData> fetch{};
                if (!PreloadStage(meta_ar, item.info, item.module, rt, fetch)) {
                    return;
                }
                seen_compute.insert(item.key);
                compute_items.push_back(std::move(item));
            } else {
                PreloadGraphicsItem item{};
                item.key.Deserialize(ar);
                item.sdata.Deserialize(ar);
                if (seen_graphics.contains(item.key)) {
                    return;
                }
                for (u32 stage = 0; stage < MaxShaderStages; ++stage) {
                    const u64 hash = item.key.stage_hashes[stage];
                    if (!hash) {
                        continue;
                    }
                    std::vector<u8> meta;
                    Store().Load(BbCache::BlobType::ShaderMeta, MetaName(hash), meta);
                    if (meta.empty()) {
                        return;
                    }
                    Serialization::Archive meta_ar{std::move(meta)};
                    if (!PreloadStage(meta_ar, item.infos[stage], item.modules[stage],
                                      item.runtime_infos[stage], item.fetch_shader)) {
                        return;
                    }
                }
                seen_graphics.insert(item.key);
                graphics_items.push_back(std::move(item));
            }
        } catch (const std::exception& e) {
            if (num_damaged++ == 0) {
                LOG_WARNING(Render, "Pipeline cache: damaged entry ({})", e.what());
            }
        }
    });

    if (num_damaged) {
        LOG_WARNING(Render, "Pipeline cache: {} damaged entries, rebuilding it", num_damaged);
        DropPreloaded();
        rebuild();
        return;
    }

    // Phase 2: Parallel Vulkan graphics & compute PSO compilation against thread-safe
    // VkPipelineCache across worker threads.
    const size_t num_compute = compute_items.size();
    const size_t num_graphics = graphics_items.size();
    const size_t total_items = num_compute + num_graphics;
    const u32 hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const u32 num_workers =
        total_items < 8
            ? 1u
            : std::min<u32>(std::clamp(hw_threads, 2u, 12u), static_cast<u32>(total_items));

    std::atomic<size_t> next_idx{0};
    std::atomic<u32> parallel_damaged{0};
    const auto worker_fn = [&] {
        for (;;) {
            if (parallel_damaged.load(std::memory_order_relaxed) != 0) {
                return;
            }
            const size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total_items) {
                return;
            }
            try {
                if (idx < num_compute) {
                    auto& item = compute_items[idx];
                    item.built = std::make_unique<ComputePipeline>(
                        instance, scheduler, desc_heap, profile, *pipeline_cache, item.key,
                        *item.info, item.module, item.sdata, true);
                } else {
                    auto& item = graphics_items[idx - num_compute];
                    item.built = std::make_unique<GraphicsPipeline>(
                        instance, scheduler, desc_heap, profile, item.key, *pipeline_cache,
                        item.infos, item.runtime_infos, item.fetch_shader, item.modules,
                        item.sdata, true);
                }
            } catch (const std::exception& e) {
                if (parallel_damaged.fetch_add(1, std::memory_order_relaxed) == 0) {
                    LOG_WARNING(Render, "Pipeline cache: damaged PSO during parallel warmup ({})",
                                e.what());
                }
                return;
            }
        }
    };

    if (num_workers <= 1) {
        worker_fn();
    } else {
        std::vector<std::thread> workers;
        workers.reserve(num_workers);
        for (u32 t = 0; t < num_workers; ++t) {
            workers.emplace_back(worker_fn);
        }
        for (auto& w : workers) {
            w.join();
        }
    }

    if (parallel_damaged.load(std::memory_order_relaxed) != 0) {
        LOG_WARNING(Render, "Pipeline cache: {} damaged PSOs, rebuilding cache",
                    parallel_damaged.load());
        compute_items.clear();
        graphics_items.clear();
        DropPreloaded();
        rebuild();
        return;
    }

    // Phase 3: Commit prewarmed PSOs into PipelineCache maps and persist driver VkPipelineCache.
    compute_pipelines.reserve(num_compute);
    for (auto& item : compute_items) {
        if (item.built) {
            compute_pipelines.emplace(item.key, std::move(item.built));
        }
    }
    {
        std::unique_lock gr_lk{graphics_pipelines_mutex};
        graphics_pipelines.reserve(num_graphics);
        for (auto& item : graphics_items) {
            if (item.built) {
                graphics_pipelines.emplace(item.key, std::move(item.built));
            }
        }
    }

    // Phase 3B: Parallel preload of all standalone pre-baked ShaderMeta entries (from --bake-shaders)
    // across `num_workers` threads so all ~5,700 game + Yebis2 + Scaleform shaders sit warm in < 250 ms.
    struct StandalonePreload {
        std::vector<u8> raw_meta;
        std::unique_ptr<Program> program;
        Shader::StageSpecialization spec{};
        size_t perm_idx = 0;
        vk::ShaderModule module{};
        bool ok = false;
    };
    std::unordered_set<u64> resident_perm_hashes;
    resident_perm_hashes.reserve(program_cache.size() * 2 + 64);
    for (const auto& [ph, prog] : program_cache) {
        if (!prog) {
            continue;
        }
        for (size_t pi = 0; pi < prog->modules.size(); ++pi) {
            if (prog->modules[pi].module) {
                resident_perm_hashes.insert(HashCombine(ph, pi));
            }
        }
    }
    std::vector<StandalonePreload> standalone_metas;
    standalone_metas.reserve(6000);
    Store().ForEachBlob(BbCache::BlobType::ShaderMeta, [&](std::vector<u8>&& data) {
        if (data.size() >= 24) {
            u64 ph_comb = 0;
            std::memcpy(&ph_comb, data.data() + 8, sizeof(u64));
            if (resident_perm_hashes.contains(ph_comb)) {
                return;
            }
        }
        standalone_metas.push_back(StandalonePreload{.raw_meta = std::move(data)});
    });
    if (!standalone_metas.empty()) {
        std::atomic<size_t> next_meta{0};
        const bool custom_native_range = Shader::Backend::SPIRV::NativeShadersCustomRange();
        const bool spirv_opt_on = Shader::Backend::SPIRV::SpirvOptEnabled();
        auto meta_worker = [&] {
            for (;;) {
                const size_t idx = next_meta.fetch_add(1, std::memory_order_relaxed);
                if (idx >= standalone_metas.size()) {
                    return;
                }
                auto& item = standalone_metas[idx];
                try {
                    Serialization::Archive ar{std::move(item.raw_meta)};
                    item.program = std::make_unique<Program>();
                    item.spec.info = &item.program->info;
                    if (!LoadShaderMeta(ar, item.program->info, item.spec, item.perm_idx,
                                        profile)) {
                        continue;
                    }
                    const u64 pgm_hash = item.program->info.pgm_hash;
                    const bool bypass_opt_cache =
                        custom_native_range && Shader::Backend::SPIRV::HasNativeShader(pgm_hash);
                    std::vector<u32> spv;
                    bool upgraded = false;
                    if (spirv_opt_on) {
                        if (!bypass_opt_cache) {
                            Store().Load(BbCache::BlobType::ShaderBinary,
                                         OptBinaryName(pgm_hash, item.perm_idx), spv);
                            if (!spv.empty() && (spv.size() < 5 || spv[0] != 0x07230203u)) {
                                spv.clear();
                            }
                        }
                        if (spv.empty()) {
                            Store().Load(BbCache::BlobType::ShaderBinary,
                                         BinaryName(pgm_hash, item.perm_idx), spv);
                            if (spv.size() < 5 || spv[0] != 0x07230203u) {
                                continue;
                            }
                            OptimizeSpirvInWarmUp(spv, pgm_hash);
                            upgraded = !bypass_opt_cache;
                        }
                    } else {
                        Store().Load(BbCache::BlobType::ShaderBinary,
                                     BinaryName(pgm_hash, item.perm_idx), spv);
                        if (spv.size() < 5 || spv[0] != 0x07230203u) {
                            continue;
                        }
                    }
                    if (Shader::Backend::SPIRV::EnsureClipDistanceInitializedInSpv(spv)) {
                        upgraded = !bypass_opt_cache;
                    }
                    auto [res, created] = instance.GetDevice().createShaderModule(
                        {.codeSize = spv.size() * sizeof(u32), .pCode = spv.data()});
                    if (res != vk::Result::eSuccess) {
                        continue;
                    }
                    if (upgraded) {
                        PipelineStore::SaveOptShaderBinary(std::move(spv), pgm_hash, item.perm_idx);
                        driver_cache_dirty.store(true, std::memory_order_relaxed);
                    }
                    item.module = created;
                    item.ok = true;
                } catch (const std::exception&) {
                }
            }
        };
        std::vector<std::thread> mworkers;
        mworkers.reserve(num_workers);
        for (u32 t = 0; t < num_workers; ++t) {
            mworkers.emplace_back(meta_worker);
        }
        for (auto& w : mworkers) {
            w.join();
        }
        program_cache.reserve(program_cache.size() + standalone_metas.size());
        for (auto& item : standalone_metas) {
            if (!item.ok || !item.program || !item.module) {
                continue;
            }
            const u64 pgm_hash = item.program->info.pgm_hash;
            auto [it_pgm, is_new] = program_cache.try_emplace(pgm_hash);
            if (is_new) {
                it_pgm.value() = std::move(item.program);
                item.spec.info = &it_pgm.value()->info;
                it_pgm.value()->info_template =
                    std::make_unique<Shader::Info>(it_pgm.value()->info);
                DirectShaders::NoteProgramCreated(pgm_hash, it_pgm.value().get());
                it_pgm.value()->InsertPermut(item.module, std::move(item.spec), item.perm_idx);
            } else {
                auto& existing = *it_pgm.value();
                item.spec.info = &existing.info;
                if (item.perm_idx < existing.modules.size() &&
                    existing.modules[item.perm_idx].module) {
                    instance.GetDevice().destroyShaderModule(item.module);
                } else {
                    const auto dup =
                        std::ranges::find(existing.modules, item.spec, &Program::Module::spec);
                    if (dup != existing.modules.end()) {
                        instance.GetDevice().destroyShaderModule(item.module);
                    } else {
                        existing.InsertPermut(item.module, std::move(item.spec), item.perm_idx);
                    }
                }
            }
        }
    }

    if ((!had_driver_cache || driver_cache_dirty.load(std::memory_order_relaxed)) &&
        (total_items > 0 || !program_cache.empty())) {
        SaveDriverPipelineCache();
    }

    const auto opt_stats = Shader::Backend::SPIRV::GetAndResetSpirvOptStats();
    if (opt_stats.modules_optimized > 0 || opt_stats.modules_failed > 0) {
        const double pct =
            opt_stats.words_before > 0
                ? 100.0 * (double(opt_stats.words_before) - double(opt_stats.words_after)) /
                      double(opt_stats.words_before)
                : 0.0;
        std::printf("[bb_vulkan] Step 5.4 SPIR-V Optimizer (HW_INTERP=%d, FAST_FP=%d, NATIVE=%d): "
                    "upgraded %u cached modules (%llu -> %llu words, -%.2f%%, NativeRepl=%u, "
                    "HwInterpMods=%u, Varyings=%llu, Channels=%llu, IntNudges=%llu, "
                    "SafeRsqMul=%llu/%llu, Vec4Out=%llu, SelfMinMax=%llu, NoContract=-%llu, "
                    "LegacyMulFold=%llu, InfNanStrip=%u, failed=%u) in %.1f ms\n",
                    Shader::Backend::SPIRV::HwInterpEnabled() ? 1 : 0,
                    Shader::Backend::SPIRV::FastFpEnabled() ? 1 : 0,
                    Shader::Backend::SPIRV::NativeShadersEnabled() ? 1 : 0,
                    opt_stats.modules_optimized,
                    static_cast<unsigned long long>(opt_stats.words_before),
                    static_cast<unsigned long long>(opt_stats.words_after), pct,
                    opt_stats.native_replaced, opt_stats.hw_interp_modules,
                    static_cast<unsigned long long>(opt_stats.hw_interp_varyings),
                    static_cast<unsigned long long>(opt_stats.hw_interp_channels),
                    static_cast<unsigned long long>(opt_stats.hw_interp_int_nudges),
                    static_cast<unsigned long long>(opt_stats.safe_rsq_muls_folded),
                    static_cast<unsigned long long>(opt_stats.safe_rsq_guards_emitted),
                    static_cast<unsigned long long>(opt_stats.vec4_out_stores_coalesced),
                    static_cast<unsigned long long>(opt_stats.self_minmax_folded),
                    static_cast<unsigned long long>(opt_stats.nocontract_stripped),
                    static_cast<unsigned long long>(opt_stats.legacy_mul_folded),
                    opt_stats.inf_nan_preserve_stripped, opt_stats.modules_failed,
                    double(opt_stats.total_ns) / 1e6);
    }

    const u32 num_pipelines =
        static_cast<u32>(compute_pipelines.size() + graphics_pipelines.size());
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[bb_vulkan] V4.2 Prewarmed %zu graphics + %zu compute PSOs (%zu programs) across "
                "%u threads in %.1f ms (VkPipelineCache=%s, %.2f MiB, SPIRV_OPT=%d, FAST_FP=%d, "
                "HW_INTERP=%d, NATIVE=%d)\n",
                graphics_pipelines.size(), compute_pipelines.size(), program_cache.size(),
                num_workers, elapsed_ms, had_driver_cache ? "HIT" : "SEEDED",
                double(loaded_driver_bytes) / (1024.0 * 1024.0),
                Shader::Backend::SPIRV::SpirvOptEnabled() ? 1 : 0,
                Shader::Backend::SPIRV::FastFpEnabled() ? 1 : 0,
                Shader::Backend::SPIRV::HwInterpEnabled() ? 1 : 0,
                Shader::Backend::SPIRV::NativeShadersEnabled() ? 1 : 0);
    LOG_INFO(Render, "Preloaded {} pipelines ({} graphics, {} compute) in {:.1f} ms",
             num_pipelines, graphics_pipelines.size(), compute_pipelines.size(), elapsed_ms);
    if (num_total > num_pipelines) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_total - num_pipelines);
    }

    // Optional developer export mode (`BB_EXPORT_RECIPES=1`): exports all extra runtime shader
    // permutations (`perm >= 1`) plus all Graphics/Compute PSO keys loaded from the local cache
    // into `RecipeCatalog`'s capture file so `tools/merge_recipes.py` can build `data/shader_recipes.bbrc`.
    if (const char* exp = std::getenv("BB_EXPORT_RECIPES"); exp && exp[0] == '1') {
        auto& cat = RecipeCatalog::Get();
        std::unordered_set<u64> pso_stage_hashes;
        pso_stage_hashes.reserve(graphics_items.size() * 2);
        for (const auto& gi : graphics_items) {
            for (u64 sh_h : gi.key.stage_hashes) {
                if (sh_h != 0) pso_stage_hashes.insert(sh_h);
            }
        }
        for (const auto& [h, prog] : program_cache) {
            if (!prog) continue;
            for (size_t i = 0; i < prog->modules.size(); ++i) {
                const auto& m = prog->modules[i];
                if (!m.module) continue;
                const u64 ph = HashCombine(h, i);
                if (i >= 1 || pso_stage_hashes.contains(ph)) {
                    cat.CaptureStageIfUnknown(prog->info, m.spec, ph, i);
                }
            }
        }
        for (const auto& gi : graphics_items) {
            cat.CaptureGraphicsPsoIfUnknown(gi.key, std::hash<GraphicsPipelineKey>{}(gi.key),
                                            gi.sdata);
        }
        for (const auto& ci : compute_items) {
            cat.CaptureComputePsoIfUnknown(ci.key);
        }
        const auto st = cat.GetStats();
        std::printf("[bb_vulkan] RecipeCatalog: BB_EXPORT_RECIPES captured %zu stage perms + "
                    "%zu gfx PSOs + %zu cs PSOs -> %s\n",
                    st.captured_stages, st.captured_gfx_psos, st.captured_cs_psos,
                    cat.CapturePath().string().c_str());
        std::fflush(stdout);
    }
    Store().FinishPreload();
}

void PipelineCache::Sync() {
    prewarm_cancel.store(true, std::memory_order_release);
    {
        std::scoped_lock lk{prewarm_join_mutex};
        if (prewarm_thread.joinable()) {
            prewarm_thread.join();
        }
    }
    StopBackgroundAutoBake();
    if (driver_cache_save_thread.joinable()) {
        driver_cache_save_thread.join();
    }
    if (driver_cache_dirty.load(std::memory_order_relaxed)) {
        SaveDriverPipelineCache();
    }
    RecipeCatalog::Get().Shutdown();
    Store().Close();
}

} // namespace Vulkan
