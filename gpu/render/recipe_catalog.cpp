// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/recipe_catalog.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <zlib.h>

#include "common/serdes.h"
#include "render/pipeline_store.h"

namespace Vulkan {

namespace {

constexpr char kMagicRaw[4]  = {'B', 'B', 'R', 'C'};
constexpr char kMagicZlib[4] = {'B', 'B', 'R', 'Z'};
constexpr size_t kMaxUncompressedCatalogBytes = 64u * 1024u * 1024u; // 64 MiB safety cap

#pragma pack(push, 1)
struct FileHeader {
    char magic[4];
    u32 format_version;
    u32 shader_meta_version;
    u32 pipeline_key_version;
};

struct RecordHeader {
    u8 kind;
    u8 flags;
    u16 reserved;
    u32 payload_bytes;
    u32 fnv1a_32;
};
#pragma pack(pop)

static_assert(sizeof(FileHeader) == 16, "BBRC FileHeader must be 16 bytes");
static_assert(sizeof(RecordHeader) == 12, "BBRC RecordHeader must be 12 bytes");

constexpr u32 Fnv1a32(const u8* data, size_t len) noexcept {
    u32 h = 2166136261u;
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}

constexpr u64 Fnv1a64(const u8* data, size_t len, u64 seed = 14695981039346656037ULL) noexcept {
    u64 h = seed;
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

bool InflateZlibPayload(const u8* src, size_t src_len, std::vector<u8>& out) {
    if (!src || src_len < 4) {
        return false;
    }
    u32 expected_size = 0;
    std::memcpy(&expected_size, src, sizeof(u32));
    if (expected_size == 0 || expected_size > kMaxUncompressedCatalogBytes) {
        return false;
    }
    out.resize(expected_size);
    uLongf dst_len = static_cast<uLongf>(expected_size);
    const int rc = ::uncompress(out.data(), &dst_len, src + 4, static_cast<uLong>(src_len - 4));
    if (rc != Z_OK) {
        out.clear();
        return false;
    }
    out.resize(static_cast<size_t>(dst_len));
    return true;
}

} // namespace

RecipeCatalog& RecipeCatalog::Get() noexcept {
    static RecipeCatalog instance;
    return instance;
}

RecipeCatalog::~RecipeCatalog() {
    Shutdown();
}

std::filesystem::path RecipeCatalog::ResolveDefaultCatalogPath() {
    if (const char* env = std::getenv("BB_RECIPE_CATALOG"); env && env[0]) {
        return std::filesystem::path(env);
    }
    if (const char* data_dir = std::getenv("BB_DATA_DIR"); data_dir && data_dir[0]) {
        const auto candidate = std::filesystem::path(data_dir) / "shader_recipes.bbrc";
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec)) {
            return candidate;
        }
    }
    return std::filesystem::path("data") / "shader_recipes.bbrc";
}

std::filesystem::path RecipeCatalog::ResolveDefaultCapturePath() {
    if (const char* env = std::getenv("BB_RECIPE_CAPTURE_PATH"); env && env[0]) {
        return std::filesystem::path(env);
    }
    return std::filesystem::path("logs") / "recipes" / "unknown_recipes.bbrc";
}

u64 RecipeCatalog::ComputeStageDigest(u64 pgm_hash, Shader::HwStage hw_stage,
                                      Shader::SwStage sw_stage,
                                      const Shader::StageSpecialization& spec) {
    Serialization::Archive ar;
    Serialization::Writer writer{ar};
    writer.Write(pgm_hash);
    writer.Write(static_cast<u8>(hw_stage));
    writer.Write(static_cast<u8>(sw_stage));
    spec.Serialize(ar);
    const auto bytes = ar.TakeOff();
    return Fnv1a64(bytes.data(), bytes.size());
}

u64 RecipeCatalog::ComputeCanonicalStageDigest(u64 pgm_hash, Shader::HwStage hw_stage,
                                               Shader::SwStage sw_stage,
                                               const Shader::StageSpecialization& spec) {
    Shader::StageSpecialization norm = spec;
    u32 binding = 0;
    for (auto& b : norm.buffers) {
        if (binding < Shader::StageSpecialization::MaxStageResources && !norm.bitset[binding]) {
            b = {};
        } else {
            b.paged = false;
        }
        ++binding;
    }
    for (auto& img : norm.images) {
        if (binding < Shader::StageSpecialization::MaxStageResources && !norm.bitset[binding]) {
            img = {};
        }
        ++binding;
    }
    return ComputeStageDigest(pgm_hash, hw_stage, sw_stage, norm);
}

void RecipeCatalog::Initialize(const std::filesystem::path& catalog_path,
                               const std::filesystem::path& capture_path) {
    std::scoped_lock lk{mutex_};
    if (initialized_.load(std::memory_order_acquire)) {
        return;
    }

    if (const char* cap_env = std::getenv("BB_RECIPE_CAPTURE"); cap_env && cap_env[0] == '0') {
        capture_enabled_ = false;
    } else {
        capture_enabled_ = true;
    }

    catalog_path_ = catalog_path.empty() ? ResolveDefaultCatalogPath() : catalog_path;
    capture_path_ = capture_path.empty() ? ResolveDefaultCapturePath() : capture_path;

    std::error_code ec;
    if (!catalog_path_.empty() && std::filesystem::is_regular_file(catalog_path_, ec)) {
        catalog_found_ = LoadStreamFile(catalog_path_, /*store_recipes=*/true);
    }

    // Also index any already-captured session recipes so we never append duplicates across runs.
    if (capture_enabled_ && !capture_path_.empty() &&
        std::filesystem::is_regular_file(capture_path_, ec)) {
        LoadStreamFile(capture_path_, /*store_recipes=*/false);
    }

    initialized_.store(true, std::memory_order_release);

    if (catalog_found_) {
        std::printf("[bb_vulkan] RecipeCatalog: loaded %zu stage perms + %zu gfx PSOs + %zu cs PSOs from %s\n",
                    stage_recipes_.size(), gfx_pso_recipes_.size(), cs_pso_recipes_.size(),
                    catalog_path_.string().c_str());
    }
}

void RecipeCatalog::Shutdown() {
    std::scoped_lock lk{mutex_};
    if (capture_stream_.is_open()) {
        capture_stream_.flush();
        capture_stream_.close();
    }
    if (captured_stages_ > 0 || captured_gfx_psos_ > 0 || captured_cs_psos_ > 0) {
        std::printf("[bb_vulkan] RecipeCatalog: known=%zu perms + %zu gfx PSOs | "
                    "captured +%zu new perms, +%zu new gfx PSOs, +%zu new cs PSOs -> %s\n",
                    stage_recipes_.size(), gfx_pso_recipes_.size(),
                    captured_stages_, captured_gfx_psos_, captured_cs_psos_,
                    capture_path_.string().c_str());
        captured_stages_ = 0;
        captured_gfx_psos_ = 0;
        captured_cs_psos_ = 0;
    }
}

bool RecipeCatalog::LoadStreamFile(const std::filesystem::path& path, bool store_recipes) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return false;
    }
    const auto sz = in.tellg();
    if (sz < static_cast<std::streamoff>(sizeof(FileHeader)) ||
        static_cast<size_t>(sz) > kMaxUncompressedCatalogBytes) {
        return false;
    }
    std::vector<u8> raw(static_cast<size_t>(sz));
    in.seekg(0, std::ios::beg);
    if (!in.read(reinterpret_cast<char*>(raw.data()), sz)) {
        return false;
    }

    FileHeader hdr{};
    std::memcpy(&hdr, raw.data(), sizeof(FileHeader));
    if (hdr.format_version != FormatVersion ||
        hdr.shader_meta_version != PipelineStore::ShaderMetaVersion ||
        hdr.pipeline_key_version != PipelineStore::PipelineKeyVersion) {
        return false;
    }

    if (std::memcmp(hdr.magic, kMagicRaw, 4) == 0) {
        return ParseStreamBytes(raw.data() + sizeof(FileHeader),
                                raw.size() - sizeof(FileHeader), store_recipes);
    }
    if (std::memcmp(hdr.magic, kMagicZlib, 4) == 0) {
        std::vector<u8> inflated;
        if (!InflateZlibPayload(raw.data() + sizeof(FileHeader),
                                raw.size() - sizeof(FileHeader), inflated)) {
            return false;
        }
        return ParseStreamBytes(inflated.data(), inflated.size(), store_recipes);
    }
    return false;
}

bool RecipeCatalog::ParseStreamBytes(const u8* data, size_t size, bool store_recipes) {
    size_t pos = 0;
    while (pos + sizeof(RecordHeader) <= size) {
        RecordHeader rh{};
        std::memcpy(&rh, data + pos, sizeof(RecordHeader));
        pos += sizeof(RecordHeader);
        if (rh.payload_bytes == 0 || pos + rh.payload_bytes > size) {
            break; // Truncated tail record (e.g. abrupt crash): keep all valid records before it.
        }
        const u8* payload = data + pos;
        pos += rh.payload_bytes;
        if (Fnv1a32(payload, rh.payload_bytes) != rh.fnv1a_32) {
            continue; // Corrupted record: skip safely.
        }

        try {
            std::vector<u8> buf(payload, payload + rh.payload_bytes);
            Serialization::Archive ar{std::move(buf)};
            Serialization::Reader reader{ar};

            switch (static_cast<RecordKind>(rh.kind)) {
            case RecordKind::Stage: {
                StageRecipe rec{};
                reader.Read(rec.pgm_hash);
                reader.Read(rec.perm_hash);
                reader.Read(rec.spec_digest);
                reader.Read(rec.perm_idx);
                reader.Read(rec.fetch_shader_sgpr_base);
                u8 hw_u8{}, sw_u8{};
                reader.Read(hw_u8);
                reader.Read(sw_u8);
                reader.Read(rec.mrt_mask);
                reader.Read(rec.info_flags);
                rec.hw_stage = static_cast<Shader::HwStage>(hw_u8);
                rec.sw_stage = static_cast<Shader::SwStage>(sw_u8);
                rec.spec.Deserialize(ar);
                reader.Read(rec.flattened_ud_buf);

                if (rec.hw_stage == Shader::HwStage::Vertex &&
                    rec.spec.runtime_info.hw.vs.motion_vectors) {
                    break; // Session-local BDA motion VS: never cache/pre-bake.
                }
                rec.spec_digest =
                    ComputeCanonicalStageDigest(rec.pgm_hash, rec.hw_stage, rec.sw_stage, rec.spec);
                if (known_stage_digests_.insert(rec.spec_digest).second && store_recipes) {
                    stage_recipes_.push_back(std::move(rec));
                }
                break;
            }
            case RecordKind::GraphicsPso: {
                GraphicsPsoRecipe rec{};
                reader.Read(rec.key_hash);
                rec.key.Deserialize(ar);
                rec.sdata.Deserialize(ar);
                if (known_gfx_pso_hashes_.insert(rec.key_hash).second && store_recipes) {
                    gfx_pso_recipes_.push_back(std::move(rec));
                }
                break;
            }
            case RecordKind::ComputePso: {
                ComputePsoRecipe rec{};
                reader.Read(rec.key_hash);
                rec.key.Deserialize(ar);
                if (known_cs_pso_hashes_.insert(rec.key_hash).second && store_recipes) {
                    cs_pso_recipes_.push_back(std::move(rec));
                }
                break;
            }
            default:
                break;
            }
        } catch (const Serialization::CorruptData&) {
            continue;
        } catch (...) {
            continue;
        }
    }
    return true;
}

bool RecipeCatalog::EnsureCaptureStreamOpenLocked() {
    if (!capture_enabled_ || capture_path_.empty()) {
        return false;
    }
    if (capture_stream_.is_open()) {
        return true;
    }
    std::error_code ec;
    if (capture_path_.has_parent_path()) {
        std::filesystem::create_directories(capture_path_.parent_path(), ec);
    }

    // Check whether an existing capture file has a valid matching FileHeader; if not, recreate.
    bool need_header = true;
    if (std::filesystem::is_regular_file(capture_path_, ec) &&
        std::filesystem::file_size(capture_path_, ec) >= sizeof(FileHeader)) {
        std::ifstream chk(capture_path_, std::ios::binary);
        FileHeader hdr{};
        if (chk && chk.read(reinterpret_cast<char*>(&hdr), sizeof(FileHeader)) &&
            std::memcmp(hdr.magic, kMagicRaw, 4) == 0 &&
            hdr.format_version == FormatVersion &&
            hdr.shader_meta_version == PipelineStore::ShaderMetaVersion &&
            hdr.pipeline_key_version == PipelineStore::PipelineKeyVersion) {
            need_header = false;
        }
    }

    const auto mode = need_header
                          ? (std::ios::binary | std::ios::out | std::ios::trunc)
                          : (std::ios::binary | std::ios::out | std::ios::app);
    capture_stream_.open(capture_path_, mode);
    if (!capture_stream_) {
        capture_enabled_ = false;
        return false;
    }
    if (need_header) {
        FileHeader hdr{};
        std::memcpy(hdr.magic, kMagicRaw, 4);
        hdr.format_version = FormatVersion;
        hdr.shader_meta_version = PipelineStore::ShaderMetaVersion;
        hdr.pipeline_key_version = PipelineStore::PipelineKeyVersion;
        capture_stream_.write(reinterpret_cast<const char*>(&hdr), sizeof(FileHeader));
        capture_stream_.flush();
    }
    return capture_stream_.good();
}

void RecipeCatalog::AppendRecordLocked(RecordKind kind, const std::vector<u8>& payload) {
    if (payload.empty() || payload.size() > 4u * 1024u * 1024u) {
        return;
    }
    if (!EnsureCaptureStreamOpenLocked()) {
        return;
    }
    RecordHeader rh{};
    rh.kind = static_cast<u8>(kind);
    rh.flags = 0u;
    rh.reserved = 0u;
    rh.payload_bytes = static_cast<u32>(payload.size());
    rh.fnv1a_32 = Fnv1a32(payload.data(), payload.size());
    capture_stream_.write(reinterpret_cast<const char*>(&rh), sizeof(RecordHeader));
    capture_stream_.write(reinterpret_cast<const char*>(payload.data()),
                          static_cast<std::streamsize>(payload.size()));
    capture_stream_.flush();
}

void RecipeCatalog::CaptureStageIfUnknown(const Shader::Info& info,
                                          const Shader::StageSpecialization& spec, u64 perm_hash,
                                          size_t perm_idx) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return;
    }
    // Skip session-local BDA motion-vector vertex shaders.
    if (info.hw_stage == Shader::HwStage::Vertex && spec.runtime_info.hw.vs.motion_vectors) {
        return;
    }

    const u64 digest = ComputeCanonicalStageDigest(info.pgm_hash, info.hw_stage, info.sw_stage, spec);

    std::scoped_lock lk{mutex_};
    if (!capture_enabled_ || !known_stage_digests_.insert(digest).second) {
        return;
    }

    Serialization::Archive ar;
    Serialization::Writer writer{ar};
    writer.Write(info.pgm_hash);
    writer.Write(perm_hash);
    writer.Write(digest);
    writer.Write(static_cast<u32>(perm_idx));
    writer.Write(info.fetch_shader_sgpr_base);
    writer.Write(static_cast<u8>(info.hw_stage));
    writer.Write(static_cast<u8>(info.sw_stage));
    writer.Write(info.mrt_mask);
    u8 flags = 0;
    if (info.has_fetch_shader) flags |= InfoFlagHasFetchShader;
    if (info.has_bitwise_xor) flags |= InfoFlagHasBitwiseXor;
    if (info.uses_dma) flags |= InfoFlagUsesDma;
    writer.Write(flags);
    spec.Serialize(ar);
    writer.Write(info.flattened_ud_buf);

    AppendRecordLocked(RecordKind::Stage, ar.TakeOff());
    ++captured_stages_;
}

void RecipeCatalog::CaptureGraphicsPsoIfUnknown(
    const GraphicsPipelineKey& key, u64 key_hash,
    const GraphicsPipeline::SerializationSupport& sdata) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return;
    }
    std::scoped_lock lk{mutex_};
    if (!capture_enabled_ || !known_gfx_pso_hashes_.insert(key_hash).second) {
        return;
    }

    Serialization::Archive ar;
    Serialization::Writer writer{ar};
    writer.Write(key_hash);
    key.Serialize(ar);
    sdata.Serialize(ar);

    AppendRecordLocked(RecordKind::GraphicsPso, ar.TakeOff());
    ++captured_gfx_psos_;
}

void RecipeCatalog::CaptureComputePsoIfUnknown(const ComputePipelineKey& key) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return;
    }
    std::scoped_lock lk{mutex_};
    if (!capture_enabled_ || !known_cs_pso_hashes_.insert(key.value).second) {
        return;
    }

    Serialization::Archive ar;
    Serialization::Writer writer{ar};
    writer.Write(key.value);
    key.Serialize(ar);

    AppendRecordLocked(RecordKind::ComputePso, ar.TakeOff());
    ++captured_cs_psos_;
}

RecipeCatalog::Stats RecipeCatalog::GetStats() const {
    std::scoped_lock lk{mutex_};
    return Stats{
        .loaded_stages = stage_recipes_.size(),
        .loaded_gfx_psos = gfx_pso_recipes_.size(),
        .loaded_cs_psos = cs_pso_recipes_.size(),
        .captured_stages = captured_stages_,
        .captured_gfx_psos = captured_gfx_psos_,
        .captured_cs_psos = captured_cs_psos_,
        .catalog_found = catalog_found_,
        .capture_enabled = capture_enabled_,
    };
}

} // namespace Vulkan
