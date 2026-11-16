// SPDX-License-Identifier: GPL-2.0-or-later
// Full-game multi-threaded Vulkan shader & PSO pre-baker ("0% -> 100%").
// Parses all 7 Dantelion2 .shaderbnd.dcx archives directly from dvdroot_ps4/shader/,
// reconstructs exact Gnm stage registers, PSSL resource types, EUD tables, and GCN fetch shaders,
// compiles all unique CS/PS/VS programs to optimized SPIR-V + VkShaderModule, and pre-warms
// Compute + paired Graphics pipelines into the persistent driver VkPipelineCache.

#include "render/pipeline_cache.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <zlib.h>

#include "common/hash.h"
#include "common/path_util.h"
#include "core/signals.h"
#include "gcn/regs_shader.h"
#include "gcn/sharps.h"
#include "render/blob_store.h"
#include "render/direct_shaders.h"
#include "render/pipeline_store.h"
#include "render/scheduler.h"
#include "shader/backend/spirv/spirv_opt.h"
#include "shim/bbport_toggles.h"
#include "vulkan/instance.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#endif

namespace Vulkan {

using Shader::HwStage;
using Shader::Output;
using Shader::SwStage;

namespace {

bool InflateZlibStream(const u8* src, size_t src_len, size_t expected_uncomp,
                       std::vector<u8>& out) {
    out.resize(expected_uncomp > 0 ? expected_uncomp : (src_len * 4 + 4096));
    z_stream zs{};
    zs.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(src));
    zs.avail_in = static_cast<uInt>(src_len);
    if (inflateInit(&zs) != Z_OK) {
        return false;
    }
    int ret = Z_OK;
    size_t total = 0;
    while (ret == Z_OK) {
        if (total == out.size()) {
            out.resize(out.size() * 2 + 65536);
        }
        zs.next_out = reinterpret_cast<Bytef*>(out.data() + total);
        zs.avail_out = static_cast<uInt>(out.size() - total);
        ret = inflate(&zs, Z_NO_FLUSH);
        total = zs.total_out;
    }
    inflateEnd(&zs);
    if (ret != Z_STREAM_END) {
        return false;
    }
    out.resize(total);
    return true;
}

u32 ReadBe32(const u8* p) noexcept {
    return (u32(p[0]) << 24) | (u32(p[1]) << 16) | (u32(p[2]) << 8) | u32(p[3]);
}

template <typename T>
T ReadLe(const u8* p) noexcept {
    T val{};
    std::memcpy(&val, p, sizeof(T));
    return val;
}

bool LoadDcxFile(const std::filesystem::path& path, std::vector<u8>& out_bnd) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return false;
    }
    const auto fsize = f.tellg();
    if (fsize <= 64) {
        return false;
    }
    std::vector<u8> raw(static_cast<size_t>(fsize));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(raw.data()), fsize)) {
        return false;
    }
    if (std::memcmp(raw.data(), "BND4", 4) == 0) {
        out_bnd = std::move(raw);
        return true;
    }
    if (std::memcmp(raw.data(), "DCX\0", 4) != 0) {
        return false;
    }
    const u32 uncomp_size = ReadBe32(raw.data() + 0x1C);
    static constexpr size_t kCandidateOffsets[] = {0x4C, 0x50, 0x48, 0x2C};
    for (size_t off : kCandidateOffsets) {
        if (off < raw.size() &&
            InflateZlibStream(raw.data() + off, raw.size() - off, uncomp_size, out_bnd) &&
            out_bnd.size() >= 64 && std::memcmp(out_bnd.data(), "BND4", 4) == 0) {
            return true;
        }
    }
    for (size_t off = 0x20; off < std::min<size_t>(raw.size(), 0x90); ++off) {
        if (raw[off] == 0x78 &&
            InflateZlibStream(raw.data() + off, raw.size() - off, uncomp_size, out_bnd) &&
            out_bnd.size() >= 64 && std::memcmp(out_bnd.data(), "BND4", 4) == 0) {
            return true;
        }
    }
    return false;
}

struct BndShaderEntry {
    std::string bundle;
    std::string name;
    std::string stem;
    std::string ext;
    std::vector<u8> bytes;
};

bool ParseBnd4Entries(const std::string& bundle_name, const std::vector<u8>& bnd,
                      std::vector<BndShaderEntry>& out_entries) {
    if (bnd.size() < 0x40 || std::memcmp(bnd.data(), "BND4", 4) != 0) {
        return false;
    }
    const u32 count = ReadLe<u32>(bnd.data() + 0x0C);
    const u64 entry_size = ReadLe<u64>(bnd.data() + 0x20);
    const bool is_unicode = bnd[0x30] != 0;
    if (entry_size < 24 || 0x40ULL + u64(count) * entry_size > bnd.size()) {
        return false;
    }
    for (u32 i = 0; i < count; ++i) {
        const u8* e = bnd.data() + 0x40 + size_t(i) * size_t(entry_size);
        const u64 comp_size = ReadLe<u64>(e + 0x08);
        u64 uncomp_size = comp_size;
        u32 data_off = 0;
        u32 name_off = 0;
        if (entry_size >= 36) {
            uncomp_size = ReadLe<u64>(e + 0x10);
            data_off = ReadLe<u32>(e + 0x18);
            name_off = ReadLe<u32>(e + 0x20);
        } else {
            data_off = ReadLe<u32>(e + 0x18);
            name_off = entry_size >= 32 ? ReadLe<u32>(e + 0x1C) : 0u;
        }
        if (u64(data_off) + comp_size > bnd.size() || comp_size == 0) {
            continue;
        }
        std::string name;
        if (name_off > 0 && name_off < bnd.size()) {
            if (is_unicode) {
                for (size_t p = name_off; p + 1 < bnd.size(); p += 2) {
                    const u16 ch = ReadLe<u16>(bnd.data() + p);
                    if (ch == 0) {
                        break;
                    }
                    name.push_back(ch < 128 ? static_cast<char>(ch) : '_');
                }
            } else {
                for (size_t p = name_off; p < bnd.size() && bnd[p] != 0; ++p) {
                    name.push_back(static_cast<char>(bnd[p]));
                }
            }
        }
        std::vector<u8> payload;
        const u8* src = bnd.data() + data_off;
        if (comp_size >= 2 && src[0] == 0x78 &&
            (src[1] == 0x9C || src[1] == 0xDA || src[1] == 0x01 || src[1] == 0x5E)) {
            if (!InflateZlibStream(src, static_cast<size_t>(comp_size),
                                   static_cast<size_t>(uncomp_size), payload)) {
                payload.assign(src, src + comp_size);
            }
        } else {
            payload.assign(src, src + comp_size);
        }

        const size_t dot = name.find_last_of('.');
        std::string stem = dot != std::string::npos ? name.substr(0, dot) : name;
        std::string ext = dot != std::string::npos ? name.substr(dot) : "";
        for (char& c : ext) {
            if (c >= 'A' && c <= 'Z') {
                c = static_cast<char>(c - 'A' + 'a');
            }
        }
        out_entries.push_back(BndShaderEntry{
            .bundle = bundle_name,
            .name = std::move(name),
            .stem = bundle_name + ":" + stem,
            .ext = std::move(ext),
            .bytes = std::move(payload),
        });
    }
    return true;
}

struct ParsedOrbShader {
    struct InputSlot {
        u8 usage_type;
        u8 api_slot;
        u8 start_reg;
        u8 flags;
    };
    struct VtxInputSem {
        u8 semantic;
        u8 vgpr;
        u8 num_elems;
    };
    struct VtxExportSem {
        u8 semantic;
        u8 out_index;
    };
    struct PsInputSem {
        u8 semantic;
        u8 flags;
    };
    struct PsslResInfo {
        u8 cat = 0xFF;
        u8 pssl_type = 0;
        u32 stride = 0;
    };

    std::string bundle;
    std::string name;
    std::string stem;
    u8 container_stage = 0; // 1=VS, 2=PS, 4=CS
    u64 hash = 0;
    std::vector<u32> code;
    u32 rsrc1 = 0;
    u32 rsrc2 = 0;
    // Stage-specific register words from Shdr body
    u32 vs_out_cntl = 0;
    u32 ps_z_fmt = 0;
    u32 ps_col_fmt = 0;
    u32 ps_input_ena = 0;
    u32 ps_input_addr = 0;
    u32 ps_in_control = 0;
    u32 ps_baryc_cntl = 0;
    u32 ps_db_shader_cntl = 0;
    u32 ps_cb_shader_mask = 0;
    std::array<u32, 3> cs_threads{1, 1, 1};

    std::vector<InputSlot> slots;
    std::vector<VtxInputSem> vs_inputs;
    std::vector<VtxExportSem> vs_exports;
    std::vector<PsInputSem> ps_inputs;
    std::unordered_map<u32, PsslResInfo> srv_by_slot;
    std::unordered_map<u32, PsslResInfo> uav_by_slot;
};

bool ParseOrbContainer(const BndShaderEntry& entry, ParsedOrbShader& out) {
    const auto& raw = entry.bytes;
    if (raw.size() < 0x60 || std::memcmp(raw.data() + 0x24, "Shdr\x07", 5) != 0) {
        return false;
    }
    // Skip .dpo domain shaders (both .vpo and .dpo use Shdr stage_type == 1).
    if (entry.ext == ".dpo" || entry.ext == ".gpo" || entry.ext == ".hpo") {
        return false;
    }
    const size_t shdr = 0x24;
    const u8 stype = raw[shdr + 8];
    if (stype != 1 && stype != 2 && stype != 4) {
        return false; // Only VS (1), PS (2), CS (4)
    }

    const size_t common = shdr + 0x10;
    const u8 num_slots = raw[common + 3];
    const u32 shdr_off = ReadLe<u32>(raw.data() + common + 8);
    const size_t ucode = common + shdr_off;
    if (ucode + 16 > raw.size()) {
        return false;
    }
    const u32* code_ptr = reinterpret_cast<const u32*>(raw.data() + ucode);
    if (code_ptr[0] != 0xBEEB03FFu) {
        return false;
    }
    const size_t orb_off = ucode + size_t(code_ptr[1] + 1u) * 8u;
    if (orb_off + sizeof(Gcn::BinaryInfo) > raw.size()) {
        return false;
    }
    const auto* bin_info = reinterpret_cast<const Gcn::BinaryInfo*>(raw.data() + orb_off);
    if (!bin_info->Valid() || bin_info->length == 0 || ucode + bin_info->length > raw.size()) {
        return false;
    }

    out.bundle = entry.bundle;
    out.name = entry.name;
    out.stem = entry.stem;
    out.container_stage = stype;
    out.hash = bin_info->shader_hash;
    out.code.assign(code_ptr, code_ptr + (bin_info->length / sizeof(u32)));

    const size_t body = common + 0x10;
    out.rsrc1 = ReadLe<u32>(raw.data() + body + 0x00);
    out.rsrc2 = ReadLe<u32>(raw.data() + body + 0x04);

    size_t slot_off = 0;
    u8 num_vs_in = 0, num_vs_exp = 0, num_ps_in = 0;
    if (stype == 1) { // VS
        if (body + 0x18 > ucode) {
            return false;
        }
        out.vs_out_cntl = ReadLe<u32>(raw.data() + body + 0x10);
        num_vs_in = raw[body + 0x14];
        num_vs_exp = raw[body + 0x15];
        slot_off = body + 0x18;
    } else if (stype == 2) { // PS
        if (body + 0x2C > ucode) {
            return false;
        }
        out.ps_z_fmt = ReadLe<u32>(raw.data() + body + 0x08);
        out.ps_col_fmt = ReadLe<u32>(raw.data() + body + 0x0C);
        out.ps_input_ena = ReadLe<u32>(raw.data() + body + 0x10);
        out.ps_input_addr = ReadLe<u32>(raw.data() + body + 0x14);
        out.ps_in_control = ReadLe<u32>(raw.data() + body + 0x18);
        out.ps_baryc_cntl = ReadLe<u32>(raw.data() + body + 0x1C);
        out.ps_db_shader_cntl = ReadLe<u32>(raw.data() + body + 0x20);
        out.ps_cb_shader_mask = ReadLe<u32>(raw.data() + body + 0x24);
        num_ps_in = raw[body + 0x28];
        slot_off = body + 0x2C;
    } else if (stype == 4) { // CS
        if (body + 0x18 > ucode) {
            return false;
        }
        out.cs_threads[0] = std::max<u32>(1u, ReadLe<u32>(raw.data() + body + 0x08) & 0xFFFFu);
        out.cs_threads[1] = std::max<u32>(1u, ReadLe<u32>(raw.data() + body + 0x0C) & 0xFFFFu);
        out.cs_threads[2] = std::max<u32>(1u, ReadLe<u32>(raw.data() + body + 0x10) & 0xFFFFu);
        slot_off = body + 0x18;
    }

    if (slot_off + size_t(num_slots) * 4u > ucode) {
        return false;
    }
    out.slots.reserve(num_slots);
    for (u8 s = 0; s < num_slots; ++s) {
        const u8* sp = raw.data() + slot_off + size_t(s) * 4u;
        out.slots.push_back({sp[0], sp[1], sp[2], sp[3]});
    }
    size_t cur = slot_off + size_t(num_slots) * 4u;
    if (stype == 1) {
        for (u8 k = 0; k < num_vs_in && cur + 4 <= ucode; ++k, cur += 4) {
            out.vs_inputs.push_back({raw[cur], raw[cur + 1], raw[cur + 2]});
        }
        for (u8 k = 0; k < num_vs_exp && cur + 2 <= ucode; ++k, cur += 2) {
            out.vs_exports.push_back({raw[cur], raw[cur + 1]});
        }
    } else if (stype == 2) {
        for (u8 k = 0; k < num_ps_in && cur + 2 <= ucode; ++k, cur += 2) {
            out.ps_inputs.push_back({raw[cur], raw[cur + 1]});
        }
    }

    // Parse PSSL resource reflection table after OrbShdr to get exact T# image dimensions (Cube/3D/Array/MSAA).
    const u32 pssl_rel = ReadLe<u32>(raw.data() + 0x10);
    const size_t pssl_off = 0x24 + size_t(pssl_rel);
    if (pssl_rel > 0 && pssl_off + 0x1C <= raw.size()) {
        const u32 nres = ReadLe<u32>(raw.data() + pssl_off);
        for (u32 r = 0; r < nres && r < 256; ++r) {
            const size_t ro = pssl_off + 0x1C + size_t(r) * 24u;
            if (ro + 24u > raw.size()) {
                break;
            }
            const u32 api_slot = ReadLe<u32>(raw.data() + ro + 0x00);
            const u32 btype = ReadLe<u32>(raw.data() + ro + 0x08);
            const u32 sz = ReadLe<u32>(raw.data() + ro + 0x0C);
            const u8 pssl_type = static_cast<u8>(btype & 0xFFu);
            const u8 cat = static_cast<u8>((btype >> 8) & 0xFFu);
            if (cat == 0x05 || cat == 0x01) {
                out.srv_by_slot[api_slot] = {cat, pssl_type, sz};
            } else if (cat == 0x00) {
                out.uav_by_slot[api_slot] = {cat, pssl_type, sz};
            }
        }
    }
    return true;
}

Gcn::Buffer MakeSyntheticBuffer(uintptr_t backing_addr, u32 stride) noexcept {
    Gcn::Buffer b{};
    b.base_address = static_cast<u64>(backing_addr) & ((1ULL << 40) - 1ULL);
    if (b.base_address == 0) {
        b.base_address = 0x100000ULL;
    }
    b.stride = stride & 0x3FFFu;
    b.num_records = stride == 0 ? 4096u : 1024u;
    b.dst_sel_x = u32(Gcn::CompSwizzle::Red);
    b.dst_sel_y = u32(Gcn::CompSwizzle::Green);
    b.dst_sel_z = u32(Gcn::CompSwizzle::Blue);
    b.dst_sel_w = u32(Gcn::CompSwizzle::Alpha);
    b.num_format = u32(Gcn::NumberFormat::Float);
    b.data_format = u32(Gcn::DataFormat::Format32_32_32_32);
    b.type = 0;
    return b;
}

Gcn::Image MakeSyntheticImage(uintptr_t valid_addr, Gcn::ImageType img_type,
                              bool is_storage) noexcept {
    Gcn::Image img{};
    const u64 shifted = (static_cast<u64>(valid_addr) >> 8) & ((1ULL << 38) - 1ULL);
    img.base_address = shifted != 0 ? shifted : 0x2000ULL;
    img.data_format = u64(is_storage ? Gcn::DataFormat::Format32_32_32_32
                                     : Gcn::DataFormat::Format8_8_8_8);
    img.num_format = u64(is_storage ? Gcn::NumberFormat::Float : Gcn::NumberFormat::Unorm);
    img.width = 1023;
    img.height = 1023;
    img.dst_sel_x = u64(Gcn::CompSwizzle::Red);
    img.dst_sel_y = u64(Gcn::CompSwizzle::Green);
    img.dst_sel_z = u64(Gcn::CompSwizzle::Blue);
    img.dst_sel_w = u64(Gcn::CompSwizzle::Alpha);
    img.base_level = 0;
    img.last_level = img_type == Gcn::ImageType::Color2DMsaa ? 2 : 0;
    img.tiling_index = u64(Gcn::TileMode::Thin1DThin);
    img.type = u64(img_type);
    img.depth = (img_type == Gcn::ImageType::Color3D) ? 31
                : (img_type == Gcn::ImageType::Color2DArray) ? 3 : 0;
    img.pitch = 1023;
    img.last_array = (img_type == Gcn::ImageType::Cube) ? 5
                     : (img_type == Gcn::ImageType::Color2DArray) ? 3 : 0;
    return img;
}

struct SyntheticStageEnv {
    alignas(64) std::array<u32, Gcn::NUM_USER_DATA> user_data{};
    alignas(64) std::array<u32, 512> eud{};
    alignas(64) std::array<u32, 256> vb_table{};
    alignas(64) std::array<u32, 256> res_table{};
    alignas(64) std::array<u32, 128> cb_table{};
    alignas(64) std::array<u32, 128> smp_table{};
    alignas(64) std::array<u32, 256> rw_table{};
    alignas(64) std::array<u32, 256> fetch_code{};
    alignas(64) std::array<u32, 256> dummy_mem{};

    void Build(const ParsedOrbShader& sh) {
        dummy_mem.fill(0);
        const uintptr_t dummy_ptr = reinterpret_cast<uintptr_t>(dummy_mem.data());
        const Gcn::Buffer default_cb = MakeSyntheticBuffer(dummy_ptr, 16);
        const Gcn::Buffer default_vb = MakeSyntheticBuffer(dummy_ptr, 16);
        const Gcn::Image default_img =
            MakeSyntheticImage(dummy_ptr, Gcn::ImageType::Color2D, false);

        for (size_t i = 0; i + 4 <= vb_table.size(); i += 4) {
            std::memcpy(&vb_table[i], &default_vb, sizeof(default_vb));
        }
        for (size_t i = 0; i + 4 <= cb_table.size(); i += 4) {
            std::memcpy(&cb_table[i], &default_cb, sizeof(default_cb));
        }
        for (size_t i = 0; i + 8 <= res_table.size(); i += 8) {
            std::memcpy(&res_table[i], &default_img, sizeof(default_img));
        }
        for (size_t i = 0; i + 8 <= rw_table.size(); i += 8) {
            const Gcn::Image rw_img = MakeSyntheticImage(dummy_ptr, Gcn::ImageType::Color2D, true);
            std::memcpy(&rw_table[i], &rw_img, sizeof(rw_img));
        }
        // Seed EUD with valid 2D image descriptors so even unlisted slots decode cleanly.
        for (size_t i = 0; i + 8 <= eud.size(); i += 8) {
            std::memcpy(&eud[i], &default_img, sizeof(default_img));
        }

        const auto write_words = [&](u32 start_reg, const void* src, size_t dwords) {
            const auto* w = static_cast<const u32*>(src);
            if (start_reg < Gcn::NUM_USER_DATA) {
                const size_t n = std::min<size_t>(dwords, Gcn::NUM_USER_DATA - start_reg);
                std::memcpy(&user_data[start_reg], w, n * sizeof(u32));
            }
            if (start_reg + dwords <= eud.size()) {
                std::memcpy(&eud[start_reg], w, dwords * sizeof(u32));
            }
        };
        const auto write_ptr = [&](u32 start_reg, const void* target) {
            const uintptr_t p = reinterpret_cast<uintptr_t>(target);
            const u32 pair[2] = {static_cast<u32>(p), static_cast<u32>(p >> 32)};
            write_words(start_reg, pair, 2);
        };

        u32 vb_sgpr = 2;
        bool has_vb_ptr = false;
        for (const auto& sl : sh.slots) {
            switch (sl.usage_type) {
            case 0x12: // SubPtrFetchShader
                write_ptr(sl.start_reg, fetch_code.data());
                break;
            case 0x17: // PtrVertexBufferTable
                vb_sgpr = sl.start_reg;
                has_vb_ptr = true;
                write_ptr(sl.start_reg, vb_table.data());
                break;
            case 0x18: // PtrResourceTable
                write_ptr(sl.start_reg, res_table.data());
                break;
            case 0x19: // PtrConstBufferTable
                write_ptr(sl.start_reg, cb_table.data());
                break;
            case 0x1A: // PtrRwResourceTable
                write_ptr(sl.start_reg, rw_table.data());
                break;
            case 0x1B: // PtrExtendedUserData
                write_ptr(sl.start_reg, eud.data());
                break;
            case 0x1D: // PtrSamplerTable
                write_ptr(sl.start_reg, smp_table.data());
                break;
            case 0x01: { // ImmSampler (4 dwords)
                const u32 smp[4] = {0, 0, 0, 0};
                write_words(sl.start_reg, smp, 4);
                break;
            }
            case 0x02: // ImmConstBuffer (4 dwords)
                write_words(sl.start_reg, &default_cb, 4);
                break;
            case 0x03: // ImmVertexBuffer (4 dwords)
                write_words(sl.start_reg, &default_vb, 4);
                break;
            case 0x00:   // ImmResource (SRV: 8-dword T# if flags&1 else 4-dword V#)
            case 0x04: { // ImmRwResource (UAV: 8-dword T# if flags&1 else 4-dword V#)
                const bool is_rw = (sl.usage_type == 0x04);
                const auto& map = is_rw ? sh.uav_by_slot : sh.srv_by_slot;
                const auto it = map.find(sl.api_slot);
                const u8 pssl_t = (it != map.end()) ? it->second.pssl_type : 0xFFu;
                const bool is_buf =
                    ((sl.flags & 1u) == 0u) ||
                    (it != map.end() &&
                     (it->second.cat == 0x01 || pssl_t == 0x00 ||
                      (pssl_t >= 0x12 && pssl_t <= 0x15)));
                if (is_buf) {
                    const u32 stride =
                        (it != map.end() && it->second.stride > 0) ? (it->second.stride * 4u) : 4u;
                    const Gcn::Buffer sb = MakeSyntheticBuffer(dummy_ptr, stride);
                    write_words(sl.start_reg, &sb, 4);
                } else {
                    Gcn::ImageType img_t = Gcn::ImageType::Color2D;
                    switch (pssl_t) {
                    case 0x01:
                        img_t = Gcn::ImageType::Color1D;
                        break;
                    case 0x03:
                    case 0x0F:
                        img_t = Gcn::ImageType::Color3D;
                        break;
                    case 0x04:
                        img_t = Gcn::ImageType::Cube;
                        break;
                    case 0x06:
                    case 0x07:
                        img_t = Gcn::ImageType::Color2DArray;
                        break;
                    case 0x08:
                    case 0x0A:
                    case 0x0B:
                        img_t = Gcn::ImageType::Color2DMsaa;
                        break;
                    default:
                        img_t = Gcn::ImageType::Color2D;
                        break;
                    }
                    const Gcn::Image im = MakeSyntheticImage(dummy_ptr, img_t, is_rw);
                    write_words(sl.start_reg, &im, 8);
                }
                break;
            }
            default:
                break;
            }
        }

        // Synthesize exact GCN FetchShader instructions for VS inputs.
        size_t pc = 0;
        if (sh.container_stage == 1) {
            if (!has_vb_ptr) {
                vb_sgpr = 2;
                write_ptr(vb_sgpr, vb_table.data());
            }
            const u32 dst_s = 12; // s[12:15] scratch V# inside fetch shader
            for (size_t k = 0; k < sh.vs_inputs.size() && pc + 6 < fetch_code.size(); ++k) {
                const auto& vin = sh.vs_inputs[k];
                const u32 elems = std::clamp<u32>(vin.num_elems, 1u, 4u);
                const u32 dfmt = (elems == 1)   ? u32(Gcn::DataFormat::Format32)
                                 : (elems == 2) ? u32(Gcn::DataFormat::Format32_32)
                                 : (elems == 3) ? u32(Gcn::DataFormat::Format32_32_32)
                                                : u32(Gcn::DataFormat::Format32_32_32_32);
                const u32 nfmt = u32(Gcn::NumberFormat::Float);
                const u32 dw_off = static_cast<u32>(k * 4u);
                // s_load_dwordx4 s[12:15], s[vb_sgpr:vb_sgpr+1], dw_off
                fetch_code[pc++] =
                    0xC0800100u | (dst_s << 15) | ((vb_sgpr / 2u) << 9) | (dw_off & 0xFFu);
                // s_waitcnt lgkmcnt(0)
                fetch_code[pc++] = 0xBF8C007Fu;
                // tbuffer_load_format_{x..xyzw} v[vgpr], v0, s[12:15], 0 idxen format:[dfmt, float]
                fetch_code[pc++] = (0x3Au << 26) | (nfmt << 23) | (dfmt << 19) |
                                   ((elems - 1u) << 16) | (1u << 13);
                fetch_code[pc++] = ((dst_s / 4u) << 16) | (u32(vin.vgpr) << 8) | 0u;
            }
            fetch_code[pc++] = 0xBF8C007Fu; // s_waitcnt 0
        }
        fetch_code[pc++] = 0xBE802000u; // s_setpc_b64 s[0:1]
    }
};

Shader::RuntimeInfo BuildSyntheticRuntimeInfo(const ParsedOrbShader& sh,
                                              const Vulkan::Instance& instance,
                                              const ParsedOrbShader* paired_vs = nullptr) {
    using Shader::HwStage;
    using Shader::SwStage;
    Shader::RuntimeInfo rt{};
    if (sh.container_stage == 4) {
        Gcn::ComputeProgram cs_prog{};
        std::memcpy(&cs_prog.settings, &sh.rsrc1, 4);
        std::memcpy(reinterpret_cast<u8*>(&cs_prog.settings) + 4, &sh.rsrc2, 4);
        rt.Initialize(HwStage::Compute, SwStage::Compute);
        rt.props.num_user_data = cs_prog.settings.num_user_regs;
        rt.props.num_allocated_vgprs = cs_prog.settings.num_vgprs * 4;
        rt.props.fp_denorm_mode32 = cs_prog.settings.fp_denorm_mode32;
        rt.props.fp_denorm_mode16_64 = cs_prog.settings.fp_denorm_mode64;
        rt.props.fp_round_mode32 = cs_prog.settings.fp_round_mode32;
        rt.props.fp_round_mode16_64 = cs_prog.settings.fp_round_mode64;
        rt.hw.cs.workgroup_size = {sh.cs_threads[0], sh.cs_threads[1], sh.cs_threads[2]};
        rt.hw.cs.tgid_enable = {cs_prog.IsTgidEnabled(0), cs_prog.IsTgidEnabled(1),
                                cs_prog.IsTgidEnabled(2)};
        rt.hw.cs.shared_memory_size = cs_prog.SharedMemSize();
        return rt;
    }

    bool has_fetch_slot = false;
    for (const auto& sl : sh.slots) {
        if (sl.usage_type == 0x12) {
            has_fetch_slot = true;
            break;
        }
    }

    Gcn::ShaderProgram gfx_prog{};
    std::memcpy(&gfx_prog.settings, &sh.rsrc1, 4);
    std::memcpy(reinterpret_cast<u8*>(&gfx_prog.settings) + 4, &sh.rsrc2, 4);
    const auto fill_common = [&] {
        rt.props.num_user_data = gfx_prog.settings.num_user_regs;
        rt.props.num_input_vgprs =
            (sh.container_stage == 1 && (has_fetch_slot || !sh.vs_inputs.empty()))
                ? 3u
                : u32(gfx_prog.settings.vgpr_comp_cnt);
        rt.props.num_allocated_vgprs = gfx_prog.NumVgprs();
        rt.props.fp_denorm_mode32 = gfx_prog.settings.fp_denorm_mode32;
        rt.props.fp_denorm_mode16_64 = gfx_prog.settings.fp_denorm_mode64;
        rt.props.fp_round_mode32 = gfx_prog.settings.fp_round_mode32;
        rt.props.fp_round_mode16_64 = gfx_prog.settings.fp_round_mode64;
    };

    if (sh.container_stage == 1) {
        rt.Initialize(HwStage::Vertex, SwStage::Vertex);
        fill_common();
        auto& vs = rt.hw.vs;
        Gcn::VsOutputControl ctl{};
        std::memcpy(&ctl, &sh.vs_out_cntl, sizeof(u32));
        u32 num_outputs = 0;
        if (ctl.vs_out_misc_enable) {
            auto& misc = vs.outputs[num_outputs++];
            misc[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
            misc[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
            misc[2] = ctl.use_vtx_kill_flag
                          ? Output::KillFlag
                          : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex
                                                           : Output::None);
            misc[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
        }
        const std::array<bool, 2> dist_vecs = {bool(ctl.vs_out_ccdist0_enable),
                                               bool(ctl.vs_out_ccdist1_enable)};
        for (u32 vec = 0; vec < dist_vecs.size(); ++vec) {
            if (dist_vecs[vec]) {
                auto& map = vs.outputs[num_outputs++];
                for (u32 comp = 0; comp < map.size(); ++comp) {
                    const u32 idx = vec * 4 + comp;
                    map[comp] = ctl.IsClipDistEnabled(idx)
                                    ? Output(u32(Output::ClipDist0) + idx)
                                    : (ctl.IsCullDistEnabled(idx)
                                           ? Output(u32(Output::CullDist0) + idx)
                                           : Output::None);
                }
            }
        }
        vs.num_outputs = num_outputs;
        vs.emulate_depth_negative_one_to_one = !instance.IsDepthClipControlSupported();
        rt.sw.vs.step_rate_0 = 0;
        rt.sw.vs.step_rate_1 = 0;
        return rt;
    }

    // Pixel Shader (stage == 2)
    rt.Initialize(HwStage::Fragment, SwStage::Fragment);
    fill_common();
    auto& fs = rt.hw.fs;
    std::memcpy(&fs.en_flags, &sh.ps_input_ena, sizeof(u32));
    std::memcpy(&fs.addr_flags, &sh.ps_input_addr, sizeof(u32));
    const u32 num_interp = std::min<u32>(sh.ps_in_control & 0x3Fu, u32(sh.ps_inputs.size()));
    fs.num_inputs = num_interp;
    fs.num_samples = 1;
    fs.z_export_format = static_cast<Gcn::ShaderExportFormat>(sh.ps_z_fmt & 0xFu);
    for (u32 i = 0; i < num_interp; ++i) {
        u8 param_idx = static_cast<u8>(i);
        if (paired_vs) {
            for (const auto& exp : paired_vs->vs_exports) {
                if (exp.semantic == sh.ps_inputs[i].semantic) {
                    param_idx = static_cast<u8>(exp.out_index & 0x1Fu);
                    break;
                }
            }
        }
        const bool is_flat = (sh.ps_inputs[i].flags & 0x02u) != 0;
        fs.inputs[i] = {
            .param_index = param_idx,
            .is_default = false,
            .is_flat = is_flat,
            .default_value = 0,
        };
    }
    // Populate exported color buffer formats matching Bloodborne's exact G-Buffer / HDR / UI RTs.
    u32 active_mrt_mask = 0;
    for (u32 cb = 0; cb < Shader::MaxColorBuffers; ++cb) {
        if (((sh.ps_col_fmt >> (cb * 4u)) & 0xFu) != 0) {
            active_mrt_mask |= (1u << cb);
        }
    }
    static constexpr Gcn::CompMapping kRgZeroMap = {
        .r = Gcn::CompSwizzle::Red,
        .g = Gcn::CompSwizzle::Green,
        .b = Gcn::CompSwizzle::Blue,
        .a = Gcn::CompSwizzle::Zero,
    };
    static constexpr Gcn::CompMapping kBgraMap = {
        .r = Gcn::CompSwizzle::Blue,
        .g = Gcn::CompSwizzle::Green,
        .b = Gcn::CompSwizzle::Red,
        .a = Gcn::CompSwizzle::Alpha,
    };
    for (u32 cb = 0; cb < Shader::MaxColorBuffers; ++cb) {
        const u32 exp_fmt = (sh.ps_col_fmt >> (cb * 4u)) & 0xFu;
        if (exp_fmt == 0) {
            continue;
        }
        auto& col = fs.color_buffers[cb];
        col.export_format = static_cast<Gcn::ShaderExportFormat>(exp_fmt);
        col.swizzle = Gcn::IdentityMapping;
        if (active_mrt_mask == 0x3Fu) {
            // 6-MRT Bloodborne G-Buffer: 8888 Unorm x2, 8888 Srgb (Albedo), 8888 Unorm, RG16F, RGBA16F
            if (cb == 2) {
                col.data_format = Gcn::DataFormat::Format8_8_8_8;
                col.num_format = Gcn::NumberFormat::Srgb;
            } else if (cb == 4) {
                col.data_format = Gcn::DataFormat::Format10_11_11;
                col.num_format = Gcn::NumberFormat::Float;
                col.swizzle = kRgZeroMap;
            } else if (cb == 5) {
                col.data_format = Gcn::DataFormat::Format16_16_16_16;
                col.num_format = Gcn::NumberFormat::Float;
            } else {
                col.data_format = Gcn::DataFormat::Format8_8_8_8;
                col.num_format = Gcn::NumberFormat::Unorm;
            }
        } else if (active_mrt_mask == 0x03u) {
            if (sh.bundle == "gxdecal") {
                col.data_format = Gcn::DataFormat::Format8_8_8_8;
                col.num_format = (cb == 0) ? Gcn::NumberFormat::Srgb : Gcn::NumberFormat::Unorm;
            } else {
                col.data_format = Gcn::DataFormat::Format10_11_11;
                col.num_format = Gcn::NumberFormat::Float;
                col.swizzle = kRgZeroMap;
            }
        } else if (sh.bundle == "gxgui") {
            col.data_format = Gcn::DataFormat::Format8_8_8_8;
            col.num_format = Gcn::NumberFormat::Unorm;
            col.swizzle = kBgraMap;
        } else {
            col.data_format = Gcn::DataFormat::Format16_16_16_16;
            col.num_format = Gcn::NumberFormat::Float;
        }
    }
    return rt;
}

void CleanupObsoleteCacheFiles(const std::filesystem::path& cache_dir) {
    std::error_code ec;
    if (!std::filesystem::exists(cache_dir, ec)) {
        return;
    }
    size_t removed = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(cache_dir, ec)) {
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const std::string fn = entry.path().filename().string();
        if (fn == "vk_driver_cache_opt31.bin" || fn == "vk_driver_cache_opt32.bin" ||
            fn == "vk_driver_cache_opt33.bin" || fn == "vk_driver_cache_opt34.bin") {
            if (std::filesystem::remove(entry.path(), ec)) {
                ++removed;
            }
        }
    }
    if (removed > 0) {
        std::printf("[BB-BAKE] Cleaned %zu obsolete driver cache files\n", removed);
        std::fflush(stdout);
    }
}

double ComputeDirectorySizeMiB(const std::filesystem::path& dir) {
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) {
        return 0.0;
    }
    u64 bytes = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dir, ec)) {
        if (entry.is_regular_file(ec)) {
            bytes += entry.file_size(ec);
        }
    }
    return double(bytes) / (1024.0 * 1024.0);
}

std::filesystem::path ResolveSiblingPatchRoot(const std::filesystem::path& game_dir) {
    std::error_code ec;
    if (const char* penv = std::getenv("BB_PATCH_DIR"); penv && *penv) {
        std::filesystem::path explicit_patch{
            std::u8string_view(reinterpret_cast<const char8_t*>(penv))};
        if (std::filesystem::exists(explicit_patch, ec)) {
            return explicit_patch;
        }
    }
    auto norm = game_dir.lexically_normal();
    if (!norm.has_filename()) {
        norm = norm.parent_path();
    }
    const auto fname = norm.filename().u8string();
    if (!fname.empty()) {
        std::u8string patch_name = fname;
        patch_name += u8"-patch";
        const auto cand = norm.parent_path() / std::filesystem::path(patch_name);
        if (std::filesystem::exists(cand, ec)) {
            return cand;
        }
    }
    return {};
}

/// Scans `eboot.bin` (or `out/boot.bin`) for embedded `Shdr\x07` shader containers:
///   - `00 03 E2 20` (`354` Scaleform GFx UI, internal GNM & Compute shaders -> `"gxgui"`),
///   - `00 03 0D 20` (`Yebis2` Post-FX shaders with `col_fmt != 9` -> `"eboot_yebis"`).
void ScanEmbeddedEbootShaders(const std::filesystem::path& game_dir,
                              std::vector<ParsedOrbShader>& cs_list,
                              std::vector<ParsedOrbShader>& ps_list,
                              std::vector<ParsedOrbShader>& vs_list,
                              std::unordered_map<u64, size_t>& cs_seen,
                              std::unordered_map<u64, size_t>& ps_seen,
                              std::unordered_map<u64, size_t>& vs_seen) {
    std::error_code ec;
    std::filesystem::path exe_path;
    const auto patch_root = ResolveSiblingPatchRoot(game_dir);
    if (!patch_root.empty() && std::filesystem::exists(patch_root / "eboot.bin", ec)) {
        exe_path = patch_root / "eboot.bin";
    } else if (std::filesystem::exists(game_dir / "eboot.bin", ec)) {
        exe_path = game_dir / "eboot.bin";
    } else if (std::filesystem::exists("out/boot.bin", ec)) {
        exe_path = "out/boot.bin";
    } else if (std::filesystem::exists("boot.bin", ec)) {
        exe_path = "boot.bin";
    } else {
        return;
    }
    std::ifstream ef(exe_path, std::ios::binary | std::ios::ate);
    if (!ef) {
        return;
    }
    const auto fsize = ef.tellg();
    if (fsize < 0x1000 || fsize > std::streamoff(256LL * 1024LL * 1024LL)) {
        return;
    }
    std::vector<u8> exe(static_cast<size_t>(fsize));
    ef.seekg(0);
    if (!ef.read(reinterpret_cast<char*>(exe.data()), fsize)) {
        return;
    }

    static constexpr u8 kShdrMagic[5] = {'S', 'h', 'd', 'r', 0x07};
    for (size_t pos = 0x24; pos + 0x60 < exe.size();) {
        const void* found = std::memchr(exe.data() + pos, 'S', exe.size() - pos - 0x60);
        if (!found) {
            break;
        }
        const size_t shdr = static_cast<const u8*>(found) - exe.data();
        pos = shdr + 1;
        if (shdr < 0x24 || std::memcmp(exe.data() + shdr, kShdrMagic, 5) != 0) {
            continue;
        }
        const size_t base = shdr - 0x24;
        const u8* p4 = exe.data() + base;
        const bool is_scaleform_gnm =
            (p4[0] == 0x00 && p4[1] == 0x03 && p4[2] == 0xE2 && p4[3] == 0x20);
        const bool is_yebis =
            (p4[0] == 0x00 && p4[1] == 0x03 && p4[2] == 0x0D && p4[3] == 0x20);
        if (!is_scaleform_gnm && !is_yebis) {
            continue;
        }
        const u8 stype = exe[shdr + 8];
        if (stype != 1 && stype != 2 && stype != 4) {
            continue;
        }
        const size_t common = shdr + 0x10;
        const u32 shdr_off = ReadLe<u32>(exe.data() + common + 8);
        const size_t ucode = common + size_t(shdr_off);
        if (ucode <= common || ucode + 16 > exe.size()) {
            continue;
        }
        if (is_yebis && stype == 2) {
            const size_t body = common + 0x10;
            if (body + 0x10 <= exe.size()) {
                const u32 col_fmt = ReadLe<u32>(exe.data() + body + 0x0C);
                // Bloodborne's Yebis2 configuration exclusively uses EXP_32_ABGR (4) and never
                // touches the 2,513 duplicate EXP_FP16_ABGR (9) shaders embedded in .rodata.
                if ((col_fmt & 0xFu) == 9u) {
                    continue;
                }
            }
        }
        const u32 dw0 = ReadLe<u32>(exe.data() + ucode);
        const u32 dw1 = ReadLe<u32>(exe.data() + ucode + 4);
        if (dw0 != 0xBEEB03FFu) {
            continue;
        }
        const size_t orb_off = ucode + size_t(dw1 + 1u) * 8u;
        if (orb_off + sizeof(Gcn::BinaryInfo) > exe.size()) {
            continue;
        }
        size_t end_off = orb_off + sizeof(Gcn::BinaryInfo);
        const u32 pssl_rel = ReadLe<u32>(exe.data() + base + 0x10);
        const size_t pssl_abs = base + 0x24 + size_t(pssl_rel);
        if (pssl_rel > 0 && pssl_abs < base + 0x40000 && pssl_abs + 0x1C <= exe.size()) {
            const u32 nres = std::min<u32>(ReadLe<u32>(exe.data() + pssl_abs), 256u);
            end_off = std::max(end_off, std::min(exe.size(), pssl_abs + 0x1C + size_t(nres) * 24u));
        }

        BndShaderEntry entry{
            .bundle = is_scaleform_gnm ? "gxgui" : "eboot_yebis",
            .name = "eboot",
            .stem = "eboot",
            .ext = stype == 1 ? ".vpo" : (stype == 2 ? ".ppo" : ".cpo"),
            .bytes = {exe.begin() + static_cast<std::ptrdiff_t>(base),
                      exe.begin() + static_cast<std::ptrdiff_t>(end_off)},
        };
        ParsedOrbShader parsed{};
        if (!ParseOrbContainer(entry, parsed)) {
            continue;
        }
        if (parsed.container_stage == 1) {
            if (!vs_seen.contains(parsed.hash)) {
                vs_seen[parsed.hash] = vs_list.size();
                vs_list.push_back(std::move(parsed));
            }
        } else if (parsed.container_stage == 2) {
            if (!ps_seen.contains(parsed.hash)) {
                ps_seen[parsed.hash] = ps_list.size();
                ps_list.push_back(std::move(parsed));
            }
        } else if (parsed.container_stage == 4) {
            if (!cs_seen.contains(parsed.hash)) {
                cs_seen[parsed.hash] = cs_list.size();
                cs_list.push_back(std::move(parsed));
            }
        }
        pos = orb_off + 8;
    }
}

} // namespace

int PipelineCache::BakeAllGameShaders(const std::filesystem::path& game_dir) {
    const auto t_start = std::chrono::steady_clock::now();
    const auto shader_dir = game_dir / "dvdroot_ps4" / "shader";
    if (!std::filesystem::exists(shader_dir)) {
        std::fprintf(stderr, "[BB-BAKE] ERROR: shader directory not found: %s\n",
                     shader_dir.string().c_str());
        std::fflush(stderr);
        return 1;
    }

    const auto cache_root = Common::FS::GetUserPath(Common::FS::PathType::CacheDir);
    CleanupObsoleteCacheFiles(cache_root);

    static constexpr const char* kBundles[] = {
        "gxshader.shaderbnd.dcx",         "gxgui.shaderbnd.dcx",
        "gxposteffect.shaderbnd.dcx",     "gxrenderershader.shaderbnd.dcx",
        "gxdecal.shaderbnd.dcx",          "gxffxshader.shaderbnd.dcx",
        "gxflvershader.shaderbnd.dcx",
    };

    std::printf("[BB-BAKE] SCANNING dir=\"%s\"\n", shader_dir.string().c_str());
    std::fflush(stdout);

    std::vector<ParsedOrbShader> cs_list;
    std::vector<ParsedOrbShader> ps_list;
    std::vector<ParsedOrbShader> vs_list;
    std::unordered_map<u64, size_t> cs_by_hash;
    std::unordered_map<u64, size_t> ps_by_hash;
    std::unordered_map<u64, size_t> vs_by_hash;
    std::unordered_map<std::string, u64> stem_vs_hash;
    std::unordered_map<std::string, u64> stem_ps_hash;

    const auto patch_root = ResolveSiblingPatchRoot(game_dir);
    const auto patch_shader_dir =
        patch_root.empty() ? std::filesystem::path{} : (patch_root / "dvdroot_ps4" / "shader");
    for (const char* bname : kBundles) {
        std::error_code bec;
        auto bpath = shader_dir / bname;
        if (!patch_shader_dir.empty() && std::filesystem::exists(patch_shader_dir / bname, bec)) {
            bpath = patch_shader_dir / bname;
        }
        if (!std::filesystem::exists(bpath, bec)) {
            continue;
        }
        std::vector<u8> bnd_bytes;
        if (!LoadDcxFile(bpath, bnd_bytes)) {
            continue;
        }
        std::string short_bundle = bname;
        if (const auto dot = short_bundle.find('.'); dot != std::string::npos) {
            short_bundle.resize(dot);
        }
        std::vector<BndShaderEntry> entries;
        if (!ParseBnd4Entries(short_bundle, bnd_bytes, entries)) {
            continue;
        }
        for (const auto& e : entries) {
            ParsedOrbShader parsed{};
            if (!ParseOrbContainer(e, parsed)) {
                continue;
            }
            const std::string qualified_stem = short_bundle + ":" + parsed.stem;
            if (parsed.container_stage == 1) {
                stem_vs_hash[qualified_stem] = parsed.hash;
                if (!vs_by_hash.contains(parsed.hash)) {
                    vs_by_hash[parsed.hash] = vs_list.size();
                    vs_list.push_back(std::move(parsed));
                }
            } else if (parsed.container_stage == 2) {
                stem_ps_hash[qualified_stem] = parsed.hash;
                if (!ps_by_hash.contains(parsed.hash)) {
                    ps_by_hash[parsed.hash] = ps_list.size();
                    ps_list.push_back(std::move(parsed));
                }
            } else if (parsed.container_stage == 4) {
                if (!cs_by_hash.contains(parsed.hash)) {
                    cs_by_hash[parsed.hash] = cs_list.size();
                    cs_list.push_back(std::move(parsed));
                }
            }
        }
    }
    ScanEmbeddedEbootShaders(game_dir, cs_list, ps_list, vs_list, cs_by_hash, ps_by_hash,
                             vs_by_hash);

    // Link paired VS <-> PS by bundle-qualified BND4 stem so PS interpolants & VS descriptor start offsets match.
    struct PassPairEntry {
        u64 v_hash;
        u64 p_hash;
        std::string bundle;
    };
    std::unordered_map<u64, u64> ps_to_vs_pair;
    std::unordered_map<u64, u64> vs_to_ps_pair;
    std::vector<PassPairEntry> pass_pairs;
    for (const auto& [qstem, v_hash] : stem_vs_hash) {
        if (const auto pit = stem_ps_hash.find(qstem); pit != stem_ps_hash.end()) {
            ps_to_vs_pair.try_emplace(pit->second, v_hash);
            vs_to_ps_pair.try_emplace(v_hash, pit->second);
            const auto colon = qstem.find(':');
            pass_pairs.push_back({
                .v_hash = v_hash,
                .p_hash = pit->second,
                .bundle = (colon != std::string::npos) ? qstem.substr(0, colon) : std::string{},
            });
        }
    }

    const size_t total_shaders = cs_list.size() + ps_list.size() + vs_list.size();
    if (total_shaders == 0) {
        std::fprintf(stderr, "[BB-BAKE] ERROR: no shaders found in %s\n",
                     shader_dir.string().c_str());
        return 1;
    }

    const u32 num_threads = std::clamp(std::thread::hardware_concurrency(), 2u, 24u);
    const auto gpu_props = instance.GetPhysicalDevice().getProperties();
    std::printf("[BB-BAKE] START total=%zu (cs=%zu ps=%zu vs=%zu pairs=%zu) threads=%u gpu=\"%s\"\n",
                total_shaders, cs_list.size(), ps_list.size(), vs_list.size(), pass_pairs.size(),
                num_threads, gpu_props.deviceName.data());
    std::fflush(stdout);

    std::atomic<size_t> done_count{0};
    std::atomic<size_t> spv_built{0};
    std::atomic<size_t> spv_cached{0};
    std::atomic<size_t> cs_pso_built{0};
    std::atomic<size_t> gfx_pso_built{0};
    std::atomic<u32> last_reported_pct{0};
    std::mutex commit_mutex;
    std::mutex print_mutex;

    const auto report_progress = [&](const char* phase_name, bool force = false) {
        const size_t cur = done_count.load(std::memory_order_relaxed);
        const u32 pct =
            total_shaders > 0
                ? static_cast<u32>(std::min<size_t>(100, (cur * 100) / total_shaders))
                : 100u;
        if (!force && pct <= last_reported_pct.load(std::memory_order_relaxed)) {
            return;
        }
        std::scoped_lock lk{print_mutex};
        if (!force && pct <= last_reported_pct.load(std::memory_order_relaxed)) {
            return;
        }
        last_reported_pct.store(pct, std::memory_order_relaxed);
        std::printf(
            "[BB-BAKE] PROGRESS pct=%u done=%zu total=%zu spv_built=%zu cached=%zu pso=%zu "
            "phase=%s\n",
            pct, cur, total_shaders, spv_built.load(std::memory_order_relaxed),
            spv_cached.load(std::memory_order_relaxed),
            cs_pso_built.load(std::memory_order_relaxed) +
                gfx_pso_built.load(std::memory_order_relaxed),
            phase_name);
        std::fflush(stdout);
    };

    const auto compile_batch = [&](const std::vector<ParsedOrbShader>& list, const char* phase_name,
                                   bool is_vs_phase) {
        std::atomic<size_t> next_idx{0};
        auto worker_fn = [&] {
            Shader::Pools local_pools;
            auto env = std::make_unique<SyntheticStageEnv>();
            for (;;) {
                const size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                if (idx >= list.size()) {
                    break;
                }
                const auto& sh = list[idx];
                bool already_have = false;
                {
                    std::scoped_lock lk{commit_mutex};
                    already_have = program_cache.contains(sh.hash);
                }
                if (already_have) {
                    spv_cached.fetch_add(1, std::memory_order_relaxed);
                    done_count.fetch_add(1, std::memory_order_relaxed);
                    report_progress(phase_name);
                    continue;
                }

                env->Build(sh);
                const ParsedOrbShader* paired_vs = nullptr;
                Shader::Backend::Bindings start_bnd{};
                if (sh.container_stage == 2) {
                    if (const auto vit = ps_to_vs_pair.find(sh.hash); vit != ps_to_vs_pair.end()) {
                        if (const auto v_idx_it = vs_by_hash.find(vit->second);
                            v_idx_it != vs_by_hash.end()) {
                            paired_vs = &vs_list[v_idx_it->second];
                        }
                    }
                } else if (is_vs_phase) {
                    if (const auto pit = vs_to_ps_pair.find(sh.hash); pit != vs_to_ps_pair.end()) {
                        std::scoped_lock lk{commit_mutex};
                        if (const auto p_found = program_cache.find(pit->second);
                            p_found != program_cache.end()) {
                            p_found->second->info.AddBindings(start_bnd);
                        }
                    }
                }

                const HwStage hw_st = (sh.container_stage == 4)   ? HwStage::Compute
                                      : (sh.container_stage == 2) ? HwStage::Fragment
                                                                  : HwStage::Vertex;
                const SwStage sw_st = (sh.container_stage == 4)   ? SwStage::Compute
                                      : (sh.container_stage == 2) ? SwStage::Fragment
                                                                  : SwStage::Vertex;

                Shader::ShaderParams params{
                    .user_data = env->user_data,
                    .code = sh.code,
                    .hash = sh.hash,
                };
                auto rt_info = BuildSyntheticRuntimeInfo(sh, instance, paired_vs);
                Shader::Info info(hw_st, sw_st, params);
                auto run_bnd = start_bnd;
                std::vector<u32> raw_spv;
                std::vector<u32> opt_spv;
                vk::ShaderModule mod = CompileModuleRaw(
                    local_pools, info, rt_info, sh.code, 0, run_bnd, &raw_spv,
                    Shader::Backend::SPIRV::SpirvOptEnabled() ? &opt_spv : nullptr);
                const bool ok = static_cast<bool>(mod);

                if (ok) {
                    auto spec = Shader::StageSpecialization(info, rt_info, profile, start_bnd);

                    // Pre-bake secondary HDR/LDR attachment specialization (`_1`) for single-MRT
                    // full-screen passes (SSR, Deferred PBR, PostFX) that cannot skip asynchronously.
                    vk::ShaderModule mod1{};
                    Shader::StageSpecialization spec1{};
                    Shader::Info info1(hw_st, sw_st, params);
                    std::vector<u32> raw_spv1, opt_spv1;
                    const bool want_alt_fs =
                        (sh.container_stage == 2) && ((sh.ps_col_fmt & ~0xFu) == 0u) &&
                        ((sh.ps_col_fmt & 0xFu) != 0u) &&
                        (sh.bundle == "gxrenderershader" || sh.bundle == "gxposteffect");
                    if (want_alt_fs) {
                        auto rt_alt = rt_info;
                        if (sh.bundle == "gxrenderershader") {
                            // Dual-HDR MRT0+MRT1 bound (SSR & Deferred lighting passes where cb1 is bound with export=0)
                            auto& cb1 = rt_alt.hw.fs.color_buffers[1];
                            cb1.data_format = Gcn::DataFormat::Format16_16_16_16;
                            cb1.num_format = Gcn::NumberFormat::Float;
                            cb1.export_format = Gcn::ShaderExportFormat::Zero;
                            cb1.swizzle = Gcn::IdentityMapping;
                        } else {
                            // LDR RGBA8 Unorm post-FX pass
                            auto& cb0 = rt_alt.hw.fs.color_buffers[0];
                            cb0.data_format = Gcn::DataFormat::Format8_8_8_8;
                            cb0.num_format = Gcn::NumberFormat::Unorm;
                            cb0.swizzle = Gcn::IdentityMapping;
                        }
                        auto run_bnd1 = start_bnd;
                        mod1 = CompileModuleRaw(
                            local_pools, info1, rt_alt, sh.code, 1, run_bnd1, &raw_spv1,
                            Shader::Backend::SPIRV::SpirvOptEnabled() ? &opt_spv1 : nullptr);
                        if (mod1) {
                            spec1 = Shader::StageSpecialization(info1, rt_alt, profile, start_bnd);
                        }
                    }

                    std::unique_ptr<ComputePipeline> built_cs;
                    ComputePipelineKey cs_key{};
                    ComputePipeline::SerializationSupport cs_sdata{};
                    if (sh.container_stage == 4) {
                        cs_key.value = HashCombine(sh.hash, 0);
                        bool have_cs = false;
                        {
                            std::scoped_lock lk{commit_mutex};
                            have_cs = compute_pipelines.contains(cs_key);
                        }
                        if (!have_cs) {
                            built_cs = std::make_unique<ComputePipeline>(
                                instance, scheduler, desc_heap, profile, *pipeline_cache, cs_key,
                                info, mod, cs_sdata, false);
                        }
                    }

                    {
                        std::scoped_lock lk{commit_mutex};
                        if (!program_cache.contains(sh.hash)) {
                            auto new_prog = std::make_unique<Program>(hw_st, sw_st, params);
                            new_prog->info = std::move(info);
                            spec.info = &new_prog->info;
                            if (!raw_spv.empty()) {
                                PipelineStore::SaveShaderBinary(std::move(raw_spv), sh.hash, 0);
                            }
                            if (!opt_spv.empty()) {
                                PipelineStore::SaveOptShaderBinary(std::move(opt_spv), sh.hash, 0);
                            }
                            const u64 perm_hash = HashCombine(sh.hash, 0);
                            PipelineStore::SaveShaderMeta(new_prog->info, spec, perm_hash, 0);
                            new_prog->AddPermut(mod, std::move(spec));
                            if (mod1) {
                                spec1.info = &new_prog->info;
                                if (!raw_spv1.empty()) {
                                    PipelineStore::SaveShaderBinary(std::move(raw_spv1), sh.hash, 1);
                                }
                                if (!opt_spv1.empty()) {
                                    PipelineStore::SaveOptShaderBinary(std::move(opt_spv1), sh.hash, 1);
                                }
                                const u64 perm_hash1 = HashCombine(sh.hash, 1);
                                PipelineStore::SaveShaderMeta(info1, spec1, perm_hash1, 1);
                                new_prog->AddPermut(mod1, std::move(spec1));
                            }
                            new_prog->info_template = std::make_unique<Shader::Info>(new_prog->info);
                            program_cache.emplace(sh.hash, std::move(new_prog));
                            spv_built.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            instance.GetDevice().destroyShaderModule(mod);
                            if (mod1) {
                                instance.GetDevice().destroyShaderModule(mod1);
                            }
                        }
                        if (built_cs && !compute_pipelines.contains(cs_key)) {
                            PipelineStore::SaveCompute(cs_key, cs_sdata);
                            compute_pipelines.emplace(cs_key, std::move(built_cs));
                            cs_pso_built.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
                done_count.fetch_add(1, std::memory_order_relaxed);
                report_progress(phase_name);
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        for (u32 t = 0; t < num_threads; ++t) {
            workers.emplace_back(worker_fn);
        }
        for (auto& w : workers) {
            w.join();
        }
    };

    // Phase 1: Compute Shaders (all 38 CS -> SPIR-V + ComputePipeline VkPipeline)
    compile_batch(cs_list, "compute", false);
    // Phase 2: Pixel Shaders (all 1,346 PS -> SPIR-V, establishing exact descriptor offsets)
    compile_batch(ps_list, "fragment", false);
    // Phase 3: Vertex Shaders (all VS -> SPIR-V using paired PS descriptor offsets)
    compile_batch(vs_list, "vertex", true);

    // Phase 4: Pre-warm driver VkPipelineCache with paired VS+PS GraphicsPipelines for core/decal/VFX passes.
    if (instance.IsVertexInputDynamicState() && !pass_pairs.empty()) {
        std::atomic<size_t> next_pair{0};
        auto pso_worker = [&] {
            for (;;) {
                const size_t idx = next_pair.fetch_add(1, std::memory_order_relaxed);
                if (idx >= pass_pairs.size()) {
                    break;
                }
                const auto& pair_entry = pass_pairs[idx];
                const u64 v_hash = pair_entry.v_hash;
                const u64 p_hash = pair_entry.p_hash;
                const Program* vp = nullptr;
                const Program* pp = nullptr;
                {
                    std::scoped_lock lk{commit_mutex};
                    const auto vit = program_cache.find(v_hash);
                    const auto pit = program_cache.find(p_hash);
                    if (vit == program_cache.end() || pit == program_cache.end() ||
                        vit->second->modules.empty() || pit->second->modules.empty()) {
                        continue;
                    }
                    vp = vit->second.get();
                    pp = pit->second.get();
                }
                // Verify VS descriptor start matches PS descriptor end before linking VkPipeline.
                Shader::Backend::Bindings check_bnd{};
                pp->info.AddBindings(check_bnd);
                if (!(vp->modules[0].spec.start == check_bnd)) {
                    continue;
                }

                const bool is_gui = (pair_entry.bundle == "gxgui");
                const bool is_fullscreen_strip = (!vp->info.has_fetch_shader && !is_gui);
                GraphicsPipelineKey key{};
                key.z_format = Gcn::DepthBuffer::ZFormat::Z32Float;
                key.stencil_format = (pp->info.mrt_mask == 0)
                                         ? Gcn::DepthBuffer::StencilFormat::Invalid
                                         : Gcn::DepthBuffer::StencilFormat::Stencil8;
                key.depth_clamp_enable = 1;
                key.depth_clip_enable = 1;
                key.clip_space = is_gui ? Gcn::ClipSpace::MinusWToW : Gcn::ClipSpace::ZeroToW;
                key.prim_type = is_fullscreen_strip ? Gcn::PrimitiveType::TriangleStrip
                                                    : Gcn::PrimitiveType::TriangleList;
                key.polygon_mode = Gcn::PolygonMode::Fill;
                key.depth_samples = 1;
                key.num_samples = 1;
                key.mrt_mask = pp->info.mrt_mask;
                key.num_color_attachments = std::bit_width(key.mrt_mask);
                key.cb_shader_mask.raw =
                    (key.num_color_attachments >= 8)
                        ? 0xFFFFFFFFu
                        : ((1u << (key.num_color_attachments * 4u)) - 1u);
                key.stage_hashes[static_cast<u32>(SwStage::Fragment)] = HashCombine(p_hash, 0);
                key.stage_hashes[static_cast<u32>(SwStage::Vertex)] = HashCombine(v_hash, 0);
                for (u32 cb = 0; cb < key.num_color_attachments; ++cb) {
                    if ((key.mrt_mask & (1u << cb)) == 0) {
                        continue;
                    }
                    key.color_buffers[cb] =
                        pp->modules[0].spec.runtime_info.hw.fs.color_buffers[cb];
                    key.write_masks[cb] = vk::ColorComponentFlagBits::eR |
                                          vk::ColorComponentFlagBits::eG |
                                          vk::ColorComponentFlagBits::eB |
                                          vk::ColorComponentFlagBits::eA;
                    key.color_samples[cb] = 1;
                }
                {
                    std::shared_lock gr_lk{graphics_pipelines_mutex};
                    if (graphics_pipelines.contains(key)) {
                        continue;
                    }
                }
                std::array<const Shader::Info*, MaxShaderStages> infos{};
                std::array<Shader::RuntimeInfo, MaxShaderStages> rts{};
                std::array<vk::ShaderModule, MaxShaderStages> mods{};
                infos[static_cast<u32>(SwStage::Fragment)] = &pp->info;
                infos[static_cast<u32>(SwStage::Vertex)] = &vp->info;
                rts[static_cast<u32>(SwStage::Fragment)] = pp->modules[0].spec.runtime_info;
                rts[static_cast<u32>(SwStage::Vertex)] = vp->modules[0].spec.runtime_info;
                mods[static_cast<u32>(SwStage::Fragment)] = pp->modules[0].module;
                mods[static_cast<u32>(SwStage::Vertex)] = vp->modules[0].module;

                GraphicsPipeline::SerializationSupport sdata{};
                auto built = std::make_unique<GraphicsPipeline>(
                    instance, scheduler, desc_heap, profile, key, *pipeline_cache, infos, rts,
                    vp->modules[0].spec.fetch_shader_data, mods, sdata, false);
                const u64 pso_hash = std::hash<GraphicsPipelineKey>{}(key);
                {
                    std::scoped_lock lk{commit_mutex};
                    std::unique_lock gr_lk{graphics_pipelines_mutex};
                    if (!graphics_pipelines.contains(key)) {
                        PipelineStore::SaveGraphics(key, pso_hash, sdata);
                        graphics_pipelines.emplace(key, std::move(built));
                        gfx_pso_built.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        for (u32 t = 0; t < num_threads; ++t) {
            workers.emplace_back(pso_worker);
        }
        for (auto& w : workers) {
            w.join();
        }
    }

    report_progress("saving", true);
    SaveProfile();
    SaveDriverPipelineCache();
    BbCache::BlobStore::Instance().Close();

    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    const double cache_mib = ComputeDirectorySizeMiB(cache_root);
    std::printf(
        "[BB-BAKE] DONE elapsed_s=%.2f programs=%zu spv_built=%zu spv_cached=%zu cs_pso=%zu "
        "gfx_pso=%zu cache_mib=%.2f\n",
        elapsed_s, program_cache.size(), spv_built.load(), spv_cached.load(),
        compute_pipelines.size(), graphics_pipelines.size(), cache_mib);
    std::fflush(stdout);
    return 0;
}

struct PipelineCache::AutoBakeReadyStage {
    HwStage hw_stage{};
    SwStage sw_stage{};
    u64 hash = 0;
    Shader::Info info{};
    Shader::StageSpecialization spec{};
    vk::ShaderModule mod{};
    bool has_cs = false;
    ComputePipelineKey cs_key{};
    std::unique_ptr<ComputePipeline> built_cs;
    ComputePipeline::SerializationSupport cs_sdata{};
};

namespace {
void SetPoliteAutoBakeThreadPriority() {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
#else
    pthread_setname_np(pthread_self(), "bb:AutoBake");
    setpriority(PRIO_PROCESS, 0, 19);
    struct sched_param sp{};
    pthread_setschedparam(pthread_self(), SCHED_IDLE, &sp);
#endif
}

} // namespace

void PipelineCache::StartBackgroundAutoBake() {
    if (const char* env = std::getenv("BB_AUTO_BAKE"); env && env[0] == '0') {
        return;
    }
    constexpr size_t kFullGameShaderThreshold = 5500;
    if (program_cache.size() >= kFullGameShaderThreshold) {
        const u32 n = static_cast<u32>(program_cache.size());
        BbStats::shader_autobake_done.store(n, std::memory_order_relaxed);
        BbStats::shader_autobake_total.store(n, std::memory_order_relaxed);
        BbStats::shader_autobake_active.store(false, std::memory_order_relaxed);
        std::printf("[bb_vulkan] Background Auto-Baker: all %zu game shaders already warm (0%% CPU)\n",
                    program_cache.size());
        return;
    }

    const char* gdir_env = std::getenv("BB_GAME_DIR");
    if (!gdir_env || !*gdir_env) {
        return;
    }
    const std::filesystem::path game_root_dir{
        std::u8string_view(reinterpret_cast<const char8_t*>(gdir_env))};
    const std::filesystem::path shader_dir = game_root_dir / "dvdroot_ps4" / "shader";
    std::error_code ec;
    if (!std::filesystem::exists(shader_dir, ec)) {
        return;
    }

    std::unordered_set<u64> warm_hashes;
    std::unordered_map<u64, Shader::Backend::Bindings> ps_bindings_map;
    warm_hashes.reserve(program_cache.size() + 512);
    for (const auto& [h, prog] : program_cache) {
        if (!prog) {
            continue;
        }
        warm_hashes.insert(h);
        if (prog->info.sw_stage == SwStage::Fragment) {
            Shader::Backend::Bindings b{};
            prog->info.AddBindings(b);
            ps_bindings_map.emplace(h, b);
        }
    }

    autobake_stop.store(false, std::memory_order_relaxed);
    BbStats::shader_autobake_active.store(true, std::memory_order_relaxed);

    autobake_thread = std::thread([this, game_root_dir, shader_dir, warm = std::move(warm_hashes),
                                   ps_bnd_init = std::move(ps_bindings_map)]() mutable {
        SetPoliteAutoBakeThreadPriority();
        const auto t0 = std::chrono::steady_clock::now();

        static constexpr const char* kBundles[] = {
            "gxshader.shaderbnd.dcx",         "gxgui.shaderbnd.dcx",
            "gxposteffect.shaderbnd.dcx",     "gxrenderershader.shaderbnd.dcx",
            "gxdecal.shaderbnd.dcx",          "gxffxshader.shaderbnd.dcx",
            "gxflvershader.shaderbnd.dcx",
        };

        std::vector<ParsedOrbShader> cs_list, ps_list, vs_list;
        std::unordered_map<u64, size_t> cs_seen, ps_seen, vs_seen;
        std::unordered_map<std::string, u64> stem_vs_hash, stem_ps_hash;

        const auto patch_root = ResolveSiblingPatchRoot(game_root_dir);
        const auto patch_shader_dir =
            patch_root.empty() ? std::filesystem::path{} : (patch_root / "dvdroot_ps4" / "shader");
        for (const char* bname : kBundles) {
            if (autobake_stop.load(std::memory_order_relaxed)) {
                BbStats::shader_autobake_active.store(false, std::memory_order_relaxed);
                return;
            }
            std::error_code bec;
            auto bpath = shader_dir / bname;
            if (!patch_shader_dir.empty() &&
                std::filesystem::exists(patch_shader_dir / bname, bec)) {
                bpath = patch_shader_dir / bname;
            }
            if (!std::filesystem::exists(bpath, bec)) {
                continue;
            }
            std::vector<u8> bnd_bytes;
            if (!LoadDcxFile(bpath, bnd_bytes)) {
                continue;
            }
            std::string short_bundle = bname;
            if (const auto dot = short_bundle.find('.'); dot != std::string::npos) {
                short_bundle.resize(dot);
            }
            std::vector<BndShaderEntry> entries;
            if (!ParseBnd4Entries(short_bundle, bnd_bytes, entries)) {
                continue;
            }
            for (const auto& e : entries) {
                ParsedOrbShader parsed{};
                if (!ParseOrbContainer(e, parsed)) {
                    continue;
                }
                const std::string qstem = short_bundle + ":" + parsed.stem;
                if (parsed.container_stage == 1) {
                    stem_vs_hash[qstem] = parsed.hash;
                    if (!vs_seen.contains(parsed.hash)) {
                        vs_seen[parsed.hash] = vs_list.size();
                        vs_list.push_back(std::move(parsed));
                    }
                } else if (parsed.container_stage == 2) {
                    stem_ps_hash[qstem] = parsed.hash;
                    if (!ps_seen.contains(parsed.hash)) {
                        ps_seen[parsed.hash] = ps_list.size();
                        ps_list.push_back(std::move(parsed));
                    }
                } else if (parsed.container_stage == 4) {
                    if (!cs_seen.contains(parsed.hash)) {
                        cs_seen[parsed.hash] = cs_list.size();
                        cs_list.push_back(std::move(parsed));
                    }
                }
            }
        }
        ScanEmbeddedEbootShaders(game_root_dir, cs_list, ps_list, vs_list, cs_seen, ps_seen,
                                 vs_seen);

        const size_t total_all = cs_list.size() + ps_list.size() + vs_list.size();
        if (total_all == 0) {
            BbStats::shader_autobake_active.store(false, std::memory_order_relaxed);
            return;
        }

        std::unordered_map<u64, u64> ps_to_vs_pair, vs_to_ps_pair;
        for (const auto& [qstem, v_hash] : stem_vs_hash) {
            if (const auto pit = stem_ps_hash.find(qstem); pit != stem_ps_hash.end()) {
                ps_to_vs_pair.try_emplace(pit->second, v_hash);
                vs_to_ps_pair.try_emplace(v_hash, pit->second);
            }
        }

        size_t already_warm_count = 0;
        for (const auto& s : cs_list) already_warm_count += warm.contains(s.hash) ? 1 : 0;
        for (const auto& s : ps_list) already_warm_count += warm.contains(s.hash) ? 1 : 0;
        for (const auto& s : vs_list) already_warm_count += warm.contains(s.hash) ? 1 : 0;

        BbStats::shader_autobake_total.store(static_cast<u32>(total_all), std::memory_order_relaxed);
        BbStats::shader_autobake_done.store(static_cast<u32>(already_warm_count),
                                            std::memory_order_relaxed);

        if (already_warm_count >= total_all) {
            BbStats::shader_autobake_active.store(false, std::memory_order_relaxed);
            return;
        }

        const u32 hw_threads = std::max(1u, std::thread::hardware_concurrency());
        const u32 polite_workers = std::clamp(hw_threads / 6u, 1u, 3u);
        std::printf("[bb_vulkan] Background Auto-Baker started politely (%zu / %zu cached, "
                    "compiling %zu remaining on %u idle-priority threads)\n",
                    already_warm_count, total_all, total_all - already_warm_count, polite_workers);
        std::fflush(stdout);

        std::mutex ps_bnd_mutex;
        std::atomic<u32> done_all{static_cast<u32>(already_warm_count)};
        std::atomic<u32> last_logged_decile{
            static_cast<u32>((already_warm_count * 10) / std::max<size_t>(1, total_all))};

        auto run_polite_phase = [&](const std::vector<ParsedOrbShader>& list, bool is_vs_phase) {
            std::atomic<size_t> next_idx{0};
            auto worker = [&] {
                SetPoliteAutoBakeThreadPriority();
                Shader::Pools local_pools;
                auto env = std::make_unique<SyntheticStageEnv>();
                for (;;) {
                    if (autobake_stop.load(std::memory_order_relaxed)) {
                        return;
                    }
                    const size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= list.size()) {
                        return;
                    }
                    const auto& sh = list[idx];
                    if (warm.contains(sh.hash)) {
                        continue;
                    }

                    env->Build(sh);
                    const ParsedOrbShader* paired_vs = nullptr;
                    Shader::Backend::Bindings start_bnd{};
                    if (sh.container_stage == 2) {
                        if (const auto vit = ps_to_vs_pair.find(sh.hash); vit != ps_to_vs_pair.end()) {
                            if (const auto v_idx = vs_seen.find(vit->second); v_idx != vs_seen.end()) {
                                paired_vs = &vs_list[v_idx->second];
                            }
                        }
                    } else if (is_vs_phase) {
                        if (const auto pit = vs_to_ps_pair.find(sh.hash); pit != vs_to_ps_pair.end()) {
                            std::scoped_lock lk{ps_bnd_mutex};
                            if (const auto bit = ps_bnd_init.find(pit->second);
                                bit != ps_bnd_init.end()) {
                                start_bnd = bit->second;
                            }
                        }
                    }

                    const HwStage hw_st = (sh.container_stage == 4)   ? HwStage::Compute
                                          : (sh.container_stage == 2) ? HwStage::Fragment
                                                                      : HwStage::Vertex;
                    const SwStage sw_st = (sh.container_stage == 4)   ? SwStage::Compute
                                          : (sh.container_stage == 2) ? SwStage::Fragment
                                                                      : SwStage::Vertex;
                    Shader::ShaderParams params{
                        .user_data = env->user_data,
                        .code = sh.code,
                        .hash = sh.hash,
                    };
                    auto rt_info = BuildSyntheticRuntimeInfo(sh, instance, paired_vs);
                    Shader::Info info(hw_st, sw_st, params);
                    auto run_bnd = start_bnd;
                    std::vector<u32> raw_spv, opt_spv;
                    vk::ShaderModule mod = CompileModuleRaw(
                        local_pools, info, rt_info, sh.code, 0, run_bnd, &raw_spv,
                        Shader::Backend::SPIRV::SpirvOptEnabled() ? &opt_spv : nullptr);

                    if (mod) {
                        if (sh.container_stage == 2) {
                            Shader::Backend::Bindings pb{};
                            info.AddBindings(pb);
                            std::scoped_lock lk{ps_bnd_mutex};
                            ps_bnd_init.emplace(sh.hash, pb);
                        }
                        auto spec = Shader::StageSpecialization(info, rt_info, profile, start_bnd);
                        auto item = std::make_shared<AutoBakeReadyStage>();
                        item->hw_stage = hw_st;
                        item->sw_stage = sw_st;
                        item->hash = sh.hash;
                        item->info = std::move(info);
                        item->spec = std::move(spec);
                        item->mod = mod;

                        if (sh.container_stage == 4) {
                            item->has_cs = true;
                            item->cs_key.value = HashCombine(sh.hash, 0);
                            item->built_cs = std::make_unique<ComputePipeline>(
                                instance, scheduler, desc_heap, profile, *pipeline_cache,
                                item->cs_key, item->info, mod, item->cs_sdata, false);
                        }

                        if (!raw_spv.empty()) {
                            PipelineStore::SaveShaderBinary(std::move(raw_spv), sh.hash, 0);
                        }
                        if (!opt_spv.empty()) {
                            PipelineStore::SaveOptShaderBinary(std::move(opt_spv), sh.hash, 0);
                        }
                        const u64 perm_hash = HashCombine(sh.hash, 0);
                        PipelineStore::SaveShaderMeta(item->info, item->spec, perm_hash, 0);
                        if (item->has_cs && item->built_cs) {
                            PipelineStore::SaveCompute(item->cs_key, item->cs_sdata);
                        }

                        {
                            std::scoped_lock qlk{autobake_mutex};
                            autobake_ready.push_back(std::move(item));
                            autobake_has_ready.store(true, std::memory_order_release);
                        }
                    }

                    const u32 cur = done_all.fetch_add(1, std::memory_order_relaxed) + 1;
                    BbStats::shader_autobake_done.store(cur, std::memory_order_relaxed);
                    const u32 dec = static_cast<u32>((size_t(cur) * 10) / total_all);
                    u32 prev_dec = last_logged_decile.load(std::memory_order_relaxed);
                    if (dec > prev_dec && dec <= 10 &&
                        last_logged_decile.compare_exchange_strong(prev_dec, dec)) {
                        std::printf("[bb_vulkan] Background Auto-Baker: %u%% (%u / %zu)\n",
                                    dec * 10, cur, total_all);
                        std::fflush(stdout);
                    }
                }
            };

            std::vector<std::thread> pool;
            pool.reserve(polite_workers);
            for (u32 t = 0; t < polite_workers; ++t) {
                pool.emplace_back(worker);
            }
            for (auto& w : pool) {
                w.join();
            }
        };

        run_polite_phase(cs_list, false);
        run_polite_phase(ps_list, false);
        run_polite_phase(vs_list, true);

        driver_cache_dirty.store(true, std::memory_order_relaxed);
        BbStats::shader_autobake_active.store(false, std::memory_order_relaxed);
        if (!autobake_stop.load(std::memory_order_relaxed)) {
            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("[bb_vulkan] Background Auto-Baker: 100%% COMPLETE (%zu shaders warm in %.1f s)\n",
                        total_all, secs);
            std::fflush(stdout);
        }
    });
}

void PipelineCache::StopBackgroundAutoBake() {
    autobake_stop.store(true, std::memory_order_relaxed);
    if (autobake_thread.joinable()) {
        autobake_thread.join();
    }
    std::scoped_lock qlk{autobake_mutex};
    for (auto& item : autobake_ready) {
        if (item && item->mod) {
            instance.GetDevice().destroyShaderModule(item->mod);
        }
    }
    autobake_ready.clear();
    autobake_has_ready.store(false, std::memory_order_relaxed);
    BbStats::shader_autobake_active.store(false, std::memory_order_relaxed);
}

void PipelineCache::DrainAutoBakeReadyStages() {
    std::vector<std::shared_ptr<AutoBakeReadyStage>> batch;
    {
        std::scoped_lock qlk{autobake_mutex};
        if (autobake_ready.empty()) {
            autobake_has_ready.store(false, std::memory_order_relaxed);
            return;
        }
        const size_t take = std::min<size_t>(autobake_ready.size(), 16);
        batch.reserve(take);
        for (size_t i = 0; i < take; ++i) {
            batch.push_back(std::move(autobake_ready.back()));
            autobake_ready.pop_back();
        }
        if (autobake_ready.empty()) {
            autobake_has_ready.store(false, std::memory_order_relaxed);
        }
    }

    bool any_committed = false;
    for (auto& item : batch) {
        if (!item || !item->mod) {
            continue;
        }
        if (program_cache.contains(item->hash)) {
            instance.GetDevice().destroyShaderModule(item->mod);
            item->mod = nullptr;
            continue;
        }
        auto new_prog = std::make_unique<Program>();
        new_prog->info = std::move(item->info);
        item->spec.info = &new_prog->info;
        new_prog->AddPermut(item->mod, std::move(item->spec));
        new_prog->info_template = std::make_unique<Shader::Info>(new_prog->info);
        Program* raw_ptr = new_prog.get();
        {
            std::unique_lock lk{programs_mutex};
            program_cache.emplace(item->hash, std::move(new_prog));
        }
        DirectShaders::NoteProgramCreated(item->hash, raw_ptr);
        if (item->has_cs && item->built_cs && !compute_pipelines.contains(item->cs_key)) {
            compute_pipelines.emplace(item->cs_key, std::move(item->built_cs));
        }
        any_committed = true;
    }
    if (any_committed) {
        NoteDriverPipelineCacheDirty();
    }
}

} // namespace Vulkan
