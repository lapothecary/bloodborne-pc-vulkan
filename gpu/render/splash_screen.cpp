// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/splash_screen.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>
#include <zlib.h>

#include "bbport_toggles.h"
#include "render/scheduler.h"
#include "vulkan/instance.h"
#include "vulkan/swapchain.h"

namespace Vulkan {

namespace {

constexpr vk::ImageSubresourceRange kColorRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
constexpr u32 kProbeWidth = 32;
constexpr u32 kProbeHeight = 18;
constexpr VkDeviceSize kProbeBytes = VkDeviceSize(kProbeWidth) * kProbeHeight * 4u;

u32 ReadBe32(const u8* p) noexcept {
    return (u32(p[0]) << 24) | (u32(p[1]) << 16) | (u32(p[2]) << 8) | u32(p[3]);
}

u16 ReadLe16(const u8* p) noexcept {
    return u16(p[0]) | (u16(p[1]) << 8);
}

u32 ReadLe32(const u8* p) noexcept {
    return u32(p[0]) | (u32(p[1]) << 8) | (u32(p[2]) << 16) | (u32(p[3]) << 24);
}

u8 PaethPredictor(u8 a, u8 b, u8 c) noexcept {
    const int p = int(a) + int(b) - int(c);
    const int pa = std::abs(p - int(a));
    const int pb = std::abs(p - int(b));
    const int pc = std::abs(p - int(c));
    if (pa <= pb && pa <= pc) {
        return a;
    }
    return pb <= pc ? b : c;
}

bool MatchUtf16LeAscii(const u8* data, size_t len, size_t off, std::string_view ascii) noexcept {
    for (size_t i = 0; i < ascii.size(); ++i) {
        if (off + i * 2 + 1 >= len) {
            return false;
        }
        if (data[off + i * 2] != u8(ascii[i]) || data[off + i * 2 + 1] != 0) {
            return false;
        }
    }
    const size_t term = off + ascii.size() * 2;
    return term + 1 < len && data[term] == 0 && data[term + 1] == 0;
}

/// Decodes a single 4x4 BC1 (DXT1) block directly into a 2D RGBA8 buffer at pixel `(px0, py0)`.
void DecodeBc1Block(const u8* blk, u8* dst_rgba, u32 stride_px, u32 px0, u32 py0) noexcept {
    const u16 c0 = ReadLe16(blk + 0);
    const u16 c1 = ReadLe16(blk + 2);
    const u32 bits = ReadLe32(blk + 4);

    const u32 r0 = ((c0 >> 11) & 31u) * 255u / 31u;
    const u32 g0 = ((c0 >> 5) & 63u) * 255u / 63u;
    const u32 b0 = (c0 & 31u) * 255u / 31u;

    const u32 r1 = ((c1 >> 11) & 31u) * 255u / 31u;
    const u32 g1 = ((c1 >> 5) & 63u) * 255u / 63u;
    const u32 b1 = (c1 & 31u) * 255u / 31u;

    u8 pal[4][4];
    pal[0][0] = u8(r0);
    pal[0][1] = u8(g0);
    pal[0][2] = u8(b0);
    pal[0][3] = 255;
    pal[1][0] = u8(r1);
    pal[1][1] = u8(g1);
    pal[1][2] = u8(b1);
    pal[1][3] = 255;
    if (c0 > c1) {
        pal[2][0] = u8((2u * r0 + r1) / 3u);
        pal[2][1] = u8((2u * g0 + g1) / 3u);
        pal[2][2] = u8((2u * b0 + b1) / 3u);
        pal[2][3] = 255;
        pal[3][0] = u8((r0 + 2u * r1) / 3u);
        pal[3][1] = u8((g0 + 2u * g1) / 3u);
        pal[3][2] = u8((b0 + 2u * b1) / 3u);
        pal[3][3] = 255;
    } else {
        pal[2][0] = u8((r0 + r1) / 2u);
        pal[2][1] = u8((g0 + g1) / 2u);
        pal[2][2] = u8((b0 + b1) / 2u);
        pal[2][3] = 255;
        pal[3][0] = 0;
        pal[3][1] = 0;
        pal[3][2] = 0;
        pal[3][3] = 255;
    }

    for (u32 y = 0; y < 4; ++y) {
        u8* row = dst_rgba + (size_t(py0 + y) * stride_px + px0) * 4u;
        for (u32 x = 0; x < 4; ++x) {
            const u32 ci = (bits >> (2u * (y * 4u + x))) & 3u;
            row[x * 4 + 0] = pal[ci][0];
            row[x * 4 + 1] = pal[ci][1];
            row[x * 4 + 2] = pal[ci][2];
            row[x * 4 + 3] = 255;
        }
    }
}

/// Decodes Bloodborne's authentic main-menu artwork (`MENU_Title_00001`, fallback `MENU_Title_00004`)
/// directly from `<game>/dvdroot_ps4/menu/title.tpf.dcx` and reassembles the exact 1920x1080
/// Scaleform `title.gfx` layout (top 1920x1024 + two 90-degree rotated 56px bottom strips packed at
/// columns 1924..1979 and 1985..2040 of the 2048x1024 BC1 texture).
bool DecodeTitleTpfDcx(const std::filesystem::path& path, std::vector<u8>& out_rgba, u32& out_w,
                       u32& out_h, std::string& out_label) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return false;
    }
    const auto fsize = f.tellg();
    if (fsize < 0x50 || fsize > 32 * 1024 * 1024) {
        return false;
    }
    std::vector<u8> raw(static_cast<size_t>(fsize));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(raw.data()), fsize)) {
        return false;
    }

    // Parse DCX\0 / DFLT header
    if (std::memcmp(raw.data(), "DCX\0", 4) != 0) {
        return false;
    }
    size_t dcs_off = 0;
    size_t dca_off = 0;
    for (size_t i = 4; i + 12 <= std::min<size_t>(raw.size(), 0x100); i += 4) {
        if (dcs_off == 0 && std::memcmp(raw.data() + i, "DCS\0", 4) == 0) {
            dcs_off = i;
        } else if (dca_off == 0 && std::memcmp(raw.data() + i, "DCA\0", 4) == 0) {
            dca_off = i;
        }
    }
    if (dcs_off == 0 || dca_off == 0) {
        return false;
    }
    const u32 uncomp_size = ReadBe32(raw.data() + dcs_off + 4);
    const u32 comp_size = ReadBe32(raw.data() + dcs_off + 8);
    const size_t data_off = dca_off + ReadBe32(raw.data() + dca_off + 4);
    if (uncomp_size < 0x40 || uncomp_size > 32 * 1024 * 1024 || data_off + comp_size > raw.size()) {
        return false;
    }

    std::vector<u8> tpf(uncomp_size);
    z_stream zs{};
    zs.next_in = raw.data() + data_off;
    zs.avail_in = comp_size;
    zs.next_out = tpf.data();
    zs.avail_out = uncomp_size;
    if (inflateInit(&zs) != Z_OK) {
        return false;
    }
    const int zret = inflate(&zs, Z_FINISH);
    const size_t produced = zs.total_out;
    inflateEnd(&zs);
    if ((zret != Z_STREAM_END && zret != Z_OK) || produced < uncomp_size) {
        return false;
    }

    // Parse PS4 TPF\0 container (platform == 4, 0x24-byte entries)
    if (std::memcmp(tpf.data(), "TPF\0", 4) != 0) {
        return false;
    }
    const u32 count = ReadLe32(tpf.data() + 8);
    const u8 platform = tpf[12];
    if (platform != 4 || count == 0 || count > 256 || 16 + size_t(count) * 0x24u > tpf.size()) {
        return false;
    }

    struct TpfEntry {
        u32 file_off = 0;
        u32 file_size = 0;
        u16 w = 0;
        u16 h = 0;
        u32 dxgi = 0;
        const char* name = nullptr;
    };
    TpfEntry chosen{};
    auto try_pick = [&](std::string_view target_name, const char* label) -> bool {
        for (u32 i = 0; i < count; ++i) {
            const u8* e = tpf.data() + 16 + size_t(i) * 0x24u;
            const u32 file_off = ReadLe32(e + 0);
            const u32 file_size = ReadLe32(e + 4);
            const u16 w = ReadLe16(e + 12);
            const u16 h = ReadLe16(e + 14);
            const u32 name_off = ReadLe32(e + 24);
            const u32 dxgi = ReadLe32(e + 32);
            if (dxgi != 71 || w < 64 || h < 64 || (w & 3u) != 0 || (h & 3u) != 0) {
                continue;
            }
            if (size_t(file_off) + size_t(file_size) > tpf.size()) {
                continue;
            }
            if (MatchUtf16LeAscii(tpf.data(), tpf.size(), name_off, target_name)) {
                chosen = {file_off, file_size, w, h, dxgi, label};
                return true;
            }
        }
        return false;
    };

    if (!try_pick("MENU_Title_00001", "MENU_Title_00001") &&
        !try_pick("MENU_Title_00004", "MENU_Title_00004")) {
        return false;
    }

    const u32 bw = chosen.w / 4u;
    const u32 bh = chosen.h / 4u;
    const u32 tiles_x = (bw + 7u) / 8u;
    const u32 tiles_y = (bh + 7u) / 8u;
    const size_t needed_bytes = size_t(tiles_x) * tiles_y * 64u * 8u;
    if (chosen.file_size < needed_bytes) {
        return false;
    }

    // Untile PS4 Thin_1dThin (8x8 Morton micro-tiles of 8-byte BC1 blocks) and decode to RGBA8
    std::vector<u8> decoded(size_t(chosen.w) * chosen.h * 4u, 0);
    const u8* src_blocks = tpf.data() + chosen.file_off;
    size_t idx = 0;
    for (u32 ty = 0; ty < tiles_y; ++ty) {
        for (u32 tx = 0; tx < tiles_x; ++tx) {
            for (u32 i = 0; i < 64; ++i) {
                const u32 lx = (i & 1u) | ((i & 4u) >> 1) | ((i & 16u) >> 2);
                const u32 ly = ((i & 2u) >> 1) | ((i & 8u) >> 2) | ((i & 32u) >> 3);
                const u32 gx = tx * 8u + lx;
                const u32 gy = ty * 8u + ly;
                if (gx < bw && gy < bh) {
                    DecodeBc1Block(src_blocks + idx, decoded.data(), chosen.w, gx * 4u, gy * 4u);
                }
                idx += 8;
            }
        }
    }

    if (chosen.w == 2048 && chosen.h == 1024) {
        // Exact 3-piece 1920x1080 Scaleform reassembly from title.gfx / title_dlc_eu.gfx
        out_w = 1920;
        out_h = 1080;
        out_rgba.assign(size_t(out_w) * out_h * 4u, 0);
        // Top 1920x1024
        for (u32 y = 0; y < 1024; ++y) {
            std::memcpy(out_rgba.data() + size_t(y) * out_w * 4u,
                        decoded.data() + size_t(y) * chosen.w * 4u, size_t(out_w) * 4u);
        }
        // Bottom 56 rows (y = 1024..1079) from the two 90-degree rotated right-hand strips:
        // Left 1024px:  src_x = 3003 - y (1979..1924), src_y = x (0..1023)
        // Right 896px:  src_x = 3064 - y (2040..1985), src_y = x - 1024 (0..895)
        for (u32 y = 1024; y < 1080; ++y) {
            u8* dst_row = out_rgba.data() + size_t(y) * out_w * 4u;
            const u32 sx_left = 3003u - y;
            for (u32 x = 0; x < 1024; ++x) {
                const u8* sp = decoded.data() + (size_t(x) * chosen.w + sx_left) * 4u;
                std::memcpy(dst_row + size_t(x) * 4u, sp, 4);
            }
            const u32 sx_right = 3064u - y;
            for (u32 x = 1024; x < 1920; ++x) {
                const u8* sp = decoded.data() + (size_t(x - 1024u) * chosen.w + sx_right) * 4u;
                std::memcpy(dst_row + size_t(x) * 4u, sp, 4);
            }
        }
    } else {
        out_w = chosen.w;
        out_h = chosen.h;
        out_rgba = std::move(decoded);
    }

    out_label = std::string("title.tpf.dcx/") + chosen.name;
    return true;
}

/// Self-contained 8-bit non-interlaced PNG decoder (Gray, RGB, GrayAlpha, RGBA) using zlib.
bool DecodePngRgba8(const std::filesystem::path& path, std::vector<u8>& out_rgba, u32& out_w,
                    u32& out_h) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return false;
    }
    const auto fsize = f.tellg();
    if (fsize < 33 || fsize > 32 * 1024 * 1024) {
        return false;
    }
    std::vector<u8> raw(static_cast<size_t>(fsize));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(raw.data()), fsize)) {
        return false;
    }

    static constexpr u8 kPngSig[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    if (std::memcmp(raw.data(), kPngSig, 8) != 0) {
        return false;
    }

    u32 width = 0;
    u32 height = 0;
    u8 bit_depth = 0;
    u8 color_type = 0;
    u8 interlace = 0;
    bool got_ihdr = false;
    std::vector<u8> idat;
    idat.reserve(raw.size());

    size_t pos = 8;
    while (pos + 12 <= raw.size()) {
        const u32 len = ReadBe32(raw.data() + pos);
        const u8* type = raw.data() + pos + 4;
        pos += 8;
        if (pos + len + 4 > raw.size()) {
            return false;
        }
        if (std::memcmp(type, "IHDR", 4) == 0 && len >= 13) {
            width = ReadBe32(raw.data() + pos + 0);
            height = ReadBe32(raw.data() + pos + 4);
            bit_depth = raw[pos + 8];
            color_type = raw[pos + 9];
            const u8 comp = raw[pos + 10];
            const u8 filt = raw[pos + 11];
            interlace = raw[pos + 12];
            if (width == 0 || height == 0 || width > 4096 || height > 4096 || bit_depth != 8 ||
                comp != 0 || filt != 0 || interlace != 0) {
                return false;
            }
            got_ihdr = true;
        } else if (std::memcmp(type, "IDAT", 4) == 0 && len > 0) {
            idat.insert(idat.end(), raw.data() + pos, raw.data() + pos + len);
        } else if (std::memcmp(type, "IEND", 4) == 0) {
            break;
        }
        pos += len + 4;
    }

    if (!got_ihdr || idat.empty()) {
        return false;
    }

    u32 bpp = 0;
    switch (color_type) {
    case 0:
        bpp = 1;
        break;
    case 2:
        bpp = 3;
        break;
    case 4:
        bpp = 2;
        break;
    case 6:
        bpp = 4;
        break;
    default:
        return false;
    }

    const size_t stride = size_t(width) * size_t(bpp);
    const size_t expected_inflated = (stride + 1) * size_t(height);
    std::vector<u8> inflated(expected_inflated);

    z_stream zs{};
    zs.next_in = idat.data();
    zs.avail_in = static_cast<uInt>(idat.size());
    zs.next_out = inflated.data();
    zs.avail_out = static_cast<uInt>(inflated.size());
    if (inflateInit(&zs) != Z_OK) {
        return false;
    }
    const int zret = inflate(&zs, Z_FINISH);
    const size_t produced = zs.total_out;
    inflateEnd(&zs);
    if ((zret != Z_STREAM_END && zret != Z_OK) || produced < expected_inflated) {
        return false;
    }

    std::vector<u8> prev_row(stride, 0);
    std::vector<u8> cur_row(stride, 0);
    out_rgba.resize(size_t(width) * size_t(height) * 4u);

    for (u32 y = 0; y < height; ++y) {
        const u8* src = inflated.data() + size_t(y) * (stride + 1);
        const u8 filter = src[0];
        const u8* row_in = src + 1;

        for (size_t x = 0; x < stride; ++x) {
            const u8 a = x >= bpp ? cur_row[x - bpp] : 0;
            const u8 b = prev_row[x];
            const u8 c = x >= bpp ? prev_row[x - bpp] : 0;
            u8 v = row_in[x];
            switch (filter) {
            case 0:
                break;
            case 1:
                v = static_cast<u8>(v + a);
                break;
            case 2:
                v = static_cast<u8>(v + b);
                break;
            case 3:
                v = static_cast<u8>(v + ((int(a) + int(b)) >> 1));
                break;
            case 4:
                v = static_cast<u8>(v + PaethPredictor(a, b, c));
                break;
            default:
                return false;
            }
            cur_row[x] = v;
        }

        u8* dst = out_rgba.data() + size_t(y) * size_t(width) * 4u;
        if (color_type == 2) {
            for (u32 x = 0; x < width; ++x) {
                dst[x * 4 + 0] = cur_row[x * 3 + 0];
                dst[x * 4 + 1] = cur_row[x * 3 + 1];
                dst[x * 4 + 2] = cur_row[x * 3 + 2];
                dst[x * 4 + 3] = 255;
            }
        } else if (color_type == 6) {
            std::memcpy(dst, cur_row.data(), size_t(width) * 4u);
        } else if (color_type == 0) {
            for (u32 x = 0; x < width; ++x) {
                const u8 g = cur_row[x];
                dst[x * 4 + 0] = g;
                dst[x * 4 + 1] = g;
                dst[x * 4 + 2] = g;
                dst[x * 4 + 3] = 255;
            }
        } else {
            for (u32 x = 0; x < width; ++x) {
                const u8 g = cur_row[x * 2 + 0];
                dst[x * 4 + 0] = g;
                dst[x * 4 + 1] = g;
                dst[x * 4 + 2] = g;
                dst[x * 4 + 3] = cur_row[x * 2 + 1];
            }
        }
        std::swap(prev_row, cur_row);
    }

    out_w = width;
    out_h = height;
    return true;
}

bool LoadGameSplashArtwork(std::vector<u8>& out_rgba, u32& out_w, u32& out_h,
                           std::string& out_label) {
    std::error_code ec;
    const char* env_vars[] = {"BB_GAME_DIR", "BB_SHADER_GAME_DIR"};

    // Priority 1: Authentic main-menu artwork (`dvdroot_ps4/menu/title.tpf.dcx` -> MENU_Title_00001)
    for (const char* ev : env_vars) {
        const char* val = std::getenv(ev);
        if (!val || !*val) {
            continue;
        }
        const std::filesystem::path root{
            std::u8string_view(reinterpret_cast<const char8_t*>(val))};
        const auto tpf_path = root / "dvdroot_ps4/menu/title.tpf.dcx";
        if (std::filesystem::is_regular_file(tpf_path, ec) &&
            DecodeTitleTpfDcx(tpf_path, out_rgba, out_w, out_h, out_label)) {
            return true;
        }
    }

    // Priority 2: PS4 system splash (`sce_sys/pic1.png` / `pic0.png`)
    const char* rel_paths[] = {"sce_sys/pic1.png", "sce_sys/pic0.png"};
    for (const char* ev : env_vars) {
        const char* val = std::getenv(ev);
        if (!val || !*val) {
            continue;
        }
        const std::filesystem::path root{
            std::u8string_view(reinterpret_cast<const char8_t*>(val))};
        for (const char* rel : rel_paths) {
            const auto cand = root / rel;
            if (std::filesystem::is_regular_file(cand, ec) &&
                DecodePngRgba8(cand, out_rgba, out_w, out_h)) {
                out_label = rel;
                return true;
            }
        }
    }
    return false;
}

vk::Rect2D LetterboxSplash(vk::Extent2D src, vk::Extent2D dst) {
    const float src_aspect = float(src.width) / float(src.height);
    const float dst_aspect = float(dst.width) / float(dst.height);
    vk::Extent2D size = dst;
    if (src_aspect > dst_aspect) {
        size.height = std::max(1u, u32(float(dst.width) / src_aspect));
    } else {
        size.width = std::max(1u, u32(float(dst.height) * src_aspect));
    }
    return {{s32(dst.width - size.width) / 2, s32(dst.height - size.height) / 2}, size};
}

} // namespace

SplashScreen::SplashScreen(Instance& instance_, Scheduler& present_scheduler, Swapchain& swapchain)
    : instance{instance_}, scheduler{present_scheduler} {
    if (const char* env = std::getenv("BB_BOOT_SPLASH"); env && env[0] == '0') {
        active = false;
        return;
    }
    LoadAndUpload(present_scheduler);
    CreateProbeResources();
    PresentInitialFrame(present_scheduler, swapchain);
}

SplashScreen::~SplashScreen() {
    DestroyImmediate();
}

bool SplashScreen::LoadAndUpload(Scheduler& present_scheduler) {
    std::vector<u8> rgba;
    u32 w = 0;
    u32 h = 0;
    std::string label;
    if (!LoadGameSplashArtwork(rgba, w, h, label)) {
        return false;
    }

    const VmaAllocator allocator = instance.GetAllocator();
    const VkDeviceSize byte_size = static_cast<VkDeviceSize>(rgba.size());

    const VkBufferCreateInfo stg_ci = vk::BufferCreateInfo{
        .size = byte_size,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
    };
    const VmaAllocationCreateInfo stg_alloc_ci{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    };
    VkBuffer raw_stg{};
    VmaAllocation stg_alloc{};
    VmaAllocationInfo stg_info{};
    if (vmaCreateBuffer(allocator, &stg_ci, &stg_alloc_ci, &raw_stg, &stg_alloc, &stg_info) !=
            VK_SUCCESS ||
        !stg_info.pMappedData) {
        return false;
    }
    std::memcpy(stg_info.pMappedData, rgba.data(), rgba.size());

    const VkImageCreateInfo img_ci = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = {w, h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    const VmaAllocationCreateInfo img_alloc_ci{
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    VkImage raw_img{};
    if (vmaCreateImage(allocator, &img_ci, &img_alloc_ci, &raw_img, &allocation, nullptr) !=
        VK_SUCCESS) {
        vmaDestroyBuffer(allocator, raw_stg, stg_alloc);
        return false;
    }
    image = raw_img;
    width = w;
    height = h;

    const auto cmdbuf = present_scheduler.CommandBuffer();
    const vk::ImageMemoryBarrier2 to_dst{
        .srcStageMask = vk::PipelineStageFlagBits2::eNone,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = image,
        .subresourceRange = kColorRange,
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_dst});

    const vk::BufferImageCopy copy_region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {w, h, 1},
    };
    cmdbuf.copyBufferToImage(vk::Buffer{raw_stg}, image, vk::ImageLayout::eTransferDstOptimal,
                             copy_region);

    const vk::ImageMemoryBarrier2 to_src{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .image = image,
        .subresourceRange = kColorRange,
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_src});

    present_scheduler.DeferOperation(
        [allocator, raw_stg, stg_alloc] { vmaDestroyBuffer(allocator, raw_stg, stg_alloc); });

    std::printf("[bb_vulkan] Boot splash loaded: %s (%ux%u)\n", label.c_str(), width, height);
    std::fflush(stdout);
    return true;
}

bool SplashScreen::CreateProbeResources() {
    const VmaAllocator allocator = instance.GetAllocator();

    const VkImageCreateInfo img_ci = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .extent = {kProbeWidth, kProbeHeight, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    const VmaAllocationCreateInfo img_alloc_ci{
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    VkImage raw_img{};
    if (vmaCreateImage(allocator, &img_ci, &img_alloc_ci, &raw_img, &probe_img_alloc, nullptr) !=
        VK_SUCCESS) {
        return false;
    }

    const VkBufferCreateInfo buf_ci = vk::BufferCreateInfo{
        .size = kProbeBytes,
        .usage = vk::BufferUsageFlagBits::eTransferDst,
    };
    const VmaAllocationCreateInfo buf_alloc_ci{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    };
    VkBuffer raw_buf{};
    VmaAllocationInfo buf_info{};
    if (vmaCreateBuffer(allocator, &buf_ci, &buf_alloc_ci, &raw_buf, &probe_buf_alloc, &buf_info) !=
            VK_SUCCESS ||
        !buf_info.pMappedData) {
        vmaDestroyImage(allocator, raw_img, probe_img_alloc);
        probe_img_alloc = nullptr;
        return false;
    }

    probe_image = raw_img;
    probe_buffer = raw_buf;
    probe_mapped = static_cast<const u8*>(buf_info.pMappedData);
    return true;
}

void SplashScreen::PresentInitialFrame(Scheduler& present_scheduler, Swapchain& swapchain) {
    if (!swapchain.IsPresentable()) {
        return;
    }
    const vk::Extent2D extent = swapchain.GetExtent();
    if (extent.width == 0 || extent.height == 0) {
        return;
    }
    if (!swapchain.AcquireNextImage()) {
        return;
    }
    const auto cmdbuf = present_scheduler.CommandBuffer();
    const vk::Image target = swapchain.Image();
    const vk::ImageMemoryBarrier2 to_transfer{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = target,
        .subresourceRange = kColorRange,
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_transfer});

    RecordBlit(cmdbuf, target, extent);

    const vk::ImageMemoryBarrier2 to_present{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eNone,
        .dstAccessMask = vk::AccessFlagBits2::eNone,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::ePresentSrcKHR,
        .image = target,
        .subresourceRange = kColorRange,
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_present});

    SubmitInfo info{};
    info.AddWait(swapchain.GetImageAcquiredSemaphore());
    info.AddSignal(swapchain.GetPresentReadySemaphore());
    present_scheduler.Flush(info);
    (void)swapchain.Present();
}

void SplashScreen::RecordProbe(vk::CommandBuffer cmdbuf, vk::Image game_image,
                               vk::Extent2D game_extent) {
    const vk::ImageMemoryBarrier2 to_dst{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = probe_image,
        .subresourceRange = kColorRange,
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_dst});

    const vk::ImageSubresourceLayers layers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    const vk::ImageBlit region{
        .srcSubresource = layers,
        .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{s32(game_extent.width), s32(game_extent.height), 1}},
        .dstSubresource = layers,
        .dstOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{s32(kProbeWidth), s32(kProbeHeight), 1}},
    };
    cmdbuf.blitImage(game_image, vk::ImageLayout::eGeneral, probe_image,
                     vk::ImageLayout::eTransferDstOptimal, region, vk::Filter::eLinear);

    const vk::ImageMemoryBarrier2 to_src{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .image = probe_image,
        .subresourceRange = kColorRange,
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_src});

    const vk::BufferImageCopy copy_region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = layers,
        .imageOffset = {0, 0, 0},
        .imageExtent = {kProbeWidth, kProbeHeight, 1},
    };
    cmdbuf.copyImageToBuffer(probe_image, vk::ImageLayout::eTransferSrcOptimal, probe_buffer,
                             copy_region);

    const vk::BufferMemoryBarrier2 to_host{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        .buffer = probe_buffer,
        .offset = 0,
        .size = kProbeBytes,
    };
    cmdbuf.pipelineBarrier2({.bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &to_host});

    probe_tick = scheduler.CurrentTick();
    probe_in_flight = true;
}

bool SplashScreen::EvaluateProbe() {
    vmaInvalidateAllocation(instance.GetAllocator(), probe_buf_alloc, 0, kProbeBytes);
    ++probe_count;

    constexpr u32 kPixelCount = kProbeWidth * kProbeHeight;
    u32 min_luma = 255;
    u32 max_luma = 0;
    u32 non_black = 0;
    u32 bright = 0;
    for (u32 i = 0; i < kPixelCount; ++i) {
        const u8* px = probe_mapped + size_t(i) * 4u;
        const u32 luma = (u32(px[0]) + u32(px[1]) + u32(px[2])) / 3u;
        min_luma = std::min(min_luma, luma);
        max_luma = std::max(max_luma, luma);
        if (luma >= 12u) {
            ++non_black;
        }
        if (luma >= 200u) {
            ++bright;
        }
    }
    const u32 contrast = max_luma > min_luma ? (max_luma - min_luma) : 0u;

    // 1) Solid gray (#808080) or solid black (#000000) / dark fade frames: keep covering.
    if (contrast < 32u || max_luma < 45u) {
        stable_probes = 0;
        prev_max_luma = max_luma;
        return false;
    }

    // 2) Full-brightness title screen (`MENU_Title_00001` white logo) or intro logo (SCE/FromSoft):
    //    wait until fade-in finishes rising (plateau for 2 consecutive probes at >= 225) so the
    //    splash hands off at 100% matched brightness with zero dimming or brightness jump.
    if (contrast >= 160u && max_luma >= 225u && bright >= 3u) {
        if (max_luma <= prev_max_luma + 1u) {
            if (++stable_probes >= 2u) {
                std::printf("[bb_vulkan] Boot splash handoff (ready): probes=%u min=%u max=%u "
                            "contrast=%u bright=%u non_black=%u\n",
                            probe_count, min_luma, max_luma, contrast, bright, non_black);
                std::fflush(stdout);
                return true;
            }
        } else {
            stable_probes = 0;
        }
        prev_max_luma = max_luma;
        return false;
    }

    // 3) Darker custom/modded non-uniform screen that finished fading in (plateau for 8 probes).
    if (contrast >= 48u && max_luma >= 80u && non_black >= 80u) {
        if (max_luma <= prev_max_luma + 2u) {
            if (++stable_probes >= 8u) {
                std::printf("[bb_vulkan] Boot splash handoff (settled): probes=%u min=%u max=%u "
                            "contrast=%u bright=%u non_black=%u\n",
                            probe_count, min_luma, max_luma, contrast, bright, non_black);
                std::fflush(stdout);
                return true;
            }
        } else {
            stable_probes = 0;
        }
    } else {
        stable_probes = 0;
    }

    prev_max_luma = max_luma;
    return false;
}

bool SplashScreen::UpdateAndShouldCover(vk::CommandBuffer cmdbuf, vk::Image game_image,
                                        vk::Extent2D game_extent) {
    if (!active) {
        return false;
    }

    if (BbStats::draws.load(std::memory_order_relaxed) == 0) {
        return true;
    }

    const auto now = std::chrono::steady_clock::now();
    if (!seen_draws) {
        seen_draws = true;
        first_draw_time = now;
    }

    // Hard safety timeout (10s after draws begin) so splash can never stay stuck on modded builds.
    if (now - first_draw_time > std::chrono::seconds(10)) {
        std::printf("[bb_vulkan] Boot splash handoff (timeout): probes=%u\n", probe_count);
        std::fflush(stdout);
        active = false;
        ScheduleRelease();
        return false;
    }

    // If probe allocation failed, fall back to a brief cover window after draws start.
    if (!probe_image || !probe_buffer || !probe_mapped) {
        if (game_image && now - first_draw_time > std::chrono::milliseconds(1500)) {
            active = false;
            ScheduleRelease();
            return false;
        }
        return true;
    }

    // Check completed async probe without stalling the GPU.
    if (probe_in_flight && scheduler.IsFree(probe_tick)) {
        probe_in_flight = false;
        if (EvaluateProbe()) {
            active = false;
            ScheduleRelease();
            return false;
        }
    }

    // Schedule next probe on the current game frame if none is in flight.
    if (!probe_in_flight && game_image && game_extent.width > 0 && game_extent.height > 0) {
        RecordProbe(cmdbuf, game_image, game_extent);
    }

    return true;
}

void SplashScreen::RecordBlit(vk::CommandBuffer cmdbuf, vk::Image target,
                              vk::Extent2D extent) const {
    const vk::ClearColorValue black{std::array{0.0f, 0.0f, 0.0f, 1.0f}};
    if (!image || width == 0 || height == 0) {
        cmdbuf.clearColorImage(target, vk::ImageLayout::eTransferDstOptimal, black, kColorRange);
        return;
    }
    const vk::Rect2D rect = LetterboxSplash({width, height}, extent);
    if (rect.extent != extent) {
        cmdbuf.clearColorImage(target, vk::ImageLayout::eTransferDstOptimal, black, kColorRange);
        const vk::MemoryBarrier2 cleared{
            .srcStageMask = vk::PipelineStageFlagBits2::eClear,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eBlit,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        };
        cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &cleared});
    }
    const vk::ImageSubresourceLayers layers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    const vk::ImageBlit region{
        .srcSubresource = layers,
        .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                 vk::Offset3D{s32(width), s32(height), 1}},
        .dstSubresource = layers,
        .dstOffsets = std::array{vk::Offset3D{rect.offset.x, rect.offset.y, 0},
                                 vk::Offset3D{rect.offset.x + s32(rect.extent.width),
                                              rect.offset.y + s32(rect.extent.height), 1}},
    };
    cmdbuf.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, target,
                     vk::ImageLayout::eTransferDstOptimal, region, vk::Filter::eLinear);
}

void SplashScreen::ScheduleRelease() {
    const VmaAllocator allocator = instance.GetAllocator();
    const VkImage raw_img = image;
    const VmaAllocation alloc = allocation;
    const VkImage raw_pimg = probe_image;
    const VmaAllocation pimg_alloc = probe_img_alloc;
    const VkBuffer raw_pbuf = probe_buffer;
    const VmaAllocation pbuf_alloc = probe_buf_alloc;

    image = nullptr;
    allocation = nullptr;
    probe_image = nullptr;
    probe_img_alloc = nullptr;
    probe_buffer = nullptr;
    probe_buf_alloc = nullptr;
    probe_mapped = nullptr;
    probe_in_flight = false;

    if (raw_img || raw_pimg || raw_pbuf) {
        scheduler.DeferOperation([allocator, raw_img, alloc, raw_pimg, pimg_alloc, raw_pbuf,
                                  pbuf_alloc] {
            if (raw_img) {
                vmaDestroyImage(allocator, raw_img, alloc);
            }
            if (raw_pimg) {
                vmaDestroyImage(allocator, raw_pimg, pimg_alloc);
            }
            if (raw_pbuf) {
                vmaDestroyBuffer(allocator, raw_pbuf, pbuf_alloc);
            }
        });
    }
}

void SplashScreen::DestroyImmediate() {
    const VmaAllocator allocator = instance.GetAllocator();
    if (image) {
        vmaDestroyImage(allocator, image, allocation);
        image = nullptr;
        allocation = nullptr;
    }
    if (probe_image) {
        vmaDestroyImage(allocator, probe_image, probe_img_alloc);
        probe_image = nullptr;
        probe_img_alloc = nullptr;
    }
    if (probe_buffer) {
        vmaDestroyBuffer(allocator, probe_buffer, probe_buf_alloc);
        probe_buffer = nullptr;
        probe_buf_alloc = nullptr;
        probe_mapped = nullptr;
    }
    probe_in_flight = false;
}

} // namespace Vulkan
