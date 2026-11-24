#include "tpf_hero_art.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace bb_launcher {
namespace {

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;

u32 ReadBe32(const u8* p) noexcept {
    return (u32(p[0]) << 24) | (u32(p[1]) << 16) | (u32(p[2]) << 8) | u32(p[3]);
}

u16 ReadLe16(const u8* p) noexcept {
    return u16(u16(p[0]) | (u16(p[1]) << 8));
}

u32 ReadLe32(const u8* p) noexcept {
    return u32(p[0]) | (u32(p[1]) << 8) | (u32(p[2]) << 16) | (u32(p[3]) << 24);
}

void WriteLe16(u8* p, u16 v) noexcept {
    p[0] = u8(v & 0xFFu);
    p[1] = u8((v >> 8) & 0xFFu);
}

void WriteLe32(u8* p, u32 v) noexcept {
    p[0] = u8(v & 0xFFu);
    p[1] = u8((v >> 8) & 0xFFu);
    p[2] = u8((v >> 16) & 0xFFu);
    p[3] = u8((v >> 24) & 0xFFu);
}

// ---------------------------------------------------------------------------
// Self-contained RFC 1950 (zlib) + RFC 1951 (DEFLATE) decompressor.
// Zero external DLL/library dependencies on both Windows MSVC /MT and Linux.
// ---------------------------------------------------------------------------
struct BitReader {
    const u8* data = nullptr;
    size_t size = 0;
    size_t pos = 0;
    uint64_t buf = 0;
    int bits = 0;

    void Refill() noexcept {
        while (bits <= 56 && pos < size) {
            buf |= uint64_t(data[pos++]) << bits;
            bits += 8;
        }
    }

    bool ReadBits(int n, u32& out) noexcept {
        if (bits < n) Refill();
        if (bits < n) return false;
        out = static_cast<u32>(buf & ((1ULL << n) - 1ULL));
        buf >>= n;
        bits -= n;
        return true;
    }

    void AlignToByte() noexcept {
        const int drop = bits & 7;
        buf >>= drop;
        bits -= drop;
    }

    bool ReadAlignedByte(u8& out) noexcept {
        if (bits >= 8) {
            out = static_cast<u8>(buf & 0xFFu);
            buf >>= 8;
            bits -= 8;
            return true;
        }
        if (pos >= size) return false;
        out = data[pos++];
        return true;
    }
};

struct HuffTable {
    static constexpr int kMaxBits = 15;
    std::array<u16, kMaxBits + 1> counts{};
    std::array<u16, 288> symbols{};

    bool Build(const u8* lengths, int num_syms) noexcept {
        counts.fill(0);
        for (int i = 0; i < num_syms; ++i) {
            if (lengths[i] > kMaxBits) return false;
            counts[lengths[i]]++;
        }
        counts[0] = 0;

        int left = 1;
        for (int len = 1; len <= kMaxBits; ++len) {
            left <<= 1;
            left -= counts[len];
            if (left < 0) return false;
        }

        std::array<u16, kMaxBits + 1> offs{};
        u16 sum = 0;
        for (int len = 1; len <= kMaxBits; ++len) {
            offs[len] = sum;
            sum = static_cast<u16>(sum + counts[len]);
        }

        for (int i = 0; i < num_syms; ++i) {
            const u8 len = lengths[i];
            if (len != 0) {
                symbols[offs[len]++] = static_cast<u16>(i);
            }
        }
        return true;
    }

    bool Decode(BitReader& br, u32& sym) const noexcept {
        int code = 0;
        int first = 0;
        int index = 0;
        for (int len = 1; len <= kMaxBits; ++len) {
            u32 bit = 0;
            if (!br.ReadBits(1, bit)) return false;
            code = (code << 1) | int(bit);
            const int count = counts[len];
            if (code - count < first) {
                sym = symbols[index + (code - first)];
                return true;
            }
            index += count;
            first = (first + count) << 1;
        }
        return false;
    }
};

bool InflateZlib(const u8* src, size_t src_len, std::vector<u8>& dst, size_t expected_size) {
    if (src_len < 6) return false;
    const u8 cmf = src[0];
    const u8 flg = src[1];
    if ((cmf & 0x0Fu) != 8u) return false;
    if (((u32(cmf) << 8) | flg) % 31u != 0u) return false;
    if ((flg & 0x20u) != 0u) return false;  // preset dictionary not used in DCX

    BitReader br{};
    br.data = src + 2;
    br.size = src_len - 2;

    dst.clear();
    dst.reserve(expected_size);

    static constexpr u16 kLenBase[29] = {
        3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
        31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    static constexpr u8 kLenExtra[29] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
        2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    static constexpr u16 kDistBase[30] = {
        1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
        33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
        1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
    static constexpr u8 kDistExtra[30] = {
        0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
        6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    static constexpr u8 kCodeLenOrder[19] = {
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

    u32 bfinal = 0;
    do {
        u32 btype = 0;
        if (!br.ReadBits(1, bfinal) || !br.ReadBits(2, btype)) return false;

        if (btype == 0) {
            br.AlignToByte();
            u8 hdr[4]{};
            for (int i = 0; i < 4; ++i) {
                if (!br.ReadAlignedByte(hdr[i])) return false;
            }
            const u16 len = ReadLe16(hdr + 0);
            const u16 nlen = ReadLe16(hdr + 2);
            if (u16(len ^ 0xFFFFu) != nlen) return false;
            if (dst.size() + len > expected_size + 65536u) return false;
            for (u16 i = 0; i < len; ++i) {
                u8 b = 0;
                if (!br.ReadAlignedByte(b)) return false;
                dst.push_back(b);
            }
        } else if (btype == 1 || btype == 2) {
            HuffTable lit_tab{};
            HuffTable dist_tab{};

            if (btype == 1) {
                std::array<u8, 288> lit_lens{};
                for (int i = 0; i <= 143; ++i) lit_lens[i] = 8;
                for (int i = 144; i <= 255; ++i) lit_lens[i] = 9;
                for (int i = 256; i <= 279; ++i) lit_lens[i] = 7;
                for (int i = 280; i <= 287; ++i) lit_lens[i] = 8;
                std::array<u8, 32> dist_lens{};
                dist_lens.fill(5);
                if (!lit_tab.Build(lit_lens.data(), 288) ||
                    !dist_tab.Build(dist_lens.data(), 32)) {
                    return false;
                }
            } else {
                u32 hlit = 0, hdist = 0, hclen = 0;
                if (!br.ReadBits(5, hlit) || !br.ReadBits(5, hdist) || !br.ReadBits(4, hclen)) {
                    return false;
                }
                const int num_lit = int(hlit) + 257;
                const int num_dist = int(hdist) + 1;
                const int num_clen = int(hclen) + 4;
                if (num_lit > 286 || num_dist > 30) return false;

                std::array<u8, 19> clen_lens{};
                for (int i = 0; i < num_clen; ++i) {
                    u32 v = 0;
                    if (!br.ReadBits(3, v)) return false;
                    clen_lens[kCodeLenOrder[i]] = static_cast<u8>(v);
                }
                HuffTable clen_tab{};
                if (!clen_tab.Build(clen_lens.data(), 19)) return false;

                std::array<u8, 320> lens{};
                const int total = num_lit + num_dist;
                int idx = 0;
                while (idx < total) {
                    u32 sym = 0;
                    if (!clen_tab.Decode(br, sym)) return false;
                    if (sym < 16u) {
                        lens[idx++] = static_cast<u8>(sym);
                    } else if (sym == 16u) {
                        if (idx == 0) return false;
                        u32 extra = 0;
                        if (!br.ReadBits(2, extra)) return false;
                        const int rep = int(extra) + 3;
                        const u8 prev = lens[idx - 1];
                        if (idx + rep > total) return false;
                        for (int r = 0; r < rep; ++r) lens[idx++] = prev;
                    } else if (sym == 17u) {
                        u32 extra = 0;
                        if (!br.ReadBits(3, extra)) return false;
                        const int rep = int(extra) + 3;
                        if (idx + rep > total) return false;
                        for (int r = 0; r < rep; ++r) lens[idx++] = 0;
                    } else if (sym == 18u) {
                        u32 extra = 0;
                        if (!br.ReadBits(7, extra)) return false;
                        const int rep = int(extra) + 11;
                        if (idx + rep > total) return false;
                        for (int r = 0; r < rep; ++r) lens[idx++] = 0;
                    } else {
                        return false;
                    }
                }
                if (!lit_tab.Build(lens.data(), num_lit) ||
                    !dist_tab.Build(lens.data() + num_lit, num_dist)) {
                    return false;
                }
            }

            for (;;) {
                u32 sym = 0;
                if (!lit_tab.Decode(br, sym)) return false;
                if (sym < 256u) {
                    if (dst.size() >= expected_size + 65536u) return false;
                    dst.push_back(static_cast<u8>(sym));
                } else if (sym == 256u) {
                    break;
                } else {
                    const u32 li = sym - 257u;
                    if (li >= 29u) return false;
                    u32 lex = 0;
                    if (kLenExtra[li] > 0 && !br.ReadBits(kLenExtra[li], lex)) return false;
                    const size_t match_len = size_t(kLenBase[li]) + lex;

                    u32 dsym = 0;
                    if (!dist_tab.Decode(br, dsym) || dsym >= 30u) return false;
                    u32 dex = 0;
                    if (kDistExtra[dsym] > 0 && !br.ReadBits(kDistExtra[dsym], dex)) return false;
                    const size_t dist = size_t(kDistBase[dsym]) + dex;
                    if (dist == 0 || dist > dst.size()) return false;
                    if (dst.size() + match_len > expected_size + 65536u) return false;

                    for (size_t k = 0; k < match_len; ++k) {
                        dst.push_back(dst[dst.size() - dist]);
                    }
                }
            }
        } else {
            return false;
        }
    } while (!bfinal);

    return dst.size() >= expected_size;
}

bool MatchUtf16LeAscii(const u8* data, size_t len, size_t off, std::string_view ascii) noexcept {
    for (size_t i = 0; i < ascii.size(); ++i) {
        if (off + i * 2 + 1 >= len) return false;
        if (data[off + i * 2] != u8(ascii[i]) || data[off + i * 2 + 1] != 0) return false;
    }
    const size_t term = off + ascii.size() * 2;
    return term + 1 < len && data[term] == 0 && data[term + 1] == 0;
}

void DecodeBc1BlockRgb(const u8* blk, u8* dst_rgb, u32 stride_px, u32 px0, u32 py0) noexcept {
    const u16 c0 = ReadLe16(blk + 0);
    const u16 c1 = ReadLe16(blk + 2);
    const u32 bits = ReadLe32(blk + 4);

    const u32 r0 = ((c0 >> 11) & 31u) * 255u / 31u;
    const u32 g0 = ((c0 >> 5) & 63u) * 255u / 63u;
    const u32 b0 = (c0 & 31u) * 255u / 31u;

    const u32 r1 = ((c1 >> 11) & 31u) * 255u / 31u;
    const u32 g1 = ((c1 >> 5) & 63u) * 255u / 63u;
    const u32 b1 = (c1 & 31u) * 255u / 31u;

    u8 pal[4][3];
    pal[0][0] = u8(r0); pal[0][1] = u8(g0); pal[0][2] = u8(b0);
    pal[1][0] = u8(r1); pal[1][1] = u8(g1); pal[1][2] = u8(b1);
    if (c0 > c1) {
        pal[2][0] = u8((2u * r0 + r1) / 3u);
        pal[2][1] = u8((2u * g0 + g1) / 3u);
        pal[2][2] = u8((2u * b0 + b1) / 3u);
        pal[3][0] = u8((r0 + 2u * r1) / 3u);
        pal[3][1] = u8((g0 + 2u * g1) / 3u);
        pal[3][2] = u8((b0 + 2u * b1) / 3u);
    } else {
        pal[2][0] = u8((r0 + r1) / 2u);
        pal[2][1] = u8((g0 + g1) / 2u);
        pal[2][2] = u8((b0 + b1) / 2u);
        pal[3][0] = 0; pal[3][1] = 0; pal[3][2] = 0;
    }

    for (u32 y = 0; y < 4; ++y) {
        u8* row = dst_rgb + (size_t(py0 + y) * stride_px + px0) * 3u;
        for (u32 x = 0; x < 4; ++x) {
            const u32 ci = (bits >> (2u * (y * 4u + x))) & 3u;
            row[x * 3 + 0] = pal[ci][0];
            row[x * 3 + 1] = pal[ci][1];
            row[x * 3 + 2] = pal[ci][2];
        }
    }
}

bool WriteRgb24Bmp(const fs::path& dst_path, const u8* rgb, u32 w, u32 h) {
    const u32 row_bytes = (w * 3u + 3u) & ~3u;
    const u32 pixel_bytes = row_bytes * h;
    const u32 file_size = 54u + pixel_bytes;

    std::vector<u8> out(file_size, 0);
    out[0] = 'B';
    out[1] = 'M';
    WriteLe32(out.data() + 2, file_size);
    WriteLe32(out.data() + 10, 54u);
    WriteLe32(out.data() + 14, 40u);
    WriteLe32(out.data() + 18, w);
    WriteLe32(out.data() + 22, h);
    WriteLe16(out.data() + 26, 1u);
    WriteLe16(out.data() + 28, 24u);
    WriteLe32(out.data() + 34, pixel_bytes);

    // Standard bottom-up BGR24 BMP
    for (u32 y = 0; y < h; ++y) {
        const u8* src_row = rgb + size_t(y) * w * 3u;
        u8* dst_row = out.data() + 54u + size_t(h - 1u - y) * row_bytes;
        for (u32 x = 0; x < w; ++x) {
            dst_row[x * 3 + 0] = src_row[x * 3 + 2];  // B
            dst_row[x * 3 + 1] = src_row[x * 3 + 1];  // G
            dst_row[x * 3 + 2] = src_row[x * 3 + 0];  // R
        }
    }

    std::error_code ec;
    fs::create_directories(dst_path.parent_path(), ec);
    const fs::path tmp_path = dst_path.string() + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
        if (!f) return false;
    }
    fs::rename(tmp_path, dst_path, ec);
    if (ec) {
        fs::remove(dst_path, ec);
        ec.clear();
        fs::rename(tmp_path, dst_path, ec);
        if (ec) {
            fs::remove(tmp_path, ec);
            return false;
        }
    }
    return true;
}

std::string PathToFileUrl(const fs::path& p) {
    std::error_code ec;
    const fs::path abs = fs::weakly_canonical(p, ec);
    auto u8s = (ec ? p : abs).u8string();
    std::string raw(reinterpret_cast<const char*>(u8s.data()), u8s.size());
    for (char& c : raw) {
        if (c == '\\') c = '/';
    }
    std::string encoded;
    encoded.reserve(raw.size() + 16);
    if (!raw.empty() && raw[0] == '/') {
        encoded = "file://";
    } else {
        encoded = "file:///";
    }
    static constexpr char kHex[] = "0123456789ABCDEF";
    for (unsigned char c : raw) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
            c == '/' || c == ':' || c == '-' || c == '_' || c == '.' || c == '~') {
            encoded.push_back(static_cast<char>(c));
        } else {
            encoded.push_back('%');
            encoded.push_back(kHex[(c >> 4) & 0xFu]);
            encoded.push_back(kHex[c & 0xFu]);
        }
    }
    return encoded;
}

bool ExtractCroppedTitleBmp(const fs::path& dcx_path, const fs::path& dst_bmp) {
    std::ifstream f(dcx_path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const auto fsize = f.tellg();
    if (fsize < 0x50 || fsize > 32 * 1024 * 1024) return false;
    std::vector<u8> raw(static_cast<size_t>(fsize));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(raw.data()), fsize)) return false;

    if (std::memcmp(raw.data(), "DCX\0", 4) != 0) return false;
    size_t dcs_off = 0;
    size_t dca_off = 0;
    for (size_t i = 4; i + 12 <= std::min<size_t>(raw.size(), 0x100); i += 4) {
        if (dcs_off == 0 && std::memcmp(raw.data() + i, "DCS\0", 4) == 0) {
            dcs_off = i;
        } else if (dca_off == 0 && std::memcmp(raw.data() + i, "DCA\0", 4) == 0) {
            dca_off = i;
        }
    }
    if (dcs_off == 0 || dca_off == 0) return false;

    const u32 uncomp_size = ReadBe32(raw.data() + dcs_off + 4);
    const u32 comp_size = ReadBe32(raw.data() + dcs_off + 8);
    const size_t data_off = dca_off + ReadBe32(raw.data() + dca_off + 4);
    if (uncomp_size < 0x40 || uncomp_size > 32 * 1024 * 1024 ||
        data_off + comp_size > raw.size()) {
        return false;
    }

    std::vector<u8> tpf;
    if (!InflateZlib(raw.data() + data_off, comp_size, tpf, uncomp_size)) return false;
    if (tpf.size() < 16 || std::memcmp(tpf.data(), "TPF\0", 4) != 0) return false;

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
    };
    TpfEntry chosen{};
    auto try_pick = [&](std::string_view target_name) -> bool {
        for (u32 i = 0; i < count; ++i) {
            const u8* e = tpf.data() + 16 + size_t(i) * 0x24u;
            const u32 file_off = ReadLe32(e + 0);
            const u32 file_size = ReadLe32(e + 4);
            const u16 w = ReadLe16(e + 12);
            const u16 h = ReadLe16(e + 14);
            const u32 name_off = ReadLe32(e + 24);
            const u32 dxgi = ReadLe32(e + 32);
            if (dxgi != 71 || w < 64 || h < 64 || (w & 3u) != 0 || (h & 3u) != 0) continue;
            if (size_t(file_off) + size_t(file_size) > tpf.size()) continue;
            if (MatchUtf16LeAscii(tpf.data(), tpf.size(), name_off, target_name)) {
                chosen = {file_off, file_size, w, h};
                return true;
            }
        }
        return false;
    };

    if (!try_pick("MENU_Title_00001") && !try_pick("MENU_Title_00004")) return false;

    const u32 bw = chosen.w / 4u;
    const u32 bh = chosen.h / 4u;
    const u32 tiles_x = (bw + 7u) / 8u;
    const u32 tiles_y = (bh + 7u) / 8u;
    const size_t needed_bytes = size_t(tiles_x) * tiles_y * 64u * 8u;
    if (chosen.file_size < needed_bytes) return false;

    // Untile PS4 Thin_1dThin (8x8 Morton micro-tiles of 8-byte BC1 blocks) -> RGB24
    std::vector<u8> decoded(size_t(chosen.w) * chosen.h * 3u, 0);
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
                    DecodeBc1BlockRgb(src_blocks + idx, decoded.data(), chosen.w, gx * 4u, gy * 4u);
                }
                idx += 8;
            }
        }
    }

    if (chosen.w == 2048 && chosen.h == 1024) {
        // Exact 3-piece 1920x1080 Scaleform reassembly from title.gfx
        constexpr u32 kFullW = 1920;
        constexpr u32 kFullH = 1080;
        std::vector<u8> full(size_t(kFullW) * kFullH * 3u, 0);

        for (u32 y = 0; y < 1024; ++y) {
            std::memcpy(full.data() + size_t(y) * kFullW * 3u,
                        decoded.data() + size_t(y) * chosen.w * 3u, size_t(kFullW) * 3u);
        }
        for (u32 y = 1024; y < 1080; ++y) {
            u8* dst_row = full.data() + size_t(y) * kFullW * 3u;
            const u32 sx_left = 3003u - y;
            for (u32 x = 0; x < 1024; ++x) {
                const u8* sp = decoded.data() + (size_t(x) * chosen.w + sx_left) * 3u;
                std::memcpy(dst_row + size_t(x) * 3u, sp, 3);
            }
            const u32 sx_right = 3064u - y;
            for (u32 x = 1024; x < 1920; ++x) {
                const u8* sp = decoded.data() + (size_t(x - 1024u) * chosen.w + sx_right) * 3u;
                std::memcpy(dst_row + size_t(x) * 3u, sp, 3);
            }
        }

        // Clean horizontal crop (x = 112..1755 -> 1644x1080): removes both left and right
        // pure-black pillarboxes AND leaves zero right-edge texture strip artifacts.
        constexpr u32 kCropX = 112;
        constexpr u32 kCropW = 1644;
        std::vector<u8> cropped(size_t(kCropW) * kFullH * 3u, 0);
        for (u32 y = 0; y < kFullH; ++y) {
            std::memcpy(cropped.data() + size_t(y) * kCropW * 3u,
                        full.data() + (size_t(y) * kFullW + kCropX) * 3u, size_t(kCropW) * 3u);
        }
        return WriteRgb24Bmp(dst_bmp, cropped.data(), kCropW, kFullH);
    }

    return WriteRgb24Bmp(dst_bmp, decoded.data(), chosen.w, chosen.h);
}

}  // namespace

bool InflateZlibStream(const uint8_t* src, size_t src_len,
                       std::vector<uint8_t>& dst, size_t expected_size) {
    return InflateZlib(src, src_len, dst, expected_size);
}

std::string LoadOrExtractHeroArtUrl(const fs::path& game_dir, const fs::path& cache_dir) {
    if (game_dir.empty()) return {};

    std::error_code ec;
    const fs::path dcx_path = game_dir / "dvdroot_ps4" / "menu" / "title.tpf.dcx";
    if (!fs::is_regular_file(dcx_path, ec)) return {};

    const fs::path bmp_path = cache_dir / "ui_assets" / "title_hero_v2.bmp";
    if (fs::is_regular_file(bmp_path, ec) && fs::file_size(bmp_path, ec) > 65536) {
        return PathToFileUrl(bmp_path);
    }

    if (!ExtractCroppedTitleBmp(dcx_path, bmp_path)) return {};
    return PathToFileUrl(bmp_path);
}

}  // namespace bb_launcher
