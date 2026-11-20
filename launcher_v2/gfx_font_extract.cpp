#include "gfx_font_extract.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace bb_launcher {
namespace {

using u8 = uint8_t;
using u16 = uint16_t;
using i16 = int16_t;
using u32 = uint32_t;
using i32 = int32_t;

u16 ReadLe16(const u8* p) noexcept {
    return u16(u16(p[0]) | (u16(p[1]) << 8));
}

i16 ReadLeS16(const u8* p) noexcept {
    return static_cast<i16>(ReadLe16(p));
}

u32 ReadLe32(const u8* p) noexcept {
    return u32(p[0]) | (u32(p[1]) << 8) | (u32(p[2]) << 16) | (u32(p[3]) << 24);
}

void PushBe16(std::vector<u8>& v, u16 val) {
    v.push_back(u8((val >> 8) & 0xFFu));
    v.push_back(u8(val & 0xFFu));
}

void PushBeS16(std::vector<u8>& v, i16 val) {
    PushBe16(v, static_cast<u16>(val));
}

void PushBe32(std::vector<u8>& v, u32 val) {
    v.push_back(u8((val >> 24) & 0xFFu));
    v.push_back(u8((val >> 16) & 0xFFu));
    v.push_back(u8((val >> 8) & 0xFFu));
    v.push_back(u8(val & 0xFFu));
}

void WriteBe32At(std::vector<u8>& v, size_t off, u32 val) {
    v[off + 0] = u8((val >> 24) & 0xFFu);
    v[off + 1] = u8((val >> 16) & 0xFFu);
    v[off + 2] = u8((val >> 8) & 0xFFu);
    v[off + 3] = u8(val & 0xFFu);
}

i16 Div10Round(i32 v) noexcept {
    const i32 r = (v >= 0) ? ((v + 5) / 10) : ((v - 5) / 10);
    return static_cast<i16>(std::clamp<i32>(r, -30000, 30000));
}

struct SwfBitReader {
    const u8* data = nullptr;
    size_t size = 0;
    size_t pos = 0;
    int bit = 0;

    u32 Ub(int n) noexcept {
        u32 v = 0;
        for (int i = 0; i < n; ++i) {
            if (pos >= size) return v;
            v = (v << 1) | u32((data[pos] >> (7 - bit)) & 1u);
            ++bit;
            if (bit == 8) {
                bit = 0;
                ++pos;
            }
        }
        return v;
    }

    i32 Sb(int n) noexcept {
        if (n <= 0) return 0;
        const u32 u = Ub(n);
        if (n < 32 && (u & (1u << (n - 1)))) {
            return static_cast<i32>(u | (~0u << n));
        }
        return static_cast<i32>(u);
    }
};

struct TtfPoint {
    i16 x = 0;
    i16 y = 0;
    bool on_curve = true;
};

struct GlyphData {
    u16 codepoint = 0;
    u16 advance = 0;
    std::vector<std::vector<TtfPoint>> contours;
    i16 x_min = 0;
    i16 y_min = 0;
    i16 x_max = 0;
    i16 y_max = 0;
    bool has_outline = false;
};

bool IsDesiredCodepoint(u16 cp) noexcept {
    if (cp >= 0x0020 && cp <= 0x007E) return true;  // Basic Latin (ASCII)
    if (cp >= 0x00A0 && cp <= 0x00FF) return true;  // Latin-1 Supplement
    if (cp >= 0x0400 && cp <= 0x045F) return true;  // Cyrillic (incl. Ё/ё)
    switch (cp) {
        case 0x2018:  // ‘
        case 0x2019:  // ’
        case 0x201C:  // “
        case 0x201D:  // ”
        case 0x2026:  // …
        case 0x25C6:  // ◆
        case 0x266A:  // ♪
            return true;
        default:
            return false;
    }
}

void ParseSwfGlyph(const u8* buf, size_t buf_size, size_t offset, GlyphData& out) {
    if (offset >= buf_size) return;
    SwfBitReader br{buf, buf_size, offset, 0};
    const int nfill = static_cast<int>(br.Ub(4));
    const int nline = static_cast<int>(br.Ub(4));

    i32 cx = 0;
    i32 cy = 0;
    std::vector<TtfPoint> cur;

    auto flush_contour = [&]() {
        if (cur.size() >= 2) {
            // Remove trailing duplicate on-curve point equal to first point
            while (cur.size() >= 2 && cur.back().on_curve && cur.front().on_curve &&
                   cur.back().x == cur.front().x && cur.back().y == cur.front().y) {
                cur.pop_back();
            }
            if (cur.size() >= 2) {
                out.contours.push_back(std::move(cur));
            }
        }
        cur.clear();
    };

    auto push_on_curve = [&](i16 tx, i16 ty) {
        if (!cur.empty() && cur.back().on_curve && cur.back().x == tx && cur.back().y == ty) {
            return;
        }
        cur.push_back({tx, ty, true});
    };

    for (int guard = 0; guard < 16384 && br.pos < buf_size; ++guard) {
        const u32 is_edge = br.Ub(1);
        if (!is_edge) {
            const u32 flags = br.Ub(5);
            if (flags == 0) break;  // EndShapeRecord
            if (flags & 0x01u) {
                flush_contour();
                const int mb = static_cast<int>(br.Ub(5));
                cx = br.Sb(mb);
                cy = br.Sb(mb);
            }
            if (flags & 0x02u) (void)br.Ub(nfill);
            if (flags & 0x04u) (void)br.Ub(nfill);
            if (flags & 0x08u) (void)br.Ub(nline);
            if (flags & 0x10u) break;  // StateNewStyles (unused in DefineFont3)
        } else {
            const u32 is_straight = br.Ub(1);
            const int nb = static_cast<int>(br.Ub(4)) + 2;
            const i16 p0x = Div10Round(cx);
            const i16 p0y = Div10Round(-cy);
            if (is_straight) {
                const u32 gen = br.Ub(1);
                i32 dx = 0;
                i32 dy = 0;
                if (gen) {
                    dx = br.Sb(nb);
                    dy = br.Sb(nb);
                } else {
                    const u32 vert = br.Ub(1);
                    if (vert) {
                        dy = br.Sb(nb);
                    } else {
                        dx = br.Sb(nb);
                    }
                }
                push_on_curve(p0x, p0y);
                cx += dx;
                cy += dy;
            } else {
                const i32 cdx = br.Sb(nb);
                const i32 cdy = br.Sb(nb);
                const i32 adx = br.Sb(nb);
                const i32 ady = br.Sb(nb);
                const i32 qx = cx + cdx;
                const i32 qy = cy + cdy;
                push_on_curve(p0x, p0y);
                cur.push_back({Div10Round(qx), Div10Round(-qy), false});
                cx = qx + adx;
                cy = qy + ady;
            }
        }
    }
    flush_contour();

    if (!out.contours.empty()) {
        i16 x0 = std::numeric_limits<i16>::max();
        i16 y0 = std::numeric_limits<i16>::max();
        i16 x1 = std::numeric_limits<i16>::min();
        i16 y1 = std::numeric_limits<i16>::min();
        for (const auto& c : out.contours) {
            for (const auto& pt : c) {
                x0 = std::min(x0, pt.x);
                y0 = std::min(y0, pt.y);
                x1 = std::max(x1, pt.x);
                y1 = std::max(y1, pt.y);
            }
        }
        out.x_min = x0;
        out.y_min = y0;
        out.x_max = x1;
        out.y_max = y1;
        out.has_outline = true;
    }
}

u32 CalcTableChecksum(const std::vector<u8>& tbl) noexcept {
    u32 sum = 0;
    const size_t n = tbl.size();
    for (size_t i = 0; i < n; i += 4) {
        u32 word = 0;
        for (size_t b = 0; b < 4; ++b) {
            word <<= 8;
            if (i + b < n) word |= tbl[i + b];
        }
        sum += word;
    }
    return sum;
}

std::vector<u8> BuildTtfBinary(const std::vector<GlyphData>& glyphs, i16 ascent, i16 descent) {
    constexpr u16 kUnitsPerEm = 2048;
    const u16 num_glyphs = static_cast<u16>(glyphs.size());

    // 1. Build `glyf` and `loca` (32-bit offsets)
    std::vector<u8> glyf;
    std::vector<u32> loca_offs;
    loca_offs.reserve(num_glyphs + 1);

    u16 max_points = 0;
    u16 max_contours = 0;
    i16 global_xmin = 0, global_ymin = 0, global_xmax = 0, global_ymax = 0;
    bool have_global_bb = false;
    u16 max_adv = 0;
    i16 min_lsb = 0, min_rsb = 0, x_max_extent = 0;
    bool have_hmetrics = false;

    for (const auto& g : glyphs) {
        loca_offs.push_back(static_cast<u32>(glyf.size()));
        max_adv = std::max(max_adv, g.advance);

        if (!g.has_outline) continue;

        if (!have_global_bb) {
            global_xmin = g.x_min;
            global_ymin = g.y_min;
            global_xmax = g.x_max;
            global_ymax = g.y_max;
            have_global_bb = true;
        } else {
            global_xmin = std::min(global_xmin, g.x_min);
            global_ymin = std::min(global_ymin, g.y_min);
            global_xmax = std::max(global_xmax, g.x_max);
            global_ymax = std::max(global_ymax, g.y_max);
        }

        const i16 lsb = g.x_min;
        const i16 rsb = static_cast<i16>(i32(g.advance) - i32(g.x_max));
        if (!have_hmetrics) {
            min_lsb = lsb;
            min_rsb = rsb;
            x_max_extent = g.x_max;
            have_hmetrics = true;
        } else {
            min_lsb = std::min(min_lsb, lsb);
            min_rsb = std::min(min_rsb, rsb);
            x_max_extent = std::max(x_max_extent, g.x_max);
        }

        const u16 nc = static_cast<u16>(g.contours.size());
        max_contours = std::max(max_contours, nc);

        PushBeS16(glyf, static_cast<i16>(nc));
        PushBeS16(glyf, g.x_min);
        PushBeS16(glyf, g.y_min);
        PushBeS16(glyf, g.x_max);
        PushBeS16(glyf, g.y_max);

        u16 total_pts = 0;
        for (const auto& c : g.contours) {
            total_pts = static_cast<u16>(total_pts + c.size());
            PushBe16(glyf, static_cast<u16>(total_pts - 1));
        }
        max_points = std::max(max_points, total_pts);

        PushBe16(glyf, 0);  // instructionLength = 0

        // Flags (1 byte per point: bit 0 = ON_CURVE; using 16-bit signed dx/dy)
        for (const auto& c : g.contours) {
            for (const auto& pt : c) {
                glyf.push_back(pt.on_curve ? 0x01u : 0x00u);
            }
        }
        // X delta coordinates (16-bit big-endian)
        i16 prev_x = 0;
        for (const auto& c : g.contours) {
            for (const auto& pt : c) {
                PushBeS16(glyf, static_cast<i16>(pt.x - prev_x));
                prev_x = pt.x;
            }
        }
        // Y delta coordinates (16-bit big-endian)
        i16 prev_y = 0;
        for (const auto& c : g.contours) {
            for (const auto& pt : c) {
                PushBeS16(glyf, static_cast<i16>(pt.y - prev_y));
                prev_y = pt.y;
            }
        }
        while (glyf.size() & 3u) glyf.push_back(0);
    }
    loca_offs.push_back(static_cast<u32>(glyf.size()));

    std::vector<u8> loca;
    loca.reserve(loca_offs.size() * 4);
    for (u32 off : loca_offs) PushBe32(loca, off);

    // 2. Build `hmtx`
    std::vector<u8> hmtx;
    hmtx.reserve(size_t(num_glyphs) * 4);
    for (const auto& g : glyphs) {
        PushBe16(hmtx, g.advance);
        PushBeS16(hmtx, g.has_outline ? g.x_min : 0);
    }

    // 3. Build `cmap` (format 4 — BMP Unicode, glyph indices sorted by codepoint)
    struct Seg {
        u16 start;
        u16 end;
        i16 delta;
    };
    std::vector<Seg> segs;
    for (u16 gid = 1; gid < num_glyphs; ++gid) {
        const u16 cp = glyphs[gid].codepoint;
        const i16 delta = static_cast<i16>(i32(gid) - i32(cp));
        if (!segs.empty() && cp == segs.back().end + 1 && delta == segs.back().delta) {
            segs.back().end = cp;
        } else {
            segs.push_back({cp, cp, delta});
        }
    }
    segs.push_back({0xFFFFu, 0xFFFFu, 1});  // required sentinel

    const u16 seg_count = static_cast<u16>(segs.size());
    const u16 seg_count_x2 = static_cast<u16>(seg_count * 2u);
    u16 search_range = 2;
    u16 entry_selector = 0;
    while (search_range * 2u <= seg_count_x2) {
        search_range = static_cast<u16>(search_range * 2u);
        ++entry_selector;
    }
    const u16 range_shift = static_cast<u16>(seg_count_x2 - search_range);

    std::vector<u8> fmt4;
    PushBe16(fmt4, 4);  // format
    const u16 fmt4_len = static_cast<u16>(16u + size_t(seg_count) * 8u);
    PushBe16(fmt4, fmt4_len);
    PushBe16(fmt4, 0);  // language
    PushBe16(fmt4, seg_count_x2);
    PushBe16(fmt4, search_range);
    PushBe16(fmt4, entry_selector);
    PushBe16(fmt4, range_shift);
    for (const auto& s : segs) PushBe16(fmt4, s.end);
    PushBe16(fmt4, 0);  // reservedPad
    for (const auto& s : segs) PushBe16(fmt4, s.start);
    for (const auto& s : segs) PushBeS16(fmt4, s.delta);
    for (size_t i = 0; i < segs.size(); ++i) PushBe16(fmt4, 0);  // idRangeOffset = 0

    std::vector<u8> cmap;
    PushBe16(cmap, 0);  // version
    PushBe16(cmap, 2);  // numTables (Platform 0 Unicode + Platform 3 Windows BMP)
    const u32 sub_off = 4u + 2u * 8u;
    PushBe16(cmap, 0); PushBe16(cmap, 3); PushBe32(cmap, sub_off);  // Unicode BMP
    PushBe16(cmap, 3); PushBe16(cmap, 1); PushBe32(cmap, sub_off);  // Windows BMP
    cmap.insert(cmap.end(), fmt4.begin(), fmt4.end());

    // 4. Build `head` (54 bytes; checkSumAdjustment filled after assembling file)
    std::vector<u8> head;
    PushBe32(head, 0x00010000u);  // major/minor
    PushBe32(head, 0x00010000u);  // fontRevision
    PushBe32(head, 0);            // checkSumAdjustment (placeholder)
    PushBe32(head, 0x5F0F3CF5u);  // magicNumber
    PushBe16(head, 0x000Bu);      // flags (baseline y=0, lsb x=0)
    PushBe16(head, kUnitsPerEm);
    PushBe32(head, 0); PushBe32(head, 0);  // created
    PushBe32(head, 0); PushBe32(head, 0);  // modified
    PushBeS16(head, global_xmin);
    PushBeS16(head, global_ymin);
    PushBeS16(head, global_xmax);
    PushBeS16(head, global_ymax);
    PushBe16(head, 0);   // macStyle
    PushBe16(head, 8);   // lowestRecPPEM
    PushBeS16(head, 2);  // fontDirectionHint
    PushBeS16(head, 1);  // indexToLocFormat = 32-bit
    PushBeS16(head, 0);  // glyphDataFormat

    // 5. Build `hhea` (36 bytes)
    const i16 line_gap = 200;
    std::vector<u8> hhea;
    PushBe32(hhea, 0x00010000u);
    PushBeS16(hhea, ascent);
    PushBeS16(hhea, static_cast<i16>(-descent));
    PushBeS16(hhea, line_gap);
    PushBe16(hhea, max_adv);
    PushBeS16(hhea, min_lsb);
    PushBeS16(hhea, min_rsb);
    PushBeS16(hhea, x_max_extent);
    PushBeS16(hhea, 1);  // caretSlopeRise
    PushBeS16(hhea, 0);  // caretSlopeRun
    PushBeS16(hhea, 0);  // caretOffset
    PushBeS16(hhea, 0); PushBeS16(hhea, 0); PushBeS16(hhea, 0); PushBeS16(hhea, 0);
    PushBeS16(hhea, 0);  // metricDataFormat
    PushBe16(hhea, num_glyphs);

    // 6. Build `maxp` (version 1.0, 32 bytes)
    std::vector<u8> maxp;
    PushBe32(maxp, 0x00010000u);
    PushBe16(maxp, num_glyphs);
    PushBe16(maxp, max_points);
    PushBe16(maxp, max_contours);
    PushBe16(maxp, 0);  // maxCompositePoints
    PushBe16(maxp, 0);  // maxCompositeContours
    PushBe16(maxp, 1);  // maxZones
    PushBe16(maxp, 0);  // maxTwilightPoints
    PushBe16(maxp, 0);  // maxStorage
    PushBe16(maxp, 0);  // maxFunctionDefs
    PushBe16(maxp, 0);  // maxInstructionDefs
    PushBe16(maxp, 0);  // maxStackElements
    PushBe16(maxp, 0);  // maxSizeOfInstructions
    PushBe16(maxp, 0);  // maxComponentElements
    PushBe16(maxp, 0);  // maxComponentDepth

    // 7. Build `OS/2` (version 4, 96 bytes — required by Windows DirectWrite & OTS)
    const u16 win_ascent = static_cast<u16>(std::max<i32>(ascent, global_ymax));
    const u16 win_descent = static_cast<u16>(std::max<i32>(descent, -global_ymin));
    const u16 first_char = (num_glyphs > 1) ? glyphs[1].codepoint : 0x0020u;
    const u16 last_char = (num_glyphs > 1) ? glyphs.back().codepoint : 0x007Eu;
    std::vector<u8> os2;
    PushBe16(os2, 4);     // version
    PushBeS16(os2, 1100); // xAvgCharWidth
    PushBe16(os2, 400);   // usWeightClass (Regular)
    PushBe16(os2, 5);     // usWidthClass (Medium)
    PushBe16(os2, 0);     // fsType
    PushBeS16(os2, 1331); PushBeS16(os2, 1433); PushBeS16(os2, 0); PushBeS16(os2, 286);  // subscript
    PushBeS16(os2, 1331); PushBeS16(os2, 1433); PushBeS16(os2, 0); PushBeS16(os2, 983);  // superscript
    PushBeS16(os2, 102);  PushBeS16(os2, 530);  // strikeout
    PushBeS16(os2, 0);    // sFamilyClass
    for (int i = 0; i < 10; ++i) os2.push_back(0);  // panose
    PushBe32(os2, 0x00000203u);  // ulUnicodeRange1 (Basic Latin, Latin-1, Cyrillic)
    PushBe32(os2, 0x00000001u);  // ulUnicodeRange2 (General Punctuation)
    PushBe32(os2, 0);
    PushBe32(os2, 0);
    os2.push_back('B'); os2.push_back('B'); os2.push_back('P'); os2.push_back('C');
    PushBe16(os2, 0x0040u);  // fsSelection = REGULAR
    PushBe16(os2, first_char);
    PushBe16(os2, last_char);
    PushBeS16(os2, ascent);
    PushBeS16(os2, static_cast<i16>(-descent));
    PushBeS16(os2, line_gap);
    PushBe16(os2, win_ascent);
    PushBe16(os2, win_descent);
    PushBe32(os2, 0x00000005u);  // ulCodePageRange1 (Latin 1 + Cyrillic)
    PushBe32(os2, 0);
    PushBeS16(os2, 1100);  // sxHeight
    PushBeS16(os2, 1570);  // sCapHeight
    PushBe16(os2, 0);      // usDefaultChar
    PushBe16(os2, 0x0020); // usBreakChar
    PushBe16(os2, 1);      // usMaxContext

    // 8. Build `name` (Windows Unicode BMP UTF-16BE — required by DirectWrite/OTS)
    struct NameRec {
        u16 id;
        const char* ascii;
    };
    static constexpr std::array<NameRec, 6> kNames = {{
        {1, "BloodborneUI"},
        {2, "Regular"},
        {3, "BloodborneUI-Local"},
        {4, "BloodborneUI Regular"},
        {5, "Version 1.0"},
        {6, "BloodborneUI-Regular"},
    }};
    std::vector<u8> str_blob;
    std::vector<u8> name;
    PushBe16(name, 0);  // format
    PushBe16(name, static_cast<u16>(kNames.size()));
    PushBe16(name, static_cast<u16>(6u + kNames.size() * 12u));
    for (const auto& nr : kNames) {
        const u16 off = static_cast<u16>(str_blob.size());
        const size_t slen = std::strlen(nr.ascii);
        for (size_t i = 0; i < slen; ++i) {
            str_blob.push_back(0);
            str_blob.push_back(static_cast<u8>(nr.ascii[i]));
        }
        PushBe16(name, 3);       // platformID = Windows
        PushBe16(name, 1);       // encodingID = Unicode BMP
        PushBe16(name, 0x0409u); // languageID = en-US
        PushBe16(name, nr.id);
        PushBe16(name, static_cast<u16>(slen * 2u));
        PushBe16(name, off);
    }
    name.insert(name.end(), str_blob.begin(), str_blob.end());

    // 9. Build `post` (format 3.0, 32 bytes)
    std::vector<u8> post;
    PushBe32(post, 0x00030000u);
    PushBe32(post, 0);        // italicAngle
    PushBeS16(post, -150);    // underlinePosition
    PushBeS16(post, 90);      // underlineThickness
    PushBe32(post, 0);        // isFixedPitch
    PushBe32(post, 0); PushBe32(post, 0); PushBe32(post, 0); PushBe32(post, 0);

    // Assemble SFNT tables strictly sorted by ASCII tag
    struct TableEntry {
        const char* tag;
        std::vector<u8>* data;
    };
    std::array<TableEntry, 10> tables = {{
        {"OS/2", &os2},
        {"cmap", &cmap},
        {"glyf", &glyf},
        {"head", &head},
        {"hhea", &hhea},
        {"hmtx", &hmtx},
        {"loca", &loca},
        {"maxp", &maxp},
        {"name", &name},
        {"post", &post},
    }};

    const u16 num_tables = static_cast<u16>(tables.size());
    u16 sfnt_sr = 16;
    u16 sfnt_es = 0;
    while (sfnt_sr * 2u <= num_tables * 16u) {
        sfnt_sr = static_cast<u16>(sfnt_sr * 2u);
        ++sfnt_es;
    }
    const u16 sfnt_rs = static_cast<u16>(num_tables * 16u - sfnt_sr);

    std::vector<u8> out;
    PushBe32(out, 0x00010000u);  // TrueType sfntVersion
    PushBe16(out, num_tables);
    PushBe16(out, sfnt_sr);
    PushBe16(out, sfnt_es);
    PushBe16(out, sfnt_rs);

    u32 cur_off = 12u + u32(num_tables) * 16u;
    size_t head_file_off = 0;
    for (const auto& t : tables) {
        if (std::memcmp(t.tag, "head", 4) == 0) head_file_off = cur_off;
        for (int i = 0; i < 4; ++i) out.push_back(static_cast<u8>(t.tag[i]));
        PushBe32(out, CalcTableChecksum(*t.data));
        PushBe32(out, cur_off);
        PushBe32(out, static_cast<u32>(t.data->size()));
        cur_off += static_cast<u32>((t.data->size() + 3u) & ~3u);
    }
    for (const auto& t : tables) {
        out.insert(out.end(), t.data->begin(), t.data->end());
        while (out.size() & 3u) out.push_back(0);
    }

    // Patch head.checkSumAdjustment
    const u32 total_sum = CalcTableChecksum(out);
    WriteBe32At(out, head_file_off + 8, 0xB1B0AFBAu - total_sum);
    return out;
}

fs::path FindGameFontGfx(const fs::path& game_dir) {
    if (game_dir.empty()) return {};
    const fs::path menu_dir = game_dir / "dvdroot_ps4" / "menu";
    std::error_code ec;
    if (!fs::is_directory(menu_dir, ec)) return {};

    static constexpr const char* kCandidates[] = {
        "engus", "enggb", "rusru", "deude", "frafr", "itait",
        "spaes", "spaar", "polpl", "porbr", "porpt", "jpnjp",
    };
    for (const char* loc : kCandidates) {
        const fs::path p = menu_dir / loc / "font.gfx";
        if (fs::is_regular_file(p, ec)) return p;
    }
    for (const auto& entry : fs::directory_iterator(menu_dir, ec)) {
        if (!entry.is_directory(ec)) continue;
        const fs::path p = entry.path() / "font.gfx";
        if (fs::is_regular_file(p, ec)) return p;
    }
    return {};
}

std::string PathToFileUrl(const fs::path& p) {
    std::error_code ec;
    const fs::path abs = fs::weakly_canonical(p, ec);
    auto u8s = (ec ? p : abs).u8string();
    std::string raw(reinterpret_cast<const char*>(u8s.data()), u8s.size());
    for (char& c : raw) {
        if (c == '\\') c = '/';
    }
    std::string encoded = (!raw.empty() && raw[0] == '/') ? "file://" : "file:///";
    encoded.reserve(raw.size() + 16);
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

bool ExtractFontGfxToTtf(const fs::path& gfx_path, const fs::path& dst_ttf) {
    std::ifstream f(gfx_path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const auto fsize = f.tellg();
    if (fsize < 32 || fsize > 32 * 1024 * 1024) return false;
    std::vector<u8> data(static_cast<size_t>(fsize));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(data.data()), fsize)) return false;

    // Uncompressed Scaleform GFX (or FWS)
    if (std::memcmp(data.data(), "GFX", 3) != 0 && std::memcmp(data.data(), "FWS", 3) != 0) {
        return false;
    }

    const size_t nbits = (data[8] >> 3) & 0x1Fu;
    const size_t rect_bytes = (5u + 4u * nbits + 7u) / 8u;
    size_t pos = 8u + rect_bytes + 4u;

    while (pos + 2 <= data.size()) {
        const u16 th = ReadLe16(data.data() + pos);
        pos += 2;
        const u16 tag = th >> 6;
        u32 len = th & 0x3Fu;
        if (len == 0x3Fu) {
            if (pos + 4 > data.size()) return false;
            len = ReadLe32(data.data() + pos);
            pos += 4;
        }
        if (pos + len > data.size()) return false;

        if (tag == 75) {  // DefineFont3
            const size_t tag_end = pos + len;
            size_t p = pos;
            if (p + 5 > tag_end) return false;
            const u8 flags = data[p + 2];
            const u8 name_len = data[p + 4];
            p += 5u + name_len;
            if (p + 2 > tag_end) return false;
            const u16 num_glyphs = ReadLe16(data.data() + p);
            p += 2;

            const bool wide_offsets = (flags & 0x08u) != 0;
            const bool wide_codes = (flags & 0x04u) != 0;
            const bool has_layout = (flags & 0x80u) != 0;
            if (!wide_offsets || !wide_codes || num_glyphs == 0) return false;

            const size_t off_base = p;
            if (off_base + size_t(num_glyphs + 1) * 4u > tag_end) return false;
            const u32 ct_rel = ReadLe32(data.data() + off_base + size_t(num_glyphs) * 4u);
            const size_t ct_pos = off_base + ct_rel;
            if (ct_pos + size_t(num_glyphs) * 2u > tag_end) return false;

            i16 ascent = 1802;
            i16 descent = 246;
            size_t adv_pos = 0;
            if (has_layout) {
                const size_t lay_pos = ct_pos + size_t(num_glyphs) * 2u;
                if (lay_pos + 6u + size_t(num_glyphs) * 2u <= tag_end) {
                    ascent = Div10Round(ReadLeS16(data.data() + lay_pos + 0));
                    descent = Div10Round(ReadLeS16(data.data() + lay_pos + 2));
                    adv_pos = lay_pos + 6u;
                }
            }

            // Collect desired glyphs sorted strictly by BMP codepoint
            std::map<u16, u16> cp_to_swf_idx;
            for (u16 i = 0; i < num_glyphs; ++i) {
                const u16 cp = ReadLe16(data.data() + ct_pos + size_t(i) * 2u);
                if (IsDesiredCodepoint(cp)) {
                    cp_to_swf_idx.try_emplace(cp, i);
                }
            }
            if (cp_to_swf_idx.size() < 60) return false;

            std::vector<GlyphData> glyphs;
            glyphs.reserve(cp_to_swf_idx.size() + 1);

            // Glyph #0: required `.notdef`
            GlyphData notdef{};
            notdef.codepoint = 0;
            notdef.advance = 1024;
            glyphs.push_back(std::move(notdef));

            for (const auto& [cp, swf_i] : cp_to_swf_idx) {
                GlyphData gd{};
                gd.codepoint = cp;
                if (adv_pos != 0) {
                    const i16 adv_raw = ReadLeS16(data.data() + adv_pos + size_t(swf_i) * 2u);
                    gd.advance = static_cast<u16>(std::max<i16>(0, Div10Round(adv_raw)));
                } else {
                    gd.advance = 1024;
                }
                const u32 g_rel = ReadLe32(data.data() + off_base + size_t(swf_i) * 4u);
                ParseSwfGlyph(data.data(), tag_end, off_base + g_rel, gd);
                glyphs.push_back(std::move(gd));
            }

            const std::vector<u8> ttf = BuildTtfBinary(glyphs, ascent, descent);
            if (ttf.size() < 512) return false;

            std::error_code ec;
            fs::create_directories(dst_ttf.parent_path(), ec);
            const fs::path tmp = dst_ttf.string() + ".tmp";
            {
                std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
                if (!out) return false;
                out.write(reinterpret_cast<const char*>(ttf.data()),
                          static_cast<std::streamsize>(ttf.size()));
                if (!out) return false;
            }
            fs::rename(tmp, dst_ttf, ec);
            if (ec) {
                fs::remove(dst_ttf, ec);
                ec.clear();
                fs::rename(tmp, dst_ttf, ec);
                if (ec) {
                    fs::remove(tmp, ec);
                    return false;
                }
            }
            return true;
        }
        if (tag == 0) break;
        pos += len;
    }
    return false;
}

}  // namespace

std::string LoadOrExtractGameFontUrl(const fs::path& game_dir, const fs::path& cache_dir) {
    const fs::path gfx = FindGameFontGfx(game_dir);
    if (gfx.empty()) return {};

    std::error_code ec;
    const fs::path ttf_path = cache_dir / "ui_assets" / "bloodborne_font_v2.ttf";
    if (fs::is_regular_file(ttf_path, ec) && fs::file_size(ttf_path, ec) > 8192) {
        return PathToFileUrl(ttf_path);
    }

    if (!ExtractFontGfxToTtf(gfx, ttf_path)) return {};
    return PathToFileUrl(ttf_path);
}

}  // namespace bb_launcher
