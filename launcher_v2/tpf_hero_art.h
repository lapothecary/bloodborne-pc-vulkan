#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace bb_launcher {

// Self-contained RFC 1950 (zlib) + RFC 1951 (DEFLATE) decompressor shared by
// launcher asset/catalog readers (zero external DLL dependencies on Windows/Linux).
bool InflateZlibStream(const uint8_t* src, size_t src_len,
                       std::vector<uint8_t>& dst, size_t expected_size);

// Decodes Bloodborne's authentic Main Menu artwork (`MENU_Title_00001`, fallback
// `MENU_Title_00004`) locally from `<game_dir>/dvdroot_ps4/menu/title.tpf.dcx`,
// reassembles the exact 1920x1080 Scaleform layout, crops out the left/right
// pillarbox bars cleanly (1644x1080), caches a 24-bit BMP in
// `<cache_dir>/ui_sfx/title_hero_v2.bmp`, and returns a properly escaped
// `file:///` URL ready for WebKitGTK / WebView2.
// Returns an empty string if `game_dir` does not contain a valid `title.tpf.dcx`,
// allowing the UI to fall back gracefully to built-in artwork.
std::string LoadOrExtractHeroArtUrl(const std::filesystem::path& game_dir,
                                    const std::filesystem::path& cache_dir);

}  // namespace bb_launcher
