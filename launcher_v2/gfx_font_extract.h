#pragma once

#include <filesystem>
#include <string>

namespace bb_launcher {

// Extracts Bloodborne's official Victorian serif font (`TT-Reimin Y10 B`) from
// `<game_dir>/dvdroot_ps4/menu/<lang>/font.gfx` (SWF DefineFont3 quadratic Beziers),
// packs the Latin + Cyrillic + UI punctuation glyphs into a standards-compliant
// TrueType `.ttf` cached at `<cache_dir>/ui_assets/bloodborne_font_v1.ttf`, and
// returns a properly escaped `file:///` URL ready for `@font-face` / `FontFace`.
// Returns an empty string if `font.gfx` is not present in `game_dir`.
std::string LoadOrExtractGameFontUrl(const std::filesystem::path& game_dir,
                                     const std::filesystem::path& cache_dir);

}  // namespace bb_launcher
