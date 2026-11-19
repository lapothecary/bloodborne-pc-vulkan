#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>

namespace bb_launcher {

// Decodes Bloodborne's authentic UI menu SFX (CURSOL_MOVE, CURSOL_OK, CURSOL_SELECT,
// CURSOL_CANCEL, mainmenu_categoly_kirikae, GAMESTART) at runtime from the user's own
// local CUSA03173/dvdroot_ps4/sound/sprj_main_*.fsb bank via LibAtrac9 into
// <cache_dir>/ui_sfx/*.wav. Zero proprietary Sony/FromSoftware audio assets are ever
// stored in this repository or release archives.
//
// Returns map: logical key ("hover", "click", "select", "cancel", "tab", "gamestart")
// -> "data:audio/wav;base64,..." ready for low-latency Web Audio playback.
std::unordered_map<std::string, std::string> LoadOrExtractUiSfx(
    const std::filesystem::path& game_dir,
    const std::filesystem::path& cache_dir);

}  // namespace bb_launcher
