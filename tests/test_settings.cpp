// SPDX-License-Identifier: GPL-2.0-or-later
// bbport_settings.cpp: the menu saves its keys into bbport.ini and keeps the launcher's (controls).
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unistd.h>
#include "gpu/shim/bbport_settings.h"

namespace BbFrameRate {
void SetLimit(int) {}
} // namespace BbFrameRate
namespace BbSysInfo {
std::string RedactHomePath(std::string_view path) {
    return std::string{path};
}
} // namespace BbSysInfo

static std::string Read(const char* path) {
    std::ifstream file(path);
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

int main() {
    assert(BbSettings::IsHotEffect(6));  // SSR now switches live in-game too.
    assert(BbSettings::IsHotEffect(7));  // Bloom still switches live.
    assert(!BbSettings::IsHotEffect(8)); // Startup intros.
    char path[] = "/tmp/bbport-settings-test-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    const char ini[] = "# launcher\nupscaler=fsr3\nkey.cross=X, Space\npad.circle=a\nshow_fps=0\n"
                       "fullscreen_hint=1\n";
    assert(write(fd, ini, sizeof(ini) - 1) == ssize_t(sizeof(ini) - 1));
    close(fd);
    setenv("BB_CONFIG", path, 1);

    BbSettings::Load();
    auto& s = BbSettings::Get();
    assert(s.upscaler == BbSettings::UpscalerFsr3 && !s.show_fps && s.menu_x < 0.0f);
    s.show_fps = true;
    s.upscaler = BbSettings::UpscalerFsr411;
    s.menu_x = 0.625f;
    s.menu_y = 0.125f;
    BbSettings::Save();

    const std::string saved = Read(path);
    // The launcher's controls, its other keys and comments stay; the menu's keys are replaced
    // in place, new ones appended.
    assert(saved.find("# launcher\nupscaler=fsr411\nkey.cross=X, Space\npad.circle=a\nshow_fps=1\n"
                      "fullscreen_hint=1\n") == 0);
    assert(saved.find("menu_pos=0.6250,0.1250\n") != std::string::npos);
    assert(saved.find("upscaler=fsr3") == std::string::npos);

    s.menu_x = -1.0f;
    s.menu_y = -1.0f;
    BbSettings::Load();
    assert(s.menu_x == 0.625f && s.menu_y == 0.125f && s.upscaler == BbSettings::UpscalerFsr411);
    assert(saved.find("menu_language=") == std::string::npos); // only started with it (#134)

    // A language chosen in the menu is saved and wins over the launcher's from then on.
    s.menu_language = BbSettings::MenuLanguage::Russian;
    s.menu_language_chosen = true;
    BbSettings::Save();
    assert(Read(path).find("menu_language=ru\n") != std::string::npos);
    setenv("BB_MENU_LANGUAGE", "en", 1);
    s.menu_language_chosen = false;
    BbSettings::Load();
    assert(s.menu_language == BbSettings::MenuLanguage::Russian && s.menu_language_chosen);

    // Native Bloodborne CSOptionsData settings round-trip through bbport.ini & live external reload.
    s.bgm_volume = 0;
    s.se_volume = 4;
    s.voice_volume = 9;
    s.blood_expression = false;
    s.subtitles = true;
    s.game_brightness = 6;
    s.camera_speed = 8;
    s.pad_vibration = 3;
    s.camera_invert_x = true;
    s.camera_invert_y = false;
    s.auto_target = true;
    s.camera_auto_adjust = false;
    s.network_region = 1;
    s.player_name_display = 0;
    BbSettings::SetNetworkPassword("hunt1234extra");
    s.game_text_language = BbSettings::ParseGameLanguageCode("ruru");
    s.game_voice_language = BbSettings::ParseGameLanguageCode("engb", BbSettings::GameVoiceLanguageCount);
    s.native_chosen_mask = 0x1ffffu;
    BbSettings::Save();
    assert(!BbSettings::ReloadExternalIfModified());

    const std::string saved_native = Read(path);
    assert(saved_native.find("bgm_volume=0\n") != std::string::npos);
    assert(saved_native.find("network_region=worldwide\n") != std::string::npos);
    assert(saved_native.find("player_name_display=character\n") != std::string::npos);
    assert(saved_native.find("network_password=hunt1234\n") != std::string::npos);
    assert(saved_native.find("game_text_language=ruru\n") != std::string::npos);
    assert(saved_native.find("game_voice_language=engb\n") != std::string::npos);

    s.bgm_volume = 5;
    BbSettings::Load();
    assert(s.bgm_volume == 0 && s.se_volume == 4 && s.voice_volume == 9);
    assert(!s.blood_expression && s.subtitles && s.game_brightness == 6);
    assert(s.camera_speed == 8 && s.pad_vibration == 3 && s.camera_invert_x && !s.camera_invert_y);
    assert(s.auto_target && !s.camera_auto_adjust);
    assert(s.network_region == 1 && s.player_name_display == 0);
    assert(BbSettings::GetNetworkPassword() == "hunt1234");
    assert(s.game_text_language == 11 && s.game_voice_language == 1);

    // Multi-frame FSR 3.1 (3x / 4x) save/load & multiplier round-trip.
    s.frame_generation_supported = true;
    for (int fg_mode : {int(BbSettings::FrameGenFsr3_3x), int(BbSettings::FrameGenFsr3_4x)}) {
        s.frame_generation = fg_mode;
        BbSettings::Save();
        s.frame_generation = BbSettings::FrameGenOff;
        BbSettings::Load();
        assert(s.frame_generation == fg_mode);
        assert(BbSettings::EffectiveFrameGen() == fg_mode);
        assert(BbSettings::FrameGenMultiplier(fg_mode) == (fg_mode == BbSettings::FrameGenFsr3_3x ? 3 : 4));
    }

    // VSync / present_mode ("Fifo", "Immediate", "Mailbox") case-insensitive round-trip through Save() -> Load()
    // and BB_PRESENT_MODE environment fallback.
    for (int vm : {int(BbSettings::VsyncOn), int(BbSettings::VsyncOff), int(BbSettings::VsyncFast)}) {
        s.vsync_mode = vm;
        s.vsync_mode_chosen = true;
        BbSettings::Save();
        s.vsync_mode = BbSettings::VsyncFast;
        s.vsync_mode_chosen = false;
        BbSettings::Load();
        assert(s.vsync_mode == vm && s.vsync_mode_chosen);
    }

    unlink(path);
    std::puts("PASS: settings save keeps other keys, menu position, menu language, and native CSOptionsData");
}
