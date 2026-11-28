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

    // Default hidden XPBD tuning values (2.50x compliance, 0.25x aero lift, 1.50x damping) before user edits.
    assert(std::abs(s.xpbd_compliance.load() - 2.50f) < 1e-3f &&
           std::abs(s.xpbd_aero.load() - 0.25f) < 1e-3f &&
           std::abs(s.xpbd_damping.load() - 1.50f) < 1e-3f);

    // Insert menu geometry, scroll, collapsible sections, cloth engine mode & XPBD/PostFX sliders round-trip.
    s.menu_x = 0.685f;
    s.menu_y = 0.045f;
    s.menu_w = 0.340f;
    s.menu_h = 0.860f;
    s.menu_scroll_y = 642.5f;
    s.menu_sec_controls = false;
    s.menu_sec_developer = true;
    s.cloth_exp_mode = 3; // Mode 1.2 XPBD
    s.cloth_exp_chosen = true;
    s.cloth_exp_min = 4;
    s.cloth_exp_max = 35;
    s.xpbd_compliance = 0.85f;
    s.xpbd_aero = 1.45f;
    s.xpbd_damping = 1.15f;
    s.xpbd_tuning_chosen = true;
    s.native_bloom_intensity = 0.215f;
    s.native_dof_strength = 1.35f;
    s.native_postfx_tuning_chosen = true;
    BbSettings::Save();

    s.menu_x = -1.0f;
    s.menu_w = -1.0f;
    s.menu_scroll_y = 0.0f;
    s.menu_sec_controls = true;
    s.menu_sec_developer = false;
    s.cloth_exp_mode = 0;
    s.xpbd_aero = 1.0f;
    s.native_dof_strength = 1.0f;
    BbSettings::Load();

    assert(std::abs(s.menu_x.load() - 0.685f) < 1e-3f && std::abs(s.menu_y.load() - 0.045f) < 1e-3f);
    assert(std::abs(s.menu_w.load() - 0.340f) < 1e-3f && std::abs(s.menu_h.load() - 0.860f) < 1e-3f);
    assert(std::abs(s.menu_scroll_y.load() - 642.5f) < 0.2f);
    assert(!s.menu_sec_controls.load() && s.menu_sec_developer.load());
    assert(s.cloth_exp_mode.load() == 3 && s.cloth_exp_min.load() == 4 && s.cloth_exp_max.load() == 35);
    assert(std::abs(s.xpbd_compliance.load() - 0.85f) < 1e-3f &&
           std::abs(s.xpbd_aero.load() - 1.45f) < 1e-3f &&
           std::abs(s.xpbd_damping.load() - 1.15f) < 1e-3f);
    assert(std::abs(s.native_bloom_intensity.load() - 0.215f) < 1e-3f &&
           std::abs(s.native_dof_strength.load() - 1.35f) < 1e-2f);

    // Upscaler Mip Bias: Auto default, manual quantization to 0.25 steps, +0.0 normalization, and round-trip.
    assert(BbSettings::IsUpscalerMipBiasAuto(s.upscaler_mip_bias.load()));
    assert(std::abs(BbSettings::QuantizeUpscalerMipBias(-0.63f) - (-0.75f)) < 1e-5f);
    assert(std::abs(BbSettings::QuantizeUpscalerMipBias(-5.0f) - (-2.0f)) < 1e-5f);
    assert(BbSettings::QuantizeUpscalerMipBias(-0.01f) == 0.0f &&
           !std::signbit(BbSettings::QuantizeUpscalerMipBias(-0.01f)));
    s.upscaler_mip_bias = -1.0f;
    BbSettings::Save();
    s.upscaler_mip_bias = BbSettings::Values::UpscalerMipBiasAuto;
    BbSettings::Load();
    assert(!BbSettings::IsUpscalerMipBiasAuto(s.upscaler_mip_bias.load()) &&
           std::abs(s.upscaler_mip_bias.load() - (-1.0f)) < 1e-5f);
    s.upscaler_mip_bias = BbSettings::Values::UpscalerMipBiasAuto;
    BbSettings::Save();
    s.upscaler_mip_bias = -0.5f;
    BbSettings::Load();
    assert(BbSettings::IsUpscalerMipBiasAuto(s.upscaler_mip_bias.load()));

    // Audio Output Mode (0=Stereo, 1=Headphones 7.1 HRTF, 2=Surround 7.1) round-trip.
    assert(s.audio_mode.load() == BbSettings::AudioStereo);
    for (int am : {BbSettings::AudioHrtf, BbSettings::AudioSurround, BbSettings::AudioStereo}) {
        s.audio_mode = am;
        BbSettings::Save();
        s.audio_mode = (am + 1) % BbSettings::AudioModeCount;
        BbSettings::Load();
        assert(s.audio_mode.load() == am);
    }

    unlink(path);
    std::puts("PASS: settings save keeps other keys, menu position, menu language, and native CSOptionsData");
}
