// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "common/types.h"

namespace BbMenuEngine {

// ============================================================================
// 1. Complete Scaleform GFx Sprite Tree of `dvdroot_ps4/menu/optionsetting.gfx`
// ============================================================================
// Extracted directly from SWF DefineSprite (tag 39), PlaceObject2/3 (tags 26/70),
// and FrameLabel (tag 43) records in `CUSA03173/dvdroot_ps4/menu/optionsetting.gfx`:
//
// - Sprite 107 (`EnvironmentSetting` / Display & Effects):
//     frames=19, labels=["FadeIn", "Loop", "FadeOut"]
//     children: StatusBar(37), Item_0_0..Item_4_0(97), Item_5_0(100=Defaults),
//               StaticText_111010, StaticText_111011, StaticText_111020
// - Sprite 138 (`OptionSetting_fla._Option_ItemList_71` / `ControllSetting`):
//     frames=19, labels=["FadeIn", "Loop", "FadeOut"]
//     children: StatusBar(37), Item_0_0..Item_5_0(97), Item_6_0(100=Defaults),
//               StaticText_113020
// - Sprite 37 (`StatusBar`):
//     children: LineHelp(35), KeyGuide(36)
// - Sprite 97 (Option Row Slot `Item_0_0` .. `Item_5_0`):
//     children:
//       - `Text_0` (id=38, depth=1)       -> Left-column row label text
//       - `Cursor` (id=43, depth=6)       -> Left-column 30-frame row highlight glow
//       - `Widgets` (id=96, depth=8)      -> Right-column widget container
// - Sprite 100 (Defaults Button Row `Item_5_0` / `Item_6_0`):
//     children: `Text_0` (id=99, d=5), `Cursor` (id=82, d=6)
// - Sprite 96 (`Widgets` container inside each row `Item_%d_0`):
//     children:
//       1. `Widgets/TextInput` (Sprite 51, depth=1):
//            children: `Text_0`(46, d=5), `TextOnEmpty`(47, d=6), `Cursor`(50, d=7)
//       2. `Widgets/ComboBox` (Sprite 57, depth=9) — 2-choice `<| Off / On |>` widget:
//            children:
//              - ornament (id=45, depth=1)
//              - `Item_0_0` (Sprite 53, depth=5) -> contains ONLY `Text_0` (id=52, depth=1)
//              - `Cursor` (Sprite 56, depth=7, 30 frames) -> contains BOTH the golden glow
//                (id=49, d=1) AND the left/right `<|` / `|>` arrows (id=55 at d=4 and d=6)!
//                NOTE: `Widgets/ComboBox` does NOT have `Item_0_0/Cursor` or `LeftItem_0`/`RightItem_0`.
//                Hiding/showing `Widgets/ComboBox/Cursor` directly toggles both the golden glow
//                and the `<|` / `|>` arrows!
//       3. `Widgets/Slider` (Sprite 74, depth=17) — `0..10` bar + integer slider widget:
//            children:
//              - ornament (id=62, depth=1)
//              - `Text_0` (id=63, depth=5)       -> Numeric value text (`0..10`)
//              - `LeftArrow` (Sprite 64, depth=6)-> `<|` arrow (30 frames)
//              - `RightArrow` (Sprite 64, depth=8)-> `|>` arrow (30 frames)
//              - `Cursor` (Sprite 67, depth=10)  -> Golden highlight glow (30 frames)
//              - `Bar` (Sprite 73, depth=12, 100 frames, label="normal") -> Frame `1 + val * 10`
//       4. `Widgets/DropdownList` (Sprite 95, depth=30) — Multi-choice pop-up dropdown widget:
//            children:
//              - `CurrentItem` (Sprite 94, depth=71) -> `Text_0` (id=93, d=1)
//              - `ItemList` (Sprite 92, depth=1)     -> Pop-up window containing:
//                  - background frame (Sprite 80, d=1)
//                  - `Item_0_0` .. `Item_5_0` (Sprite 83: `Text_0` d=5, `Cursor` d=6)
//                  - `ScrollBarV` (Sprite 91: `ArrowDown` d=1, `ArrowUp` d=3, `Bar` d=5)
//
//   Title Screen Layout (`menu/title.gfx`, `title_dlc.gfx`, `title_dlc_eu.gfx`):
//     Root places `Top` (Sprite 43, tx=0, ty=0):
//       - `StaticRegionText_401001` (Sprite 24, d=1, ty=1000.95) — bottom copyright line
//       - `CommandList` (Sprite 32, d=7, tx=1179.45, ty=703.50)  — Title Menu item list:
//           Vanilla places 5 rows (`Item_4_0 .. Item_0_0`, Sprite 31, tx=-514.15, ty=+112.00..-47.40,
//           step=+39.85 px / +797 twips, depths 1, 6, 11, 16, 21). `src/runtime_file.c`
//           (`fix_title_layout`) appends `Item_5_0` at `ty=+151.85` (`+3037` twips, stage y=855.35)
//           and bumps `Item_4_0..Item_0_0` depths by +5 so all 6 Title Menu rows (`Continue`,
//           `Load Game`, `New Game`, `System`, `Log In` / `Information`, `Exit Game`) display
//           simultaneously without scrolling.
//       - `Sprite 31` (Title row template): `EditText 27` (`Text_0`, d=10) + `Sprite 30` (`Cursor`, d=14, 30f)
//       - `PressStart` (Sprite 34, d=29), `ProgressInfo` (Sprite 39, d=34), `Info` (Sprite 42, d=38)

// ============================================================================
// 2. Decompiled `eboot.bin` VMA Constants (file_offset = VMA + 0x4000)
// ============================================================================
namespace Vma {
// Message repository lookup & text helpers
inline constexpr u64 MsgLookup            = 0x0f6f340; // (repo, table, category, id) -> const char16_t*
inline constexpr u64 MakeText             = 0x1ae8cc0; // (out_entry 0x40B, category, id)

// Title Screen menu builder (`sub_1b4a3d0`, FMG 0xc8 IDs 401301..401307) & CommandList binding
inline constexpr u64 StartupMenuBuilder   = 0x1b39030; // Play Offline / Play Online 2-item startup prompt
inline constexpr u64 TitleMenuBuilder     = 0x1b4a3d0; // Main Title Menu builder (Continue/Load/New/System/LogIn)
inline constexpr u64 AddActionItem        = 0x1b4c9a0; // (menu, name_help_texts, action_fn, out)
inline constexpr u64 BindCommandList      = 0x1bea820; // (out_step, menu, cmd_list_mc) -> binds items to Top.CommandList
inline constexpr u64 TitleMenuBindRet     = 0x1b4adb8; // Return VMA of BindCommandList call inside TitleMenuBuilder

// Scaleform GFx MovieClip (`0x60`-byte wrapper) & Value operations
inline constexpr u64 SetMovieClipPosition = 0x1acb8c0; // (GFxMovieClip* mc, int x, int y)
inline constexpr u64 SetMovieClipScale    = 0x1acba50; // (GFxMovieClip* mc, float scale_x, float scale_y)
inline constexpr u64 GetSubMovieClip      = 0x1adb220; // (GFxMovieClip* out, const GFxMovieClip* parent, const char* fmt, ...)
inline constexpr u64 LookupChildPath      = 0x1f6c8a0; // (GFxValueHolder* out, const GFxValueHolder* parent, const String* path)
inline constexpr u64 SetMovieClipVisible  = 0x1adb7f0; // vtable+0x08 on GFxMovieClip: (GFxMovieClip* mc, bool visible)
inline constexpr u64 SetMovieClipIntText  = 0x1ad9e90; // (GFxMovieClip* mc_plus_8, int value)
inline constexpr u64 ScrollBarVInit       = 0x1adbd20; // (ScrollBarV* self, const GFxMovieClip* clip, int visible_rows)
inline constexpr u64 ScrollBarVSetScroll  = 0x1adc580; // (ScrollBarV* self, int scroll_index)
inline constexpr u64 ScrollBarVSetTotal   = 0x1adc670; // (ScrollBarV* self, int total_items)
inline constexpr u64 MovieClipBaseVtable  = 0x539a070; // Base vtable (+0x10) restored on stack GFxMovieClip destruction

// Scaleform `ListComponent` (`0x2b8` bytes)
inline constexpr u64 ListCtor             = 0x1ace690; // (ListComponent* self, const GFxMovieClip* clip, bool wrap)
inline constexpr u64 ListInit             = 0x1aceca0; // (ListComponent* self)
inline constexpr u64 ListHideAllCursors   = 0x1acf0e0; // (ListComponent* self)
inline constexpr u64 ListSetItemCount     = 0x1acf2c0; // (ListComponent* self, u32 count)
inline constexpr u64 ListPlayCursorLoop   = 0x1acf480; // (ListComponent* self) -> GotoAndStopLabel("Loop") on current item Cursor
inline constexpr u64 ListSelect           = 0x1acf6b0; // (ListComponent* self, u32 index) -> selects index & shows its Cursor
inline constexpr u64 ListUnfocus          = 0x1acfb20; // (ListComponent* self) -> GotoAndStopLabel("Normal") & hides arrows
inline constexpr u64 ListShowSelected     = 0x1acfd80; // (ListComponent* self) -> GotoAndStopLabel("Selected") on current item Cursor
inline constexpr u64 ListHandleInput      = 0x1acff30; // (ListComponent* self, const u8* input)
inline constexpr u64 ListRefreshItems     = 0x1ad06e0; // (ListComponent* self) -> invokes callback at +0x280 to populate Text_0
inline constexpr u64 ListGetItemClip      = 0x1ad1f50; // (GFxMovieClip* out, const ListComponent* self, int relative_slot)

// OptionPage & OptionRow constructors and navigation
inline constexpr u64 AddPageItem          = 0x1b4e2a0; // (menu, name_help_texts, page_factory, out)
inline constexpr u64 FactoryToFunction    = 0x1b6ed90; // ({vtable, fn}, out std::function)
inline constexpr u64 CreatePage           = 0x1bb4c70; // (out, arg, content std::function)
inline constexpr u64 OpenPage             = 0x1b20900; // (a, b, layout_name, rows_fn, 0, 0)
inline constexpr u64 GetNextRowMovieClip  = 0x1b2ba20; // (GFxMovieClip* out, OptionPage* page) -> looks up "Item_%d_0" on page+0xdf0
inline constexpr u64 SelectRowCtor        = 0x1b29370; // (OptionPage* page, texts, int* val, val_list, int* def_val) -> 0xcf0B row
inline constexpr u64 PairRowCtor          = 0x1b2a100; // (OptionPage* page, texts, u8* val, pair_list, u8* def_val) -> 0x430B row
inline constexpr u64 SliderRowCtor        = 0x1b2ac00; // (OptionPage* page, texts, u8* val, u8* def_val) -> 0xce0B row
inline constexpr u64 ListRowCtor          = 0x1b78ab0; // Pop-up list row builder
inline constexpr u64 ListPushChoice       = 0x1b1bbd0; // (val_list, {u32 val, text})
inline constexpr u64 PairPushChoice       = 0x1b2c200; // (pair_list, {u8 val, text})
inline constexpr u64 SelectRowHandleInput = 0x1b26b00; // (SelectRow* row, u64 input, ..., double frame_time)
inline constexpr u64 SelectRowShowValue   = 0x1b2b7c0; // (SelectRow* row) -> updates CurrentItem/Text_0
inline constexpr u64 PairRowRefreshItems  = 0x1b263b0; // Callback at PairRow+0x368 for ListRefreshItems
inline constexpr u64 UpdatePageHelpBar    = 0x1b1ab20; // (OptionPage* page) -> updates StatusBar/LineHelp
inline constexpr u64 UpdatePageButtonGuide= 0x1ad6450; // (OptionPage* page) -> updates StatusBar/KeyGuide
inline constexpr u64 OptionsPageNavigate  = 0x1b1b520; // (OptionPage* page, u8* input, float frame_time)
inline constexpr u64 ControlsContent      = 0x1b211a0; // (OptionPage* page, u8* game_settings)

// Menu pad/mouse input predicates (const u8* input)
inline constexpr u64 GetHorizontalStep    = 0x1adfdd0; // (const u8* input) -> int32_t (-1, 0, +1)
inline constexpr u64 IsConfirmPressed     = 0x1adfec0; // (const u8* input) -> bool
inline constexpr u64 IsLeftPressed        = 0x1ae0890; // (const u8* input) -> bool
inline constexpr u64 IsRightPressed       = 0x1ae0970; // (const u8* input) -> bool

// GameDataMan, CSOptionsData (`0x80`B at `userdata0010 + 0x04`), and FrpgSaveLoadMan
inline constexpr u64 GameDataManPtr          = 0x553b130; // GameDataMan** (+0x48 = CSOptionsData*)
inline constexpr u64 FrpgSaveLoadManPtr      = 0x5556678; // FrpgSaveLoadMan** (+0x1548 = req_save_sys, +0x1554 = save_state)
inline constexpr u64 DeserializeOptionsData  = 0x14ef0b0; // (CSOptionsData* self, DLStream* stream) -> int
inline constexpr u64 CommitPageToOptions     = 0x14ef550; // (CSOptionsData* dst, const OptionPageSettings* src)
inline constexpr u64 CopyOptionsToPage       = 0x14ef730; // (const CSOptionsData* src, OptionPageSettings* dst)
inline constexpr u64 NotifyOptionPageChanged = 0x1b23ee0; // (OptionPage* page) -> invokes [page + 0xea0] -> CommitPageToOptions
inline constexpr u64 RequestSaveSystemData   = 0x156d8a0; // (bool skip_profile, bool clear_only, bool arg)
} // namespace Vma

// ============================================================================
// 3. Decompiled Scaleform GFx & Menu Struct Layouts
// ============================================================================

/// Scaleform GFx Value (`0x30` bytes returned by `(**inner)(&gfval, inner)`).
struct alignas(16) GFxValue {
    u64 reserved0 = 0;       // +0x00
    u64 node_prev = ~0ULL;   // +0x08 (passed as 2nd arg to mgr->ReleaseValue)
    u64 node_next = ~0ULL;   // +0x10
    void* mgr = nullptr;     // +0x18: Scaleform GFx::MovieImpl / ObjectManager
    u32 flags = 0;           // +0x20: & 0x8f = valid object, & 0x40 = holds refcount
    u32 pad24 = 0;           // +0x24
    u64 obj = 0;             // +0x28: Scaleform ASMovieClip / TextField handle
};
static_assert(sizeof(GFxValue) == 0x30);

/// Stack/inline Scaleform MovieClip wrapper (`0x60` bytes).
/// Note: When embedded inside a row object (e.g. `SelectRow` at `row + 0x10`),
/// the 0x60-byte `GFxMovieClip` starts at `row + 0x10`.
struct alignas(16) GFxMovieClip {
    u64 vtable = 0;          // +0x00: vtable (SetVisible is at vtable + 0x08)
    void* prev = nullptr;    // +0x08: intrusive tree self-pointer (initialized to `this`)
    void* next = nullptr;    // +0x10: intrusive tree self-pointer (initialized to `this`)
    void* self_ref = nullptr;// +0x18: self-pointer (`this`), whose vtable[0] populates GFxValue
    void* context = nullptr; // +0x20: GFx movie context
    u64 holder_vtbl = 0;     // +0x28: inner value holder vtable (`0x535ac90`)
    u64 node_prev = ~0ULL;   // +0x30: refcount node prev (passed to mgr->Release)
    u64 node_next = ~0ULL;   // +0x38: refcount node next
    void* mgr = nullptr;     // +0x40: Scaleform ObjectManager
    u32 flags = 0;           // +0x48: & 0x40 = needs mgr->Release(mgr, &node_prev, obj)
    u32 pad4c = 0;           // +0x4c
    u64 obj = 0;             // +0x50: Scaleform ASMovieClip handle
    u64 extra = 0;           // +0x58

    u8* Raw() { return reinterpret_cast<u8*>(this); }
    const u8* Raw() const { return reinterpret_cast<const u8*>(this); }
};
static_assert(sizeof(GFxMovieClip) == 0x60);

/// Offsets inside `SelectRow` (`0xcf0` bytes, constructed by `0x1b29370`).
namespace SelectRowLayout {
inline constexpr std::size_t kDropdownClip  = 0x010; // GFxMovieClip for "Widgets/DropdownList" (0x60 bytes)
inline constexpr std::size_t kCurrentText   = 0x0a0; // GFxMovieClip for "CurrentItem/Text_0" (0x60 bytes)
inline constexpr std::size_t kListComponent = 0x100; // Embedded ListComponent for popup "ItemList" (0x2b8 bytes)
inline constexpr std::size_t kIsPopupOpen   = 0x3c0; // u8: 1 while popup dropdown is open, 0 when closed
inline constexpr std::size_t kValuePtr      = 0x3c8; // int*: pointer to bound setting value
inline constexpr std::size_t kCurrentValue  = 0x3d0; // int: current value
inline constexpr std::size_t kDefaultValue  = 0x3d4; // int: default value
inline constexpr std::size_t kChoicesArray  = 0x3d8; // 8-aligned array of 32 choice entries (0x48 bytes each)
inline constexpr std::size_t kChoiceStride  = 0x048; // {u32 value, pad, UTF-16 String[0x38]}
inline constexpr std::size_t kChoicesCount  = 0xce0; // u64: number of valid choices (1..32)
} // namespace SelectRowLayout

/// Offsets inside `OptionPage` (constructed by `0x1b20900`).
namespace OptionPageLayout {
inline constexpr std::size_t kRowListComponent = 0xa70; // ListComponent managing the 6 visible rows + Defaults
inline constexpr std::size_t kSelectedRowIndex = 0xae4; // int32_t: selected row index (when +0xae8 < 0)
inline constexpr std::size_t kHoveredRowIndex  = 0xae8; // int32_t: hovered/override row index (-1 if none)
inline constexpr std::size_t kRootMovieClip    = 0xdf0; // GFxMovieClip for page root ("Item_%d_0" parent)
inline constexpr std::size_t kRowsVectorBegin  = 0xe58; // OptionRowEntry* begin (stride 0x90)
inline constexpr std::size_t kRowsVectorEnd    = 0xe60; // OptionRowEntry* end
inline constexpr std::size_t kRowEntryStride   = 0x090; // {Text name[0x40], Text help[0x40], void* row_widget, u64 flags}
inline constexpr std::size_t kOnChangeCallback = 0xea0; // Functor invoked by NotifyOptionPageChanged (0x1b23ee0)
inline constexpr std::size_t kPageSettings     = 0xeb0; // Inline OptionPageSettings working copy
} // namespace OptionPageLayout

/// Decompiled `CSOptionsData` (`0x80` bytes stored verbatim at `SPRJ0005/userdata0010 + 0x04..+0x83`
/// after `u32 version = 0x1a` at `+0x00`, and pointed to by `(*0x553b130) + 0x48`).
/// Verified instruction-for-instruction against `InitDefaults` (`0x14eeaa4`), `Deserialize` (`0x14ef0b0`),
/// `CommitPageToOptions` (`0x14ef550`), and `CopyOptionsToPage` (`0x14ef730`).
struct CSOptionsData {
    u8 camera_speed        = 5;  // +0x00: Game.Option.Control.RotSpeed (0..10, default 5)
    u8 pad_vibration       = 10; // +0x01: Game.Option.Control.Rumble (0..10, default 10)
    u8 brightness          = 5;  // +0x02: Game.Option.Disp.Brightness (0..10, default 5)
    u8 sound_type          = 2;  // +0x03: Game.Option.Sound.SoundType (0..2, default 2)
    u8 music_volume        = 7;  // +0x04: Game.Option.Sound.MusicVol (0..10, default 7; read live at 0x17f4e83)
    u8 se_volume           = 7;  // +0x05: Game.Option.Sound.SeVol (0..10, default 7)
    u8 voice_volume        = 7;  // +0x06: Game.Option.Sound.VoiceVol (0..10, default 7)
    u8 hud_display         = 1;  // +0x07: internal HUD flag (default 1, untouched by menu)
    u8 blood_expression    = 1;  // +0x08: Game.Option.Disp.Blood (0 = Off, 1 = On; default 1)
    u8 subtitles           = 1;  // +0x09: Game.Option.Disp.Subtitle (0 = Off, 1 = On; default 1)
    u8 camera_up_down      = 1;  // +0x0a: Game.Option.Control.RotUD (1 = Normal, 0 = Inverted; default 1)
    u8 camera_left_right   = 1;  // +0x0b: Game.Option.Control.RotLR (1 = Normal, 0 = Inverted; default 1)
    u8 auto_target         = 0;  // +0x0c: Auto-Target Switch (0 = Off, 1 = On; default 0)
    u8 camera_auto_adjust  = 1;  // +0x0d: Camera Wall Auto-Adjust (0 = Off, 1 = On; default 1)
    u8 unk_0e              = 0;  // +0x0e
    u8 unk_0f              = 0;  // +0x0f
    u8 network_region      = 0;  // +0x10: Matching Regions (0 = Local, 1 = Worldwide; default 0)
    u8 player_name_display = 1;  // +0x11: Player Name Display (0 = Character Name, 1 = Online ID; default 1)
    u8 unk_12              = 0;  // +0x12
    u8 unk_13              = 0;  // +0x13: cleared to 0 on Deserialize
    u8 pad_14[4]           = {}; // +0x14..+0x17
    s32 text_language      = -1; // +0x18: Text language index in GAME_LANGUAGES (0..20, or -1 = system default)
    s32 voice_language     = -1; // +0x1c: Voice language index in GAME_LANGUAGES (0..7, or -1 = system default)
    char16_t password[9]   = {}; // +0x20..+0x31: Network password (up to 8 UTF-16LE code units + NUL at +0x30)
    u8 reserved_32[0x4e]   = {}; // +0x32..+0x7f
};
static_assert(sizeof(CSOptionsData) == 0x80);
static_assert(offsetof(CSOptionsData, music_volume) == 0x04);
static_assert(offsetof(CSOptionsData, blood_expression) == 0x08);
static_assert(offsetof(CSOptionsData, camera_up_down) == 0x0a);
static_assert(offsetof(CSOptionsData, camera_auto_adjust) == 0x0d);
static_assert(offsetof(CSOptionsData, network_region) == 0x10);
static_assert(offsetof(CSOptionsData, player_name_display) == 0x11);
static_assert(offsetof(CSOptionsData, text_language) == 0x18);
static_assert(offsetof(CSOptionsData, voice_language) == 0x1c);
static_assert(offsetof(CSOptionsData, password) == 0x20);

/// Working-copy layout inside `OptionPage + 0xeb0` (`OptionPageSettings`, mapped by `0x14ef550` / `0x14ef730`).
/// Note that `CSOptionsData::hud_display` (`+0x07`) is omitted here, shifting `+0x08..+0x1c` down by 1 / 4 bytes!
namespace OptionPageSettingsLayout {
inline constexpr std::size_t kCameraSpeed       = 0x00; // u8 -> CSOptionsData + 0x00
inline constexpr std::size_t kPadVibration      = 0x01; // u8 -> CSOptionsData + 0x01
inline constexpr std::size_t kBrightness        = 0x02; // u8 -> CSOptionsData + 0x02
inline constexpr std::size_t kSoundType         = 0x03; // u8 -> CSOptionsData + 0x03
inline constexpr std::size_t kMusicVolume       = 0x04; // u8 -> CSOptionsData + 0x04
inline constexpr std::size_t kSeVolume          = 0x05; // u8 -> CSOptionsData + 0x05
inline constexpr std::size_t kVoiceVolume       = 0x06; // u8 -> CSOptionsData + 0x06
inline constexpr std::size_t kBloodExpression   = 0x07; // u8 -> CSOptionsData + 0x08
inline constexpr std::size_t kSubtitles         = 0x08; // u8 -> CSOptionsData + 0x09
inline constexpr std::size_t kCameraUpDown      = 0x09; // u8 -> CSOptionsData + 0x0a
inline constexpr std::size_t kCameraLeftRight   = 0x0a; // u8 -> CSOptionsData + 0x0b
inline constexpr std::size_t kAutoTarget        = 0x0b; // u8 -> CSOptionsData + 0x0c
inline constexpr std::size_t kCameraAutoAdjust  = 0x0c; // u8 -> CSOptionsData + 0x0d
inline constexpr std::size_t kNetworkRegion     = 0x0f; // u8 -> CSOptionsData + 0x10
inline constexpr std::size_t kPlayerNameDisplay = 0x10; // u8 -> CSOptionsData + 0x11
inline constexpr std::size_t kTextLanguage      = 0x14; // s32 -> CSOptionsData + 0x18
inline constexpr std::size_t kVoiceLanguage     = 0x18; // s32 -> CSOptionsData + 0x1c
inline constexpr std::size_t kPasswordString    = 0x20; // DLBasicString<char16_t> -> CSOptionsData + 0x20
} // namespace OptionPageSettingsLayout

/// Offsets inside `FrpgSaveLoadMan` (`*0x5556678`).
namespace FrpgSaveLoadManLayout {
inline constexpr std::size_t kActiveSaveSlot        = 0x1490; // s32 (-1 on Title Screen, 0..9 in-game)
inline constexpr std::size_t kRequestSaveSlot       = 0x1547; // u8
inline constexpr std::size_t kRequestSaveSystemData = 0x1548; // u8 (polled at 0x193fa3a when save_state == 0)
inline constexpr std::size_t kRequestSaveSystemArg  = 0x1549; // u8
inline constexpr std::size_t kSaveState             = 0x1554; // u32 (0 = Idle, 1..3 = SLSession active)
inline constexpr std::size_t kSaveEnabled           = 0x1578; // u8 (1 once system save initialized)
} // namespace FrpgSaveLoadManLayout

/// Offsets inside FromSoftware's native `ScrollBarV` (`0x178` bytes, initialized by `0x1adbd20`,
/// updated by `0x1adc580` / `0x1adc670` on `Sprite 91` `ScrollBarV`).
namespace ScrollBarVLayout {
inline constexpr std::size_t kRootClip      = 0x000; // GFxMovieClip for ScrollBarV root
inline constexpr std::size_t kArrowUpClip   = 0x058; // GFxMovieClip for "ArrowUp" (d=3, y=-2.5)
inline constexpr std::size_t kBarClip       = 0x0b0; // GFxMovieClip for "Bar" (d=5, x=10.1, y=-1.1, sy=0.96720886)
inline constexpr std::size_t kArrowDownClip = 0x108; // GFxMovieClip for "ArrowDown" (d=1, x=-0.9, y=334.3)
inline constexpr std::size_t kBarOriginY    = 0x160; // float: initial Bar._y (-1.1f)
inline constexpr std::size_t kBarOriginX    = 0x164; // float: initial Bar._x (10.1f)
inline constexpr std::size_t kTrackSpanY    = 0x168; // float: ArrowDown._y - Bar._y (335.4f)
inline constexpr std::size_t kScrollIndex   = 0x16c; // int32_t: current top/scroll index
inline constexpr std::size_t kTotalItems    = 0x170; // int32_t: total item count
inline constexpr std::size_t kVisibleRows   = 0x174; // int32_t: visible row count
inline constexpr float kNativeBarX          = 10.1f;
inline constexpr float kNativeBarOriginY    = -1.1f;
inline constexpr float kNativeTrackSpanY    = 335.4f;
inline constexpr float kNativeBarScaleY     = 0.96720886f;
} // namespace ScrollBarVLayout

// ============================================================================
// 4. Clean C++20 Methods Operating on Scaleform GFx MovieClips
// ============================================================================

inline void SetVisible(u8* mc, bool visible) {
    if (!mc) return;
    const u64 vtbl = *reinterpret_cast<u64*>(mc);
    void* const inner = *reinterpret_cast<void**>(mc + 0x18);
    if (!vtbl || !inner) return;
    auto fn = *reinterpret_cast<void (PS4_SYSV_ABI**)(void*, int)>(vtbl + 0x08);
    if (fn) fn(mc, visible ? 1 : 0);
}

inline void SetPosition(u8* mc, u64 image_base, int x, int y) {
    if (!mc || !image_base || !*reinterpret_cast<void**>(mc + 0x18)) return;
    using Fn = void (PS4_SYSV_ABI*)(void*, int, int);
    reinterpret_cast<Fn>(image_base + Vma::SetMovieClipPosition)(mc, x, y);
}

inline void SetScale(u8* mc, u64 image_base, float scale_x, float scale_y) {
    if (!mc || !image_base || !*reinterpret_cast<void**>(mc + 0x18)) return;
    using Fn = void (PS4_SYSV_ABI*)(void*, float, float);
    reinterpret_cast<Fn>(image_base + Vma::SetMovieClipScale)(mc, scale_x, scale_y);
}

inline void DestroyClip(u8* mc, u64 image_base) {
    if (!mc) return;
    *reinterpret_cast<u64*>(mc) = image_base + Vma::MovieClipBaseVtable;
    const u32 flags = *reinterpret_cast<u32*>(mc + 0x48);
    if (flags & 0x40) {
        void* mgr = *reinterpret_cast<void**>(mc + 0x40);
        const u64 val = *reinterpret_cast<u64*>(mc + 0x50);
        if (mgr) {
            const u64 vtbl = *reinterpret_cast<u64*>(mgr);
            auto rel = *reinterpret_cast<void (PS4_SYSV_ABI**)(void*, void*, u64)>(vtbl + 0x18);
            rel(mgr, mc + 0x30, val);
        }
        *reinterpret_cast<u64*>(mc + 0x40) = 0;
    }
    *reinterpret_cast<u32*>(mc + 0x48) = 0;
}

inline void SyncPortScrollBar(u8* page_root_mc, u64 image_base, int scroll, int visible_slots,
                              int total_items) {
    if (!page_root_mc || !image_base || !*reinterpret_cast<void**>(page_root_mc + 0x18)) return;
    using GetSubFn = void* (PS4_SYSV_ABI*)(void*, const void*, const char*, ...);
    auto get_sub = reinterpret_cast<GetSubFn>(image_base + Vma::GetSubMovieClip);

    GFxMovieClip sb_mc{};
    get_sub(sb_mc.Raw(), page_root_mc, "PortScrollBarV");
    const bool scrollable = total_items > visible_slots && visible_slots > 0;
    SetVisible(sb_mc.Raw(), scrollable);
    if (!scrollable || !sb_mc.self_ref) {
        DestroyClip(sb_mc.Raw(), image_base);
        return;
    }

    const int max_scroll = std::max(1, total_items - visible_slots);
    const int clamped_scroll = std::clamp(scroll, 0, max_scroll);
    const float thumb_frac = std::clamp(float(visible_slots) / float(total_items), 0.10f, 1.0f);
    const float scale_y = thumb_frac * ScrollBarVLayout::kNativeBarScaleY;
    const float travel = ScrollBarVLayout::kNativeTrackSpanY * (1.0f - thumb_frac);
    const int bar_y = int(std::lround(ScrollBarVLayout::kNativeBarOriginY +
                                      travel * (float(clamped_scroll) / float(max_scroll))));

    GFxMovieClip bar_mc{};
    get_sub(bar_mc.Raw(), sb_mc.Raw(), "Bar");
    SetPosition(bar_mc.Raw(), image_base, int(ScrollBarVLayout::kNativeBarX), bar_y);
    SetScale(bar_mc.Raw(), image_base, 1.0f, scale_y);
    DestroyClip(bar_mc.Raw(), image_base);

    GFxMovieClip up_mc{};
    get_sub(up_mc.Raw(), sb_mc.Raw(), "GothicUp");
    SetScale(up_mc.Raw(), image_base, 0.95f / 1.62f, 1.0f);
    SetVisible(up_mc.Raw(), clamped_scroll > 0);
    DestroyClip(up_mc.Raw(), image_base);

    GFxMovieClip down_mc{};
    get_sub(down_mc.Raw(), sb_mc.Raw(), "GothicDown");
    SetScale(down_mc.Raw(), image_base, 0.95f / 1.62f, 1.0f);
    SetVisible(down_mc.Raw(), clamped_scroll < max_scroll);
    DestroyClip(down_mc.Raw(), image_base);

    DestroyClip(sb_mc.Raw(), image_base);
}

template <typename Fn>
inline void WithGFxObject(u8* mc, Fn&& callback) {
    if (!mc) return;
    void* inner = *reinterpret_cast<void**>(mc + 0x18);
    if (!inner) return;
    GFxValue gfval{};
    auto get_val = **reinterpret_cast<void (PS4_SYSV_ABI***)(void*, void*)>(inner);
    get_val(&gfval, inner);
    if ((gfval.flags & 0x8f) && gfval.mgr) {
        const u64 vtbl = *reinterpret_cast<u64*>(gfval.mgr);
        callback(gfval.mgr, vtbl, gfval.obj);
    }
    if ((gfval.flags & 0x40) && gfval.mgr) {
        const u64 vtbl = *reinterpret_cast<u64*>(gfval.mgr);
        auto rel = *reinterpret_cast<void (PS4_SYSV_ABI**)(void*, void*, u64)>(vtbl + 0x18);
        rel(gfval.mgr, &gfval.node_prev, gfval.obj);
        gfval.mgr = nullptr;
    }
}

inline void GotoAndStopLabel(u8* mc, const char* label) {
    if (!label) return;
    WithGFxObject(mc, [&](void* mgr, u64 vtbl, u64 obj) {
        auto fn = *reinterpret_cast<void (PS4_SYSV_ABI**)(void*, u64, const char*, u32)>(vtbl + 0x150);
        fn(mgr, obj, label, 0);
    });
}

inline void GotoAndStopFrame(u8* mc, u64 frame_1based) {
    WithGFxObject(mc, [&](void* mgr, u64 vtbl, u64 obj) {
        auto fn = *reinterpret_cast<void (PS4_SYSV_ABI**)(void*, u64, u64, u32)>(vtbl + 0x158);
        fn(mgr, obj, frame_1based, 1);
    });
}

inline void SetWideText(u8* text_mc, const char16_t* str) {
    WithGFxObject(text_mc, [&](void* mgr, u64 vtbl, u64 obj) {
        auto fn = *reinterpret_cast<void (PS4_SYSV_ABI**)(void*, u64, const char16_t*, u32)>(vtbl + 0x138);
        fn(mgr, obj, str ? str : u"", 1);
    });
}

} // namespace BbMenuEngine
