// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <algorithm>
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

// ============================================================================
// 2. Decompiled `eboot.bin` VMA Constants (file_offset = VMA + 0x4000)
// ============================================================================
namespace Vma {
// Message repository lookup & text helpers
inline constexpr u64 MsgLookup            = 0x0f6f340; // (repo, table, category, id) -> const char16_t*
inline constexpr u64 MakeText             = 0x1ae8cc0; // (out_entry 0x40B, category, id)

// Scaleform GFx MovieClip (`0x60`-byte wrapper) & Value operations
inline constexpr u64 GetSubMovieClip      = 0x1adb220; // (GFxMovieClip* out, const GFxMovieClip* parent, const char* fmt, ...)
inline constexpr u64 LookupChildPath      = 0x1f6c8a0; // (GFxValueHolder* out, const GFxValueHolder* parent, const String* path)
inline constexpr u64 SetMovieClipVisible  = 0x1adb7f0; // vtable+0x08 on GFxMovieClip: (GFxMovieClip* mc, bool visible)
inline constexpr u64 SetMovieClipIntText  = 0x1ad9e90; // (GFxMovieClip* mc_plus_8, int value)
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
} // namespace OptionPageLayout

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
