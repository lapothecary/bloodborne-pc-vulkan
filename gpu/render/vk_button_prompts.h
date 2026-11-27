// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the port's button prompt icons on the GPU (gpu/shim/bbport_button_prompts.h) and
// Hybrid GPU UI & Icon Super-Resolution (2x / 4x) for 1-mip 2D UI textures (title logo,
// inventory icons, runes, HUD atlases).
#pragma once

#include <map>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "vulkan/vk_common.h"
#include "textures/image.h"

namespace VideoCore {
class BufferCache;
struct ImageViewInfo;
} // namespace VideoCore

namespace Vulkan {

class Instance;
class Runtime;
class Scheduler;

class ButtonPrompts {
public:
    ButtonPrompts(const Instance& instance, Scheduler& scheduler, VideoCore::BufferCache& buffer_cache);
    /// Lightweight constructor without BufferCache (for UI upscale cache unit tests).
    ButtonPrompts(const Instance& instance, Scheduler& scheduler);
    ~ButtonPrompts();

    /// The icons for the device in use, uploaded when they changed (the draw recording thread;
    /// ends the render pass then, once per change of device or bindings).
    void Update();

    /// Whether any icon is the port's now.
    bool Active() const noexcept {
        return active;
    }

    /// The port's icon for the game's icon `icon` (Image::prompt_icon), viewed as `info` views
    /// the game's; null: the game's own.
    vk::ImageView View(u32 icon, const VideoCore::ImageViewInfo& info);

    /// The copy of the HUD atlas `image` (Image::prompt_atlas) with the port's icons in its
    /// cells, viewed as `info` views the game's; null: the game's own. Made (in stream order, out
    /// of a render pass) when the icons or the atlas's bytes changed.
    vk::ImageView AtlasView(Runtime& runtime, VideoCore::Image& image,
                            const VideoCore::ImageViewInfo& info);

    /// True if `image` is a 1-mip 2D block-compressed UI texture eligible for UI Super-Resolution.
    bool IsUiTexture(const VideoCore::Image& image) const noexcept;

    /// Returns the 2x/4x upscaled view of 1-mip 2D UI texture `image` (logos, item/weapon/armor
    /// icons, HUD widgets), computed once per upload on the GPU via `ui_upscale.comp` and cached
    /// in VRAM within a strict byte budget.
    vk::ImageView UiView(Runtime& runtime, VideoCore::Image& image,
                         const VideoCore::ImageViewInfo& info);

    /// Invoked when TextureCache retires/destroys an image so its upscaled UI/atlas copy and
    /// source VkImageView are immediately released instead of leaking VRAM.
    void OnImageDeleted(u64 image_uid);

    /// Once-per-frame cache maintenance: releases retired GPU-free images, evicts idle upscaled
    /// UI/atlas copies, and clears the UI upscale cache immediately if `ui_upscale` is disabled.
    void Maintenance();

    /// Live byte size and entry count of the GPU-upscaled UI texture cache (for diagnostics/tests).
    [[nodiscard]] u64 UiCacheBytes() const noexcept;
    [[nodiscard]] std::size_t UiCacheCount() const noexcept;
    [[nodiscard]] std::size_t RetiredCount() const noexcept;

    /// Scale factor for 1-mip UI textures (0 = skip upscaling and sample native BC directly).
    static u32 ComputeUiUpscaleFactor(u32 width, u32 height, bool low_vram) noexcept;
    /// Scale factor for HUD button-prompt atlases (1 = native 1x, 2 = 2x upscaled).
    static u32 ComputeAtlasUpscaleFactor(u32 width, u32 height, bool upscale_on,
                                         bool low_vram) noexcept;
    /// Byte budget for cached uncompressed RGBA8 UI upscaled textures.
    static u64 ComputeUiCacheBudgetBytes(bool low_vram) noexcept;
    /// Entry cap for cached uncompressed RGBA8 UI upscaled textures.
    static std::size_t ComputeUiCacheMaxEntries(bool low_vram) noexcept;

private:
    struct Atlas {
        VideoCore::HostImage image;
        vk::Image src_image{};
        vk::UniqueImageView src_view;
        vk::UniqueImageView storage_view;
        u32 upload = 0;
        u32 generation = 0;
        u32 scale = 0;
        u64 last_used_tick = 0;
        u64 last_used_second = 0;
        std::map<std::tuple<bool, bool, u32>, vk::UniqueImageView> views;
    };

    bool EnsureUpscaleViews(Atlas& entry, VideoCore::Image& src_image, u32 dst_w, u32 dst_h);
    void RetireOrDestroyLocked(Atlas&& entry, std::vector<Atlas>& destroy_now);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::BufferCache* buffer_cache = nullptr;
    VideoCore::HostImage image;
    vk::UniqueDescriptorSetLayout upscale_set_layout;
    vk::UniquePipelineLayout upscale_pipeline_layout;
    vk::UniquePipeline upscale_pipeline;
    u32 layers = 0;
    u32 generation = 0;
    bool active = false;
    std::vector<bool> drawn;
    std::vector<u32> small;  ///< the icons at 1x game size (32x32), a layer after another
    std::vector<u32> medium; ///< the icons at 2x size (64x64) for upscaled HUD atlases
    mutable std::mutex cache_mutex;
    std::unordered_map<u64, Atlas> atlases;     ///< HUD prompt atlases by Image::image_uid
    std::unordered_map<u64, Atlas> ui_textures; ///< Upscaled 1-mip UI textures by Image::image_uid
    std::vector<Atlas> retired_entries;         ///< Deferred GPU-safe destruction queue
    u64 ui_textures_bytes = 0;
    u64 last_maintenance_second = 0;
    std::map<std::tuple<u32, bool, bool, u32>, vk::UniqueImageView> views;
};

} // namespace Vulkan
