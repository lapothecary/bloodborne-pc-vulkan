// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_button_prompts.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "bbgpu/shaders/ui_upscale_comp.h"
#include "bbport_button_prompts.h"
#include "bbport_settings.h"
#include "buffers/buffer_cache.h"
#include "vulkan/instance.h"
#include "vulkan/shader_module.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "textures/image_view.h"

namespace Vulkan {
namespace {

struct UpscalePush {
    s32 src_w;
    s32 src_h;
    s32 dst_w;
    s32 dst_h;
    s32 scale_factor;
    s32 is_block;
};

bool IsSrgb(vk::Format format) {
    switch (format) {
    case vk::Format::eBc1RgbaSrgbBlock:
    case vk::Format::eBc1RgbSrgbBlock:
    case vk::Format::eBc2SrgbBlock:
    case vk::Format::eBc3SrgbBlock:
    case vk::Format::eBc7SrgbBlock:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eA8B8G8R8SrgbPack32:
        return true;
    default:
        return false;
    }
}

vk::Format UnormFormat(vk::Format format) {
    switch (format) {
    case vk::Format::eBc1RgbSrgbBlock:
        return vk::Format::eBc1RgbUnormBlock;
    case vk::Format::eBc1RgbaSrgbBlock:
        return vk::Format::eBc1RgbaUnormBlock;
    case vk::Format::eBc2SrgbBlock:
        return vk::Format::eBc2UnormBlock;
    case vk::Format::eBc3SrgbBlock:
        return vk::Format::eBc3UnormBlock;
    case vk::Format::eBc7SrgbBlock:
        return vk::Format::eBc7UnormBlock;
    case vk::Format::eR8G8B8A8Srgb:
        return vk::Format::eR8G8B8A8Unorm;
    case vk::Format::eB8G8R8A8Srgb:
        return vk::Format::eB8G8R8A8Unorm;
    case vk::Format::eA8B8G8R8SrgbPack32:
        return vk::Format::eA8B8G8R8UnormPack32;
    default:
        return format;
    }
}

bool IsSupportedUiFormat(vk::Format format) {
    switch (format) {
    case vk::Format::eBc1RgbUnormBlock:
    case vk::Format::eBc1RgbSrgbBlock:
    case vk::Format::eBc1RgbaUnormBlock:
    case vk::Format::eBc1RgbaSrgbBlock:
    case vk::Format::eBc2UnormBlock:
    case vk::Format::eBc2SrgbBlock:
    case vk::Format::eBc3UnormBlock:
    case vk::Format::eBc3SrgbBlock:
    case vk::Format::eBc4UnormBlock:
    case vk::Format::eBc7UnormBlock:
    case vk::Format::eBc7SrgbBlock:
        return true;
    default:
        return false;
    }
}

void DownsampleIcons(const std::vector<u32>& src, u32 layers, u32 dst_dim, std::vector<u32>& dst) {
    const u32 factor = BbPrompts::IconSize / dst_dim;
    dst.assign(std::size_t(layers) * dst_dim * dst_dim, 0);
    for (u32 layer = 0; layer < layers; ++layer) {
        const u32* icon = src.data() + std::size_t(layer) * BbPrompts::IconSize * BbPrompts::IconSize;
        for (u32 y = 0; y < dst_dim; ++y) {
            for (u32 x = 0; x < dst_dim; ++x) {
                u32 sum[4]{};
                for (u32 dy = 0; dy < factor; ++dy) {
                    for (u32 dx = 0; dx < factor; ++dx) {
                        const u32 texel = icon[(y * factor + dy) * BbPrompts::IconSize + x * factor + dx];
                        for (u32 c = 0; c < 4; ++c) {
                            sum[c] += (texel >> (8 * c)) & 0xff;
                        }
                    }
                }
                u32 out = 0;
                for (u32 c = 0; c < 4; ++c) {
                    out |= (sum[c] + factor * factor / 2) / (factor * factor) << (8 * c);
                }
                dst[(std::size_t(layer) * dst_dim + y) * dst_dim + x] = out;
            }
        }
    }
}

u64 CurrentSteadySeconds() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

constexpr u64 MiB = 1ull << 20;
constexpr std::size_t MaxCachedAtlases = 8;

} // namespace

u32 ButtonPrompts::ComputeUiUpscaleFactor(u32 width, u32 height, bool low_vram) noexcept {
    if (width < 32u || height < 32u || width > 2048u || height > 2048u) {
        return 0u;
    }
    if (low_vram) {
        // On <= 4.5 GiB GPUs, 2048x2048 sheets are already native 2K crisp; upscaling them to
        // uncompressed 4096x4096 RGBA8 would consume 64 MiB for a single UI sheet.
        if (width > 1024u || height > 1024u) {
            return 0u;
        }
        // Keep 4x for small icons/badges (<= 256x256 -> <= 4 MiB); use 2x for 512..1024 sheets
        // (4..16 MiB instead of 16..32 MiB) which already exceeds 1080p/1680x1050 output density.
        return (width <= 256u && height <= 256u) ? 4u : 2u;
    }
    // 4x Super-Resolution for logos (1024x256, 1024x512) and individual icons/widgets (<= 512K px);
    // 2x Super-Resolution for large 1024x1024 / 2048x2048 multi-icon sheets (up to 4096x4096).
    return (width * height <= 1024u * 512u && width <= 1024u && height <= 1024u) ? 4u : 2u;
}

u32 ButtonPrompts::ComputeAtlasUpscaleFactor(u32 width, u32 height, bool upscale_on,
                                             bool low_vram) noexcept {
    if (!upscale_on) {
        return 1u;
    }
    const u32 max_dim = low_vram ? 1024u : 2048u;
    return (width <= max_dim && height <= max_dim) ? 2u : 1u;
}

u64 ButtonPrompts::ComputeUiCacheBudgetBytes(bool low_vram) noexcept {
    static const u64 env_mb = []() -> u64 {
        const char* env = std::getenv("BB_UI_UPSCALE_CACHE_MB");
        if (!env || !*env) {
            return 0ull;
        }
        return std::clamp<u64>(std::strtoull(env, nullptr, 10), 16ull, 1024ull);
    }();
    if (env_mb != 0ull) {
        return env_mb * MiB;
    }
    return (low_vram ? 64ull : 256ull) * MiB;
}

std::size_t ButtonPrompts::ComputeUiCacheMaxEntries(bool low_vram) noexcept {
    return low_vram ? 32u : 128u;
}

ButtonPrompts::ButtonPrompts(const Instance& instance_, Scheduler& scheduler_,
                             VideoCore::BufferCache& buffer_cache_)
    : ButtonPrompts(instance_, scheduler_) {
    buffer_cache = &buffer_cache_;
    BbPrompts::Start();
}

ButtonPrompts::ButtonPrompts(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_},
      image{instance_.GetDevice(), instance_.GetAllocator()} {
    const auto device = instance.GetDevice();
    const std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
        {.binding = 0,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 1,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
    }};
    upscale_set_layout = Check(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::PushConstantRange push_range{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(UpscalePush),
    };
    upscale_pipeline_layout = Check(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &*upscale_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    }));
    const auto module = CompileSPV(UI_UPSCALE_COMP, device);
    upscale_pipeline = Check(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                          .module = module,
                          .pName = "main"},
                .layout = *upscale_pipeline_layout,
            }));
    device.destroyShaderModule(module);
}

ButtonPrompts::~ButtonPrompts() = default;

void ButtonPrompts::Update() {
    BbPrompts::Set set;
    if (!BbPrompts::Update(generation, set)) {
        return;
    }
    generation = set.generation;
    drawn = set.drawn;
    active = std::find(drawn.begin(), drawn.end(), true) != drawn.end();
    if (!active) {
        return; // the game's icons: nothing to upload
    }
    if (!image) {
        layers = set.count;
        image.Create(vk::ImageCreateInfo{
            .flags = vk::ImageCreateFlagBits::eMutableFormat,
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .extent = {BbPrompts::IconSize, BbPrompts::IconSize, 1},
            .mipLevels = 1,
            .arrayLayers = layers,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
    }
    if (set.count != layers) {
        active = false;
        return;
    }
    // Precompute 1x (32x32) and 2x (64x64) icons for HUD atlases.
    DownsampleIcons(set.pixels, layers, BbPrompts::GameIconSize, small);
    DownsampleIcons(set.pixels, layers, BbPrompts::GameIconSize * 2, medium);

    const u64 bytes = u64(set.pixels.size()) * sizeof(u32);
    if (!buffer_cache) {
        return;
    }
    auto& stream = buffer_cache->GetStreamBuffer();
    const u64 offset = stream.Copy(set.pixels.data(), bytes, 16);
    // In stream order, ahead of the draws that sample it; the draws before it are done reading it
    // (the barrier's source stages).
    scheduler.EndRendering();
    scheduler.Record([image = vk::Image{image.image}, buffer = stream.Handle(), offset,
                      layers = layers](vk::CommandBuffer cmdbuf) {
        const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, layers};
        const vk::ImageMemoryBarrier2 to_copy{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = image,
            .subresourceRange = range,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                   .pImageMemoryBarriers = &to_copy});
        const vk::BufferImageCopy copy{
            .bufferOffset = offset,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, layers},
            .imageOffset = {0, 0, 0},
            .imageExtent = {BbPrompts::IconSize, BbPrompts::IconSize, 1},
        };
        cmdbuf.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, copy);
        const vk::ImageMemoryBarrier2 to_sample{
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = image,
            .subresourceRange = range,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                   .pImageMemoryBarriers = &to_sample});
    });
}

vk::ImageView ButtonPrompts::View(u32 icon, const VideoCore::ImageViewInfo& info) {
    if (!active || icon == 0 || icon > drawn.size() || !drawn[icon - 1]) {
        return {};
    }
    const bool array = info.type == Gcn::ImageType::Color2DArray;
    if (!array && info.type != Gcn::ImageType::Color2D) {
        return {};
    }
    const bool srgb = IsSrgb(info.format);
    const u32 mapping = u32(info.mapping.r) | u32(info.mapping.g) << 8 | u32(info.mapping.b) << 16 |
                        u32(info.mapping.a) << 24;
    auto& view = views[{icon, srgb, array, mapping}];
    if (!view) {
        auto [result, created] = instance.GetDevice().createImageViewUnique(vk::ImageViewCreateInfo{
            .image = image.image,
            .viewType = array ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D,
            .format = srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm,
            .components = info.mapping,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, icon - 1, 1},
        });
        if (result != vk::Result::eSuccess) {
            std::printf("Button prompts: no image view (%s)\n", vk::to_string(result).c_str());
            return {};
        }
        view = std::move(created);
    }
    return *view;
}

bool ButtonPrompts::EnsureUpscaleViews(Atlas& entry, VideoCore::Image& src_image,
                                       u32 dst_w, u32 dst_h) {
    const auto device = instance.GetDevice();
    if (!entry.image) {
        entry.image = VideoCore::HostImage{device, instance.GetAllocator()};
        entry.image.Create(vk::ImageCreateInfo{
            .flags = vk::ImageCreateFlagBits::eMutableFormat,
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .extent = {dst_w, dst_h, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
                     vk::ImageUsageFlagBits::eTransferDst,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
    }
    if (!entry.storage_view) {
        auto [res, created] = device.createImageViewUnique(vk::ImageViewCreateInfo{
            .image = entry.image.image,
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        });
        if (res != vk::Result::eSuccess) {
            return false;
        }
        entry.storage_view = std::move(created);
    }
    if (entry.src_image != src_image.GetImage() || !entry.src_view) {
        entry.src_image = src_image.GetImage();
        const vk::Format raw_fmt = UnormFormat(src_image.backing->image.image_ci.format);
        auto [res, created] = device.createImageViewUnique(vk::ImageViewCreateInfo{
            .image = entry.src_image,
            .viewType = vk::ImageViewType::e2D,
            .format = raw_fmt,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        });
        if (res != vk::Result::eSuccess) {
            entry.src_view.reset();
            return false;
        }
        entry.src_view = std::move(created);
    }
    return true;
}

u64 ButtonPrompts::UiCacheBytes() const noexcept {
    std::scoped_lock lk{cache_mutex};
    return ui_textures_bytes;
}

std::size_t ButtonPrompts::UiCacheCount() const noexcept {
    std::scoped_lock lk{cache_mutex};
    return ui_textures.size();
}

std::size_t ButtonPrompts::RetiredCount() const noexcept {
    std::scoped_lock lk{cache_mutex};
    return retired_entries.size();
}

void ButtonPrompts::RetireOrDestroyLocked(Atlas&& entry, std::vector<Atlas>& destroy_now) {
    // Source VkImage may be about to be destroyed by TextureCache; drop src_view immediately.
    entry.src_view.reset();
    entry.src_image = VK_NULL_HANDLE;
    if (entry.last_used_tick == 0 || scheduler.IsFree(entry.last_used_tick)) {
        destroy_now.push_back(std::move(entry));
    } else {
        retired_entries.push_back(std::move(entry));
    }
}

void ButtonPrompts::OnImageDeleted(u64 image_uid) {
    if (image_uid == 0) {
        return;
    }
    std::vector<Atlas> destroy_now;
    {
        std::scoped_lock lk{cache_mutex};
        if (auto it = ui_textures.find(image_uid); it != ui_textures.end()) {
            const u64 bytes = it->second.image.size_bytes;
            ui_textures_bytes = ui_textures_bytes > bytes ? (ui_textures_bytes - bytes) : 0ull;
            RetireOrDestroyLocked(std::move(it->second), destroy_now);
            ui_textures.erase(it);
        }
        if (auto it = atlases.find(image_uid); it != atlases.end()) {
            RetireOrDestroyLocked(std::move(it->second), destroy_now);
            atlases.erase(it);
        }
    }
    // Destroy retired HostImages outside cache_mutex.
    destroy_now.clear();
}

void ButtonPrompts::Maintenance() {
    const bool upscale_on = BbSettings::Get().ui_upscale.load(std::memory_order_relaxed);
    const u64 now_sec = CurrentSteadySeconds();
    const bool low_vram = instance.IsLowVramDevice();
    const u64 ui_idle_secs = low_vram ? 5ull : 20ull;
    const u64 atlas_idle_secs = low_vram ? 15ull : 30ull;

    std::vector<Atlas> destroy_now;
    {
        std::scoped_lock lk{cache_mutex};
        // Always drain deferred GPU-retired entries as soon as the GPU finishes their tick.
        for (std::size_t i = 0; i < retired_entries.size();) {
            if (retired_entries[i].last_used_tick == 0 ||
                scheduler.IsFree(retired_entries[i].last_used_tick)) {
                destroy_now.push_back(std::move(retired_entries[i]));
                retired_entries[i] = std::move(retired_entries.back());
                retired_entries.pop_back();
            } else {
                ++i;
            }
        }
        // Release all upscaled UI textures immediately when UI & Icons Upscale is toggled Off.
        if (!upscale_on && !ui_textures.empty()) {
            for (auto& [uid, entry] : ui_textures) {
                RetireOrDestroyLocked(std::move(entry), destroy_now);
            }
            ui_textures.clear();
            ui_textures_bytes = 0;
        }
        // Once per second: evict idle upscaled UI sheets and prompt atlases (e.g. closed menus,
        // title screen, loading screens, or area transitions) so VRAM stays bounded and flat.
        if (now_sec != last_maintenance_second) {
            last_maintenance_second = now_sec;
            for (auto it = ui_textures.begin(); it != ui_textures.end();) {
                const Atlas& entry = it->second;
                if (entry.last_used_second != 0 && now_sec >= entry.last_used_second + ui_idle_secs &&
                    (entry.last_used_tick == 0 || scheduler.IsFree(entry.last_used_tick))) {
                    const u64 bytes = entry.image.size_bytes;
                    ui_textures_bytes =
                        ui_textures_bytes > bytes ? (ui_textures_bytes - bytes) : 0ull;
                    destroy_now.push_back(std::move(it->second));
                    it = ui_textures.erase(it);
                } else {
                    ++it;
                }
            }
            for (auto it = atlases.begin(); it != atlases.end();) {
                const Atlas& entry = it->second;
                if (entry.last_used_second != 0 &&
                    now_sec >= entry.last_used_second + atlas_idle_secs &&
                    (entry.last_used_tick == 0 || scheduler.IsFree(entry.last_used_tick))) {
                    destroy_now.push_back(std::move(it->second));
                    it = atlases.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
    destroy_now.clear();
}

vk::ImageView ButtonPrompts::AtlasView(Runtime& runtime, VideoCore::Image& image,
                                       const VideoCore::ImageViewInfo& info) {
    if (!active || !buffer_cache || image.prompt_atlas == 0 || info.type != Gcn::ImageType::Color2D ||
        !image.backing) {
        return {};
    }
    const auto cells = BbPrompts::AtlasCells(image.prompt_atlas);
    const auto is_drawn = [&](const BbPrompts::AtlasCell& cell) {
        return cell.icon <= drawn.size() && drawn[cell.icon - 1] && cell.icon <= layers;
    };
    if (std::none_of(cells.begin(), cells.end(), is_drawn)) {
        return {};
    }
    const vk::Format source_format = image.backing->image.image_ci.format;
    if (source_format != vk::Format::eBc7UnormBlock && source_format != vk::Format::eBc7SrgbBlock) {
        return {};
    }
    const u32 width = image.info.size.width, height = image.info.size.height;
    const bool upscale_on = BbSettings::Get().ui_upscale.load(std::memory_order_relaxed);
    const u32 desired_scale =
        ComputeAtlasUpscaleFactor(width, height, upscale_on, instance.IsLowVramDevice());
    const u32 dst_w = width * desired_scale, dst_h = height * desired_scale;
    const u64 uid = image.image_uid;
    const bool srgb = IsSrgb(info.format);
    const u32 mapping = u32(info.mapping.r) | u32(info.mapping.g) << 8 | u32(info.mapping.b) << 16 |
                        u32(info.mapping.a) << 24;

    bool need_dispatch = false;
    vk::ImageView src_view_h{};
    vk::ImageView dst_view_h{};
    vk::Image target_h{};
    vk::ImageView out_view{};
    std::vector<Atlas> destroy_now;
    {
        std::scoped_lock lk{cache_mutex};
        auto it = atlases.find(uid);
        if (it == atlases.end() && atlases.size() >= MaxCachedAtlases) {
            const u64 cur_tick = scheduler.CurrentTick();
            auto victim = atlases.end();
            for (auto cand = atlases.begin(); cand != atlases.end(); ++cand) {
                if (cur_tick > cand->second.last_used_tick + 4u &&
                    scheduler.IsFree(cand->second.last_used_tick)) {
                    if (victim == atlases.end() ||
                        cand->second.last_used_tick < victim->second.last_used_tick) {
                        victim = cand;
                    }
                }
            }
            if (victim != atlases.end()) {
                destroy_now.push_back(std::move(victim->second));
                atlases.erase(victim);
            }
        }
        const bool inserted = (it == atlases.end());
        Atlas& atlas = inserted ? atlases[uid] : it->second;
        if (atlas.scale != desired_scale) {
            if (atlas.image) {
                Atlas old_copy;
                old_copy.image = std::move(atlas.image);
                old_copy.storage_view = std::move(atlas.storage_view);
                old_copy.views = std::move(atlas.views);
                old_copy.last_used_tick = atlas.last_used_tick;
                RetireOrDestroyLocked(std::move(old_copy), destroy_now);
            } else {
                atlas.views.clear();
                atlas.storage_view.reset();
            }
            atlas.scale = desired_scale;
            atlas.upload = 0;
            atlas.generation = 0;
        }
        if (!EnsureUpscaleViews(atlas, image, dst_w, dst_h)) {
            if (inserted) {
                atlases.erase(uid);
            }
            return {};
        }
        if (atlas.upload != image.prompt_upload || atlas.generation != generation) {
            atlas.upload = image.prompt_upload;
            atlas.generation = generation;
            need_dispatch = true;
            src_view_h = *atlas.src_view;
            dst_view_h = *atlas.storage_view;
            target_h = vk::Image{atlas.image.image};
        }
        atlas.last_used_tick = scheduler.CurrentTick();
        atlas.last_used_second = CurrentSteadySeconds();
        auto& view = atlas.views[{srgb, false, mapping}];
        if (!view) {
            auto [result, created] = instance.GetDevice().createImageViewUnique(vk::ImageViewCreateInfo{
                .image = atlas.image.image,
                .viewType = vk::ImageViewType::e2D,
                .format = srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm,
                .components = info.mapping,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            });
            if (result != vk::Result::eSuccess) {
                std::printf("Button prompts: no atlas view (%s)\n", vk::to_string(result).c_str());
                return {};
            }
            view = std::move(created);
        }
        out_view = *view;
    }
    destroy_now.clear();

    if (need_dispatch) {
        const u32 cell_dim = BbPrompts::GameIconSize * desired_scale;
        const auto& icon_pool = (desired_scale == 2u) ? medium : small;
        std::vector<vk::BufferImageCopy> copies;
        std::vector<u32> pixels;
        for (const auto& cell : cells) {
            if (!is_drawn(cell) || cell.x + BbPrompts::GameIconSize > width ||
                cell.y + BbPrompts::GameIconSize > height) {
                continue;
            }
            copies.push_back(vk::BufferImageCopy{
                .bufferOffset = pixels.size() * sizeof(u32),
                .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .imageOffset = {s32(cell.x * desired_scale), s32(cell.y * desired_scale), 0},
                .imageExtent = {cell_dim, cell_dim, 1},
            });
            const u32* icon = icon_pool.data() + std::size_t(cell.icon - 1) * cell_dim * cell_dim;
            pixels.insert(pixels.end(), icon, icon + cell_dim * cell_dim);
        }
        auto& stream = buffer_cache->GetStreamBuffer();
        const u64 offset = stream.Copy(pixels.data(), pixels.size() * sizeof(u32), 16);
        for (auto& copy : copies) {
            copy.bufferOffset += offset;
        }
        scheduler.EndRendering();
        runtime.Transit(&image, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader |
                            vk::PipelineStageFlagBits2::eAllGraphics,
                        vk::AccessFlagBits2::eShaderRead);
        runtime.FlushBarriers();
        scheduler.BumpPushConstantsEpoch();

        const UpscalePush push{
            .src_w = s32(width),
            .src_h = s32(height),
            .dst_w = s32(dst_w),
            .dst_h = s32(dst_h),
            .scale_factor = s32(desired_scale),
            .is_block = 1,
        };
        scheduler.Record([pipeline = *upscale_pipeline, layout = *upscale_pipeline_layout,
                          src_view = src_view_h, dst_view = dst_view_h, target = target_h,
                          buffer = stream.Handle(), copies = std::move(copies),
                          push](vk::CommandBuffer cmdbuf) {
            const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            const vk::ImageMemoryBarrier2 to_storage{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = target,
                .subresourceRange = range,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                       .pImageMemoryBarriers = &to_storage});

            const vk::DescriptorImageInfo src_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = src_view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
            const vk::DescriptorImageInfo dst_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = dst_view,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const std::array<vk::WriteDescriptorSet, 2> writes{{
                {.dstBinding = 0,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &src_info},
                {.dstBinding = 1,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &dst_info},
            }};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
            cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
            cmdbuf.dispatch(u32(push.dst_w + 15) / 16, u32(push.dst_h + 15) / 16, 1);

            const vk::ImageMemoryBarrier2 to_copy{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .image = target,
                .subresourceRange = range,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                       .pImageMemoryBarriers = &to_copy});
            cmdbuf.copyBufferToImage(buffer, target, vk::ImageLayout::eTransferDstOptimal, copies);
            const vk::ImageMemoryBarrier2 to_sample{
                .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .image = target,
                .subresourceRange = range,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                       .pImageMemoryBarriers = &to_sample});
        });
        std::scoped_lock lk{cache_mutex};
        if (auto it = atlases.find(uid); it != atlases.end()) {
            it->second.last_used_tick = scheduler.CurrentTick();
        }
    }
    return out_view;
}

bool ButtonPrompts::IsUiTexture(const VideoCore::Image& image) const noexcept {
    if (!image.backing || !image.info.props.is_block || image.info.props.is_depth ||
        image.info.type != Gcn::ImageType::Color2D ||
        image.info.resources.levels != 1 || image.info.resources.layers != 1 ||
        image.binding.is_target || image.scene_proxy || image.prompt_icon != 0 ||
        VideoCore::True(image.flags & VideoCore::ImageFlagBits::GpuModified)) {
        return false;
    }
    if (!IsSupportedUiFormat(image.backing->image.image_ci.format)) {
        return false;
    }
    const u32 w = image.info.size.width;
    const u32 h = image.info.size.height;
    return w >= 32 && h >= 32 && w <= 2048 && h <= 2048;
}

vk::ImageView ButtonPrompts::UiView(Runtime& runtime, VideoCore::Image& image,
                                    const VideoCore::ImageViewInfo& info) {
    if (info.type != Gcn::ImageType::Color2D || !IsUiTexture(image)) {
        return {};
    }
    const u32 width = image.info.size.width;
    const u32 height = image.info.size.height;
    const bool low_vram = instance.IsLowVramDevice();
    const u32 scale = ComputeUiUpscaleFactor(width, height, low_vram);
    if (scale == 0u) {
        return {};
    }
    const u32 dst_w = width * scale;
    const u32 dst_h = height * scale;
    const u64 est_bytes = u64(dst_w) * u64(dst_h) * 4ull;
    const u64 max_bytes = ComputeUiCacheBudgetBytes(low_vram);
    const std::size_t max_entries = ComputeUiCacheMaxEntries(low_vram);
    if (est_bytes > max_bytes) {
        return {};
    }

    const u64 uid = image.image_uid;
    const u32 cur_upload = image.prompt_upload ? image.prompt_upload : 1u;
    const bool srgb = IsSrgb(info.format);
    const u32 mapping = u32(info.mapping.r) | u32(info.mapping.g) << 8 | u32(info.mapping.b) << 16 |
                        u32(info.mapping.a) << 24;

    bool need_dispatch = false;
    vk::ImageView src_view_h{};
    vk::ImageView dst_view_h{};
    vk::Image target_h{};
    vk::ImageView out_view{};
    std::vector<Atlas> destroy_now;
    {
        std::scoped_lock lk{cache_mutex};
        auto it = ui_textures.find(uid);
        if (it == ui_textures.end()) {
            const u64 cur_tick = scheduler.CurrentTick();
            // Evict LRU GPU-free entries until both entry count and byte budget have headroom.
            while (ui_textures.size() >= max_entries ||
                   ui_textures_bytes + est_bytes > max_bytes) {
                auto victim = ui_textures.end();
                for (auto cand = ui_textures.begin(); cand != ui_textures.end(); ++cand) {
                    if (cur_tick > cand->second.last_used_tick + 4u &&
                        scheduler.IsFree(cand->second.last_used_tick)) {
                        if (victim == ui_textures.end() ||
                            cand->second.last_used_tick < victim->second.last_used_tick) {
                            victim = cand;
                        }
                    }
                }
                if (victim == ui_textures.end()) {
                    // All cached entries are actively in-flight this frame: fall back to native
                    // BC sampling instead of exceeding the VRAM budget.
                    return {};
                }
                const u64 freed = victim->second.image.size_bytes;
                ui_textures_bytes = ui_textures_bytes > freed ? (ui_textures_bytes - freed) : 0ull;
                destroy_now.push_back(std::move(victim->second));
                ui_textures.erase(victim);
            }
        }

        const bool inserted = (it == ui_textures.end());
        Atlas& entry = inserted ? ui_textures[uid] : it->second;
        entry.scale = scale;
        const bool had_image = bool(entry.image);
        if (!EnsureUpscaleViews(entry, image, dst_w, dst_h)) {
            if (inserted) {
                ui_textures.erase(uid);
            }
            return {};
        }
        if (!had_image && entry.image) {
            ui_textures_bytes += entry.image.size_bytes;
        }
        if (entry.upload != cur_upload) {
            entry.upload = cur_upload;
            need_dispatch = true;
            src_view_h = *entry.src_view;
            dst_view_h = *entry.storage_view;
            target_h = vk::Image{entry.image.image};
        }
        entry.last_used_tick = scheduler.CurrentTick();
        entry.last_used_second = CurrentSteadySeconds();
        auto& view = entry.views[{srgb, false, mapping}];
        if (!view) {
            auto [result, created] = instance.GetDevice().createImageViewUnique(vk::ImageViewCreateInfo{
                .image = entry.image.image,
                .viewType = vk::ImageViewType::e2D,
                .format = srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm,
                .components = info.mapping,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            });
            if (result != vk::Result::eSuccess) {
                std::printf("UI upscale: no image view (%s)\n", vk::to_string(result).c_str());
                return {};
            }
            view = std::move(created);
        }
        out_view = *view;
    }
    destroy_now.clear();

    if (need_dispatch) {
        scheduler.EndRendering();
        runtime.Transit(&image, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader |
                            vk::PipelineStageFlagBits2::eAllGraphics,
                        vk::AccessFlagBits2::eShaderRead);
        runtime.FlushBarriers();
        scheduler.BumpPushConstantsEpoch();

        const UpscalePush push{
            .src_w = s32(width),
            .src_h = s32(height),
            .dst_w = s32(dst_w),
            .dst_h = s32(dst_h),
            .scale_factor = s32(scale),
            .is_block = 1,
        };
        scheduler.Record([pipeline = *upscale_pipeline, layout = *upscale_pipeline_layout,
                          src_view = src_view_h, dst_view = dst_view_h, target = target_h,
                          push](vk::CommandBuffer cmdbuf) {
            const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            const vk::ImageMemoryBarrier2 to_storage{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = target,
                .subresourceRange = range,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                       .pImageMemoryBarriers = &to_storage});

            const vk::DescriptorImageInfo src_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = src_view,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };
            const vk::DescriptorImageInfo dst_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = dst_view,
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            const std::array<vk::WriteDescriptorSet, 2> writes{{
                {.dstBinding = 0,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &src_info},
                {.dstBinding = 1,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &dst_info},
            }};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
            cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
            cmdbuf.dispatch(u32(push.dst_w + 15) / 16, u32(push.dst_h + 15) / 16, 1);

            const vk::ImageMemoryBarrier2 to_sample{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .image = target,
                .subresourceRange = range,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                       .pImageMemoryBarriers = &to_sample});
        });
        std::scoped_lock lk{cache_mutex};
        if (auto it = ui_textures.find(uid); it != ui_textures.end()) {
            it->second.last_used_tick = scheduler.CurrentTick();
        }
    }
    return out_view;
}

} // namespace Vulkan
