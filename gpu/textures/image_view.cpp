// SPDX-License-Identifier: GPL-2.0-or-later
#include "textures/image_view.h"

#include <algorithm>

#include "bbport_settings.h"
#include "common/logging/log.h"
#include "shader/resource.h"
#include "textures/image.h"
#include "render/gcn_to_vk.h"
#include "vulkan/instance.h"

namespace VideoCore {

namespace {

namespace ToVk = Vulkan::GcnToVk;

vk::ImageViewType ViewType(Gcn::ImageType type) {
    switch (type) {
    case Gcn::ImageType::Color1D:
        return vk::ImageViewType::e1D;
    case Gcn::ImageType::Color1DArray:
        return vk::ImageViewType::e1DArray;
    case Gcn::ImageType::Color2D:
    case Gcn::ImageType::Color2DMsaa:
        return vk::ImageViewType::e2D;
    case Gcn::ImageType::Color2DArray:
        return vk::ImageViewType::e2DArray;
    case Gcn::ImageType::Color3D:
        return vk::ImageViewType::e3D;
    default:
        UNREACHABLE_MSG("view of image type {}", u64(type));
    }
}

/// Whether Vulkan allows a view of `view` type on an image of `image` type (2D views of 3D
/// images with the 2D-array compatible flag).
bool ViewFitsImage(Gcn::ImageType view, Gcn::ImageType image) {
    switch (view) {
    case Gcn::ImageType::Color1D:
    case Gcn::ImageType::Color1DArray:
        return image == Gcn::ImageType::Color1D;
    case Gcn::ImageType::Color2D:
    case Gcn::ImageType::Color2DArray:
    case Gcn::ImageType::Color2DMsaa:
    case Gcn::ImageType::Color2DMsaaArray:
        return image == Gcn::ImageType::Color2D || image == Gcn::ImageType::Color3D;
    case Gcn::ImageType::Color3D:
        return image == Gcn::ImageType::Color3D;
    default:
        return false;
    }
}

/// One layer gives a 2D view, more an array.
Gcn::ImageType TargetViewType(u32 layers) {
    return layers > 1 ? Gcn::ImageType::Color2DArray : Gcn::ImageType::Color2D;
}

} // namespace

ImageViewInfo::ImageViewInfo(const Gcn::Image& image, const Shader::ImageResource& desc) noexcept
    : is_storage{desc.is_written} {
    auto nfmt = image.GetNumberFmt();
    if (is_storage && nfmt == Gcn::NumberFormat::Srgb) {
        nfmt = Gcn::NumberFormat::Unorm; // storage views have no sRGB formats
    }
    format = ToVk::SurfaceFormat(image.GetDataFmt(), nfmt);
    if (desc.is_depth) {
        format = ToVk::PromoteFormatToDepth(format);
    }
    range.base.level = image.base_level;
    range.base.layer = std::min<u32>(image.base_array, image.NumLayers() - 1);
    range.extent.levels = image.NumViewLevels(desc.is_array);
    range.extent.layers = image.NumViewLayers(desc.is_array);
    type = image.GetViewType(desc.is_array);
    min_lod = u32(image.min_lod);
    // Live Texture Quality mip skip (0=Default, 1=Medium, 2=Low, 3=Very Low) for world 2D BC
    // textures with a full mip chain (>=128x128, >=5 levels, single layer); never touches UI,
    // fonts, LUTs, cubemaps, depth/shadow maps, or render targets.
    if (!is_storage && !desc.is_depth && !image.IsCube() && image.NumLayers() == 1u &&
        (type == Gcn::ImageType::Color2D || type == Gcn::ImageType::Color2DArray) &&
        Gcn::IsBlockCoded(image.GetDataFmt()) && image.width >= 127u && image.height >= 127u &&
        range.extent.levels >= 5u) {
        const int tq =
            std::clamp(BbSettings::Get().texture_quality.load(std::memory_order_relaxed), 0, 3);
        if (tq > 0) {
            const u32 skip = std::min<u32>(static_cast<u32>(tq), range.extent.levels - 2u);
            range.base.level += skip;
            range.extent.levels -= skip;
        }
    }
    if (!is_storage) {
        mapping = ToVk::ComponentMapping(image.DstSelect());
    }
}

ImageViewInfo::ImageViewInfo(const Gcn::ColorBuffer& buffer) noexcept {
    range.base.layer = buffer.BaseSlice();
    range.extent.layers = buffer.NumSlices() - range.base.layer;
    type = TargetViewType(range.extent.layers);
    format = ToVk::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
}

ImageViewInfo::ImageViewInfo(const Gcn::DepthBuffer& buffer, Gcn::DepthView view,
                             Gcn::DepthControl control) {
    format = ToVk::DepthFormat(buffer.z_info.format, buffer.stencil_info.format);
    is_storage = control.depth_write_enable;
    range.base.layer = view.slice_start;
    range.extent.layers = view.NumSlices() - range.base.layer;
    type = TargetViewType(range.extent.layers);
}

ImageView::ImageView(const Vulkan::Instance& instance, const ImageViewInfo& info_,
                     const Image& image)
    : info{info_} {
    vk::ImageViewUsageCreateInfo usage{.usage = image.usage_flags};
    if (!info.is_storage) {
        usage.usage &= ~vk::ImageUsageFlagBits::eStorage;
    }
    const u32 mip_off = image.host_mip_offset;
    const u32 host_levels = image.HostMipLevels();
    const u32 vk_base_mip = info.range.base.level > mip_off
                                ? std::min(info.range.base.level - mip_off, host_levels - 1u)
                                : 0u;
    const u32 max_avail = host_levels - vk_base_mip;
    const u32 vk_level_count = std::clamp(info.range.extent.levels, 1u, max_avail);

    vk::ImageViewMinLodCreateInfoEXT min_lod{};
    const float eff_min_lod = std::max(0.0f, float(info.min_lod) / 256.0f - float(mip_off));
    if (eff_min_lod > 0.0f && instance.IsImageViewMinLodSupported()) {
        const u32 last_level = vk_base_mip + vk_level_count - 1u;
        min_lod.minLod = std::min(eff_min_lod, float(last_level));
        usage.pNext = &min_lod;
    }

    // Depth images sampled through a T# name a color format (R32, R16): view their depth, or
    // their stencil for 8-bit formats.
    vk::Format format = info.format;
    vk::ImageAspectFlags aspect = image.aspect_mask;
    if (image.aspect_mask & vk::ImageAspectFlagBits::eDepth &&
        ToVk::IsFormatDepthCompatible(format)) {
        format = image.info.pixel_format;
        aspect = vk::ImageAspectFlagBits::eDepth;
    }
    if (image.aspect_mask & vk::ImageAspectFlagBits::eStencil &&
        ToVk::IsFormatStencilCompatible(format)) {
        format = image.info.pixel_format;
        aspect = vk::ImageAspectFlagBits::eStencil;
    }

    if (!ViewFitsImage(info.type, image.info.type)) {
        LOG_ERROR(Render_Vulkan, "image view type {} does not fit image type {}", u64(info.type),
                  u64(image.info.type));
    }
    const vk::ImageViewCreateInfo create_info = {
        .pNext = &usage,
        .image = image.GetImage(),
        .viewType = ViewType(info.type),
        .format = instance.GetSupportedFormat(format, image.format_features),
        .components = info.mapping,
        .subresourceRange{
            .aspectMask = aspect,
            .baseMipLevel = vk_base_mip,
            .levelCount = vk_level_count,
            .baseArrayLayer = info.range.base.layer,
            .layerCount = info.range.extent.layers,
        },
    };
    auto [result, view] = instance.GetDevice().createImageViewUnique(create_info);
    ASSERT_MSG(result == vk::Result::eSuccess, "image view: {}", vk::to_string(result));
    image_view = std::move(view);

    const char* aspect_name = aspect & vk::ImageAspectFlagBits::eDepth     ? "depth"
                              : aspect & vk::ImageAspectFlagBits::eStencil ? "stencil"
                                                                           : "color";
    Vulkan::SetObjectName(instance.GetDevice(), *image_view,
                          "view {}x{}x{} {:#x}+{:#x} levels {}..{} layers {}..{} {}",
                          image.info.size.width, image.info.size.height, image.info.size.depth,
                          image.info.guest_address, image.info.guest_size, info.range.base.level,
                          info.range.base.level + info.range.extent.levels - 1,
                          info.range.base.layer,
                          info.range.base.layer + info.range.extent.layers - 1, aspect_name);
}

ImageView::~ImageView() = default;

} // namespace VideoCore
