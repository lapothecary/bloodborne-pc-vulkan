// SPDX-License-Identifier: GPL-2.0-or-later
// The layout half of ImageInfo (no renderer dependencies: tested on its own).
#include <algorithm>
#include <bit>

#include "common/assert.h"
#include "textures/format_class.h"
#include "textures/image_info.h"
#include "textures/surface_layout.h"

namespace VideoCore {

bool ImageInfo::IsCompatible(const ImageInfo& info) const {
    return (IsVulkanFormatCompatible(pixel_format, info.pixel_format) ||
            IsVulkanFormatCompatible(info.pixel_format, pixel_format)) &&
           num_samples == info.num_samples && num_bits == info.num_bits;
}

void ImageInfo::UpdateSize() {
    guest_size = 0;
    micro_tiled_mips = 0;
    const bool macro = Gcn::IsMacroTiled(array_mode);
    const u32 thickness = Gcn::GetMicroTileThickness(array_mode);
    for (u32 level = 0; level < resources.levels; ++level) {
        // In blocks for compressed formats; at least one; powers of two when padded.
        const u32 block = props.is_block ? 4 : 1;
        u32 width = std::max((pitch >> level) / block + ((pitch >> level) % block != 0), 1u);
        u32 height =
            std::max((size.height >> level) / block + ((size.height >> level) % block != 0), 1u);
        u32 depth = std::max(size.depth >> level, 1u);
        if (props.is_pow2) {
            width = std::bit_ceil(width);
            height = std::bit_ceil(height);
            depth = std::bit_ceil(depth);
        }

        LevelSize layout;
        u32 level_thickness = 1;
        if (array_mode == Gcn::ArrayMode::ArrayLinearAligned) {
            layout = LinearAlignedSize(width, height, num_bits, num_samples);
        } else if (array_mode == Gcn::ArrayMode::ArrayLinearGeneral) {
            UNREACHABLE_MSG("linear general textures");
        } else if (macro && IsMacroTiledMip(width, height, num_bits, num_samples, tile_mode,
                                            level, alt_tile)) {
            // Tiled modes keep `thickness` depth slices in a tile.
            level_thickness = thickness;
            layout = MacroTiledSize(width, height, num_bits, num_samples, tile_mode, alt_tile);
        } else {
            // Micro tiled, or a macro tiled level smaller than a macro tile (stored 1D, at most
            // four slices thick, as the hardware does).
            level_thickness = std::min(thickness, 4u);
            layout = MicroTiledSize(width, height, level_thickness, num_bits, num_samples);
            if (macro) {
                micro_tiled_mips |= 1u << level;
            }
        }

        auto& mip = mips_layout[level];
        mip.pitch = layout.pitch;
        mip.height = layout.height;
        if (props.is_block) {
            mip.pitch = std::max(mip.pitch * 4, 32u);
            mip.height = std::max(mip.height * 4, 32u);
        }
        // Slices (depth times layers) padded to whole tiles.
        const u32 slices = (depth * resources.layers + level_thickness - 1) / level_thickness *
                           level_thickness;
        mip.size = u32(layout.bytes) * slices;
        mip.offset = guest_size;
        guest_size += mip.size;
    }
}

namespace {

/// Whether a single-level sub-mip with `sub_mode` matches mip `mip` of `parent`.
/// On PS4 GCN, macro-tiled mipmapped surfaces (`Array2DTiledThin1` / `Array2DTiledThick`) store
/// mip levels smaller than a macro-tile (`parent.micro_tiled_mips`) in 1D micro-tiled layout
/// (`Array1DTiledThin1` / `Array1DTiledThick`), so ColorBuffer sub-mip draws bind them with that
/// 1D tile mode rather than the parent's 2D macro-tiled mode.
bool MipArrayModeMatches(const ImageInfo& parent, u32 mip, Gcn::ArrayMode sub_mode) noexcept {
    if (sub_mode == parent.array_mode) {
        return true;
    }
    if (!Gcn::IsMacroTiled(parent.array_mode) || mip >= 32u ||
        (parent.micro_tiled_mips & (1u << mip)) == 0u) {
        return false;
    }
    const u32 thickness = Gcn::GetMicroTileThickness(parent.array_mode);
    return thickness == 1u ? (sub_mode == Gcn::ArrayMode::Array1DTiledThin1)
                           : (sub_mode == Gcn::ArrayMode::Array1DTiledThick);
}

} // namespace

s32 ImageInfo::MipOf(const ImageInfo& info) const {
    if (!IsCompatible(info) || resources.levels != 1) {
        return -1;
    }
    const u32 info_shift = info.props.is_block ? 2 : 0;
    const u32 this_shift = props.is_block ? 2 : 0;

    // The level of `info` this image starts a slice of, with the same pitch in blocks.
    s32 mip = -1;
    for (u32 m = 0; m < info.resources.levels; ++m) {
        if (!MipArrayModeMatches(info, m, array_mode)) {
            continue;
        }
        const auto& level = info.mips_layout[m];
        const VAddr begin = info.guest_address + level.offset;
        const u32 slice_size = level.size / info.resources.layers;
        if (slice_size == 0) {
            continue;
        }
        if (guest_address >= begin && guest_address < begin + level.size &&
            (guest_address - begin) % slice_size == 0 &&
            (pitch >> this_shift) == (level.pitch >> info_shift)) {
            mip = s32(m);
            break;
        }
    }
    if (mip < 0) {
        return -1;
    }

    // The same size in blocks.
    if (std::max(size.width >> this_shift, 1u) != std::max(info.size.width >> (mip + info_shift), 1u) ||
        std::max(size.height >> this_shift, 1u) !=
            std::max(info.size.height >> (mip + info_shift), 1u)) {
        return -1;
    }

    if (info.type == Gcn::ImageType::Color3D && type == Gcn::ImageType::Color2D) {
        // A 2D array copied into a 3D level: as many layers as the level is deep.
        if (resources.layers != std::max(info.size.depth >> mip, 1u)) {
            return -1;
        }
    } else if (type != info.type) {
        return -1;
    }
    return mip;
}

s32 ImageInfo::SliceOf(const ImageInfo& info, s32 mip) const {
    if (!IsCompatible(info) || type != info.type) {
        return -1;
    }
    // The same size and pitch in blocks as that level.
    const u32 info_shift = info.props.is_block ? 2 : 0;
    const u32 this_shift = props.is_block ? 2 : 0;
    const auto& level = info.mips_layout[mip];
    if (std::max(size.width >> this_shift, 1u) != std::max(info.size.width >> (mip + info_shift), 1u) ||
        std::max(size.height >> this_shift, 1u) !=
            std::max(info.size.height >> (mip + info_shift), 1u) ||
        std::max(pitch >> this_shift, 1u) != std::max(level.pitch >> info_shift, 1u)) {
        return -1;
    }
    // Whole slices, at a slice boundary.
    const u32 slice_size = level.size / info.resources.layers;
    if (guest_size % slice_size != 0) {
        return -1;
    }
    const VAddr distance = guest_address - (info.guest_address + level.offset);
    if (distance % guest_size != 0) {
        return -1;
    }
    return s32(distance / guest_size);
}

} // namespace VideoCore
