// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: Rasterizer, dispatches done without running the shader: compute shaders that
// clear metadata (CMASK/HTILE) or copy or clear an image through its memory are recognized by
// their resources and replaced by the tracked metadata state or a Vulkan copy or clear (the
// images live in their own Vulkan memory, not in the buffers the shader would write).

#include "render/raster_internal.h"
#include "textures/direct_textures.h"

namespace Vulkan {

namespace {

/// The shader's own buffers: the page table the info collection pass appends to every shader
/// with guest buffers (BB_LAYER_MEMORY) does not count.
size_t ShaderBufferCount(const Shader::Info& info) {
    size_t count = info.buffers.size();
    if (count != 0 && !info.uses_dma &&
        info.buffers.back().buffer_type == Shader::BufferType::BdaPagetable) {
        --count;
    }
    return count;
}

/// 64 threads, two own buffers, no images: the copy and clear shaders' shape.
bool IsTwoBufferShader(const Gcn::ComputeProgram& cs, const Shader::Info& info) {
    return cs.num_thread_x.full == 64 && ShaderBufferCount(info) == 2 && info.images.empty();
}

} // namespace

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }
    // A shader writing metadata almost always clears it: the dispatch is skipped and the
    // tracked state updated (reading cleared metadata, as HTILE introspection would, needs the
    // full emulation anyway).
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    for (const auto& desc : info.buffers) {
        if (!desc.IsSpecial() && !desc.is_written &&
            texture_cache.IsMeta(desc.GetSharp(info).base_address)) {
            return false; // reads metadata: a copy
        }
    }
    // Encoding tiled metadata (HTILE from a depth image) computes tile addresses with an xor:
    // such a shader is not a clear.
    if (info.has_bitwise_xor) {
        return false;
    }
    for (const auto& desc : info.buffers) {
        if (!desc.IsSpecial() && desc.is_written &&
            texture_cache.ClearMeta(desc.GetSharp(info).base_address)) {
            begin_memo.valid = false;
            return true; // all slices taken as cleared
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline, bool dry_run) {
    if (!pipeline->IsCompute()) {
        return false;
    }
    // Two formatted buffers of one size, one read and one written, a dword per thread: a copy
    // between the memory of two images of the same layout.
    const auto& cs = CsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (!IsTwoBufferShader(cs, info)) {
        return false;
    }
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }
    const Gcn::Buffer buf0 = desc0.GetSharp(info);
    const Gcn::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs.dim_x != buf0.GetSize() / 256) {
        return false;
    }
    static const bool copy_log = std::getenv("BB_IMAGE_COPY_LOG") != nullptr;
    const u64 lookup_max_tail = std::max<u64>(4_MB, buf0.GetSize() / 3);
    auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize(),
                                                      !desc0.is_written, lookup_max_tail);
    auto image1_id = texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(),
                                                      !desc1.is_written, lookup_max_tail);
    if (image0_id && !image1_id && !desc0.is_written && desc1.is_written && !dry_run) {
        const auto& src_img = texture_cache.GetImage(image0_id);
        const u64 max_img_tail = std::max<u64>(4_MB, src_img.info.guest_size / 2);
        if (buf0.GetSize() >= src_img.info.guest_size &&
            buf0.GetSize() - src_img.info.guest_size <= max_img_tail) {
            VideoCore::TextureCache::ImageDesc dst_desc{};
            dst_desc.info = src_img.info;
            dst_desc.info.guest_address = buf1.base_address;
            dst_desc.info.meta_info = {};
            dst_desc.info.stencil_addr = 0;
            dst_desc.info.stencil_size = 0;
            dst_desc.type = src_img.info.props.is_depth
                                ? VideoCore::TextureCache::BindingType::DepthTarget
                                : VideoCore::TextureCache::BindingType::RenderTarget;
            image1_id = texture_cache.FindImage(dst_desc, true);
        }
    } else if (image1_id && !image0_id && !desc1.is_written && desc0.is_written && !dry_run) {
        const auto& src_img = texture_cache.GetImage(image1_id);
        const u64 max_img_tail = std::max<u64>(4_MB, src_img.info.guest_size / 2);
        if (buf1.GetSize() >= src_img.info.guest_size &&
            buf1.GetSize() - src_img.info.guest_size <= max_img_tail) {
            VideoCore::TextureCache::ImageDesc dst_desc{};
            dst_desc.info = src_img.info;
            dst_desc.info.guest_address = buf0.base_address;
            dst_desc.info.meta_info = {};
            dst_desc.info.stencil_addr = 0;
            dst_desc.info.stencil_size = 0;
            dst_desc.type = src_img.info.props.is_depth
                                ? VideoCore::TextureCache::BindingType::DepthTarget
                                : VideoCore::TextureCache::BindingType::RenderTarget;
            image0_id = texture_cache.FindImage(dst_desc, true);
        }
    }
    if (!image0_id || !image1_id) {
        if (copy_log && BbStats::gpu_frames.load(std::memory_order_relaxed) >= 30) {
            static std::atomic<u32> rej{0};
            if (rej.fetch_add(1) < 20) {
                std::printf("Image copy HLE miss: cs %016llx groups %u buf0 %#llx+%u (id %u:%s) "
                            "buf1 %#llx+%u (id %u:%s)\n",
                            (unsigned long long)info.pgm_hash, cs.dim_x,
                            (unsigned long long)buf0.base_address, u32(buf0.GetSize()),
                            image0_id.index,
                            texture_cache.DescribeImagesIn(buf0.base_address, buf0.GetSize()).c_str(),
                            (unsigned long long)buf1.base_address, u32(buf1.GetSize()),
                            image1_id.index,
                            texture_cache.DescribeImagesIn(buf1.base_address, buf1.GetSize()).c_str());
            }
        }
        return false;
    }
    auto& image0 = texture_cache.GetImage(image0_id);
    auto& image1 = texture_cache.GetImage(image1_id);
    // The buffers may run past the images (padding up to the next one: the light shafts' 960x540
    // targets are copied as 0x241800 bytes of 0x240000): those bytes are copied as a buffer.
    const u64 image_size = image0.info.guest_size;
    const u64 max_img_tail = std::max<u64>(4_MB, image_size / 2);
    if (image_size != image1.info.guest_size || image0.info.pitch != image1.info.pitch ||
        buf0.GetSize() < image_size || buf0.GetSize() - image_size > max_img_tail ||
        image0.info.num_bits != image1.info.num_bits) {
        if (copy_log && BbStats::gpu_frames.load(std::memory_order_relaxed) >= 30) {
            static std::atomic<u32> rej2{0};
            if (rej2.fetch_add(1) < 20) {
                std::printf("Image copy HLE size mismatch: buf %u, img0 %#llx %ux%u pitch %u "
                            "guest_size %llu bits %u fmt %u, img1 %#llx %ux%u pitch %u guest_size "
                            "%llu bits %u fmt %u\n",
                            u32(buf0.GetSize()), (unsigned long long)image0.info.guest_address,
                            image0.info.size.width, image0.info.size.height, image0.info.pitch,
                            (unsigned long long)image0.info.guest_size, image0.info.num_bits,
                            u32(image0.info.pixel_format),
                            (unsigned long long)image1.info.guest_address, image1.info.size.width,
                            image1.info.size.height, image1.info.pitch,
                            (unsigned long long)image1.info.guest_size, image1.info.num_bits,
                            u32(image1.info.pixel_format));
            }
        }
        return false;
    }
    auto& src = desc0.is_written ? image1 : image0;
    auto& dst = desc0.is_written ? image0 : image1;
    // BB_IMAGE_COPY_LOG=1 (diagnostics): the copies recognized.
    if (copy_log) {
        static std::atomic<u32> printed{0};
        if (printed.fetch_add(1) < 40) {
            std::printf("Image copy: %#llx (%ux%u fmt %u flags %#x) -> %#llx (%ux%u fmt %u flags "
                        "%#x), %u bytes\n",
                        (unsigned long long)src.info.guest_address, src.info.size.width,
                        src.info.size.height, u32(src.info.pixel_format), u32(src.flags),
                        (unsigned long long)dst.info.guest_address, dst.info.size.width,
                        dst.info.size.height, u32(dst.info.pixel_format), u32(dst.flags),
                        u32(buf0.GetSize()));
        }
    }
    if (dry_run) {
        return false;
    }
    const auto dst_id = desc0.is_written ? image0_id : image1_id;
    if (VideoCore::True(dst.flags & VideoCore::ImageFlagBits::DirectImmutable)) {
        texture_cache.DemoteDirectImmutable(dst_id);
    }
    runtime.CopyColorAndDepth(&src, &dst);
    dst.gpu_write_tick = scheduler.CurrentTick();
    // Before: the shader ran (5 times a frame over the light shafts), writing 2.3 MiB into the
    // game's memory each time (in place: those blocks are dynamic), the target then uploaded
    // from there again: 3.4 ms of GPU time a frame with BB_GUEST_UNCACHED on an RX 7800 XT.
    if (const u64 tail = buf0.GetSize() - image_size) {
        const VAddr src_tail = src.info.guest_address + image_size;
        const VAddr dst_tail = dst.info.guest_address + image_size;
        const auto tail_img_id = texture_cache.FindImageFromRange(dst_tail, tail, false);
        if (tail_img_id &&
            texture_cache.GetImage(tail_img_id).info.pixel_format != vk::Format::eUndefined) {
            const auto [src_buffer, src_offset] =
                buffer_cache.ObtainBuffer(src_tail, u32(tail), false);
            const auto [dst_buffer, dst_offset] =
                buffer_cache.ObtainBuffer(dst_tail, u32(tail), true);
            const vk::BufferCopy copy{src_offset, dst_offset, tail};
            runtime.CopyBuffer(src_buffer, dst_buffer, {&copy, 1});
            texture_cache.InvalidateMemoryFromGPU(dst_tail, tail);
        }
    }
    return true;
}

/// BB_IMAGE_CLEAR_LOG=1: why a clear shader (cs 8b355b5a) went to the real shader, once per reason
/// and group count: the number of the failing check in IsComputeImageClear.
static bool RejectClear(int reason, u32 dim_x, VAddr addr = 0, u64 size = 0,
                        const std::string& desc = {}) {
    static const bool log = [] {
        const char* env = std::getenv("BB_IMAGE_CLEAR_LOG");
        return env && env[0] == '1';
    }();
    if (log && BbStats::gpu_frames.load(std::memory_order_relaxed) >= 30) {
        static std::mutex mutex;
        static std::set<std::tuple<int, u32, VAddr>> told;
        std::scoped_lock lock{mutex};
        if (reason > 2 && told.size() < 40 && told.emplace(reason, dim_x, addr).second) {
            std::printf("Image clear HLE: rejected at check %d (%u groups, %#llx+%#llx%s)\n",
                        reason, dim_x, (unsigned long long)addr, (unsigned long long)size,
                        desc.c_str());
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return RejectClear(1, 0);
    }
    const auto& cs = CsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    // Phase 2.3: Yebis 2 8x8x1 2D UAV Image Clear (`cs 0x000000009a9cf8a9`):
    // Replaces 3.68M-8.29M OpImageWrite invocations + eGeneral DCC decompression with a single
    // hardware DCC Fast-Clear (`vkCmdClearColorImage`). Note: `upscaler->OnDispatch(cs.pgm_hash)`
    // in `DispatchRecord` runs before `BindResources`, so temporal upscaling trigger order is
    // 100% preserved.
    if (info.pgm_hash == 0x000000009a9cf8a9ULL) {
        if (cs.num_thread_x.full != 8 || cs.num_thread_y.full != 8 || cs.num_thread_z.full != 1 ||
            (cs.start_x | cs.start_y | cs.start_z) != 0 || cs.dim_z != 1 ||
            ShaderBufferCount(info) != 1 || info.images.size() != 1 || !info.samplers.empty()) {
            return false;
        }
        const auto& value_desc = info.buffers[0];
        const auto& image_res = info.images[0];
        if (value_desc.IsSpecial() || value_desc.is_formatted || value_desc.is_written ||
            !image_res.is_written || image_res.is_depth ||
            image_res.mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
            return false;
        }
        const PreparedStage* prepared = FindPreparedStage(info);
        const float* color_ptr = nullptr;
        for (u32 i = 0; i < num_ring_stages; ++i) {
            if (ring_stages[i].info == &info) {
                for (u32 k = 0; k < ring_stages[i].count; ++k) {
                    const auto& rb = ring_stages[i].bindings[k];
                    if (rb.index == 0 && rb.offset < RingNotCopied && rb.size >= 16) {
                        const u8* ring_data = rb.ring == 0
                                                  ? constant_ring->Data(rb.offset)
                                                  : worker_rings[rb.ring - 1]->Data(rb.offset);
                        color_ptr = reinterpret_cast<const float*>(ring_data);
                        break;
                    }
                }
                break;
            }
        }
        if (!color_ptr) {
            const Gcn::Buffer vsharp =
                prepared ? prepared->buffer_sharps[0] : value_desc.GetSharp(info);
            if (vsharp.base_address == 0 || vsharp.GetSize() < 16 ||
                memory->ClampRangeSize(vsharp.base_address, 16) < 16 ||
                buffer_cache.IsRegionGpuModified(vsharp.base_address, 16) ||
                buffer_cache.OverlapsInPassShadow(vsharp.base_address, 16) ||
                (HasPendingHleCopies() &&
                 PendingHleCopiesConflict(vsharp.base_address, 16, false))) {
                return false;
            }
            color_ptr = reinterpret_cast<const float*>(vsharp.base_address);
        }

        Gcn::Image tsharp = prepared ? prepared->image_sharps[0] : image_res.GetSharp(info);
        u8 srv_gen8 = 0u;
        const u16 srv_uid = VideoCore::DirectTextures::UnpackAndCleanSrv(tsharp, srv_gen8);
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == Gcn::DataFormat::FormatInvalid ||
            !memory->IsValidGpuMapping(tsharp.Address(), 0) || !IsKnownFormat(data_fmt, num_fmt) ||
            tsharp.GetNumberConversion() != Gcn::NumberConversion::None ||
            Gcn::GetNumberClass(num_fmt) != Gcn::NumberClass::Float) {
            return false;
        }
        ++bind_epoch;
        auto& desc_entry = ResolveImageDescEntry(
            tsharp, srv_uid, srv_gen8, image_res,
            prepared ? &prepared->image_hashes[0] : nullptr);
        VideoCore::TextureCache::ImageDesc desc = desc_entry.desc;
        if (image_res.mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
            desc.view_info.range.base.level += image_res.constant_mip_index;
            desc.view_info.range.extent.levels = 1;
        }
        if (desc.view_info.range.extent.levels != 1 || desc.view_info.range.extent.layers != 1) {
            return false;
        }
        const VideoCore::ImageId image_id = texture_cache.FindImage(desc);
        if (!image_id) {
            return false;
        }
        auto& image = texture_cache.GetImage(image_id);
        if (image.info.props.is_depth || image.info.size.depth != 1 || image.info.num_samples > 1) {
            return false;
        }
        const u32 mip_w = std::max(1u, image.info.size.width >> desc.view_info.range.base.level);
        const u32 mip_h = std::max(1u, image.info.size.height >> desc.view_info.range.base.level);
        if (u64(cs.dim_x) * 8ULL < mip_w || u64(cs.dim_y) * 8ULL < mip_h) {
            return false;
        }

        const std::array<float, 4> raw_rgba{color_ptr[0], color_ptr[1], color_ptr[2], color_ptr[3]};
        std::array<float, 4> swz_rgba = tsharp.DstSelect().Inverse().Apply(raw_rgba);
        const vk::Format view_fmt = desc.view_info.format;
        const vk::Format img_fmt = image.info.pixel_format;
        if (view_fmt != img_fmt) {
            const bool is_zero_or_one =
                (swz_rgba[0] == 0.0f || swz_rgba[0] == 1.0f) &&
                (swz_rgba[1] == 0.0f || swz_rgba[1] == 1.0f) &&
                (swz_rgba[2] == 0.0f || swz_rgba[2] == 1.0f) &&
                (swz_rgba[3] == 0.0f || swz_rgba[3] == 1.0f);
            if ((view_fmt == vk::Format::eR8G8B8A8Unorm && img_fmt == vk::Format::eB8G8R8A8Unorm) ||
                (view_fmt == vk::Format::eB8G8R8A8Unorm && img_fmt == vk::Format::eR8G8B8A8Unorm)) {
                std::swap(swz_rgba[0], swz_rgba[2]);
            } else if (is_zero_or_one && view_fmt == vk::Format::eR8G8B8A8Unorm &&
                       img_fmt == vk::Format::eR8G8B8A8Srgb) {
                // 0.0 and 1.0 encode identically in linear UNORM and sRGB.
            } else if (is_zero_or_one &&
                       ((view_fmt == vk::Format::eR8G8B8A8Unorm &&
                         img_fmt == vk::Format::eB8G8R8A8Srgb) ||
                        (view_fmt == vk::Format::eB8G8R8A8Unorm &&
                         img_fmt == vk::Format::eR8G8B8A8Srgb))) {
                std::swap(swz_rgba[0], swz_rgba[2]);
            } else {
                return false;
            }
        }

        const vk::ClearValue clear{.color = {.float32 = swz_rgba}};
        GpuProfiler::HudMarkPass(&scheduler, BbStats::GpuSub_Post_BloomGlare);
        if (auto* profiler = GpuProfiler::Get()) {
            profiler->Mark(info.pgm_hash ^ 0xD15Aull, [&] {
                return fmt::format("hle clear cs {:016x} ({}x{})", info.pgm_hash, mip_w, mip_h);
            });
        }
        if (VideoCore::True(image.flags & VideoCore::ImageFlagBits::DirectImmutable)) {
            texture_cache.DemoteDirectImmutable(image_id);
        }
        runtime.ClearImage(&image, desc.view_info.range, clear);
        image.gpu_write_tick = scheduler.CurrentTick();
        return true;
    }

    // A raw vec4 read and a formatted buffer over one layer of a color image written, 128
    // elements per workgroup: a clear of that image to the vec4.
    if (!IsTwoBufferShader(cs, info)) {
        return RejectClear(2, cs.dim_x);
    }
    const auto& value_desc = info.buffers[0];
    const auto& target_desc = info.buffers[1];
    if (value_desc.is_formatted || !target_desc.is_formatted || value_desc.is_written ||
        !target_desc.is_written) {
        return RejectClear(3, cs.dim_x);
    }
    const Gcn::Buffer value = value_desc.GetSharp(info);
    const Gcn::Buffer target = target_desc.GetSharp(info);
    const u32 bits = Gcn::NumBitsPerBlock(target.GetDataFmt());
    if (value.GetSize() != 16 || cs.dim_x * 128ULL * (bits / 8) != target.GetSize()) {
        return RejectClear(4, cs.dim_x, target.base_address, target.GetSize());
    }
    constexpr u64 MaxClearTail = 512_KB;
    thread_local boost::container::static_vector<VideoCore::ImageInfo, 8> clear_templates;
    s32 slice_layer = -1;
    auto image_id = texture_cache.FindImageFromRange(target.base_address, target.GetSize(),
                                                     false, MaxClearTail);
    if (image_id) {
        const auto& cand = texture_cache.GetImage(image_id);
        if (target.GetSize() < cand.info.guest_size) {
            image_id = {};
        }
    }
    if (!image_id) {
        std::tie(image_id, slice_layer) =
            texture_cache.FindImageSliceFromRange(target.base_address, target.GetSize());
    }
    if (!image_id && cs.dim_x >= 4096) {
        for (const auto& tmpl : clear_templates) {
            if (tmpl.guest_size == target.GetSize() && tmpl.num_bits == bits) {
                VideoCore::TextureCache::ImageDesc dst_desc{};
                dst_desc.info = tmpl;
                dst_desc.info.guest_address = target.base_address;
                dst_desc.info.meta_info = {};
                dst_desc.info.stencil_addr = 0;
                dst_desc.info.stencil_size = 0;
                dst_desc.type = VideoCore::TextureCache::BindingType::RenderTarget;
                image_id = texture_cache.FindImage(dst_desc, true);
                break;
            }
        }
    }
    if (!image_id) {
        return RejectClear(5, cs.dim_x, target.base_address, target.GetSize(),
                           texture_cache.DescribeImagesIn(target.base_address, target.GetSize()));
    }
    auto& image = texture_cache.GetImage(image_id);
    if (slice_layer < 0 &&
        (target.GetSize() < image.info.guest_size ||
         target.GetSize() - image.info.guest_size > MaxClearTail)) {
        return RejectClear(6, cs.dim_x, target.base_address, target.GetSize(),
                           texture_cache.DescribeImagesIn(target.base_address, target.GetSize()));
    }
    if (image.info.num_bits != bits || image.info.props.is_depth) {
        return RejectClear(6, cs.dim_x, target.base_address, target.GetSize(),
                           texture_cache.DescribeImagesIn(target.base_address, target.GetSize()));
    }
    if (slice_layer < 0 && cs.dim_x >= 4096 && image.info.resources.levels == 1 &&
        image.info.resources.layers == 1 && image.info.guest_size == target.GetSize()) {
        bool known = false;
        for (auto& tmpl : clear_templates) {
            if (tmpl.guest_size == image.info.guest_size && tmpl.num_bits == image.info.num_bits) {
                tmpl = image.info;
                known = true;
                break;
            }
        }
        if (!known && clear_templates.size() < clear_templates.capacity()) {
            clear_templates.push_back(image.info);
        }
    }
    const auto* color = reinterpret_cast<const float*>(value.base_address);
    const vk::ClearValue clear{
        .color = {.float32 = std::array<float, 4>{color[0], color[1], color[2], color[3]}},
    };
    const VideoCore::SubresourceRange range =
        slice_layer >= 0
            ? VideoCore::SubresourceRange{.base = {.level = 0, .layer = u32(slice_layer)},
                                          .extent = {.levels = 1, .layers = 1}}
            : VideoCore::SubresourceRange{.base = {.level = 0, .layer = 0},
                                          .extent = image.info.resources};
    if (VideoCore::True(image.flags & VideoCore::ImageFlagBits::DirectImmutable)) {
        texture_cache.DemoteDirectImmutable(image_id);
    }
    runtime.ClearImage(&image, range, clear);
    image.gpu_write_tick = scheduler.CurrentTick();
    return true;
}

} // namespace Vulkan
