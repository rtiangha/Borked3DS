// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

#include "common/literals.h"
#include "common/profiling.h"
#include "common/scope_exit.h"
#include "video_core/custom_textures/material.h"
#include "video_core/rasterizer_cache/texture_codec.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/renderer_vulkan/pica_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_render_manager.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_format_traits.hpp>

// Ignore the -Wclass-memaccess warning on memcpy for non-trivially default constructible objects.
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif

namespace Vulkan {

vk::ImageSubresourceRange MakeSubresourceRange(vk::ImageAspectFlags aspect, u32 level, u32 levels,
                                               u32 layer) {
    return vk::ImageSubresourceRange{
        .aspectMask = aspect,
        .baseMipLevel = level,
        .levelCount = levels,
        .baseArrayLayer = layer,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };
}

namespace {

using VideoCore::MapType;
using VideoCore::PixelFormat;
using VideoCore::SurfaceType;
using VideoCore::TextureType;
using namespace Common::Literals;

struct RecordParams {
    vk::ImageAspectFlags aspect;
    vk::Filter filter;
    vk::PipelineStageFlags pipeline_flags;
    vk::AccessFlags src_access;
    vk::AccessFlags dst_access;
    vk::Image src_image;
    vk::Image dst_image;
};

vk::Filter MakeFilter(VideoCore::PixelFormat pixel_format) {
    switch (pixel_format) {
    case VideoCore::PixelFormat::D16:
    case VideoCore::PixelFormat::D24:
    case VideoCore::PixelFormat::D24S8:
        return vk::Filter::eNearest;
    default:
        return vk::Filter::eLinear;
    }
}

[[nodiscard]] vk::ClearValue MakeClearValue(VideoCore::ClearValue clear) {
    static_assert(sizeof(VideoCore::ClearValue) == sizeof(vk::ClearValue));

    vk::ClearValue value{};
    std::memcpy(&value, &clear, sizeof(vk::ClearValue));
    return value;
}

[[nodiscard]] vk::ClearColorValue MakeClearColorValue(Common::Vec4f color) {
    return vk::ClearColorValue{
        .float32 = std::array{color[0], color[1], color[2], color[3]},
    };
}

[[nodiscard]] vk::ClearDepthStencilValue MakeClearDepthStencilValue(VideoCore::ClearValue clear) {
    return vk::ClearDepthStencilValue{
        .depth = clear.depth,
        .stencil = clear.stencil,
    };
}

u32 UnpackDepthStencil(const VideoCore::StagingData& data, vk::Format dest) {
    u32 depth_offset = 0;
    u32 stencil_offset = 4 * data.size / 5;
    const auto& mapped = data.mapped;

    switch (dest) {
    case vk::Format::eD24UnormS8Uint: {
        for (; stencil_offset < data.size; depth_offset += 4) {
            u8* ptr = mapped.data() + depth_offset;
            const u32 d24s8 = VideoCore::MakeInt<u32>(ptr);
            const u32 d24 = d24s8 >> 8;
            mapped[stencil_offset] = d24s8 & 0xFF;
            std::memcpy(ptr, &d24, 4);
            stencil_offset++;
        }
        break;
    }
    case vk::Format::eD32SfloatS8Uint: {
        for (; stencil_offset < data.size; depth_offset += 4) {
            u8* ptr = mapped.data() + depth_offset;
            const u32 d24s8 = VideoCore::MakeInt<u32>(ptr);
            const float d32 = (d24s8 >> 8) / 16777215.f;
            mapped[stencil_offset] = d24s8 & 0xFF;
            std::memcpy(ptr, &d32, 4);
            stencil_offset++;
        }
        break;
    }
    default:
        LOG_ERROR(Render_Vulkan, "Unimplemented convertion for depth format {}",
                  vk::to_string(dest));
        UNREACHABLE();
    }

    ASSERT(depth_offset == 4 * data.size / 5);
    return depth_offset;
}

boost::container::small_vector<vk::ImageMemoryBarrier, 3> MakeInitBarriers(
    vk::ImageAspectFlags aspect, vk::AccessFlags dst_access, std::span<const vk::Image> images) {
    boost::container::small_vector<vk::ImageMemoryBarrier, 3> barriers;
    for (const vk::Image& image : images) {
        barriers.push_back(vk::ImageMemoryBarrier{
            .srcAccessMask = vk::AccessFlagBits::eNone,
            .dstAccessMask = dst_access,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = image,
            .subresourceRange = MakeSubresourceRange(aspect, 0, VK_REMAINING_MIP_LEVELS, 0)});
    }
    return barriers;
}

Handle MakeHandle(const Instance* instance, u32 width, u32 height, u32 levels, TextureType type,
                  vk::Format format, vk::ImageUsageFlags usage, vk::ImageCreateFlags flags,
                  vk::ImageAspectFlags aspect, bool need_format_list,
                  std::string_view debug_name = {}) {
    const u32 layers = type == TextureType::CubeMap ? 6 : 1;

    // Apply texture size limits if high quality textures are disabled
    if (!TextureConfig{}.high_quality_textures) {
        width = std::min(width, TextureConfig{}.max_texture_size);
        height = std::min(height, TextureConfig{}.max_texture_size);
    }

    const std::array format_list = {
        vk::Format::eR8G8B8A8Unorm,
        vk::Format::eR32Uint,
    };
    const vk::ImageFormatListCreateInfo image_format_list = {
        .viewFormatCount = static_cast<u32>(format_list.size()),
        .pViewFormats = format_list.data(),
    };

    const vk::ImageCreateInfo image_info = {
        .pNext = need_format_list ? &image_format_list : nullptr,
        .flags = flags,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = levels,
        .arrayLayers = layers,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = usage,
    };

    // Get memory requirements using Vulkan-Hpp
    const vk::Device device = instance->GetDevice();
    const vk::UniqueImage temp_image = device.createImageUnique(image_info);
    const vk::MemoryRequirements mem_requirements =
        device.getImageMemoryRequirements(temp_image.get());

    const std::array<VmaAllocationCreateFlags, 2> alloc_flags = {
        VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT, // Fallback
    };

    const std::array<VmaMemoryUsage, 2> memory_usages = {
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_MEMORY_USAGE_AUTO,
    };

    VkImage unsafe_image{};
    VmaAllocation allocation{};
    VkResult result = VK_ERROR_OUT_OF_HOST_MEMORY;

    // Convert to VkImageCreateInfo first to avoid temporary address issues
    VkImageCreateInfo vk_image_info = static_cast<VkImageCreateInfo>(image_info);

    bool allocation_succeeded = false;

    // Try different combinations of flags and memory usage
    for (const auto image_flags : alloc_flags) {
        for (const auto mem_usage : memory_usages) {
            const VmaAllocationCreateInfo alloc_info = {
                .flags = image_flags,
                .usage = mem_usage,
                .requiredFlags = mem_requirements.memoryTypeBits,
                .preferredFlags = 0,
                .pool = VK_NULL_HANDLE,
                .pUserData = nullptr,
            };

            result = vmaCreateImage(instance->GetAllocator(),
                                    &vk_image_info, // Use pre-converted struct
                                    &alloc_info, &unsafe_image, &allocation, nullptr);

            if (result == VK_SUCCESS) {
                allocation_succeeded = true;
                break;
            }
        }
        if (allocation_succeeded) {
            break;
        }
    }

    if (!allocation_succeeded) {
        // Fallback attempt in separate scope
        const VmaAllocationCreateInfo fallback_info = {
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
            .requiredFlags = 0,
            .preferredFlags = mem_requirements.memoryTypeBits,
            .pool = VK_NULL_HANDLE,
            .pUserData = nullptr,
        };
        result = vmaCreateImage(instance->GetAllocator(),
                                &vk_image_info, // Use pre-converted struct
                                &fallback_info, &unsafe_image, &allocation, nullptr);
    }

    if (result != VK_SUCCESS) {
        LOG_CRITICAL(Render_Vulkan, "Failed to allocate image ({}): {}x{} mips:{} format:{}",
                     result, width, height, levels, vk::to_string(format));
        throw std::runtime_error("Failed to allocate Vulkan image");
    }

    const vk::Image image{unsafe_image};
    const vk::ImageViewCreateInfo view_info = {
        .image = image,
        .viewType =
            type == TextureType::CubeMap ? vk::ImageViewType::eCube : vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange =
            {
                .aspectMask = aspect,
                .baseMipLevel = 0,
                .levelCount = levels,
                .baseArrayLayer = 0,
                .layerCount = layers,
            },
    };

    vk::UniqueImageView image_view;
    try {
        image_view = instance->GetDevice().createImageViewUnique(view_info);
    } catch (const vk::SystemError& err) {
        vmaDestroyImage(instance->GetAllocator(), image, allocation);
        LOG_CRITICAL(Render_Vulkan, "Failed to create image view: {}", err.what());
        throw;
    }

    if (!debug_name.empty() && instance->HasDebuggingToolAttached()) {
        SetObjectName(instance->GetDevice(), image, debug_name);
        SetObjectName(instance->GetDevice(), image_view.get(), "{} View", debug_name);
    }

    return Handle{
        .alloc = allocation,
        .image = image,
        .image_view = std::move(image_view),
    };
}

constexpr u64 UPLOAD_BUFFER_SIZE = 512_MiB;
constexpr u64 DOWNLOAD_BUFFER_SIZE = 16_MiB;

} // Anonymous namespace

vk::UniqueFramebuffer MakeFramebuffer(vk::Device device, vk::RenderPass render_pass, u32 width,
                                      u32 height, std::span<const vk::ImageView> attachments) {
    const vk::FramebufferCreateInfo framebuffer_info = {
        .renderPass = render_pass,
        .attachmentCount = static_cast<u32>(attachments.size()),
        .pAttachments = attachments.data(),
        .width = width,
        .height = height,
        .layers = 1,
    };
    return device.createFramebufferUnique(framebuffer_info);
}

TextureRuntime::TextureRuntime(const Instance& instance, Scheduler& scheduler,
                               RenderManager& renderpass_cache, DescriptorUpdateQueue& update_queue,
                               u32 num_swapchain_images_)
    : instance{instance}, scheduler{scheduler}, renderpass_cache{renderpass_cache},
      blit_helper{instance, scheduler, renderpass_cache, update_queue},
      upload_buffer{instance, scheduler, vk::BufferUsageFlagBits::eTransferSrc, UPLOAD_BUFFER_SIZE,
                    BufferType::Upload},
      download_buffer{instance, scheduler,
                      vk::BufferUsageFlagBits::eTransferDst |
                          vk::BufferUsageFlagBits::eStorageBuffer,
                      DOWNLOAD_BUFFER_SIZE, BufferType::Download},
      num_swapchain_images{num_swapchain_images_} {}

TextureRuntime::~TextureRuntime() = default;

VideoCore::StagingData TextureRuntime::FindStaging(u32 size, bool upload) {
    StreamBuffer& buffer = upload ? upload_buffer : download_buffer;
    const auto [data, offset, invalidate] = buffer.Map(size, 16);
    return VideoCore::StagingData{
        .size = size,
        .offset = offset,
        .mapped = std::span{data, size},
    };
}

u32 TextureRuntime::RemoveThreshold() {
    return num_swapchain_images;
}

void TextureRuntime::Finish() {
    scheduler.Finish();
}

bool TextureRuntime::Reinterpret(Surface& source, Surface& dest,
                                 const VideoCore::TextureCopy& copy) {
    const PixelFormat src_format = source.pixel_format;
    const PixelFormat dst_format = dest.pixel_format;
    ASSERT_MSG(src_format != dst_format, "Reinterpretation with same format is invalid");

    // Validate format reinterpretation support
    const auto& src_traits = instance.GetTraits(src_format);
    const auto& dst_traits = instance.GetTraits(dst_format);
    if (!src_traits.transfer_support || !dst_traits.transfer_support) {
        LOG_ERROR(Render_Vulkan, "Reinterpret requires transfer support (src: {}, dst: {})",
                  src_traits.transfer_support, dst_traits.transfer_support);
        return false;
    }

    // Transition source/dest to TRANSFER_SRC/DST_OPTIMAL
    source.TransitionLayout(vk::ImageLayout::eTransferSrcOptimal, 0);
    dest.TransitionLayout(vk::ImageLayout::eTransferDstOptimal, 0);

    bool success = false;
    if (src_format == PixelFormat::D24S8 && dst_format == PixelFormat::RGBA8) {
        success = blit_helper.ConvertDS24S8ToRGBA8(source, dest, copy);
    } else if (!src_traits.needs_conversion && !dst_traits.needs_conversion) {
        // Direct copy if no conversion needed
        const std::array<VideoCore::TextureCopy, 1> copies = {copy};
        success = CopyTextures(source, dest, copies);
    } else {
        LOG_ERROR(Render_Vulkan, "Unimplemented reinterpretation: {} -> {}",
                  VideoCore::PixelFormatAsString(src_format),
                  VideoCore::PixelFormatAsString(dst_format));
    }

    // Transition back to GENERAL regardless of result
    source.TransitionLayout(vk::ImageLayout::eGeneral, 0);
    dest.TransitionLayout(vk::ImageLayout::eGeneral, 0);
    return success;
}

bool TextureRuntime::ClearTexture(Surface& surface, const VideoCore::TextureClear& clear) {
    // Validate clear operation support
    if (!surface.traits.attachment_support) {
        LOG_ERROR(Render_Vulkan, "Surface does not support attachment operations");
        return false;
    }

    // Transition to TRANSFER_DST_OPTIMAL for clear
    surface.TransitionLayout(vk::ImageLayout::eTransferDstOptimal, 0);

    const RecordParams params = {
        .aspect = surface.Aspect(),
        .pipeline_flags = surface.PipelineStageFlags(),
        .src_access = surface.AccessFlags(),
        .src_image = surface.Image(),
    };

    scheduler.Record([params, clear](vk::CommandBuffer cmdbuf) {
        const vk::ImageSubresourceRange range =
            MakeSubresourceRange(params.aspect, clear.texture_level, 1, 0);

        if (params.aspect & vk::ImageAspectFlagBits::eColor) {
            cmdbuf.clearColorImage(params.src_image, vk::ImageLayout::eTransferDstOptimal,
                                   MakeClearColorValue(clear.value.color), range);
        } else {
            cmdbuf.clearDepthStencilImage(params.src_image, vk::ImageLayout::eTransferDstOptimal,
                                          MakeClearDepthStencilValue(clear.value), range);
        }
    });

    // Transition back to GENERAL
    surface.TransitionLayout(vk::ImageLayout::eGeneral, 0);
    return true;
}

void TextureRuntime::ClearTextureWithRenderpass(Surface& surface,
                                                const VideoCore::TextureClear& clear) {
    const bool is_color = surface.type != VideoCore::SurfaceType::Depth &&
                          surface.type != VideoCore::SurfaceType::DepthStencil;

    const auto color_format = is_color ? surface.pixel_format : PixelFormat::Invalid;
    const auto depth_format = is_color ? PixelFormat::Invalid : surface.pixel_format;
    const auto render_pass = renderpass_cache.GetRenderpass(color_format, depth_format, true);

    const RecordParams params = {
        .aspect = surface.Aspect(),
        .pipeline_flags = surface.PipelineStageFlags(),
        .src_access = surface.AccessFlags(),
        .src_image = surface.Image(),
    };

    scheduler.Record([params, is_color, clear, render_pass,
                      framebuffer = surface.Framebuffer()](vk::CommandBuffer cmdbuf) {
        const vk::AccessFlags access_flags =
            is_color ? (vk::AccessFlagBits::eColorAttachmentRead |
                        vk::AccessFlagBits::eColorAttachmentWrite)
                     : (vk::AccessFlagBits::eDepthStencilAttachmentRead |
                        vk::AccessFlagBits::eDepthStencilAttachmentWrite);

        const vk::PipelineStageFlags stage_flags =
            is_color ? vk::PipelineStageFlagBits::eColorAttachmentOutput
                     : vk::PipelineStageFlagBits::eEarlyFragmentTests;

        const vk::ImageMemoryBarrier pre_barrier{
            .srcAccessMask = params.src_access,
            .dstAccessMask = access_flags,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = is_color ? vk::ImageLayout::eColorAttachmentOptimal
                                  : vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .image = params.src_image,
            .subresourceRange = MakeSubresourceRange(params.aspect, clear.texture_level, 1, 0)};

        const vk::ImageMemoryBarrier post_barrier{
            .srcAccessMask = access_flags,
            .dstAccessMask = params.src_access,
            .oldLayout = is_color ? vk::ImageLayout::eColorAttachmentOptimal
                                  : vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = params.src_image,
            .subresourceRange = MakeSubresourceRange(params.aspect, clear.texture_level, 1, 0)};

        cmdbuf.pipelineBarrier(params.pipeline_flags, stage_flags,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barrier);
        auto clear_value = MakeClearValue(clear.value);
        const vk::RenderPassBeginInfo renderpass_info = {
            .renderPass = render_pass,
            .framebuffer = framebuffer,
            .renderArea = {.offset = {static_cast<s32>(clear.texture_rect.left),
                                      static_cast<s32>(clear.texture_rect.bottom)},
                           .extent = {clear.texture_rect.GetWidth(),
                                      clear.texture_rect.GetHeight()}},
            .clearValueCount = 1,
            .pClearValues = &clear_value,
        };
        cmdbuf.beginRenderPass(renderpass_info, vk::SubpassContents::eInline);
        cmdbuf.endRenderPass();

        cmdbuf.pipelineBarrier(stage_flags, params.pipeline_flags,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);
    });
}

bool TextureRuntime::CopyTextures(Surface& source, Surface& dest,
                                  std::span<const VideoCore::TextureCopy> copies) {
    renderpass_cache.EndRendering();

    const RecordParams params = {
        .aspect = source.Aspect(),
        .filter = MakeFilter(source.pixel_format),
        .pipeline_flags = source.PipelineStageFlags() | dest.PipelineStageFlags(),
        .src_access = source.AccessFlags(),
        .dst_access = dest.AccessFlags(),
        .src_image = source.Image(),
        .dst_image = dest.Image(),
    };

    boost::container::small_vector<vk::ImageCopy, 2> vk_copies;
    std::ranges::transform(copies, std::back_inserter(vk_copies), [&](const auto& copy) {
        return vk::ImageCopy{
            .srcSubresource{
                .aspectMask = params.aspect,
                .mipLevel = copy.src_level,
                .baseArrayLayer = copy.src_layer,
                .layerCount = 1,
            },
            .srcOffset = {static_cast<s32>(copy.src_offset.x), static_cast<s32>(copy.src_offset.y),
                          0},
            .dstSubresource{
                .aspectMask = params.aspect,
                .mipLevel = copy.dst_level,
                .baseArrayLayer = copy.dst_layer,
                .layerCount = 1,
            },
            .dstOffset = {static_cast<s32>(copy.dst_offset.x), static_cast<s32>(copy.dst_offset.y),
                          0},
            .extent = {copy.extent.width, copy.extent.height, 1},
        };
    });

    scheduler.Record([params, copies = std::move(vk_copies)](vk::CommandBuffer cmdbuf) {
        const bool self_copy = params.src_image == params.dst_image;
        const vk::ImageLayout new_src_layout =
            self_copy ? vk::ImageLayout::eGeneral : vk::ImageLayout::eTransferSrcOptimal;
        const vk::ImageLayout new_dst_layout =
            self_copy ? vk::ImageLayout::eGeneral : vk::ImageLayout::eTransferDstOptimal;

        const std::array pre_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = params.src_access,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = new_src_layout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange =
                    MakeSubresourceRange(params.aspect, 0, VK_REMAINING_MIP_LEVELS, 0),
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = params.dst_access,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = new_dst_layout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.dst_image,
                .subresourceRange =
                    MakeSubresourceRange(params.aspect, 0, VK_REMAINING_MIP_LEVELS, 0),
            },
        };
        const std::array post_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eNone,
                .dstAccessMask = vk::AccessFlagBits::eNone,
                .oldLayout = new_src_layout,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange =
                    MakeSubresourceRange(params.aspect, 0, VK_REMAINING_MIP_LEVELS, 0),
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = params.dst_access,
                .oldLayout = new_dst_layout,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.dst_image,
                .subresourceRange =
                    MakeSubresourceRange(params.aspect, 0, VK_REMAINING_MIP_LEVELS, 0),
            },
        };

        cmdbuf.pipelineBarrier(params.pipeline_flags, vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barriers);

        cmdbuf.copyImage(params.src_image, new_src_layout, params.dst_image, new_dst_layout,
                         copies);

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, params.pipeline_flags,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barriers);
    });

    return true;
}

bool TextureRuntime::BlitTextures(Surface& source, Surface& dest,
                                  const VideoCore::TextureBlit& blit) {
    // Validate blit compatibility
    const auto& src_traits = instance.GetTraits(source.pixel_format);
    const auto& dst_traits = instance.GetTraits(dest.pixel_format);
    if (!src_traits.blit_support || !dst_traits.blit_support) {
        LOG_ERROR(Render_Vulkan, "Blit not supported for formats {} -> {}",
                  VideoCore::PixelFormatAsString(source.pixel_format),
                  VideoCore::PixelFormatAsString(dest.pixel_format));
        return false;
    }

    // Transition layouts for blit
    source.TransitionLayout(vk::ImageLayout::eTransferSrcOptimal, 0);
    dest.TransitionLayout(vk::ImageLayout::eTransferDstOptimal, 0);

    const RecordParams params = {
        .aspect = source.Aspect(),
        .filter = src_traits.blit_support ? vk::Filter::eLinear : vk::Filter::eNearest,
        .pipeline_flags = source.PipelineStageFlags() | dest.PipelineStageFlags(),
        .src_access = source.AccessFlags(),
        .dst_access = dest.AccessFlags(),
        .src_image = source.Image(),
        .dst_image = dest.Image(),
    };

    scheduler.Record([params, blit](vk::CommandBuffer cmdbuf) {
        const std::array source_offsets = {
            vk::Offset3D{static_cast<s32>(blit.src_rect.left),
                         static_cast<s32>(blit.src_rect.bottom), 0},
            vk::Offset3D{static_cast<s32>(blit.src_rect.right), static_cast<s32>(blit.src_rect.top),
                         1},
        };
        const std::array dest_offsets = {
            vk::Offset3D{static_cast<s32>(blit.dst_rect.left),
                         static_cast<s32>(blit.dst_rect.bottom), 0},
            vk::Offset3D{static_cast<s32>(blit.dst_rect.right), static_cast<s32>(blit.dst_rect.top),
                         1},
        };

        const vk::ImageBlit blit_region = {
            .srcSubresource = {params.aspect, blit.src_level, blit.src_layer, 1},
            .srcOffsets = source_offsets,
            .dstSubresource = {params.aspect, blit.dst_level, blit.dst_layer, 1},
            .dstOffsets = dest_offsets,
        };

        cmdbuf.blitImage(params.src_image, vk::ImageLayout::eTransferSrcOptimal, params.dst_image,
                         vk::ImageLayout::eTransferDstOptimal, blit_region, params.filter);
    });

    // Transition back to GENERAL for general use
    source.TransitionLayout(vk::ImageLayout::eGeneral, 0);
    dest.TransitionLayout(vk::ImageLayout::eGeneral, 0);
    return true;
}

void TextureRuntime::GenerateMipmaps(Surface& surface) {
    if (VideoCore::IsCustomFormatCompressed(surface.custom_format)) {
        LOG_ERROR(Render_Vulkan, "Generating mipmaps for compressed formats unsupported!");
        return;
    }

    surface.TransitionLayout(vk::ImageLayout::eTransferSrcOptimal, 0);

    auto [width, height] = surface.RealExtent();
    for (u32 i = 1; i < surface.levels; i++) {
        surface.TransitionLayout(vk::ImageLayout::eTransferDstOptimal, i);

        const VideoCore::TextureBlit blit = {
            .src_level = i - 1,
            .dst_level = i,
            .src_rect = {0, height, width, 0},
            .dst_rect = {0, height >> 1, width >> 1, 0},
        };
        BlitTextures(surface, surface, blit);

        surface.TransitionLayout(vk::ImageLayout::eGeneral, i - 1);
        width = std::max(width >> 1, 1u);
        height = std::max(height >> 1, 1u);
    }

    surface.TransitionLayout(vk::ImageLayout::eGeneral, surface.levels - 1);
}

bool TextureRuntime::NeedsConversion(VideoCore::PixelFormat format) const {
    const FormatTraits traits = instance.GetTraits(format);
    return traits.needs_conversion &&
           // DepthStencil formats are handled elsewhere due to de-interleaving.
           traits.aspect != (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil);
}

Surface::Surface(TextureRuntime& runtime_, const VideoCore::SurfaceParams& params)
    : SurfaceBase{params}, runtime{&runtime_}, instance{&runtime_.GetInstance()},
      scheduler{&runtime_.GetScheduler()}, traits{instance->GetTraits(pixel_format)} {

    if (pixel_format == VideoCore::PixelFormat::Invalid) {
        return;
    }

    const bool is_mutable = pixel_format == VideoCore::PixelFormat::RGBA8;
    const vk::Format format = traits.native;

    ASSERT_MSG(format != vk::Format::eUndefined && levels >= 1,
               "Invalid image allocation parameters");

    boost::container::static_vector<vk::Image, 3> raw_images;

    vk::ImageCreateFlags flags{};
    if (texture_type == VideoCore::TextureType::CubeMap) {
        flags |= vk::ImageCreateFlagBits::eCubeCompatible;
    }
    if (is_mutable) {
        flags |= vk::ImageCreateFlagBits::eMutableFormat;
    }

    const bool need_format_list = is_mutable && instance->IsImageFormatListSupported();
    const std::string debug_name = DebugName(false);

    try {
        handles[0] = MakeHandle(instance, width, height, levels, texture_type, format, traits.usage,
                                flags, traits.aspect, need_format_list, debug_name);
        raw_images.emplace_back(handles[0].image);

        if (res_scale != 1) {
            handles[1] =
                MakeHandle(instance, GetScaledWidth(), GetScaledHeight(), levels, texture_type,
                           format, traits.usage, flags, traits.aspect, false, DebugName(true));
            raw_images.emplace_back(handles[1].image);
        }
    } catch (const std::runtime_error& err) {
        LOG_CRITICAL(Render_Vulkan, "Failed to create surface: {}", err.what());
        throw;
    }

    runtime->renderpass_cache.EndRendering();

    const vk::AccessFlags dst_access =
        traits.aspect & vk::ImageAspectFlagBits::eColor
            ? vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eShaderRead
            : vk::AccessFlagBits::eDepthStencilAttachmentWrite |
                  vk::AccessFlagBits::eDepthStencilAttachmentRead;

    scheduler->Record([raw_images, aspect = traits.aspect, dst_access](vk::CommandBuffer cmdbuf) {
        const auto barriers = MakeInitBarriers(aspect, dst_access, raw_images);
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                               vk::PipelineStageFlagBits::eAllCommands,
                               vk::DependencyFlagBits::eByRegion, {}, {}, barriers);
    });
}

// Custom material surface constructor
Surface::Surface(TextureRuntime& runtime_, const VideoCore::SurfaceBase& surface,
                 const VideoCore::Material* mat)
    : SurfaceBase{surface}, runtime{&runtime_}, instance{&runtime_.GetInstance()},
      scheduler{&runtime_.GetScheduler()}, traits{instance->GetTraits(mat->format)} {

    if (!traits.transfer_support) {
        return;
    }

    const bool has_normal = mat && mat->Map(MapType::Normal);
    const vk::Format format = traits.native;

    boost::container::static_vector<vk::Image, 2> raw_images;
    const std::string debug_name = DebugName(false, true);

    try {
        vk::ImageUsageFlags usage = traits.usage;
        if (static_cast<VideoCore::PixelFormat>(mat->format) == VideoCore::PixelFormat::RGBA8) {
            usage |= vk::ImageUsageFlagBits::eStorage;
        }

        handles[0] = MakeHandle(instance, mat->width, mat->height, levels, texture_type, format,
                                usage, {}, traits.aspect, false, debug_name);
        raw_images.emplace_back(handles[0].image);

        if (res_scale != 1) {
            handles[1] =
                MakeHandle(instance, mat->width, mat->height, levels, texture_type,
                           vk::Format::eR8G8B8A8Unorm, usage, {}, traits.aspect, false, debug_name);
            raw_images.emplace_back(handles[1].image);
        }
        if (has_normal) {
            handles[2] = MakeHandle(instance, mat->width, mat->height, levels, texture_type, format,
                                    usage, {}, traits.aspect, false, debug_name);
            raw_images.emplace_back(handles[2].image);
        }
    } catch (const std::runtime_error& err) {
        LOG_CRITICAL(Render_Vulkan, "Failed to create material surface: {}", err.what());
        throw;
    }

    runtime->renderpass_cache.EndRendering();

    const vk::AccessFlags dst_access = vk::AccessFlagBits::eShaderRead |
                                       vk::AccessFlagBits::eTransferRead |
                                       vk::AccessFlagBits::eTransferWrite;

    scheduler->Record([raw_images, aspect = traits.aspect, dst_access](vk::CommandBuffer cmdbuf) {
        const auto barriers = MakeInitBarriers(aspect, dst_access, raw_images);
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                               vk::PipelineStageFlagBits::eFragmentShader,
                               vk::DependencyFlagBits::eByRegion, {}, {}, barriers);
    });

    custom_format = mat->format;
    material = mat;
}

Surface::~Surface() {
    if (!handles[0].image_view) {
        return;
    }
    scheduler->Finish();
    for (const auto& [alloc, image, image_view] : handles) {
        if (image) {
            vmaDestroyImage(instance->GetAllocator(), image, alloc);
        }
    }
    if (copy_handle.image_view) {
        vmaDestroyImage(instance->GetAllocator(), copy_handle.image, copy_handle.alloc);
    }
}

void Surface::SyncDualAspect() {
    // Validate dual-aspect requirement
    ASSERT_MSG((Aspect() & (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)) ==
                   (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil),
               "SyncDualAspect requires depth/stencil image");

    // Use TransitionLayout to ensure valid state transitions
    TransitionLayout(vk::ImageLayout::eDepthStencilReadOnlyOptimal, 0);
}

void Surface::Upload(const VideoCore::BufferTextureCopy& upload,
                     const VideoCore::StagingData& staging) {
    ASSERT_MSG(traits.transfer_support, "Surface does not support transfer operations!");
    runtime->renderpass_cache.EndRendering();

    const u32 index = 0; // Main surface handle
    TransitionLayout(vk::ImageLayout::eTransferDstOptimal, index);

    const bool is_depth_stencil =
        (traits.aspect & (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)) !=
        vk::ImageAspectFlags{};

    const RecordParams params = {
        .aspect = traits.aspect,
        .pipeline_flags = PipelineStageFlags(),
        .src_access = AccessFlags(),
        .src_image = Image(index),
    };

    scheduler->Record([buffer = runtime->upload_buffer.Handle(), params, upload, is_depth_stencil,
                       format = traits.native, staging](vk::CommandBuffer cmdbuf) {
        boost::container::static_vector<vk::BufferImageCopy, 2> copies;
        const auto rect = upload.texture_rect;

        copies.emplace_back(vk::BufferImageCopy{
            .bufferOffset = upload.buffer_offset,
            .imageSubresource =
                {
                    .aspectMask = params.aspect,
                    .mipLevel = upload.texture_level,
                    .layerCount = 1,
                },
            .imageOffset = {static_cast<s32>(rect.left), static_cast<s32>(rect.bottom), 0},
            .imageExtent = {rect.GetWidth(), rect.GetHeight(), 1},
        });

        if (is_depth_stencil) {
            auto& stencil_copy = copies.emplace_back(copies[0]);
            stencil_copy.bufferOffset += UnpackDepthStencil(staging, format);
            stencil_copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eStencil;
        }

        cmdbuf.copyBufferToImage(buffer, params.src_image, vk::ImageLayout::eTransferDstOptimal,
                                 copies);
    });

    TransitionLayout(vk::ImageLayout::eGeneral, index);
    runtime->upload_buffer.Commit(staging.size);

    if (res_scale != 1) {
        const VideoCore::TextureBlit blit = {
            .src_level = upload.texture_level,
            .dst_level = upload.texture_level,
            .src_rect = upload.texture_rect,
            .dst_rect = upload.texture_rect * res_scale,
        };
        BlitScale(blit, true);
    }
}

void Surface::UploadCustom(const VideoCore::Material* material, u32 level) {
    const u32 width = material->width;
    const u32 height = material->height;
    const Common::Rectangle rect{0U, height, width, 0U};

    const auto upload = [&](u32 index, VideoCore::CustomTexture* texture) {
        TransitionLayout(vk::ImageLayout::eTransferDstOptimal, index);

        const u32 custom_size = static_cast<u32>(texture->data.size());
        const auto [data, offset, invalidate] = runtime->upload_buffer.Map(custom_size, 0);
        std::memcpy(data, texture->data.data(), custom_size);
        runtime->upload_buffer.Commit(custom_size);

        scheduler->Record([buffer = runtime->upload_buffer.Handle(), level, image = Image(index),
                           aspect = traits.aspect, rect, offset](vk::CommandBuffer cmdbuf) {
            const vk::BufferImageCopy copy{
                .bufferOffset = offset,
                .imageSubresource =
                    {
                        .aspectMask = aspect,
                        .mipLevel = level,
                        .layerCount = 1,
                    },
                .imageOffset = {static_cast<s32>(rect.left), static_cast<s32>(rect.bottom), 0},
                .imageExtent = {rect.GetWidth(), rect.GetHeight(), 1},
            };
            cmdbuf.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, copy);
        });

        TransitionLayout(vk::ImageLayout::eGeneral, index);
    };

    upload(0, material->textures[0]);

    for (u32 i = 1; i < VideoCore::MAX_MAPS; i++) {
        if (auto texture = material->textures[i]) {
            upload(i + 1, texture);
        }
    }
}

void Surface::Download(const VideoCore::BufferTextureCopy& download,
                       const VideoCore::StagingData& staging) {
    SCOPE_EXIT({ runtime->download_buffer.Commit(staging.size); });
    runtime->renderpass_cache.EndRendering();

    const u32 index = res_scale != 1 ? 1 : 0;
    if (pixel_format == PixelFormat::D24S8) {
        runtime->blit_helper.DepthToBuffer(*this, runtime->download_buffer.Handle(), download);
        return;
    }

    if (res_scale != 1) {
        const VideoCore::TextureBlit blit = {
            .src_level = download.texture_level,
            .dst_level = download.texture_level,
            .src_rect = download.texture_rect * res_scale,
            .dst_rect = download.texture_rect,
        };
        BlitScale(blit, false);
    }

    TransitionLayout(vk::ImageLayout::eTransferSrcOptimal, index);

    scheduler->Record([buffer = runtime->download_buffer.Handle(), image = Image(index),
                       aspect = traits.aspect, download](vk::CommandBuffer cmdbuf) {
        const auto rect = download.texture_rect;
        const vk::BufferImageCopy copy{
            .bufferOffset = download.buffer_offset,
            .imageSubresource =
                {
                    .aspectMask = aspect,
                    .mipLevel = download.texture_level,
                    .layerCount = 1,
                },
            .imageOffset = {static_cast<s32>(rect.left), static_cast<s32>(rect.bottom), 0},
            .imageExtent = {rect.GetWidth(), rect.GetHeight(), 1},
        };
        cmdbuf.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, buffer, copy);
    });

    TransitionLayout(vk::ImageLayout::eGeneral, index);
}

void Surface::ScaleUp(u32 new_scale) {
    if (res_scale == new_scale || new_scale == 1) {
        return;
    }

    res_scale = new_scale;

    const bool is_mutable = pixel_format == VideoCore::PixelFormat::RGBA8;

    vk::ImageCreateFlags flags{};
    if (texture_type == VideoCore::TextureType::CubeMap) {
        flags |= vk::ImageCreateFlagBits::eCubeCompatible;
    }
    if (is_mutable) {
        flags |= vk::ImageCreateFlagBits::eMutableFormat;
    }

    handles[1] =
        MakeHandle(instance, GetScaledWidth(), GetScaledHeight(), levels, texture_type,
                   traits.native, traits.usage, flags, traits.aspect, false, DebugName(true));

    runtime->renderpass_cache.EndRendering();

    const vk::AccessFlags dst_access = vk::AccessFlagBits::eTransferWrite;
    scheduler->Record([raw_images = std::array{Image()}, aspect = traits.aspect,
                       dst_access](vk::CommandBuffer cmdbuf) {
        const auto barriers = MakeInitBarriers(aspect, dst_access, raw_images);
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                               vk::PipelineStageFlagBits::eTopOfPipe,
                               vk::DependencyFlagBits::eByRegion, {}, {}, barriers);
    });

    for (u32 level = 0; level < levels; level++) {
        const VideoCore::TextureBlit blit = {
            .src_level = level,
            .dst_level = level,
            .src_rect = GetRect(level),
            .dst_rect = GetScaledRect(level),
        };
        BlitScale(blit, true);
    }
}

u32 Surface::GetInternalBytesPerPixel() const {
    // Request 5 bytes for D24S8 as well because we can use the
    // extra space when deinterleaving the data during upload
    if (traits.native == vk::Format::eD24UnormS8Uint) {
        return 5;
    }

    return vk::blockSize(traits.native);
}

vk::AccessFlags Surface::AccessFlags() const noexcept {
    const bool is_color = static_cast<bool>(Aspect() & vk::ImageAspectFlagBits::eColor);
    const vk::AccessFlags attachment_flags =
        is_color
            ? vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite
            : vk::AccessFlagBits::eDepthStencilAttachmentRead |
                  vk::AccessFlagBits::eDepthStencilAttachmentWrite;

    return vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead |
           vk::AccessFlagBits::eTransferWrite |
           (is_framebuffer ? attachment_flags : vk::AccessFlagBits::eNone) |
           (is_storage ? vk::AccessFlagBits::eShaderWrite : vk::AccessFlagBits::eNone);
}

vk::PipelineStageFlags Surface::PipelineStageFlags() const noexcept {
    const bool is_color = static_cast<bool>(Aspect() & vk::ImageAspectFlagBits::eColor);
    const vk::PipelineStageFlags attachment_flags =
        is_color ? vk::PipelineStageFlagBits::eColorAttachmentOutput
                 : vk::PipelineStageFlagBits::eEarlyFragmentTests |
                       vk::PipelineStageFlagBits::eLateFragmentTests;

    return vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eFragmentShader |
           (is_framebuffer ? attachment_flags : vk::PipelineStageFlagBits::eNone) |
           (is_storage ? vk::PipelineStageFlagBits::eComputeShader
                       : vk::PipelineStageFlagBits::eNone);
}

vk::Image Surface::Image(u32 index) const noexcept {
    const vk::Image image = handles[index].image;
    if (!image) {
        return handles[0].image;
    }
    return image;
}

vk::ImageView Surface::CopyImageView() noexcept {
    vk::ImageLayout copy_layout = vk::ImageLayout::eGeneral;
    if (!copy_handle.image) {
        vk::ImageCreateFlags flags{};
        if (texture_type == VideoCore::TextureType::CubeMap) {
            flags |= vk::ImageCreateFlagBits::eCubeCompatible;
        }
        copy_handle =
            MakeHandle(instance, GetScaledWidth(), GetScaledHeight(), levels, texture_type,
                       traits.native, traits.usage, flags, traits.aspect, false);
        copy_layout = vk::ImageLayout::eUndefined;
    }

    runtime->renderpass_cache.EndRendering();

    const RecordParams params = {
        .aspect = Aspect(),
        .pipeline_flags = PipelineStageFlags(),
        .src_access = AccessFlags(),
        .src_image = Image(),
        .dst_image = copy_handle.image,
    };

    scheduler->Record([params, copy_layout, levels = this->levels, width = GetScaledWidth(),
                       height = GetScaledHeight()](vk::CommandBuffer cmdbuf) {
        std::array pre_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, levels, 0),
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eShaderRead,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .oldLayout = copy_layout,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.dst_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, levels, 0),
            },
        };
        std::array post_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferRead,
                .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.src_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, levels, 0),
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = params.dst_image,
                .subresourceRange = MakeSubresourceRange(params.aspect, 0, levels, 0),
            },
        };

        boost::container::small_vector<vk::ImageCopy, 3> image_copies;
        for (u32 level = 0; level < levels; level++) {
            image_copies.push_back(vk::ImageCopy{
                .srcSubresource{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = level,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .srcOffset = {0, 0, 0},
                .dstSubresource{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = level,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .dstOffset = {0, 0, 0},
                .extent = {width >> level, height >> level, 1},
            });
        }

        cmdbuf.pipelineBarrier(params.pipeline_flags, vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barriers);

        cmdbuf.copyImage(params.src_image, vk::ImageLayout::eTransferSrcOptimal, params.dst_image,
                         vk::ImageLayout::eTransferDstOptimal, image_copies);

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, params.pipeline_flags,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barriers);
    });

    return copy_handle.image_view.get();
}

vk::ImageView Surface::ImageView(u32 index) const noexcept {
    const auto& image_view = handles[index].image_view.get();
    if (!image_view) {
        return handles[0].image_view.get();
    }
    return image_view;
}

vk::ImageView Surface::FramebufferView() noexcept {
    is_framebuffer = true;
    return ImageView();
}

vk::ImageView Surface::DepthView() noexcept {
    if (depth_view) {
        return depth_view.get();
    }

    const vk::ImageViewCreateInfo view_info = {
        .image = Image(),
        .viewType = vk::ImageViewType::e2D,
        .format = instance->GetTraits(pixel_format).native,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eDepth,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        },
    };

    depth_view = instance->GetDevice().createImageViewUnique(view_info);
    return depth_view.get();
}

vk::ImageView Surface::StencilView() noexcept {
    if (stencil_view) {
        return stencil_view.get();
    }

    const vk::ImageViewCreateInfo view_info = {
        .image = Image(),
        .viewType = vk::ImageViewType::e2D,
        .format = instance->GetTraits(pixel_format).native,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eStencil,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        },
    };

    stencil_view = instance->GetDevice().createImageViewUnique(view_info);
    return stencil_view.get();
}

vk::ImageView Surface::StorageView() noexcept {
    if (storage_view) {
        return storage_view.get();
    }

    if (pixel_format != VideoCore::PixelFormat::RGBA8) {
        LOG_WARNING(Render_Vulkan,
                    "Attempted to retrieve storage view from unsupported surface with format {}",
                    VideoCore::PixelFormatAsString(pixel_format));
        return ImageView();
    }

    is_storage = true;

    const vk::ImageViewCreateInfo storage_view_info = {
        .image = Image(),
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR32Uint,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        },
    };
    storage_view = instance->GetDevice().createImageViewUnique(storage_view_info);
    return storage_view.get();
}

vk::Framebuffer Surface::Framebuffer() noexcept {
    const u32 index = res_scale == 1 ? 0u : 1u;
    if (framebuffers[index]) {
        return framebuffers[index].get();
    }

    const bool is_depth = type == SurfaceType::Depth || type == SurfaceType::DepthStencil;
    const auto color_format = is_depth ? PixelFormat::Invalid : pixel_format;
    const auto depth_format = is_depth ? pixel_format : PixelFormat::Invalid;
    const auto render_pass =
        runtime->renderpass_cache.GetRenderpass(color_format, depth_format, false);
    const auto attachments = std::array{ImageView()};
    framebuffers[index] = MakeFramebuffer(instance->GetDevice(), render_pass, GetScaledWidth(),
                                          GetScaledHeight(), attachments);
    return framebuffers[index].get();
}

void Surface::TransitionLayout(vk::ImageLayout new_layout, u32 index) {
    if (current_layouts[index] == new_layout) {
        return; // No-op if layout is unchanged
    }

    vk::AccessFlags src_access = GetAccessMask(current_layouts[index]);
    vk::AccessFlags dst_access = GetAccessMask(new_layout);

    vk::PipelineStageFlags src_stage = vk::PipelineStageFlagBits::eTopOfPipe;
    vk::PipelineStageFlags dst_stage = vk::PipelineStageFlagBits::eBottomOfPipe;

    // Define stage/access mappings for common layout transitions
    switch (current_layouts[index]) {
    case vk::ImageLayout::eUndefined:
        src_access = vk::AccessFlagBits::eNone;
        src_stage = vk::PipelineStageFlagBits::eTopOfPipe;
        break;
    case vk::ImageLayout::eGeneral:
        src_access = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
        src_stage = vk::PipelineStageFlagBits::eAllCommands;
        break;
    case vk::ImageLayout::eTransferDstOptimal:
        src_access = vk::AccessFlagBits::eTransferWrite;
        src_stage = vk::PipelineStageFlagBits::eTransfer;
        break;
    case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        src_access = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        src_stage = vk::PipelineStageFlagBits::eLateFragmentTests;
        break;
    case vk::ImageLayout::eDepthStencilReadOnlyOptimal:
        src_access = vk::AccessFlagBits::eDepthStencilAttachmentRead;
        src_stage = vk::PipelineStageFlagBits::eFragmentShader;
        break;
    default:
        LOG_ERROR(Render_Vulkan, "Unsupported old layout: {}",
                  vk::to_string(current_layouts[index]));
        UNREACHABLE();
    }

    switch (new_layout) {
    case vk::ImageLayout::eGeneral:
        dst_access = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
        dst_stage = vk::PipelineStageFlagBits::eAllCommands;
        break;
    case vk::ImageLayout::eTransferDstOptimal:
        dst_access = vk::AccessFlagBits::eTransferWrite;
        dst_stage = vk::PipelineStageFlagBits::eTransfer;
        break;
    case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        dst_access = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        dst_stage = vk::PipelineStageFlagBits::eEarlyFragmentTests |
                    vk::PipelineStageFlagBits::eLateFragmentTests;
        break;
    case vk::ImageLayout::eDepthStencilReadOnlyOptimal:
        dst_access = vk::AccessFlagBits::eDepthStencilAttachmentRead;
        dst_stage = vk::PipelineStageFlagBits::eFragmentShader;
        break;
    default:
        LOG_ERROR(Render_Vulkan, "Unsupported new layout: {}", vk::to_string(new_layout));
        UNREACHABLE();
    }

    scheduler->Record([=, this](vk::CommandBuffer cmdbuf) {
        const vk::ImageMemoryBarrier barrier{
            .srcAccessMask = src_access,
            .dstAccessMask = dst_access,
            .oldLayout = current_layouts[index],
            .newLayout = new_layout,
            .image = Image(index),
            .subresourceRange = MakeSubresourceRange(traits.aspect, 0, VK_REMAINING_MIP_LEVELS, 0)};
        cmdbuf.pipelineBarrier(src_stage, dst_stage, vk::DependencyFlagBits::eByRegion, {}, {},
                               barrier);
    });

    current_layouts[index] = new_layout;
}

void Surface::BlitScale(const VideoCore::TextureBlit& blit, bool up_scale) {
    const u32 src_index = up_scale ? 0 : 1;
    const u32 dst_index = up_scale ? 1 : 0;

    // Validate scaled surfaces exist
    ASSERT_MSG(handles[src_index].image && handles[dst_index].image,
               "BlitScale called on invalid scaled surface handles");

    // Transition source to TRANSFER_SRC_OPTIMAL and destination to TRANSFER_DST_OPTIMAL
    TransitionLayout(vk::ImageLayout::eTransferSrcOptimal, src_index);
    TransitionLayout(vk::ImageLayout::eTransferDstOptimal, dst_index);

    // For depth/stencil, validate blit support
    if (traits.aspect & (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)) {
        const auto& depth_traits = instance->GetTraits(pixel_format);
        ASSERT_MSG(depth_traits.blit_support, "Depth/stencil blit unsupported by hardware");
    }

    const vk::Image src_image = Image(src_index);
    const vk::Image dst_image = Image(dst_index);

    const vk::Filter filter = traits.blit_support ? vk::Filter::eLinear : vk::Filter::eNearest;
    scheduler->Record(
        [blit, src_image, dst_image, filter, aspect = traits.aspect](vk::CommandBuffer cmdbuf) {
            const std::array source_offsets = {
                vk::Offset3D{static_cast<s32>(blit.src_rect.left),
                             static_cast<s32>(blit.src_rect.bottom), 0},
                vk::Offset3D{static_cast<s32>(blit.src_rect.right),
                             static_cast<s32>(blit.src_rect.top), 1},
            };
            const std::array dest_offsets = {
                vk::Offset3D{static_cast<s32>(blit.dst_rect.left),
                             static_cast<s32>(blit.dst_rect.bottom), 0},
                vk::Offset3D{static_cast<s32>(blit.dst_rect.right),
                             static_cast<s32>(blit.dst_rect.top), 1},
            };

            const vk::ImageBlit blit_region = {
                .srcSubresource = {aspect, blit.src_level, blit.src_layer, 1},
                .srcOffsets = source_offsets,
                .dstSubresource = {aspect, blit.dst_level, blit.dst_layer, 1},
                .dstOffsets = dest_offsets,
            };

            cmdbuf.blitImage(src_image, vk::ImageLayout::eTransferSrcOptimal, dst_image,
                             vk::ImageLayout::eTransferDstOptimal, blit_region, filter);
        });

    // Transition back to GENERAL for general use
    TransitionLayout(vk::ImageLayout::eGeneral, src_index);
    TransitionLayout(vk::ImageLayout::eGeneral, dst_index);
}

Framebuffer::Framebuffer(TextureRuntime& runtime, const VideoCore::FramebufferParams& params,
                         Surface* color, Surface* depth)
    : VideoCore::FramebufferParams{params},
      res_scale{color ? color->res_scale : (depth ? depth->res_scale : 1u)} {
    auto& renderpass_cache = runtime.GetRenderpassCache();
    if (shadow_rendering && !color) {
        return;
    }

    width = height = std::numeric_limits<u32>::max();

    const auto prepare = [&](u32 index, Surface* surface) {
        const VideoCore::Extent extent = surface->RealExtent();
        width = std::min(width, extent.width);
        height = std::min(height, extent.height);
        if (!shadow_rendering) {
            formats[index] = surface->pixel_format;
        }
        images[index] = surface->Image();
        aspects[index] = surface->Aspect();
        image_views[index] = shadow_rendering ? surface->StorageView() : surface->FramebufferView();
    };

    boost::container::static_vector<vk::ImageView, 2> attachments;

    if (color) {
        prepare(0, color);
        attachments.emplace_back(image_views[0]);
    }

    if (depth) {
        prepare(1, depth);
        attachments.emplace_back(image_views[1]);
    }

    const vk::Device device = runtime.GetInstance().GetDevice();
    if (shadow_rendering) {
        render_pass =
            renderpass_cache.GetRenderpass(PixelFormat::Invalid, PixelFormat::Invalid, false);
        framebuffer = MakeFramebuffer(device, render_pass, color->GetScaledWidth(),
                                      color->GetScaledHeight(), {});
    } else {
        render_pass = renderpass_cache.GetRenderpass(formats[0], formats[1], false);
        framebuffer = MakeFramebuffer(device, render_pass, width, height, attachments);
    }
}

Framebuffer::~Framebuffer() = default;

Sampler::Sampler(TextureRuntime& runtime, const VideoCore::SamplerParams& params) {
    const Instance& instance = runtime.GetInstance();
    const vk::PhysicalDevice physical_device = instance.GetPhysicalDevice();
    const vk::PhysicalDeviceProperties properties = physical_device.getProperties();
    const vk::PhysicalDeviceFeatures features = physical_device.getFeatures();

    // Fixed anisotropy reference
    const u32 max_anisotropy = std::clamp(params.max_anisotropy, 1u,
                                          static_cast<u32>(properties.limits.maxSamplerAnisotropy));

    // Handle custom border color support
    vk::SamplerCustomBorderColorCreateInfoEXT border_color_info;
    const bool use_custom_border =
        instance.IsCustomBorderColorSupported() &&
        (params.wrap_s == Pica::TexturingRegs::TextureConfig::ClampToBorder ||
         params.wrap_t == Pica::TexturingRegs::TextureConfig::ClampToBorder);

    if (use_custom_border) {
        const auto color = PicaToVK::ColorRGBA8(params.border_color);
        border_color_info = {.customBorderColor = MakeClearColorValue(color),
                             .format = vk::Format::eUndefined};
    }

    // Configure wrap modes
    auto wrap_u = PicaToVK::WrapMode(params.wrap_s, runtime.GetInstance());
    auto wrap_v = PicaToVK::WrapMode(params.wrap_t, runtime.GetInstance());
    auto wrap_w = PicaToVK::WrapMode(params.wrap_r, runtime.GetInstance());

    // Configure compare op
    const bool compare_enabled = params.compare_enabled;
    const auto compare_op = PicaToVK::CompareFunc(params.compare_op);

    const vk::SamplerCreateInfo sampler_info = {
        .pNext = use_custom_border ? &border_color_info : nullptr,
        .magFilter = PicaToVK::TextureFilterMode(params.mag_filter),
        .minFilter = PicaToVK::TextureFilterMode(params.min_filter),
        .mipmapMode = PicaToVK::TextureMipFilterMode(params.mip_filter),
        .addressModeU = wrap_u,
        .addressModeV = wrap_v,
        .addressModeW = wrap_w,
        .mipLodBias = static_cast<float>(params.lod_bias),
        .anisotropyEnable = features.samplerAnisotropy && (max_anisotropy > 1),
        .maxAnisotropy = static_cast<float>(max_anisotropy),
        .compareEnable = compare_enabled,
        .compareOp = compare_op,
        .minLod = static_cast<float>(params.lod_min),
        .maxLod = static_cast<float>(params.lod_max),
        .borderColor =
            use_custom_border ? vk::BorderColor::eFloatCustomEXT : vk::BorderColor::eIntOpaqueBlack,
        .unnormalizedCoordinates = false};

    try {
        sampler = instance.GetDevice().createSamplerUnique(sampler_info);
    } catch (const vk::SystemError& err) {
        LOG_CRITICAL(Render_Vulkan, "Failed to create sampler: {}", err.what());
        throw;
    }

    // Debug labeling
    if (instance.HasDebuggingToolAttached()) {
        SetObjectName(instance.GetDevice(), *sampler, "Sampler");
    }
}

Sampler::~Sampler() = default;

DebugScope::DebugScope(TextureRuntime& runtime, Common::Vec4f color, std::string_view label)
    : scheduler{runtime.GetScheduler()},
      has_debug_tool{runtime.GetInstance().HasDebuggingToolAttached()} {
    if (!has_debug_tool) {
        return;
    }
    scheduler.Record([color, label = std::string(label)](vk::CommandBuffer cmdbuf) {
        const vk::DebugUtilsLabelEXT debug_label = {
            .pLabelName = label.data(),
            .color = std::array{color[0], color[1], color[2], color[3]},
        };
        cmdbuf.beginDebugUtilsLabelEXT(debug_label);
    });
}

DebugScope::~DebugScope() {
    if (!has_debug_tool) {
        return;
    }
    scheduler.Record([](vk::CommandBuffer cmdbuf) { cmdbuf.endDebugUtilsLabelEXT(); });
}

} // namespace Vulkan
