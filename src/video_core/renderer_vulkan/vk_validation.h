// Copyright 2025 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include "common/logging/log.h"
#include "video_core/pica/regs_external.h"
#include "video_core/pica/regs_framebuffer.h"
#include "video_core/pica_types.h"
#include "video_core/rasterizer_cache/rasterizer_cache_base.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_types.h"
#include "video_core/texture/texture_decode.h"

namespace Vulkan {

struct ScreenInfo;
struct TextureInfo;

class ValidationHelper {
public:
    static bool Validate3DSScreenDimensions(u32 width, u32 height, bool is_top_screen) {
        const u32 expected_width = is_top_screen ? TOP_SCREEN_WIDTH : BOTTOM_SCREEN_WIDTH;
        const u32 expected_height = is_top_screen ? TOP_SCREEN_HEIGHT : BOTTOM_SCREEN_HEIGHT;

        if (width != expected_width || height != expected_height) {
            LOG_CRITICAL(Render_Vulkan, "Invalid {} screen dimensions: {}x{}, expected: {}x{}",
                         is_top_screen ? "top" : "bottom", width, height, expected_width,
                         expected_height);
            return false;
        }
        return true;
    }

    static bool ValidateImageLayout(vk::Image image, vk::ImageLayout current_layout,
                                    vk::ImageLayout new_layout) {
        if (!image) {
            LOG_CRITICAL(Render_Vulkan, "Null image handle in layout transition");
            return false;
        }

        // Validate invalid layout transitions
        if (current_layout == vk::ImageLayout::eUndefined &&
            new_layout == vk::ImageLayout::ePreinitialized) {
            LOG_CRITICAL(Render_Vulkan, "Invalid layout transition: Undefined -> Preinitialized");
            return false;
        }

        if (current_layout == vk::ImageLayout::ePreinitialized &&
            new_layout == vk::ImageLayout::eUndefined) {
            LOG_CRITICAL(Render_Vulkan, "Invalid layout transition: Preinitialized -> Undefined");
            return false;
        }

        return true;
    }

    static bool ValidateSchedulerWait(u64 tick, u64 current_tick) {
        if (tick == 0) {
            LOG_CRITICAL(Render_Vulkan, "Invalid tick value: 0");
            return false;
        }

        // This matches your scheduler's check
        if (tick >= current_tick) {
            LOG_WARNING(Render_Vulkan, "Waiting for future tick {} >= {}", tick, current_tick);
            return true; // Not a fatal error since Scheduler::Wait handles this
        }

        return true;
    }

    static bool ValidateSemaphoreSubmission(vk::CommandBuffer cmdbuf, vk::Semaphore wait_semaphore,
                                            vk::Semaphore signal_semaphore, u64 signal_value) {
        if (!cmdbuf) {
            LOG_CRITICAL(Render_Vulkan, "Null command buffer in semaphore submission");
            return false;
        }

        if (signal_value == 0) {
            LOG_CRITICAL(Render_Vulkan, "Invalid signal value: 0");
            return false;
        }

        // Signal semaphore is required for timeline operation
        if (!signal_semaphore && signal_value != 0) {
            LOG_CRITICAL(Render_Vulkan, "Timeline signal requested but no semaphore provided");
            return false;
        }

        return true;
    }

    static bool ValidateBufferAllocation(VmaAllocator allocator,
                                         const vk::BufferCreateInfo& buffer_info,
                                         const VmaAllocationCreateInfo& alloc_info,
                                         VkBuffer& buffer, VmaAllocation& allocation) {
        // Validate buffer size
        if (buffer_info.size == 0) {
            LOG_CRITICAL(Render_Vulkan, "Attempted to create buffer with size 0");
            return false;
        }

        // Create buffer
        VkBufferCreateInfo unsafe_buffer_info = static_cast<VkBufferCreateInfo>(buffer_info);
        VkResult result = vmaCreateBuffer(allocator, &unsafe_buffer_info, &alloc_info, &buffer,
                                          &allocation, nullptr);

        if (result != VK_SUCCESS) {
            LOG_CRITICAL(Render_Vulkan, "Failed to create buffer: {}", result);
            return false;
        }

        // Validate allocation
        VmaAllocationInfo allocation_info;
        vmaGetAllocationInfo(allocator, allocation, &allocation_info);
        if (allocation_info.size < buffer_info.size) {
            LOG_CRITICAL(Render_Vulkan, "Allocated buffer size {} is smaller than requested {}",
                         allocation_info.size, buffer_info.size);
            vmaDestroyBuffer(allocator, buffer, allocation);
            return false;
        }

        return true;
    }

    static bool ValidateImageProperties(const vk::ImageCreateInfo& info,
                                        const vk::PhysicalDeviceLimits& limits) {
        if (info.extent.width > limits.maxImageDimension2D ||
            info.extent.height > limits.maxImageDimension2D) {
            LOG_CRITICAL(Render_Vulkan, "Image dimensions exceed device limits: {}x{}",
                         info.extent.width, info.extent.height);
            return false;
        }
        return true;
    }

    static bool ValidateFrameDimensions(u32 width, u32 height,
                                        const vk::PhysicalDeviceLimits& limits) {
        if (width == 0 || height == 0) {
            LOG_CRITICAL(Render_Vulkan, "Frame dimensions cannot be zero");
            return false;
        }

        if (width > limits.maxFramebufferWidth || height > limits.maxFramebufferHeight) {
            LOG_CRITICAL(Render_Vulkan, "Frame dimensions {}x{} exceed device limits {}x{}", width,
                         height, limits.maxFramebufferWidth, limits.maxFramebufferHeight);
            return false;
        }

        return true;
    }

    static bool ValidateRenderPassBegin(const vk::RenderPassBeginInfo& begin_info) {
        if (!begin_info.renderPass) {
            LOG_CRITICAL(Render_Vulkan, "Render pass is null in begin info");
            return false;
        }

        if (!begin_info.framebuffer) {
            LOG_CRITICAL(Render_Vulkan, "Framebuffer is null in begin info");
            return false;
        }

        if (begin_info.renderArea.extent.width == 0 || begin_info.renderArea.extent.height == 0) {
            LOG_CRITICAL(Render_Vulkan, "Invalid render area dimensions: {}x{}",
                         begin_info.renderArea.extent.width, begin_info.renderArea.extent.height);
            return false;
        }

        return true;
    }

    static bool ValidateDrawParameters(Frame* frame, const Layout::FramebufferLayout& layout,
                                       const std::array<ScreenInfo, 3>& screen_infos) {
        if (!frame) {
            LOG_CRITICAL(Render_Vulkan, "Null frame passed to draw operation");
            return false;
        }

        if (layout.width == 0 || layout.height == 0) {
            LOG_CRITICAL(Render_Vulkan, "Invalid layout dimensions: {}x{}", layout.width,
                         layout.height);
            return false;
        }

        for (size_t i = 0; i < screen_infos.size(); i++) {
            if (!ValidateScreenInfo(screen_infos[i])) {
                LOG_CRITICAL(Render_Vulkan, "Invalid screen info at index {}", i);
                return false;
            }
        }

        return true;
    }

    static bool ValidateViewport(const vk::Viewport& viewport,
                                 const vk::Extent2D& framebuffer_extent) {
        if (viewport.width <= 0.0f || viewport.height <= 0.0f) {
            LOG_CRITICAL(Render_Vulkan, "Invalid viewport dimensions: {}x{}", viewport.width,
                         viewport.height);
            return false;
        }

        if (viewport.width > framebuffer_extent.width ||
            viewport.height > framebuffer_extent.height) {
            LOG_CRITICAL(Render_Vulkan, "Viewport {}x{} exceeds framebuffer dimensions {}x{}",
                         viewport.width, viewport.height, framebuffer_extent.width,
                         framebuffer_extent.height);
            return false;
        }

        return true;
    }

    static bool ValidateVertexBufferParameters(const ScreenRectVertex* vertices,
                                               size_t vertex_count, u64 buffer_size) {
        if (!vertices) {
            LOG_CRITICAL(Render_Vulkan, "Null vertex buffer");
            return false;
        }

        if (vertex_count == 0) {
            LOG_CRITICAL(Render_Vulkan, "Zero vertices provided");
            return false;
        }

        const u64 required_size = sizeof(ScreenRectVertex) * vertex_count;
        if (required_size > buffer_size) {
            LOG_CRITICAL(Render_Vulkan, "Vertex data size {} exceeds buffer size {}", required_size,
                         buffer_size);
            return false;
        }

        return true;
    }

    static bool ValidateScissor(const vk::Rect2D& scissor, const vk::Extent2D& framebuffer_extent) {
        if (scissor.extent.width == 0 || scissor.extent.height == 0) {
            LOG_CRITICAL(Render_Vulkan, "Invalid scissor dimensions: {}x{}", scissor.extent.width,
                         scissor.extent.height);
            return false;
        }

        if (scissor.offset.x + scissor.extent.width > framebuffer_extent.width ||
            scissor.offset.y + scissor.extent.height > framebuffer_extent.height) {
            LOG_CRITICAL(Render_Vulkan, "Scissor region exceeds framebuffer dimensions");
            return false;
        }

        return true;
    }

    static bool ValidatePushConstantRange(const vk::PushConstantRange& range,
                                          const vk::PhysicalDeviceLimits& limits) {
        if (range.offset + range.size > limits.maxPushConstantsSize) {
            LOG_CRITICAL(Render_Vulkan, "Push constant range {}+{} exceeds device limit {}",
                         range.offset, range.size, limits.maxPushConstantsSize);
            return false;
        }

        if (range.size == 0) {
            LOG_CRITICAL(Render_Vulkan, "Push constant range size cannot be 0");
            return false;
        }

        if (range.offset % 4 != 0 || range.size % 4 != 0) {
            LOG_CRITICAL(Render_Vulkan, "Push constant range offset and size must be aligned to 4");
            return false;
        }

        return true;
    }

    static bool ValidateRenderPassCompatibility(const vk::RenderPass& renderpass,
                                                const vk::Framebuffer& framebuffer) {
        if (!renderpass) {
            LOG_CRITICAL(Render_Vulkan, "Render pass is null");
            return false;
        }

        if (!framebuffer) {
            LOG_CRITICAL(Render_Vulkan, "Framebuffer is null");
            return false;
        }

        return true;
    }

    static bool ValidateFramebufferConfig(const Pica::FramebufferConfig& config,
                                          PAddr framebuffer_addr, u32 pixel_stride) {
        if (framebuffer_addr == 0) {
            LOG_CRITICAL(Render_Vulkan, "Null framebuffer address");
            return false;
        }

        if (config.width.Value() == 0 || config.height.Value() == 0) {
            LOG_CRITICAL(Render_Vulkan, "Invalid framebuffer dimensions: {}x{}",
                         config.width.Value(), config.height.Value());
            return false;
        }

        if (pixel_stride < config.width.Value()) {
            LOG_CRITICAL(Render_Vulkan, "Stride {} is less than width {}", pixel_stride,
                         config.width.Value());
            return false;
        }

        return true;
    }

    static bool ValidateScreenInfo(const ScreenInfo& info) {
        if (!info.texture.image || !info.texture.image_view) {
            LOG_CRITICAL(Render_Vulkan, "Invalid screen texture or image view");
            return false;
        }

        // Change the texture coordinate validation to use the proper accessors
        const auto width = info.texcoords.GetWidth();
        const auto height = info.texcoords.GetHeight();
        if (width <= 0 || height <= 0) {
            LOG_CRITICAL(Render_Vulkan, "Invalid texture coordinates");
            return false;
        }

        if (!info.image_view) {
            LOG_CRITICAL(Render_Vulkan, "Invalid image view");
            return false;
        }

        return true;
    }

    static bool ValidateTextureInfo(const Vulkan::TextureInfo& info) {
        if (info.width == 0 || info.height == 0) {
            LOG_CRITICAL(Render_Vulkan, "Invalid texture dimensions: {}x{}", info.width,
                         info.height);
            return false;
        }

        if (!info.image) {
            LOG_CRITICAL(Render_Vulkan, "Texture image handle is null");
            return false;
        }

        if (!info.image_view) {
            LOG_CRITICAL(Render_Vulkan, "Texture image view handle is null");
            return false;
        }

        return true;
    }

    static bool ValidateShaderModules(const vk::Device& device,
                                      const std::vector<vk::ShaderModule>& shaders) {
        for (const auto& shader : shaders) {
            // Validate shader module exists
            if (!shader) {
                LOG_CRITICAL(Render_Vulkan, "Invalid shader module");
                return false;
            }

            // Additional shader validation properties could be checked here
            // if using VK_EXT_validation_cache or VK_EXT_shader_object
        }
        return true;
    }

    static bool ValidateSamplerProperties(const vk::SamplerCreateInfo& info,
                                          const vk::PhysicalDeviceLimits& limits) {
        if (info.maxAnisotropy > limits.maxSamplerAnisotropy) {
            LOG_CRITICAL(Render_Vulkan, "Anisotropy level {} exceeds device maximum {}",
                         info.maxAnisotropy, limits.maxSamplerAnisotropy);
            return false;
        }

        if (info.mipLodBias > limits.maxSamplerLodBias) {
            LOG_CRITICAL(Render_Vulkan, "LOD bias {} exceeds device maximum {}", info.mipLodBias,
                         limits.maxSamplerLodBias);
            return false;
        }
        return true;
    }

    static bool ValidateFormatSupport(const vk::PhysicalDevice& physical_device, vk::Format format,
                                      vk::FormatFeatureFlags required_features) {
        vk::FormatProperties props = physical_device.getFormatProperties(format);
        if ((props.optimalTilingFeatures & required_features) != required_features) {
            LOG_CRITICAL(Render_Vulkan, "Format {} doesn't support required features",
                         vk::to_string(format));
            return false;
        }
        return true;
    }

    static bool ValidateQueueFamilyProperties(const vk::QueueFamilyProperties& props,
                                              vk::QueueFlags required_flags) {
        if ((props.queueFlags & required_flags) != required_flags) {
            LOG_CRITICAL(Render_Vulkan, "Queue family doesn't support required operations");
            return false;
        }
        return true;
    }

    static bool ValidateImageCreateInfo(const vk::ImageCreateInfo& create_info) {
        if (create_info.extent.width == 0 || create_info.extent.height == 0 ||
            create_info.extent.depth == 0) {
            LOG_CRITICAL(Render_Vulkan, "Invalid image dimensions: {}x{}x{}",
                         create_info.extent.width, create_info.extent.height,
                         create_info.extent.depth);
            return false;
        }

        if (create_info.mipLevels == 0) {
            LOG_CRITICAL(Render_Vulkan, "Invalid mip level count: 0");
            return false;
        }

        if (create_info.arrayLayers == 0) {
            LOG_CRITICAL(Render_Vulkan, "Invalid array layer count: 0");
            return false;
        }

        return true;
    }

    static bool ValidateFence(const vk::Fence& fence, const char* name) {
        if (!fence) {
            LOG_CRITICAL(Render_Vulkan, "Null fence handle: {}", name);
            return false;
        }
        return true;
    }

    static bool ValidateBuffer(const vk::Buffer& buffer, VmaAllocation allocation,
                               VkDeviceSize size, const char* name) {
        if (!buffer) {
            LOG_CRITICAL(Render_Vulkan, "Null buffer handle: {}", name);
            return false;
        }

        if (!allocation) {
            LOG_CRITICAL(Render_Vulkan, "Null allocation handle: {}", name);
            return false;
        }

        if (size == 0) {
            LOG_CRITICAL(Render_Vulkan, "Zero-sized buffer: {}", name);
            return false;
        }

        return true;
    }

    static bool ValidateBufferProperties(const vk::PhysicalDeviceProperties& props,
                                         vk::DeviceSize size, vk::BufferUsageFlags usage) {
        if (size > props.limits.maxStorageBufferRange) {
            LOG_CRITICAL(Render_Vulkan, "Buffer size {} exceeds device limit {}", size,
                         props.limits.maxStorageBufferRange);
            return false;
        }

        // Add any other buffer property validations needed

        return true;
    }

    static bool ValidateSemaphore(const vk::Semaphore& semaphore, const char* name) {
        if (!semaphore) {
            LOG_CRITICAL(Render_Vulkan, "Null semaphore handle: {}", name);
            return false;
        }
        return true;
    }

    static bool ValidateMemoryType(const vk::PhysicalDeviceMemoryProperties& props, u32 type_index,
                                   vk::MemoryPropertyFlags required_props) {
        if (type_index >= props.memoryTypeCount) {
            LOG_CRITICAL(Render_Vulkan, "Invalid memory type index");
            return false;
        }

        if ((props.memoryTypes[type_index].propertyFlags & required_props) != required_props) {
            LOG_CRITICAL(Render_Vulkan, "Memory type doesn't support required properties");
            return false;
        }
        return true;
    }

    static bool ValidateSurfaceCapabilities(const vk::SurfaceCapabilitiesKHR& caps,
                                            const vk::Extent2D& desired_extent) {
        if (desired_extent.width < caps.minImageExtent.width ||
            desired_extent.width > caps.maxImageExtent.width ||
            desired_extent.height < caps.minImageExtent.height ||
            desired_extent.height > caps.maxImageExtent.height) {
            LOG_CRITICAL(Render_Vulkan, "Surface extent {}x{} out of bounds", desired_extent.width,
                         desired_extent.height);
            return false;
        }
        return true;
    }

    static bool ValidateCommandBuffer(const vk::CommandBuffer& cmd_buffer) {
        if (!cmd_buffer) {
            LOG_CRITICAL(Render_Vulkan, "Command buffer is null");
            return false;
        }
        return true;
    }

    static bool ValidatePipelineState(const vk::Pipeline& pipeline,
                                      const vk::PipelineLayout& layout) {
        if (!pipeline) {
            LOG_CRITICAL(Render_Vulkan, "Pipeline is null");
            return false;
        }

        if (!layout) {
            LOG_CRITICAL(Render_Vulkan, "Pipeline layout is null");
            return false;
        }

        return true;
    }

    static bool ValidateDescriptorSet(const vk::DescriptorSet& set) {
        if (!set) {
            LOG_CRITICAL(Render_Vulkan, "Descriptor set is null");
            return false;
        }
        return true;
    }

    static bool ValidateSwapchain(const vk::SwapchainKHR& swapchain,
                                  const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (!swapchain) {
            LOG_CRITICAL(Render_Vulkan, "Null swapchain handle");
            return false;
        }

        const auto min_image_count = capabilities.minImageCount;
        const auto max_image_count = capabilities.maxImageCount;

        if (min_image_count < 2) {
            LOG_CRITICAL(Render_Vulkan, "Swapchain minimum image count too low: {}",
                         min_image_count);
            return false;
        }

        if (max_image_count > 0 && min_image_count > max_image_count) {
            LOG_CRITICAL(Render_Vulkan, "Invalid swapchain image count range: {} - {}",
                         min_image_count, max_image_count);
            return false;
        }

        return true;
    }

    static bool ValidateSwapchainFormat(const Instance& instance, vk::SurfaceKHR surface,
                                        const vk::SurfaceFormatKHR& desired_format) {
        const auto formats = instance.GetPhysicalDevice().getSurfaceFormatsKHR(surface);

        bool format_supported = false;
        for (const auto& format : formats) {
            if (format.format == desired_format.format &&
                format.colorSpace == desired_format.colorSpace) {
                format_supported = true;
                break;
            }
        }

        if (!format_supported) {
            LOG_CRITICAL(Render_Vulkan, "Surface format not supported");
            return false;
        }

        return true;
    }

    static bool ValidatePresentMode(const Instance& instance, vk::SurfaceKHR surface,
                                    vk::PresentModeKHR desired_mode) {
        const auto present_modes = instance.GetPhysicalDevice().getSurfacePresentModesKHR(surface);

        if (std::find(present_modes.begin(), present_modes.end(), desired_mode) ==
            present_modes.end()) {
            LOG_CRITICAL(Render_Vulkan, "Present mode not supported");
            return false;
        }

        return true;
    }

    static bool HasCompatibleMemoryType(
        u32 type_bits, vk::MemoryPropertyFlags required_properties,
        const vk::PhysicalDeviceMemoryProperties& memory_properties) {
        for (u32 i = 0; i < memory_properties.memoryTypeCount; i++) {
            if ((type_bits & (1 << i)) && (memory_properties.memoryTypes[i].propertyFlags &
                                           required_properties) == required_properties) {
                return true;
            }
        }

        LOG_CRITICAL(Render_Vulkan, "No compatible memory type found for properties: {}",
                     static_cast<u32>(required_properties));
        return false;
    }

    static bool ValidateSwapchainProperties(const Instance& instance, vk::SurfaceKHR surface,
                                            u32 width, u32 height) {
        const auto physical_device = instance.GetPhysicalDevice();

        // First validate surface support
        VkBool32 supported = false;
        auto result = physical_device.getSurfaceSupportKHR(instance.GetPresentQueueFamilyIndex(),
                                                           surface, &supported);

        if (result != vk::Result::eSuccess || !supported) {
            LOG_CRITICAL(Render_Vulkan, "Surface not supported by physical device");
            return false;
        }

        // Then validate surface capabilities
        const auto caps = physical_device.getSurfaceCapabilitiesKHR(surface);
        if (!ValidateSurfaceCapabilities(caps, vk::Extent2D{width, height})) {
            return false;
        }

        return true;
    }
}; // class ValidationHelper
} // namespace Vulkan
