// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <exception>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/memory_detect.h"
#include "common/profiling.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "video_core/gpu.h"
#include "video_core/pica/pica_core.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_memory_util.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#include "video_core/host_shaders/vulkan_present_anaglyph_dubois_frag.h"
#include "video_core/host_shaders/vulkan_present_anaglyph_rendepth_frag.h"
#include "video_core/host_shaders/vulkan_present_frag.h"
#include "video_core/host_shaders/vulkan_present_interlaced_frag.h"
#include "video_core/host_shaders/vulkan_present_vert.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

constexpr u32 VERTEX_BUFFER_SIZE = sizeof(ScreenRectVertex) * 8192;

constexpr std::array<f32, 4 * 4> MakeOrthographicMatrix(u32 width, u32 height) {
    // clang-format off
    return { 2.f / width, 0.f,         0.f, -1.f,
            0.f,         2.f / height, 0.f, -1.f,
            0.f,         0.f,          1.f,  0.f,
            0.f,         0.f,          0.f,  1.f};
    // clang-format on
}

constexpr static std::array<vk::DescriptorSetLayoutBinding, 1> PRESENT_BINDINGS = {{
    {0, vk::DescriptorType::eCombinedImageSampler, 3, vk::ShaderStageFlagBits::eFragment},
}};

RendererVulkan::RendererVulkan(Core::System& system, Pica::PicaCore& pica_,
                               Frontend::EmuWindow& window, Frontend::EmuWindow* secondary_window)
    : RendererBase{system, window, secondary_window},
      validation{std::make_unique<ValidationHelper>()}, memory{system.Memory()}, pica{pica_},
      instance{window, Settings::values.physical_device.GetValue()}, scheduler{instance},
      renderpass_cache{instance, scheduler}, main_window{window, instance, scheduler},
      vertex_buffer{instance, scheduler, vk::BufferUsageFlagBits::eVertexBuffer,
                    VERTEX_BUFFER_SIZE},
      update_queue{instance},
      rasterizer{
          memory,   pica,      system.CustomTexManager(), *this,        render_window,
          instance, scheduler, renderpass_cache,          update_queue, main_window.ImageCount()},
      present_heap{instance, scheduler.GetMasterSemaphore(), PRESENT_BINDINGS, 32} {
    try {
        // Initial device validation
        ValidateDeviceProperties();

        // Validate vertex buffer size and alignment
        if (!validation->ValidateBufferProperties(
                instance.GetPhysicalDevice().getProperties(), // Use physical device properties
                VERTEX_BUFFER_SIZE, vk::BufferUsageFlagBits::eVertexBuffer)) {
            throw std::runtime_error("Invalid buffer configuration");
        }

        // Validate window surface
        if (!validation->ValidateSurfaceCapabilities(
                instance.GetPhysicalDevice().getSurfaceCapabilitiesKHR(window.GetWindowSurface()),
                vk::Extent2D{window.GetWidth(), window.GetHeight()})) {
            throw std::runtime_error("Invalid surface configuration");
        }

        // Proceed with initialization if validation passes
        if (!CompileShaders()) {
            throw std::runtime_error("Failed to compile shaders");
        }

        if (!BuildLayouts()) {
            throw std::runtime_error("Failed to build layouts");
        }

        if (!BuildPipelines()) {
            throw std::runtime_error("Failed to build pipelines");
        }

        ReloadPipeline();

        // Validate secondary window if present
        if (secondary_window) {
            second_window = std::make_unique<PresentWindow>(*secondary_window, instance, scheduler);
            if (!second_window ||
                !validation->ValidateSurfaceCapabilities(
                    instance.GetPhysicalDevice().getSurfaceCapabilitiesKHR(
                        secondary_window->GetWindowSurface()),
                    vk::Extent2D{static_cast<u32>(secondary_window->GetFramebufferWidth()),
                                 static_cast<u32>(secondary_window->GetFramebufferHeight())})) {
                throw std::runtime_error("Secondary window validation failed");
            }
        }

        // Final validation of the complete renderer state
        if (!ValidateRenderState()) {
            throw std::runtime_error("Final renderer state validation failed");
        }

    } catch (const vk::SystemError& e) {
        LOG_CRITICAL(Render_Vulkan, "Vulkan error during renderer initialization: {}", e.what());
        throw;
    } catch (const std::exception& e) {
        LOG_CRITICAL(Render_Vulkan, "Error during renderer initialization: {}", e.what());
        throw;
    }
}

RendererVulkan::~RendererVulkan() {
    vk::Device device = instance.GetDevice();
    scheduler.Finish();
    main_window.WaitPresent();
    device.waitIdle();

    device.destroyShaderModule(present_vertex_shader);
    for (u32 i = 0; i < PRESENT_PIPELINES; i++) {
        device.destroyPipeline(present_pipelines[i]);
        device.destroyShaderModule(present_shaders[i]);
    }

    for (auto& sampler : present_samplers) {
        device.destroySampler(sampler);
    }

    for (auto& info : screen_infos) {
        device.destroyImageView(info.texture.image_view);
        vmaDestroyImage(instance.GetAllocator(), info.texture.image, info.texture.allocation);
    }
}

void RendererVulkan::Sync() {
    rasterizer.SyncEntireState();
}

void RendererVulkan::ValidateDeviceProperties() {
    const auto& props = instance.GetPhysicalDevice().getProperties();

    // Validate maximum dimensions
    if (props.limits.maxImageDimension2D < std::max(TOP_SCREEN_WIDTH, BOTTOM_SCREEN_WIDTH)) {
        LOG_CRITICAL(Render_Vulkan, "Device doesn't support required image dimensions");
        throw std::runtime_error("Inadequate device capabilities");
    }

    // Validate queue family support
    bool found_graphics_queue = false;
    for (const auto& queue_family : instance.GetPhysicalDevice().getQueueFamilyProperties()) {
        if (ValidationHelper::ValidateQueueFamilyProperties(queue_family,
                                                            vk::QueueFlagBits::eGraphics)) {
            found_graphics_queue = true;
            break;
        }
    }
    if (!found_graphics_queue) {
        throw std::runtime_error("No suitable graphics queue found");
    }

    // Validate format support for 3DS textures
    const auto required_features =
        vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferDst;
    if (!ValidationHelper::ValidateFormatSupport(instance.GetPhysicalDevice(),
                                                 vk::Format::eR8G8B8A8Unorm, required_features)) {
        throw std::runtime_error("Required texture format not supported");
    }
}

void RendererVulkan::PrepareRendertarget() {
    LOG_DEBUG(Render_Vulkan, "Preparing rendertarget");
    const auto& framebuffer_config = pica.regs.framebuffer_config;
    const auto& regs_lcd = pica.regs_lcd;

    try {
        for (u32 i = 0; i < 3; i++) {
            const u32 fb_id = i == 2 ? 1 : 0;
            const auto& framebuffer = framebuffer_config[fb_id];
            auto& texture = screen_infos[i].texture;

            // Validate texture before use
            if (!texture.image || !texture.image_view) {
                LOG_ERROR(Render_Vulkan, "Texture at index {} is not properly initialized", i);
                if (!ConfigureFramebufferTexture(texture, framebuffer)) {
                    LOG_CRITICAL(Render_Vulkan, "Failed to initialize texture at index {}", i);
                    continue;
                }
            }

            LOG_DEBUG(Render_Vulkan, "Processing framebuffer {} with format {}", i,
                      static_cast<u32>(texture.format));

            const auto color_fill =
                fb_id == 0 ? regs_lcd.color_fill_top : regs_lcd.color_fill_bottom;
            if (color_fill.is_enabled) {
                LOG_DEBUG(Render_Vulkan, "Color fill enabled for buffer {}", i);
                screen_infos[i].image_view = texture.image_view;
                FillScreen(color_fill.AsVector(), texture);
                continue;
            }

            if (texture.width != framebuffer.width || texture.height != framebuffer.height ||
                texture.format != framebuffer.color_format) {
                LOG_DEBUG(Render_Vulkan, "Reconfiguring framebuffer {} due to size/format change",
                          i);
                if (!ConfigureFramebufferTexture(texture, framebuffer)) {
                    LOG_ERROR(Render_Vulkan, "Failed to reconfigure framebuffer {}", i);
                    continue;
                }
            }
            if (!LoadFBToScreenInfo(framebuffer, screen_infos[i], i == 1)) {
                LOG_ERROR(Render_Vulkan, "Failed to load framebuffer for screen {}", i);
                continue;
            }
        }
        LOG_DEBUG(Render_Vulkan, "PrepareRendertarget completed successfully");
    } catch (const vk::SystemError& e) {
        LOG_CRITICAL(Render_Vulkan, "Vulkan error in PrepareRendertarget: {}", e.what());
    } catch (const std::exception& err) {
        LOG_CRITICAL(Render_Vulkan, "Error in PrepareRendertarget: {}", err.what());
    }
}

void RendererVulkan::PrepareDraw(Frame* frame, const Layout::FramebufferLayout& layout) {
    if (!frame) {
        LOG_CRITICAL(Render_Vulkan, "Null frame in PrepareDraw");
        return;
    }

    const auto sampler = present_samplers[!Settings::values.filter_mode.GetValue()];
    const auto present_set = present_heap.Commit();

    if (!validation->ValidateDescriptorSet(present_set)) {
        return;
    }

    for (u32 index = 0; index < screen_infos.size(); index++) {
        if (!validation->ValidateScreenInfo(screen_infos[index])) {
            LOG_CRITICAL(Render_Vulkan, "Invalid screen info at index {}", index);
            continue;
        }
        update_queue.AddImageSampler(present_set, 0, index, screen_infos[index].image_view,
                                     sampler);
    }

    renderpass_cache.EndRendering();
    scheduler.Record([this, layout, frame, present_set, renderpass = main_window.Renderpass(),
                      index = current_pipeline](vk::CommandBuffer cmdbuf) {
        if (!validation->ValidateCommandBuffer(cmdbuf)) {
            return;
        }

        const vk::Viewport viewport = {
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(layout.width),
            .height = static_cast<float>(layout.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        const vk::Rect2D scissor = {
            .offset = {0, 0},
            .extent = {layout.width, layout.height},
        };

        const vk::Extent2D framebuffer_extent = {frame->width, frame->height};
        if (!validation->ValidateViewport(viewport, framebuffer_extent) ||
            !validation->ValidateScissor(scissor, framebuffer_extent)) {
            return;
        }

        cmdbuf.setViewport(0, viewport);
        cmdbuf.setScissor(0, scissor);

        const vk::ClearValue clear{.color = clear_color};
        const vk::PipelineLayout layout{*present_pipeline_layout};

        if (!validation->ValidatePipelineState(present_pipelines[index], layout)) {
            return;
        }

        const vk::RenderPassBeginInfo renderpass_begin_info = {
            .renderPass = renderpass,
            .framebuffer = frame->framebuffer,
            .renderArea =
                vk::Rect2D{
                    .offset = {0, 0},
                    .extent = {frame->width, frame->height},
                },
            .clearValueCount = 1,
            .pClearValues = &clear,
        };

        if (!validation->ValidateRenderPassBegin(renderpass_begin_info)) {
            return;
        }

        cmdbuf.beginRenderPass(renderpass_begin_info, vk::SubpassContents::eInline);
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, present_pipelines[index]);
        cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, present_set, {});
    });
}

void RendererVulkan::RenderToWindow(PresentWindow& window, const Layout::FramebufferLayout& layout,
                                    bool flipped) {
    Frame* frame = window.GetRenderFrame();

    if (!validation->ValidateFrameDimensions(layout.width, layout.height,
                                             instance.GetPhysicalDevice().getProperties().limits)) {
        return;
    }

    if (layout.width != frame->width || layout.height != frame->height) {
        window.WaitPresent();
        scheduler.Finish();
        window.RecreateFrame(frame, layout.width, layout.height);
    }

    DrawScreens(frame, layout, flipped);
    scheduler.Flush(frame->render_ready);

    window.Present(frame);
}

bool RendererVulkan::LoadFBToScreenInfo(const Pica::FramebufferConfig& framebuffer,
                                        ScreenInfo& screen_info, bool right_eye) {
    if (!ValidationHelper::ValidateFramebufferConfig(
            framebuffer, right_eye ? framebuffer.address_right1 : framebuffer.address_left1,
            framebuffer.stride)) {
        return false;
    }
    try {
        LOG_DEBUG(Render_Vulkan, "Loading framebuffer to screen info");

        if (framebuffer.address_right1 == 0 || framebuffer.address_right2 == 0) {
            LOG_DEBUG(Render_Vulkan, "No right eye buffer available, forcing left eye");
            right_eye = false;
        }

        const PAddr framebuffer_addr =
            framebuffer.active_fb == 0
                ? (right_eye ? framebuffer.address_right1 : framebuffer.address_left1)
                : (right_eye ? framebuffer.address_right2 : framebuffer.address_left2);

        LOG_DEBUG(Render_Vulkan, "Using framebuffer address 0x{:08x}", framebuffer_addr);
        LOG_TRACE(Render_Vulkan, "0x{:08x} bytes from 0x{:08x}({}x{}), fmt {:x}",
                  framebuffer.stride * framebuffer.height, framebuffer_addr,
                  framebuffer.width.Value(), framebuffer.height.Value(), framebuffer.format);

        const u32 bpp = Pica::BytesPerPixel(framebuffer.color_format);
        const std::size_t pixel_stride = framebuffer.stride / bpp;

        if (pixel_stride * bpp != framebuffer.stride) {
            LOG_ERROR(Render_Vulkan, "Invalid pixel stride");
            return false;
        }

        if (pixel_stride % 4 != 0) {
            LOG_ERROR(Render_Vulkan, "Pixel stride not aligned to 4");
            return false;
        }

        LOG_DEBUG(Render_Vulkan, "Accelerating display...");
        if (!rasterizer.AccelerateDisplay(framebuffer, framebuffer_addr,
                                          static_cast<u32>(pixel_stride), screen_info)) {
            LOG_ERROR(Render_Vulkan, "Failed to accelerate display");
            // Reset the screen info's display texture to its own permanent texture
            screen_info.image_view = screen_info.texture.image_view;
            screen_info.texcoords = {0.f, 0.f, 1.f, 1.f};
            LOG_ERROR(Render_Vulkan, "Failed to accelerate display");

            // Add a check to ensure the image is valid
            if (!screen_info.texture.image) {
                LOG_CRITICAL(Render_Vulkan, "Invalid image handle in screen_info.texture.image");
                return false;
            }
            return false;
        }

        if (!validation->ValidateScreenInfo(screen_info)) {
            return false;
        }

        LOG_DEBUG(Render_Vulkan, "Framebuffer loaded successfully");
        return true;
    } catch (const vk::SystemError& error) {
        LOG_CRITICAL(Render_Vulkan, "Vulkan error in LoadFBToScreenInfo: {}", error.what());
        return false;
    } catch (const std::exception& error) {
        LOG_CRITICAL(Render_Vulkan, "Error in LoadFBToScreenInfo: {}", error.what());
        return false;
    }
}

bool RendererVulkan::CompileShaders() {
    try {
        const vk::Device device = instance.GetDevice();
        const std::string_view preamble =
            instance.IsImageArrayDynamicIndexSupported() ? "#define ARRAY_DYNAMIC_INDEX" : "";
        present_vertex_shader =
            Compile(HostShaders::VULKAN_PRESENT_VERT, vk::ShaderStageFlagBits::eVertex, device);
        present_shaders[0] = Compile(HostShaders::VULKAN_PRESENT_FRAG,
                                     vk::ShaderStageFlagBits::eFragment, device, preamble);
        present_shaders[1] = Compile(HostShaders::VULKAN_PRESENT_ANAGLYPH_RENDEPTH_FRAG,
                                     vk::ShaderStageFlagBits::eFragment, device, preamble);
        present_shaders[2] = Compile(HostShaders::VULKAN_PRESENT_ANAGLYPH_DUBOIS_FRAG,
                                     vk::ShaderStageFlagBits::eFragment, device, preamble);
        present_shaders[3] = Compile(HostShaders::VULKAN_PRESENT_INTERLACED_FRAG,
                                     vk::ShaderStageFlagBits::eFragment, device, preamble);

        auto properties = instance.GetPhysicalDevice().getProperties();
        for (std::size_t i = 0; i < present_samplers.size(); i++) {
            const vk::Filter filter_mode = i == 0 ? vk::Filter::eLinear : vk::Filter::eNearest;
            const vk::SamplerCreateInfo sampler_info = {
                .magFilter = filter_mode,
                .minFilter = filter_mode,
                .mipmapMode = vk::SamplerMipmapMode::eLinear,
                .addressModeU = vk::SamplerAddressMode::eClampToEdge,
                .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                .anisotropyEnable = instance.IsAnisotropicFilteringSupported(),
                .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
                .compareEnable = false,
                .compareOp = vk::CompareOp::eAlways,
                .borderColor = vk::BorderColor::eIntOpaqueBlack,
                .unnormalizedCoordinates = false,
            };

            present_samplers[i] = device.createSampler(sampler_info);
        }
        return true;
    } catch (const std::exception& e) {
        LOG_CRITICAL(Render_Vulkan, "Shader compilation failed: {}", e.what());
        return false;
    }
}

bool RendererVulkan::BuildLayouts() {
    try {
        const vk::PushConstantRange push_range = {
            .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            .offset = 0,
            .size = sizeof(PresentUniformData),
        };

        const auto descriptor_set_layout = present_heap.Layout();
        const vk::PipelineLayoutCreateInfo layout_info = {
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        present_pipeline_layout = instance.GetDevice().createPipelineLayoutUnique(layout_info);
        return true;
    } catch (const std::exception& e) {
        LOG_CRITICAL(Render_Vulkan, "Layout building failed: {}", e.what());
        return false;
    }
}

bool RendererVulkan::BuildPipelines() {
    try {
        // Add validation for pipeline creation parameters
        if (!present_pipeline_layout) {
            LOG_CRITICAL(Render_Vulkan, "Pipeline layout not created");
            return false;
        }

        // Validate vertex input
        const auto& device_props = instance.GetPhysicalDevice().getProperties();
        if (sizeof(ScreenRectVertex) > device_props.limits.maxVertexInputBindingStride) {
            LOG_CRITICAL(Render_Vulkan, "Vertex size exceeds device limits");
            return false;
        }

        const vk::VertexInputBindingDescription binding = {
            .binding = 0,
            .stride = sizeof(ScreenRectVertex),
            .inputRate = vk::VertexInputRate::eVertex,
        };

        const std::array attributes = {
            vk::VertexInputAttributeDescription{
                .location = 0,
                .binding = 0,
                .format = vk::Format::eR32G32Sfloat,
                .offset = offsetof(ScreenRectVertex, position),
            },
            vk::VertexInputAttributeDescription{
                .location = 1,
                .binding = 0,
                .format = vk::Format::eR32G32Sfloat,
                .offset = offsetof(ScreenRectVertex, tex_coord),
            },
        };

        const vk::PipelineVertexInputStateCreateInfo vertex_input_info = {
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &binding,
            .vertexAttributeDescriptionCount = static_cast<u32>(attributes.size()),
            .pVertexAttributeDescriptions = attributes.data(),
        };

        const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
            .topology = vk::PrimitiveTopology::eTriangleStrip,
            .primitiveRestartEnable = false,
        };

        const vk::PipelineRasterizationStateCreateInfo raster_state = {
            .depthClampEnable = false,
            .rasterizerDiscardEnable = false,
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eClockwise,
            .depthBiasEnable = false,
            .lineWidth = 1.0f,
        };

        const vk::PipelineMultisampleStateCreateInfo multisampling = {
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = false,
        };

        const vk::PipelineColorBlendAttachmentState colorblend_attachment = {
            .blendEnable = false,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };

        const vk::PipelineColorBlendStateCreateInfo color_blending = {
            .logicOpEnable = false,
            .attachmentCount = 1,
            .pAttachments = &colorblend_attachment,
            .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f},
        };

        const vk::Viewport placeholder_viewport = vk::Viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
        const vk::Rect2D placeholder_scissor = vk::Rect2D{{0, 0}, {1, 1}};
        const vk::PipelineViewportStateCreateInfo viewport_info = {
            .viewportCount = 1,
            .pViewports = &placeholder_viewport,
            .scissorCount = 1,
            .pScissors = &placeholder_scissor,
        };

        const std::array dynamic_states = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
        };

        const vk::PipelineDynamicStateCreateInfo dynamic_info = {
            .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
            .pDynamicStates = dynamic_states.data(),
        };

        const vk::PipelineDepthStencilStateCreateInfo depth_info = {
            .depthTestEnable = false,
            .depthWriteEnable = false,
            .depthCompareOp = vk::CompareOp::eAlways,
            .depthBoundsTestEnable = false,
            .stencilTestEnable = false,
        };

        for (u32 i = 0; i < PRESENT_PIPELINES; i++) {
            const std::array shader_stages = {
                vk::PipelineShaderStageCreateInfo{
                    .stage = vk::ShaderStageFlagBits::eVertex,
                    .module = present_vertex_shader,
                    .pName = "main",
                },
                vk::PipelineShaderStageCreateInfo{
                    .stage = vk::ShaderStageFlagBits::eFragment,
                    .module = present_shaders[i],
                    .pName = "main",
                },
            };

            const vk::GraphicsPipelineCreateInfo pipeline_info = {
                .stageCount = static_cast<u32>(shader_stages.size()),
                .pStages = shader_stages.data(),
                .pVertexInputState = &vertex_input_info,
                .pInputAssemblyState = &input_assembly,
                .pViewportState = &viewport_info,
                .pRasterizationState = &raster_state,
                .pMultisampleState = &multisampling,
                .pDepthStencilState = &depth_info,
                .pColorBlendState = &color_blending,
                .pDynamicState = &dynamic_info,
                .layout = *present_pipeline_layout,
                .renderPass = main_window.Renderpass(),
            };

            const auto [result, pipeline] =
                instance.GetDevice().createGraphicsPipeline({}, pipeline_info);
            ASSERT_MSG(result == vk::Result::eSuccess, "Unable to build present pipelines");
            present_pipelines[i] = pipeline;
        }
        return true;
    } catch (const vk::SystemError& err) {
        LOG_CRITICAL(Render_Vulkan, "Failed to build pipelines: {}", err.what());
        return false;
    } catch (const std::exception& e) {
        LOG_CRITICAL(Render_Vulkan, "Pipeline building failed: {}", e.what());
        return false;
    }
}

bool RendererVulkan::ConfigureFramebufferTexture(TextureInfo& texture,
                                                 const Pica::FramebufferConfig& framebuffer) {
    try {
        LOG_DEBUG(Render_Vulkan, "Configuring framebuffer texture {}x{} format={}",
                  framebuffer.width.Value(), framebuffer.height.Value(),
                  static_cast<u32>(framebuffer.color_format.Value()));

        if (!ValidationHelper::ValidateFramebufferConfig(framebuffer, fb_addr,
                                                         framebuffer.stride)) {
            return false;
        }

        vk::Device device = instance.GetDevice();
        if (texture.image_view) {
            LOG_DEBUG(Render_Vulkan, "Destroying old image view");
            main_window.WaitPresent();
            device.destroyImageView(texture.image_view);
        }
        if (texture.image) {
            LOG_DEBUG(Render_Vulkan, "Destroying old image");
            main_window.WaitPresent();
            vmaDestroyImage(instance.GetAllocator(), texture.image, texture.allocation);
        }

        const VideoCore::PixelFormat pixel_format =
            VideoCore::PixelFormatFromGPUPixelFormat(framebuffer.color_format);
        const vk::Format format = instance.GetTraits(pixel_format).native;
        LOG_DEBUG(Render_Vulkan, "Using Vulkan format {}", vk::to_string(format));

        const vk::ImageCreateInfo image_info = {
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {framebuffer.width, framebuffer.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .usage = vk::ImageUsageFlagBits::eSampled,
        };

        // Validate image creation parameters
        if (!ValidationHelper::ValidateImageProperties(
                image_info, instance.GetPhysicalDevice().getProperties().limits)) {
            return false;
        }

        const VmaAllocationCreateInfo alloc_info = {
            .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags = 0,
            .preferredFlags = 0,
            .pool = VK_NULL_HANDLE,
            .pUserData = nullptr,
        };

        VkImage unsafe_image{};
        VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);

        LOG_DEBUG(Render_Vulkan, "Creating image...");
        VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info,
                                         &unsafe_image, &texture.allocation, nullptr);
        if (result != VK_SUCCESS) [[unlikely]] {
            LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}", result);
            return false;
        }
        texture.image = vk::Image{unsafe_image};
        LOG_DEBUG(Render_Vulkan, "Image created successfully");

        const vk::ImageViewCreateInfo view_info = {
            .image = texture.image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        LOG_DEBUG(Render_Vulkan, "Creating image view...");
        texture.image_view = device.createImageView(view_info);
        LOG_DEBUG(Render_Vulkan, "Image view created successfully");

        texture.width = framebuffer.width;
        texture.height = framebuffer.height;
        texture.format = framebuffer.color_format;

        LOG_DEBUG(Render_Vulkan, "Framebuffer texture configured successfully");
        return true;
    } catch (const vk::SystemError& error) {
        LOG_CRITICAL(Render_Vulkan, "Vulkan error in ConfigureFramebufferTexture: {}",
                     error.what());
        return false;
    } catch (const std::exception& error) {
        LOG_CRITICAL(Render_Vulkan, "Error in ConfigureFramebufferTexture: {}", error.what());
        return false;
    }
}

void RendererVulkan::FillScreen(Common::Vec3<u8> color, const TextureInfo& texture) {
    const vk::ClearColorValue clear_color = {
        .float32 =
            std::array{
                color.r() / 255.0f,
                color.g() / 255.0f,
                color.b() / 255.0f,
                1.0f,
            },
    };

    renderpass_cache.EndRendering();
    scheduler.Record([image = texture.image, clear_color](vk::CommandBuffer cmdbuf) {
        const vk::ImageSubresourceRange range = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        };

        const vk::ImageMemoryBarrier pre_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        };

        const vk::ImageMemoryBarrier post_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = range,
        };

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barrier);

        cmdbuf.clearColorImage(image, vk::ImageLayout::eTransferDstOptimal, clear_color, range);

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eFragmentShader,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);
    });
}

void RendererVulkan::ReloadPipeline() {
    const Settings::StereoRenderOption render_3d = Settings::values.render_3d.GetValue();
    switch (render_3d) {
    case Settings::StereoRenderOption::Anaglyph:
        current_pipeline = 1;
        break;
    case Settings::StereoRenderOption::Interlaced:
    case Settings::StereoRenderOption::ReverseInterlaced:
        current_pipeline = 2;
        draw_info.reverse_interlaced = render_3d == Settings::StereoRenderOption::ReverseInterlaced;
        break;
    default:
        current_pipeline = 0;
        break;
    }
}

void RendererVulkan::DrawSingleScreen(u32 screen_id, float x, float y, float w, float h,
                                      Layout::DisplayOrientation orientation) {
    // Validate parameters
    if (screen_id >= screen_infos.size()) {
        LOG_CRITICAL(Render_Vulkan, "Invalid screen_id: {}", screen_id);
        return;
    }

    if (w <= 0.0f || h <= 0.0f) {
        LOG_CRITICAL(Render_Vulkan, "Invalid dimensions: {}x{}", w, h);
        return;
    }

    const ScreenInfo& screen_info = screen_infos[screen_id];
    if (!ValidationHelper::ValidateScreenInfo(screen_info)) {
        return;
    }

    const auto& texcoords = screen_info.texcoords;

    std::array<ScreenRectVertex, 4> vertices;
    switch (orientation) {
    case Layout::DisplayOrientation::Landscape:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x, y + h, texcoords.top, texcoords.left),
            ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.right),
        }};
        break;
    case Layout::DisplayOrientation::Portrait:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x + w, y, texcoords.top, texcoords.right),
            ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.left),
        }};
        std::swap(h, w);
        break;
    case Layout::DisplayOrientation::LandscapeFlipped:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.top, texcoords.right),
            ScreenRectVertex(x + w, y, texcoords.top, texcoords.left),
            ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x + w, y + h, texcoords.bottom, texcoords.left),
        }};
        break;
    case Layout::DisplayOrientation::PortraitFlipped:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.top, texcoords.left),
            ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x, y + h, texcoords.top, texcoords.right),
            ScreenRectVertex(x + w, y + h, texcoords.bottom, texcoords.right),
        }};
        std::swap(h, w);
        break;
    default:
        LOG_ERROR(Render_Vulkan, "Unknown DisplayOrientation: {}", orientation);
        break;
    }

    const u64 size = sizeof(ScreenRectVertex) * vertices.size();
    auto [data, offset, invalidate] = vertex_buffer.Map(size, 16);
    std::memcpy(data, vertices.data(), size);
    vertex_buffer.Commit(size);

    const u32 scale_factor = GetResolutionScaleFactor();
    draw_info.i_resolution =
        Common::MakeVec(static_cast<f32>(screen_info.texture.width * scale_factor),
                        static_cast<f32>(screen_info.texture.height * scale_factor),
                        1.0f / static_cast<f32>(screen_info.texture.width * scale_factor),
                        1.0f / static_cast<f32>(screen_info.texture.height * scale_factor));
    draw_info.o_resolution = Common::MakeVec(h, w, 1.0f / h, 1.0f / w);
    draw_info.screen_id_l = screen_id;

    scheduler.Record([this, offset = offset, info = draw_info](vk::CommandBuffer cmdbuf) {
        const u32 first_vertex = static_cast<u32>(offset) / sizeof(ScreenRectVertex);
        cmdbuf.pushConstants(*present_pipeline_layout,
                             vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                             0, sizeof(info), &info);

        cmdbuf.bindVertexBuffers(0, vertex_buffer.Handle(), {0});
        cmdbuf.draw(4, 1, first_vertex, 0);
    });
}

void RendererVulkan::DrawSingleScreenStereo(u32 screen_id_l, u32 screen_id_r, float x, float y,
                                            float w, float h,
                                            Layout::DisplayOrientation orientation) {
    const ScreenInfo& screen_info_l = screen_infos[screen_id_l];
    const auto& texcoords = screen_info_l.texcoords;

    std::array<ScreenRectVertex, 4> vertices;
    switch (orientation) {
    case Layout::DisplayOrientation::Landscape:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x, y + h, texcoords.top, texcoords.left),
            ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.right),
        }};
        break;
    case Layout::DisplayOrientation::Portrait:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x + w, y, texcoords.top, texcoords.right),
            ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.left),
        }};
        std::swap(h, w);
        break;
    case Layout::DisplayOrientation::LandscapeFlipped:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.top, texcoords.right),
            ScreenRectVertex(x + w, y, texcoords.top, texcoords.left),
            ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.right),
            ScreenRectVertex(x + w, y + h, texcoords.bottom, texcoords.left),
        }};
        break;
    case Layout::DisplayOrientation::PortraitFlipped:
        vertices = {{
            ScreenRectVertex(x, y, texcoords.top, texcoords.left),
            ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.left),
            ScreenRectVertex(x, y + h, texcoords.top, texcoords.right),
            ScreenRectVertex(x + w, y + h, texcoords.bottom, texcoords.right),
        }};
        std::swap(h, w);
        break;
    default:
        LOG_ERROR(Render_Vulkan, "Unknown DisplayOrientation: {}", orientation);
        break;
    }

    const u64 size = sizeof(ScreenRectVertex) * vertices.size();
    auto [data, offset, invalidate] = vertex_buffer.Map(size, 16);
    std::memcpy(data, vertices.data(), size);
    vertex_buffer.Commit(size);

    const u32 scale_factor = GetResolutionScaleFactor();
    draw_info.i_resolution =
        Common::MakeVec(static_cast<f32>(screen_info_l.texture.width * scale_factor),
                        static_cast<f32>(screen_info_l.texture.height * scale_factor),
                        1.0f / static_cast<f32>(screen_info_l.texture.width * scale_factor),
                        1.0f / static_cast<f32>(screen_info_l.texture.height * scale_factor));
    draw_info.o_resolution = Common::MakeVec(h, w, 1.0f / h, 1.0f / w);
    draw_info.screen_id_l = screen_id_l;
    draw_info.screen_id_r = screen_id_r;

    scheduler.Record([this, offset = offset, info = draw_info](vk::CommandBuffer cmdbuf) {
        const u32 first_vertex = static_cast<u32>(offset) / sizeof(ScreenRectVertex);
        cmdbuf.pushConstants(*present_pipeline_layout,
                             vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                             0, sizeof(info), &info);

        cmdbuf.bindVertexBuffers(0, vertex_buffer.Handle(), {0});
        cmdbuf.draw(4, 1, first_vertex, 0);
    });
}

void RendererVulkan::DrawTopScreen(const Layout::FramebufferLayout& layout,
                                   const Common::Rectangle<u32>& top_screen) {
    if (!layout.top_screen_enabled) {
        return;
    }
    int leftside, rightside;
    leftside = Settings::values.swap_eyes_3d.GetValue() ? 1 : 0;
    rightside = Settings::values.swap_eyes_3d.GetValue() ? 0 : 1;
    const float top_screen_left = static_cast<float>(top_screen.left);
    const float top_screen_top = static_cast<float>(top_screen.top);
    const float top_screen_width = static_cast<float>(top_screen.GetWidth());
    const float top_screen_height = static_cast<float>(top_screen.GetHeight());

    const auto orientation = layout.is_rotated ? Layout::DisplayOrientation::Landscape
                                               : Layout::DisplayOrientation::Portrait;
    switch (Settings::values.render_3d.GetValue()) {
    case Settings::StereoRenderOption::Off: {
        const int eye = static_cast<int>(Settings::values.mono_render_option.GetValue());
        DrawSingleScreen(eye, top_screen_left, top_screen_top, top_screen_width, top_screen_height,
                         orientation);
        break;
    }
    case Settings::StereoRenderOption::SideBySide: {
        DrawSingleScreen(leftside, top_screen_left / 2, top_screen_top, top_screen_width / 2,
                         top_screen_height, orientation);
        draw_info.layer = 1;
        DrawSingleScreen(rightside, static_cast<float>((top_screen_left / 2) + (layout.width / 2)),
                         top_screen_top, top_screen_width / 2, top_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::SideBySideFull: {
        DrawSingleScreen(leftside, top_screen_left, top_screen_top, top_screen_width,
                         top_screen_height, orientation);
        draw_info.layer = 1;
        DrawSingleScreen(rightside, top_screen_left + layout.width / 2, top_screen_top,
                         top_screen_width, top_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::CardboardVR: {
        DrawSingleScreen(leftside, top_screen_left, top_screen_top, top_screen_width,
                         top_screen_height, orientation);
        draw_info.layer = 1;
        DrawSingleScreen(
            rightside,
            static_cast<float>(layout.cardboard.top_screen_right_eye + (layout.width / 2)),
            top_screen_top, top_screen_width, top_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::Anaglyph:
    case Settings::StereoRenderOption::Interlaced:
    case Settings::StereoRenderOption::ReverseInterlaced: {
        DrawSingleScreenStereo(leftside, rightside, top_screen_left, top_screen_top,
                               top_screen_width, top_screen_height, orientation);
        break;
    }
    }
}

void RendererVulkan::DrawBottomScreen(const Layout::FramebufferLayout& layout,
                                      const Common::Rectangle<u32>& bottom_screen) {
    if (!layout.bottom_screen_enabled) {
        return;
    }

    const float bottom_screen_left = static_cast<float>(bottom_screen.left);
    const float bottom_screen_top = static_cast<float>(bottom_screen.top);
    const float bottom_screen_width = static_cast<float>(bottom_screen.GetWidth());
    const float bottom_screen_height = static_cast<float>(bottom_screen.GetHeight());

    const auto orientation = layout.is_rotated ? Layout::DisplayOrientation::Landscape
                                               : Layout::DisplayOrientation::Portrait;

    bool separate_win = false;
#ifndef ANDROID
    separate_win =
        (Settings::values.layout_option.GetValue() == Settings::LayoutOption::SeparateWindows);
#endif

    switch (Settings::values.render_3d.GetValue()) {
    case Settings::StereoRenderOption::Off: {
        DrawSingleScreen(2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                         bottom_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::SideBySide: {
        if (separate_win) {
            DrawSingleScreen(2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                             bottom_screen_height, orientation);
        } else {

            DrawSingleScreen(2, bottom_screen_left / 2, bottom_screen_top, bottom_screen_width / 2,
                             bottom_screen_height, orientation);
            draw_info.layer = 1;
            DrawSingleScreen(2, static_cast<float>((bottom_screen_left / 2) + (layout.width / 2)),
                             bottom_screen_top, bottom_screen_width / 2, bottom_screen_height,
                             orientation);
        }
        break;
    }
    case Settings::StereoRenderOption::SideBySideFull: {
        DrawSingleScreen(2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                         bottom_screen_height, orientation);
        draw_info.layer = 1;
        DrawSingleScreen(2, bottom_screen_left + layout.width / 2, bottom_screen_top,
                         bottom_screen_width, bottom_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::CardboardVR: {
        DrawSingleScreen(2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                         bottom_screen_height, orientation);
        draw_info.layer = 1;
        DrawSingleScreen(
            2, static_cast<float>(layout.cardboard.bottom_screen_right_eye + (layout.width / 2)),
            bottom_screen_top, bottom_screen_width, bottom_screen_height, orientation);
        break;
    }
    case Settings::StereoRenderOption::Anaglyph:
    case Settings::StereoRenderOption::Interlaced:
    case Settings::StereoRenderOption::ReverseInterlaced: {
        if (separate_win) {
            DrawSingleScreen(2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                             bottom_screen_height, orientation);
        } else {
            DrawSingleScreenStereo(2, 2, bottom_screen_left, bottom_screen_top, bottom_screen_width,
                                   bottom_screen_height, orientation);
        }
        break;
    }
    }
}

void RendererVulkan::DrawScreens(Frame* frame, const Layout::FramebufferLayout& layout,
                                 bool flipped) {
    try {
        if (!ValidationHelper::ValidateDrawParameters(frame, layout, screen_infos)) {
            return;
        }

        LOG_DEBUG(Render_Vulkan, "Starting DrawScreens with frame size: {}x{}", layout.width,
                  layout.height);

        // Validate screen info textures
        for (size_t i = 0; i < screen_infos.size(); i++) {
            if (!ValidationHelper::ValidateScreenInfo(screen_infos[i])) {
                LOG_CRITICAL(Render_Vulkan, "Invalid screen info at index {}", i);
                return;
            }
        }
        if (settings.bg_color_update_requested.exchange(false)) {
            clear_color.float32[0] = Settings::values.bg_red.GetValue();
            clear_color.float32[1] = Settings::values.bg_green.GetValue();
            clear_color.float32[2] = Settings::values.bg_blue.GetValue();
        }
        if (settings.shader_update_requested.exchange(false)) {
            ReloadPipeline();
        }

        PrepareDraw(frame, layout);

        const auto& top_screen = layout.top_screen;
        const auto& bottom_screen = layout.bottom_screen;
        draw_info.modelview = MakeOrthographicMatrix(layout.width, layout.height);

        draw_info.layer = 0;
        if (!Settings::values.swap_screen.GetValue()) {
            DrawTopScreen(layout, top_screen);
            draw_info.layer = 0;
            DrawBottomScreen(layout, bottom_screen);
        } else {
            DrawBottomScreen(layout, bottom_screen);
            draw_info.layer = 0;
            DrawTopScreen(layout, top_screen);
        }

        if (layout.additional_screen_enabled) {
            const auto& additional_screen = layout.additional_screen;
            if (!Settings::values.swap_screen.GetValue()) {
                DrawTopScreen(layout, additional_screen);
            } else {
                DrawBottomScreen(layout, additional_screen);
            }
        }

        LOG_DEBUG(Render_Vulkan, "DrawScreens completed successfully");
    } catch (const vk::SystemError& err) {
        LOG_CRITICAL(Render_Vulkan, "Vulkan error in DrawScreens: {}", err.what());
    } catch (const std::exception& err) {
        LOG_CRITICAL(Render_Vulkan, "Error in DrawScreens: {}", err.what());
    }
    scheduler.Record([](vk::CommandBuffer cmdbuf) { cmdbuf.endRenderPass(); });
}

void RendererVulkan::SwapBuffers() {
    const Layout::FramebufferLayout& layout = render_window.GetFramebufferLayout();
    PrepareRendertarget();
    RenderScreenshot();
    RenderToWindow(main_window, layout, false);
#ifndef ANDROID
    if (Settings::values.layout_option.GetValue() == Settings::LayoutOption::SeparateWindows) {
        ASSERT(secondary_window);
        const auto& secondary_layout = secondary_window->GetFramebufferLayout();
        if (!second_window) {
            second_window = std::make_unique<PresentWindow>(*secondary_window, instance, scheduler);
        }
        RenderToWindow(*second_window, secondary_layout, false);
        secondary_window->PollEvents();
    }
#endif
    rasterizer.TickFrame();
    EndFrame();
}

void RendererVulkan::RenderScreenshot() {
    if (!settings.screenshot_requested.exchange(false)) {
        return;
    }

    if (!TryRenderScreenshotWithHostMemory()) {
        RenderScreenshotWithStagingCopy();
    }

    settings.screenshot_complete_callback(false);
}

void RendererVulkan::RenderScreenshotWithStagingCopy() {
    const vk::Device device = instance.GetDevice();

    const Layout::FramebufferLayout layout{settings.screenshot_framebuffer_layout};
    const u32 width = layout.width;
    const u32 height = layout.height;

    const vk::BufferCreateInfo staging_buffer_info = {
        .size = width * height * 4,
        .usage = vk::BufferUsageFlagBits::eTransferDst,
    };

    const VmaAllocationCreateInfo alloc_create_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkBuffer unsafe_buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo alloc_info{};

    if (!ValidationHelper::ValidateBufferAllocation(instance.GetAllocator(), staging_buffer_info,
                                                    alloc_create_info, unsafe_buffer, allocation)) {
        return;
    }

    // Get the allocation info which contains the mapped pointer
    vmaGetAllocationInfo(instance.GetAllocator(), allocation, &alloc_info);

    vk::Buffer staging_buffer{unsafe_buffer};

    Frame frame{};
    main_window.RecreateFrame(&frame, width, height);

    DrawScreens(&frame, layout, false);

    scheduler.Record(
        [width, height, source_image = frame.image, staging_buffer](vk::CommandBuffer cmdbuf) {
            const vk::ImageMemoryBarrier read_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = source_image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
            const vk::ImageMemoryBarrier write_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eTransferRead,
                .dstAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = source_image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
            static constexpr vk::MemoryBarrier memory_write_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
            };

            const vk::BufferImageCopy image_copy = {
                .bufferOffset = 0,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource =
                    {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                .imageOffset = {0, 0, 0},
                .imageExtent = {width, height, 1},
            };

            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                   vk::PipelineStageFlagBits::eTransfer,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, read_barrier);
            cmdbuf.copyImageToBuffer(source_image, vk::ImageLayout::eTransferSrcOptimal,
                                     staging_buffer, image_copy);
            cmdbuf.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
                vk::DependencyFlagBits::eByRegion, memory_write_barrier, {}, write_barrier);
        });

    // Ensure the copy is fully completed before saving the screenshot
    scheduler.Finish();

    // Copy backing image data to the QImage screenshot buffer
    std::memcpy(settings.screenshot_bits, alloc_info.pMappedData, staging_buffer_info.size);

    // Destroy allocated resources
    vmaDestroyBuffer(instance.GetAllocator(), staging_buffer, allocation);
    vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
    device.destroyFramebuffer(frame.framebuffer);
    device.destroyImageView(frame.image_view);
}

bool RendererVulkan::TryRenderScreenshotWithHostMemory() {
    // If the host-memory import alignment matches the allocation granularity of the platform, then
    // the entire span of memory can be trivially imported
    const bool trivial_import =
        instance.IsExternalMemoryHostSupported() &&
        instance.GetMinImportedHostPointerAlignment() == Common::GetPageSize();
    if (!trivial_import) {
        return false;
    }

    const vk::Device device = instance.GetDevice();

    const Layout::FramebufferLayout layout{settings.screenshot_framebuffer_layout};
    const u32 width = layout.width;
    const u32 height = layout.height;

    // For a span of memory [x, x + s], import [AlignDown(x, alignment), AlignUp(x + s, alignment)]
    // and maintain an offset to the start of the data
    const u64 import_alignment = instance.GetMinImportedHostPointerAlignment();
    const uintptr_t address = reinterpret_cast<uintptr_t>(settings.screenshot_bits);
    void* aligned_pointer = reinterpret_cast<void*>(Common::AlignDown(address, import_alignment));
    const u64 offset = address % import_alignment;
    const u64 aligned_size = Common::AlignUp(offset + width * height * 4ull, import_alignment);

    // Buffer<->Image mapping for the imported imported buffer
    const vk::BufferImageCopy buffer_image_copy = {
        .bufferOffset = offset,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, 1},
    };

    const vk::MemoryHostPointerPropertiesEXT import_properties =
        device.getMemoryHostPointerPropertiesEXT(
            vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, aligned_pointer);

    if (!import_properties.memoryTypeBits) {
        // Could not import memory
        return false;
    }

    const std::optional<u32> memory_type_index = FindMemoryType(
        instance.GetPhysicalDevice().getMemoryProperties(),
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        import_properties.memoryTypeBits);

    if (!memory_type_index.has_value()) {
        // Could not find memory type index
        return false;
    }

    const vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryHostPointerInfoEXT>
        allocation_chain = {
            vk::MemoryAllocateInfo{
                .allocationSize = aligned_size,
                .memoryTypeIndex = memory_type_index.value(),
            },
            vk::ImportMemoryHostPointerInfoEXT{
                .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
                .pHostPointer = aligned_pointer,
            },
        };

    // Import host memory
    const vk::UniqueDeviceMemory imported_memory =
        device.allocateMemoryUnique(allocation_chain.get());

    const vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfo> buffer_info =
        {
            vk::BufferCreateInfo{
                .size = aligned_size,
                .usage = vk::BufferUsageFlagBits::eTransferDst,
                .sharingMode = vk::SharingMode::eExclusive,
            },
            vk::ExternalMemoryBufferCreateInfo{
                .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
            },
        };

    // Bind imported memory to buffer
    const vk::UniqueBuffer imported_buffer = device.createBufferUnique(buffer_info.get());
    device.bindBufferMemory(imported_buffer.get(), imported_memory.get(), 0);

    Frame frame{};
    main_window.RecreateFrame(&frame, width, height);

    DrawScreens(&frame, layout, false);

    scheduler.Record([buffer_image_copy, source_image = frame.image,
                      imported_buffer = imported_buffer.get()](vk::CommandBuffer cmdbuf) {
        const vk::ImageMemoryBarrier read_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = source_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        const vk::ImageMemoryBarrier write_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferRead,
            .dstAccessMask = vk::AccessFlagBits::eMemoryWrite,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = source_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        static constexpr vk::MemoryBarrier memory_write_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
            .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
        };

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, read_barrier);
        cmdbuf.copyImageToBuffer(source_image, vk::ImageLayout::eTransferSrcOptimal,
                                 imported_buffer, buffer_image_copy);
        cmdbuf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
            vk::DependencyFlagBits::eByRegion, memory_write_barrier, {}, write_barrier);
    });

    // Ensure the copy is fully completed before saving the screenshot
    scheduler.Finish();

    // Image data has been copied directly to host memory
    device.destroyFramebuffer(frame.framebuffer);
    device.destroyImageView(frame.image_view);

    return true;
}

} // namespace Vulkan
