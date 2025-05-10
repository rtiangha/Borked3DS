// Copyright 2025 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_types.h"
#include "common/math_util.h"
#include "video_core/pica/regs_framebuffer.h"
#include "video_core/pica_types.h"
#include "video_core/rasterizer_cache/rasterizer_cache_base.h"
#include "video_core/texture/texture_decode.h"

namespace Vulkan {

// 3DS specific constants
static constexpr u32 TOP_SCREEN_WIDTH = 400;
static constexpr u32 TOP_SCREEN_HEIGHT = 240;
static constexpr u32 BOTTOM_SCREEN_WIDTH = 320;
static constexpr u32 BOTTOM_SCREEN_HEIGHT = 240;

struct TextureInfo {
    u32 width;
    u32 height;
    Pica::PixelFormat format;
    vk::Image image;
    vk::ImageView image_view;
    VmaAllocation allocation;
};

struct ScreenInfo {
    TextureInfo texture;
    Common::Rectangle<f32> texcoords;
    vk::ImageView image_view;
};

struct ScreenRectVertex {
    ScreenRectVertex() = default;
    ScreenRectVertex(float x, float y, float u, float v) : position{x, y}, tex_coord{u, v} {}

    std::array<float, 2> position;
    std::array<float, 2> tex_coord;
};
} // namespace Vulkan
