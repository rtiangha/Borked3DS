// Copyright 2025 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <glad/gl.h>

namespace OpenGL {

class GLCompatibility {
public:
    static bool Initialize();

    static bool HasTextureBuffer() {
        return has_texture_buffer;
    }
    static bool HasImageAtomic() {
        return has_image_atomic;
    }
    static bool HasBlendMinMax() {
        return has_blend_minmax;
    }
    static bool HasClipDistance() {
        return has_clip_distance;
    }
    static bool HasGeometryShader() {
        return has_geometry_shader;
    }
    static bool IsOpenGLES() {
        return is_gles;
    }

private:
    static bool has_texture_buffer;
    static bool has_image_atomic;
    static bool has_blend_minmax;
    static bool has_clip_distance;
    static bool has_geometry_shader;
    static bool is_gles;
};

// Utility functions for feature emulation
namespace GLFeatures {
void EmulateTextureBuffer(GLuint& texture, GLuint& buffer, GLenum format);
void EmulateImageAtomic(GLuint unit, GLuint texture, GLenum access, GLenum format);
void EmulateBlendMinMax(GLenum equation, GLuint src_texture, GLuint dst_texture);
} // namespace GLFeatures

} // namespace OpenGL
