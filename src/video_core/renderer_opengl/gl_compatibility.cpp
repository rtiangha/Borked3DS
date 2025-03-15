// Copyright 2025 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/logging/log.h"
#include "video_core/renderer_opengl/gl_compatibility.h"
#include "video_core/renderer_opengl/gl_shader_util.h"

namespace OpenGL {

bool GLCompatibility::has_texture_buffer = false;
bool GLCompatibility::has_image_atomic = false;
bool GLCompatibility::has_blend_minmax = false;
bool GLCompatibility::has_clip_distance = false;
bool GLCompatibility::has_geometry_shader = false;
bool GLCompatibility::is_gles = false;

bool GLCompatibility::Initialize() {
    is_gles = GLAD_GL_ES_VERSION_3_0;

    // Check for required extensions
    has_texture_buffer = GLAD_GL_EXT_texture_buffer;
    has_image_atomic = GLAD_GL_OES_shader_image_atomic;
    has_blend_minmax = GLAD_GL_EXT_blend_minmax;
    has_clip_distance = GLAD_GL_EXT_clip_cull_distance;
    has_geometry_shader = GLAD_GL_EXT_geometry_shader;

    LOG_INFO(Render_OpenGL, "OpenGL Compatibility Layer Initialized:");
    LOG_INFO(Render_OpenGL, "  GLES Mode: {}", is_gles ? "Yes" : "No");
    LOG_INFO(Render_OpenGL, "  Texture Buffer: {}", has_texture_buffer ? "Hardware" : "Emulated");
    LOG_INFO(Render_OpenGL, "  Image Atomic: {}", has_image_atomic ? "Hardware" : "Emulated");
    LOG_INFO(Render_OpenGL, "  Blend Min/Max: {}", has_blend_minmax ? "Hardware" : "Emulated");
    LOG_INFO(Render_OpenGL, "  Clip Distance: {}", has_clip_distance ? "Supported" : "Unsupported");
    LOG_INFO(Render_OpenGL, "  Geometry Shader: {}",
             has_geometry_shader ? "Supported" : "Unsupported");

    return true;
}

namespace GLFeatures {

// Texture Buffer emulation using 2D texture array
void EmulateTextureBuffer(GLuint& texture, GLuint& buffer, GLenum format) {
    if (GLCompatibility::HasTextureBuffer()) {
        glBindTexture(GL_TEXTURE_BUFFER_EXT, texture);
        glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, format, buffer);
        return;
    }

    // Emulation using 2D texture array
    GLint buffer_size;
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &buffer_size);

    // Calculate dimensions for 2D texture array
    const GLsizei width = 1024; // Adjust based on your needs
    const GLsizei height = (buffer_size + width - 1) / width;

    glBindTexture(GL_TEXTURE_2D_ARRAY, texture);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, format, width, height, 1, 0, format, GL_UNSIGNED_BYTE,
                 nullptr);

    // Copy buffer data to texture
    GLvoid* data = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, width, height, 1, format, GL_UNSIGNED_BYTE,
                    data);
    glUnmapBuffer(GL_ARRAY_BUFFER);
}

// Image atomic emulation using regular textures and custom shaders
void EmulateImageAtomic(GLuint unit, GLuint texture, GLenum access, GLenum format) {
    if (GLCompatibility::HasImageAtomic()) {
        glBindImageTexture(unit, texture, 0, GL_FALSE, 0, access, format);
        return;
    }

    // Fallback to regular texture binding
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, texture);
}

// Blend min/max emulation using custom shader
void EmulateBlendMinMax(GLenum equation, GLuint src_texture, GLuint dst_texture) {
    if (GLCompatibility::HasBlendMinMax()) {
        glBlendEquation(equation);
        return;
    }

    static const GLchar* blend_vs = R"(#version 300 es
        precision mediump float;
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 texcoord;
        out vec2 v_texcoord;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            v_texcoord = texcoord;
        }
    )";

    static const GLchar* blend_fs = R"(#version 300 es
        precision mediump float;
        uniform sampler2D src_tex;
        uniform sampler2D dst_tex;
        uniform bool is_min;
        in vec2 v_texcoord;
        out vec4 frag_color;
        void main() {
            vec4 src = texture(src_tex, v_texcoord);
            vec4 dst = texture(dst_tex, v_texcoord);
            frag_color = is_min ? min(src, dst) : max(src, dst);
        }
    )";

    static GLuint blend_program = 0;
    if (!blend_program) {
        blend_program = CreateProgram(blend_vs, blend_fs);
    }

    glUseProgram(blend_program);
    glUniform1i(glGetUniformLocation(blend_program, "src_tex"), 0);
    glUniform1i(glGetUniformLocation(blend_program, "dst_tex"), 1);
    glUniform1i(glGetUniformLocation(blend_program, "is_min"), equation == GL_MIN ? 1 : 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, src_texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, dst_texture);
}

} // namespace GLFeatures
} // namespace OpenGL
