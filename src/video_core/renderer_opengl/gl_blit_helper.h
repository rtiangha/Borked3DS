// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/math_util.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/renderer_opengl/gl_compatibility.h"
#include "video_core/renderer_opengl/gl_resource_manager.h"
#include "video_core/renderer_opengl/gl_state.h"

namespace VideoCore {
struct Extent;
struct TextureBlit;
struct TextureCopy;
} // namespace VideoCore

namespace OpenGL {

class Driver;
class Surface;

class BlitHelper {
public:
    explicit BlitHelper(const Driver& driver);
    ~BlitHelper();

    bool Filter(Surface& surface, const VideoCore::TextureBlit& blit);

    bool ConvertDS24S8ToRGBA8(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy);

    bool ConvertRGBA4ToRGB5A1(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy);

    /// Blit source texture to draw framebuffer
    bool BlitTexture(GLuint src_tex, const Common::Rectangle<u32>& src_rect,
                     const Common::Rectangle<u32>& dst_rect, GLuint read_fb_handle,
                     GLuint draw_fb_handle, GLenum buffer, GLenum filter);

    /// Same as above but with a 3D texture source
    bool BlitTextures(const std::array<GLuint, 2>& src_textures,
                      const Common::Rectangle<u32>& src_rect,
                      const Common::Rectangle<u32>& dst_rect, GLuint read_fb_handle,
                      GLuint draw_fb_handle, GLenum buffer, GLenum filter);

    /// Blit depth/stencil texture to draw framebuffer using provided vertex shader
    bool BlitDepthStencil(GLuint src_tex, const Common::Rectangle<u32>& src_rect,
                          const Common::Rectangle<u32>& dst_rect,
                          GLuint read_fb_handle, GLuint draw_fb_handle);

private:
    void FilterAnime4K(Surface& surface, const VideoCore::TextureBlit& blit);
    void FilterBicubic(Surface& surface, const VideoCore::TextureBlit& blit);
    void FilterScaleForce(Surface& surface, const VideoCore::TextureBlit& blit);
    void FilterXbrz(Surface& surface, const VideoCore::TextureBlit& blit);
    void FilterMMPX(Surface& surface, const VideoCore::TextureBlit& blit);

    void SetParams(OGLProgram& program, const VideoCore::Extent& src_extent,
                   Common::Rectangle<u32> src_rect);
    void Draw(OGLProgram& program, GLuint dst_tex, GLuint dst_fbo, u32 dst_level,
              Common::Rectangle<u32> dst_rect);
    /// Setup vertices for blitting
    void SetupVertices(const Common::Rectangle<u32>& src_rect,
                       const Common::Rectangle<u32>& dst_rect);

    /// Build shader programs
    bool BuildProgramGLES();  // New: GLES specific shader program
    bool BuildProgramGL();    // Original GL shader program
    bool BuildDepthStencilProgram();
private:
    const Driver& driver;
    OGLVertexArray vao;
    OpenGLState state;
    OGLFramebuffer draw_fbo;
    OGLSampler linear_sampler;
    OGLSampler nearest_sampler;

    OGLProgram bicubic_program;
    OGLProgram scale_force_program;
    OGLProgram xbrz_program;
    OGLProgram mmpx_program;
    OGLProgram gradient_x_program;
    OGLProgram gradient_y_program;
    OGLProgram refine_program;
    OGLProgram d24s8_to_rgba8;
    OGLProgram rgba4_to_rgb5a1;

    OGLTexture temp_tex;
    VideoCore::Extent temp_extent{};
    bool use_texture_view{true};

    GLint blit_color_loc;
    OGLProgram program;
    OGLProgram depth_program;
    OGLBuffer vertex_buffer;
    OGLVertexArray vertex_array;
    GLint copy_vertices_loc;
    GLint positionLoc;
    GLint texCoordLoc;

    struct CommonUniforms {
        GLint src_tex;
        GLint src_tex_2;
        GLint sampler_2;
    } uniforms{};

    bool uses_gles_path{false};     // New: Track if using GLES path
    bool has_copy_image{false};     // New: Track if GL_EXT_copy_image is available
    bool initialized{false};
};

} // namespace OpenGL
