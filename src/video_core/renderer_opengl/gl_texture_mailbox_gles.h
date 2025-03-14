// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <glad/gl.h>
#include "video_core/renderer_opengl/gl_resource_manager.h"
#include "video_core/renderer_opengl/gl_texture_mailbox.h"
#include "common/common_types.h"

namespace OpenGL {

class OGLTextureMailboxGLES : public OGLTextureMailbox {
public:
    explicit OGLTextureMailboxGLES(bool debug, const Driver* driver) 
        : OGLTextureMailbox(debug, driver) {}

    Frontend::Frame* GetRenderFrame() override {
        if (!initialized) {
            Initialize();
        }
        return &swap_chain[render_index];
    }

    void ReleaseRenderFrame(Frontend::Frame* frame) override {
        render_index = (render_index + 1) % swap_chain.size();
    }

    Frontend::Frame* TryGetPresentFrame(int timeout_ms) override {
        if (!initialized) {
            return nullptr;
        }
        return &swap_chain[present_index];
    }

    void ReloadPresentFrame(Frontend::Frame* frame, u32 height, u32 width) override {
        frame->present.Release();
        frame->present.Create();
        glBindFramebuffer(GL_FRAMEBUFFER, frame->present.handle);

        // For GLES we use renderbuffer instead of texture for better performance
        GLuint color_buffer;
        glGenRenderbuffers(1, &color_buffer);
        glBindRenderbuffer(GL_RENDERBUFFER, color_buffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_buffer);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOG_CRITICAL(Render_OpenGL, "Failed to recreate present framebuffer!");
        }

        frame->color_reloaded = false;
        frame->width = width;
        frame->height = height;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

private:
    void Initialize() {
        if (initialized) {
            return;
        }
        for (auto& frame : swap_chain) {
            frame.render.Create();
            frame.present.Create();
        }
        initialized = true;
    }

    static constexpr std::size_t SWAP_CHAIN_SIZE = 3;
    std::array<Frontend::Frame, SWAP_CHAIN_SIZE> swap_chain{};
    std::size_t render_index = 0;
    std::size_t present_index = 0;
    bool initialized = false;
};

} // namespace OpenGL
