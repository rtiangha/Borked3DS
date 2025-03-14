// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/logging/log.h"
#include "video_core/renderer_opengl/gl_driver.h"
#include "video_core/renderer_opengl/gl_state.h"
#include "video_core/renderer_opengl/gl_texture_mailbox.h"

namespace OpenGL {

OGLTextureMailbox::OGLTextureMailbox(bool has_debug_tool_, const Driver* driver)
    : has_debug_tool{has_debug_tool_}, is_gles{driver ? driver->IsOpenGLES() : false} {
    for (auto& frame : swap_chain) {
        frame.is_gles = is_gles;
        free_queue.push(&frame);
    }
}

OGLTextureMailbox::~OGLTextureMailbox() {
    // Lock the mutex and clear out the present and free_queues and notify any people who are
    // blocked to prevent deadlock on shutdown
    std::scoped_lock lock(swap_chain_lock);

    // Clean up GLES textures
    if (is_gles) {
        for (auto& frame : swap_chain) {
            DeleteGLESTexture(&frame);
        }
    }

    free_queue = {};
    present_queue.clear();
    present_cv.notify_all();
    free_cv.notify_all();
}

bool OGLTextureMailbox::SupportsRenderbufferSharing() const {
    if (!is_gles)
        return true;

    return false; // For now, always use textures for GLES
}

void OGLTextureMailbox::CreateGLESTexture(Frontend::Frame* frame, u32 width, u32 height) {
    if (frame->texture != 0) {
        glDeleteTextures(1, &frame->texture);
    }

    glGenTextures(1, &frame->texture);
    glBindTexture(GL_TEXTURE_2D, frame->texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void OGLTextureMailbox::DeleteGLESTexture(Frontend::Frame* frame) {
    if (frame->texture != 0) {
        glDeleteTextures(1, &frame->texture);
        frame->texture = 0;
    }
}

void OGLTextureMailbox::ReloadPresentFrame(Frontend::Frame* frame, u32 height, u32 width) {
    frame->present.Release();
    frame->present.Create();
    GLint previous_draw_fbo{};
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previous_draw_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, frame->present.handle);

    if (is_gles) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frame->texture,
                               0);
    } else {
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
                                  frame->color.handle);
    }

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        LOG_CRITICAL(Render_OpenGL, "Failed to recreate present FBO!");
    }

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, previous_draw_fbo);
    frame->color_reloaded = false;
}

void OGLTextureMailbox::ReloadRenderFrame(Frontend::Frame* frame, u32 width, u32 height) {
    OpenGLState prev_state = OpenGLState::GetCurState();
    OpenGLState state = OpenGLState::GetCurState();

    // For GLES, always use textures
    if (is_gles) {
        // Delete old texture if it exists
        if (frame->texture != 0) {
            glDeleteTextures(1, &frame->texture);
        }

        // Create new texture
        glGenTextures(1, &frame->texture);
        glBindTexture(GL_TEXTURE_2D, frame->texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Create and setup FBO
        frame->render.Release();
        frame->render.Create();
        state.draw.read_framebuffer = frame->render.handle;
        state.draw.draw_framebuffer = frame->render.handle;
        state.Apply();

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, frame->texture,
                               0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOG_CRITICAL(Render_OpenGL, "Failed to recreate render FBO!");
        }
    } else {
        // Original desktop GL path
        frame->color.Release();
        frame->color.Create();
        state.renderbuffer = frame->color.handle;
        state.Apply();
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);

        frame->render.Release();
        frame->render.Create();
        state.draw.read_framebuffer = frame->render.handle;
        state.draw.draw_framebuffer = frame->render.handle;
        state.Apply();

        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
                                  frame->color.handle);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            LOG_CRITICAL(Render_OpenGL, "Failed to recreate render FBO!");
        }
    }

    prev_state.Apply();
    frame->width = width;
    frame->height = height;
    frame->color_reloaded = true;
}

Frontend::Frame* OGLTextureMailbox::GetRenderFrame() {
    std::unique_lock lock{swap_chain_lock};

    // If theres no free frames, we will reuse the oldest render frame
    if (free_queue.empty()) {
        auto frame = present_queue.back();
        present_queue.pop_back();
        return frame;
    }

    Frontend::Frame* frame = free_queue.front();
    free_queue.pop();
    return frame;
}

void OGLTextureMailbox::ReleaseRenderFrame(Frontend::Frame* frame) {
    std::unique_lock lock{swap_chain_lock};
    present_queue.push_front(frame);
    present_cv.notify_one();

    DebugNotifyNextFrame();
}

void OGLTextureMailbox::LoadPresentFrame() {
    // Free the previous frame and add it back to the free queue
    if (previous_frame) {
        free_queue.push(previous_frame);
        free_cv.notify_one();
    }

    // The newest entries are pushed to the front of the queue
    Frontend::Frame* frame = present_queue.front();
    present_queue.pop_front();
    // Remove all old entries from the present queue and move them back to the free_queue
    for (auto f : present_queue) {
        free_queue.push(f);
    }
    present_queue.clear();
    previous_frame = frame;
}

Frontend::Frame* OGLTextureMailbox::TryGetPresentFrame(int timeout_ms) {
    DebugWaitForNextFrame();

    std::unique_lock lock{swap_chain_lock};
    // Wait for new entries in the present_queue
    present_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [&] { return !present_queue.empty(); });
    if (present_queue.empty()) {
        // Timed out waiting for a frame to draw so return the previous frame
        return previous_frame;
    }

    LoadPresentFrame();
    return previous_frame;
}

void OGLTextureMailbox::DebugNotifyNextFrame() {
    if (!has_debug_tool) {
        return;
    }
    frame_for_debug++;
    std::scoped_lock lock{debug_synch_mutex};
    debug_synch_condition.notify_one();
}

void OGLTextureMailbox::DebugWaitForNextFrame() {
    if (!has_debug_tool) {
        return;
    }
    const int last_frame = frame_for_debug;
    std::unique_lock lock{debug_synch_mutex};
    debug_synch_condition.wait(lock, [this, last_frame] { return frame_for_debug > last_frame; });
}

Frontend::Frame* OGLVideoDumpingMailbox::GetRenderFrame() {
    std::unique_lock lock{swap_chain_lock};

    // If theres no free frames, we will wait until one shows up
    if (free_queue.empty()) {
        free_cv.wait(lock, [&] { return (!free_queue.empty() || quit); });
        if (quit) {
            throw OGLTextureMailboxException("VideoDumpingMailbox quitting");
        }

        if (free_queue.empty()) {
            LOG_CRITICAL(Render_OpenGL, "Could not get free frame");
            return nullptr;
        }
    }

    Frontend::Frame* frame = free_queue.front();
    free_queue.pop();
    return frame;
}

void OGLVideoDumpingMailbox::LoadPresentFrame() {
    // Free the previous frame and add it back to the free queue
    if (previous_frame) {
        free_queue.push(previous_frame);
        free_cv.notify_one();
    }

    Frontend::Frame* frame = present_queue.back();
    present_queue.pop_back();
    previous_frame = frame;

    // Do not remove entries from the present_queue, as video dumping would require
    // that we preserve all frames
}

Frontend::Frame* OGLVideoDumpingMailbox::TryGetPresentFrame(int timeout_ms) {
    std::unique_lock lock{swap_chain_lock};
    // Wait for new entries in the present_queue
    present_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [&] { return !present_queue.empty(); });
    if (present_queue.empty()) {
        // Timed out waiting for a frame
        return nullptr;
    }

    LoadPresentFrame();
    return previous_frame;
}

} // namespace OpenGL
