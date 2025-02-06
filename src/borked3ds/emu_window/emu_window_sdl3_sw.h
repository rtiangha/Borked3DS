// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include "borked3ds/emu_window/emu_window_sdl3.h"

struct SDL_Renderer;
struct SDL_Surface;

namespace VideoCore {
enum class ScreenId : u32;
}

namespace Core {
class System;
}

class EmuWindow_SDL3_SW final : public EmuWindow_SDL3 {
public:
    explicit EmuWindow_SDL3_SW(Core::System& system, bool fullscreen, bool is_secondary);
    ~EmuWindow_SDL3_SW();

    void Present() override;
    std::unique_ptr<Frontend::GraphicsContext> CreateSharedContext() const override;
    void MakeCurrent() override {}
    void DoneCurrent() override {}

private:
    /// Loads a framebuffer to an SDL surface
    SDL_Surface* LoadFramebuffer(VideoCore::ScreenId screen_id);

    /// The system class.
    Core::System& system;

    /// The SDL software renderer
    SDL_Renderer* renderer;

    /// The window surface
    SDL_Surface* window_surface;
};
