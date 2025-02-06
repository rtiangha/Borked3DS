// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <SDL3/SDL_video.h>
#include "borked3ds/emu_window/emu_window_sdl2.h"

struct SDL_Window;

namespace Core {
class System;
}

class EmuWindow_SDL2_GL : public EmuWindow_SDL2 {
public:
    explicit EmuWindow_SDL2_GL(Core::System& system_, bool fullscreen, bool is_secondary);
    ~EmuWindow_SDL2_GL();

    void Present() override;
    std::unique_ptr<Frontend::GraphicsContext> CreateSharedContext() const override;
    void MakeCurrent() override;
    void DoneCurrent() override;
    void SaveContext() override;
    void RestoreContext() override;

private:
    SDL_GLContext window_context;                            ///< Primary OpenGL context
    SDL_GLContext last_saved_context;                        ///< Context state for Save/Restore
    std::unique_ptr<Frontend::GraphicsContext> core_context; ///< Shared context for core operations
};
