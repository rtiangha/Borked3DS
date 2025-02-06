// Copyright 2016 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <utility>
#include <SDL3/SDL.h>
#include "common/common_types.h"
#include "core/frontend/emu_window.h"

union SDL_Event;
struct SDL_Window;

namespace Core {
class System;
}

class EmuWindow_SDL2 : public Frontend::EmuWindow {
public:
    explicit EmuWindow_SDL2(Core::System& system_, bool is_secondary);
    ~EmuWindow_SDL2();
    virtual void Present() {}

    static void InitializeSDL2();

    void PollEvents() override;

    bool IsOpen() const;
    void RequestClose();

protected:
    u32 GetEventWindowId(const SDL_Event& event) const;
    void OnKeyEvent(int key, u8 state);
    void OnMouseMotion(s32 x, s32 y);
    void OnMouseButton(u32 button, u8 state, s32 x, s32 y);
    std::pair<unsigned, unsigned> TouchToPixelPos(float touch_x, float touch_y) const;
    void OnFingerDown(float x, float y);
    void OnFingerMotion(float x, float y);
    void OnFingerUp();
    void OnResize();
    void Fullscreen();
    void OnMinimalClientAreaChangeRequest(std::pair<u32, u32> minimal_size) override;
    void UpdateFramerateCounter();

    bool is_open = true;
    SDL_Window* render_window;
    u32 render_window_id{};
    SDL_Window* dummy_window;
    u32 last_time = 0;
    Core::System& system;
};
