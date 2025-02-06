// Copyright 2016 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <cstdlib>
#include <string>
#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include "borked3ds/emu_window/emu_window_sdl2.h"
#include "common/logging/log.h"
#include "common/scm_rev.h"
#include "core/core.h"
#include "input_common/keyboard.h"
#include "input_common/main.h"
#include "input_common/motion_emu.h"
#include "network/network.h"

void EmuWindow_SDL2::OnMouseMotion(s32 x, s32 y) {
    TouchMoved((unsigned)std::max(x, 0), (unsigned)std::max(y, 0));
    InputCommon::GetMotionEmu()->Tilt(x, y);
}

void EmuWindow_SDL2::OnMouseButton(u32 button, u8 state, s32 x, s32 y) {
    if (button == SDL_BUTTON_LEFT) {
        if (state) { // SDL_EVENT_PRESSED is 1 in SDL3
            TouchPressed((unsigned)std::max(x, 0), (unsigned)std::max(y, 0));
        } else {
            TouchReleased();
        }
    } else if (button == SDL_BUTTON_RIGHT) {
        if (state) {
            InputCommon::GetMotionEmu()->BeginTilt(x, y);
        } else {
            InputCommon::GetMotionEmu()->EndTilt();
        }
    }
}

std::pair<unsigned, unsigned> EmuWindow_SDL2::TouchToPixelPos(float touch_x, float touch_y) const {
    int w, h;
    SDL_GetWindowSize(render_window, &w, &h);

    touch_x *= w;
    touch_y *= h;

    return {static_cast<unsigned>(std::max(std::round(touch_x), 0.0f)),
            static_cast<unsigned>(std::max(std::round(touch_y), 0.0f))};
}

void EmuWindow_SDL2::OnFingerDown(float x, float y) {
    const auto [px, py] = TouchToPixelPos(x, y);
    TouchPressed(px, py);
}

void EmuWindow_SDL2::OnFingerMotion(float x, float y) {
    const auto [px, py] = TouchToPixelPos(x, y);
    TouchMoved(px, py);
}

void EmuWindow_SDL2::OnFingerUp() {
    TouchReleased();
}

void EmuWindow_SDL2::OnKeyEvent(int key, u8 state) {
    if (state) {
        InputCommon::GetKeyboard()->PressKey(key);
    } else {
        InputCommon::GetKeyboard()->ReleaseKey(key);
    }
}

bool EmuWindow_SDL2::IsOpen() const {
    return is_open;
}

void EmuWindow_SDL2::RequestClose() {
    is_open = false;
}

void EmuWindow_SDL2::OnResize() {
    int width, height;
    SDL_GetWindowSizeInPixels(render_window, &width, &height);
    UpdateCurrentFramebufferLayout(width, height);
}

void EmuWindow_SDL2::Fullscreen() {
    // Attempt borderless fullscreen first
    if (SDL_SetWindowFullscreenMode(render_window, nullptr) &&
        SDL_SetWindowFullscreen(render_window, true)) {
        return;
    }

    LOG_ERROR(Frontend, "Fullscreening failed: {}", SDL_GetError());

    // Fallback to maximize window
    SDL_MaximizeWindow(render_window);
}

EmuWindow_SDL2::EmuWindow_SDL2(Core::System& system_, bool is_secondary)
    : EmuWindow(is_secondary), system(system_) {}

EmuWindow_SDL2::~EmuWindow_SDL2() {
    SDL_Quit();
}

void EmuWindow_SDL2::InitializeSDL2() {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD) != 0) {
        LOG_CRITICAL(Frontend, "Failed to initialize SDL: {}! Exiting...", SDL_GetError());
        exit(1);
    }

    InputCommon::Init();
    Network::Init();
}

u32 EmuWindow_SDL2::GetEventWindowId(const SDL_Event& event) const {
    switch (event.type) {
    case SDL_EVENT_WINDOW_RESIZED:
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
    case SDL_EVENT_WINDOW_MAXIMIZED:
    case SDL_EVENT_WINDOW_RESTORED:
    case SDL_EVENT_WINDOW_MINIMIZED:
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
        return event.window.windowID;
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
        return event.key.windowID;
    case SDL_EVENT_MOUSE_MOTION:
        return event.motion.windowID;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
        return event.button.windowID;
    case SDL_EVENT_MOUSE_WHEEL:
        return event.wheel.windowID;
    case SDL_EVENT_FINGER_DOWN:
    case SDL_EVENT_FINGER_MOTION:
    case SDL_EVENT_FINGER_UP:
        return event.tfinger.windowID;
    case SDL_EVENT_TEXT_EDITING:
        return event.edit.windowID;
    case SDL_EVENT_TEXT_INPUT:
        return event.text.windowID;
    case SDL_EVENT_DROP_BEGIN:
    case SDL_EVENT_DROP_FILE:
    case SDL_EVENT_DROP_TEXT:
    case SDL_EVENT_DROP_COMPLETE:
        return event.drop.windowID;
    default:
        return render_window_id;
    }
}

void EmuWindow_SDL2::PollEvents() {
    SDL_Event event;
    std::vector<SDL_Event> other_window_events;

    while (SDL_PollEvent(&event)) {
        if (GetEventWindowId(event) != render_window_id) {
            other_window_events.push_back(event);
            continue;
        }

        switch (event.type) {
        case SDL_EVENT_WINDOW_RESIZED:
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_MAXIMIZED:
        case SDL_EVENT_WINDOW_RESTORED:
        case SDL_EVENT_WINDOW_MINIMIZED:
            OnResize();
            break;
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            RequestClose();
            break;
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP:
            OnKeyEvent(event.key.scancode, // Direct scancode access
                       (event.type == SDL_EVENT_KEY_DOWN) ? 1 : 0);
            break;
        case SDL_EVENT_MOUSE_MOTION:
            if (event.motion.which != SDL_TOUCH_MOUSEID)
                OnMouseMotion(event.motion.x, event.motion.y);
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (event.button.which != SDL_TOUCH_MOUSEID) {
                OnMouseButton(event.button.button,
                              event.type == SDL_EVENT_MOUSE_BUTTON_DOWN ? 1 : 0, // Use event type
                              event.button.x, event.button.y);
            }
            break;
        case SDL_EVENT_FINGER_DOWN:
            OnFingerDown(event.tfinger.x, event.tfinger.y);
            break;
        case SDL_EVENT_FINGER_MOTION:
            OnFingerMotion(event.tfinger.x, event.tfinger.y);
            break;
        case SDL_EVENT_FINGER_UP:
            OnFingerUp();
            break;
        case SDL_EVENT_QUIT:
            RequestClose();
            break;
        default:
            break;
        }
    }

    for (auto& e : other_window_events) {
        SDL_PushEvent(&e);
    }

    if (!is_secondary) {
        UpdateFramerateCounter();
    }
}

void EmuWindow_SDL2::OnMinimalClientAreaChangeRequest(std::pair<u32, u32> minimal_size) {
    SDL_SetWindowMinimumSize(render_window, minimal_size.first, minimal_size.second);
}

void EmuWindow_SDL2::UpdateFramerateCounter() {
    const u32 current_time = SDL_GetTicks();
    if (current_time > last_time + 2000) {
        const auto results = system.GetAndResetPerfStats();
        const auto title =
            fmt::format("Borked3DS {} | {}-{} | FPS: {:.0f} ({:.0f}%)", Common::g_build_fullname,
                        Common::g_scm_branch, Common::g_scm_desc, results.game_fps,
                        results.emulation_speed * 100.0f);
        SDL_SetWindowTitle(render_window, title.c_str());
        last_time = current_time;
    }
}
