// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstdlib>
#include <memory>
#include <string>
#include <SDL3/SDL.h>
#include <fmt/format.h>
#include "borked3ds/emu_window/emu_window_sdl3_vk.h"
#include "common/logging/log.h"
#include "common/scm_rev.h"
#include "core/frontend/emu_window.h"

class DummyContext : public Frontend::GraphicsContext {};

EmuWindow_SDL3_VK::EmuWindow_SDL3_VK(Core::System& system, bool fullscreen, bool is_secondary)
    : EmuWindow_SDL3{system, is_secondary} {
    const std::string window_title = fmt::format("Borked3DS {} | {}-{}", Common::g_build_fullname,
                                                 Common::g_scm_branch, Common::g_scm_desc);
    render_window =
        SDL_CreateWindow(window_title.c_str(), Core::kScreenTopWidth,
                         Core::kScreenTopHeight + Core::kScreenBottomHeight,
                         SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);

    if (!render_window) {
        LOG_CRITICAL(Frontend, "Failed to create SDL window: {}", SDL_GetError());
        exit(EXIT_FAILURE);
    }

    SDL_PropertiesID props = SDL_GetWindowProperties(render_window);

    // Windows
    void* hwnd = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    if (hwnd) {
        window_info.type = Frontend::WindowSystemType::Windows;
        window_info.render_surface = hwnd;
    }
    // X11
    else if (SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0)) {
        window_info.type = Frontend::WindowSystemType::X11;
        window_info.display_connection =
            SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr);
        window_info.render_surface = reinterpret_cast<void*>(
            SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0));
    }
    // Wayland
    else if (SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr)) {
        window_info.type = Frontend::WindowSystemType::Wayland;
        window_info.display_connection =
            SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, nullptr);
        window_info.render_surface =
            SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr);
    }
    // macOS (Metal)
    else if (SDL_GetNumberProperty(props, SDL_PROP_WINDOW_COCOA_METAL_VIEW_TAG_NUMBER, 0)) {
        window_info.type = Frontend::WindowSystemType::MacOS;
        window_info.render_surface = reinterpret_cast<void*>(
            SDL_GetNumberProperty(props, SDL_PROP_WINDOW_COCOA_METAL_VIEW_TAG_NUMBER, 0));
    }
    // Android
    else if (SDL_GetPointerProperty(props, SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, nullptr)) {
        window_info.type = Frontend::WindowSystemType::Android;
        window_info.render_surface =
            SDL_GetPointerProperty(props, SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, nullptr);
    } else {
        LOG_CRITICAL(Frontend, "Unsupported window manager subsystem");
        exit(EXIT_FAILURE);
    }

    if (fullscreen) {
        Fullscreen();
        SDL_HideCursor();
    }

    render_window_id = SDL_GetWindowID(render_window);

    OnResize();
    OnMinimalClientAreaChangeRequest(GetActiveConfig().min_client_area_size);
    SDL_PumpEvents();
}

EmuWindow_SDL3_VK::~EmuWindow_SDL3_VK() = default;

std::unique_ptr<Frontend::GraphicsContext> EmuWindow_SDL3_VK::CreateSharedContext() const {
    return std::make_unique<DummyContext>();
}
