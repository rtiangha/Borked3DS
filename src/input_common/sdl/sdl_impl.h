// Copyright 2018 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>
#include "common/settings.h"
#include "common/threadsafe_queue.h"
#include "input_common/sdl/sdl.h"

union SDL_Event;
using SDL_Joystick = struct SDL_Joystick;
using SDL_JoystickID = u32;
using SDL_Gamepad = struct SDL_Gamepad;

namespace InputCommon::SDL {

class SDLJoystick;
class SDLAnalogFactory;
class SDLMotionFactory;
class SDLState;

class SDLButtonFactory : public Input::Factory<Input::ButtonDevice> {
public:
    explicit SDLButtonFactory(SDLState& state_);
    std::unique_ptr<Input::ButtonDevice> Create(const Common::ParamPackage& params) override;

private:
    SDLState& state;
};

class SDLState : public State {
public:
    /// Initializes and registers SDL device factories
    SDLState();

    /// Unregisters SDL device factories and shut them down.
    ~SDLState() override;

    Pollers GetPollers(Polling::DeviceType type) override;

    /// Handle SDL_Events for joysticks from SDL_PollEvent
    void HandleGamepadEvent(const SDL_Event& event);

    std::shared_ptr<SDLJoystick> GetSDLJoystickBySDLID(SDL_JoystickID sdl_id);
    std::shared_ptr<SDLJoystick> GetSDLJoystickByGUID(const std::string& guid, int port);

    Common::ParamPackage GetSDLGamepadButtonBindByGUID(const std::string& guid, int port,
                                                       Settings::NativeButton::Values button);
    Common::ParamPackage GetSDLGamepadAnalogBindByGUID(const std::string& guid, int port,
                                                       Settings::NativeAnalog::Values analog);

    /// Get all DevicePoller that use the SDL backend for a specific device type
    Pollers GetPollers(Polling::DeviceType type) override;

    /// Used by the Pollers during config
    std::atomic<bool> polling = false;
    Common::SPSCQueue<SDL_Event> event_queue;

private:
    void InitGamepad(SDL_JoystickID instance_id);
    void CloseGamepad(SDL_JoystickID instance_id);

    /// Needs to be called before SDL_QuitSubSystem.
    void CloseGamepads();

    /// Map of GUID of a list of corresponding virtual Joysticks
    std::unordered_map<std::string, std::vector<std::shared_ptr<SDLJoystick>>> joystick_map;
    std::mutex joystick_map_mutex;

    std::shared_ptr<SDLButtonFactory> button_factory;
    std::shared_ptr<SDLAnalogFactory> analog_factory;
    std::shared_ptr<SDLMotionFactory> motion_factory;

    bool start_thread = false;
    std::atomic<bool> initialized = false;

    std::thread poll_thread;
};
} // namespace InputCommon::SDL
