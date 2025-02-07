// Copyright 2018 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <SDL3/SDL.h>
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/math_util.h"
#include "common/param_package.h"
#include "common/threadsafe_queue.h"
#include "core/frontend/input.h"
#include "input_common/sdl/sdl_impl.h"

namespace InputCommon::SDL {
// Constructor implementation
SDLState::SDLState() {
    // Initialize SDL input subsystems
    SDL_InitSubSystem(SDL_INIT_GAMEPAD | SDL_INIT_JOYSTICK);
}

static std::string GetGUID(SDL_Joystick* joystick) {
    SDL_GUID guid = SDL_GetJoystickGUID(joystick);
    char guid_str[33];

    // Manually convert GUID bytes to hexadecimal string
    const Uint8* bytes = guid.data;
    for (int i = 0; i < 16; ++i) {
        snprintf(guid_str + (i * 2), 3, "%02x", bytes[i]);
    }
    guid_str[32] = '\0';

    return guid_str;
}

/// Creates a ParamPackage from an SDL_Event that can directly be used to create a ButtonDevice
std::string GetHatDirectionString(Uint8 hat_mask) {
    switch (hat_mask) {
    case SDL_HAT_UP:
        return "up";
    case SDL_HAT_DOWN:
        return "down";
    case SDL_HAT_LEFT:
        return "left";
    case SDL_HAT_RIGHT:
        return "right";
    default:
        return "unknown";
    }
}

static Common::ParamPackage SDLEventToButtonParamPackage(SDLState& state, const SDL_Event& event);

[[maybe_unused]] static int SDLEventWatcher(void* userdata, SDL_Event* event) {
    SDLState* sdl_state = reinterpret_cast<SDLState*>(userdata);
    if (sdl_state->polling) {
        sdl_state->event_queue.Push(*event);
    } else {
        sdl_state->HandleGamepadEvent(*event);
    }
    return 0;
}

constexpr std::array<SDL_GamepadButton, Settings::NativeButton::NumButtons> xinput_to_3ds_mapping =
    {{
        SDL_GAMEPAD_BUTTON_EAST,
        SDL_GAMEPAD_BUTTON_SOUTH,
        SDL_GAMEPAD_BUTTON_NORTH,
        SDL_GAMEPAD_BUTTON_WEST,
        SDL_GAMEPAD_BUTTON_DPAD_UP,
        SDL_GAMEPAD_BUTTON_DPAD_DOWN,
        SDL_GAMEPAD_BUTTON_DPAD_LEFT,
        SDL_GAMEPAD_BUTTON_DPAD_RIGHT,
        SDL_GAMEPAD_BUTTON_LEFT_SHOULDER,
        SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER,
        SDL_GAMEPAD_BUTTON_START,
        SDL_GAMEPAD_BUTTON_BACK,
        SDL_GAMEPAD_BUTTON_INVALID,
        SDL_GAMEPAD_BUTTON_INVALID,
        SDL_GAMEPAD_BUTTON_INVALID,
        SDL_GAMEPAD_BUTTON_INVALID,
        SDL_GAMEPAD_BUTTON_GUIDE,
        SDL_GAMEPAD_BUTTON_INVALID,
    }};

class SDLJoystick {
public:
    SDLJoystick(std::string guid_, int port_, SDL_Gamepad* gamepad)
        : guid{std::move(guid_)}, port{port_}, sdl_gamepad{gamepad, &SDL_CloseGamepad} {
        EnableMotion();
    }

    void EnableMotion() {
        if (!sdl_gamepad) {
            return;
        }
        if (HasMotion()) {
            SDL_SetGamepadSensorEnabled(sdl_gamepad.get(), SDL_SENSOR_ACCEL, false);
            SDL_SetGamepadSensorEnabled(sdl_gamepad.get(), SDL_SENSOR_GYRO, false);
        }
        has_accel = SDL_GamepadHasSensor(sdl_gamepad.get(), SDL_SENSOR_ACCEL);
        has_gyro = SDL_GamepadHasSensor(sdl_gamepad.get(), SDL_SENSOR_GYRO);
        if (has_accel) {
            SDL_SetGamepadSensorEnabled(sdl_gamepad.get(), SDL_SENSOR_ACCEL, true);
        }
        if (has_gyro) {
            SDL_SetGamepadSensorEnabled(sdl_gamepad.get(), SDL_SENSOR_GYRO, true);
        }
    }

    bool HasMotion() const {
        return has_gyro || has_accel;
    }

    void SetSDLGamepad(SDL_Gamepad* gamepad) {
        std::lock_guard lock{mutex};
        sdl_gamepad.reset(gamepad);
        EnableMotion();
    }

    void SetButton(int button, bool value) {
        std::lock_guard lock{mutex};
        state.buttons[button] = value;
    }

    bool GetButton(int button) const {
        std::lock_guard lock{mutex};
        return state.buttons.at(button);
    }

    void SetAxis(int axis, Sint16 value) {
        std::lock_guard lock{mutex};
        state.axes[axis] = value;
    }

    float GetAxis(int axis) const {
        std::lock_guard lock{mutex};
        return state.axes.at(axis) / 32767.0f;
    }

    void SetHat(int hat, Uint8 value) {
        std::lock_guard lock{mutex};
        state.hats[hat] = value;
    }

    bool GetHatDirection(int hat, Uint8 direction) const {
        std::lock_guard lock{mutex};
        return (state.hats.at(hat) & direction) != 0;
    }

    std::tuple<float, float> GetAnalog(int axis_x, int axis_y) const {
        float x = GetAxis(axis_x);
        float y = -GetAxis(axis_y); // Invert Y-axis for 3DS
        float r = std::sqrt(x * x + y * y);
        if (r > 1.0f) {
            x /= r;
            y /= r;
        }
        return {x, y};
    }

    void SetAccel(float x, float y, float z) {
        std::lock_guard lock{mutex};
        state.accel = {x, y, z};
    }

    void SetGyro(float pitch, float yaw, float roll) {
        std::lock_guard lock{mutex};
        state.gyro = {pitch, yaw, roll};
    }

    std::tuple<Common::Vec3<float>, Common::Vec3<float>> GetMotion() const {
        std::lock_guard lock{mutex};
        return {state.accel, state.gyro};
    }

    /**
     * The guid of the joystick
     */
    const std::string& GetGUID() const {
        return guid;
    }

    /**
     * The number of joystick from the same type that were connected before this joystick
     */
    int GetPort() const {
        return port;
    }
    SDL_Gamepad* GetSDLGamepad() const {
        return sdl_gamepad.get();
    }

private:
    struct State {
        std::unordered_map<int, Uint8> hats;
        std::unordered_map<int, bool> buttons;
        std::unordered_map<int, Sint16> axes;
        Common::Vec3<float> accel;
        Common::Vec3<float> gyro;
    } state;
    std::string guid;
    int port;
    bool has_gyro = false;
    bool has_accel = false;
    std::unique_ptr<SDL_Gamepad, decltype(&SDL_CloseGamepad)> sdl_gamepad;
    mutable std::mutex mutex;
};

class SDLAxisButton : public Input::ButtonDevice {
public:
    SDLAxisButton(std::shared_ptr<SDLJoystick> joystick_, int axis_, float threshold_,
                  bool positive_)
        : joystick(joystick_), axis(axis_), threshold(threshold_), positive(positive_) {}

    bool GetStatus() const override {
        const float value = joystick->GetAxis(axis);
        return positive ? (value > threshold) : (value < threshold);
    }

private:
    std::shared_ptr<SDLJoystick> joystick;
    int axis;
    float threshold;
    bool positive;
};

class SDLDirectionButton : public Input::ButtonDevice {
public:
    SDLDirectionButton(std::shared_ptr<SDLJoystick> joystick_, int hat_, Uint8 direction_)
        : joystick(joystick_), hat(hat_), direction(direction_) {}

    bool GetStatus() const override {
        return joystick->GetHatDirection(hat, direction);
    }

private:
    std::shared_ptr<SDLJoystick> joystick;
    int hat;
    Uint8 direction;
};

/**
 * Get the nth joystick with the corresponding GUID
 */
std::shared_ptr<SDLJoystick> SDLState::GetSDLJoystickByGUID(const std::string& guid, int port) {
    std::lock_guard lock{joystick_map_mutex};
    auto& list = joystick_map[guid];
    if (port >= static_cast<int>(list.size())) {
        list.resize(port + 1);
    }
    if (!list[port]) {
        list[port] = std::make_shared<SDLJoystick>(guid, port, nullptr);
    }
    return list[port];
}

void SDLState::InitGamepad(SDL_JoystickID instance_id) {
    if (!SDL_IsGamepad(instance_id)) {
        LOG_WARNING(Input, "Instance ID {} is not a gamepad", instance_id);
        return;
    }
    SDL_Gamepad* gamepad = SDL_OpenGamepad(instance_id);
    if (!gamepad) {
        LOG_ERROR(Input, "Failed to open gamepad {}: {}", instance_id, SDL_GetError());
        return;
    }

    SDL_Joystick* joystick = SDL_GetGamepadJoystick(gamepad);
    std::string guid = GetGUID(joystick);

    std::scoped_lock lock{joystick_map_mutex};
    auto& list = joystick_map[guid];
    for (auto& entry : list) {
        if (!entry->GetSDLGamepad()) {
            entry = std::make_shared<SDLJoystick>(guid, entry->GetPort(), gamepad);
            return;
        }
    }
    int port = static_cast<int>(list.size());
    list.push_back(std::make_shared<SDLJoystick>(guid, port, gamepad));
}

std::shared_ptr<SDLJoystick> SDLState::GetSDLJoystickBySDLID(SDL_JoystickID sdl_id) {
    std::scoped_lock lock{joystick_map_mutex};
    for (auto& [guid, list] : joystick_map) {
        for (auto& entry : list) {
            if (entry->GetSDLGamepad()) {
                SDL_Joystick* joystick = SDL_GetGamepadJoystick(entry->GetSDLGamepad());
                if (SDL_GetJoystickID(joystick) == sdl_id) {
                    return entry;
                }
            }
        }
    }
    return nullptr;
}

void SDLState::HandleGamepadEvent(const SDL_Event& event) {
    switch (event.type) {
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN: {
        auto joystick = GetSDLJoystickBySDLID(event.gbutton.which);
        if (joystick) {
            joystick->SetButton(event.gbutton.button, event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN);
        }
        break;
    }
    case SDL_EVENT_GAMEPAD_AXIS_MOTION: {
        auto joystick = GetSDLJoystickBySDLID(event.gaxis.which);
        if (joystick) {
            joystick->SetAxis(event.gaxis.axis, event.gaxis.value);
        }
        break;
    }
    case SDL_EVENT_GAMEPAD_SENSOR_UPDATE: {
        auto joystick = GetSDLJoystickBySDLID(event.gsensor.which);
        if (joystick) {
            switch (event.gsensor.sensor) {
            case SDL_SENSOR_ACCEL:
                // Convert from m/s² to G-force
                joystick->SetAccel(event.gsensor.data[0] / SDL_STANDARD_GRAVITY,
                                   -event.gsensor.data[1] / SDL_STANDARD_GRAVITY, // Y inversion
                                   event.gsensor.data[2] / SDL_STANDARD_GRAVITY);
                break;
            case SDL_SENSOR_GYRO:
                // Convert from rad/s to deg/s
                joystick->SetGyro(-event.gsensor.data[0] * (180.0f / Common::PI),  // Pitch
                                  event.gsensor.data[1] * (180.0f / Common::PI),   // Yaw
                                  -event.gsensor.data[2] * (180.0f / Common::PI)); // Roll
                break;
            }
        }
        break;
    }
    case SDL_EVENT_JOYSTICK_HAT_MOTION: {
        auto joystick = GetSDLJoystickBySDLID(event.jhat.which);
        if (joystick) {
            joystick->SetHat(event.jhat.hat, event.jhat.value);
        }
        break;
    }
    case SDL_EVENT_GAMEPAD_REMOVED:
        CloseGamepad(event.gdevice.which);
        break;
    case SDL_EVENT_GAMEPAD_ADDED:
        InitGamepad(event.gdevice.which);
        break;
    case SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_UP:
        // Currently not handled
        break;
    }
}

void SDLState::CloseGamepads() {
    std::scoped_lock lock{joystick_map_mutex};
    for (auto& [guid, list] : joystick_map) {
        for (auto& entry : list) {
            if (SDL_Gamepad* gamepad = entry->GetSDLGamepad()) {
                SDL_CloseGamepad(gamepad);
                entry->SetSDLGamepad(nullptr);
            }
        }
        list.clear();
    }
    joystick_map.clear();
}

void SDLState::CloseGamepad(SDL_JoystickID instance_id) {
    std::scoped_lock lock{joystick_map_mutex};
    for (auto& [guid, list] : joystick_map) {
        for (auto& entry : list) {
            if (entry->GetSDLGamepad() &&
                SDL_GetJoystickID(SDL_GetGamepadJoystick(entry->GetSDLGamepad())) == instance_id) {
                SDL_CloseGamepad(entry->GetSDLGamepad());
                entry->SetSDLGamepad(nullptr);
                return;
            }
        }
    }
}

// SDLButton implementation
class SDLButton final : public Input::ButtonDevice {
public:
    explicit SDLButton(std::shared_ptr<SDLJoystick> joystick_, SDL_GamepadButton button_)
        : joystick(std::move(joystick_)), button(button_) {}

    bool GetStatus() const override {
        if (SDL_Gamepad* gamepad = joystick->GetSDLGamepad()) {
            return SDL_GetGamepadButton(gamepad, button);
        }
        return joystick->GetButton(button);
    }

private:
    std::shared_ptr<SDLJoystick> joystick;
    SDL_GamepadButton button;
};

// SDLAnalog implementation
class SDLAnalog final : public Input::AnalogDevice {
public:
    SDLAnalog(std::shared_ptr<SDLJoystick> joystick_, SDL_GamepadAxis axis_x_,
              SDL_GamepadAxis axis_y_)
        : joystick(std::move(joystick_)), axis_x(axis_x_), axis_y(axis_y_) {}

    std::tuple<float, float> GetStatus() const override {
        if (SDL_Gamepad* gamepad = joystick->GetSDLGamepad()) {
            const float x = SDL_GetGamepadAxis(gamepad, axis_x) / 32767.0f;
            const float y = -SDL_GetGamepadAxis(gamepad, axis_y) / 32767.0f; // Invert Y-axis
            return {x, y};
        }
        return joystick->GetAnalog(axis_x, axis_y);
    }

private:
    std::shared_ptr<SDLJoystick> joystick;
    SDL_GamepadAxis axis_x;
    SDL_GamepadAxis axis_y;
};

Common::ParamPackage SDLState::GetSDLGamepadAnalogBindByGUID(
    const std::string& guid, int port, Settings::NativeAnalog::Values analog) {
    Common::ParamPackage params({{"engine", "sdl"}});
    return params;
}

Common::ParamPackage SDLState::GetSDLGamepadButtonBindByGUID(
    const std::string& guid, int port, Settings::NativeButton::Values button) {
    Common::ParamPackage params({{"engine", "sdl"}});
    auto joystick = GetSDLJoystickByGUID(guid, port);
    SDL_Gamepad* gamepad = joystick->GetSDLGamepad();

    if (!gamepad) {
        LOG_WARNING(Input, "Gamepad not connected: {}", guid);
        return params;
    }

    const auto mapped_button = xinput_to_3ds_mapping[static_cast<int>(button)];
    // Add validation
    if (mapped_button == SDL_GAMEPAD_BUTTON_INVALID) {
        return params;
    }

    int num_bindings = 0;
    SDL_GamepadBinding** bindings = SDL_GetGamepadBindings(gamepad, &num_bindings);
    SDL_GamepadBinding bind{};

    for (int i = 0; i < num_bindings; ++i) {
        if (bindings[i]->output_type == SDL_GAMEPAD_BINDTYPE_BUTTON &&
            bindings[i]->output.button == mapped_button) {
            bind = *bindings[i];
            break;
        }
    }

    SDL_free(bindings);

    switch (bind.input_type) {
    case SDL_GAMEPAD_BINDTYPE_BUTTON:
        params.Set("button", bind.input.button);
        break;
    case SDL_GAMEPAD_BINDTYPE_AXIS:
        params.Set("axis", bind.input.axis.axis);
        params.Set("threshold", 0.5f);
        params.Set("direction", bind.input.axis.axis_min < bind.input.axis.axis_max ? "+" : "-");
        break;
    case SDL_GAMEPAD_BINDTYPE_HAT:
        params.Set("hat", bind.input.hat.hat);
        params.Set("direction", GetHatDirectionString(bind.input.hat.hat_mask));
        break;
    default:
        break;
    }

    return params;
}

// Factory implementations
std::unique_ptr<Input::ButtonDevice> SDLButtonFactory::Create(const Common::ParamPackage& params) {
    const std::string guid = params.Get("guid", "");
    const int port = params.Get("port", 0);
    auto joystick = state.GetSDLJoystickByGUID(guid, port);

    // Add null checks
    if (!joystick || !joystick->GetSDLGamepad()) {
        throw std::runtime_error("Gamepad not available");
    }

    if (params.Has("button")) {
        const auto button = static_cast<SDL_GamepadButton>(params.Get("button", 0));
        return std::make_unique<SDLButton>(joystick, button);
    }

    if (params.Has("axis")) {
        const int axis = params.Get("axis", 0);
        const float threshold = params.Get("threshold", 0.5f);
        const std::string direction = params.Get("direction", "+");
        return std::make_unique<SDLAxisButton>(joystick, axis, threshold, direction == "+");
    }

    if (params.Has("hat")) {
        const int hat = params.Get("hat", 0);
        const std::string direction_str = params.Get("direction", "up");
        Uint8 direction = SDL_HAT_CENTERED;

        if (direction_str == "up")
            direction = SDL_HAT_UP;
        else if (direction_str == "down")
            direction = SDL_HAT_DOWN;
        else if (direction_str == "left")
            direction = SDL_HAT_LEFT;
        else if (direction_str == "right")
            direction = SDL_HAT_RIGHT;

        return std::make_unique<SDLDirectionButton>(joystick, hat, direction);
    }

    throw std::invalid_argument("Invalid button configuration");
}

// Motion sensor implementation
class SDLMotion final : public Input::MotionDevice {
public:
    explicit SDLMotion(std::shared_ptr<SDLJoystick> joystick_) : joystick(std::move(joystick_)) {}

    std::tuple<Common::Vec3<float>, Common::Vec3<float>> GetStatus() const override {
        return joystick->GetMotion();
    }

private:
    std::shared_ptr<SDLJoystick> joystick;
};

// Event polling updates
[[maybe_unused]] Common::ParamPackage SDLEventToButtonParamPackage(SDLState& state,
                                                                   const SDL_Event& event) {
    Common::ParamPackage params;

    switch (event.type) {
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_BUTTON_UP: {
        auto joystick = state.GetSDLJoystickBySDLID(event.gbutton.which);
        if (!joystick) {
            return {};
        }
        params.Set("engine", "sdl");
        params.Set("guid", joystick->GetGUID());
        params.Set("port", joystick->GetPort());
        params.Set("button", event.gbutton.button);
        break;
    }
    case SDL_EVENT_GAMEPAD_AXIS_MOTION: {
        auto joystick = state.GetSDLJoystickBySDLID(event.gaxis.which);
        if (!joystick) {
            return {};
        }

        params.Set("engine", "sdl");
        params.Set("guid", joystick->GetGUID());
        params.Set("port", joystick->GetPort());
        params.Set("axis", event.gaxis.axis);

        // Automatic axis direction detection
        const float threshold = 0.5f;
        const bool positive = event.gaxis.value > 0;
        params.Set("threshold", positive ? threshold : -threshold);
        params.Set("direction", positive ? "+" : "-");
        break;
    }
    default:
        return {};
    }

    return params;
} // namespace SDL
} // namespace InputCommon::SDL
