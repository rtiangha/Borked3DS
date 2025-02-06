// Copyright 2016 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <string>
#include <vector>
#include <SDL3/SDL.h>
#include "audio_core/audio_types.h"
#include "audio_core/sdl2_sink.h"
#include "common/assert.h"
#include "common/logging/log.h"

namespace AudioCore {

struct SDL2Sink::Impl {
    SDL_AudioStream* stream = nullptr;
    unsigned int sample_rate = 0;
    std::function<void(s16*, std::size_t)> cb;

    static void Callback(void* userdata, SDL_AudioStream* stream, int additional_amount,
                         int total_amount);
};

SDL2Sink::SDL2Sink(std::string device_name) : impl(std::make_unique<Impl>()) {
    if (SDL_Init(SDL_INIT_AUDIO) != 0) {
        LOG_CRITICAL(Audio_Sink, "SDL_Init(SDL_INIT_AUDIO) failed with: {}", SDL_GetError());
        return;
    }

    SDL_AudioSpec spec;
    SDL_zero(spec);
    spec.format = SDL_AUDIO_S16LE;
    spec.channels = 2;
    spec.freq = native_sample_rate;

    SDL_AudioDeviceID devid = SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;

    if (device_name != auto_device_name && !device_name.empty()) {
        int count = 0;
        SDL_AudioDeviceID* devices = SDL_GetAudioPlaybackDevices(&count);
        for (int i = 0; i < count; ++i) {
            const char* name = SDL_GetAudioDeviceName(devices[i]);
            if (name && device_name == name) {
                devid = devices[i];
                break;
            }
        }
        SDL_free(devices);
    }

    impl->stream = SDL_OpenAudioDeviceStream(devid, &spec, &Impl::Callback, impl.get());
    if (!impl->stream) {
        LOG_CRITICAL(Audio_Sink, "SDL_OpenAudioDeviceStream failed for device \"{}\": {}",
                     device_name, SDL_GetError());
        return;
    }

    SDL_ResumeAudioStreamDevice(impl->stream);

    // Get the actual sample rate from the stream's output format
    SDL_AudioSpec dst_spec;
    if (SDL_GetAudioStreamFormat(impl->stream, nullptr, &dst_spec)) {
        impl->sample_rate = dst_spec.freq;
    } else {
        impl->sample_rate = native_sample_rate;
    }
}

SDL2Sink::~SDL2Sink() {
    if (impl->stream) {
        SDL_DestroyAudioStream(impl->stream);
    }
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
}

unsigned int SDL2Sink::GetNativeSampleRate() const {
    return impl->sample_rate;
}

void SDL2Sink::SetCallback(std::function<void(s16*, std::size_t)> cb) {
    impl->cb = cb;
}

void SDL2Sink::Impl::Callback(void* userdata, SDL_AudioStream* stream, int additional_amount,
                              int total_amount) {
    Impl* impl = static_cast<Impl*>(userdata);
    if (!impl || !impl->cb)
        return;

    const std::size_t num_samples = total_amount / sizeof(s16);
    const std::size_t num_frames = num_samples / 2; // stereo

    std::vector<s16> buffer(num_samples);
    impl->cb(buffer.data(), num_frames);

    // Explicit cast to int with range check
    const int buffer_size = static_cast<int>(num_samples * sizeof(s16));
    if (buffer_size > 0) {
        SDL_PutAudioStreamData(stream, buffer.data(), buffer_size);
    }
}

std::vector<std::string> ListSDL2SinkDevices() {
    if (SDL_InitSubSystem(SDL_INIT_AUDIO) != 0) {
        LOG_CRITICAL(Audio_Sink, "SDL_InitSubSystem failed with: {}", SDL_GetError());
        return {};
    }

    std::vector<std::string> device_list;
    int count = 0;
    SDL_AudioDeviceID* devices = SDL_GetAudioPlaybackDevices(&count);
    for (int i = 0; i < count; ++i) {
        const char* name = SDL_GetAudioDeviceName(devices[i]);
        if (name) {
            device_list.emplace_back(name);
        }
    }
    SDL_free(devices);

    SDL_QuitSubSystem(SDL_INIT_AUDIO);

    return device_list;
}

} // namespace AudioCore
