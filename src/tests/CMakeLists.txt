add_executable(tests
    common/bit_field.cpp
    common/file_util.cpp
    common/param_package.cpp
    core/core_timing.cpp
    core/file_sys/path_parser.cpp
    core/hle/kernel/hle_ipc.cpp
    core/memory/memory.cpp
    core/memory/vm_manager.cpp
    precompiled_headers.h
    audio_core/hle/hle.cpp
    audio_core/hle/source.cpp
    audio_core/lle/lle.cpp
    audio_core/audio_fixures.h
    audio_core/decoder_tests.cpp
    video_core/pica_float.cpp
    video_core/shader.cpp
    audio_core/merryhime_3ds_audio/merry_audio/merry_audio.cpp
    audio_core/merryhime_3ds_audio/merry_audio/merry_audio.h
    audio_core/merryhime_3ds_audio/merry_audio/service_fixture.cpp
    audio_core/merryhime_3ds_audio/merry_audio/service_fixture.h
    audio_core/merryhime_3ds_audio/audio_test_biquad_filter.cpp
)

if (MSVC AND ENABLE_LTO)
  target_compile_options(tests PRIVATE
    /wd5049 # 'string': Embedding a full path may result in machine-dependent output (breaks LTO on MSVC)
  )
endif()

create_target_directory_groups(tests)

# Add dependency on scm_rev.cpp generation
add_dependencies(tests borked3ds_common)

# Fix linking order with repeated libraries to handle circular dependencies
target_link_libraries(tests PRIVATE
    borked3ds_core
    video_core
    audio_core
    input_common
    network
    borked3ds_common
    web_service
    cryptopp
    lodepng
    network # Repeat for circular dependencies
    web_service
    enet
    ssl
    crypto
    dynarmic
    mcl
    glad
    sirit
    SPIRV
    glslang
    SPIRV-Tools-opt
    SPIRV-Tools
    faad2
    SoundTouch
    teakra
    cubeb
    openal
    dl
    atomic
    borked3ds_common # Repeat for circular dependencies
    gamemode
    fmt
    boost_serialization
    boost_iostreams
    zstd
    tracy
    SDL2::SDL2
    libusb
    catch2
    nihstro-headers
    ${PLATFORM_LIBRARIES}
    Threads::Threads
    rt
    m
)

# For LTO support
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(tests PRIVATE
        $<$<BOOL:${ENABLE_LTO}>:-flto>
    )
endif()

add_test(NAME tests COMMAND tests)

if (BORKED3DS_USE_PRECOMPILED_HEADERS)
    target_precompile_headers(tests PRIVATE precompiled_headers.h)
endif()

# Bundle in-place on MSVC so dependencies can be resolved by builds.
if (MSVC)
    include(BundleTarget)
    bundle_target_in_place(tests)
endif()
