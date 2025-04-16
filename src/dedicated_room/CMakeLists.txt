set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMakeModules)

add_executable(borked3ds-room
    precompiled_headers.h
    borked3ds-room.cpp
    borked3ds-room.rc
)

# Add dependency on scm_rev.cpp generation
add_dependencies(borked3ds-room borked3ds_common)

if (MSVC AND ENABLE_LTO)
  target_compile_options(borked3ds-room PRIVATE
    /wd5049 # 'string': Embedding a full path may result in machine-dependent output (breaks LTO on MSVC)
  )
endif()

create_target_directory_groups(borked3ds-room)

# Fix linking order with repeated libraries to handle circular dependencies
target_link_libraries(borked3ds-room PRIVATE
    network
    borked3ds_common
    web_service
    network
    borked3ds_common
    enet
    Boost::serialization
    Boost::iostreams
    fmt
    gamemode
    zstd
    tracy
    ssl
    crypto
    cryptopp
    network # Third time to resolve any remaining circular dependencies
    borked3ds_common # Third time to resolve any remaining circular dependencies
    ${PLATFORM_LIBRARIES}
    Threads::Threads
    rt
    m
)

if (MSVC)
    target_link_libraries(borked3ds-room PRIVATE getopt)
endif()

if(UNIX AND NOT APPLE)
    install(TARGETS borked3ds-room RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endif()

if (BORKED3DS_USE_PRECOMPILED_HEADERS)
    target_precompile_headers(borked3ds-room PRIVATE precompiled_headers.h)
endif()

# For LTO support
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(borked3ds-room PRIVATE
        -fuse-ld=mold
        $<$<BOOL:${ENABLE_LTO}>:-flto>
    )
endif()

# Bundle in-place on MSVC so dependencies can be resolved by builds.
if (MSVC)
    include(BundleTarget)
    bundle_target_in_place(borked3ds-room)
endif()
