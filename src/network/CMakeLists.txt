add_library(network STATIC
    announce_multiplayer_session.cpp
    announce_multiplayer_session.h
    artic_base/artic_base_client.cpp
    artic_base/artic_base_client.h
    artic_base/artic_base_common.h
    network.cpp
    network.h
    network_settings.cpp
    network_settings.h
    packet.cpp
    packet.h
    precompiled_headers.h
    room.cpp
    room.h
    room_member.cpp
    room_member.h
    socket_manager.cpp
    socket_manager.h
    verify_user.cpp
    verify_user.h
)

if (MSVC AND ENABLE_LTO)
  target_compile_options(network PRIVATE
    /wd5049 # 'string': Embedding a full path may result in machine-dependent output (breaks LTO on MSVC)
  )
endif()

create_target_directory_groups(network)

# Update library dependencies
target_link_libraries(network
    PUBLIC
        borked3ds_common
    PRIVATE
        enet
        Boost::serialization
        httplib
        $<$<BOOL:${ENABLE_WEB_SERVICE}>:web_service>
)

# For LTO support
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(network PRIVATE
        $<$<BOOL:${ENABLE_LTO}>:-flto>
    )
endif()

if (BORKED3DS_USE_PRECOMPILED_HEADERS)
    target_precompile_headers(network PRIVATE precompiled_headers.h)
endif()
