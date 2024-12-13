set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMakeModules)

add_executable(borked3ds-room
    precompiled_headers.h
    borked3ds-room.cpp
    borked3ds-room.rc
)

set_target_properties(borked3ds-room PROPERTIES OUTPUT_NAME "citra-room")

if (MSVC AND ENABLE_LTO)
  target_compile_options(borked3ds-room PRIVATE
    /wd5049 # 'string': Embedding a full path may result in machine-dependent output (breaks LTO on MSVC)
  )
endif()

create_target_directory_groups(borked3ds-room)

target_link_libraries(borked3ds-room PRIVATE borked3ds_common network)

if (NOT MSVC)
    target_link_libraries(borked3ds-room PRIVATE m)
endif()

if (ENABLE_WEB_SERVICE)
    target_link_libraries(borked3ds-room PRIVATE web_service)
endif()

target_link_libraries(borked3ds-room PRIVATE cryptopp)
if (MSVC)
    target_link_libraries(borked3ds-room PRIVATE getopt)
endif()
target_link_libraries(borked3ds-room PRIVATE ${PLATFORM_LIBRARIES} Threads::Threads)

if(UNIX AND NOT APPLE)
    install(TARGETS borked3ds-room RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endif()

if (BORKED3DS_USE_PRECOMPILED_HEADERS)
    target_precompile_headers(borked3ds-room PRIVATE precompiled_headers.h)
endif()

# Bundle in-place on MSVC so dependencies can be resolved by builds.
if (MSVC)
    include(BundleTarget)
    bundle_target_in_place(borked3ds-room)
endif()
