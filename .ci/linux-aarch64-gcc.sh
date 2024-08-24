#!/bin/bash -ex

if [ "$TARGET" = "appimage-aarch64-gcc" ]; then
    # Compile the AppImage we distribute with Clang.
    export EXTRA_CMAKE_FLAGS=(-DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++-12 -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc-12 -DCMAKE_EXE_LINKER_FLAGS="-lstdc++" -DCMAKE_SHARED_LINKER_FLAGS="-lstdc++" -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu -DCMAKE_PREFIX_PATH=${GITHUB_WORKSPACE}/6.7.2/gcc_arm64)
    # Cross-compile vars
    export ARCHGCC=aarch64-linux-gnu
    export ARCH=aarch64
    # Host strip does not work in a cross-compile environment
    export NO_STRIP=1
    # Bundle required QT wayland libraries
    export EXTRA_QT_PLUGINS="waylandcompositor"
    export EXTRA_PLATFORM_PLUGINS="libqwayland-egl.so;libqwayland-generic.so"
fi

# Cheap hack...
cp /usr/lib/qt6/libexec/moc ${GITHUB_WORKSPACE}/6.7.2/gcc_arm64/libexec/
cp /usr/lib/qt6/libexec/uic ${GITHUB_WORKSPACE}/6.7.2/gcc_arm64/libexec/
cp /usr/lib/qt6/libexec/rcc ${GITHUB_WORKSPACE}/6.7.2/gcc_arm64/libexec/
cp /usr/lib/qt6/bin/lrelease ${GITHUB_WORKSPACE}/6.7.2/gcc_arm64/bin/

mkdir build && cd build
cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    "${EXTRA_CMAKE_FLAGS[@]}" \
    -DUSE_SYSTEM_BOOST=OFF \
    -DUSE_SYSTEM_CATCH2=OFF \
    -DUSE_SYSTEM_CRYPTOPP=OFF \
    -DUSE_SYSTEM_FMT=OFF \
    -DUSE_SYSTEM_XBYAK=OFF \
    -DUSE_SYSTEM_DYNARMIC=OFF \
    -DUSE_SYSTEM_INIH=OFF \
    -DUSE_SYSTEM_FFMPEG_HEADERS=OFF \
    -DUSE_SYSTEM_SOUNDTOUCH=OFF \
    -DUSE_SYSTEM_SDL2=OFF \
    -DUSE_SYSTEM_LIBUSB=OFF \
    -DUSE_SYSTEM_ZSTD=OFF \
    -DUSE_SYSTEM_ENET=OFF \
    -DUSE_SYSTEM_CUBEB=OFF \
    -DUSE_SYSTEM_JSON=OFF \
    -DUSE_SYSTEM_OPENSSL=OFF \
    -DUSE_SYSTEM_CPP_HTTPLIB=OFF \
    -DUSE_SYSTEM_CPP_JWT=OFF \
    -DUSE_SYSTEM_LODEPNG=OFF \
    -DUSE_SYSTEM_OPENAL=OFF \
    -DUSE_SYSTEM_GLSLANG=OFF \
    -DUSE_SYSTEM_VULKAN_HEADERS=OFF \
    -DUSE_SYSTEM_VMA=OFF \
    -DCITRA_USE_EXTERNAL_VULKAN_SPIRV_TOOLS=ON \
    -DCITRA_ENABLE_COMPATIBILITY_REPORTING=ON \
    -DCITRA_USE_PRECOMPILED_HEADERS=OFF \
    -DCITRA_CROSS_COMPILE_AARCH64=ON \
    -DENABLE_LTO=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_QT_TRANSLATION=ON \
    -DENABLE_COMPATIBILITY_LIST_DOWNLOAD=ON \
    -DUSE_DISCORD_PRESENCE=OFF
ninja

if [ "$TARGET" = "appimage-aarch64-gcc" ]; then
    ninja bundle
    # TODO: Our AppImage environment currently uses an older ccache version without the verbose flag.
    ccache -s
else
    ccache -s -v
fi

ctest -VV -C Release
