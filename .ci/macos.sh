#!/bin/bash -ex

BUILD_MOLTENVK=true

if [ "$BUILD_MOLTENVK" = true ]; then
    export EXTRA_CMAKE_FLAGS=(-DBORKED3DS_USE_EXTERNAL_MOLTENVK=ON)
    # Build MoltenVK
    cd externals/MoltenVK
    ./fetchDependencies --macos
    xcodebuild build -quiet -project MoltenVKPackaging.xcodeproj -scheme "MoltenVK Package (macOS only)" -configuration "Release"
    cd ../..
    mkdir -p build/externals/MoltenVK/MoltenVK
    mv externals/MoltenVK/Package/Release/MoltenVK build/externals/MoltenVK/
    cd build/externals
    tar cf MoltenVK.tar MoltenVK
    rm -rf MoltenVK
    cd ..
else
    mkdir build && cd build
fi

# Build Borked3DS
cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES="$TARGET" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DENABLE_QT_TRANSLATION=ON \
    -DUSE_SYSTEM_GLSLANG=OFF \
    -DUSE_SYSTEM_VULKAN_HEADERS=OFF \
    -DUSE_SYSTEM_VMA=OFF \
    -DBORKED3DS_USE_EXTERNAL_VULKAN_SPIRV_TOOLS=ON \
    "${EXTRA_CMAKE_FLAGS[@]}" \
    -DBORKED3DS_ENABLE_COMPATIBILITY_REPORTING=OFF \
    -DUSE_DISCORD_PRESENCE=ON
ninja
ninja bundle

ccache -s -v

CURRENT_ARCH=`arch`
if [ "$TARGET" = "$CURRENT_ARCH" ]; then
  ctest -VV -C Release
fi
