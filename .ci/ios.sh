#!/bin/bash -ex

# Build MoltenVK
cd externals/MoltenVK
./fetchDependencies --ios
xcodebuild build -quiet -project MoltenVKPackaging.xcodeproj -scheme "MoltenVK Package (iOS only)" -configuration "Release"
cd ../..
mkdir -p build/externals/MoltenVK/MoltenVK
mv externals/MoltenVK/Package/Release/MoltenVK/dynamic build/externals/MoltenVK/MoltenVK/
cd build/externals
tar cf MoltenVK.tar MoltenVK
rm -rf MoltenVK
cd ..

# Build Borked3DS
cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DENABLE_QT_TRANSLATION=ON \
    -DBORKED3DS_USE_PRECOMPILED_HEADERS=OFF \
    -DBORKED3DS_ENABLE_COMPATIBILITY_REPORTING=OFF \
    -DBORKED3DS_USE_EXTERNAL_VULKAN_SPIRV_TOOLS=ON \
    -DBORKED3DS_USE_EXTERNAL_MOLTENVK=ON \
    -DENABLE_COMPATIBILITY_LIST_DOWNLOAD=OFF
ninja

ccache -s -v
