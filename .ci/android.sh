#!/bin/bash -ex

export NDK_CCACHE=$(which ccache)

if [ ! -z "${DROID_KEYSTORE_B64}" ]; then
    export DROID_KEYSTORE_FILE="${GITHUB_WORKSPACE}/ks.jks"
    base64 --decode <<< "${DROID_KEYSTORE_B64}" > "${DROID_KEYSTORE_FILE}"
fi

# Build Vulkan-ValidationLayers
mkdir -p src/android/app/build/tmp
mkdir -p assets/libs/lib/arm64-v8a
mkdir -p assets/libs/lib/x86_64

cd externals/Vulkan-ValidationLayers

# Build vvl release binary for arm64-v8a
cmake -S . -B build \
  -D CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -D CMAKE_C_COMPILER_LAUNCHER=ccache \
  -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -D CMAKE_CXX_COMPILER=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++ \
  -D CMAKE_C_COMPILER=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang \
  -D CMAKE_CXX_FLAGS=-O2 \
  -D CMAKE_C_FLAGS=-O2 \
  -D CMAKE_EXE_LINKER_FLAGS=-flto=thin \
  -D CMAKE_SHARED_LINKER_FLAGS=-flto=thin \
  -D ANDROID_PLATFORM=28 \
  -D CMAKE_ANDROID_ARCH_ABI=arm64-v8a \
  -D CMAKE_ANDROID_STL_TYPE=c++_static \
  -D ANDROID_USE_LEGACY_TOOLCHAIN_FILE=NO \
  -D CMAKE_BUILD_TYPE=Release \
  -D UPDATE_DEPS=ON \
  -G Ninja

cd build
ninja
cd ..

cmake --install build --prefix build/libs/arm64-v8a
mv build/libs/arm64-v8a/lib/libVkLayer_khronos_validation.so ../../assets/libs/lib/arm64-v8a

rm -rf build

# Build vvl release binary for x86_64
cmake -S . -B build \
  -D CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -D CMAKE_C_COMPILER_LAUNCHER=ccache \
  -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -D CMAKE_CXX_COMPILER=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android28-clang++ \
  -D CMAKE_C_COMPILER=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android28-clang \
  -D CMAKE_CXX_FLAGS=-O2 \
  -D CMAKE_C_FLAGS=-O2 \
  -D CMAKE_EXE_LINKER_FLAGS=-flto=thin \
  -D CMAKE_SHARED_LINKER_FLAGS=-flto=thin \
  -D ANDROID_PLATFORM=28 \
  -D CMAKE_ANDROID_ARCH_ABI=x86_64 \
  -D CMAKE_ANDROID_STL_TYPE=c++_static \
  -D ANDROID_USE_LEGACY_TOOLCHAIN_FILE=NO \
  -D CMAKE_BUILD_TYPE=Release \
  -D UPDATE_DEPS=ON \
  -G Ninja

cd build
ninja
cd ..

cmake --install build --prefix build/libs/x86_64
mv build/libs/x86_64/lib/libVkLayer_khronos_validation.so ../../assets/libs/lib/x86_64

cd ../../assets/libs
zip -r Vulkan-ValidationLayers.zip lib
mv Vulkan-ValidationLayers.zip ../../src/android/app/build/tmp
cd ../..

# Build Borked3DS
cd src/android
chmod +x ./gradlew
./gradlew assembleRelease
./gradlew bundleRelease

ccache -s -v

if [ ! -z "${DROID_KEYSTORE_B64}" ]; then
    rm "${DROID_KEYSTORE_FILE}"
fi
