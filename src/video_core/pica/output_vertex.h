// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstddef>
#include "common/common_funcs.h"
#include "common/vector_math.h"
#include "video_core/pica_types.h"

namespace Pica {

struct RasterizerRegs;

using AttributeBuffer = std::array<Common::Vec4<f24>, 16>;

struct alignas(16) OutputVertex {
    OutputVertex() = default;
    explicit OutputVertex(const RasterizerRegs& regs, const AttributeBuffer& output);

    union {
        struct {
            Common::Vec4<f24> pos;   // 16 bytes
            Common::Vec4<f24> quat;  // 16 bytes
            Common::Vec4<f24> color; // 16 bytes
            Common::Vec2<f24> tc0;   // 8 bytes
            Common::Vec2<f24> tc1;   // 8 bytes
            f24 tc0_w;               // 4 bytes
            u32 pad1;                // 4 bytes padding
            Common::Vec3<f24> view;  // 12 bytes
            u32 pad2;                // 4 bytes padding
            Common::Vec2<f24> tc2;   // 8 bytes
        };
        std::array<u32, 24> raw; // Force size to be exactly 96 bytes
    };

private:
    template <class Archive>
    void serialize(Archive& ar, const u32) {
        ar & pos;
        ar & quat;
        ar & color;
        ar & tc0;
        ar & tc1;
        ar & tc0_w;
        ar & view;
        ar & tc2;
    }
    friend class boost::serialization::access;
};

static_assert(std::is_standard_layout_v<OutputVertex>, "Structure is not standard layout");
static_assert(sizeof(OutputVertex) == 24 * sizeof(f32), "OutputVertex has invalid size");
static_assert(offsetof(OutputVertex, pos) == 0, "Invalid pos offset");
static_assert(offsetof(OutputVertex, quat) == 16, "Invalid quat offset");
static_assert(offsetof(OutputVertex, color) == 32, "Invalid color offset");
static_assert(offsetof(OutputVertex, tc0) == 48, "Invalid tc0 offset");
static_assert(offsetof(OutputVertex, tc1) == 56, "Invalid tc1 offset");
static_assert(offsetof(OutputVertex, tc0_w) == 64, "Invalid tc0_w offset");
static_assert(offsetof(OutputVertex, view) == 72, "Invalid view offset");
static_assert(offsetof(OutputVertex, tc2) == 88, "Invalid tc2 offset");

} // namespace Pica
