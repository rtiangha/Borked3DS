// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_funcs.h"
#include "common/vector_math.h"
#include "video_core/pica_types.h"

namespace Pica {

struct RasterizerRegs;

using AttributeBuffer = std::array<Common::Vec4<f24>, 16>;

struct alignas(16) OutputVertex { // Align to 16-byte boundary for Vec4
    OutputVertex() = default;
    explicit OutputVertex(const RasterizerRegs& regs, const AttributeBuffer& output);

    alignas(16) Common::Vec4<f24> pos;   // 16 bytes
    alignas(16) Common::Vec4<f24> quat;  // 16 bytes
    alignas(16) Common::Vec4<f24> color; // 16 bytes
    alignas(8) Common::Vec2<f24> tc0;    // 8 bytes
    alignas(8) Common::Vec2<f24> tc1;    // 8 bytes
    alignas(4) f24 tc0_w;                // 4 bytes
    INSERT_PADDING_WORDS(1);             // 4 bytes
    alignas(16) Common::Vec3<f24> view;  // 12 bytes + 4 bytes padding
    INSERT_PADDING_WORDS(1);             // 4 bytes
    alignas(8) Common::Vec2<f24> tc2;    // 8 bytes
                                         // Total: 96 bytes (24 * sizeof(f32))

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

} // namespace Pica
