// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <cstddef>
#include "common/common_funcs.h"
#include "common/vector_math.h"
#include "video_core/pica_types.h"

namespace Pica {

struct RasterizerRegs;
using AttributeBuffer = std::array<Common::Vec4<f24>, 16>;

#pragma pack(push, 1)
struct alignas(16) OutputVertex {
    OutputVertex() = default;
    explicit OutputVertex(const RasterizerRegs& regs, const AttributeBuffer& output);

    struct {
        alignas(16) std::array<f24, 4> pos;   // 0-15
        alignas(16) std::array<f24, 4> quat;  // 16-31
        alignas(16) std::array<f24, 4> color; // 32-47
        alignas(8) std::array<f24, 2> tc0;    // 48-55
        alignas(8) std::array<f24, 2> tc1;    // 56-63
        alignas(4) f24 tc0_w;                 // 64-67
        u32 pad1;                             // 68-71
        std::array<f24, 3> view;              // 72-83
        u32 pad2;                             // 84-87
        alignas(8) std::array<f24, 2> tc2;    // 88-95
    };

    // Vector accessors
    Common::Vec4<f24>& pos_vec() {
        return *reinterpret_cast<Common::Vec4<f24>*>(pos.data());
    }
    Common::Vec4<f24>& quat_vec() {
        return *reinterpret_cast<Common::Vec4<f24>*>(quat.data());
    }
    Common::Vec4<f24>& color_vec() {
        return *reinterpret_cast<Common::Vec4<f24>*>(color.data());
    }
    Common::Vec2<f24>& tc0_vec() {
        return *reinterpret_cast<Common::Vec2<f24>*>(tc0.data());
    }
    Common::Vec2<f24>& tc1_vec() {
        return *reinterpret_cast<Common::Vec2<f24>*>(tc1.data());
    }
    Common::Vec3<f24>& view_vec() {
        return *reinterpret_cast<Common::Vec3<f24>*>(view.data());
    }
    Common::Vec2<f24>& tc2_vec() {
        return *reinterpret_cast<Common::Vec2<f24>*>(tc2.data());
    }

    const Common::Vec4<f24>& pos_vec() const {
        return *reinterpret_cast<const Common::Vec4<f24>*>(pos.data());
    }
    const Common::Vec4<f24>& quat_vec() const {
        return *reinterpret_cast<const Common::Vec4<f24>*>(quat.data());
    }
    const Common::Vec4<f24>& color_vec() const {
        return *reinterpret_cast<const Common::Vec4<f24>*>(color.data());
    }
    const Common::Vec2<f24>& tc0_vec() const {
        return *reinterpret_cast<const Common::Vec2<f24>*>(tc0.data());
    }
    const Common::Vec2<f24>& tc1_vec() const {
        return *reinterpret_cast<const Common::Vec2<f24>*>(tc1.data());
    }
    const Common::Vec3<f24>& view_vec() const {
        return *reinterpret_cast<const Common::Vec3<f24>*>(view.data());
    }
    const Common::Vec2<f24>& tc2_vec() const {
        return *reinterpret_cast<const Common::Vec2<f24>*>(tc2.data());
    }

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
#pragma pack(pop)

static_assert(std::is_standard_layout_v<OutputVertex>, "Structure is not standard layout");
static_assert(sizeof(OutputVertex) == 96, "OutputVertex has invalid size");
static_assert(alignof(OutputVertex) == 16, "OutputVertex has invalid alignment");

static_assert(offsetof(OutputVertex, pos) == 0, "Invalid pos offset");
static_assert(offsetof(OutputVertex, quat) == 16, "Invalid quat offset");
static_assert(offsetof(OutputVertex, color) == 32, "Invalid color offset");
static_assert(offsetof(OutputVertex, tc0) == 48, "Invalid tc0 offset");
static_assert(offsetof(OutputVertex, tc1) == 56, "Invalid tc1 offset");
static_assert(offsetof(OutputVertex, tc0_w) == 64, "Invalid tc0_w offset");
static_assert(offsetof(OutputVertex, view) == 72, "Invalid view offset");
static_assert(offsetof(OutputVertex, tc2) == 88, "Invalid tc2 offset");

} // namespace Pica
