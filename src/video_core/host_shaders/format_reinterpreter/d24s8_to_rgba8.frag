// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

//? #version 430 core

precision highp int;
precision highp float;
precision highp sampler2D;

layout(location = 0) in mediump vec2 tex_coord;
layout(location = 0) out lowp vec4 frag_color;

#if defined(GL_ES) && __VERSION__ < 320
// Single combined sampler for OpenGL ES
uniform sampler2D depth_stencil;
#else
layout(binding = 0) uniform highp sampler2D depth;
layout(binding = 1) uniform lowp usampler2D stencil;
#endif

void main() {
#if defined(GL_ES)
    // OpenGL ES path
    vec2 coord = tex_coord * vec2(textureSize(depth_stencil, 0));
    ivec2 tex_icoord = ivec2(coord);
    
    vec4 depth_stencil_val = texelFetch(depth_stencil, tex_icoord, 0);
    uint depth_val = uint(depth_stencil_val.x * float(0xFFFFFF));
    uint combined_val = uint(depth_stencil_val.x * float(0xFFFFFFFF));
    uint stencil_val = (combined_val >> 24u) & 0xFFu;
    
    vec4 components = vec4(
        float(stencil_val),
        float((depth_val >> 16u) & 0xFFu),
        float((depth_val >> 8u) & 0xFFu),
        float(depth_val & 0xFFu)
    );
    
    frag_color = components / 255.0;
#else
    mediump vec2 coord = tex_coord * vec2(textureSize(depth, 0));
    mediump ivec2 tex_icoord = ivec2(coord);
    highp uint depth_val =
        uint(texelFetch(depth, tex_icoord, 0).x * (exp2(32.0) - 1.0));
    lowp uint stencil_val = texelFetch(stencil, tex_icoord, 0).x;
    highp uvec4 components =
        uvec4(stencil_val, (uvec3(depth_val) >> uvec3(24u, 16u, 8u)) & 0x000000FFu);
    frag_color = vec4(components) / (exp2(8.0) - 1.0);
#endif
}
