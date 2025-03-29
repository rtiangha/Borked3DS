// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

//? #version 430 core

// Anaglyph Red-Cyan shader based on Dubois algorithm
// Constants taken from the paper:
// "Conversion of a Stereo Pair to Anaglyph with
// the Least-Squares Projection Method"
// Eric Dubois, March 2009
const mat3 l = mat3( 0.437f, 0.449f, 0.164f,
              -0.062f,-0.062f,-0.024f,
              -0.048f,-0.050f,-0.017f);
const mat3 r = mat3(-0.011f,-0.032f,-0.007f,
               0.377f, 0.761f, 0.009f,
              -0.026f,-0.093f, 1.234f);

layout(location = 0) in vec2 frag_tex_coord;
layout(location = 0) out vec4 color;

layout(binding = 0) uniform sampler2D color_texture;
layout(binding = 1) uniform sampler2D color_texture_r;

uniform vec4 resolution;
uniform int layer;

void main() {
    vec4 color_tex_l = texture(color_texture, frag_tex_coord);
    vec4 color_tex_r = texture(color_texture_r, frag_tex_coord);
    color = vec4(color_tex_l.rgb*l+color_tex_r.rgb*r, color_tex_l.a);
}
