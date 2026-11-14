// SPDX-License-Identifier: GPL-2.0-or-later
// Phase 2.3: Native Vulkan 1.3 GLSL 460 replacement for ps 0x00000000e80526cc
// (Yebis 2 7-Tap Weighted Bokeh Filter Pass — sequential tap streaming cuts live VGPRs ~64 -> ~12).
#version 460
#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 2) in vec4 fs_in_attr2_p;
layout(location = 3) in vec4 fs_in_attr3_p;
layout(location = 4) in vec4 fs_in_attr4_p;
layout(location = 5) in vec4 fs_in_attr5_p;
layout(location = 6) in vec4 fs_in_attr6_p;

layout(location = 0) out vec4 frag_color0;

layout(push_constant) uniform AuxData {
    float xoffset;
    float yoffset;
    float xscale;
    float yscale;
    uvec4 ud_regs0;
    uvec4 ud_regs1;
    uvec4 ud_regs2;
    uvec4 ud_regs3;
    uvec4 buf_offsets0;
    uvec4 buf_offsets1;
    uvec2 buf_offsets2;
    uint motion_param;
    uint scene_size;
} push_data;

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { uint data[]; } ssbo_1;
layout(set = 0, binding = 1) uniform texture2D fs_img0;
layout(set = 0, binding = 2) uniform sampler fs_samp0;

vec4 s1_v4(uint base) {
    return vec4(uintBitsToFloat(ssbo_1.data[base + 0u]),
                uintBitsToFloat(ssbo_1.data[base + 1u]),
                uintBitsToFloat(ssbo_1.data[base + 2u]),
                uintBitsToFloat(ssbo_1.data[base + 3u]));
}

void main() {
    const uint d0 = (push_data.buf_offsets0.x & 0xFFu) >> 2u;
    vec4 acc = textureLod(sampler2D(fs_img0, fs_samp0), fs_in_attr0_p.xy, 0.0) * s1_v4(d0 + 616u);
    acc = fma(textureLod(sampler2D(fs_img0, fs_samp0), fs_in_attr1_p.xy, 0.0), s1_v4(d0 + 620u), acc);
    acc = fma(textureLod(sampler2D(fs_img0, fs_samp0), fs_in_attr2_p.xy, 0.0), s1_v4(d0 + 624u), acc);
    acc = fma(textureLod(sampler2D(fs_img0, fs_samp0), fs_in_attr3_p.xy, 0.0), s1_v4(d0 + 628u), acc);
    acc = fma(textureLod(sampler2D(fs_img0, fs_samp0), fs_in_attr4_p.xy, 0.0), s1_v4(d0 + 632u), acc);
    acc = fma(textureLod(sampler2D(fs_img0, fs_samp0), fs_in_attr5_p.xy, 0.0), s1_v4(d0 + 636u), acc);
    acc = fma(textureLod(sampler2D(fs_img0, fs_samp0), fs_in_attr6_p.xy, 0.0), s1_v4(d0 + 640u), acc);
    frag_color0 = acc;
}
