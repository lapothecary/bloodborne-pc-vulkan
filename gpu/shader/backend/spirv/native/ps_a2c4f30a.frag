// SPDX-License-Identifier: GPL-2.0-or-later
// Phase 2.3: Native Vulkan 1.3 GLSL 460 replacement for ps 0x00000000a2c4f30a
// (Yebis 2 Fullscreen DoF Near/Far Bokeh Composite Pass).
#version 460
#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;

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
layout(set = 0, binding = 2) uniform texture2D fs_img1;
layout(set = 0, binding = 3) uniform sampler fs_samp0;
layout(set = 0, binding = 4) uniform sampler fs_samp1;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }

void main() {
    const uint d0 = (push_data.buf_offsets0.x & 0xFFu) >> 2u;
    const vec4 c0 = textureLod(sampler2D(fs_img0, fs_samp0), fs_in_attr0_p.xy, 0.0);
    const vec4 c1 = textureLod(sampler2D(fs_img1, fs_samp1), fs_in_attr1_p.xy, 0.0);

    const float far_t = clamp(fma(c0.w, s1f(d0 + 76u), s1f(d0 + 77u)), 0.0, 1.0);
    const float near_t = sqrt(clamp(c1.w * 4.0, 0.0, 1.0));
    // Exact algebraic collapse of mix(mix(c0.rgb, c1.rgb, far_t), c1.rgb, near_t):
    const float t = fma(near_t, 1.0 - far_t, far_t);

    frag_color0 = vec4(mix(c0.rgb, c1.rgb, t), c0.w);
}
