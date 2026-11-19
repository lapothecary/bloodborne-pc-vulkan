// SPDX-License-Identifier: GPL-2.0-or-later
// Step 2.3: Native Vulkan 1.3 GLSL 460 replacement for ps 0x0000000001ae5388
// (5b. SSR Hi-Z 4-Tap Minimum Depth Pyramid Downsample Pass).
#version 460
#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) in vec4 fs_in_attr0_p;
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

void main() {
    const uint d0 = bitfieldExtract(push_data.buf_offsets0.x, 0, 8) >> 2u;
    const vec2 uv = fs_in_attr0_p.xy;

    const vec2 o0 = vec2(uintBitsToFloat(ssbo_1.data[d0 + 0u]),
                         uintBitsToFloat(ssbo_1.data[d0 + 1u]));
    const vec2 o1 = vec2(uintBitsToFloat(ssbo_1.data[d0 + 4u]),
                         uintBitsToFloat(ssbo_1.data[d0 + 5u]));
    const vec2 o2 = vec2(uintBitsToFloat(ssbo_1.data[d0 + 8u]),
                         uintBitsToFloat(ssbo_1.data[d0 + 9u]));
    const vec2 o3 = vec2(uintBitsToFloat(ssbo_1.data[d0 + 12u]),
                         uintBitsToFloat(ssbo_1.data[d0 + 13u]));

    const float s2 = texture(sampler2D(fs_img0, fs_samp0), uv + o2).x;
    const float s3 = texture(sampler2D(fs_img0, fs_samp0), uv + o3).x;
    const float s1 = texture(sampler2D(fs_img0, fs_samp0), uv + o1).x;
    const float s0 = texture(sampler2D(fs_img0, fs_samp0), uv + o0).x;

    frag_color0 = vec4(min(min(s0, s1), min(s2, s3)), 0.0, 0.0, 1.0);
}
