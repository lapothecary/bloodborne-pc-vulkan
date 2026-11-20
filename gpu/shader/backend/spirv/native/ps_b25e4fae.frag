// SPDX-License-Identifier: GPL-2.0-or-later
// Native Vulkan 1.3 GLSL replacement for ps 0x00000000b25e4fae
// (Motion Blur alpha-cutout velocity pass, 960x540 / half-res, 3 draws/frame).
// Replaces software BaryCoordKHR interpolation with native hardware interpolation and
// early-outs before SSBO2 / NDC division on alpha-discarded foliage/hair/cloth fragments.
#version 460
#extension GL_EXT_demote_to_helper_invocation : require

#ifndef ATTR2_LOC
#define ATTR2_LOC 2
#endif

layout(location = 0) in vec4 fs_in_attr0;
layout(location = 1) in vec4 fs_in_attr1;
layout(location = ATTR2_LOC) in vec4 fs_in_attr2;
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

layout(set = 0, binding = 0) readonly buffer SSBO1 {
    uint data[];
} ssbo_1;

layout(set = 0, binding = 1) readonly buffer SSBO2 {
    uint data[];
} ssbo_2;

layout(set = 0, binding = 2) uniform texture2D fs_img0;
layout(set = 0, binding = 3) uniform sampler fs_samp0;

void main() {
    const uint b0 = (push_data.buf_offsets0.x & 0xFFu) >> 2u;
    const vec2 uv_scale = vec2(uintBitsToFloat(ssbo_1.data[b0 + 0u]),
                               uintBitsToFloat(ssbo_1.data[b0 + 1u]));
    const vec2 uv_bias  = vec2(uintBitsToFloat(ssbo_1.data[b0 + 2u]),
                               uintBitsToFloat(ssbo_1.data[b0 + 3u]));
    const vec2 uv = fma(fs_in_attr2.xy, uv_scale, uv_bias);

    const float alpha = texture(sampler2D(fs_img0, fs_samp0), uv).a;
    if (alpha < 0.00392156886) {
        demote;
        return;
    }

    const uint b1 = ((push_data.buf_offsets0.x >> 8u) & 0xFFu) >> 2u;
    const vec2 vel_scale = vec2(uintBitsToFloat(ssbo_2.data[b1 + 0u]),
                                uintBitsToFloat(ssbo_2.data[b1 + 1u]));

    const float inv_cur_w  = 1.0 / fs_in_attr0.w;
    const float inv_prev_w = 1.0 / fs_in_attr1.w;
    const vec2 vel = (fs_in_attr0.xy * inv_cur_w - fs_in_attr1.xy * inv_prev_w) * vel_scale;
    const vec2 enc = clamp(fma(vel, vec2(0.5), vec2(0.5)), vec2(0.0), vec2(1.0));
    const float z_enc = fs_in_attr0.w / (fs_in_attr0.w + 10.0);

    frag_color0 = vec4(enc.x, enc.y, z_enc, 1.0);
}
