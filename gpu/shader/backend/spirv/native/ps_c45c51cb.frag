// SPDX-License-Identifier: GPL-2.0-or-later
// Phase 2 (v1.2.4): Native Vulkan 1.3 GLSL 460 replacement for ps 0x00000000c45c51cb
// (gxposteffect ScreenSpaceAO2_BilateralUpScale.ppo — 2x2 depth-aware SSAO bilateral upscale).
#version 460
#extension GL_EXT_samplerless_texture_functions : require

layout(constant_id = 0) const uint kUnnormDepthUv = 0u;

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

layout(set = 0, binding = 0, std430) readonly buffer SSBO1 { uint data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer SSBO2 { uint data[]; } ssbo_2;

layout(set = 0, binding = 2) uniform texture2D fs_img0;
layout(set = 0, binding = 3) uniform texture2D fs_img1;
layout(set = 0, binding = 4) uniform sampler fs_samp0;
layout(set = 0, binding = 5) uniform sampler fs_samp1;

void main() {
    const uint b0 = (push_data.buf_offsets0.x & 0xFFu) >> 2u;
    const uint b1 = ((push_data.buf_offsets0.x >> 8u) & 0xFFu) >> 2u;

    const vec2 src_size = vec2(uintBitsToFloat(ssbo_1.data[b0 + 16u]),
                               uintBitsToFloat(ssbo_1.data[b0 + 17u]));
    const vec2 inv_size = vec2(uintBitsToFloat(ssbo_1.data[b0 + 18u]),
                               uintBitsToFloat(ssbo_1.data[b0 + 19u]));
    const vec2 uv = fs_in_attr0_p.xy;
    const vec2 hs = 0.5 * inv_size;

    const vec2 d_uv = (kUnnormDepthUv != 0u) ? (uv / vec2(textureSize(fs_img1, 0))) : uv;

    // Issue all 5 texture fetches up front to hide latency.
    const vec4 s0 = textureLod(sampler2D(fs_img0, fs_samp0), vec2(uv.x + hs.x, uv.y - hs.y), 0.0);
    const vec4 s1 = textureLod(sampler2D(fs_img0, fs_samp0), vec2(uv.x - hs.x, uv.y - hs.y), 0.0);
    const vec4 s2 = textureLod(sampler2D(fs_img0, fs_samp0), vec2(uv.x - hs.x, uv.y + hs.y), 0.0);
    const vec4 s3 = textureLod(sampler2D(fs_img0, fs_samp0), vec2(uv.x + hs.x, uv.y + hs.y), 0.0);
    const float d_full = texture(sampler2D(fs_img1, fs_samp1), d_uv).r;

    const float z_a = uintBitsToFloat(ssbo_2.data[b1 + 0u]);
    const float z_b = uintBitsToFloat(ssbo_2.data[b1 + 1u]);
    const float inv_lin_z = (z_a - d_full) / (-z_b);

    const vec3 kPack = vec3(1.53787005e-5, 0.00392156886, 1.0);
    const vec4 z_taps = vec4(dot(s0.gba, kPack), dot(s1.gba, kPack),
                             dot(s2.gba, kPack), dot(s3.gba, kPack));

    const vec2 f = fract(uv * src_size);
    const vec2 of = 1.0 - f;
    const vec2 f2 = f * f;
    const vec2 of2 = of * of;
    const vec4 sp_d2 = vec4(f2.x + of2.y, of2.x + of2.y, of2.x + f2.y, f2.x + f2.y);

    const vec4 w_z = 1.0 / (abs(fma(vec4(inv_lin_z), z_taps, vec4(-1.0))) + 0.01);
    const vec4 w = w_z * inversesqrt(sp_d2);

    const float ao = dot(vec4(s0.r, s1.r, s2.r, s3.r), w) / dot(w, vec4(1.0));
    frag_color0 = vec4(ao);
}
