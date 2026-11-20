// SPDX-License-Identifier: GPL-2.0-or-later
// Native Vulkan 1.3 GLSL replacement for Filter\GXSssBlurX.ppo (0x2eacb983, SSS_AXIS_Y=0)
// and Filter\GXSssBlurY.ppo (0x8c13b984, SSS_AXIS_Y=1) — 17-tap Separable SSS Blur.
//
// Eliminates software BaryCoordKHR interpolation, collapses ~90 VGPRs -> ~12 VGPRs, hoists
// textureSize() out of the 16-tap loop (when kUnnormSampler1 != 0), and skips 31 redundant
// full-resolution texture fetches on non-skin pixels (step_scale == 0.0, ~98% of the screen).
#version 460
#extension GL_EXT_samplerless_texture_functions : require

#ifndef SSS_AXIS_Y
#define SSS_AXIS_Y 0
#endif

#ifndef ATTR0_LOC
#define ATTR0_LOC 0
#endif

layout(constant_id = 0) const uint kUnnormSampler1 = 0u;

layout(location = ATTR0_LOC) in vec4 fs_in_attr0;
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
    float data[];
} ssbo_2;

layout(set = 0, binding = 2) uniform texture2D fs_img0;
layout(set = 0, binding = 3) uniform texture2D fs_img1;
layout(set = 0, binding = 4) uniform texture2D fs_img2;
layout(set = 0, binding = 5) uniform sampler fs_samp0;
layout(set = 0, binding = 6) uniform sampler fs_samp1;

void main() {
    const uint b0 = (push_data.buf_offsets0.x & 0xFFu) >> 2u;
    const uint b1 = ((push_data.buf_offsets0.x >> 8u) & 0xFFu) >> 2u;

    const vec2 uv = fs_in_attr0.xy;
    const float gbuf_a = textureLod(sampler2D(fs_img0, fs_samp0), uv, 0.0).a;
    const vec4 center_col = textureLod(sampler2D(fs_img1, fs_samp0), uv, 0.0);
    const float center_d = textureLod(sampler2D(fs_img2, fs_samp0), uv, 0.0).r;

    const float proj_a = uintBitsToFloat(ssbo_1.data[b0 + 68u]);
    const float proj_b = uintBitsToFloat(ssbo_1.data[b0 + 70u]);
    const float neg_proj_b = -proj_b;
    const float center_z = neg_proj_b / (proj_a - center_d);

    const uint sss_id = uint(int(gbuf_a * 255.0));
    const uint base = sss_id * 72u + b1;
    const float sss_width = ssbo_2.data[base];
    const vec3 w0 = vec3(ssbo_2.data[base + 4u],
                         ssbo_2.data[base + 5u],
                         ssbo_2.data[base + 6u]);

    const float step_scale = (1.0 / center_z) * sss_width * center_col.a * 0.333333343;
    const vec2 inv_img1_sz = (kUnnormSampler1 != 0u) ? (1.0 / vec2(textureSize(fs_img1, 0))) : vec2(1.0);

    // Exact algebraic fast-path when step_scale == 0.0 (non-skin / zero-width / zero-strength):
    // every tap samples the exact same `uv` with `abs(center_z - tap_z) == 0.0`, eliminating
    // 31 full-resolution texture fetches while preserving exact profile weight sums.
    if (step_scale == 0.0) {
        const vec2 uv1 = (kUnnormSampler1 != 0u) ? (uv * inv_img1_sz) : uv;
        const vec3 tap_col0 = textureLod(sampler2D(fs_img1, fs_samp1), uv1, 0.0).rgb;
        vec3 w_sum = vec3(0.0);
        for (uint k = 2u; k < 18u; ++k) {
            const uint off = base + (k << 2u);
            w_sum += vec3(ssbo_2.data[off], ssbo_2.data[off + 1u], ssbo_2.data[off + 2u]);
        }
        frag_color0 = vec4(fma(w_sum, tap_col0, w0 * center_col.rgb), center_col.a);
        return;
    }

    const float falloff_scale = 300.0 * sss_width;
    vec3 acc = w0 * center_col.rgb;

    for (uint k = 2u; k < 18u; ++k) {
        const uint off = base + (k << 2u);
        const vec3 wk = vec3(ssbo_2.data[off], ssbo_2.data[off + 1u], ssbo_2.data[off + 2u]);
        const float tap_off = ssbo_2.data[off + 3u];
#if SSS_AXIS_Y
        const vec2 tap_uv = vec2(uv.x, fma(tap_off, step_scale, uv.y));
#else
        const vec2 tap_uv = vec2(fma(tap_off, step_scale, uv.x), uv.y);
#endif
        const vec2 tap_uv1 = (kUnnormSampler1 != 0u) ? (tap_uv * inv_img1_sz) : tap_uv;
        const vec3 tap_col = textureLod(sampler2D(fs_img1, fs_samp1), tap_uv1, 0.0).rgb;
        const float tap_d = textureLod(sampler2D(fs_img2, fs_samp0), tap_uv, 0.0).r;
        const float tap_z = neg_proj_b / (proj_a - tap_d);
        const float s = clamp(falloff_scale * abs(center_z - tap_z), 0.0, 1.0);
        acc += wk * mix(tap_col, center_col.rgb, s);
    }

    frag_color0 = vec4(acc, center_col.a);
}
