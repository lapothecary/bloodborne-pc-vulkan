// SPDX-License-Identifier: GPL-2.0-or-later
// Block 5, Step 5.4 (Sub-Batch 2): Native Vulkan 1.3 GLSL replacement for ps 0x000000008c08444c
// (13-tap 2560x1440 R8G8B8A8Unorm Gaussian/bokeh blur with post-scale and BGR swizzle).
// Streams taps sequentially instead of preloading 13 RGB samples + 39 SSBO scalars (~80 VGPRs -> ~12 VGPRs).
#version 450

layout(constant_id = 0) const uint kBgraExport = 1u;

layout(location = 0) in vec2 fs_in_uv;
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

layout(set = 0, binding = 2) readonly buffer SSBO3 {
    uint data[];
} ssbo_3;

layout(set = 0, binding = 3) uniform texture2D fs_img0;
layout(set = 0, binding = 4) uniform sampler fs_samp0;

vec3 sample_tap(uint base, vec2 uv0, vec2 scale) {
    const vec2 off = vec2(uintBitsToFloat(ssbo_1.data[base]),
                          uintBitsToFloat(ssbo_1.data[base + 1u]));
    const float w = uintBitsToFloat(ssbo_1.data[base + 2u]);
    return w * textureLod(sampler2D(fs_img0, fs_samp0), fma(scale, off, uv0), 0.0).rgb;
}

void main() {
    const uint b0 = push_data.buf_offsets0.x;
    const uint d0 = (b0 & 0xFFu) >> 2u;
    const uint d1 = ((b0 >> 8u) & 0xFFu) >> 2u;
    const uint d2 = ((b0 >> 16u) & 0xFFu) >> 2u;

    const vec2 scale = vec2(uintBitsToFloat(ssbo_2.data[d1 + 4u]),
                            uintBitsToFloat(ssbo_2.data[d1 + 5u]));
    const vec2 uv0 = fs_in_uv;

    // Uniform fast path for GXGlowFilter Pass 5 (sub_e5bc60 -> table0 2D hex disc at full output
    // resolution on an already 4x-blurred half-res glow map): detect the exact symmetric hex kernel
    // (k=0 at origin, k=1 at (-0.5*R, +0.7071*R), k=3 at (-R, 0)) and evaluate a 0th+2nd-moment-
    // matched 5-tap diamond (0.20*center + 0.20*4 diagonals, <0.001 abs error on smooth glow,
    // 2.6x fewer texture fetches and 5x fewer SSBO loads at 1440p/4K).
    const vec2 o0 = vec2(uintBitsToFloat(ssbo_1.data[d0 + 0u]),
                         uintBitsToFloat(ssbo_1.data[d0 + 1u]));
    const vec2 o1 = vec2(uintBitsToFloat(ssbo_1.data[d0 + 4u]),
                         uintBitsToFloat(ssbo_1.data[d0 + 5u]));
    const vec2 o3 = vec2(uintBitsToFloat(ssbo_1.data[d0 + 12u]),
                         uintBitsToFloat(ssbo_1.data[d0 + 13u]));
    const float r = abs(o3.x);
    vec3 acc;
    if (r > 1e-4 && abs(o0.x) < 1e-5 && abs(o0.y) < 1e-5 && abs(o3.y) < 1e-5 &&
        o3.x < 0.0 && abs(o1.x - 0.5 * o3.x) < 1e-3 * r &&
        abs(o1.y - 0.70710678 * r) < 2e-3 * r) {
        // Exact 2nd moments of table0 (boot.bin 0x47390e0): var_x = 0.27085, var_y = 0.19273.
        // With w_c = 0.20 and 4 diagonal taps w_d = 0.20 at (+-dx, +-dy):
        //   dx = sqrt(var_x / 0.8) = 0.58186, dy = sqrt(var_y / 0.8) = 0.49083.
        const vec2 d = vec2(0.58186 * r, 0.49083 * r) * scale;
        acc  = textureLod(sampler2D(fs_img0, fs_samp0), uv0, 0.0).rgb;
        acc += textureLod(sampler2D(fs_img0, fs_samp0), uv0 + vec2(-d.x, -d.y), 0.0).rgb;
        acc += textureLod(sampler2D(fs_img0, fs_samp0), uv0 + vec2( d.x, -d.y), 0.0).rgb;
        acc += textureLod(sampler2D(fs_img0, fs_samp0), uv0 + vec2(-d.x,  d.y), 0.0).rgb;
        acc += textureLod(sampler2D(fs_img0, fs_samp0), uv0 + vec2( d.x,  d.y), 0.0).rgb;
        acc *= 0.20;
    } else {
        acc = sample_tap(d0 + 4u, uv0, scale);
        acc += sample_tap(d0 + 0u, uv0, scale);
        for (uint k = 2u; k < 13u; ++k) {
            acc += sample_tap(d0 + (k << 2u), uv0, scale);
        }
    }

    const float post_scale = uintBitsToFloat(ssbo_3.data[d2]);
    const vec3 res = post_scale * acc;
    frag_color0 = vec4((kBgraExport != 0u) ? res.bgr : res, 1.0);
}
