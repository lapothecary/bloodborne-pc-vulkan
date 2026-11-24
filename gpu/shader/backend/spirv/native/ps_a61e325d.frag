// SPDX-License-Identifier: GPL-2.0-or-later
// Pack #5: Native Vulkan 1.3 GLSL 460 replacement for ps 0x00000000a61e325d
// (gxposteffect Sample13WeightforGlowKernel.ppo — 13-tap RGBA Glow blur + alpha post-scale).
// Streams taps sequentially instead of preloading 13 RGBA samples + 39 SSBO scalars.
#version 460

layout(constant_id = 0) const uint kBgraExport = 0u;

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
layout(set = 0, binding = 2, std430) readonly buffer SSBO3 { uint data[]; } ssbo_3;

layout(set = 0, binding = 3) uniform texture2D fs_img0;
layout(set = 0, binding = 4) uniform sampler fs_samp0;

vec4 sample_tap(uint base, vec2 uv0, vec2 scale) {
    const vec2 off = vec2(uintBitsToFloat(ssbo_1.data[base]),
                          uintBitsToFloat(ssbo_1.data[base + 1u]));
    const float w = uintBitsToFloat(ssbo_1.data[base + 2u]);
    return w * textureLod(sampler2D(fs_img0, fs_samp0), fma(scale, off, uv0), 0.0);
}

void main() {
    const uint b0 = push_data.buf_offsets0.x;
    const uint d0 = (b0 & 0xFFu) >> 2u;
    const uint d1 = ((b0 >> 8u) & 0xFFu) >> 2u;
    const uint d2 = ((b0 >> 16u) & 0xFFu) >> 2u;

    const vec2 scale = vec2(uintBitsToFloat(ssbo_2.data[d1 + 4u]),
                            uintBitsToFloat(ssbo_2.data[d1 + 5u]));
    const vec2 uv0 = fs_in_attr0_p.xy;

    vec4 acc = sample_tap(d0 + 4u, uv0, scale);
    acc += sample_tap(d0 + 0u, uv0, scale);
    for (uint k = 2u; k < 7u; ++k) {
        acc += sample_tap(d0 + (k << 2u), uv0, scale);
    }
    // Bit-exact uniform deduplication of FromSoftware's duplicated k=7/k=8 (+0.6667) tap in
    // boot.bin 0x4739150 / 0x47391c0 (saves 1 texture fetch per pixel across all 4 glow passes).
    const vec2 off7 = vec2(uintBitsToFloat(ssbo_1.data[d0 + 28u]),
                           uintBitsToFloat(ssbo_1.data[d0 + 29u]));
    const vec2 off8 = vec2(uintBitsToFloat(ssbo_1.data[d0 + 32u]),
                           uintBitsToFloat(ssbo_1.data[d0 + 33u]));
    const float w7 = uintBitsToFloat(ssbo_1.data[d0 + 30u]);
    const float w8 = uintBitsToFloat(ssbo_1.data[d0 + 34u]);
    if (off7 == off8) {
        acc += (w7 + w8) * textureLod(sampler2D(fs_img0, fs_samp0), fma(scale, off7, uv0), 0.0);
    } else {
        acc += w7 * textureLod(sampler2D(fs_img0, fs_samp0), fma(scale, off7, uv0), 0.0);
        acc += w8 * textureLod(sampler2D(fs_img0, fs_samp0), fma(scale, off8, uv0), 0.0);
    }
    for (uint k = 9u; k < 13u; ++k) {
        acc += sample_tap(d0 + (k << 2u), uv0, scale);
    }

    const float alpha_scale = uintBitsToFloat(ssbo_3.data[d2 + 1u]);
    frag_color0 = vec4((kBgraExport != 0u) ? acc.bgr : acc.rgb, alpha_scale * acc.a);
}
