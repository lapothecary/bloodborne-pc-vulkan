// SPDX-License-Identifier: GPL-2.0-or-later
// Step 2.3: Native Vulkan 1.3 GLSL 460 replacement for ps 0x000000004eb5a42f
// (8c. Yebis 2 Bright-Pass Glare Soft-Knee Threshold Pass).
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
    const vec4 s = textureLod(sampler2D(fs_img0, fs_samp0), fs_in_attr0_p.xy, 0.0);

    const vec3 pre_scale = vec3(uintBitsToFloat(ssbo_1.data[d0 + 616u]),
                                uintBitsToFloat(ssbo_1.data[d0 + 617u]),
                                uintBitsToFloat(ssbo_1.data[d0 + 618u]));
    const float thresh = uintBitsToFloat(ssbo_1.data[d0 + 627u]);

    const vec3 c = s.rgb * pre_scale;
    const float lum = (c.r * 0.333333343) + ((c.g * 0.333333343) + (c.b * 0.333333343));
    const float excess = max(0.0, lum - thresh);

    vec3 out_rgb = vec3(0.0);
    if (excess > 0.0) {
        const float inv_lum = 1.0 / (lum + 2.38418579e-07);
        const float k0 = uintBitsToFloat(ssbo_1.data[d0 + 836u]);
        const float k1 = uintBitsToFloat(ssbo_1.data[d0 + 839u]);
        const float gain = k1 * ((excess + sqrt(fma(excess, k0 + excess, 1.0))) - 1.0);
        const vec3 out_scale = vec3(uintBitsToFloat(ssbo_1.data[d0 + 620u]),
                                    uintBitsToFloat(ssbo_1.data[d0 + 621u]),
                                    uintBitsToFloat(ssbo_1.data[d0 + 622u]));
        out_rgb = clamp((c * (inv_lum * gain)) * out_scale, vec3(0.0), vec3(64000.0));
    }

    frag_color0 = vec4(out_rgb, s.a);
}
