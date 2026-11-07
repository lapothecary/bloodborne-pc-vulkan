#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Auxiliary data block
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

// VFX Dual-Frame Blend 2-MRT — ps_c575bc29
layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 5) flat in vec4 fs_in_attr2_p;

layout(location = 0) out vec4 frag_color0;
layout(location = 1) out vec4 frag_color1;

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { uint data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { uint data[]; } ssbo_2;

layout(set = 0, binding = 2) uniform texture2D fs_img0;
layout(set = 0, binding = 3) uniform sampler fs_samp0;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }

vec3 eval_overlay_sq(vec4 tex, vec3 vcol, bool mode2) {
    vec3 inv_v = 1.0 - vcol;
    vec3 inv_v2 = inv_v * 2.0;
    vec3 ov = vec3(
        ((tex.x * tex.w) < 0.5) ? ((tex.w * (tex.x * vcol.x)) * 2.0) : (1.0 - (inv_v2.x - (tex.w * (tex.x * inv_v.x)) * 2.0)),
        ((tex.y * tex.w) < 0.5) ? ((tex.w * (tex.y * vcol.y)) * 2.0) : (1.0 - (inv_v2.y - (tex.w * (tex.y * inv_v.y)) * 2.0)),
        (0.5 > (tex.z * tex.w)) ? ((tex.w * (tex.z * vcol.z)) * 2.0) : (1.0 - (inv_v2.z - (tex.w * (tex.z * inv_v.z)) * 2.0))
    );
    vec3 mul = tex.xyz * vcol;
    return (!mode2) ? (mul * mul) : (mode2 ? (ov * ov) : vec3(0.0));
}

void main() {
    uint u0 = push_data.buf_offsets0.x;
    uint d0 = bitfieldExtract(u0, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(u0, 8, 8) >> 2u;

    vec4 tex0 = texture(sampler2D(fs_img0, fs_samp0), fs_in_attr0_p.xy);
    vec4 tex1 = texture(sampler2D(fs_img0, fs_samp0), fs_in_attr0_p.zw);
    bool mode2 = (ssbo_1.data[d0 + 95u] == 2u);

    vec3 col0 = eval_overlay_sq(tex0, fs_in_attr1_p.xyz, mode2);
    vec3 col1 = eval_overlay_sq(tex1, fs_in_attr1_p.xyz, mode2);
    float a0 = tex0.w * fs_in_attr1_p.w;
    float a1 = tex1.w * fs_in_attr1_p.w;

    float blend_t = unpackHalf2x16((floatBitsToUint(fs_in_attr2_p.x) >> 16u) & 0xFFFFu).x;
    vec3 col_mix = blend_t * (col1 - col0) + col0;
    float a_mix = clamp(blend_t * (a1 - a0) + a0, 0.0, 1.0);

    vec3 tint = vec3(s2f(d1 + 3u), s2f(d1 + 7u), s2f(d1 + 11u));
    vec3 final_rgb = clamp((0.0 < a_mix) ? (tint * col_mix) : col_mix, 0.0, 1.0);
    float final_a = clamp(a_mix, 0.0, 1.0);

    frag_color1 = vec4(
        s1f(d0 + 88u) * final_rgb.x,
        s1f(d0 + 89u) * final_rgb.y,
        s1f(d0 + 90u) * final_rgb.z,
        clamp(final_a * s1f(d0 + 91u), 0.0, 1.0)
    );
    frag_color0 = vec4(final_rgb, final_a);
}
