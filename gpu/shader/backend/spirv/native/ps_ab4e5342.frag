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

// VFX Dual-Frame Blend + Atmospheric Inscattering & Fog 2-MRT — ps_ab4e5342
layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 3) in vec4 fs_in_attr2_p;
layout(location = 5) flat in vec4 fs_in_attr3_p;

layout(location = 0) out vec4 frag_color0;
layout(location = 1) out vec4 frag_color1;

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { uint data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { uint data[]; } ssbo_2;
layout(set = 0, binding = 2, std430) readonly buffer ssbo_3_buf { uint data[]; } ssbo_3;

layout(set = 0, binding = 3) uniform texture2D fs_img0;
layout(set = 0, binding = 4) uniform sampler fs_samp0;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
float s3f(uint i) { return uintBitsToFloat(ssbo_3.data[i]); }

float safe_rsq(float x) {
    return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0;
}

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
    uint d2 = bitfieldExtract(u0, 16, 8) >> 2u;

    vec4 tex0 = texture(sampler2D(fs_img0, fs_samp0), fs_in_attr0_p.xy);
    vec4 tex1 = texture(sampler2D(fs_img0, fs_samp0), fs_in_attr0_p.zw);
    bool mode2 = (ssbo_1.data[d0 + 95u] == 2u);

    vec3 col0 = eval_overlay_sq(tex0, fs_in_attr1_p.xyz, mode2);
    vec3 col1 = eval_overlay_sq(tex1, fs_in_attr1_p.xyz, mode2);
    float a0 = tex0.w * fs_in_attr1_p.w;
    float a1 = tex1.w * fs_in_attr1_p.w;

    float blend_t = unpackHalf2x16((floatBitsToUint(fs_in_attr3_p.x) >> 16u) & 0xFFFFu).x;
    vec3 col_mix = blend_t * (col1 - col0) + col0;
    float a_mix = clamp(blend_t * (a1 - a0) + a0, 0.0, 1.0);

    vec3 pos_sq = fs_in_attr2_p.xyz * fs_in_attr2_p.xyz;
    float dist4 = sqrt(pos_sq.x + (pos_sq.y + (pos_sq.z + fs_in_attr2_p.w * fs_in_attr2_p.w)));
    float rsq3 = safe_rsq(pos_sq.x + (pos_sq.y + pos_sq.z));
    vec3 view_dir = fs_in_attr2_p.xyz * rsq3;
    float cos_theta = dot(vec3(s3f(d2 + 52u), s3f(d2 + 53u), s3f(d2 + 54u)), view_dir);

    vec3 tint = vec3(s2f(d1 + 3u), s2f(d1 + 7u), s2f(d1 + 11u));
    vec3 lit_rgb = tint * col_mix;
    float luma_bias = -1.0 + (lit_rgb.z * 0.114480004 + (lit_rgb.x * 0.298909992 + 0.586610019 * lit_rgb.y));
    float rad_mod = s3f(d2 + 7u) * luma_bias + 1.0;
    float hgt_mod = s3f(d2 + 19u) * luma_bias + 1.0;

    float dist_xz = sqrt((-fs_in_attr2_p.x) * (-fs_in_attr2_p.x) + (-fs_in_attr2_p.z) * (-fs_in_attr2_p.z));
    float fog_rad = clamp(s3f(d2 + 8u) + s3f(d2 + 8u) * (-exp2(-(s3f(d2 + 10u) * dist_xz + (-(s3f(d2 + 9u) * s3f(d2 + 10u)))))), 0.0, 1.0);
    vec3 rad_col = vec3(s3f(d2 + 4u), s3f(d2 + 5u), s3f(d2 + 6u)) * rad_mod;
    vec3 after_rad = fog_rad * (-col_mix * tint + rad_col) + lit_rgb;

    float fog_hgt = clamp(s3f(d2 + 12u) * clamp(s3f(d2 + 14u) * (s3f(d2 + 15u) - fs_in_attr2_p.y) + (-(s3f(d2 + 13u) * s3f(d2 + 14u))), 0.0, 1.0), 0.0, 1.0);
    vec3 hgt_col = vec3(s3f(d2 + 16u), s3f(d2 + 17u), s3f(d2 + 18u)) * hgt_mod;
    vec3 fogged = fog_hgt * (hgt_col + (-after_rad)) + after_rad;

    float hg_denom = max(9.99999997e-07, (-cos_theta) * s3f(d2 + 38u) + s3f(d2 + 37u));
    float hg_phase = (1.0 / hg_denom) * (s3f(d2 + 36u) * safe_rsq(hg_denom));
    float ray_phase = fma(cos_theta, cos_theta, 1.0);
    vec3 phase_col = vec3(s3f(d2 + 40u), s3f(d2 + 41u), s3f(d2 + 42u)) * ray_phase
                   + vec3(s3f(d2 + 44u), s3f(d2 + 45u), s3f(d2 + 46u)) * hg_phase;

    float ext_r = exp2((1.44269502 * (s3f(d2 + 55u) * (dist4 * (-s3f(d2 + 24u))))) * 1.44269502);
    float ext_g = exp2(((s3f(d2 + 55u) * (dist4 * (-s3f(d2 + 25u)))) * 1.44269502) * 1.44269502);
    float ext_b = exp2(((s3f(d2 + 55u) * (dist4 * (-s3f(d2 + 26u)))) * 1.44269502) * 1.44269502);
    vec3 ext = vec3(ext_r, ext_g, ext_b);

    vec3 ext_tint = vec3(s3f(d2 + 28u), s3f(d2 + 29u), s3f(d2 + 30u));
    vec3 inscat_scale = s3f(d2 + 31u) * (vec3(s3f(d2 + 32u), s3f(d2 + 33u), s3f(d2 + 34u)) * vec3(s3f(d2 + 48u), s3f(d2 + 49u), s3f(d2 + 50u)));
    vec3 atmos_rgb = s3f(d2 + 51u) * (((ext_tint * fogged) * ext + (-ext * phase_col + phase_col) * inscat_scale) - fogged) + fogged;

    vec3 final_rgb = clamp((0.0 < a_mix) ? atmos_rgb : col_mix, 0.0, 1.0);
    float final_a = clamp(a_mix, 0.0, 1.0);

    frag_color1 = vec4(
        s1f(d0 + 88u) * final_rgb.x,
        s1f(d0 + 89u) * final_rgb.y,
        s1f(d0 + 90u) * final_rgb.z,
        clamp(final_a * s1f(d0 + 91u), 0.0, 1.0)
    );
    frag_color0 = vec4(final_rgb, final_a);
}
