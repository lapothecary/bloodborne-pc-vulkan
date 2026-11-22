// SPDX-License-Identifier: GPL-2.0-or-later
// Pack #5: Native Vulkan 1.3 GLSL 460 replacement for ps 0x00000000133eb139
// (gxdecal Deferred\Decal_DifSpc.ppo — 5-MRT Deferred PBR Blood/Surface Decal).
#version 460
#extension GL_EXT_demote_to_helper_invocation : require

layout(location = 0) in vec4 fs_in_attr0_p;

layout(location = 0) out vec4 frag_color0;
layout(location = 1) out vec4 frag_color1;
layout(location = 2) out vec4 frag_color2;
layout(location = 3) out vec4 frag_color3;
layout(location = 4) out vec4 frag_color4;

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
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { uint data[]; } ssbo_2;
layout(set = 0, binding = 2, std430) readonly buffer ssbo_3_buf { uint data[]; } ssbo_3;

layout(set = 0, binding = 3) uniform texture2D fs_img0;
layout(set = 0, binding = 4) uniform texture2D fs_img1;
layout(set = 0, binding = 5) uniform texture2D fs_img2;
layout(set = 0, binding = 6) uniform texture2D fs_img3;
layout(set = 0, binding = 7) uniform texture2D fs_img4;
layout(set = 0, binding = 8) uniform texture2DArray fs_img5;
layout(set = 0, binding = 9) uniform texture2DArray fs_img6;

layout(set = 0, binding = 10) uniform sampler fs_samp0;
layout(set = 0, binding = 11) uniform sampler fs_samp1;
layout(set = 0, binding = 12) uniform sampler fs_samp2;
layout(set = 0, binding = 13) uniform sampler fs_samp3;
layout(set = 0, binding = 14) uniform sampler fs_samp4;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
float s3f(uint i) { return uintBitsToFloat(ssbo_3.data[i]); }

vec4 s1v4(uint i) {
    return vec4(s1f(i), s1f(i + 1u), s1f(i + 2u), s1f(i + 3u));
}

vec3 safe_normalize(vec3 v) {
    float len2 = dot(v, v);
    float inv = (len2 > 0.0 && len2 <= 3.40282347e+38)
                    ? inversesqrt(min(max(len2, 1.17549435e-38), 3.40282347e+38))
                    : 0.0;
    return v * inv;
}

vec3 cube_dir_to_2darray(vec3 r) {
    vec3 ar = abs(r);
    bool z_dom = (ar.z >= ar.x) && (ar.z >= ar.y);
    bool y_dom = ar.y >= ar.x;
    float ma = z_dom ? (r.z * 2.0) : (y_dom ? (r.y * 2.0) : (r.x * 2.0));
    float sc = z_dom ? ((r.z < 0.0) ? -r.x : r.x)
                     : (y_dom ? r.x : ((r.x < 0.0) ? r.z : -r.z));
    float tc = z_dom ? -r.y : (y_dom ? ((r.y < 0.0) ? -r.z : r.z) : -r.y);
    float face = z_dom ? ((r.z < 0.0) ? 5.0 : 4.0)
                       : (y_dom ? ((r.y < 0.0) ? 3.0 : 2.0)
                                : ((r.x < 0.0) ? 1.0 : 0.0));
    float inv_ma = 1.0 / abs(ma);
    return vec3(sc * inv_ma + 0.5, tc * inv_ma + 0.5, face);
}

void main() {
    const uint b0 = push_data.buf_offsets0.x;
    const uint d0 = (b0 & 0xFFu) >> 2u;
    const uint d1 = ((b0 >> 8u) & 0xFFu) >> 2u;
    const uint d2 = ((b0 >> 16u) & 0xFFu) >> 2u;

    const vec2 ndc = fs_in_attr0_p.xy * (1.0 / fs_in_attr0_p.w);
    const vec2 screen_uv = vec2(fma(0.5, ndc.x, 0.5), fma(-0.5, ndc.y, 0.5));
    const float depth = texture(sampler2D(fs_img0, fs_samp0), screen_uv).x;

    const vec4 clip_p = vec4(ndc.x, ndc.y, depth, 1.0);
    const float loc_w = dot(s1v4(d0 + 12u), clip_p);
    const float inv_loc_w = 1.0 / loc_w;
    const float loc_x = dot(s1v4(d0 + 0u), clip_p) * inv_loc_w;
    const float loc_y = dot(s1v4(d0 + 4u), clip_p) * inv_loc_w;
    const float loc_z = dot(s1v4(d0 + 8u), clip_p) * inv_loc_w;

    const bool in_box = !(loc_y < 0.0 || loc_y > 1.0 ||
                          loc_z < 0.0 || loc_z > 1.0 ||
                          loc_x < 0.0 || loc_x > 1.0);
    if (!in_box) {
        demote;
    }

    const vec2 decal_uv = vec2(loc_x, loc_z);
    const vec4 albedo_s = texture(sampler2D(fs_img1, fs_samp1), decal_uv);

    const float fade_lo = clamp(loc_y * s1f(d0 + 97u) - s1f(d0 + 96u) * s1f(d0 + 97u), 0.0, 1.0);
    const float fade_hi = clamp(loc_y * s1f(d0 + 99u) - s1f(d0 + 98u) * s1f(d0 + 99u), 0.0, 1.0);
    const float edge_fade = fade_lo - fade_lo * fade_hi;
    const float alpha = s2f(d1 + 3u) * (edge_fade * (s1f(d0 + 83u) * albedo_s.a));
    const float out_alpha = s1f(d0 + 70u) * alpha;

    if (!in_box || !(0.0 < out_alpha)) {
        demote;
    }

    const vec3 n_raw = texture(sampler2D(fs_img2, fs_samp2), screen_uv).xyz;
    const float shin_raw = texture(sampler2D(fs_img3, fs_samp3), decal_uv).x;
    const vec3 spec_raw = texture(sampler2D(fs_img4, fs_samp3), decal_uv).xyz;

    if (helperInvocationEXT()) {
        return;
    }

    const float wp_w = dot(s1v4(d0 + 28u), clip_p);
    const float inv_wp_w = 1.0 / wp_w;
    const vec4 wp = vec4(dot(s1v4(d0 + 16u), clip_p) * inv_wp_w,
                         dot(s1v4(d0 + 20u), clip_p) * inv_wp_w,
                         dot(s1v4(d0 + 24u), clip_p) * inv_wp_w,
                         1.0);
    const vec3 view_pos = vec3(dot(s1v4(d0 + 32u), wp),
                               dot(s1v4(d0 + 36u), wp),
                               dot(s1v4(d0 + 40u), wp));
    const vec3 cam_pos = vec3(s1f(d0 + 35u), s1f(d0 + 39u), s1f(d0 + 43u));

    const vec3 V = safe_normalize(cam_pos - view_pos);
    const vec3 N = safe_normalize(fma(vec3(2.0), n_raw, vec3(-1.0)));
    const float NoV = dot(N, V);
    const vec3 R = N * (NoV * 2.0) - V;

    const float shin = s1f(d0 + 87u) * shin_raw;
    const float ibl_lod = min(6.0, 5.5 - shin * 6.5);
    const vec3 cube_uv = cube_dir_to_2darray(R);
    const vec3 ibl_a = textureLod(sampler2DArray(fs_img5, fs_samp4), cube_uv, ibl_lod).xyz;
    const vec3 ibl_b = textureLod(sampler2DArray(fs_img6, fs_samp4), cube_uv, ibl_lod).xyz;

    const float p_exp = exp2(13.0 * shin);
    const float p_norm = p_exp / (30.0 + p_exp);
    const float one_minus_nov = 1.0 - clamp(NoV, 0.0, 1.0);
    const float log_omv = log2(one_minus_nov);
    const float fresnel_diff = p_norm * exp2(log_omv * fma(4.0, p_norm, 1.0));
    const float fresnel_spec = exp2(5.0 * log_omv);

    const bool spec_mono = (spec_raw.z == 0.0) && (spec_raw.y == 0.0);
    const vec3 spec_in = spec_mono ? vec3(spec_raw.x) : spec_raw;
    const vec3 f0 = vec3(s1f(d0 + 84u), s1f(d0 + 85u), s1f(d0 + 86u)) * spec_in;

    const vec3 f_ibl = f0 + fresnel_spec * (max(f0, vec3(shin)) - f0);
    const vec3 ibl_blend = mix(s2f(d1 + 5u) * ibl_b, s2f(d1 + 6u) * ibl_a, s2f(d1 + 4u));
    const vec3 ibl_col = ibl_blend * (1.0 / (4.0 - 3.0 * shin));

    const vec4 n4 = vec4(N, 1.0);
    const vec3 sh_amb = max(vec3(0.0), vec3(dot(vec4(s3f(d2 + 0u), s3f(d2 + 1u), s3f(d2 + 2u), s3f(d2 + 3u)), n4),
                                            dot(vec4(s3f(d2 + 4u), s3f(d2 + 5u), s3f(d2 + 6u), s3f(d2 + 7u)), n4),
                                            dot(vec4(s3f(d2 + 8u), s3f(d2 + 9u), s3f(d2 + 10u), s3f(d2 + 11u)), n4)));
    const vec3 f_diff = f0 + fresnel_diff - f0 * fresnel_diff;
    const vec3 diff_rgb = vec3(s1f(d0 + 80u), s1f(d0 + 81u), s1f(d0 + 82u)) * albedo_s.rgb * alpha;

    frag_color0 = vec4(f_ibl * ibl_col, out_alpha);
    frag_color1 = vec4(shin, 0.0, 1.0, out_alpha);
    frag_color2 = vec4(diff_rgb, out_alpha);
    frag_color3 = vec4(sqrt(f0), out_alpha);
    frag_color4 = vec4(sh_amb * (1.0 - f_diff), out_alpha);
}
