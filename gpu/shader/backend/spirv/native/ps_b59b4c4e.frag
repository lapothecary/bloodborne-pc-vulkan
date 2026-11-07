#version 460
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_demote_to_helper_invocation : require

layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 0) out vec4 frag_color0;
layout(location = 1) out vec4 frag_color1;

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
    uint vsharp_base_offset;
    uint tsharp_base_offset;
} push_data;

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { uint data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { uint data[]; } ssbo_2;

layout(set = 0, binding = 2) uniform texture2D fs_img0;
layout(set = 0, binding = 3) uniform texture2D fs_img1;
layout(set = 0, binding = 4) uniform texture2D fs_img2;
layout(set = 0, binding = 5) uniform texture2D fs_img3;
layout(set = 0, binding = 6) uniform texture2D fs_img4;
layout(set = 0, binding = 7) uniform sampler fs_samp0;
layout(set = 0, binding = 8) uniform sampler fs_samp1;

float safe_rsq(float x) { return (x > 0.0) ? inversesqrt(x) : 0.0; }
float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }

vec4 load_s1_v4(uint i) { return vec4(s1f(i), s1f(i + 1u), s1f(i + 2u), s1f(i + 3u)); }
vec3 load_s1_v3(uint i) { return vec3(s1f(i), s1f(i + 1u), s1f(i + 2u)); }
vec3 load_s2_v3(uint i) { return vec3(s2f(i), s2f(i + 1u), s2f(i + 2u)); }

mat2x3 eval_point_light_phong(
    uint base, vec3 pos_v, vec3 N, vec3 neg_V, vec3 R_v,
    float exp_g, float exp_g_scaled, float trans, float trans_denom, float rim_base
) {
    vec3 L_vec = -pos_v + load_s2_v3(base + 0u);
    float d2 = dot(L_vec, L_vec);
    vec3 L_dir = L_vec * safe_rsq(d2);
    float inv_r2 = s2f(base + 3u);
    vec3 d_col = load_s2_v3(base + 4u);
    float l_exp = s2f(base + 7u);
    vec3 s_col = load_s2_v3(base + 8u);
    float rim_scale = s2f(base + 12u);
    float rim_pow = s2f(base + 13u);

    float falloff = 1.0 - sqrt(clamp(inv_r2 * d2, 0.0, 1.0));
    float atten = falloff * falloff;
    float NdotL = dot(N, L_dir);
    float NdotL_c = clamp(NdotL, 0.0, 1.0);

    vec3 diff;
    if (trans == 0.0) {
        diff = d_col * (atten * NdotL_c);
    } else {
        float LdotNegV_c = clamp(dot(L_dir, neg_V), 0.0, 1.0);
        float back_pow = exp2((rim_pow * 32.0) * log2(LdotNegV_c));
        float wrap = atten * (max(0.0, trans + NdotL) / trans_denom) + trans * (atten * (back_pow * (rim_scale * rim_base)));
        diff = d_col * wrap;
    }

    float spec_exp = max(1.0, (l_exp * exp_g) / (exp_g_scaled + l_exp));
    float RdotL_c = clamp(dot(R_v, L_dir), 0.0, 1.0);
    float spec_pow = exp2((0.25 * spec_exp) * log2(RdotL_c)) * (0.125 * spec_exp + 0.25);
    vec3 spec = atten * (s_col * spec_pow);

    return mat2x3(diff, spec);
}

void main() {
    uint bo0 = push_data.buf_offsets0.x;
    uint d0 = bitfieldExtract(bo0, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(bo0, 8, 8) >> 2u;

    vec2 uv = fs_in_attr1_p.xy;
    vec4 t_depth = texture(sampler2D(fs_img0, fs_samp0), uv);
    vec4 clip_p = vec4(fs_in_attr0_p.xy, t_depth.x, 1.0);
    float w_denom = dot(load_s1_v4(d0 + 12u), clip_p);
    vec3 clip_xyz = vec3(
        dot(load_s1_v4(d0 + 0u), clip_p),
        dot(load_s1_v4(d0 + 4u), clip_p),
        dot(load_s1_v4(d0 + 8u), clip_p)
    );

    vec3 neg_pos = clip_xyz * (-1.0 / w_denom);
    bool any_inside = false;
    for (uint i = 0u; i < 9u; ++i) {
        uint b = d1 + i * 16u;
        vec3 dv = neg_pos + load_s2_v3(b);
        if (dot(dv, dv) * s2f(b + 3u) < 1.0) any_inside = true;
    }
    if (!any_inside) {
        frag_color0 = vec4(0.0, 0.0, 0.0, 1.0);
        frag_color1 = vec4(0.0, 0.0, 0.0, 1.0);
        demote;
        return;
    }

    vec4 t_norm = texture(sampler2D(fs_img1, fs_samp0), uv);
    vec4 t_mat  = texture(sampler2D(fs_img2, fs_samp0), uv);
    vec4 t_spec = texture(sampler2D(fs_img3, fs_samp0), uv);
    vec4 t_mask = texture(sampler2D(fs_img4, fs_samp1), uv);

    vec3 pos_v = clip_xyz * (1.0 / w_denom);
    vec3 n_raw = fma(vec3(2.0), t_norm.xyz, vec3(-1.0));
    vec3 n_ts = n_raw * safe_rsq(dot(n_raw, n_raw));
    vec3 N = vec3(
        dot(load_s1_v3(d0 + 16u), n_ts),
        dot(load_s1_v3(d0 + 20u), n_ts),
        dot(load_s1_v3(d0 + 24u), n_ts)
    );

    vec3 neg_V = pos_v * safe_rsq(dot(pos_v, pos_v));
    vec3 R_v = neg_V - N * (2.0 * dot(N, neg_V));
    float rim = max(0.0, 1.0 + dot(-neg_V, N));
    float rim_base = 0.25 * (rim * rim);

    float exp_g = exp2(13.0 * t_mat.x);
    float exp_g_scaled = exp_g * 0.0100021362;
    float trans = 10.0 * t_mat.y;
    float trans_denom = fma(2.0, trans, 1.0);

    vec3 acc_diff = vec3(0.0);
    vec3 acc_spec = vec3(0.0);
    for (uint i = 0u; i < 9u; ++i) {
        mat2x3 l = eval_point_light_phong(d1 + i * 16u, pos_v, N, neg_V, R_v, exp_g, exp_g_scaled, trans, trans_denom, rim_base);
        acc_diff += l[0];
        acc_spec += l[1];
    }

    vec3 f0 = t_spec.xyz * t_spec.xyz;
    frag_color1 = vec4(f0 * (t_mask.x * acc_spec), 1.0);
    frag_color0 = vec4(t_mask.x * (acc_diff - (f0 * acc_diff) * 0.5), 1.0);
}
