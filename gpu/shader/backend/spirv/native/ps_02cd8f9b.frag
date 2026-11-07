#version 460
#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 0) out vec4 frag_color0;
layout(location = 1) out vec4 frag_color1;

layout(push_constant) uniform AuxData {
    float vtx_x_offset;
    float vtx_y_offset;
    float vtx_x_scale;
    float vtx_y_scale;
    uvec4 vs_cb_offset;
    uvec4 fs_cb_offset;
    uvec4 cs_cb_offset;
    uvec4 hs_cb_offset;
    uvec4 buf_offsets0;
    uvec4 buf_offsets1;
    uvec2 gs_cb_offset;
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
vec4 load_s2_v4(uint i) { return vec4(s2f(i), s2f(i + 1u), s2f(i + 2u), s2f(i + 3u)); }

mat2x3 eval_spot_light(
    uint base, vec3 pos_v, vec3 N, vec3 neg_V, vec3 R_v,
    float exp_g, float trans, float trans_denom, float rim_base
) {
    float pz = dot(load_s2_v4(base + 24u), vec4(pos_v, 1.0));
    if (0.0 >= pz) {
        return mat2x3(vec3(0.0), vec3(0.0));
    }
    float px = dot(load_s2_v4(base + 16u), vec4(pos_v, 1.0));
    float py = dot(load_s2_v4(base + 20u), vec4(pos_v, 1.0));
    float pw = dot(load_s2_v4(base + 28u), vec4(pos_v, 1.0));

    vec3 L_vec = -pos_v + vec3(s2f(base + 0u), s2f(base + 1u), s2f(base + 2u));
    float d2 = dot(L_vec, L_vec);
    vec3 L_dir = L_vec * safe_rsq(d2);
    float NdotL = dot(L_dir, N);
    float NdotL_c = clamp(NdotL, 0.0, 1.0);

    float inv_pw = 1.0 / pw;
    float nx = px * inv_pw;
    float ny = py * inv_pw;
    float cone_c = clamp(1.0 - (ny * ny + nx * nx), 0.0, 1.0);

    float r = s2f(base + 3u);
    vec3 d_col = vec3(s2f(base + 4u), s2f(base + 5u), s2f(base + 6u));
    float l_exp = s2f(base + 7u);
    vec3 s_col = vec3(s2f(base + 8u), s2f(base + 9u), s2f(base + 10u));
    float rim_scale = s2f(base + 12u);
    float rim_pow = s2f(base + 13u);
    float spec_exp = max(1.0, (l_exp * exp_g) / (exp_g * 0.0100021362 + l_exp));
    float spec_norm = 0.125 * spec_exp + 0.25;

    float atten_d;
    vec3 spec;
    if (s2f(base + 15u) > 0.0) {
        float inv_r2 = 1.0 / (r * r);
        atten_d = max(0.0, (1.0 / d2) - inv_r2) * cone_c;
        float atten_s = (1.0 - clamp(d2 * inv_r2, 0.0, 1.0)) * cone_c;
        vec3 H = (-neg_V + L_dir) * safe_rsq(dot(-neg_V + L_dir, -neg_V + L_dir));
        float NdotH_c = clamp(dot(H, N), 0.0, 1.0);
        float spec_pow = exp2(spec_exp * log2(NdotH_c));
        float nl_pow = exp2(0.0999755859 * log2(NdotL_c));
        spec = spec_norm * (atten_s * (nl_pow * (s_col * spec_pow)));
    } else {
        float z_c = clamp(1.0 - pw * (1.0 / r), 0.0, 1.0);
        atten_d = (cone_c * cone_c) * (z_c * z_c);
        float RdotL_c = clamp(dot(L_dir, R_v), 0.0, 1.0);
        float spec_pow = exp2((0.25 * spec_exp) * log2(RdotL_c)) * spec_norm;
        spec = atten_d * (s_col * spec_pow);
    }

    vec3 diff;
    if (trans == 0.0) {
        diff = atten_d * (d_col * NdotL_c);
    } else {
        float LdotNegV_c = clamp(dot(L_dir, neg_V), 0.0, 1.0);
        float back_pow = exp2((rim_pow * 32.0) * log2(LdotNegV_c));
        float wrap = atten_d * (max(0.0, trans + NdotL) / trans_denom) + trans * (atten_d * (rim_scale * rim_base * back_pow));
        diff = d_col * wrap;
    }

    return mat2x3(diff, spec);
}

void main() {
    uint bo0 = push_data.buf_offsets0.x;
    uint d0 = bitfieldExtract(bo0, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(bo0, 8, 8) >> 2u;

    vec2 uv = fs_in_attr1_p.xy;
    vec4 t_depth = texture(sampler2D(fs_img0, fs_samp0), uv);
    vec4 t_norm  = texture(sampler2D(fs_img1, fs_samp0), uv);
    vec4 t_mat   = texture(sampler2D(fs_img2, fs_samp0), uv);
    vec4 t_spec  = texture(sampler2D(fs_img3, fs_samp0), uv);
    vec4 t_mask  = texture(sampler2D(fs_img4, fs_samp1), uv);

    vec4 clip_p = vec4(fs_in_attr0_p.xy, t_depth.x, 1.0);
    float inv_w = 1.0 / dot(load_s1_v4(d0 + 12u), clip_p);
    vec3 pos_v = vec3(
        dot(load_s1_v4(d0 + 0u), clip_p),
        dot(load_s1_v4(d0 + 4u), clip_p),
        dot(load_s1_v4(d0 + 8u), clip_p)
    ) * inv_w;

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
    float trans = 10.0 * t_mat.y;
    float trans_denom = fma(2.0, trans, 1.0);

    mat2x3 l0 = eval_spot_light(d1 + 0u,  pos_v, N, neg_V, R_v, exp_g, trans, trans_denom, rim_base);
    mat2x3 l1 = eval_spot_light(d1 + 32u, pos_v, N, neg_V, R_v, exp_g, trans, trans_denom, rim_base);
    mat2x3 l2 = eval_spot_light(d1 + 64u, pos_v, N, neg_V, R_v, exp_g, trans, trans_denom, rim_base);
    vec3 acc_diff = l0[0] + l1[0] + l2[0];
    vec3 acc_spec = l0[1] + l1[1] + l2[1];

    vec3 f0 = t_spec.xyz * t_spec.xyz;
    frag_color1 = vec4(f0 * (t_mask.x * acc_spec), 1.0);
    frag_color0 = vec4(t_mask.x * (acc_diff - (f0 * acc_diff) * 0.5), 1.0);
}
