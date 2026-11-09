#version 460
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_demote_to_helper_invocation : require

layout(early_fragment_tests) in;
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
layout(set = 0, binding = 7) uniform texture2DArray fs_img5;
layout(set = 0, binding = 8) uniform sampler fs_samp0;
layout(set = 0, binding = 9) uniform sampler fs_samp1;

float safe_rsq(float x) { return (x > 0.0) ? inversesqrt(x) : 0.0; }
float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }

vec4 load_s1_v4(uint i) { return vec4(s1f(i), s1f(i + 1u), s1f(i + 2u), s1f(i + 3u)); }
vec3 load_s1_v3(uint i) { return vec3(s1f(i), s1f(i + 1u), s1f(i + 2u)); }
vec3 load_s2_v3(uint i) { return vec3(s2f(i), s2f(i + 1u), s2f(i + 2u)); }

float eval_cube_vsm(uint base, vec3 pos_v) {
    vec3 sv = -pos_v + load_s2_v3(base + 20u);
    vec3 nsv = -sv;
    bool is_z = (abs(nsv.z) >= abs(nsv.x)) && (abs(nsv.z) >= abs(nsv.y));
    bool is_y = (abs(nsv.y) >= abs(nsv.x));
    float face = is_z ? ((nsv.z < 0.0) ? 5.0 : 4.0)
               : (is_y ? ((nsv.y < 0.0) ? 3.0 : 2.0)
                       : ((nsv.x < 0.0) ? 1.0 : 0.0));
    float inv_ma = 1.0 / abs(is_z ? (sv.z * -2.0) : (is_y ? (sv.y * -2.0) : (sv.x * -2.0)));
    float sc = is_z ? ((nsv.z < 0.0) ? sv.x : -sv.x)
             : (is_y ? -sv.x : ((nsv.x < 0.0) ? -sv.z : sv.z));
    float tc = is_z ? sv.y : (is_y ? ((nsv.y < 0.0) ? sv.z : -sv.z) : sv.y);
    float layer = fma(floor(face * 0.125), -2.0, face);
    vec4 sm = texture(sampler2DArray(fs_img5, fs_samp1), vec3(sc * inv_ma + 0.5, tc * inv_ma + 0.5, layer));

    float far_d = (-s2f(base + 17u)) / (s2f(base + 16u) + -1.0);
    float dist_n = max(abs(sv.z), max(abs(sv.x), abs(sv.y))) / far_d;
    float lit = (sm.x >= dist_n) ? 1.0 : 0.0;
    float var = max(0.00100000005 / far_d, (-sm.x) * sm.x + sm.y);
    float diff_d = dist_n - sm.x;
    float p_max = max(lit, var / (diff_d * diff_d + var));
    if (((push_data.buf_offsets2.y >> 24u) & 0x02u) != 0u) {
        p_max = smoothstep(0.18, 1.0, p_max);
    }
    return s2f(base + 19u) * (-1.0 + p_max) + 1.0;
}

void main() {
    uint bo0 = push_data.buf_offsets0.x;
    uint d0 = bitfieldExtract(bo0, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(bo0, 8, 8) >> 2u;

    vec2 uv = fs_in_attr1_p.xy;
    vec4 t_depth = texture(sampler2D(fs_img1, fs_samp0), uv);
    vec4 clip_p = vec4(fs_in_attr0_p.xy, t_depth.x, 1.0);
    float inv_w = 1.0 / dot(load_s1_v4(d0 + 12u), clip_p);
    vec3 pos_v = vec3(
        dot(load_s1_v4(d0 + 0u), clip_p),
        dot(load_s1_v4(d0 + 4u), clip_p),
        dot(load_s1_v4(d0 + 8u), clip_p)
    ) * inv_w;

    vec3 L_vec = -pos_v + load_s2_v3(d1 + 0u);
    float d2 = dot(L_vec, L_vec);
    float inv_r2 = s2f(d1 + 3u);
    if (inv_r2 * d2 >= 1.0) {
        demote;
        frag_color1 = vec4(0.0, 0.0, 0.0, 1.0);
        frag_color0 = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec4 t_norm = texture(sampler2D(fs_img0, fs_samp0), uv);
    vec4 t_mask = texture(sampler2D(fs_img2, fs_samp1), uv);
    vec4 t_spec = texture(sampler2D(fs_img3, fs_samp0), uv);
    vec4 t_mat  = texture(sampler2D(fs_img4, fs_samp0), uv);

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

    vec3 L_dir = L_vec * safe_rsq(d2);
    float NdotL = dot(N, L_dir);
    float NdotL_c = clamp(NdotL, 0.0, 1.0);
    float falloff = 1.0 - sqrt(clamp(inv_r2 * d2, 0.0, 1.0));
    float atten = falloff * falloff;
    float shadow = eval_cube_vsm(d1, pos_v);

    vec3 d_col = load_s2_v3(d1 + 4u);
    float l_exp = s2f(d1 + 7u);
    vec3 s_col = load_s2_v3(d1 + 8u);
    float rim_scale = s2f(d1 + 12u);
    float rim_pow = s2f(d1 + 13u);

    float exp_g = exp2(13.0 * t_mat.x);
    float trans = 10.0 * t_mat.y;
    float trans_denom = fma(2.0, trans, 1.0);

    vec3 acc_diff;
    if (trans == 0.0) {
        acc_diff = shadow * (d_col * (atten * NdotL_c));
    } else {
        float LdotNegV_c = clamp(dot(L_dir, neg_V), 0.0, 1.0);
        float back_pow = exp2((rim_pow * 32.0) * log2(LdotNegV_c));
        float wrap = (atten * shadow) * (max(0.0, trans + NdotL) / trans_denom) + trans * (atten * (back_pow * (rim_scale * rim_base)));
        acc_diff = d_col * wrap;
    }

    float spec_exp = max(1.0, (l_exp * exp_g) / (exp_g * 0.0100021362 + l_exp));
    float RdotL_c = clamp(dot(R_v, L_dir), 0.0, 1.0);
    float spec_pow = exp2((0.25 * spec_exp) * log2(RdotL_c)) * (0.125 * spec_exp + 0.25);
    vec3 acc_spec = shadow * (atten * (s_col * spec_pow));

    vec3 f0 = t_spec.xyz * t_spec.xyz;
    frag_color1 = vec4(f0 * (t_mask.x * acc_spec), 1.0);
    frag_color0 = vec4(t_mask.x * (acc_diff - (f0 * acc_diff) * 0.5), 1.0);
}
