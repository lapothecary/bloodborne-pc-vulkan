#version 460
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_demote_to_helper_invocation : require

// Bloodborne Native GLSL 460 — 0x0ef50047 (4-Point-Light Phong PBR + Cubemap VSM Shadow)
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
    uint motion_param;
    uint scene_size;
} push_data;

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { uint data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { uint data[]; } ssbo_2;

layout(set = 0, binding = 2) uniform texture2D fs_img0;
layout(set = 0, binding = 3) uniform texture2D fs_img1;
layout(set = 0, binding = 4) uniform texture2D fs_img2;
layout(set = 0, binding = 5) uniform texture2D fs_img3;
layout(set = 0, binding = 6) uniform texture2D fs_img4;
layout(set = 0, binding = 7) uniform texture2DArray fs_img5;
layout(set = 0, binding = 8) uniform texture2DArray fs_img6;
layout(set = 0, binding = 9) uniform texture2DArray fs_img7;
layout(set = 0, binding = 10) uniform texture2DArray fs_img8;
layout(set = 0, binding = 11) uniform sampler fs_samp0;
layout(set = 0, binding = 12) uniform sampler fs_samp1;

float safe_rsq(float x) {
    return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0;
}

float s1f(uint idx) { return uintBitsToFloat(ssbo_1.data[idx]); }
float s2f(uint idx) { return uintBitsToFloat(ssbo_2.data[idx]); }
vec3 s1_v3(uint idx) { return vec3(s1f(idx), s1f(idx + 1u), s1f(idx + 2u)); }
vec4 s1_v4(uint idx) { return vec4(s1f(idx), s1f(idx + 1u), s1f(idx + 2u), s1f(idx + 3u)); }
vec3 s2_v3(uint idx) { return vec3(s2f(idx), s2f(idx + 1u), s2f(idx + 2u)); }

vec3 cube_uv_face(vec3 sv) {
    vec3 d = -sv;
    vec3 ad = abs(d);
    bool is_z = (ad.z >= ad.x) && (ad.z >= ad.y);
    bool is_y = (ad.y >= ad.x);
    float face = is_z ? ((d.z < 0.0) ? 5.0 : 4.0)
               : (is_y ? ((d.y < 0.0) ? 3.0 : 2.0)
                       : ((d.x < 0.0) ? 1.0 : 0.0));
    float inv_2m = 1.0 / abs(is_z ? (sv.z * -2.0) : (is_y ? (sv.y * -2.0) : (sv.x * -2.0)));
    float u = (is_z ? ((d.z < 0.0) ? sv.x : -sv.x)
             : (is_y ? -sv.x : ((d.x < 0.0) ? -sv.z : sv.z))) * inv_2m + 0.5;
    float v = (is_z ? sv.y
             : (is_y ? ((d.y < 0.0) ? sv.z : -sv.z) : sv.y)) * inv_2m + 0.5;
    float layer = fma(floor(face * 0.125), -2.0, face);
    return vec3(u, v, layer);
}

mat2x3 eval_vsm_point_light(uint base, vec3 L_vec, float d2, vec4 sm, vec3 pos, vec3 N, vec3 V_dir, vec3 R_v, float exp_13, float trans, float rim_factor) {
    vec3 L = L_vec * safe_rsq(d2);
    float NdotL = dot(N, L);
    float att = 1.0 - sqrt(clamp(s2f(base + 3u) * d2, 0.0, 1.0));
    float att2 = att * att;

    vec3 sv = s2_v3(base + 20u) - pos;
    float sh_range = (-s2f(base + 17u)) / (s2f(base + 16u) - 1.0);
    float sh_dist = max(abs(sv.z), max(abs(sv.x), abs(sv.y))) / sh_range;
    float var = max(0.00100000005 / sh_range, (-sm.x * sm.x) + sm.y);
    float p_max = max((sm.x >= sh_dist) ? 1.0 : 0.0, var / ((sh_dist - sm.x) * (sh_dist - sm.x) + var));
    if (((push_data.buf_offsets2.y >> 24u) & 0x02u) != 0u) {
        p_max = smoothstep(0.18, 1.0, p_max);
    }
    float sh_factor = s2f(base + 19u) * (p_max - 1.0) + 1.0;

    float spec_pow = max(1.0, (s2f(base + 7u) * exp_13) / (exp_13 * 0.0100021362 + s2f(base + 7u)));
    float spec_brdf = exp2((0.25 * spec_pow) * log2(clamp(dot(R_v, L), 0.0, 1.0))) * (0.125 * spec_pow + 0.25);
    vec3 spec_contrib = (att2 * sh_factor) * (s2_v3(base + 8u) * spec_brdf);

    vec3 light_col = s2_v3(base + 4u);
    vec3 diff_contrib;
    if (trans == 0.0) {
        diff_contrib = sh_factor * (light_col * (att2 * clamp(NdotL, 0.0, 1.0)));
    } else {
        float wrap = (att2 * sh_factor) * (max(0.0, trans + NdotL) / fma(2.0, trans, 1.0));
        float trans_spec = exp2((s2f(base + 13u) * 32.0) * log2(clamp(dot(L, V_dir), 0.0, 1.0))) * (s2f(base + 12u) * rim_factor);
        diff_contrib = light_col * (wrap + trans * (att2 * trans_spec));
    }
    return mat2x3(diff_contrib, spec_contrib);
}

void main() {
    uint d0 = bitfieldExtract(push_data.buf_offsets0.x, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(push_data.buf_offsets0.x, 8, 8) >> 2u;

    vec2 uv = fs_in_attr1_p.xy;
    float depth = texture(sampler2D(fs_img1, fs_samp0), uv).x;

    vec4 clip_p = vec4(fs_in_attr0_p.xy, depth, 1.0);
    float inv_w = 1.0 / dot(s1_v4(d0 + 12u), clip_p);
    vec3 pos = vec3(
        dot(s1_v4(d0 + 0u), clip_p),
        dot(s1_v4(d0 + 4u), clip_p),
        dot(s1_v4(d0 + 8u), clip_p)
    ) * inv_w;

    vec3 L_vec0 = s2_v3(d1 + 0u)  - pos;
    vec3 L_vec1 = s2_v3(d1 + 24u) - pos;
    vec3 L_vec2 = s2_v3(d1 + 48u) - pos;
    vec3 L_vec3 = s2_v3(d1 + 72u) - pos;
    float d2_0 = dot(L_vec0, L_vec0);
    float d2_1 = dot(L_vec1, L_vec1);
    float d2_2 = dot(L_vec2, L_vec2);
    float d2_3 = dot(L_vec3, L_vec3);

    bool in0 = (s2f(d1 + 3u)  * d2_0) < 1.0;
    bool in1 = (s2f(d1 + 27u) * d2_1) < 1.0;
    bool in2 = (s2f(d1 + 51u) * d2_2) < 1.0;
    bool in3 = (s2f(d1 + 75u) * d2_3) < 1.0;
    if (!in0 && !in1 && !in2 && !in3) {
        demote;
        frag_color1 = vec4(0.0, 0.0, 0.0, 1.0);
        frag_color0 = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec4 t_norm = texture(sampler2D(fs_img0, fs_samp0), uv);
    vec3 n_raw = fma(vec3(2.0), t_norm.xyz, vec3(-1.0));
    vec3 n_ts = n_raw * safe_rsq(dot(n_raw, n_raw));
    vec3 N = vec3(
        dot(s1_v3(d0 + 16u), n_ts),
        dot(s1_v3(d0 + 20u), n_ts),
        dot(s1_v3(d0 + 24u), n_ts)
    );

    vec3 V_dir = pos * safe_rsq(dot(pos, pos));
    vec3 R_v = V_dir - N * (2.0 * dot(N, V_dir));
    float rim = max(0.0, 1.0 + dot(-V_dir, N));
    float rim_factor = 0.25 * (rim * rim);

    float ao = texture(sampler2D(fs_img2, fs_samp1), uv).x;
    vec3 albedo = texture(sampler2D(fs_img3, fs_samp0), uv).xyz;
    vec3 albedo2 = albedo * albedo;
    vec4 t_mat = texture(sampler2D(fs_img4, fs_samp0), uv);

    vec4 sm0 = in0 ? texture(sampler2DArray(fs_img5, fs_samp1), cube_uv_face(s2_v3(d1 + 20u) - pos)) : vec4(1.0);
    vec4 sm1 = in1 ? texture(sampler2DArray(fs_img6, fs_samp1), cube_uv_face(s2_v3(d1 + 44u) - pos)) : vec4(1.0);
    vec4 sm2 = in2 ? texture(sampler2DArray(fs_img7, fs_samp1), cube_uv_face(s2_v3(d1 + 68u) - pos)) : vec4(1.0);
    vec4 sm3 = in3 ? texture(sampler2DArray(fs_img8, fs_samp1), cube_uv_face(s2_v3(d1 + 92u) - pos)) : vec4(1.0);

    float exp_13 = exp2(13.0 * t_mat.x);
    float trans = 10.0 * t_mat.y;

    mat2x3 r0 = eval_vsm_point_light(d1 + 0u,  L_vec0, d2_0, sm0, pos, N, V_dir, R_v, exp_13, trans, rim_factor);
    mat2x3 r1 = eval_vsm_point_light(d1 + 24u, L_vec1, d2_1, sm1, pos, N, V_dir, R_v, exp_13, trans, rim_factor);
    mat2x3 r2 = eval_vsm_point_light(d1 + 48u, L_vec2, d2_2, sm2, pos, N, V_dir, R_v, exp_13, trans, rim_factor);
    mat2x3 r3 = eval_vsm_point_light(d1 + 72u, L_vec3, d2_3, sm3, pos, N, V_dir, R_v, exp_13, trans, rim_factor);

    vec3 diff_acc = r0[0] + r1[0] + r2[0] + r3[0];
    vec3 spec_acc = r0[1] + r1[1] + r2[1] + r3[1];

    frag_color1 = vec4(albedo2 * (ao * spec_acc), 1.0);
    frag_color0 = vec4(ao * (diff_acc - (albedo2 * diff_acc) * 0.5), 1.0);
}
