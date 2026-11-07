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

// Deferred 2-Point-Light Omnidirectional VSM Cubemap Shadow PBR 2-MRT — ps_b5b52d63
layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;

layout(location = 0) out vec4 frag_color0;
layout(location = 1) out vec4 frag_color1;

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { uint data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { uint data[]; } ssbo_2;

layout(set = 0, binding = 2) uniform texture2D fs_img0;
layout(set = 0, binding = 3) uniform texture2D fs_img1;
layout(set = 0, binding = 4) uniform texture2DArray fs_img2;
layout(set = 0, binding = 5) uniform texture2D fs_img3;
layout(set = 0, binding = 6) uniform texture2D fs_img4;
layout(set = 0, binding = 7) uniform texture2D fs_img5;
layout(set = 0, binding = 8) uniform texture2DArray fs_img6;
layout(set = 0, binding = 9) uniform sampler fs_samp0;
layout(set = 0, binding = 10) uniform sampler fs_samp1;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
vec4 load_s1_v4(uint i) { return vec4(s1f(i), s1f(i + 1u), s1f(i + 2u), s1f(i + 3u)); }

float safe_rsq(float x) {
    return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0;
}

vec3 shadow_cube_coords(vec3 v) {
    vec3 nr = -v;
    vec3 ar = abs(nr);
    bool z_maj = (ar.z >= ar.x) && (ar.z >= ar.y);
    bool y_maj = (ar.y >= ar.x);
    float ma = z_maj ? (v.z * -2.0) : (y_maj ? (v.y * -2.0) : (v.x * -2.0));
    float inv_ma = 1.0 / abs(ma);
    float sc = z_maj ? ((nr.z < 0.0) ? v.x : -v.x) : (y_maj ? -v.x : ((nr.x < 0.0) ? -v.z : v.z));
    float tc = z_maj ? v.y : (y_maj ? ((nr.y < 0.0) ? v.z : -v.z) : v.y);
    float face = z_maj ? ((nr.z < 0.0) ? 5.0 : 4.0) : (y_maj ? ((nr.y < 0.0) ? 3.0 : 2.0) : ((nr.x < 0.0) ? 1.0 : 0.0));
    float layer = fma(floor(face * 0.125), -2.0, face);
    return vec3(sc * inv_ma + 0.5, tc * inv_ma + 0.5, layer);
}

void eval_shadowed_point_light(
    uint base, vec4 sh_tex, vec3 sh_vec,
    vec3 pos_v, vec3 V, vec3 neg_V, vec3 N,
    float gloss, float exp_g, float trans, float rim_term,
    out vec3 out_diff, out vec3 out_spec
) {
    vec3 L_vec = -pos_v + vec3(s2f(base + 0u), s2f(base + 1u), s2f(base + 2u));
    float d2 = dot(L_vec, L_vec);
    vec3 L_dir = L_vec * safe_rsq(d2);
    vec3 H = (V + L_dir) * safe_rsq(dot(V + L_dir, V + L_dir));

    float sh_scale = (-s2f(base + 17u)) * (1.0 / (s2f(base + 16u) + -1.0));
    float sh_var = max(0.00100000005 / sh_scale, (-sh_tex.x) * sh_tex.x + sh_tex.y);
    float sh_dist = max(abs(sh_vec.z), max(abs(sh_vec.x), abs(sh_vec.y))) * (1.0 / sh_scale);
    float cheb = max((sh_tex.x >= sh_dist) ? 1.0 : 0.0, sh_var * (1.0 / ((sh_dist - sh_tex.x) * (sh_dist - sh_tex.x) + sh_var)));
    float shadow = s2f(base + 19u) * (-1.0 + cheb) + 1.0;

    float NdotL = dot(N, L_dir);
    float NdotL_c = clamp(NdotL, 0.0, 1.0);
    float atten_d = max(0.0, (1.0 / d2) - s2f(base + 3u));
    float atten_s = 1.0 - clamp(s2f(base + 3u) * d2, 0.0, 1.0);

    vec3 l_col = vec3(s2f(base + 4u), s2f(base + 5u), s2f(base + 6u));
    if (trans == 0.0) {
        out_diff = atten_d * (shadow * (l_col * NdotL_c));
    } else {
        float wrap = (atten_d * shadow) * (max(0.0, trans + NdotL) * (1.0 / fma(2.0, trans, 1.0)));
        float back_cos = clamp(dot(L_dir, neg_V), 0.0, 1.0);
        float back = trans * (atten_d * (exp2((s2f(base + 13u) * 32.0) * log2(back_cos)) * (s2f(base + 12u) * rim_term)));
        out_diff = l_col * (wrap + back);
    }

    float l_pow = s2f(base + 7u);
    float spec_exp = max(1.0, (l_pow * exp_g) * (1.0 / (exp_g * 0.0100021362 + l_pow)));
    float blinn = exp2(spec_exp * log2(clamp(dot(H, N), 0.0, 1.0)));
    float nrm_fade = exp2(0.0999755859 * log2(NdotL_c));
    vec3 s_col = vec3(s2f(base + 8u), s2f(base + 9u), s2f(base + 10u));
    out_spec = (atten_s * (shadow * ((s_col * blinn) * nrm_fade))) * (0.125 * spec_exp + 0.25);
}

void main() {
    uint u0 = push_data.buf_offsets0.x;
    uint d0 = bitfieldExtract(u0, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(u0, 8, 8) >> 2u;

    vec2 uv = fs_in_attr1_p.xy;
    vec4 tex_depth = texture(sampler2D(fs_img0, fs_samp0), uv);
    vec4 tex_nrm   = texture(sampler2D(fs_img1, fs_samp0), uv);

    vec4 clip_p = vec4(fs_in_attr0_p.xy, tex_depth.x, 1.0);
    float inv_w = 1.0 / dot(load_s1_v4(d0 + 12u), clip_p);
    vec3 pos_u = vec3(
        dot(load_s1_v4(d0 + 0u), clip_p),
        dot(load_s1_v4(d0 + 4u), clip_p),
        dot(load_s1_v4(d0 + 8u), clip_p)
    );
    vec3 pos_v = pos_u * inv_w;

    vec3 sh_vec0 = -pos_u * inv_w + vec3(s2f(d1 + 20u), s2f(d1 + 21u), s2f(d1 + 22u));
    vec4 sh_tex0 = texture(sampler2DArray(fs_img2, fs_samp1), shadow_cube_coords(sh_vec0));
    vec4 tex_mat = texture(sampler2D(fs_img3, fs_samp0), uv);
    vec4 tex_spec = texture(sampler2D(fs_img4, fs_samp0), uv);
    vec4 tex_ao = texture(sampler2D(fs_img5, fs_samp1), uv);
    vec3 sh_vec1 = -pos_u * inv_w + vec3(s2f(d1 + 44u), s2f(d1 + 45u), s2f(d1 + 46u));
    vec4 sh_tex1 = texture(sampler2DArray(fs_img6, fs_samp1), shadow_cube_coords(sh_vec1));

    vec3 N_w = fma(vec3(2.0), tex_nrm.xyz, vec3(-1.0));
    N_w *= safe_rsq(dot(N_w, N_w));
    vec3 N = vec3(
        dot(load_s1_v4(d0 + 16u).xyz, N_w),
        dot(load_s1_v4(d0 + 20u).xyz, N_w),
        dot(load_s1_v4(d0 + 24u).xyz, N_w)
    );

    float pos_rsq = safe_rsq(dot(pos_v, pos_v));
    vec3 V = (-pos_v) * pos_rsq;
    vec3 neg_V = pos_v * pos_rsq;

    float rim = max(0.0, 1.0 + dot(-neg_V, N));
    float rim_term = 0.25 * (rim * rim);
    float exp_g = exp2(13.0 * tex_mat.x);
    float trans = 10.0 * tex_mat.y;

    vec3 diff0, spec0, diff1, spec1;
    eval_shadowed_point_light(d1 + 0u,  sh_tex0, sh_vec0, pos_v, V, neg_V, N, tex_mat.x, exp_g, trans, rim_term, diff0, spec0);
    eval_shadowed_point_light(d1 + 24u, sh_tex1, sh_vec1, pos_v, V, neg_V, N, tex_mat.x, exp_g, trans, rim_term, diff1, spec1);

    vec3 diff_sum = diff0 + diff1;
    vec3 spec_sum = spec0 + spec1;
    vec3 spec_sq = tex_spec.xyz * tex_spec.xyz;

    frag_color1 = vec4(spec_sq * (tex_ao.x * spec_sum), 1.0);
    frag_color0 = vec4(tex_ao.x * (diff_sum - (spec_sq * diff_sum) * 0.5), 1.0);
}
