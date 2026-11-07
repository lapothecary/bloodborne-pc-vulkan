#version 460
#extension GL_EXT_samplerless_texture_functions : require
// ps_bd1d09f0 — Clean vectorized forward-lit 1-MRT transparent/foliage PBR shader

layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 2) in vec4 fs_in_attr2_p;
layout(location = 3) in vec4 fs_in_attr3_p;
layout(location = 4) in vec4 fs_in_attr4_p;
layout(location = 5) in vec4 fs_in_attr5_p;

layout(location = 0) out vec4 frag_color0;

layout(push_constant) uniform AuxData {
    float xoffset, yoffset, xscale, yscale;
    uvec4 ud_regs0, ud_regs1, ud_regs2, ud_regs3;
    uvec4 buf_offsets0, buf_offsets1;
    uvec2 buf_offsets2;
    uint motion_param, scene_size;
} push_data;

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { uint data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { uint data[]; } ssbo_2;
layout(set = 0, binding = 2, std430) readonly buffer ssbo_3_buf { float data[]; } ssbo_3;
layout(set = 0, binding = 3, std430) readonly buffer ssbo_4_buf { uint data[]; } ssbo_4;
layout(set = 0, binding = 4, std430) readonly buffer ssbo_5_buf { uint data[]; } ssbo_5;
layout(set = 0, binding = 5, std430) readonly buffer ssbo_6_buf { uint data[]; } ssbo_6;

layout(set = 0, binding = 6) uniform texture2D fs_img0;
layout(set = 0, binding = 7) uniform texture2D fs_img1;
layout(set = 0, binding = 8) uniform texture2D fs_img2;
layout(set = 0, binding = 9) uniform texture2D fs_img3;
layout(set = 0, binding = 10) uniform texture2DArray fs_img4;
layout(set = 0, binding = 11) uniform texture2D fs_img5;
layout(set = 0, binding = 12) uniform texture2DArray fs_img6;
layout(set = 0, binding = 13) uniform texture2D fs_img7;
layout(set = 0, binding = 14) uniform sampler fs_samp0;
layout(set = 0, binding = 15) uniform sampler fs_samp1;
layout(set = 0, binding = 16) uniform sampler fs_samp2;
layout(set = 0, binding = 17) uniform sampler fs_samp3;
layout(set = 0, binding = 18) uniform sampler fs_samp4;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
float s4f(uint i) { return uintBitsToFloat(ssbo_4.data[i]); }
float s5f(uint i) { return uintBitsToFloat(ssbo_5.data[i]); }
float s6f(uint i) { return uintBitsToFloat(ssbo_6.data[i]); }
vec3 s4v3(uint i) { return vec3(s4f(i), s4f(i + 1u), s4f(i + 2u)); }

float safe_rsq(float x) { return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0; }
vec3 safe_norm(vec3 v) { return v * safe_rsq(dot(v, v)); }
vec2 xform_uv(uint off, vec2 uv) {
    return vec2(uv.x * s1f(off) + s1f(off + 2u), uv.y * s1f(off + 1u) + s1f(off + 3u));
}
vec3 cube_uv(vec3 r) {
    bool iz = (abs(r.z) >= abs(r.x)) && (abs(r.z) >= abs(r.y));
    bool iy = (abs(r.y) >= abs(r.x));
    float ma = iz ? (r.z * 2.0) : (iy ? (r.y * 2.0) : (r.x * 2.0));
    float inv_ma = 1.0 / abs(ma);
    float face = iz ? ((r.z < 0.0) ? 5.0 : 4.0)
                    : (iy ? ((r.y < 0.0) ? 3.0 : 2.0) : ((r.x < 0.0) ? 1.0 : 0.0));
    float sc = iz ? ((r.z < 0.0) ? -r.x : r.x)
                  : (iy ? r.x : ((r.x < 0.0) ? r.z : -r.z));
    float tc = iz ? -r.y : (iy ? ((r.y < 0.0) ? -r.z : r.z) : -r.y);
    return vec3(sc * inv_ma + 0.5, tc * inv_ma + 0.5, fma(floor(face * 0.125), -2.0, face));
}
float eval_sh(uint base, vec3 n) {
    float d = n.z * ssbo_3.data[base + 2u] + (n.x * ssbo_3.data[base] + n.y * ssbo_3.data[base + 1u]);
    float c = ssbo_3.data[base + 3u];
    return max(0.0, max(c + d, (c - d) * 0.85));
}

void main() {
    uvec4 bo0 = push_data.buf_offsets0;
    uint d0 = (bo0.x >> 2u) & 0x3Fu;
    uint d1 = (bo0.x >> 10u) & 0x3Fu;
    uint d2 = (bo0.x >> 18u) & 0x3Fu;
    uint d3 = (bo0.x >> 26u) & 0x3Fu;
    uint d4 = (bo0.y >> 2u) & 0x3Fu;
    uint d5 = (bo0.y >> 10u) & 0x3Fu;

    bool is_front = gl_FrontFacing;

    float inv_w = 1.0 / fs_in_attr3_p.w;
    vec2 screen_uv = vec2(fma(0.5, fs_in_attr3_p.x * inv_w, 0.5), fma(-0.5, fs_in_attr3_p.y * inv_w, 0.5));
    vec4 tex0 = texture(sampler2D(fs_img0, fs_samp0), xform_uv(d0 + 12u, fs_in_attr5_p.xy));
    vec4 tex1 = texture(sampler2D(fs_img1, fs_samp1), xform_uv(d0 + 8u, fs_in_attr5_p.xy));
    vec4 tex2 = texture(sampler2D(fs_img2, fs_samp1), xform_uv(d0 + 4u, fs_in_attr5_p.xy));
    vec4 tex3 = texture(sampler2D(fs_img3, fs_samp2), screen_uv);

    vec3 n_unit = safe_norm(fs_in_attr0_p.xyz);
    vec3 N_vert = is_front ? n_unit : -n_unit;
    vec3 T_vert = safe_norm(fs_in_attr1_p.xyz);
    vec3 B_vert = safe_norm(cross(fs_in_attr0_p.xyz, fs_in_attr1_p.xyz)) * fs_in_attr1_p.w;

    float nx = fma(2.0, tex0.x, -1.0) * fma(2.0, tex0.w, -1.0);
    float ny = fma(2.0, tex0.y, -1.0);
    float nz = sqrt(1.0 - clamp(nx * nx + ny * ny, 0.0, 1.0));
    vec3 N = safe_norm(B_vert * nx + (T_vert * ny + N_vert * nz));

    vec3 V = safe_norm(fs_in_attr4_p.xyz);
    float NdotV = dot(N, V);
    vec3 R = -V + N * (NdotV * 2.0);

    float gloss = tex1.x * (s2f(d1 + 8u) * 0.00390625);
    float mip_lod = min(6.0, 6.5 - gloss * 6.5);
    vec3 c_uv = cube_uv(is_front ? R : vec3(R.x, abs(R.y), R.z));
    vec4 env0 = textureLod(sampler2DArray(fs_img4, fs_samp2), c_uv, mip_lod);
    vec4 ssr = textureLod(sampler2D(fs_img5, fs_samp3), screen_uv, mip_lod);
    vec4 env1 = textureLod(sampler2DArray(fs_img6, fs_samp2), c_uv, mip_lod);
    vec4 tex7 = texture(sampler2D(fs_img7, fs_samp4), xform_uv(d0, fs_in_attr5_p.xy));

    uint inst_off = uint(fs_in_attr0_p.w * 1.00001526) * 13u;
    uint sh_base = (inst_off + 8u) * 4u + d2;
    float sh_shadow = ssbo_3.data[(inst_off + 12u) * 4u + d2];
    float ao = ((0.0 != fs_in_attr2_p.x) ? fs_in_attr2_p.x : 1.0) * ((s2f(d1 + 12u) * tex3.x + 1.0) - s2f(d1 + 12u));

    bool mono_spec = (tex2.z == 0.0) && (tex2.y == 0.0);
    vec3 spec = vec3(tex2.x, mono_spec ? tex2.x : tex2.y, mono_spec ? tex2.x : tex2.z);
    float fres = exp2(5.0 * log2(1.0 - clamp(NdotV, 0.0, 1.0))) * clamp(((NdotV + ao) * (NdotV + ao) + ao) + -1.0, 0.0, 1.0);
    vec3 spec_fres = (fres - spec * fres) * (gloss * gloss) + spec;

    float horizon = clamp(dot(R, N_vert) + 1.0, 0.0, 1.0);
    float env_mask = ao * ((horizon * horizon) * (1.0 - ssr.w));
    float e_w0 = ssbo_3.data[sh_base];
    float e_w1 = ssbo_3.data[sh_base + 1u];
    float e_w2 = ssbo_3.data[sh_base + 2u];
    vec3 env_rgb = ao * ssr.xyz + (e_w0 * (-e_w1 * env1.xyz + e_w2 * env0.xyz) + e_w1 * env1.xyz) * env_mask;
    vec3 sh_rgb = vec3(eval_sh(sh_base + 4u, N), eval_sh(sh_base + 8u, N), eval_sh(sh_base + 12u, N));

    float nl0 = max(dot(s4v3(d3), N), -dot(s4v3(d3), N) * 0.85);
    float nl1 = max(dot(s4v3(d3 + 4u), N), -dot(s4v3(d3 + 4u), N) * 0.85);
    vec3 diff_acc = (s4v3(d3 + 16u) * sh_shadow) * ((-spec) * clamp(nl0, 0.0, 1.0) + clamp(nl0, 0.0, 1.0))
                  + s4v3(d3 + 20u) * ((-spec) * clamp(nl1, 0.0, 1.0) + clamp(nl1, 0.0, 1.0));

    float spec_exp_raw = exp2(13.0 * gloss);
    float spec_denom_bias = spec_exp_raw * 0.0100021362;
    vec3 H0 = safe_norm(V + s4v3(d3));
    vec3 H1 = safe_norm(V + s4v3(d3 + 4u));
    float exp0 = max(1.0, (s4f(d3 + 3u) * spec_exp_raw) * (1.0 / (spec_denom_bias + s4f(d3 + 3u))));
    float exp1 = max(1.0, (s4f(d3 + 7u) * spec_exp_raw) * (1.0 / (spec_denom_bias + s4f(d3 + 7u))));
    float nh0 = exp2(exp0 * log2(clamp(dot(H0, N), 0.0, 1.0)));
    float nh1 = exp2(exp1 * log2(clamp(dot(H1, N), 0.0, 1.0)));
    float vh0 = exp2(5.0 * log2(1.0 - clamp(dot(V, H0), 0.0, 1.0)));
    float vh1 = exp2(5.0 * log2(1.0 - clamp(dot(V, H1), 0.0, 1.0)));
    vec3 spec_acc = (clamp(nl0 * 16.0, 0.0, 1.0) * (nh0 * ((s4v3(d3 + 32u) * sh_shadow) * (0.125 * exp0 + 0.25)))) * ((-spec) * vh0 + (spec + vh0))
                  + ((-spec) * vh1 + (spec + vh1)) * (clamp(nl1 * 16.0, 0.0, 1.0) * (nh1 * (s4v3(d3 + 36u) * (exp1 * 0.125 + 0.25))));

    if (ssbo_4.data[d3 + 64u] > 0u) {
        float hw2 = fma(0.5, dot(s4v3(d3 + 8u), N), 0.5);
        float c2 = clamp(hw2 * hw2, 0.0, 1.0);
        diff_acc = s4v3(d3 + 24u) * ((-spec) * c2 + c2) + diff_acc;
        vec3 H2 = safe_norm(V + s4v3(d3 + 8u));
        float exp2_v = max(1.0, (s4f(d3 + 11u) * spec_exp_raw) * (1.0 / (spec_denom_bias + s4f(d3 + 11u))));
        float nh2 = exp2(exp2_v * log2(clamp(dot(H2, N), 0.0, 1.0)));
        float vh2 = exp2(5.0 * log2(1.0 - clamp(dot(V, H2), 0.0, 1.0)));
        spec_acc = (nh2 * (s4v3(d3 + 40u) * (0.125 * exp2_v + 0.25))) * ((-spec) * vh2 + (spec + vh2)) + spec_acc;
    }
    if (ssbo_4.data[d3 + 65u] > 0u) {
        float hw3 = fma(0.5, dot(s4v3(d3 + 12u), N), 0.5);
        float c3 = clamp(hw3 * hw3, 0.0, 1.0);
        diff_acc = s4v3(d3 + 28u) * ((-spec) * c3 + c3) + diff_acc;
        vec3 H3 = safe_norm(V + s4v3(d3 + 12u));
        float exp3_v = max(1.0, (s4f(d3 + 15u) * spec_exp_raw) * (1.0 / (spec_denom_bias + s4f(d3 + 15u))));
        float nh3 = exp2(exp3_v * log2(clamp(dot(H3, N), 0.0, 1.0)));
        float vh3 = exp2(5.0 * log2(1.0 - clamp(dot(V, H3), 0.0, 1.0)));
        spec_acc = (nh3 * (s4v3(d3 + 44u) * (0.125 * exp3_v + 0.25))) * ((-spec) * vh3 + (spec + vh3)) + spec_acc;
    }

    vec3 alb = vec3(s2f(d1), s2f(d1 + 1u), s2f(d1 + 2u)) * tex7.xyz;
    vec3 env_tint = vec3(s2f(d1 + 4u), s2f(d1 + 5u), s2f(d1 + 6u)) * s5f(d4 + 12u);
    vec3 lit = alb * (ao * ((-spec) * sh_rgb + sh_rgb) + diff_acc) + (spec_fres * env_rgb + spec_acc) * env_tint;

    uint tint_base = (inst_off + 7u) * 4u + d2;
    vec3 inst_tint = vec3(ssbo_3.data[tint_base], ssbo_3.data[tint_base + 1u], ssbo_3.data[tint_base + 2u]);
    vec3 col_tinted = lit * inst_tint;
    float luma_bias = -1.0 + (col_tinted.z * 0.114480004 + (col_tinted.x * 0.298909992 + 0.586610019 * col_tinted.y));

    float dist_xz = sqrt(fs_in_attr4_p.x * fs_in_attr4_p.x + fs_in_attr4_p.z * fs_in_attr4_p.z);
    float fog_rad = clamp(s6f(d5 + 8u) + s6f(d5 + 8u) * (-exp2(-(s6f(d5 + 10u) * (dist_xz - s6f(d5 + 9u))))), 0.0, 1.0);
    vec3 col_rad = fog_rad * (-col_tinted + vec3(s6f(d5 + 4u), s6f(d5 + 5u), s6f(d5 + 6u)) * (s6f(d5 + 7u) * luma_bias + 1.0)) + col_tinted;

    float fog_h = clamp(s6f(d5 + 12u) * clamp(s6f(d5 + 14u) * ((s6f(d5 + 15u) - fs_in_attr4_p.y) - s6f(d5 + 13u)), 0.0, 1.0), 0.0, 1.0);
    vec3 col_final = fog_h * (vec3(s6f(d5 + 16u), s6f(d5 + 17u), s6f(d5 + 18u)) * (s6f(d5 + 19u) * luma_bias + 1.0) - col_rad) + col_rad;

    frag_color0 = vec4(col_final, (tex7.w * fs_in_attr2_p.w) * ssbo_3.data[tint_base + 3u]);
}
