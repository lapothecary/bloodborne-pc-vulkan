#version 460
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_demote_to_helper_invocation : require
// ps_c31bd920 — Clean vectorized forward-lit 1-MRT lightmapped + IBL/SSR PBR shader with 4-tap CSM, wrap/SSS, and dither cutout
// Early cutout demote + helperInvocationEXT() exit skips 4-tap 4K CSM, 2 IBL cubemaps, SSR, lightmap spec, wrap/SSS, 4 lights, and fog.

layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 2) in vec4 fs_in_attr2_p;
layout(location = 3) in vec4 fs_in_attr3_p;
layout(location = 5) in vec4 fs_in_attr4_p;
layout(location = 6) in vec4 fs_in_attr5_p;

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
layout(set = 0, binding = 2, std430) readonly buffer ssbo_3_buf { uint data[]; } ssbo_3;
layout(set = 0, binding = 3, std430) readonly buffer ssbo_4_buf { uint data[]; } ssbo_4;
layout(set = 0, binding = 4, std430) readonly buffer ssbo_5_buf { float data[]; } ssbo_5;
layout(set = 0, binding = 5, std430) readonly buffer ssbo_6_buf { uint data[]; } ssbo_6;
layout(set = 0, binding = 6, std430) readonly buffer ssbo_7_buf { uint data[]; } ssbo_7;
layout(set = 0, binding = 7, std430) readonly buffer ssbo_8_buf { uint data[]; } ssbo_8;
layout(set = 0, binding = 8, std430) readonly buffer ssbo_9_buf { uint data[]; } ssbo_9;
layout(set = 0, binding = 9, std430) readonly buffer srt_flatbuf_buf { uint data[]; } srt_flatbuf;

layout(set = 0, binding = 10) uniform texture2D fs_img0;
layout(set = 0, binding = 11) uniform texture2D fs_img1;
layout(set = 0, binding = 12) uniform texture2D fs_img2;
layout(set = 0, binding = 13) uniform texture2D fs_img3;
layout(set = 0, binding = 14) uniform texture2D fs_img4;
layout(set = 0, binding = 15) uniform texture2D fs_img5;
layout(set = 0, binding = 16) uniform texture2D fs_img6;
layout(set = 0, binding = 17) uniform texture2D fs_img7;
layout(set = 0, binding = 18) uniform texture2DArray fs_img8;
layout(set = 0, binding = 19) uniform texture2D fs_img9;
layout(set = 0, binding = 20) uniform texture2DArray fs_img10;
layout(set = 0, binding = 21) uniform texture2D fs_img11;
layout(set = 0, binding = 22) uniform sampler fs_samp0;
layout(set = 0, binding = 23) uniform sampler fs_samp1;
layout(set = 0, binding = 24) uniform sampler fs_samp2;
layout(set = 0, binding = 25) uniform sampler fs_samp3;
layout(set = 0, binding = 26) uniform samplerShadow fs_samp4;
layout(set = 0, binding = 27) uniform sampler fs_samp5;
layout(set = 0, binding = 28) uniform sampler fs_samp6;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
float s3f(uint i) { return uintBitsToFloat(ssbo_3.data[i]); }
float s4f(uint i) { return uintBitsToFloat(ssbo_4.data[i]); }
float s6f(uint i) { return uintBitsToFloat(ssbo_6.data[i]); }
float s7f(uint i) { return uintBitsToFloat(ssbo_7.data[i]); }
float s8f(uint i) { return uintBitsToFloat(ssbo_8.data[i]); }
float s9f(uint i) { return uintBitsToFloat(ssbo_9.data[i]); }
vec4 s1v4(uint i) { return vec4(s1f(i), s1f(i + 1u), s1f(i + 2u), s1f(i + 3u)); }
vec3 s6v3(uint i) { return vec3(s6f(i), s6f(i + 1u), s6f(i + 2u)); }
vec3 s8v3(uint i) { return vec3(s8f(i), s8f(i + 1u), s8f(i + 2u)); }

float safe_rsq(float x) { return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0; }
vec3 safe_norm(vec3 v) { return v * safe_rsq(dot(v, v)); }
vec2 xform_uv(uint off, vec2 uv) {
    return vec2(uv.x * s3f(off) + s3f(off + 2u), uv.y * s3f(off + 1u) + s3f(off + 3u));
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

void main() {
    uvec4 bo0 = push_data.buf_offsets0;
    // Reference binding 9 (srt_flatbuf) so pre-opt interface matches raw PS4 SPV (29 descs);
    // spirv-opt folds `(& 0u)` away so zero runtime SSBO load is emitted.
    uint d0 = ((bo0.x >> 2u) & 0x3Fu) | (srt_flatbuf.data[0] & 0u);
    uint d1 = (bo0.x >> 10u) & 0x3Fu;
    uint d2 = (bo0.x >> 18u) & 0x3Fu;
    uint d3 = (bo0.x >> 26u) & 0x3Fu;
    uint d4 = (bo0.y >> 2u) & 0x3Fu;
    uint d5 = (bo0.y >> 10u) & 0x3Fu;
    uint d6 = (bo0.y >> 18u) & 0x3Fu;
    uint d7 = (bo0.y >> 26u) & 0x3Fu;
    uint d8 = (bo0.z >> 2u) & 0x3Fu;

    float inv_w = 1.0 / fs_in_attr4_p.w;
    vec3 ndc = fs_in_attr4_p.xyz * inv_w;
    vec2 screen_uv = vec2(fma(0.5, ndc.x, 0.5), fma(-0.5, ndc.y, 0.5));
    vec2 dither_uv = vec2(0.25 * s1f(d0 + 4u) * screen_uv.x, 0.25 * s1f(d0 + 5u) * screen_uv.y);

    // Issue all implicit-LOD textures in one parallel wave (no serial stall on live pixels,
    // exact 2x2 quad derivatives preserved), then early-exit before 4-tap CSM + 2 cubemaps + SSR + ALU.
    vec4 tex0 = texture(sampler2D(fs_img0, fs_samp0), dither_uv);
    vec4 tex10 = texture(sampler2D(fs_img11, fs_samp6), xform_uv(d2, fs_in_attr3_p.xy));
    vec4 tex1 = texture(sampler2D(fs_img1, fs_samp1), xform_uv(d2 + 12u, fs_in_attr3_p.xy));
    vec4 tex2 = texture(sampler2D(fs_img2, fs_samp2), xform_uv(d2 + 8u, fs_in_attr3_p.xy));
    vec4 tex3 = texture(sampler2D(fs_img3, fs_samp3), fs_in_attr3_p.zw);
    vec4 tex4 = texture(sampler2D(fs_img4, fs_samp2), xform_uv(d2 + 4u, fs_in_attr3_p.xy));
    vec4 tex5 = texture(sampler2D(fs_img5, fs_samp0), screen_uv);
    vec4 tex6 = texture(sampler2D(fs_img7, fs_samp3), fs_in_attr3_p.zw);

    uint inst_off = uint(fs_in_attr0_p.w * 1.00001526) * 13u;
    uint tint_base = (inst_off + 7u) * 4u + d4;
    float alpha_base = tex10.w * fs_in_attr2_p.w;
    float out_alpha = alpha_base * ssbo_5.data[tint_base + 3u];
    bool do_demote = (s4f(d3 + 20u) > s4f(d3 + 21u) * (tex0.w - 0.5) + alpha_base) || (out_alpha == 0.0);
    if (do_demote) {
        demote;
        return;
    }

    vec3 n_unit = safe_norm(fs_in_attr0_p.xyz);
    vec3 N_vert = gl_FrontFacing ? n_unit : -n_unit;
    vec3 T_vert = safe_norm(fs_in_attr1_p.xyz);
    vec3 B_vert = safe_norm(cross(fs_in_attr0_p.xyz, fs_in_attr1_p.xyz) * fs_in_attr1_p.w);

    float nx = fma(2.0, tex1.x, -1.0) * fma(2.0, tex1.w, -1.0);
    float ny = fma(2.0, tex1.y, -1.0);
    float nz = sqrt(1.0 - clamp(nx * nx + ny * ny, 0.0, 1.0));
    vec3 N = safe_norm(B_vert * nx + (T_vert * ny + N_vert * nz));
    vec3 V = safe_norm(fs_in_attr5_p.xyz);
    float NdotV = dot(N, V);
    vec3 R = -V + N * (NdotV * 2.0);

    // View-space position & 4-tap CSM shadow lookup
    vec4 ndc4 = vec4(ndc, 1.0);
    float inv_vw = 1.0 / dot(s1v4(d0 + 48u), ndc4);
    vec3 view_pos = vec3(dot(s1v4(d0 + 36u), ndc4), dot(s1v4(d0 + 40u), ndc4), dot(s1v4(d0 + 44u), ndc4)) * inv_vw;
    float view_z = view_pos.z;

    uint c_idx = ((s1f(d0 + 176u) < view_z) ? 1u : 0u)
               + ((view_z > s1f(d0 + 177u)) ? 1u : 0u)
               + ((view_z > s1f(d0 + 178u)) ? 1u : 0u);
    uint c_base = c_idx << 2u;
    float sx = dot(s1v4((c_base + 28u) * 4u + d0), ndc4);
    float sy = dot(s1v4((c_base + 29u) * 4u + d0), ndc4);
    float sz = dot(s1v4((c_base + 30u) * 4u + d0), ndc4);
    float sw = dot(s1v4((c_base + 31u) * 4u + d0), ndc4);
    float inv_sw = 1.0 / sw;
    float dref = sz * inv_sw;
    float pcf_o0 = s1f(d0 + 74u);
    float pcf_o1 = s1f(d0 + 75u);
    float sh0 = textureLod(sampler2DShadow(fs_img6, fs_samp4), vec3((sx + sw * pcf_o1) * inv_sw, (sy - sw * pcf_o0) * inv_sw, dref), 0.0);
    float sh1 = textureLod(sampler2DShadow(fs_img6, fs_samp4), vec3((sx + sw * pcf_o0) * inv_sw, (sy + sw * pcf_o1) * inv_sw, dref), 0.0);
    float sh2 = textureLod(sampler2DShadow(fs_img6, fs_samp4), vec3((sx - sw * pcf_o0) * inv_sw, (sy - sw * pcf_o1) * inv_sw, dref), 0.0);
    float sh3 = textureLod(sampler2DShadow(fs_img6, fs_samp4), vec3((sx - sw * pcf_o1) * inv_sw, (sy + sw * pcf_o0) * inv_sw, dref), 0.0);
    float pcf = 0.25 * (sh3 + (sh2 + (sh1 + sh0)));

    float gloss = tex2.x * (s4f(d3 + 8u) * 0.00390625);
    float mip_lod = min(6.0, 6.5 - gloss * 6.5);
    vec3 c_uv = cube_uv(R);
    vec4 env0 = textureLod(sampler2DArray(fs_img8, fs_samp0), c_uv, mip_lod);
    vec4 ssr = textureLod(sampler2D(fs_img9, fs_samp5), screen_uv, mip_lod);
    vec4 env1 = textureLod(sampler2DArray(fs_img10, fs_samp0), c_uv, mip_lod);

    uint sh_base = (inst_off + 8u) * 4u + d4;
    float sh_inst = ssbo_5.data[(inst_off + 12u) * 4u + d4];
    float e_w0 = ssbo_5.data[sh_base];
    float e_w1 = ssbo_5.data[sh_base + 1u];
    float e_w2 = ssbo_5.data[sh_base + 2u];
    float e_w3 = ssbo_5.data[sh_base + 3u];
    float ao = ((fs_in_attr2_p.x != 0.0) ? fs_in_attr2_p.x : 1.0) * ((s4f(d3 + 12u) * tex5.x + 1.0) - s4f(d3 + 12u));

    float shadow_fade = sh_inst * clamp(s1f(d0 + 73u) * (s1f(d0 + 72u) - length(view_pos)), 0.0, 1.0);
    vec3 shadow_rgb = (pcf - 1.0) * (vec3(s1f(d0 + 76u), s1f(d0 + 77u), s1f(d0 + 78u)) * shadow_fade) + vec3(sh_inst);
    vec3 env_sh_shadow = sh_inst * fma(vec3(e_w3), shadow_rgb - 1.0, vec3(1.0));
    vec3 shadow_lm = tex6.w * shadow_rgb;

    bool mono_spec = (tex4.z == 0.0) && (tex4.y == 0.0);
    vec3 spec = vec3(tex4.x, mono_spec ? tex4.x : tex4.y, mono_spec ? tex4.x : tex4.z);
    float spec_exp_raw = exp2(13.0 * gloss);
    float spec_denom_bias = spec_exp_raw * 0.0100021362;

    // Environment cubemap + SSR reflection
    float fres = exp2(5.0 * log2(1.0 - clamp(NdotV, 0.0, 1.0))) * clamp(((NdotV + ao) * (NdotV + ao) + ao) + -1.0, 0.0, 1.0);
    vec3 spec_fres = (fres - spec * fres) * (gloss * gloss) + spec;
    float horizon = clamp(dot(R, N_vert) + 1.0, 0.0, 1.0);
    float env_mask = ao * ((horizon * horizon) * (1.0 - ssr.w));
    vec3 env_rgb = ao * ssr.xyz + (e_w0 * (-e_w1 * env1.xyz + e_w2 * env0.xyz) + e_w1 * env1.xyz) * env_mask;

    // Directional lightmap diffuse & specular (blended with IBL via env_weight)
    float env_weight = s7f(d6 + 36u);
    vec3 lm_dir = safe_norm(fma(vec3(2.0), tex3.xyz, vec3(-1.0)));
    float lm_factor = clamp(s2f(d1 + 41u) + tex3.w, 0.0, 1.0);
    vec3 lm_col = s2f(d1 + 43u) * (tex6.xyz * (s6v3(d5) * tex6.xyz));
    vec3 lm_dir_col = lm_factor * lm_col;
    vec3 lm_amb_col = lm_col - lm_dir_col;
    float lm_nl = clamp(dot(lm_dir, N), 0.0, 1.0);
    vec3 lm_diff_raw = ((-spec) * lm_nl + lm_nl) * lm_dir_col + lm_amb_col;
    vec3 lm_diff = ao * ((-spec) * lm_diff_raw + lm_diff_raw);

    float lm_area = max(0.00100000005, exp2(s2f(d1 + 42u) * log2(lm_factor))) * (100.0 - s2f(d1 + 40u)) + s2f(d1 + 40u);
    float lm_exp = max(1.0, (spec_exp_raw * lm_area) * (1.0 / (spec_denom_bias + lm_area)));
    vec3 H_lm = safe_norm(V + lm_dir);
    float lm_nh = exp2(lm_exp * log2(clamp(dot(H_lm, N), 0.0, 1.0)));
    float lm_vh = exp2(5.0 * log2(1.0 - clamp(dot(V, H_lm), 0.0, 1.0)));
    float lm_spec_scale = (lm_nh * (0.125 * lm_exp + 0.25)) * (ao * (1.0 - env_weight));
    vec3 lm_spec = ((-spec) * lm_vh + (spec + lm_vh)) * (lm_dir_col * lm_spec_scale);
    vec3 env_plus_lm_spec = (env_weight * spec_fres) * env_rgb + lm_spec;

    // Direct Lights 0 & 1 (standard or subsurface-wrap)
    vec3 L0 = s8v3(d7);
    vec3 L1 = s8v3(d7 + 4u);
    float wrap = s7f(d6 + 4u) * fs_in_attr2_p.y;
    vec3 diff_acc;
    if (wrap == 0.0) {
        float nl0 = clamp(dot(L0, N), 0.0, 1.0);
        float nl1 = clamp(dot(L1, N), 0.0, 1.0);
        diff_acc = (s8v3(d7 + 16u) * shadow_lm) * ((-spec) * nl0 + nl0)
                 + s8v3(d7 + 20u) * ((-spec) * nl1 + nl1);
    } else {
        float inv_wrap = 1.0 / fma(2.0, wrap, 1.0);
        float w0 = max(0.0, wrap + dot(L0, N)) * inv_wrap;
        float w1 = max(0.0, wrap + dot(L1, N)) * inv_wrap;
        diff_acc = (s8v3(d7 + 16u) * shadow_lm) * w0 + s8v3(d7 + 20u) * w1;
    }

    vec3 H0 = safe_norm(V + L0);
    vec3 H1 = safe_norm(V + L1);
    float exp0 = max(1.0, (s8f(d7 + 3u) * spec_exp_raw) * (1.0 / (spec_denom_bias + s8f(d7 + 3u))));
    float exp1 = max(1.0, (s8f(d7 + 7u) * spec_exp_raw) * (1.0 / (spec_denom_bias + s8f(d7 + 7u))));
    float nh0 = exp2(exp0 * log2(clamp(dot(H0, N), 0.0, 1.0)));
    float nh1 = exp2(exp1 * log2(clamp(dot(H1, N), 0.0, 1.0)));
    float vh0 = exp2(5.0 * log2(1.0 - clamp(dot(V, H0), 0.0, 1.0)));
    float vh1 = exp2(5.0 * log2(1.0 - clamp(dot(V, H1), 0.0, 1.0)));
    vec3 spec_acc = (clamp(dot(L0, N) * 16.0, 0.0, 1.0) * (nh0 * ((s8v3(d7 + 32u) * shadow_lm) * (exp0 * 0.125 + 0.25)))) * ((-spec) * vh0 + (spec + vh0))
                  + ((-spec) * vh1 + (spec + vh1)) * (clamp(dot(L1, N) * 16.0, 0.0, 1.0) * (nh1 * (s8v3(d7 + 36u) * (0.125 * exp1 + 0.25))));

    if (ssbo_8.data[d7 + 64u] > 0u) {
        vec3 L2 = s8v3(d7 + 8u);
        float hw2 = fma(0.5, dot(L2, N), 0.5);
        float c2 = clamp(hw2 * hw2, 0.0, 1.0);
        diff_acc = s8v3(d7 + 24u) * ((-spec) * c2 + c2) + diff_acc;
        vec3 H2 = safe_norm(V + L2);
        float exp2_v = max(1.0, (s8f(d7 + 11u) * spec_exp_raw) * (1.0 / (spec_denom_bias + s8f(d7 + 11u))));
        float nh2 = exp2(exp2_v * log2(clamp(dot(H2, N), 0.0, 1.0)));
        float vh2 = exp2(5.0 * log2(1.0 - clamp(dot(V, H2), 0.0, 1.0)));
        spec_acc = (nh2 * (s8v3(d7 + 40u) * (0.125 * exp2_v + 0.25))) * ((-spec) * vh2 + (spec + vh2)) + spec_acc;
    }
    if (ssbo_8.data[d7 + 65u] > 0u) {
        vec3 L3 = s8v3(d7 + 12u);
        float hw3 = fma(0.5, dot(L3, N), 0.5);
        float c3 = clamp(hw3 * hw3, 0.0, 1.0);
        diff_acc = s8v3(d7 + 28u) * ((-spec) * c3 + c3) + diff_acc;
        vec3 H3 = safe_norm(V + L3);
        float exp3_v = max(1.0, (s8f(d7 + 15u) * spec_exp_raw) * (1.0 / (spec_denom_bias + s8f(d7 + 15u))));
        float nh3 = exp2(exp3_v * log2(clamp(dot(H3, N), 0.0, 1.0)));
        float vh3 = exp2(5.0 * log2(1.0 - clamp(dot(V, H3), 0.0, 1.0)));
        spec_acc = (nh3 * (s8v3(d7 + 44u) * (0.125 * exp3_v + 0.25))) * ((-spec) * vh3 + (spec + vh3)) + spec_acc;
    }

    // Backlit forward-scatter subsurface translucency when wrap > 0
    if (wrap > 0.0) {
        float ndv = max(0.0, 1.0 + NdotV);
        float sss_scale = 0.25 * (ndv * ndv);
        float b0 = exp2(32.0 * log2(clamp(dot(L0, -V), 0.0, 1.0))) * sss_scale;
        float b1 = exp2(32.0 * log2(clamp(dot(L1, -V), 0.0, 1.0))) * sss_scale;
        diff_acc += wrap * (s8v3(d7 + 16u) * b0 + s8v3(d7 + 20u) * b1);
    }

    vec3 alb = vec3(s4f(d3), s4f(d3 + 1u), s4f(d3 + 2u)) * tex10.xyz;
    vec3 env_tint = vec3(s4f(d3 + 4u), s4f(d3 + 5u), s4f(d3 + 6u)) * s2f(d1 + 12u);
    vec3 lit = alb * (lm_diff * env_sh_shadow + diff_acc)
             + env_tint * (env_plus_lm_spec * env_sh_shadow + spec_acc);

    vec3 inst_tint = vec3(ssbo_5.data[tint_base], ssbo_5.data[tint_base + 1u], ssbo_5.data[tint_base + 2u]);
    vec3 col_tinted = lit * inst_tint;
    float luma_bias = -1.0 + (col_tinted.z * 0.114480004 + (col_tinted.x * 0.298909992 + 0.586610019 * col_tinted.y));

    float dist_xz = sqrt(fs_in_attr5_p.x * fs_in_attr5_p.x + fs_in_attr5_p.z * fs_in_attr5_p.z);
    float fog_rad = clamp(s9f(d8 + 8u) + s9f(d8 + 8u) * (-exp2(-(s9f(d8 + 10u) * (dist_xz - s9f(d8 + 9u))))), 0.0, 1.0);
    vec3 col_rad = fog_rad * (-col_tinted + vec3(s9f(d8 + 4u), s9f(d8 + 5u), s9f(d8 + 6u)) * (s9f(d8 + 7u) * luma_bias + 1.0)) + col_tinted;

    float fog_h = clamp(s9f(d8 + 12u) * clamp(s9f(d8 + 14u) * ((s9f(d8 + 15u) - fs_in_attr5_p.y) - s9f(d8 + 13u)), 0.0, 1.0), 0.0, 1.0);
    vec3 col_final = fog_h * (vec3(s9f(d8 + 16u), s9f(d8 + 17u), s9f(d8 + 18u)) * (s9f(d8 + 19u) * luma_bias + 1.0) - col_rad) + col_rad;

    frag_color0 = vec4(col_final, out_alpha);
}
