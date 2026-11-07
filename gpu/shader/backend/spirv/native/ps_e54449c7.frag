#version 460
#extension GL_EXT_samplerless_texture_functions : require
// ps_e54449c7 — Clean vectorized G-Buffer 2-layer 6-MRT PBR shader (dual TBN + cubemap + lightmap)

layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 2) in vec4 fs_in_attr2_p;
layout(location = 3) in vec4 fs_in_attr3_p;
layout(location = 4) in vec4 fs_in_attr4_p;
layout(location = 5) in vec4 fs_in_attr5_p;
layout(location = 6) in vec4 fs_in_attr6_p;

layout(location = 0) out vec4 frag_color0;
layout(location = 1) out vec4 frag_color1;
layout(location = 2) out vec4 frag_color2;
layout(location = 3) out vec4 frag_color3;
layout(location = 4) out vec4 frag_color4;
layout(location = 5) out vec4 frag_color5;

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
layout(set = 0, binding = 4, std430) readonly buffer ssbo_5_buf { uint data[]; } ssbo_5;
layout(set = 0, binding = 5, std430) readonly buffer ssbo_6_buf { float data[]; } ssbo_6;
layout(set = 0, binding = 6, std430) readonly buffer ssbo_7_buf { uint data[]; } ssbo_7;

layout(set = 0, binding = 7) uniform texture2D fs_img0;
layout(set = 0, binding = 8) uniform texture2D fs_img1;
layout(set = 0, binding = 9) uniform texture2D fs_img2;
layout(set = 0, binding = 10) uniform texture2D fs_img3;
layout(set = 0, binding = 11) uniform texture2D fs_img4;
layout(set = 0, binding = 12) uniform texture2D fs_img5;
layout(set = 0, binding = 13) uniform texture2D fs_img6;
layout(set = 0, binding = 14) uniform texture2D fs_img7;
layout(set = 0, binding = 15) uniform texture2D fs_img8;
layout(set = 0, binding = 16) uniform texture2DArray fs_img9;
layout(set = 0, binding = 17) uniform texture2DArray fs_img10;
layout(set = 0, binding = 18) uniform texture2D fs_img11;
layout(set = 0, binding = 19) uniform texture2D fs_img12;
layout(set = 0, binding = 20) uniform sampler fs_samp0;
layout(set = 0, binding = 21) uniform sampler fs_samp1;
layout(set = 0, binding = 22) uniform sampler fs_samp2;
layout(set = 0, binding = 23) uniform sampler fs_samp3;
layout(set = 0, binding = 24) uniform sampler fs_samp4;
layout(set = 0, binding = 25) uniform sampler fs_samp5;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
float s3f(uint i) { return uintBitsToFloat(ssbo_3.data[i]); }
float s4f(uint i) { return uintBitsToFloat(ssbo_4.data[i]); }
float s5f(uint i) { return uintBitsToFloat(ssbo_5.data[i]); }
float s7f(uint i) { return uintBitsToFloat(ssbo_7.data[i]); }

float safe_rsq(float x) { return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0; }
vec3 safe_norm(vec3 v) { return v * safe_rsq(dot(v, v)); }
float s3_row_dot(uint off, vec4 v) {
    return s3f(off) * v.x + (s3f(off + 1u) * v.y + (s3f(off + 2u) * v.z + s3f(off + 3u) * v.w));
}
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

void main() {
    uvec4 bo0 = push_data.buf_offsets0;
    uint d0 = (bo0.x >> 2u) & 0x3Fu;
    uint d1 = (bo0.x >> 10u) & 0x3Fu;
    uint d2 = (bo0.x >> 18u) & 0x3Fu;
    uint d3 = (bo0.x >> 26u) & 0x3Fu;
    uint d4 = (bo0.y >> 2u) & 0x3Fu;
    uint d5 = (bo0.y >> 10u) & 0x3Fu;
    uint d6 = (bo0.y >> 18u) & 0x3Fu;

    bool is_front = gl_FrontFacing;

    vec4 tex0 = texture(sampler2D(fs_img0, fs_samp0), xform_uv(d0 + 56u, fs_in_attr5_p.zw));
    float mask_bias = s2f(d1 + 12u) * tex0.x + s2f(d1 + 14u);
    float mask_lin = clamp((0.0 < mask_bias) ? (mask_bias * fs_in_attr0_p.w + fs_in_attr0_p.w)
                                             : (mask_bias * (1.0 - fs_in_attr0_p.w) + fs_in_attr0_p.w), 0.0, 1.0);
    float blend_w = (mask_lin * mask_lin) * 3.0 + (mask_lin * (mask_lin * mask_lin)) * -2.0;

    vec4 tex1 = texture(sampler2D(fs_img1, fs_samp1), xform_uv(d0 + 12u, fs_in_attr4_p.xy));
    vec4 tex2 = texture(sampler2D(fs_img2, fs_samp1), xform_uv(d0 + 40u, fs_in_attr4_p.zw));
    vec4 tex3 = texture(sampler2D(fs_img3, fs_samp2), xform_uv(d0 + 8u, fs_in_attr4_p.xy));
    vec4 tex4 = texture(sampler2D(fs_img4, fs_samp2), xform_uv(d0 + 36u, fs_in_attr4_p.zw));
    vec4 tex5 = texture(sampler2D(fs_img5, fs_samp3), fs_in_attr5_p.xy);
    vec4 tex6 = texture(sampler2D(fs_img6, fs_samp2), xform_uv(d0 + 4u, fs_in_attr4_p.xy));
    vec4 tex7 = texture(sampler2D(fs_img7, fs_samp2), xform_uv(d0 + 32u, fs_in_attr4_p.zw));
    vec4 tex8 = texture(sampler2D(fs_img8, fs_samp3), fs_in_attr5_p.xy);

    vec3 n_unit = safe_norm(fs_in_attr1_p.xyz);
    vec3 N_vert = is_front ? n_unit : -n_unit;
    vec3 T0 = safe_norm(fs_in_attr2_p.xyz);
    vec3 B0 = safe_norm(cross(fs_in_attr1_p.xyz, fs_in_attr2_p.xyz) * fs_in_attr2_p.w);
    vec3 T1 = safe_norm(fs_in_attr3_p.xyz);
    vec3 B1 = safe_norm(cross(fs_in_attr1_p.xyz, fs_in_attr3_p.xyz) * fs_in_attr3_p.w);

    float nx0 = fma(2.0, tex1.x, -1.0) * fma(2.0, tex1.w, -1.0);
    float ny0 = fma(2.0, tex1.y, -1.0);
    float nz0 = sqrt(1.0 - clamp(nx0 * nx0 + ny0 * ny0, 0.0, 1.0));
    vec3 N0 = B0 * nx0 + (T0 * ny0 + N_vert * nz0);

    float nx1 = fma(2.0, tex2.x, -1.0) * fma(2.0, tex2.w, -1.0);
    float ny1 = fma(2.0, tex2.y, -1.0);
    float nz1 = sqrt(1.0 - clamp(nx1 * nx1 + ny1 * ny1, 0.0, 1.0));
    vec3 N1 = B1 * nx1 + (T1 * ny1 + N_vert * nz1);
    vec3 N = safe_norm(blend_w * (N1 - N0) + N0);

    float inv_proj_w = 1.0 / s3_row_dot(d2 + 32u, fs_in_attr6_p);
    vec3 V = safe_norm(vec3(
        -s3_row_dot(d2 + 20u, fs_in_attr6_p) * inv_proj_w + s3f(d2 + 183u),
        -s3_row_dot(d2 + 24u, fs_in_attr6_p) * inv_proj_w + s3f(d2 + 187u),
        -s3_row_dot(d2 + 28u, fs_in_attr6_p) * inv_proj_w + s3f(d2 + 191u)
    ));
    float NdotV = dot(N, V);
    vec3 R = -V + N * (NdotV * 2.0);

    float gloss_scale = s5f(d4 + 8u) * 0.00390625;
    float gloss = (blend_w * (tex4.x - tex3.x) + tex3.x) * gloss_scale;
    float mip_lod = min(6.0, 6.5 - gloss * 6.5);
    vec3 c_uv = cube_uv(R);
    vec4 env0 = textureLod(sampler2DArray(fs_img9, fs_samp4), c_uv, mip_lod);
    vec4 env1 = textureLod(sampler2DArray(fs_img10, fs_samp4), c_uv, mip_lod);
    vec4 tex11 = texture(sampler2D(fs_img11, fs_samp5), xform_uv(d0, fs_in_attr4_p.xy));
    vec4 tex12 = texture(sampler2D(fs_img12, fs_samp5), xform_uv(d0 + 28u, fs_in_attr4_p.zw));

    vec3 L_dir = safe_norm(fma(vec3(2.0), tex5.xyz, vec3(-1.0)));
    vec3 H_dir = safe_norm(V + L_dir);
    float lm_w = clamp(s4f(d3 + 41u) + tex5.w, 0.0, 1.0);
    float lm_pow = max(0.00100000005, exp2(s4f(d3 + 42u) * log2(lm_w))) * (100.0 - s4f(d3 + 40u)) + s4f(d3 + 40u);
    float vh_fres = exp2(5.0 * log2(1.0 - clamp(dot(V, H_dir), 0.0, 1.0)));
    float spec_exp_raw = exp2(13.0 * gloss);
    float spec_exp = max(1.0, (spec_exp_raw * lm_pow) * (1.0 / (spec_exp_raw * 0.0100021362 + lm_pow)));
    float nh_spec = exp2(spec_exp * log2(clamp(dot(H_dir, N), 0.0, 1.0))) * ((0.125 * spec_exp + 0.25) * (1.0 - s2f(d1 + 36u)));
    float nl_wrap = clamp(max(dot(L_dir, N), -dot(L_dir, N) * 0.85), 0.0, 1.0);

    bool mono0 = (tex6.z == 0.0) && (tex6.y == 0.0);
    bool mono1 = (tex7.z == 0.0) && (tex7.y == 0.0);
    vec3 sp0 = vec3(tex6.x, mono0 ? tex6.x : tex6.y, mono0 ? tex6.x : tex6.z);
    vec3 sp1 = vec3(tex7.x, mono1 ? tex7.x : tex7.y, mono1 ? tex7.x : tex7.z);
    vec3 spec = blend_w * (sp1 - sp0) + sp0;
    float fres = clamp((1.0 + NdotV) * (1.0 + NdotV), 0.0, 1.0) * exp2(5.0 * log2(1.0 - clamp(NdotV, 0.0, 1.0)));
    vec3 spec_fres = (fres - fres * spec) * (gloss * gloss) + spec;

    uint inst_off = uint(fs_in_attr1_p.w * 1.00001526) * 13u;
    uint sh_base = (inst_off + 8u) * 4u + d5;
    float e_w0 = ssbo_6.data[sh_base];
    float e_w1 = ssbo_6.data[sh_base + 1u];
    float e_w2 = ssbo_6.data[sh_base + 2u];
    float horizon = clamp(dot(R, N_vert) + 1.0, 0.0, 1.0);
    vec3 env_blend = (horizon * horizon) * (s2f(d1 + 36u) * (e_w0 * (-e_w1 * env1.xyz + e_w2 * env0.xyz) + e_w1 * env1.xyz));
    vec3 lm_rgb = s4f(d3 + 43u) * (tex8.xyz * (vec3(s7f(d6), s7f(d6 + 1u), s7f(d6 + 2u)) * tex8.xyz));
    vec3 dir_spec = ((-spec) * vh_fres + (spec + vh_fres)) * ((lm_w * lm_rgb) * nh_spec);
    vec3 env_tint = vec3(s5f(d4 + 4u), s5f(d4 + 5u), s5f(d4 + 6u)) * s4f(d3 + 12u);

    vec4 alb_blend = blend_w * (tex12 - tex11) + tex11;
    float alpha_val = alb_blend.w * ssbo_6.data[(inst_off + 7u) * 4u + 3u + d5];
    bool alpha_pass = (s5f(d4 + 22u) != 0.0);
    float alpha_zero = alpha_pass ? alpha_val : 0.0;
    float mat_id = 0.00392156886 * float(floatBitsToUint(ssbo_6.data[(inst_off + 6u) * 4u + d5]));

    frag_color5 = vec4((env_blend * spec_fres + dir_spec) * env_tint, alpha_pass ? alpha_val : ssbo_6.data[sh_base + 3u]);
    frag_color4 = vec4((((-spec) * nl_wrap + nl_wrap) * (lm_w * lm_rgb) + (lm_rgb - lm_w * lm_rgb)) * (1.0 - spec), alpha_pass ? alpha_val : 1.0);
    frag_color3 = vec4(sqrt(spec), alpha_pass ? alpha_val : mat_id);
    frag_color2 = vec4(vec3(s5f(d4), s5f(d4 + 1u), s5f(d4 + 2u)) * alb_blend.xyz, alpha_val);
    frag_color1 = vec4(gloss_scale * gloss, 0.100000001 * (s2f(d1 + 4u) * fs_in_attr0_p.y), tex8.w, alpha_zero);
    frag_color0 = vec4(fma(vec3(0.5), N, vec3(0.500488758)), alpha_zero);
}
