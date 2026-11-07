#version 460
#extension GL_EXT_samplerless_texture_functions : require
// ps_bb10e440_gbuf — Clean vectorized G-Buffer main 6-MRT PBR shader (lightmap Blinn-Phong + wrap diffuse)

layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 2) in vec4 fs_in_attr2_p;
layout(location = 3) in vec4 fs_in_attr3_p;
layout(location = 4) in vec4 fs_in_attr4_p;

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
layout(set = 0, binding = 3, std430) readonly buffer ssbo_4_buf { float data[]; } ssbo_4;
layout(set = 0, binding = 4, std430) readonly buffer ssbo_5_buf { uint data[]; } ssbo_5;
layout(set = 0, binding = 5, std430) readonly buffer ssbo_6_buf { uint data[]; } ssbo_6;
layout(set = 0, binding = 6, std430) readonly buffer ssbo_7_buf { uint data[]; } ssbo_7;

layout(set = 0, binding = 7) uniform texture2D fs_img0;
layout(set = 0, binding = 8) uniform texture2D fs_img1;
layout(set = 0, binding = 9) uniform texture2D fs_img2;
layout(set = 0, binding = 10) uniform texture2D fs_img3;
layout(set = 0, binding = 11) uniform texture2D fs_img4;
layout(set = 0, binding = 12) uniform texture2D fs_img5;
layout(set = 0, binding = 13) uniform sampler fs_samp0;
layout(set = 0, binding = 14) uniform sampler fs_samp1;
layout(set = 0, binding = 15) uniform sampler fs_samp2;
layout(set = 0, binding = 16) uniform sampler fs_samp3;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
float s3f(uint i) { return uintBitsToFloat(ssbo_3.data[i]); }
float s5f(uint i) { return uintBitsToFloat(ssbo_5.data[i]); }
float s6f(uint i) { return uintBitsToFloat(ssbo_6.data[i]); }
float s7f(uint i) { return uintBitsToFloat(ssbo_7.data[i]); }

float safe_rsq(float x) { return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0; }
vec3 safe_norm(vec3 v) { return v * safe_rsq(dot(v, v)); }
float s1_row_dot(uint off, vec4 v) {
    return s1f(off) * v.x + (s1f(off + 1u) * v.y + (s1f(off + 2u) * v.z + s1f(off + 3u) * v.w));
}
vec2 xform_uv(uint off, vec2 uv) {
    return vec2(uv.x * s3f(off) + s3f(off + 2u), uv.y * s3f(off + 1u) + s3f(off + 3u));
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

    vec4 tex0 = texture(sampler2D(fs_img0, fs_samp0), xform_uv(d2 + 12u, fs_in_attr3_p.xy));
    vec4 tex1 = texture(sampler2D(fs_img1, fs_samp1), fs_in_attr3_p.zw);
    vec4 tex2 = texture(sampler2D(fs_img2, fs_samp1), fs_in_attr3_p.zw);
    vec4 tex3 = texture(sampler2D(fs_img3, fs_samp2), xform_uv(d2 + 8u, fs_in_attr3_p.xy));
    vec4 tex4 = texture(sampler2D(fs_img4, fs_samp2), xform_uv(d2 + 4u, fs_in_attr3_p.xy));
    vec4 tex5 = texture(sampler2D(fs_img5, fs_samp3), xform_uv(d2, fs_in_attr3_p.xy));

    vec3 n_unit = safe_norm(fs_in_attr0_p.xyz);
    vec3 N_vert = is_front ? n_unit : -n_unit;
    vec3 T_vert = safe_norm(fs_in_attr1_p.xyz);
    vec3 B_vert = safe_norm(cross(fs_in_attr0_p.xyz, fs_in_attr1_p.xyz) * fs_in_attr1_p.w);

    float nx = fma(2.0, tex0.x, -1.0) * fma(2.0, tex0.w, -1.0);
    float ny = fma(2.0, tex0.y, -1.0);
    float nz = sqrt(1.0 - clamp(nx * nx + ny * ny, 0.0, 1.0));
    vec3 N = safe_norm(B_vert * nx + (T_vert * ny + N_vert * nz));

    float inv_proj_w = 1.0 / s1_row_dot(d0 + 32u, fs_in_attr4_p);
    vec3 V = safe_norm(vec3(
        -s1_row_dot(d0 + 20u, fs_in_attr4_p) * inv_proj_w + s1f(d0 + 183u),
        -s1_row_dot(d0 + 24u, fs_in_attr4_p) * inv_proj_w + s1f(d0 + 187u),
        -s1_row_dot(d0 + 28u, fs_in_attr4_p) * inv_proj_w + s1f(d0 + 191u)
    ));

    float gloss_scale = s6f(d5 + 8u) * 0.00390625;
    float gloss = tex3.x * gloss_scale;

    vec3 L_dir = safe_norm(fma(vec3(2.0), tex1.xyz, vec3(-1.0)));
    vec3 H_dir = safe_norm(V + L_dir);
    float lm_w = clamp(s2f(d1 + 41u) + tex1.w, 0.0, 1.0);
    float lm_pow = max(0.00100000005, exp2(s2f(d1 + 42u) * log2(lm_w))) * (100.0 - s2f(d1 + 40u)) + s2f(d1 + 40u);
    float vh_fres = exp2(5.0 * log2(1.0 - clamp(dot(V, H_dir), 0.0, 1.0)));
    float spec_exp_raw = exp2(13.0 * gloss);
    float spec_exp = max(1.0, (spec_exp_raw * lm_pow) * (1.0 / (spec_exp_raw * 0.0100021362 + lm_pow)));
    float nh_spec = exp2(spec_exp * log2(clamp(dot(H_dir, N), 0.0, 1.0))) * (0.125 * spec_exp + 0.25);
    float nl_wrap = clamp(max(dot(L_dir, N), -dot(L_dir, N) * 0.85), 0.0, 1.0);

    bool mono_spec = (tex4.z == 0.0) && (tex4.y == 0.0);
    vec3 spec = vec3(tex4.x, mono_spec ? tex4.x : tex4.y, mono_spec ? tex4.x : tex4.z);
    vec3 lm_rgb = s2f(d1 + 43u) * (tex2.xyz * (vec3(s7f(d6), s7f(d6 + 1u), s7f(d6 + 2u)) * tex2.xyz));
    vec3 dir_spec = ((-spec) * vh_fres + (spec + vh_fres)) * ((lm_w * lm_rgb) * nh_spec);
    vec3 env_tint = vec3(s6f(d5 + 4u), s6f(d5 + 5u), s6f(d5 + 6u)) * s2f(d1 + 12u);

    uint inst_off = uint(fs_in_attr0_p.w * 1.00001526) * 13u;
    float alpha_val = (tex5.w * fs_in_attr2_p.w) * ssbo_4.data[(inst_off + 7u) * 4u + 3u + d3];
    bool alpha_pass = (s6f(d5 + 22u) != 0.0);
    float alpha_zero = alpha_pass ? alpha_val : 0.0;
    float mat_id = 0.00392156886 * float(floatBitsToUint(ssbo_4.data[(inst_off + 6u) * 4u + d3]));

    frag_color5 = vec4(dir_spec * env_tint, alpha_pass ? alpha_val : ssbo_4.data[(inst_off + 8u) * 4u + 3u + d3]);
    frag_color4 = vec4((((-spec) * nl_wrap + nl_wrap) * (lm_w * lm_rgb) + (lm_rgb - lm_w * lm_rgb)) * (1.0 - spec), alpha_pass ? alpha_val : 1.0);
    frag_color3 = vec4(sqrt(spec), alpha_pass ? alpha_val : mat_id);
    frag_color2 = vec4(vec3(s6f(d5), s6f(d5 + 1u), s6f(d5 + 2u)) * tex5.xyz, alpha_val);
    frag_color1 = vec4(gloss_scale * gloss, 0.100000001 * (s5f(d4 + 4u) * fs_in_attr2_p.y), tex2.w, alpha_zero);
    frag_color0 = vec4(fma(vec3(0.5), N, vec3(0.500488758)), alpha_zero);
}
