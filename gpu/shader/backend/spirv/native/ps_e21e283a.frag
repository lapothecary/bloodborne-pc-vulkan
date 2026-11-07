#version 460
#extension GL_EXT_samplerless_texture_functions : require
// ps_e21e283a — Clean vectorized G-Buffer foliage/cloth shader (opaque two-sided + detail + Fresnel)

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

layout(set = 0, binding = 5) uniform texture2D fs_img0;
layout(set = 0, binding = 6) uniform texture2D fs_img1;
layout(set = 0, binding = 7) uniform texture2D fs_img2;
layout(set = 0, binding = 8) uniform texture2D fs_img3;
layout(set = 0, binding = 9) uniform texture2D fs_img4;
layout(set = 0, binding = 10) uniform texture2DArray fs_img5;
layout(set = 0, binding = 11) uniform texture2DArray fs_img6;
layout(set = 0, binding = 12) uniform texture2D fs_img7;
layout(set = 0, binding = 13) uniform sampler fs_samp0;
layout(set = 0, binding = 14) uniform sampler fs_samp1;
layout(set = 0, binding = 15) uniform sampler fs_samp2;
layout(set = 0, binding = 16) uniform sampler fs_samp3;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
float s3f(uint i) { return uintBitsToFloat(ssbo_3.data[i]); }
float s5f(uint i) { return uintBitsToFloat(ssbo_5.data[i]); }

float safe_rsq(float x) { return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0; }
vec3 safe_norm(vec3 v) { return v * safe_rsq(dot(v, v)); }
float s1_row_dot(uint off, vec4 v) {
    return s1f(off) * v.x + (s1f(off + 1u) * v.y + (s1f(off + 2u) * v.z + s1f(off + 3u) * v.w));
}
vec2 xform_uv(uint off, vec2 uv) {
    return vec2(uv.x * s2f(off) + s2f(off + 2u), uv.y * s2f(off + 1u) + s2f(off + 3u));
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
    float d = n.z * ssbo_4.data[base + 2u] + (n.x * ssbo_4.data[base] + n.y * ssbo_4.data[base + 1u]);
    float c = ssbo_4.data[base + 3u];
    return max(0.0, max(c + d, (c - d) * 0.85));
}

void main() {
    uvec4 bo0 = push_data.buf_offsets0;
    uint d0 = (bo0.x >> 2u) & 0x3Fu;
    uint d1 = (bo0.x >> 10u) & 0x3Fu;
    uint d2 = (bo0.x >> 18u) & 0x3Fu;
    uint d3 = (bo0.x >> 26u) & 0x3Fu;
    uint d4 = (bo0.y >> 2u) & 0x3Fu;

    bool is_front = gl_FrontFacing;

    vec4 tex0 = texture(sampler2D(fs_img0, fs_samp0), fs_in_attr3_p.zw);
    vec4 tex1 = texture(sampler2D(fs_img1, fs_samp1), xform_uv(d1 + 12u, fs_in_attr3_p.xy));
    vec4 tex2 = texture(sampler2D(fs_img2, fs_samp2), xform_uv(d1 + 8u, fs_in_attr3_p.xy));
    vec4 tex3 = texture(sampler2D(fs_img3, fs_samp2), fs_in_attr3_p.zw);
    vec4 tex4 = texture(sampler2D(fs_img4, fs_samp2), xform_uv(d1 + 4u, fs_in_attr3_p.xy));

    vec3 n_unit = safe_norm(fs_in_attr0_p.xyz);
    vec3 N_vert = is_front ? n_unit : -n_unit;
    vec3 T_vert = safe_norm(fs_in_attr1_p.xyz);
    vec3 B_vert = safe_norm(cross(fs_in_attr0_p.xyz, fs_in_attr1_p.xyz) * fs_in_attr1_p.w);

    float nx = fma(2.0, tex1.x, -1.0) * fma(2.0, tex1.w, -1.0);
    float ny = fma(2.0, tex1.y, -1.0);
    float nz = sqrt(1.0 - clamp(nx * nx + ny * ny, 0.0, 1.0));
    vec3 N = safe_norm(B_vert * nx + (T_vert * ny + N_vert * nz));

    float inv_proj_w = 1.0 / s1_row_dot(d0 + 32u, fs_in_attr4_p);
    vec3 V = safe_norm(vec3(
        -s1_row_dot(d0 + 20u, fs_in_attr4_p) * inv_proj_w + s1f(d0 + 183u),
        -s1_row_dot(d0 + 24u, fs_in_attr4_p) * inv_proj_w + s1f(d0 + 187u),
        -s1_row_dot(d0 + 28u, fs_in_attr4_p) * inv_proj_w + s1f(d0 + 191u)
    ));
    float NdotV = dot(N, V);
    vec3 R = -V + N * (NdotV * 2.0);

    float det_w = s3f(d2 + 24u);
    float gloss_scale = s3f(d2 + 8u) * 0.00390625;
    float gloss = (det_w * ((tex0.w * tex2.x + tex3.x) - tex2.x) + tex2.x) * gloss_scale;
    float mip_lod = min(6.0, 6.5 - gloss * 6.5);
    vec3 c_uv = cube_uv(is_front ? R : vec3(R.x, abs(R.y), R.z));
    vec4 env0 = textureLod(sampler2DArray(fs_img5, fs_samp3), c_uv, mip_lod);
    vec4 env1 = textureLod(sampler2DArray(fs_img6, fs_samp3), c_uv, mip_lod);
    vec4 tex7 = texture(sampler2D(fs_img7, fs_samp0), xform_uv(d1, fs_in_attr3_p.xy));

    bool mono_spec = (tex4.z == 0.0) && (tex4.y == 0.0);
    vec3 spec_tex = vec3(tex4.x, mono_spec ? tex4.x : tex4.y, mono_spec ? tex4.x : tex4.z);
    vec3 spec = det_w * ((tex0.w * spec_tex + tex3.y * tex3.y) - spec_tex) + spec_tex;
    float fres = clamp((1.0 + NdotV) * (1.0 + NdotV), 0.0, 1.0) * exp2(5.0 * log2(1.0 - clamp(NdotV, 0.0, 1.0)));
    vec3 spec_fres = (fres - fres * spec) * (gloss * gloss) + spec;

    uint inst_off = uint(fs_in_attr0_p.w * 1.00001526) * 13u;
    uint sh_base = (inst_off + 8u) * 4u + d3;
    float e_w0 = ssbo_4.data[sh_base];
    float e_w1 = ssbo_4.data[sh_base + 1u];
    float e_w2 = ssbo_4.data[sh_base + 2u];
    float horizon = clamp(dot(R, N_vert) + 1.0, 0.0, 1.0);
    vec3 env_blend = (e_w0 * (-e_w1 * env1.xyz + e_w2 * env0.xyz) + e_w1 * env1.xyz) * (horizon * horizon);
    vec3 env_tint = vec3(s3f(d2 + 4u), s3f(d2 + 5u), s3f(d2 + 6u)) * s5f(d4 + 12u);

    vec3 sh_rgb = vec3(eval_sh(sh_base + 4u, N), eval_sh(sh_base + 8u, N), eval_sh(sh_base + 12u, N));
    vec3 alb_base = det_w * ((tex0.w * tex7.xyz + tex0.xyz) - tex7.xyz) + tex7.xyz;

    float alpha_val = (tex7.w * fs_in_attr2_p.w) * ssbo_4.data[(inst_off + 7u) * 4u + 3u + d3];
    bool alpha_pass = (s3f(d2 + 22u) != 0.0);
    float alpha_zero = alpha_pass ? alpha_val : 0.0;
    float mat_id = 0.00392156886 * float(floatBitsToUint(ssbo_4.data[(inst_off + 6u) * 4u + d3]));

    frag_color5 = vec4((spec_fres * env_blend) * env_tint, alpha_pass ? alpha_val : ssbo_4.data[sh_base + 3u]);
    frag_color4 = vec4(-spec * sh_rgb + sh_rgb, alpha_pass ? alpha_val : 1.0);
    frag_color3 = vec4(sqrt(spec), alpha_pass ? alpha_val : mat_id);
    frag_color2 = vec4(vec3(s3f(d2), s3f(d2 + 1u), s3f(d2 + 2u)) * alb_base, alpha_val);
    frag_color1 = vec4(gloss_scale * gloss, is_front ? 0.0 : 0.08, ssbo_4.data[sh_base + 16u], alpha_zero);
    frag_color0 = vec4(fma(vec3(0.5), N, vec3(0.500488758)), alpha_zero);
}
