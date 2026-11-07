#version 460
#extension GL_EXT_samplerless_texture_functions : require
// Native Vulkan 1.3 GLSL PS 0x000000001c3953c6 (G-Buffer cutout 6-MRT PBR foliage)

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
layout(set = 0, binding = 13) uniform texture2D fs_img6;
layout(set = 0, binding = 14) uniform sampler fs_samp0;
layout(set = 0, binding = 15) uniform sampler fs_samp1;
layout(set = 0, binding = 16) uniform sampler fs_samp2;
layout(set = 0, binding = 17) uniform sampler fs_samp3;
layout(set = 0, binding = 18) uniform sampler fs_samp4;

float safe_rsq(float x) { return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0; }
vec3 safe_norm(vec3 v) { return v * safe_rsq(dot(v, v)); }
float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
float s3f(uint i) { return uintBitsToFloat(ssbo_3.data[i]); }
float s5f(uint i) { return uintBitsToFloat(ssbo_5.data[i]); }
float s6f(uint i) { return uintBitsToFloat(ssbo_6.data[i]); }
float s7f(uint i) { return uintBitsToFloat(ssbo_7.data[i]); }
float s1_row_dot(uint r, vec4 v) {
    return s1f(r) * v.x + (s1f(r + 1u) * v.y + (s1f(r + 2u) * v.z + s1f(r + 3u) * v.w));
}
vec2 xform_uv(uint b, vec2 uv) {
    return uv * vec2(s3f(b), s3f(b + 1u)) + vec2(s3f(b + 2u), s3f(b + 3u));
}

void main() {
    const uint bo0 = push_data.buf_offsets0.x, bo1 = push_data.buf_offsets0.y;
    const uint d0 = bitfieldExtract(bo0, 0, 8) >> 2u, d1 = bitfieldExtract(bo0, 8, 8) >> 2u;
    const uint d2 = bitfieldExtract(bo0, 16, 8) >> 2u, d3 = bitfieldExtract(bo0, 24, 8) >> 2u;
    const uint d4 = bitfieldExtract(bo1, 0, 8) >> 2u, d5 = bitfieldExtract(bo1, 8, 8) >> 2u;
    const uint d6 = bitfieldExtract(bo1, 16, 8) >> 2u;

    const vec3 n_in = fs_in_attr0_p.xyz;
    const bool is_front = gl_FrontFacing;
    const vec3 N_vert = (is_front ? 1.0 : -1.0) * safe_norm(n_in);

    const vec4 clip_in = fs_in_attr4_p;
    const vec2 screen_uv = fma(vec2(0.5, -0.5) * clip_in.xy, vec2(1.0 / clip_in.w), vec2(0.5));
    const float dither_a = texture(sampler2D(fs_img0, fs_samp0), 0.25 * vec2(s1f(d0 + 4u), s1f(d0 + 5u)) * screen_uv).w - 0.5;

    const vec2 uv0 = fs_in_attr3_p.xy;
    const vec4 base_tex = texture(sampler2D(fs_img1, fs_samp1), xform_uv(d2 + 0u, uv0));
    const uint mat_row = uint(fs_in_attr0_p.w * 1.00001526) * 13u;
    const float alpha_raw = base_tex.w * fs_in_attr2_p.w;
    const float out_alpha = alpha_raw * ssbo_4.data[d3 + (mat_row + 7u) * 4u + 3u];
    if (s5f(d4 + 20u) > (s5f(d4 + 21u) * dither_a + alpha_raw) || out_alpha == 0.0) {
        discard;
    }

    const vec3 wpos = vec3(s1_row_dot(d0 + 20u, clip_in), s1_row_dot(d0 + 24u, clip_in), s1_row_dot(d0 + 28u, clip_in));
    const float inv_w = 1.0 / s1_row_dot(d0 + 32u, clip_in);
    const vec3 cam_pos = vec3(s1f(d0 + 183u), s1f(d0 + 187u), s1f(d0 + 191u));
    const vec3 V = safe_norm((-wpos) * inv_w + cam_pos);

    const vec4 nrm_tex = texture(sampler2D(fs_img2, fs_samp2), xform_uv(d2 + 12u, uv0));
    const vec4 ldir_tex = texture(sampler2D(fs_img3, fs_samp3), fs_in_attr3_p.zw);
    const vec4 lcol_tex = texture(sampler2D(fs_img4, fs_samp3), fs_in_attr3_p.zw);
    const float gloss_tex = texture(sampler2D(fs_img5, fs_samp4), xform_uv(d2 + 8u, uv0)).x;
    const vec3 spec_tex = texture(sampler2D(fs_img6, fs_samp4), xform_uv(d2 + 4u, uv0)).rgb;

    const vec3 T = safe_norm(fs_in_attr1_p.xyz);
    const vec3 B = safe_norm(cross(n_in, fs_in_attr1_p.xyz) * fs_in_attr1_p.w);
    const float nx = fma(2.0, nrm_tex.x, -1.0) * fma(2.0, nrm_tex.w, -1.0);
    const float ny = fma(2.0, nrm_tex.y, -1.0);
    const float nz = sqrt(1.0 - clamp(nx * nx + ny * ny, 0.0, 1.0));
    const vec3 N = safe_norm(B * nx + (T * ny + N_vert * nz));

    const vec3 L = safe_norm(fma(vec3(2.0), ldir_tex.rgb, vec3(-1.0)));
    const vec3 H = safe_norm(V + L);
    const float gloss_scale = s5f(d4 + 8u) * 0.00390625;
    const float gloss = gloss_tex * gloss_scale;
    const float spec_pow = exp2(13.0 * gloss);
    const float lmap_w = clamp(s2f(d1 + 41u) + ldir_tex.w, 0.0, 1.0);
    const float atten_pow = max(0.00100000005, exp2(s2f(d1 + 42u) * log2(lmap_w)));
    const float atten_mix = atten_pow * (100.0 - s2f(d1 + 40u)) + s2f(d1 + 40u);
    const float spec_exp = max(1.0, (spec_pow * atten_mix) / (spec_pow * 0.0100021362 + atten_mix));

    const float VdotH = clamp(dot(V, H), 0.0, 1.0), NdotH = clamp(dot(N, H), 0.0, 1.0);
    const float ndotl_raw = dot(N, L);
    const float NdotL = clamp(max(ndotl_raw, -ndotl_raw * 0.85), 0.0, 1.0);
    const float fresnel = exp2(5.0 * log2(1.0 - VdotH));
    const vec3 F0 = (spec_tex.z == 0.0 && spec_tex.y == 0.0) ? vec3(spec_tex.x) : spec_tex;
    const vec3 F = fma(-F0, vec3(fresnel), F0 + vec3(fresnel));

    const vec3 lcol_sq = vec3(s7f(d6 + 0u), s7f(d6 + 1u), s7f(d6 + 2u)) * lcol_tex.rgb * lcol_tex.rgb * s2f(d1 + 43u);
    const vec3 lcol_dir = lcol_sq * lmap_w;
    const float spec_brdf = exp2(spec_exp * log2(NdotH)) * (0.125 * spec_exp + 0.25);
    const vec3 out_spec = (F * (lcol_dir * spec_brdf)) * (vec3(s5f(d4 + 4u), s5f(d4 + 5u), s5f(d4 + 6u)) * s2f(d1 + 12u));
    const vec3 out_diff = (fma(-F0, vec3(NdotL), vec3(NdotL)) * lcol_dir + (lcol_sq - lcol_dir)) * (1.0 - F0);

    const vec3 albedo = base_tex.rgb * vec3(s5f(d4 + 0u), s5f(d4 + 1u), s5f(d4 + 2u));
    const bool blend_alpha = (s5f(d4 + 22u) != 0.0);
    const float a01 = blend_alpha ? out_alpha : 0.0;
    const float mat_id = 0.00392156886 * float(floatBitsToUint(ssbo_4.data[d3 + (mat_row + 6u) * 4u]));
    const float amb_w = ssbo_4.data[d3 + (mat_row + 8u) * 4u + 3u];

    frag_color0 = vec4(fma(vec3(0.5), N, vec3(0.500488758)), a01);
    frag_color1 = vec4(gloss_scale * gloss, 0.100000001 * (s6f(d5 + 4u) * fs_in_attr2_p.y), lcol_tex.w, a01);
    frag_color2 = vec4(albedo, out_alpha);
    frag_color3 = vec4(sqrt(F0), blend_alpha ? out_alpha : mat_id);
    frag_color4 = vec4(out_diff, blend_alpha ? out_alpha : 1.0);
    frag_color5 = vec4(out_spec, blend_alpha ? out_alpha : amb_w);
}
