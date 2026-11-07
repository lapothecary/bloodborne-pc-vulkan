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

// G-Buffer Foliage (2-layer blend, dual TBN, lightmap Blinn-Phong) — hash 0xdbb222eb
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

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { uint data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { uint data[]; } ssbo_2;
layout(set = 0, binding = 2, std430) readonly buffer ssbo_3_buf { uint data[]; } ssbo_3;
layout(set = 0, binding = 3, std430) readonly buffer ssbo_4_buf { uint data[]; } ssbo_4;
layout(set = 0, binding = 4, std430) readonly buffer ssbo_5_buf { float data[]; } ssbo_5;
layout(set = 0, binding = 5, std430) readonly buffer ssbo_6_buf { uint data[]; } ssbo_6;
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
layout(set = 0, binding = 16) uniform texture2D fs_img9;
layout(set = 0, binding = 17) uniform texture2D fs_img10;
layout(set = 0, binding = 18) uniform sampler fs_samp0;
layout(set = 0, binding = 19) uniform sampler fs_samp1;
layout(set = 0, binding = 20) uniform sampler fs_samp2;
layout(set = 0, binding = 21) uniform sampler fs_samp3;
layout(set = 0, binding = 22) uniform sampler fs_samp4;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
float s3f(uint i) { return uintBitsToFloat(ssbo_3.data[i]); }
float s4f(uint i) { return uintBitsToFloat(ssbo_4.data[i]); }
float s6f(uint i) { return uintBitsToFloat(ssbo_6.data[i]); }
float s7f(uint i) { return uintBitsToFloat(ssbo_7.data[i]); }

vec4 load_s3_v4(uint i) { return vec4(s3f(i), s3f(i + 1u), s3f(i + 2u), s3f(i + 3u)); }

vec3 safe_normalize(vec3 v) {
    float l2 = dot(v, v);
    float rsq = (l2 > 0.0 && l2 <= 3.40282347e+38) ? inversesqrt(clamp(l2, 1.17549435e-38, 3.40282347e+38)) : 0.0;
    return v * rsq;
}

vec2 xform_uv1(uint base, vec2 uv) {
    return vec2(uv.x * s1f(base) + s1f(base + 2u), uv.y * s1f(base + 1u) + s1f(base + 3u));
}

void main() {
    uint u0 = push_data.buf_offsets0.x;
    uint d0 = bitfieldExtract(u0, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(u0, 8, 8) >> 2u;
    uint d2 = bitfieldExtract(u0, 16, 8) >> 2u;
    uint d3 = bitfieldExtract(u0, 24, 8) >> 2u;
    uint u1 = push_data.buf_offsets0.y;
    uint d4 = bitfieldExtract(u1, 0, 8) >> 2u;
    uint d5 = bitfieldExtract(u1, 8, 8) >> 2u;
    uint d6 = bitfieldExtract(u1, 16, 8) >> 2u;

    vec2 uv0 = fs_in_attr4_p.xy;
    vec2 uv1 = fs_in_attr4_p.zw;
    vec4 tex_mask = texture(sampler2D(fs_img0, fs_samp0), xform_uv1(d0 + 56u, fs_in_attr5_p.zw));
    float mask_val = s2f(d1 + 12u) * tex_mask.x + s2f(d1 + 14u);
    float mask_t = clamp((0.0 < mask_val)
        ? (mask_val * fs_in_attr0_p.w + fs_in_attr0_p.w)
        : (mask_val * (1.0 - fs_in_attr0_p.w) + fs_in_attr0_p.w), 0.0, 1.0);
    float blend_w = (mask_t * mask_t) * 3.0 + (mask_t * (mask_t * mask_t)) * -2.0;

    vec4 tex_nrm0 = texture(sampler2D(fs_img1, fs_samp1), xform_uv1(d0 + 12u, uv0));
    vec4 tex_nrm1 = texture(sampler2D(fs_img2, fs_samp1), xform_uv1(d0 + 40u, uv1));
    vec4 tex_lmap_dir = texture(sampler2D(fs_img3, fs_samp2), fs_in_attr5_p.xy);
    vec4 tex_rgh0 = texture(sampler2D(fs_img4, fs_samp3), xform_uv1(d0 + 8u, uv0));
    vec4 tex_rgh1 = texture(sampler2D(fs_img5, fs_samp3), xform_uv1(d0 + 36u, uv1));
    vec4 tex_spec0 = texture(sampler2D(fs_img6, fs_samp3), xform_uv1(d0 + 4u, uv0));
    vec4 tex_spec1 = texture(sampler2D(fs_img7, fs_samp3), xform_uv1(d0 + 32u, uv1));
    vec4 tex_lmap_col = texture(sampler2D(fs_img8, fs_samp2), fs_in_attr5_p.xy);
    vec4 tex_alb0 = texture(sampler2D(fs_img9, fs_samp4), xform_uv1(d0 + 0u, uv0));
    vec4 tex_alb1 = texture(sampler2D(fs_img10, fs_samp4), xform_uv1(d0 + 28u, uv1));

    bool front = gl_FrontFacing;
    vec3 N_vert = safe_normalize(fs_in_attr1_p.xyz) * (front ? 1.0 : -1.0);
    vec3 T0 = safe_normalize(fs_in_attr2_p.xyz);
    vec3 B0 = safe_normalize(cross(fs_in_attr1_p.xyz, fs_in_attr2_p.xyz) * fs_in_attr2_p.w);
    vec3 T1 = safe_normalize(fs_in_attr3_p.xyz);
    vec3 B1 = safe_normalize(cross(fs_in_attr1_p.xyz, fs_in_attr3_p.xyz) * fs_in_attr3_p.w);

    float nx0 = fma(2.0, tex_nrm0.x, -1.0) * fma(2.0, tex_nrm0.w, -1.0);
    float ny0 = fma(2.0, tex_nrm0.y, -1.0);
    float nz0 = sqrt(1.0 - clamp(nx0 * nx0 + ny0 * ny0, 0.0, 1.0));
    vec3 N0 = B0 * nx0 + T0 * ny0 + N_vert * nz0;

    float nx1 = fma(2.0, tex_nrm1.x, -1.0) * fma(2.0, tex_nrm1.w, -1.0);
    float ny1 = fma(2.0, tex_nrm1.y, -1.0);
    float nz1 = sqrt(1.0 - clamp(nx1 * nx1 + ny1 * ny1, 0.0, 1.0));
    vec3 N1 = B1 * nx1 + T1 * ny1 + N_vert * nz1;

    vec3 N = safe_normalize(blend_w * (N1 - N0) + N0);

    float inv_w = 1.0 / dot(load_s3_v4(d2 + 32u), fs_in_attr6_p);
    vec3 cam_pos = vec3(s3f(d2 + 183u), s3f(d2 + 187u), s3f(d2 + 191u));
    vec3 view_vec = vec3(
        -dot(load_s3_v4(d2 + 20u), fs_in_attr6_p) * inv_w + cam_pos.x,
        -dot(load_s3_v4(d2 + 24u), fs_in_attr6_p) * inv_w + cam_pos.y,
        -dot(load_s3_v4(d2 + 28u), fs_in_attr6_p) * inv_w + cam_pos.z
    );
    vec3 V = safe_normalize(view_vec);
    vec3 L_dir = safe_normalize(fma(vec3(2.0), tex_lmap_dir.xyz, vec3(-1.0)));
    vec3 H = safe_normalize(V + L_dir);

    float gloss_scale = s6f(d5 + 8u) * 0.00390625;
    float gloss = (blend_w * (tex_rgh1.x - tex_rgh0.x) + tex_rgh0.x) * gloss_scale;

    float lm_w = clamp(s4f(d3 + 41u) + tex_lmap_dir.w, 0.0, 1.0);
    float lm_pow = max(0.00100000005, exp2(s4f(d3 + 42u) * log2(lm_w))) * (100.0 - s4f(d3 + 40u)) + s4f(d3 + 40u);
    float exp_g = exp2(13.0 * gloss);
    float spec_exp = max(1.0, (exp_g * lm_pow) * (1.0 / (exp_g * 0.0100021362 + lm_pow)));

    bool mono0 = (tex_spec0.z == 0.0) && (tex_spec0.y == 0.0);
    bool mono1 = (tex_spec1.z == 0.0) && (tex_spec1.y == 0.0);
    vec3 spec0 = vec3(tex_spec0.x, mono0 ? tex_spec0.x : tex_spec0.y, mono0 ? tex_spec0.x : tex_spec0.z);
    vec3 spec1 = vec3(tex_spec1.x, mono1 ? tex_spec1.x : tex_spec1.y, mono1 ? tex_spec1.x : tex_spec1.z);
    vec3 spec_col = blend_w * (spec1 - spec0) + spec0;

    vec3 lm_tint = vec3(s7f(d6 + 0u), s7f(d6 + 1u), s7f(d6 + 2u));
    vec3 lm_rgb = s4f(d3 + 43u) * (tex_lmap_col.xyz * (lm_tint * tex_lmap_col.xyz));

    float VdotH = clamp(dot(V, H), 0.0, 1.0);
    float fresnel = exp2(5.0 * log2(1.0 - VdotH));
    float NdotL = clamp(dot(L_dir, N), 0.0, 1.0);
    float NdotH = clamp(dot(H, N), 0.0, 1.0);
    float blinn = exp2(spec_exp * log2(NdotH)) * (0.125 * spec_exp + 0.25);

    uint inst_base = uint(fs_in_attr1_p.w * 1.00001526) * 13u;
    float alpha_mix = blend_w * (tex_alb1.w - tex_alb0.w) + tex_alb0.w;
    float alpha_out = alpha_mix * ssbo_5.data[(inst_base + 7u) * 4u + 3u + d4];
    bool use_alpha = (s6f(d5 + 22u) != 0.0);
    float alpha_opt = use_alpha ? alpha_out : 0.0;

    vec3 albedo_tint = vec3(s6f(d5 + 0u), s6f(d5 + 1u), s6f(d5 + 2u));
    vec3 albedo = albedo_tint * (blend_w * (tex_alb1.xyz - tex_alb0.xyz) + tex_alb0.xyz);
    vec3 spec_scale = vec3(s6f(d5 + 4u), s6f(d5 + 5u), s6f(d5 + 6u)) * s4f(d3 + 12u);

    frag_color0 = vec4(fma(vec3(0.5), N, vec3(0.500488758)), alpha_opt);
    frag_color1 = vec4(gloss_scale * gloss, 0.100000001 * (s2f(d1 + 4u) * fs_in_attr0_p.y), tex_lmap_col.w, alpha_opt);
    frag_color2 = vec4(albedo, alpha_out);
    frag_color3 = vec4(
        sqrt(spec_col),
        use_alpha ? alpha_out : (0.00392156886 * float(floatBitsToUint(ssbo_5.data[(inst_base + 6u) * 4u + d4])))
    );
    frag_color4 = vec4(
        ((-spec_col * NdotL + NdotL) * (lm_w * lm_rgb) + (lm_rgb - lm_w * lm_rgb)) * (1.0 - spec_col),
        use_alpha ? alpha_out : 1.0
    );
    frag_color5 = vec4(
        ((-spec_col * fresnel + (spec_col + fresnel)) * ((lm_w * lm_rgb) * blinn)) * spec_scale,
        use_alpha ? alpha_out : ssbo_5.data[(inst_base + 8u) * 4u + 3u + d4]
    );
}
