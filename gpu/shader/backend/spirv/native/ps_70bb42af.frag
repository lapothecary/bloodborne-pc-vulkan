#version 460
#extension GL_EXT_samplerless_texture_functions : require
// Native Vulkan 1.3 GLSL PS 0x0000000070bb42af (G-Buffer 6-MRT detail-blend + cubemap SH foliage)

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
layout(set = 0, binding = 2, std430) readonly buffer ssbo_3_buf { float data[]; } ssbo_3;
layout(set = 0, binding = 3, std430) readonly buffer ssbo_4_buf { uint data[]; } ssbo_4;
layout(set = 0, binding = 4, std430) readonly buffer ssbo_5_buf { uint data[]; } ssbo_5;

layout(set = 0, binding = 5) uniform texture2D fs_img0;
layout(set = 0, binding = 6) uniform texture2D fs_img1;
layout(set = 0, binding = 7) uniform texture2D fs_img2;
layout(set = 0, binding = 8) uniform texture2D fs_img3;
layout(set = 0, binding = 9) uniform texture2D fs_img4;
layout(set = 0, binding = 10) uniform texture2D fs_img5;
layout(set = 0, binding = 11) uniform texture2D fs_img6;
layout(set = 0, binding = 12) uniform texture2DArray fs_img7;
layout(set = 0, binding = 13) uniform texture2DArray fs_img8;
layout(set = 0, binding = 14) uniform sampler fs_samp0;
layout(set = 0, binding = 15) uniform sampler fs_samp1;
layout(set = 0, binding = 16) uniform sampler fs_samp2;
layout(set = 0, binding = 17) uniform sampler fs_samp3;

float safe_rsq(float x) { return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0; }
vec3 safe_norm(vec3 v) { return v * safe_rsq(dot(v, v)); }
float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
float s4f(uint i) { return uintBitsToFloat(ssbo_4.data[i]); }
float s5f(uint i) { return uintBitsToFloat(ssbo_5.data[i]); }
vec4 load_s3_v4(uint i) { return vec4(ssbo_3.data[i], ssbo_3.data[i + 1u], ssbo_3.data[i + 2u], ssbo_3.data[i + 3u]); }
float s1_row_dot(uint r, vec4 v) { return s1f(r) * v.x + (s1f(r + 1u) * v.y + (s1f(r + 2u) * v.z + s1f(r + 3u) * v.w)); }
vec2 xform_uv(uint b, vec2 uv) { return uv * vec2(s2f(b), s2f(b + 1u)) + vec2(s2f(b + 2u), s2f(b + 3u)); }

vec3 cube_uv(vec3 r) {
    const vec3 ar = abs(r);
    const bool z_max = (ar.z >= ar.x) && (ar.z >= ar.y);
    const bool y_max = (ar.y >= ar.x);
    const float inv_2a = 1.0 / abs(z_max ? (r.z * 2.0) : (y_max ? (r.y * 2.0) : (r.x * 2.0)));
    const float u_num = z_max ? ((r.z < 0.0) ? -r.x : r.x) : (y_max ? r.x : ((r.x < 0.0) ? r.z : -r.z));
    const float v_num = z_max ? -r.y : (y_max ? ((r.y < 0.0) ? -r.z : r.z) : -r.y);
    const float face = z_max ? ((r.z < 0.0) ? 5.0 : 4.0) : (y_max ? ((r.y < 0.0) ? 3.0 : 2.0) : ((r.x < 0.0) ? 1.0 : 0.0));
    return vec3(u_num * inv_2a + 0.5, v_num * inv_2a + 0.5, fma(floor(face * 0.125), -2.0, face));
}

void main() {
    const uint bo0 = push_data.buf_offsets0.x, bo1 = push_data.buf_offsets0.y;
    const uint d0 = bitfieldExtract(bo0, 0, 8) >> 2u, d1 = bitfieldExtract(bo0, 8, 8) >> 2u;
    const uint d2 = bitfieldExtract(bo0, 16, 8) >> 2u, d3 = bitfieldExtract(bo0, 24, 8) >> 2u;
    const uint d4 = bitfieldExtract(bo1, 0, 8) >> 2u;

    const vec3 n_in = fs_in_attr0_p.xyz;
    const bool is_front = gl_FrontFacing;
    const vec3 N_vert = (is_front ? 1.0 : -1.0) * safe_norm(n_in);

    const vec4 clip_in = fs_in_attr4_p;
    const vec2 screen_uv = fma(vec2(0.5, -0.5) * clip_in.xy, vec2(1.0 / clip_in.w), vec2(0.5));
    const float dither_a = texture(sampler2D(fs_img0, fs_samp0), 0.25 * vec2(s1f(d0 + 4u), s1f(d0 + 5u)) * screen_uv).w - 0.5;

    const vec2 uv0 = fs_in_attr3_p.xy, uv1 = fs_in_attr3_p.zw;
    const vec4 det_alb = texture(sampler2D(fs_img1, fs_samp1), uv1);
    const vec4 base_tex = texture(sampler2D(fs_img2, fs_samp1), xform_uv(d1 + 0u, uv0));
    const uint mat_row = uint(fs_in_attr0_p.w * 1.00001526) * 13u;
    const float alpha_raw = base_tex.w * fs_in_attr2_p.w;
    const float out_alpha = alpha_raw * ssbo_3.data[d2 + (mat_row + 7u) * 4u + 3u];
    if (s4f(d3 + 20u) > (s4f(d3 + 21u) * dither_a + alpha_raw) || out_alpha == 0.0) {
        discard;
    }

    const vec3 wpos = vec3(s1_row_dot(d0 + 20u, clip_in), s1_row_dot(d0 + 24u, clip_in), s1_row_dot(d0 + 28u, clip_in));
    const vec3 V_raw = (-wpos) * (1.0 / s1_row_dot(d0 + 32u, clip_in)) + vec3(s1f(d0 + 183u), s1f(d0 + 187u), s1f(d0 + 191u));
    const vec3 V = safe_norm(V_raw);

    const vec4 nrm_tex = texture(sampler2D(fs_img3, fs_samp2), xform_uv(d1 + 12u, uv0));
    const float gloss_base = texture(sampler2D(fs_img4, fs_samp3), xform_uv(d1 + 8u, uv0)).x;
    const vec2 det_gs = texture(sampler2D(fs_img5, fs_samp3), uv1).xy;
    const vec3 spec_tex = texture(sampler2D(fs_img6, fs_samp3), xform_uv(d1 + 4u, uv0)).rgb;

    const vec3 T = safe_norm(fs_in_attr1_p.xyz);
    const vec3 B = safe_norm(cross(n_in, fs_in_attr1_p.xyz) * fs_in_attr1_p.w);
    const float nx = fma(2.0, nrm_tex.x, -1.0) * fma(2.0, nrm_tex.w, -1.0);
    const float ny = fma(2.0, nrm_tex.y, -1.0);
    const vec3 N = safe_norm(B * nx + (T * ny + N_vert * sqrt(1.0 - clamp(nx * nx + ny * ny, 0.0, 1.0))));

    const float NdotV = dot(N, V);
    const vec3 R = -V + N * (2.0 * NdotV);
    const float det_w = s4f(d3 + 24u);
    const float gloss_scale = s4f(d3 + 8u) * 0.00390625;
    const float gloss = (det_w * ((det_alb.w * gloss_base + det_gs.x) - gloss_base) + gloss_base) * gloss_scale;
    const float mip_lod = min(6.0, 6.5 - gloss * 6.5);
    const vec3 cuv = cube_uv(is_front ? R : vec3(R.x, abs(R.y), R.z));
    const vec3 env0 = textureLod(sampler2DArray(fs_img7, fs_samp0), cuv, mip_lod).rgb;
    const vec3 env1 = textureLod(sampler2DArray(fs_img8, fs_samp0), cuv, mip_lod).rgb;

    const vec4 sh_8 = load_s3_v4(d2 + (mat_row + 8u) * 4u);
    const vec4 sh_9 = load_s3_v4(d2 + (mat_row + 9u) * 4u);
    const vec4 sh_10 = load_s3_v4(d2 + (mat_row + 10u) * 4u);
    const vec4 sh_11 = load_s3_v4(d2 + (mat_row + 11u) * 4u);
    const vec3 sh_dot = vec3(dot(N, sh_9.xyz), dot(N, sh_10.xyz), dot(N, sh_11.xyz));
    const vec3 sh_w = vec3(sh_9.w, sh_10.w, sh_11.w);
    const vec3 sh_amb = max(vec3(0.0), max(sh_w + sh_dot, (sh_w - sh_dot) * 0.85));

    const float r_occ = clamp(dot(R, N_vert) + 1.0, 0.0, 1.0);
    const vec3 env_col = (sh_8.x * (sh_8.z * env0 - sh_8.y * env1) + sh_8.y * env1) * (r_occ * r_occ);
    const vec3 F0_base = (spec_tex.z == 0.0 && spec_tex.y == 0.0) ? vec3(spec_tex.x) : spec_tex;
    const vec3 F0 = det_w * ((det_alb.w * F0_base + vec3(det_gs.y * det_gs.y)) - F0_base) + F0_base;
    const float fresnel = clamp((1.0 + NdotV) * (1.0 + NdotV), 0.0, 1.0) * exp2(5.0 * log2(1.0 - clamp(NdotV, 0.0, 1.0)));
    const vec3 out_spec = (((fresnel - fresnel * F0) * (gloss * gloss) + F0) * env_col) * (vec3(s4f(d3 + 4u), s4f(d3 + 5u), s4f(d3 + 6u)) * s5f(d4 + 12u));

    const vec3 alb_mix = det_w * ((det_alb.w * base_tex.rgb + det_alb.rgb) - base_tex.rgb) + base_tex.rgb;
    const vec3 albedo = alb_mix * vec3(s4f(d3 + 0u), s4f(d3 + 1u), s4f(d3 + 2u));
    const bool blend_alpha = (s4f(d3 + 22u) != 0.0);
    const float a01 = blend_alpha ? out_alpha : 0.0;
    const float mat_id = 0.00392156886 * float(floatBitsToUint(ssbo_3.data[d2 + (mat_row + 6u) * 4u]));
    const float vel_z = ssbo_3.data[d2 + (mat_row + 12u) * 4u];

    frag_color0 = vec4(fma(vec3(0.5), N, vec3(0.500488758)), a01);
    frag_color1 = vec4(gloss_scale * gloss, is_front ? 0.0 : 0.08, vel_z, a01);
    frag_color2 = vec4(albedo, out_alpha);
    frag_color3 = vec4(sqrt(F0), blend_alpha ? out_alpha : mat_id);
    frag_color4 = vec4(-F0 * sh_amb + sh_amb, blend_alpha ? out_alpha : 1.0);
    frag_color5 = vec4(out_spec, blend_alpha ? out_alpha : sh_8.w);
}
