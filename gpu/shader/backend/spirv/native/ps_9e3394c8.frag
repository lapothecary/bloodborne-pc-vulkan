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

// G-Buffer Foliage (2-sided, opaque, alpha-weighted color bias) — hash 0x9e3394c8
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

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { uint data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { uint data[]; } ssbo_2;
layout(set = 0, binding = 2, std430) readonly buffer ssbo_3_buf { uint data[]; } ssbo_3;
layout(set = 0, binding = 3, std430) readonly buffer ssbo_4_buf { float data[]; } ssbo_4;
layout(set = 0, binding = 4, std430) readonly buffer ssbo_5_buf { uint data[]; } ssbo_5;
layout(set = 0, binding = 5, std430) readonly buffer ssbo_6_buf { uint data[]; } ssbo_6;

layout(set = 0, binding = 6) uniform texture2D fs_img0;
layout(set = 0, binding = 7) uniform texture2D fs_img1;
layout(set = 0, binding = 8) uniform texture2D fs_img2;
layout(set = 0, binding = 9) uniform texture2DArray fs_img3;
layout(set = 0, binding = 10) uniform texture2DArray fs_img4;
layout(set = 0, binding = 11) uniform texture2D fs_img5;
layout(set = 0, binding = 12) uniform sampler fs_samp0;
layout(set = 0, binding = 13) uniform sampler fs_samp1;
layout(set = 0, binding = 14) uniform sampler fs_samp2;
layout(set = 0, binding = 15) uniform sampler fs_samp3;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
float s3f(uint i) { return uintBitsToFloat(ssbo_3.data[i]); }
float s5f(uint i) { return uintBitsToFloat(ssbo_5.data[i]); }
float s6f(uint i) { return uintBitsToFloat(ssbo_6.data[i]); }

vec4 load_s1_v4(uint i) { return vec4(s1f(i), s1f(i + 1u), s1f(i + 2u), s1f(i + 3u)); }
vec4 load_s4_v4(uint i) { return vec4(ssbo_4.data[i], ssbo_4.data[i + 1u], ssbo_4.data[i + 2u], ssbo_4.data[i + 3u]); }

vec3 safe_normalize(vec3 v) {
    float l2 = dot(v, v);
    float rsq = (l2 > 0.0 && l2 <= 3.40282347e+38) ? inversesqrt(clamp(l2, 1.17549435e-38, 3.40282347e+38)) : 0.0;
    return v * rsq;
}

vec2 xform_uv2(uint base, vec2 uv) {
    return vec2(uv.x * s2f(base) + s2f(base + 2u), uv.y * s2f(base + 1u) + s2f(base + 3u));
}

vec3 cubemap_coords(vec3 r) {
    vec3 ar = abs(r);
    bool z_maj = (ar.z >= ar.x) && (ar.z >= ar.y);
    bool y_maj = (ar.y >= ar.x);
    float ma = z_maj ? (r.z * 2.0) : (y_maj ? (r.y * 2.0) : (r.x * 2.0));
    float inv_ma = 1.0 / abs(ma);
    float sc = z_maj ? ((r.z < 0.0) ? -r.x : r.x) : (y_maj ? r.x : ((r.x < 0.0) ? r.z : -r.z));
    float tc = z_maj ? -r.y : (y_maj ? ((r.y < 0.0) ? -r.z : r.z) : -r.y);
    float face = z_maj ? ((r.z < 0.0) ? 5.0 : 4.0) : (y_maj ? ((r.y < 0.0) ? 3.0 : 2.0) : ((r.x < 0.0) ? 1.0 : 0.0));
    float layer = fma(floor(face * 0.125), -2.0, face);
    return vec3(sc * inv_ma + 0.5, tc * inv_ma + 0.5, layer);
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

    vec2 uv = fs_in_attr4_p.xy;
    vec4 tex_nrm = texture(sampler2D(fs_img0, fs_samp0), xform_uv2(d1 + 12u, uv));
    vec4 tex_rgh = texture(sampler2D(fs_img1, fs_samp1), xform_uv2(d1 + 8u, uv));
    vec4 tex_spec = texture(sampler2D(fs_img2, fs_samp1), xform_uv2(d1 + 4u, uv));
    vec4 tex_alb = texture(sampler2D(fs_img5, fs_samp3), xform_uv2(d1 + 0u, uv));

    bool front = gl_FrontFacing;
    vec3 N_vert = safe_normalize(fs_in_attr0_p.xyz) * (front ? 1.0 : -1.0);
    vec3 T_vert = safe_normalize(fs_in_attr1_p.xyz);
    vec3 B_vert = safe_normalize(cross(fs_in_attr0_p.xyz, fs_in_attr1_p.xyz)) * fs_in_attr1_p.w;

    float nx = fma(2.0, tex_nrm.x, -1.0) * fma(2.0, tex_nrm.w, -1.0);
    float ny = fma(2.0, tex_nrm.y, -1.0);
    float nz = sqrt(1.0 - clamp(nx * nx + ny * ny, 0.0, 1.0));
    vec3 N = safe_normalize(B_vert * nx + T_vert * ny + N_vert * nz);

    float inv_w = 1.0 / dot(load_s1_v4(d0 + 32u), fs_in_attr3_p);
    vec3 cam_pos = vec3(s1f(d0 + 183u), s1f(d0 + 187u), s1f(d0 + 191u));
    vec3 view_vec = vec3(
        -dot(load_s1_v4(d0 + 20u), fs_in_attr3_p) * inv_w + cam_pos.x,
        -dot(load_s1_v4(d0 + 24u), fs_in_attr3_p) * inv_w + cam_pos.y,
        -dot(load_s1_v4(d0 + 28u), fs_in_attr3_p) * inv_w + cam_pos.z
    );
    vec3 V = safe_normalize(view_vec);
    vec3 R = -V + N * (dot(N, V) * 2.0);

    float gloss_scale = s3f(d2 + 8u) * 0.00390625;
    float gloss = tex_rgh.x * gloss_scale;
    float mip = min(6.0, 6.5 - gloss * 6.5);
    vec3 cube_tc = cubemap_coords(front ? R : vec3(R.x, abs(R.y), R.z));
    vec4 env0 = textureLod(sampler2DArray(fs_img3, fs_samp2), cube_tc, mip);
    vec4 env1 = textureLod(sampler2DArray(fs_img4, fs_samp2), cube_tc, mip);

    uint inst_base = uint(fs_in_attr0_p.w * 1.00001526) * 13u;
    vec4 env_w = load_s4_v4((inst_base + 8u) * 4u + d3);
    float horizon = clamp(dot(R, N_vert) + 1.0, 0.0, 1.0);
    vec3 env_col = (env_w.x * (-env_w.y * env1.xyz + env_w.z * env0.xyz) + env_w.y * env1.xyz) * (horizon * horizon);

    bool mono_spec = (tex_spec.z == 0.0) && (tex_spec.y == 0.0);
    vec3 spec_col = vec3(tex_spec.x, mono_spec ? tex_spec.x : tex_spec.y, mono_spec ? tex_spec.x : tex_spec.z);
    vec3 spec_scale = vec3(s3f(d2 + 4u), s3f(d2 + 5u), s3f(d2 + 6u)) * s6f(d5 + 12u);

    vec4 sh_r = load_s4_v4((inst_base + 9u) * 4u + d3);
    vec4 sh_g = load_s4_v4((inst_base + 10u) * 4u + d3);
    vec4 sh_b = load_s4_v4((inst_base + 11u) * 4u + d3);
    float dr = dot(N, sh_r.xyz);
    float dg = dot(N, sh_g.xyz);
    float db = dot(N, sh_b.xyz);
    vec3 sh_amb = vec3(
        max(0.0, max(sh_r.w + dr, (sh_r.w - dr) * 0.85)),
        max(0.0, max(sh_g.w + dg, (sh_g.w - dg) * 0.85)),
        max(0.0, max(sh_b.w + db, (sh_b.w - db) * 0.85))
    );

    vec3 col_bias = vec3(s5f(d4 + 24u), s5f(d4 + 25u), s5f(d4 + 26u));
    vec3 alb_sqrt = clamp(col_bias * tex_alb.w + sqrt(tex_alb.xyz), 0.0, 1.0);
    vec3 albedo_tint = vec3(s3f(d2 + 0u), s3f(d2 + 1u), s3f(d2 + 2u));
    vec3 albedo = albedo_tint * (alb_sqrt * alb_sqrt);

    float alpha_out = fs_in_attr2_p.w * ssbo_4.data[(inst_base + 7u) * 4u + 3u + d3];
    bool use_alpha = (s3f(d2 + 22u) != 0.0);
    float alpha_opt = use_alpha ? alpha_out : 0.0;

    frag_color0 = vec4(fma(vec3(0.5), N, vec3(0.500488758)), alpha_opt);
    frag_color1 = vec4(gloss_scale * gloss, front ? 0.0 : 0.08, ssbo_4.data[(inst_base + 12u) * 4u + d3], alpha_opt);
    frag_color2 = vec4(albedo, alpha_out);
    frag_color3 = vec4(
        sqrt(spec_col),
        use_alpha ? alpha_out : (0.00392156886 * float(floatBitsToUint(ssbo_4.data[(inst_base + 6u) * 4u + d3])))
    );
    frag_color4 = vec4(-spec_col * sh_amb + sh_amb, use_alpha ? alpha_out : 1.0);
    frag_color5 = vec4((spec_col * env_col) * spec_scale, use_alpha ? alpha_out : env_w.w);
}
