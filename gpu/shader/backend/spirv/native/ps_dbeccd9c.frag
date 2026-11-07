#version 460
#extension GL_EXT_samplerless_texture_functions : require
// Native Vulkan 1.3 GLSL PS 0x00000000dbeccd9c (G-Buffer unbaked cutout 6-MRT foliage)

layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 2) in vec4 fs_in_attr2_p;
layout(location = 3) in vec4 fs_in_attr3_p;

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

layout(set = 0, binding = 4) uniform texture2D fs_img0;
layout(set = 0, binding = 5) uniform texture2D fs_img1;
layout(set = 0, binding = 6) uniform sampler fs_samp0;
layout(set = 0, binding = 7) uniform sampler fs_samp1;

float safe_rsq(float x) {
    return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0;
}
float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
float s4f(uint i) { return uintBitsToFloat(ssbo_4.data[i]); }
vec4 load_s3_v4(uint i) {
    return vec4(ssbo_3.data[i], ssbo_3.data[i + 1u], ssbo_3.data[i + 2u], ssbo_3.data[i + 3u]);
}

void main() {
    const uint bo0 = push_data.buf_offsets0.x;
    const uint d0 = bitfieldExtract(bo0, 0, 8) >> 2u;
    const uint d1 = bitfieldExtract(bo0, 8, 8) >> 2u;
    const uint d2 = bitfieldExtract(bo0, 16, 8) >> 2u;
    const uint d3 = bitfieldExtract(bo0, 24, 8) >> 2u;

    const vec3 n_in = fs_in_attr0_p.xyz;
    const float n_len2 = dot(n_in, n_in);
    const bool is_front = gl_FrontFacing;
    const vec3 N_world = (is_front ? 1.0 : -1.0) * (n_in * safe_rsq(n_len2));

    const vec2 screen_uv = fma(vec2(0.5, -0.5) * fs_in_attr2_p.xy, vec2(1.0 / fs_in_attr2_p.w), vec2(0.5));
    const vec2 dither_uv = 0.25 * vec2(s1f(d0 + 4u), s1f(d0 + 5u)) * screen_uv;
    const float dither_a = texture(sampler2D(fs_img0, fs_samp0), dither_uv).w - 0.5;

    const vec2 uv = fs_in_attr3_p.xy * vec2(s2f(d1 + 0u), s2f(d1 + 1u)) + vec2(s2f(d1 + 2u), s2f(d1 + 3u));
    const vec4 base_tex = texture(sampler2D(fs_img1, fs_samp1), uv);

    const uint mat_row = uint(fs_in_attr0_p.w * 1.00001526) * 13u;
    const float alpha_raw = base_tex.w * fs_in_attr1_p.w;
    const float out_alpha = alpha_raw * ssbo_3.data[d2 + (mat_row + 7u) * 4u + 3u];
    if (s4f(d3 + 20u) > (s4f(d3 + 21u) * dither_a + alpha_raw) || out_alpha == 0.0) {
        discard;
    }

    const vec4 sh_r = load_s3_v4(d2 + (mat_row + 9u) * 4u);
    const vec4 sh_g = load_s3_v4(d2 + (mat_row + 10u) * 4u);
    const vec4 sh_b = load_s3_v4(d2 + (mat_row + 11u) * 4u);
    const vec3 sh_dot = vec3(dot(N_world, sh_r.xyz), dot(N_world, sh_g.xyz), dot(N_world, sh_b.xyz));
    const vec3 sh_w = vec3(sh_r.w, sh_g.w, sh_b.w);
    const vec3 sh_amb = max(vec3(0.0), max(sh_w + sh_dot, (sh_w - sh_dot) * 0.85));

    const vec3 albedo = base_tex.rgb * vec3(s4f(d3 + 0u), s4f(d3 + 1u), s4f(d3 + 2u));
    const bool blend_alpha = (s4f(d3 + 22u) != 0.0);
    const float a01 = blend_alpha ? out_alpha : 0.0;
    const float mat_id = 0.00392156886 * float(floatBitsToUint(ssbo_3.data[d2 + (mat_row + 6u) * 4u]));
    const float amb_w = ssbo_3.data[d2 + (mat_row + 8u) * 4u + 3u];
    const float vel_z = ssbo_3.data[d2 + (mat_row + 12u) * 4u];

    frag_color0 = vec4(fma(vec3(0.5), N_world, vec3(0.500488758)), a01);
    frag_color1 = vec4(0.0, is_front ? 0.0 : 0.08, vel_z, a01);
    frag_color2 = vec4(albedo, out_alpha);
    frag_color3 = vec4(0.0, 0.0, 0.0, blend_alpha ? out_alpha : mat_id);
    frag_color4 = vec4(sh_amb, blend_alpha ? out_alpha : 1.0);
    frag_color5 = vec4(0.0, 0.0, 0.0, blend_alpha ? out_alpha : amb_w);
}
