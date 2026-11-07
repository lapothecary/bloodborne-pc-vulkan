#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Bloodborne Native GLSL 460 — 0x765450df (Dithered Alpha-Test Forward + Distance/Height Fog)
layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 2) in vec4 fs_in_attr2_p;
layout(location = 3) in vec4 fs_in_attr3_p;
layout(location = 4) in vec4 fs_in_attr4_p;

layout(location = 0) out vec4 frag_color0;

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
layout(set = 0, binding = 7) uniform sampler fs_samp0;
layout(set = 0, binding = 8) uniform sampler fs_samp1;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
float s4f(uint i) { return uintBitsToFloat(ssbo_4.data[i]); }
float s5f(uint i) { return uintBitsToFloat(ssbo_5.data[i]); }

void main() {
    uint u0 = push_data.buf_offsets0.x;
    uint d0 = bitfieldExtract(u0, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(u0, 8, 8) >> 2u;
    uint d2 = bitfieldExtract(u0, 16, 8) >> 2u;
    uint d3 = bitfieldExtract(u0, 24, 8) >> 2u;
    uint d4 = bitfieldExtract(push_data.buf_offsets0.y, 0, 8) >> 2u;

    uint inst_idx = uint(fs_in_attr0_p.w * 1.00001526);
    uint sb3_base = (inst_idx * 13u + 7u) * 4u + d2;
    vec4 inst_col = vec4(ssbo_3.data[sb3_base], ssbo_3.data[sb3_base + 1u], ssbo_3.data[sb3_base + 2u], ssbo_3.data[sb3_base + 3u]);

    float inv_w = 1.0 / fs_in_attr3_p.w;
    vec2 scr_uv = vec2(fma(0.5 * fs_in_attr3_p.x, inv_w, 0.5), fma(-0.5 * fs_in_attr3_p.y, inv_w, 0.5));
    vec2 dither_uv = 0.25 * (vec2(s1f(d0 + 4u), s1f(d0 + 5u)) * scr_uv);
    float dither_a = texture(sampler2D(fs_img0, fs_samp0), dither_uv).w;

    vec2 alb_uv = fs_in_attr4_p.xy * vec2(s2f(d1 + 0u), s2f(d1 + 1u)) + vec2(s2f(d1 + 2u), s2f(d1 + 3u));
    vec4 tex = texture(sampler2D(fs_img1, fs_samp1), alb_uv);

    float base_a = tex.w * fs_in_attr2_p.w;
    float out_a = base_a * inst_col.w;
    if ((s4f(d3 + 20u) > (s4f(d3 + 21u) * (-0.5 + dither_a) + base_a)) || (out_a == 0.0)) {
        discard;
    }

    vec3 mat_tint = vec3(s4f(d3 + 0u), s4f(d3 + 1u), s4f(d3 + 2u)) * tex.xyz;
    vec3 col = mat_tint * inst_col.xyz;
    float luma_m1 = -1.0 + (col.z * 0.114480004 + (col.x * 0.298909992 + 0.586610019 * col.y));

    float xz_dist = sqrt(fs_in_attr1_p.x * fs_in_attr1_p.x + fs_in_attr1_p.z * fs_in_attr1_p.z);
    float dist_fog_t = clamp(s5f(d4 + 8u) + s5f(d4 + 8u) * (-exp2(-(s5f(d4 + 10u) * (xz_dist - s5f(d4 + 9u))))), 0.0, 1.0);
    vec3 dist_fog_col = vec3(s5f(d4 + 4u), s5f(d4 + 5u), s5f(d4 + 6u)) * (s5f(d4 + 7u) * luma_m1 + 1.0);
    vec3 col_df = dist_fog_t * ((-mat_tint) * inst_col.xyz + dist_fog_col) + col;

    float h_fog_raw = clamp(s5f(d4 + 14u) * ((s5f(d4 + 15u) - fs_in_attr1_p.y) - s5f(d4 + 13u)), 0.0, 1.0);
    float h_fog_t = clamp(s5f(d4 + 12u) * h_fog_raw, 0.0, 1.0);
    vec3 h_fog_col = vec3(s5f(d4 + 16u), s5f(d4 + 17u), s5f(d4 + 18u)) * (s5f(d4 + 19u) * luma_m1 + 1.0);
    vec3 out_rgb = h_fog_t * (h_fog_col - col_df) + col_df;

    frag_color0 = vec4(out_rgb, out_a);
}
