#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Bloodborne Native GLSL 460 — 0xdda49e89 (Opaque/Blended Forward + Distance/Height Fog)
layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 2) in vec4 fs_in_attr2_p;
layout(location = 4) in vec4 fs_in_attr3_p;

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
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { float data[]; } ssbo_2;
layout(set = 0, binding = 2, std430) readonly buffer ssbo_3_buf { uint data[]; } ssbo_3;
layout(set = 0, binding = 3, std430) readonly buffer ssbo_4_buf { uint data[]; } ssbo_4;

layout(set = 0, binding = 4) uniform texture2D fs_img0;
layout(set = 0, binding = 5) uniform sampler fs_samp0;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s3f(uint i) { return uintBitsToFloat(ssbo_3.data[i]); }
float s4f(uint i) { return uintBitsToFloat(ssbo_4.data[i]); }

void main() {
    uint u0 = push_data.buf_offsets0.x;
    uint d0 = bitfieldExtract(u0, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(u0, 8, 8) >> 2u;
    uint d2 = bitfieldExtract(u0, 16, 8) >> 2u;
    uint d3 = bitfieldExtract(u0, 24, 8) >> 2u;

    uint inst_idx = uint(fs_in_attr0_p.w * 1.00001526);
    uint sb2_base = (inst_idx * 13u + 7u) * 4u + d1;
    vec4 inst_col = vec4(ssbo_2.data[sb2_base], ssbo_2.data[sb2_base + 1u], ssbo_2.data[sb2_base + 2u], ssbo_2.data[sb2_base + 3u]);

    vec2 alb_uv = fs_in_attr3_p.xy * vec2(s1f(d0 + 0u), s1f(d0 + 1u)) + vec2(s1f(d0 + 2u), s1f(d0 + 3u));
    vec4 tex = texture(sampler2D(fs_img0, fs_samp0), alb_uv);

    vec3 mat_tint = vec3(s3f(d2 + 0u), s3f(d2 + 1u), s3f(d2 + 2u)) * tex.xyz;
    vec3 col = mat_tint * inst_col.xyz;
    float luma_m1 = -1.0 + (col.z * 0.114480004 + (col.x * 0.298909992 + 0.586610019 * col.y));

    float xz_dist = sqrt(fs_in_attr1_p.x * fs_in_attr1_p.x + fs_in_attr1_p.z * fs_in_attr1_p.z);
    float dist_fog_t = clamp(s4f(d3 + 8u) + s4f(d3 + 8u) * (-exp2(-(s4f(d3 + 10u) * (xz_dist - s4f(d3 + 9u))))), 0.0, 1.0);
    vec3 dist_fog_col = vec3(s4f(d3 + 4u), s4f(d3 + 5u), s4f(d3 + 6u)) * (s4f(d3 + 7u) * luma_m1 + 1.0);
    vec3 col_df = dist_fog_t * ((-mat_tint) * inst_col.xyz + dist_fog_col) + col;

    float h_fog_raw = clamp(s4f(d3 + 14u) * ((s4f(d3 + 15u) - fs_in_attr1_p.y) - s4f(d3 + 13u)), 0.0, 1.0);
    float h_fog_t = clamp(s4f(d3 + 12u) * h_fog_raw, 0.0, 1.0);
    vec3 h_fog_col = vec3(s4f(d3 + 16u), s4f(d3 + 17u), s4f(d3 + 18u)) * (s4f(d3 + 19u) * luma_m1 + 1.0);
    vec3 out_rgb = h_fog_t * (h_fog_col - col_df) + col_df;
    float out_a = (tex.w * fs_in_attr2_p.w) * inst_col.w;

    frag_color0 = vec4(out_rgb, out_a);
}
