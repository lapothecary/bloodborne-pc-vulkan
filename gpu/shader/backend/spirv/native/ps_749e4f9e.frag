#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Bloodborne Native GLSL 460 — 0x749e4f9e (Alpha-Tested Screen-Space Velocity + Depth)
layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 3) in vec4 fs_in_attr2_p;

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

layout(set = 0, binding = 2) uniform texture2D fs_img0;
layout(set = 0, binding = 3) uniform sampler fs_samp0;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }

void main() {
    uint u0 = push_data.buf_offsets0.x;
    uint d0 = bitfieldExtract(u0, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(u0, 8, 8) >> 2u;

    vec2 uv = vec2(s1f(d0 + 0u), s1f(d0 + 1u)) * fs_in_attr2_p.xy + vec2(s1f(d0 + 2u), s1f(d0 + 3u));
    float tex_a = texture(sampler2D(fs_img0, fs_samp0), uv).w;
    if (0.00392156886 > tex_a) {
        discard;
    }

    const float fps_q = float((push_data.buf_offsets2.y >> 8u) & 0xFFu) * 0.0625;
    const float obj_fps_scale = (fps_q >= 1.0) ? fps_q : 1.0;
    const bool native_postfx = ((push_data.buf_offsets2.y >> 20u) & 1u) != 0u;
    const vec2 enc_scale = native_postfx ? vec2(4.0) : vec2(s2f(d1 + 0u), s2f(d1 + 1u));

    vec2 cur_ndc  = fs_in_attr0_p.xy * (1.0 / fs_in_attr0_p.w);
    vec2 prev_ndc = fs_in_attr1_p.xy * (1.0 / fs_in_attr1_p.w);
    vec2 vel = clamp((enc_scale * (cur_ndc - prev_ndc) * obj_fps_scale) * 0.5 + 0.5, 0.0, 1.0);
    float depth_norm = fs_in_attr0_p.w * (1.0 / (10.0 + fs_in_attr0_p.w));

    frag_color0 = vec4(vel, depth_norm, 1.0);
}
