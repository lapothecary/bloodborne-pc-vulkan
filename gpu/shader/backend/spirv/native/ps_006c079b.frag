#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Bloodborne Native GLSL 460 — 0x006c079b (Depth-Only Alpha Test)
layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;

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

    vec2 uv = fs_in_attr1_p.zw * vec2(s1f(d0 + 0u), s1f(d0 + 1u)) + vec2(s1f(d0 + 2u), s1f(d0 + 3u));
    float alpha = texture(sampler2D(fs_img0, fs_samp0), uv).w * fs_in_attr0_p.w;
    if ((s2f(d1 + 20u) > (s2f(d1 + 21u) * -0.5 + alpha)) || (alpha == 0.0)) {
        discard;
    }
}
