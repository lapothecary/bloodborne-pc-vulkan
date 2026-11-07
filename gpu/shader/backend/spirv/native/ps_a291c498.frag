#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Bloodborne Native GLSL 460 — 0xa291c498 (Depth / CoC Linearization)
layout(location = 1) in vec4 fs_in_attr0_p;

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

layout(set = 0, binding = 1) uniform texture2D fs_img0;
layout(set = 0, binding = 2) uniform texture2D fs_img1;
layout(set = 0, binding = 3) uniform sampler fs_samp0;
layout(set = 0, binding = 4) uniform sampler fs_samp1;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }

void main() {
    uint d0 = bitfieldExtract(push_data.buf_offsets0.x, 0, 8) >> 2u;

    vec2 uv = fs_in_attr0_p.xy;
    float z  = texture(sampler2D(fs_img0, fs_samp0), uv).z;
    float x1 = texture(sampler2D(fs_img1, fs_samp1), uv).x;
    vec2 coc = vec2(s1f(d0 + 1u), s1f(d0 + 2u)) * z + vec2(s1f(d0 + 5u), s1f(d0 + 6u));

    frag_color0 = vec4(x1, coc, 1.0);
}
