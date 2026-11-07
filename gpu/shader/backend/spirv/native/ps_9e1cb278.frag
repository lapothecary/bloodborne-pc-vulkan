#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Bloodborne Native GLSL 460 — 0x9e1cb278 (Post-FX CoC / Alpha Lerp)
layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;

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

    vec4 c0 = textureLod(sampler2D(fs_img0, fs_samp0), fs_in_attr0_p.xy, 0.0);
    vec4 c1 = textureLod(sampler2D(fs_img1, fs_samp1), fs_in_attr1_p.xy, 0.0);

    float blend = clamp(2.0 - (s1f(d0 + 76u) * c0.w + s1f(d0 + 77u)), 0.0, 1.0);
    frag_color0 = vec4(blend * (c0.xyz - c1.xyz) + c1.xyz, c0.w);
}
