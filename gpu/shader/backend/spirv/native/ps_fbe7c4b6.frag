#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Bloodborne Native GLSL 460 — 0xfbe7c4b6 (Post-FX LUT / Color Scale)
layout(location = 0) in vec4 fs_in_attr0_p;

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
layout(set = 0, binding = 3) uniform texture2D fs_img2;
layout(set = 0, binding = 4) uniform sampler fs_samp0;
layout(set = 0, binding = 5) uniform sampler fs_samp1;
layout(set = 0, binding = 6) uniform sampler fs_samp2;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }

void main() {
    uint d0 = bitfieldExtract(push_data.buf_offsets0.x, 0, 8) >> 2u;

    vec2 uv = fs_in_attr0_p.xy;
    vec4 c0 = textureLod(sampler2D(fs_img0, fs_samp0), uv, 0.0);
    float luma = textureLod(sampler2D(fs_img1, fs_samp1), uv, 0.0).x;

    vec2 lut_uv = vec2(s1f(d0 + 0u) * luma + s1f(d0 + 1u), s1f(d0 + 200u));
    float lut_x = textureLod(sampler2D(fs_img2, fs_samp2), lut_uv, 0.0).x;

    vec3 rgb = (s1f(d0 + 1024u) == 1.0)
             ? (vec3(s1f(d0 + 616u), s1f(d0 + 617u), s1f(d0 + 618u)) * c0.xyz)
             : vec3(0.0);
    float a = s1f(d0 + 68u) * lut_x + s1f(d0 + 69u);

    frag_color0 = vec4(rgb, a);
}
