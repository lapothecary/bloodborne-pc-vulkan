#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Bloodborne Native GLSL 460 — 0xccbf44a6 (Post-FX Non-Negative Clamp)
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

layout(set = 0, binding = 0) uniform texture2D fs_img0;
layout(set = 0, binding = 1) uniform sampler fs_samp0;

void main() {
    vec4 tex = textureLod(sampler2D(fs_img0, fs_samp0), fs_in_attr0_p.xy, 0.0);
    frag_color0 = max(vec4(0.0), tex);
}
