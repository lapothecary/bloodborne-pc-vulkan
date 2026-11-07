#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Bloodborne Native GLSL 460 — 0x8650418f (Dual-Texture RGB Modulate)
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

layout(set = 0, binding = 0) uniform texture2D fs_img0;
layout(set = 0, binding = 1) uniform texture2D fs_img1;
layout(set = 0, binding = 2) uniform sampler fs_samp0;

void main() {
    vec2 uv = fs_in_attr0_p.xy;
    vec4 c0 = texture(sampler2D(fs_img0, fs_samp0), uv);
    vec4 c1 = texture(sampler2D(fs_img1, fs_samp0), uv);
    frag_color0 = vec4(c0.xyz * c1.xyz, c1.w);
}
