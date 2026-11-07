#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Bloodborne Native GLSL 460 — 0x5e4596f4 (UI/VFX BGRA + Alpha Modulate)
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

layout(set = 0, binding = 0) uniform texture2D fs_img0;
layout(set = 0, binding = 1) uniform sampler fs_samp0;

void main() {
    float tex_a = texture(sampler2D(fs_img0, fs_samp0), fs_in_attr1_p.xy).w;
    frag_color0 = vec4(fs_in_attr0_p.zyx, fs_in_attr0_p.w * tex_a);
}
