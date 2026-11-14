// SPDX-License-Identifier: GPL-2.0-or-later
// Phase 2.3: Native Vulkan 1.3 GLSL 460 replacement for ps 0x00000000ff902740
// (Yebis 2 DoF CoC Prefilter Pass).
#version 460
#extension GL_EXT_samplerless_texture_functions : require

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
layout(set = 0, binding = 2) uniform sampler fs_samp0;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }

void main() {
    const uint d0 = (push_data.buf_offsets0.x & 0xFFu) >> 2u;
    const vec4 c = textureLod(sampler2D(fs_img0, fs_samp0), fs_in_attr0_p.xy, 0.0);
    const float coc = max(0.0, fma(s1f(d0 + 619u), c.w, s1f(d0 + 623u)));
    frag_color0 = vec4(c.rgb, clamp(0.25 * (coc * coc), 0.0, 1.0));
}
