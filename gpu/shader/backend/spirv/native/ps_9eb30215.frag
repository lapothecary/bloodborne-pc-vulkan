// SPDX-License-Identifier: GPL-2.0-or-later
// Step 2.3: Native Vulkan 1.3 GLSL 460 replacement for ps 0x000000009eb30215
// (6b. Forward Alpha / Soft Particle Constant-Color Fill Pass).
#version 460

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

void main() {
    const uint d0 = bitfieldExtract(push_data.buf_offsets0.x, 0, 8) >> 2u;
    frag_color0 = vec4(uintBitsToFloat(ssbo_1.data[d0 + 0u]),
                       uintBitsToFloat(ssbo_1.data[d0 + 1u]),
                       uintBitsToFloat(ssbo_1.data[d0 + 2u]),
                       uintBitsToFloat(ssbo_1.data[d0 + 3u]));
}
