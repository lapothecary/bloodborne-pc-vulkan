// SPDX-License-Identifier: GPL-2.0-or-later
// Step 2.3: Native Vulkan 1.3 GLSL 460 replacement for ps 0x00000000abf92450
// (8b. Motion Blur 8-Neighbor Directional Velocity Tile Dilation Pass).
// Streams all 8 compass neighbors sequentially (~8 VGPRs vs ~55 VGPRs in auto-SPIR-V).
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

const vec2 kDirs[8] = vec2[8](
    vec2( 0.0,          1.0),
    vec2( 0.707106829,  0.707106769),
    vec2( 1.0,          0.0),
    vec2( 0.707106769, -0.707106829),
    vec2( 0.0,         -1.0),
    vec2(-0.707106829, -0.707106769),
    vec2(-1.0,          0.0),
    vec2(-0.707106769,  0.707106829)
);

void main() {
    const uint d0 = bitfieldExtract(push_data.buf_offsets0.x, 0, 8) >> 2u;
    const vec2 scale_dir = vec2(uintBitsToFloat(ssbo_1.data[d0 + 28u]),
                                uintBitsToFloat(ssbo_1.data[d0 + 29u]));
    const vec2 step_uv = vec2(uintBitsToFloat(ssbo_1.data[d0 + 30u]),
                              uintBitsToFloat(ssbo_1.data[d0 + 31u]));
    const vec2 uv = fs_in_attr0_p.xy;

    const vec4 c = textureLod(sampler2D(fs_img0, fs_samp0), uv, 0.0);
    vec2 best_v = fma(c.xy, vec2(0.400000006), vec2(-0.199215695));
    float best_z = c.z;
    float best_w = c.w;

    for (int i = 0; i < 8; ++i) {
        const vec2 dir = kDirs[i];
        const vec4 s = textureLod(sampler2D(fs_img0, fs_samp0), fma(step_uv, dir, uv), 0.0);
        const vec2 v = fma(s.xy, vec2(0.400000006), vec2(-0.199215695));
        if (best_z > s.z && dot(dir, scale_dir * v) > 0.0) {
            best_v = v;
            best_z = s.z;
            best_w = s.w;
        }
    }

    frag_color0 = vec4(fma(best_v, vec2(2.5), vec2(0.5)), best_z, best_w);
}
