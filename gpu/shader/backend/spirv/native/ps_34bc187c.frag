// SPDX-License-Identifier: GPL-2.0-or-later
// Native Vulkan 1.3 GLSL 460 replacement for ps 0x0000000034bc187c
// (Pass 0x2e Opaque Object-Velocity + Depth encoder, R8G8B8A8Unorm).
// Pre-scales per-frame (cur_ndc - prev_ndc) by FPS normalization factor (buf_offsets[37] in Q4.4)
// before 8-bit UNORM quantization so high-FPS (90..300+ FPS) weapon swings/dodges retain full
// 30-FPS 8-bit precision instead of quantizing to 0x7F/0x80.
#version 460

layout(location = 0) in vec4 fs_in_attr0; // cur clip pos
layout(location = 1) in vec4 fs_in_attr1; // prev clip pos
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

layout(set = 0, binding = 0, std430) readonly buffer SSBO1 {
    uint data[];
} ssbo_1;

void main() {
    const uint b0 = (push_data.buf_offsets0.x & 0xFFu) >> 2u;
    const vec2 vel_scale = vec2(uintBitsToFloat(ssbo_1.data[b0 + 0u]),
                                uintBitsToFloat(ssbo_1.data[b0 + 1u]));

    const float fps_q = float((push_data.buf_offsets2.y >> 8u) & 0xFFu) * 0.0625;
    const float obj_fps_scale = (fps_q >= 1.0) ? fps_q : 1.0;
    const bool native_postfx = ((push_data.buf_offsets2.y >> 20u) & 1u) != 0u;
    const vec2 enc_scale = native_postfx ? vec2(4.0) : vel_scale;

    const vec2 cur_ndc  = fs_in_attr0.xy * (1.0 / fs_in_attr0.w);
    const vec2 prev_ndc = fs_in_attr1.xy * (1.0 / fs_in_attr1.w);
    const vec2 vel = (cur_ndc - prev_ndc) * enc_scale * obj_fps_scale;
    const vec2 enc = clamp(fma(vel, vec2(0.5), vec2(0.5)), vec2(0.0), vec2(1.0));
    const float z_enc = fs_in_attr0.w / (10.0 + fs_in_attr0.w);

    frag_color0 = vec4(enc.x, enc.y, z_enc, 1.0);
}
