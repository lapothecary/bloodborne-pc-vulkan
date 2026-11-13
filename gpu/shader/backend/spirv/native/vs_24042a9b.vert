// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 11: Native Vulkan 1.3 GLSL replacement for Scaleform GFx 2D TexGen + CxForm VS (`0x0000000024042a9b`).
// Supports both:
//   - Batched multi-quad Scaleform draws (`vs_in_attr1.w < -0.5`, encoding `-float(quad_idx + 1)`
//     in `vs_in_attr1.w`), indexing `cb0.data[buf_dword_off + quad_idx * 24u + 0..23]` so an
//     arbitrary run of HUD sprites sharing the same atlas texture & blend state coalesces into
//     a single `vkCmdDraw` without fragment shader changes.
//   - Unbatched Scaleform draws (`vs_in_attr1.w >= -0.5`, where 2-component guest `vs_in_attr1`
//     supplies `w == +1.0`), reading `cb0.data[buf_dword_off + 0..23]` identically to the PS4 ISA.
#version 460

layout(constant_id = 0) const uint kBufSlot = 0u;

layout(location = 0) in vec4 vs_in_attr0;
layout(location = 1) in vec4 vs_in_attr1;

layout(location = 0) out vec4 out_attr0;
layout(location = 1) out vec4 out_attr1;
layout(location = 2) out vec4 out_attr2;
layout(location = 3) out vec4 out_attr3;

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

layout(set = 0, binding = 0, std430) readonly buffer SSBO0 {
    uint data[];
} cb0;

void main() {
    uint slot = kBufSlot;
    const uint word_idx = slot >> 2u;
    const int bit_shift = int((slot & 3u) << 3u);
    uint packed_off = 0u;
    if (word_idx == 0u) {
        packed_off = push_data.buf_offsets0.x;
    } else if (word_idx == 1u) {
        packed_off = push_data.buf_offsets0.y;
    } else if (word_idx == 2u) {
        packed_off = push_data.buf_offsets0.z;
    } else if (word_idx == 3u) {
        packed_off = push_data.buf_offsets0.w;
    } else if (word_idx == 4u) {
        packed_off = push_data.buf_offsets1.x;
    } else if (word_idx == 5u) {
        packed_off = push_data.buf_offsets1.y;
    } else if (word_idx == 6u) {
        packed_off = push_data.buf_offsets1.z;
    } else if (word_idx == 7u) {
        packed_off = push_data.buf_offsets1.w;
    } else if (word_idx == 8u) {
        packed_off = push_data.buf_offsets2.x;
    } else {
        packed_off = push_data.buf_offsets2.y;
    }
    uint base = bitfieldExtract(packed_off, bit_shift, 8) >> 2u;
    if (vs_in_attr1.w < -0.5) {
        base += uint(-vs_in_attr1.w - 0.5) * 24u;
    }

    const float px = vs_in_attr1.x;
    const float py = vs_in_attr1.y;

    out_attr0 = vs_in_attr0;
    out_attr1 = vec4(uintBitsToFloat(cb0.data[base + 0u]),
                     uintBitsToFloat(cb0.data[base + 1u]),
                     uintBitsToFloat(cb0.data[base + 2u]),
                     uintBitsToFloat(cb0.data[base + 3u]));
    out_attr2 = vec4(uintBitsToFloat(cb0.data[base + 4u]),
                     uintBitsToFloat(cb0.data[base + 5u]),
                     uintBitsToFloat(cb0.data[base + 6u]),
                     uintBitsToFloat(cb0.data[base + 7u]));

    const float pos_x = uintBitsToFloat(cb0.data[base + 8u]) * px +
                        uintBitsToFloat(cb0.data[base + 9u]) * py +
                        uintBitsToFloat(cb0.data[base + 11u]);
    const float pos_y = uintBitsToFloat(cb0.data[base + 12u]) * px +
                        uintBitsToFloat(cb0.data[base + 13u]) * py +
                        uintBitsToFloat(cb0.data[base + 15u]);

    const float uv_x = uintBitsToFloat(cb0.data[base + 16u]) * px +
                       uintBitsToFloat(cb0.data[base + 17u]) * py +
                       uintBitsToFloat(cb0.data[base + 19u]);
    const float uv_y = uintBitsToFloat(cb0.data[base + 20u]) * px +
                       uintBitsToFloat(cb0.data[base + 21u]) * py +
                       uintBitsToFloat(cb0.data[base + 23u]);
    out_attr3 = vec4(uv_x, uv_y, 0.0, 1.0);

    vec4 pos = vec4(pos_x, pos_y, 0.0, 1.0);
    const uint mp = push_data.motion_param;
    if (mp != 0u) {
        const float lim_x = float(mp >> 16u) * 1.52590219e-05;
        const float lim_y = float((mp & 65534u) >> 1u) * 3.05185094e-05;
        const bool flip_y = (mp & 1u) != 0u;
        const float abs_w = abs(pos.w);
        const float sy = flip_y ? pos.y : -pos.y;
        const bool cond_y_gt = sy > (abs_w * 0.400000006);
        const float abs_y = abs(pos.y);
        const bool out_y = abs_y >= (abs_w * (lim_y - 0.00200000009));
        const float abs_x = abs(pos.x);
        const bool out_x = abs_x >= (abs_w * (lim_x - 0.00200000009));
        const bool both_out = out_x && out_y;
        const bool in_x_max = abs_x <= (abs_w * (lim_x + 0.00400000019));
        if (both_out || (cond_y_gt && out_x && in_x_max)) {
            pos.x = (pos.x * abs_w) / abs_x;
        }
        if (both_out) {
            pos.y = (pos.y * abs_w) / abs_y;
        }
    }
    gl_Position = pos;
}
