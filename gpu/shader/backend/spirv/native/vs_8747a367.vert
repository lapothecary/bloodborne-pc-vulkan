// SPDX-License-Identifier: GPL-2.0-or-later
// Stage 8: Native Vulkan 1.3 GLSL replacement for Filter\GXLightAcc.vpo (`0x000000008747a367`).
// Supports both:
//   - Batched multi-tile local-light quads (`vs_in_attr0.w < -0.5`, encoding
//     `-(tile_ssbo2_vec4_offset + 1)` in `.w` and `tile_vtx_z` in `.z`), forwarding
//     `+float(tile_ssbo2_vec4_offset + 1)` in `out_attr1.z` (`fs_in_attr1_p.z`) and setting
//     `gl_Position = out_attr0 = vec4(vs_in_attr0.xyz, 1.0)` (zero SSBO fetch per vertex!).
//   - Unbatched fullscreen passes (`vs_in_attr0.w >= -0.5`, e.g. Directional/IBL/Fog `w == +1.0`),
//     loading `cb0.data[buf_dword_off]` into `.z`, setting `out_attr1 = vec4(uv, 0.0, 1.0)`, and
//     applying `push_data.motion_param` ultrawide UI edge stretch.
#version 460

layout(constant_id = 0) const uint kBufSlot = 0u;

layout(location = 0) in vec4 vs_in_attr0;
layout(location = 1) in vec4 vs_in_attr1;

layout(location = 0) out vec4 out_attr0;
layout(location = 1) out vec4 out_attr1;

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
    if (vs_in_attr0.w < -0.5) {
        const vec4 pos = vec4(vs_in_attr0.x, vs_in_attr0.y, vs_in_attr0.z, 1.0);
        out_attr0 = pos;
        out_attr1 = vec4(vs_in_attr1.x, vs_in_attr1.y, -vs_in_attr0.w, 1.0);
        gl_Position = pos;
        return;
    }

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
    const uint buf_dword_off = bitfieldExtract(packed_off, bit_shift, 8) >> 2u;
    const float cb_z = uintBitsToFloat(cb0.data[buf_dword_off]);

    vec4 pos = vec4(vs_in_attr0.x, vs_in_attr0.y, cb_z, vs_in_attr0.w);
    out_attr0 = pos;
    out_attr1 = vec4(vs_in_attr1.x, vs_in_attr1.y, 0.0, 1.0);

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
