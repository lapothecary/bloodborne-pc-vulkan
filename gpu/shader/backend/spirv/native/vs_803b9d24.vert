// SPDX-License-Identifier: GPL-2.0-or-later
// Phase 5 (Step 5.3): Native Vulkan 1.3 GLSL replacement for 1-bone static/indexed 2-UV
// FLVER G-Buffer foliage/cutout VS (`0x00000000803b9d24`, 106.7k+ indices/frame).
// Supports all PushData `buf_offsets` slot permutations via `kFirstBufSlot` (`constant_id = 0`).
#version 460

layout(constant_id = 0) const uint kFirstBufSlot = 5u;

layout(location = 0) in vec4 vs_in_attr0;
layout(location = 1) in uvec4 vs_in_attr1;
layout(location = 2) in uvec4 vs_in_attr2;
layout(location = 3) in vec4 vs_in_attr3;
layout(location = 4) in ivec4 vs_in_attr4;

layout(location = 0) out vec4 out_attr0;
layout(location = 1) out vec4 out_attr1;
layout(location = 2) out vec4 out_attr2;
layout(location = 3) out vec4 out_attr3;
layout(location = 4) out vec4 out_attr4;

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

layout(set = 0, binding = 0, std430) readonly buffer ssbo_a_buf { uint data[]; } ssbo_a;
layout(set = 0, binding = 1, std430) readonly buffer ssbo_b_buf { float data[]; } ssbo_b;
layout(set = 0, binding = 2, std430) readonly buffer ssbo_c_buf { uint data[]; } ssbo_c;
layout(set = 0, binding = 3, std430) readonly buffer ssbo_d_buf { uint data[]; } ssbo_d;

uint get_buf_dword_off(uint slot) {
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
    return bitfieldExtract(packed_off, bit_shift, 8) >> 2u;
}

vec4 load_bone_row(uint base) {
    return vec4(ssbo_b.data[base + 0u], ssbo_b.data[base + 1u], ssbo_b.data[base + 2u], ssbo_b.data[base + 3u]);
}

float c_f(uint i) { return uintBitsToFloat(ssbo_c.data[i]); }
float d_f(uint i) { return uintBitsToFloat(ssbo_d.data[i]); }

void main() {
    const uint dA = get_buf_dword_off(kFirstBufSlot + 0u);
    const uint dB = get_buf_dword_off(kFirstBufSlot + 1u);
    const uint dC = get_buf_dword_off(kFirstBufSlot + 2u);
    const uint dD = get_buf_dword_off(kFirstBufSlot + 3u);

    out_attr2 = vs_in_attr3;

    const uint bone = ssbo_a.data[dA + 20u] + vs_in_attr1.w;
    const uint r0 = bone * 12u + dB;
    const vec4 m0 = load_bone_row(r0 + 0u);
    const vec4 m1 = load_bone_row(r0 + 4u);
    const vec4 m2 = load_bone_row(r0 + 8u);

    const vec3 pos_in = vs_in_attr0.xyz;
    const float wx = dot(m0.xyz, pos_in) + m0.w;
    const float wy = dot(m1.xyz, pos_in) + m1.w;
    const float wz = dot(m2.xyz, pos_in) + m2.w;

    const vec3 rel = vec3(wx - c_f(dC + 183u), wy - c_f(dC + 187u), wz - c_f(dC + 191u));
    vec4 clip_pos = vec4(
        dot(rel, vec3(c_f(dC + 200u), c_f(dC + 201u), c_f(dC + 202u))) + c_f(dC + 203u),
        dot(rel, vec3(c_f(dC + 204u), c_f(dC + 205u), c_f(dC + 206u))) + c_f(dC + 207u),
        dot(rel, vec3(c_f(dC + 208u), c_f(dC + 209u), c_f(dC + 210u))) + c_f(dC + 211u),
        dot(rel, vec3(c_f(dC + 212u), c_f(dC + 213u), c_f(dC + 214u))) + c_f(dC + 215u)
    );

    const vec3 n_in = (vec3(vs_in_attr1.xyz) - 127.0) * 0.00787401572;
    const vec4 t_in = (vec4(vs_in_attr2) - 127.0) * 0.00787401572;

    out_attr0 = vec4(dot(m0.xyz, n_in), dot(m1.xyz, n_in), dot(m2.xyz, n_in), 0.0);
    out_attr1 = vec4(dot(m0.xyz, t_in.xyz), dot(m1.xyz, t_in.xyz), dot(m2.xyz, t_in.xyz), t_in.w);

    const vec4 uv_in = vec4(vs_in_attr4) * 0.00048828125;
    out_attr3 = vec4(
        fma(uv_in.x, d_f(dD + 0u), d_f(dD + 2u)),
        fma(uv_in.y, d_f(dD + 1u), d_f(dD + 3u)),
        fma(uv_in.z, d_f(dD + 4u), d_f(dD + 6u)),
        fma(uv_in.w, d_f(dD + 5u), d_f(dD + 7u))
    );
    out_attr4 = clip_pos;

    const uint mp = push_data.motion_param;
    if (mp != 0u) {
        const float lim_x = float(mp >> 16u) * 1.52590219e-05;
        const float lim_y = float((mp & 65534u) >> 1u) * 3.05185094e-05;
        const bool flip_y = (mp & 1u) != 0u;
        const float abs_w = abs(clip_pos.w);
        const float sy = flip_y ? clip_pos.y : -clip_pos.y;
        const bool cond_y_gt = sy > (abs_w * 0.400000006);
        const float abs_y = abs(clip_pos.y);
        const bool out_y = abs_y >= (abs_w * (lim_y - 0.00200000009));
        const float abs_x = abs(clip_pos.x);
        const bool out_x = abs_x >= (abs_w * (lim_x - 0.00200000009));
        const bool both_out = out_x && out_y;
        const bool in_x_max = abs_x <= (abs_w * (lim_x + 0.00400000019));
        if (both_out || (cond_y_gt && out_x && in_x_max)) {
            clip_pos.x = (clip_pos.x * abs_w) / abs_x;
        }
        if (both_out) {
            clip_pos.y = (clip_pos.y * abs_w) / abs_y;
        }
    }
    gl_Position = clip_pos;
}
