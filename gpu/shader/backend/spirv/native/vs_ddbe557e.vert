// SPDX-License-Identifier: GPL-2.0-or-later
// Phase 5 (Step 5.3): Native Vulkan 1.3 GLSL replacement for 4-bone skinned 3-tangent,
// 2-color, 3-UV FLVER G-Buffer foliage/cutout VS (`0x00000000ddbe557e`).
// Supports all PushData `buf_offsets` slot permutations via `kFirstBufSlot` (`constant_id = 0`)
// and skips zero-weight bone palette SSBO loads.
#version 460

layout(constant_id = 0) const uint kFirstBufSlot = 7u;

layout(location = 0) in vec4 vs_in_attr0;
layout(location = 1) in uvec4 vs_in_attr1;
layout(location = 2) in uvec4 vs_in_attr2;
layout(location = 3) in uvec4 vs_in_attr3;
layout(location = 4) in uvec4 vs_in_attr4;
layout(location = 5) in uvec4 vs_in_attr5;
layout(location = 6) in vec4 vs_in_attr6;
layout(location = 7) in vec4 vs_in_attr7;
layout(location = 8) in vec4 vs_in_attr8;
layout(location = 9) in ivec4 vs_in_attr9;
layout(location = 10) in ivec4 vs_in_attr10;

layout(location = 0) out vec4 out_attr0;
layout(location = 1) out vec4 out_attr1;
layout(location = 2) out vec4 out_attr2;
layout(location = 3) out vec4 out_attr3;
layout(location = 4) out vec4 out_attr4;
layout(location = 5) out vec4 out_attr5;
layout(location = 6) out vec4 out_attr6;
layout(location = 7) out vec4 out_attr7;
layout(location = 8) out vec4 out_attr8;

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

    out_attr4 = vec4(vs_in_attr8.xyz, vs_in_attr7.w);
    out_attr6 = vs_in_attr7;

    const uint base_bone = ssbo_a.data[dA + 20u];
    const uvec4 bones = vs_in_attr5 + base_bone;

    const float w_sum = (vs_in_attr6.x + vs_in_attr6.y) + (vs_in_attr6.z + vs_in_attr6.w);
    const vec4 w = vs_in_attr6 * (1.0 / max(0.00100000005, w_sum));

    const uint r0_0 = bones.x * 12u + dB;
    vec4 m0 = load_bone_row(r0_0 + 0u) * w.x;
    vec4 m1 = load_bone_row(r0_0 + 4u) * w.x;
    vec4 m2 = load_bone_row(r0_0 + 8u) * w.x;

    if (w.y != 0.0) {
        const uint r1_0 = bones.y * 12u + dB;
        m0 = fma(load_bone_row(r1_0 + 0u), vec4(w.y), m0);
        m1 = fma(load_bone_row(r1_0 + 4u), vec4(w.y), m1);
        m2 = fma(load_bone_row(r1_0 + 8u), vec4(w.y), m2);
    }
    if (w.z != 0.0) {
        const uint r2_0 = bones.z * 12u + dB;
        m0 = fma(load_bone_row(r2_0 + 0u), vec4(w.z), m0);
        m1 = fma(load_bone_row(r2_0 + 4u), vec4(w.z), m1);
        m2 = fma(load_bone_row(r2_0 + 8u), vec4(w.z), m2);
    }
    if (w.w != 0.0) {
        const uint r3_0 = bones.w * 12u + dB;
        m0 = fma(load_bone_row(r3_0 + 0u), vec4(w.w), m0);
        m1 = fma(load_bone_row(r3_0 + 4u), vec4(w.w), m1);
        m2 = fma(load_bone_row(r3_0 + 8u), vec4(w.w), m2);
    }

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
    const vec4 t0_in = (vec4(vs_in_attr2) - 127.0) * 0.00787401572;
    const vec4 t1_in = (vec4(vs_in_attr3) - 127.0) * 0.00787401572;
    const vec4 t2_in = (vec4(vs_in_attr4) - 127.0) * 0.00787401572;

    out_attr0 = vec4(dot(m0.xyz, n_in), dot(m1.xyz, n_in), dot(m2.xyz, n_in), 0.0);
    out_attr1 = vec4(dot(m0.xyz, t0_in.xyz), dot(m1.xyz, t0_in.xyz), dot(m2.xyz, t0_in.xyz), t0_in.w);
    out_attr2 = vec4(dot(m0.xyz, t1_in.xyz), dot(m1.xyz, t1_in.xyz), dot(m2.xyz, t1_in.xyz), t1_in.w);
    out_attr3 = vec4(dot(m0.xyz, t2_in.xyz), dot(m1.xyz, t2_in.xyz), dot(m2.xyz, t2_in.xyz), t2_in.w);

    const vec4 uv01_in = vec4(vs_in_attr9) * 0.00048828125;
    out_attr5 = vec4(
        fma(uv01_in.x, d_f(dD + 0u), d_f(dD + 2u)),
        fma(uv01_in.y, d_f(dD + 1u), d_f(dD + 3u)),
        fma(uv01_in.z, d_f(dD + 4u), d_f(dD + 6u)),
        fma(uv01_in.w, d_f(dD + 5u), d_f(dD + 7u))
    );
    out_attr7 = clip_pos;

    const vec2 uv2_in = vec2(vs_in_attr10.xy) * 0.00048828125;
    out_attr8 = vec4(
        fma(uv2_in.x, d_f(dD + 8u), d_f(dD + 10u)),
        fma(uv2_in.y, d_f(dD + 9u), d_f(dD + 11u)),
        0.0,
        1.0
    );

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
