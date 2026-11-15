// SPDX-License-Identifier: GPL-2.0-or-later
// Phase 5 (Step 5.3): Native Vulkan 1.3 GLSL replacement for dual-stage 4-bone skinned
// FLVER G-Buffer foliage/cloth VS (`0x00000000b10c4f55`, 80.7k+ indices/frame).
// Supports all PushData `buf_offsets` slot permutations via `kFirstBufSlot` (`constant_id = 0`)
// and skips zero-weight bone loads across BOTH local (`ssbo_c`) and world (`ssbo_d`) palettes.
#version 460

layout(constant_id = 0) const uint kFirstBufSlot = 5u;

layout(location = 0) in vec4 vs_in_attr0;
layout(location = 1) in uvec4 vs_in_attr1;
layout(location = 2) in uvec4 vs_in_attr2;
layout(location = 3) in uvec4 vs_in_attr3;
layout(location = 4) in vec4 vs_in_attr4;
layout(location = 5) in vec4 vs_in_attr5;
layout(location = 6) in ivec4 vs_in_attr6;

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
layout(set = 0, binding = 1, std430) readonly buffer ssbo_b_buf { uint data[]; } ssbo_b;
layout(set = 0, binding = 2, std430) readonly buffer ssbo_c_buf { float data[]; } ssbo_c;
layout(set = 0, binding = 3, std430) readonly buffer ssbo_d_buf { float data[]; } ssbo_d;
layout(set = 0, binding = 4, std430) readonly buffer ssbo_e_buf { uint data[]; } ssbo_e;

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

vec4 load_c_row(uint base) {
    return vec4(ssbo_c.data[base + 0u], ssbo_c.data[base + 1u], ssbo_c.data[base + 2u], ssbo_c.data[base + 3u]);
}

vec4 load_d_row(uint base) {
    return vec4(ssbo_d.data[base + 0u], ssbo_d.data[base + 1u], ssbo_d.data[base + 2u], ssbo_d.data[base + 3u]);
}

float b_f(uint i) { return uintBitsToFloat(ssbo_b.data[i]); }
float e_f(uint i) { return uintBitsToFloat(ssbo_e.data[i]); }

void main() {
    const uint dA = get_buf_dword_off(kFirstBufSlot + 0u);
    const uint dB = get_buf_dword_off(kFirstBufSlot + 1u);
    const uint dC = get_buf_dword_off(kFirstBufSlot + 2u);
    const uint dD = get_buf_dword_off(kFirstBufSlot + 3u);
    const uint dE = get_buf_dword_off(kFirstBufSlot + 4u);

    out_attr2 = vs_in_attr5;

    const uint base_bone = ssbo_a.data[dA + 20u];
    const uvec4 lbones = vs_in_attr3;
    const uvec4 wbones = lbones + base_bone;

    const float w_sum = (vs_in_attr4.x + vs_in_attr4.y) + (vs_in_attr4.z + vs_in_attr4.w);
    const vec4 w = vs_in_attr4 * (1.0 / max(0.00100000005, w_sum));

    const uint lr0 = lbones.x * 12u + dC;
    vec4 l0 = load_c_row(lr0 + 0u) * w.x;
    vec4 l1 = load_c_row(lr0 + 4u) * w.x;
    vec4 l2 = load_c_row(lr0 + 8u) * w.x;

    const uint wr0 = wbones.x * 12u + dD;
    vec4 m0 = load_d_row(wr0 + 0u) * w.x;
    vec4 m1 = load_d_row(wr0 + 4u) * w.x;
    vec4 m2 = load_d_row(wr0 + 8u) * w.x;

    if (w.y != 0.0) {
        const uint lr1 = lbones.y * 12u + dC;
        l0 = fma(load_c_row(lr1 + 0u), vec4(w.y), l0);
        l1 = fma(load_c_row(lr1 + 4u), vec4(w.y), l1);
        l2 = fma(load_c_row(lr1 + 8u), vec4(w.y), l2);

        const uint wr1 = wbones.y * 12u + dD;
        m0 = fma(load_d_row(wr1 + 0u), vec4(w.y), m0);
        m1 = fma(load_d_row(wr1 + 4u), vec4(w.y), m1);
        m2 = fma(load_d_row(wr1 + 8u), vec4(w.y), m2);
    }
    if (w.z != 0.0) {
        const uint lr2 = lbones.z * 12u + dC;
        l0 = fma(load_c_row(lr2 + 0u), vec4(w.z), l0);
        l1 = fma(load_c_row(lr2 + 4u), vec4(w.z), l1);
        l2 = fma(load_c_row(lr2 + 8u), vec4(w.z), l2);

        const uint wr2 = wbones.z * 12u + dD;
        m0 = fma(load_d_row(wr2 + 0u), vec4(w.z), m0);
        m1 = fma(load_d_row(wr2 + 4u), vec4(w.z), m1);
        m2 = fma(load_d_row(wr2 + 8u), vec4(w.z), m2);
    }
    if (w.w != 0.0) {
        const uint lr3 = lbones.w * 12u + dC;
        l0 = fma(load_c_row(lr3 + 0u), vec4(w.w), l0);
        l1 = fma(load_c_row(lr3 + 4u), vec4(w.w), l1);
        l2 = fma(load_c_row(lr3 + 8u), vec4(w.w), l2);

        const uint wr3 = wbones.w * 12u + dD;
        m0 = fma(load_d_row(wr3 + 0u), vec4(w.w), m0);
        m1 = fma(load_d_row(wr3 + 4u), vec4(w.w), m1);
        m2 = fma(load_d_row(wr3 + 8u), vec4(w.w), m2);
    }

    const vec3 pos_in = vs_in_attr0.xyz;
    const vec3 pos_mid = vec3(
        dot(l0.xyz, pos_in) + l0.w,
        dot(l1.xyz, pos_in) + l1.w,
        dot(l2.xyz, pos_in) + l2.w
    );
    const float wx = dot(m0.xyz, pos_mid) + m0.w;
    const float wy = dot(m1.xyz, pos_mid) + m1.w;
    const float wz = dot(m2.xyz, pos_mid) + m2.w;

    const vec3 rel = vec3(wx - b_f(dB + 183u), wy - b_f(dB + 187u), wz - b_f(dB + 191u));
    vec4 clip_pos = vec4(
        dot(rel, vec3(b_f(dB + 200u), b_f(dB + 201u), b_f(dB + 202u))) + b_f(dB + 203u),
        dot(rel, vec3(b_f(dB + 204u), b_f(dB + 205u), b_f(dB + 206u))) + b_f(dB + 207u),
        dot(rel, vec3(b_f(dB + 208u), b_f(dB + 209u), b_f(dB + 210u))) + b_f(dB + 211u),
        dot(rel, vec3(b_f(dB + 212u), b_f(dB + 213u), b_f(dB + 214u))) + b_f(dB + 215u)
    );

    const vec3 n_in = (vec3(vs_in_attr1.xyz) - 127.0) * 0.00787401572;
    const vec4 t_in = (vec4(vs_in_attr2) - 127.0) * 0.00787401572;

    const vec3 n_mid = vec3(dot(l0.xyz, n_in), dot(l1.xyz, n_in), dot(l2.xyz, n_in));
    const vec3 t_mid = vec3(dot(l0.xyz, t_in.xyz), dot(l1.xyz, t_in.xyz), dot(l2.xyz, t_in.xyz));

    out_attr0 = vec4(dot(m0.xyz, n_mid), dot(m1.xyz, n_mid), dot(m2.xyz, n_mid), 0.0);
    out_attr1 = vec4(dot(m0.xyz, t_mid), dot(m1.xyz, t_mid), dot(m2.xyz, t_mid), t_in.w);

    const vec4 uv_in = vec4(vs_in_attr6) * 0.00048828125;
    out_attr3 = vec4(
        fma(uv_in.x, e_f(dE + 0u), e_f(dE + 2u)),
        fma(uv_in.y, e_f(dE + 1u), e_f(dE + 3u)),
        fma(uv_in.z, e_f(dE + 4u), e_f(dE + 6u)),
        fma(uv_in.w, e_f(dE + 5u), e_f(dE + 7u))
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
