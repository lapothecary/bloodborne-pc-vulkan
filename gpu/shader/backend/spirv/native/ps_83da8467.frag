// SPDX-License-Identifier: GPL-2.0-or-later
// Phase 2 (v1.2.4): Native Vulkan 1.3 GLSL 460 replacement for ps 0x0000000083da8467
// (gxposteffect ScreenSpaceAO2_WithNormalReProj.ppo — 8-pair normal-oriented SSAO + temporal reprojection).
#version 460

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

layout(set = 0, binding = 0, std430) readonly buffer SSBO1 { uint data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer SSBO2 { uint data[]; } ssbo_2;
layout(set = 0, binding = 2, std430) readonly buffer SSBO3 { uint data[]; } ssbo_3;

layout(set = 0, binding = 3) uniform texture2D fs_img0;
layout(set = 0, binding = 4) uniform texture2D fs_img1;
layout(set = 0, binding = 5) uniform texture2D fs_img2;
layout(set = 0, binding = 6) uniform texture2D fs_img3;
layout(set = 0, binding = 7) uniform sampler fs_samp0;
layout(set = 0, binding = 8) uniform sampler fs_samp1;
layout(set = 0, binding = 9) uniform sampler fs_samp2;
layout(set = 0, binding = 10) uniform sampler fs_samp3;

vec4 load_vec4_b0(uint off) {
    return vec4(uintBitsToFloat(ssbo_1.data[off + 0u]),
                uintBitsToFloat(ssbo_1.data[off + 1u]),
                uintBitsToFloat(ssbo_1.data[off + 2u]),
                uintBitsToFloat(ssbo_1.data[off + 3u]));
}

vec3 unproject(vec2 uv, float depth, vec4 r0, vec4 r1, vec4 r2, vec4 r3) {
    const vec4 clip = vec4(fma(2.0, uv.x, -1.0), fma(-2.0, uv.y, 1.0), depth, 1.0);
    const float inv_w = 1.0 / dot(r3, clip);
    return vec3(dot(r0, clip), dot(r1, clip), dot(r2, clip)) * inv_w;
}

void main() {
    const uint b0 = (push_data.buf_offsets0.x & 0xFFu) >> 2u;
    const uint b1 = ((push_data.buf_offsets0.x >> 8u) & 0xFFu) >> 2u;
    const uint b2 = ((push_data.buf_offsets0.x >> 16u) & 0xFFu) >> 2u;

    const vec2 uv = fs_in_attr0_p.xy;
    const float d0 = textureLod(sampler2D(fs_img0, fs_samp0), uv, 0.0).r;
    const vec3 n_raw = texture(sampler2D(fs_img1, fs_samp1), uv).rgb;

    const vec4 r0 = load_vec4_b0(b0 + 0u);
    const vec4 r1 = load_vec4_b0(b0 + 4u);
    const vec4 r2 = load_vec4_b0(b0 + 8u);
    const vec4 r3 = load_vec4_b0(b0 + 12u);

    const vec3 p_center = unproject(uv, d0, r0, r1, r2, r3);
    float ao_final = 1.0;

    if (d0 != 1.0) {
        const vec4 dim_params = load_vec4_b0(b0 + 16u); // (w, h, inv_w, inv_h)
        const vec2 rot_raw =
            textureLod(sampler2D(fs_img2, fs_samp2), uv * dim_params.xy * 0.015625, 0.0).rg;
        const vec2 rot = fma(vec2(2.0), rot_raw, vec2(-1.0));

        const vec3 n_dec = fma(vec3(2.0), n_raw, vec3(-1.0));
        const vec3 n_world = n_dec * inversesqrt(max(dot(n_dec, n_dec), 1e-20));
        const vec3 nm0 = vec3(uintBitsToFloat(ssbo_1.data[b0 + 20u]),
                              uintBitsToFloat(ssbo_1.data[b0 + 21u]),
                              uintBitsToFloat(ssbo_1.data[b0 + 22u]));
        const vec3 nm1 = vec3(uintBitsToFloat(ssbo_1.data[b0 + 24u]),
                              uintBitsToFloat(ssbo_1.data[b0 + 25u]),
                              uintBitsToFloat(ssbo_1.data[b0 + 26u]));
        const vec3 nm2 = vec3(uintBitsToFloat(ssbo_1.data[b0 + 28u]),
                              uintBitsToFloat(ssbo_1.data[b0 + 29u]),
                              uintBitsToFloat(ssbo_1.data[b0 + 30u]));
        const vec3 n_view = vec3(dot(nm0, n_world), dot(nm1, n_world), dot(nm2, n_world));

        const float inv_z = 1.0 / p_center.z;
        const float rad_scale = uintBitsToFloat(ssbo_1.data[b0 + 40u]);
        const float rad_min = uintBitsToFloat(ssbo_1.data[b0 + 41u]);
        const float rad_max = uintBitsToFloat(ssbo_1.data[b0 + 42u]);
        const float rad = clamp(rad_scale * inv_z, rad_min, rad_max);
        const vec2 step_scale = dim_params.zw * rad;

        float ao_acc = 0.0;
        for (uint i = 0u; i < 8u; ++i) {
            const uint ko = b1 + (i << 2u);
            const float kx = uintBitsToFloat(ssbo_2.data[ko + 0u]);
            const float ky = uintBitsToFloat(ssbo_2.data[ko + 1u]);
            const float kw = uintBitsToFloat(ssbo_2.data[ko + 2u]);

            const vec2 delta = vec2(kx * rot.y + ky * rot.x, -kx * rot.x + ky * rot.y) * step_scale;
            const vec2 uv_p = uv + delta;
            const vec2 uv_m = uv - delta;

            const float dp = textureLod(sampler2D(fs_img0, fs_samp0), uv_p, 0.0).r;
            const float dm = textureLod(sampler2D(fs_img0, fs_samp0), uv_m, 0.0).r;

            const vec3 pp = unproject(uv_p, dp, r0, r1, r2, r3);
            const vec3 pm = unproject(uv_m, dm, r0, r1, r2, r3);

            const vec3 vp = pp - p_center;
            const vec3 vm = pm - p_center;
            const vec3 dir_p = vp * inversesqrt(max(dot(vp, vp), 1e-20));
            const vec3 dir_m = vm * inversesqrt(max(dot(vm, vm), 1e-20));

            const float cp = clamp(1.0 - dot(n_view, dir_p), 0.0, 1.0);
            const float cm = clamp(1.0 - dot(n_view, dir_m), 0.0, 1.0);
            const float fp = clamp(10.0 * inv_z * (0.9 * p_center.z - pp.z), 0.0, 1.0);
            const float fm = clamp(10.0 * inv_z * (0.9 * p_center.z - pm.z), 0.0, 1.0);

            const float avg = 0.5 * (cp + cm);
            const float avg2 = avg * avg;
            const float occ = avg2 + max(fp, fm) * (1.0 - avg2);
            ao_acc += kw * occ;
        }

        // Temporal reprojection blend with previous frame's packed AO + depth.
        const vec4 rp0 = load_vec4_b0(b0 + 44u);
        const vec4 rp1 = load_vec4_b0(b0 + 48u);
        const vec4 rp2 = load_vec4_b0(b0 + 52u);
        const vec4 rp3 = load_vec4_b0(b0 + 56u);
        const vec4 cur_p = vec4(uv, d0, 1.0);
        const float inv_rw = 1.0 / dot(rp3, cur_p);
        const vec2 prev_uv = vec2(dot(rp0, cur_p), dot(rp1, cur_p)) * inv_rw;
        const float prev_z_raw = dot(rp2, cur_p);

        const vec3 hist_gba = textureLod(sampler2D(fs_img3, fs_samp1), prev_uv, 0.0).gba;
        const float hist_ao = textureLod(sampler2D(fs_img3, fs_samp3), prev_uv, 0.0).r;

        const float za = uintBitsToFloat(ssbo_3.data[b2 + 0u]);
        const float zb = uintBitsToFloat(ssbo_3.data[b2 + 1u]);
        const float z_scale = uintBitsToFloat(ssbo_3.data[b2 + 2u]);

        const float cur_prev_z = (-zb) / (za - prev_z_raw * inv_rw);
        const float hist_z = dot(hist_gba, vec3(1.53787005e-5, 0.00392156886, 1.0));
        const float cur_z_scaled = z_scale * cur_prev_z;
        const float z_tol = uintBitsToFloat(ssbo_1.data[b0 + 62u]) * cur_z_scaled;
        const float z_diff = clamp(abs(cur_z_scaled - z_scale * hist_z) / z_tol, 0.0, 1.0);

        const float w_base = uintBitsToFloat(ssbo_1.data[b0 + 60u]);
        const float w_fade = uintBitsToFloat(ssbo_1.data[b0 + 61u]);
        const float alpha = (prev_z_raw > 0.0) ? (w_base * (1.0 - w_fade * z_diff)) : 0.0;
        ao_final = mix(ao_acc, hist_ao, alpha);
    }

    const float z_norm = p_center.z / uintBitsToFloat(ssbo_3.data[b2 + 2u]);
    const float ao_scale = uintBitsToFloat(ssbo_1.data[b0 + 36u]);
    const float ao_bias = uintBitsToFloat(ssbo_1.data[b0 + 37u]);

    const float f255 = fract(255.0 * z_norm);
    const float f65025 = fract(65025.0 * z_norm);
    const float kInv255 = 0.00392156886;
    frag_color0 = vec4(ao_scale * ao_final + ao_bias, f65025, f255 - f65025 * kInv255,
                       z_norm - f255 * kInv255);
}
