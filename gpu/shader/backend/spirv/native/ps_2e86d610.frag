// SPDX-License-Identifier: GPL-2.0-or-later
// Remaster v2 Stage 2.0: Native Vulkan 1.3 GLSL replacement for ps 0x000000002e86d610
// (Tessellated GXFfx fire ribbons / bonfire flame tongues / Molotov fire streams + radial/height fog).
// When Enhanced Fire & Embers (fx_mask & 0x80) is active on additive warm-flame draws
// (mb_mask & 0x20), upgrades ribbons with 170-FPS 3D FBM flame sculpting, physical Blackbody
// HDR radiation (1400K..4800K), and rising glowing micro-embers.
#version 450

layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 2) in vec4 fs_in_attr2_p;
layout(location = 3) in vec4 fs_in_attr3_p;

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

layout(set = 0, binding = 0) readonly buffer SSBO1 {
    uint data[];
} ssbo_1;

layout(set = 0, binding = 1) readonly buffer SSBO2 {
    uint data[];
} ssbo_2;

layout(set = 0, binding = 2) uniform texture2D fs_img0;
layout(set = 0, binding = 3) uniform sampler fs_samp0;

float s1f(uint idx) {
    return uintBitsToFloat(ssbo_1.data[idx]);
}

float s2f(uint idx) {
    return uintBitsToFloat(ssbo_2.data[idx]);
}

float FireHash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

vec3 FireHash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

float FireNoise3(vec3 p) {
    const vec3 i = floor(p);
    const vec3 f = fract(p);
    const vec3 u = f * f * (3.0 - 2.0 * f);
    const float n000 = FireHash31(i + vec3(0.0, 0.0, 0.0));
    const float n100 = FireHash31(i + vec3(1.0, 0.0, 0.0));
    const float n010 = FireHash31(i + vec3(0.0, 1.0, 0.0));
    const float n110 = FireHash31(i + vec3(1.0, 1.0, 0.0));
    const float n001 = FireHash31(i + vec3(0.0, 0.0, 1.0));
    const float n101 = FireHash31(i + vec3(1.0, 0.0, 1.0));
    const float n011 = FireHash31(i + vec3(0.0, 1.0, 1.0));
    const float n111 = FireHash31(i + vec3(1.0, 1.0, 1.0));
    const float nx00 = mix(n000, n100, u.x);
    const float nx10 = mix(n010, n110, u.x);
    const float nx01 = mix(n001, n101, u.x);
    const float nx11 = mix(n011, n111, u.x);
    return mix(mix(nx00, nx10, u.y), mix(nx01, nx11, u.y), u.z);
}

vec3 FireBlackbodyHdr(float t) {
    const vec3 c0 = vec3(1.05, 0.06, 0.004);
    const vec3 c1 = vec3(2.15, 0.50, 0.030);
    const vec3 c2 = vec3(3.55, 2.05, 0.52);
    const vec3 c3 = vec3(5.60, 4.45, 2.55);
    vec3 col = mix(c0, c1, smoothstep(0.08, 0.42, t));
    col = mix(col, c2, smoothstep(0.40, 0.80, t));
    col = mix(col, c3, smoothstep(0.78, 1.25, t));
    return col;
}

void ApplyEnhancedFire2026(inout vec3 rgb, inout float alpha, vec3 world_pos, vec2 uv_lo, vec2 uv_hi) {
    const uint b_hi = push_data.buf_offsets2.y;
    const uint fx_mask = (b_hi >> 24u) & 0xFFu;
    const uint mb_mask = (b_hi >> 16u) & 0xFFu;
    if ((fx_mask & 0x80u) == 0u || (mb_mask & 0x20u) == 0u) {
        return;
    }
    const vec2 duv = uv_hi - uv_lo;
    if (dot(duv, duv) < 1e-6) {
        return;
    }
    const float max_c = max(max(rgb.r, rgb.g), rgb.b);
    if (max_c < 1e-4 || alpha < 1e-4) {
        return;
    }
    const float eps = max(0.002 * max_c, 1e-5);
    const float r_over_b = rgb.r / max(rgb.b, eps);
    const float r_over_g = rgb.r / max(rgb.g, eps);
    const float g_over_b = rgb.g / max(rgb.b, eps);
    const float fire_mask = smoothstep(1.40, 2.35, r_over_b) *
                            smoothstep(0.95, 1.22, r_over_g) *
                            smoothstep(1.05, 1.65, max(r_over_b * 0.55, g_over_b));
    if (fire_mask < 1e-3) {
        return;
    }

    const float t_sec = uintBitsToFloat(push_data.buf_offsets2.x);
    const float intensity = 0.40 + float(b_hi & 0xFFu) * (2.00 / 255.0);
    const float turb = float((push_data.buf_offsets1.w >> 24u) & 0xFFu) * (2.00 / 255.0);
    const float embers = float((push_data.buf_offsets1.w >> 16u) & 0xFFu) * (2.00 / 255.0);

    const vec3 p0 = world_pos * 2.7 - vec3(0.0, t_sec * 4.6, 0.0);
    const float wx = FireNoise3(p0 * 0.85 + vec3(3.1, 0.0, 7.4)) - 0.5;
    const float wz = FireNoise3(p0 * 0.85 + vec3(11.7, 2.3, 1.9)) - 0.5;
    const vec3 p_adv = p0 + vec3(wx * 1.15, -0.30 * wx, wz * 1.15);

    const float n1 = FireNoise3(p_adv);
    const float n2 = FireNoise3(p_adv * 2.15 - vec3(0.0, t_sec * 3.8, 0.0));
    const float n3 = FireNoise3(p_adv * 4.50 - vec3(0.0, t_sec * 7.2, 0.0));
    const float fbm = n1 * 0.52 + n2 * 0.31 + n3 * 0.17;
    const float ridge = 1.0 - abs(n2 * 2.0 - 1.0);

    const float raw_e = max(rgb.r * 0.55 + rgb.g * 0.35 + rgb.b * 0.10, max_c * 0.65);
    const float raw_d = clamp(alpha * (0.55 + 0.45 * clamp(raw_e, 0.0, 2.0)), 0.0, 1.0);

    const float turb_mix = clamp(turb * 0.80, 0.0, 1.0);
    const float carve = mix(1.0, clamp(0.38 + 1.15 * fbm + 0.30 * (ridge - 0.45), 0.0, 1.60), turb_mix);
    const float eroded = smoothstep(0.05 * min(turb, 1.5), 0.55, raw_d * carve);

    const float temp = clamp((0.42 * clamp(raw_e, 0.0, 2.2) + 0.58 * eroded) *
                             mix(1.0, 0.60 + 0.72 * fbm + 0.24 * ridge, turb_mix),
                             0.0, 1.38);
    const vec3 bb_col = FireBlackbodyHdr(temp);
    const float energy_scale = min(raw_e * 0.82, 2.35) * intensity;
    vec3 enh_rgb = bb_col * energy_scale;
    float enh_alpha = alpha * mix(1.0, mix(0.28, 1.32, eroded) * clamp(0.48 + 0.82 * fbm, 0.15, 1.38), turb_mix);

    if (embers > 0.02) {
        const vec3 sway = vec3(sin(t_sec * 2.1 + world_pos.y * 2.4) * 0.55,
                               t_sec * 7.4,
                               cos(t_sec * 1.8 + world_pos.x * 2.4) * 0.55);
        const vec3 ep = world_pos * 13.5 - sway;
        const vec3 ec = floor(ep);
        const vec3 ef = fract(ep);
        const vec3 rnd = FireHash33(ec);
        if (rnd.x > 0.76) {
            const vec3 center = 0.2 + 0.6 * rnd;
            const float d_sp = length(ef - center);
            const float twinkle = 0.45 + 0.55 * sin(t_sec * 17.0 + rnd.y * 43.0);
            const float sp = smoothstep(0.16, 0.025, d_sp) * twinkle * smoothstep(0.03, 0.28, alpha);
            enh_rgb += vec3(8.5, 2.8, 0.38) * (sp * embers * intensity);
            enh_alpha = max(enh_alpha, sp * 0.85 * min(embers, 1.2));
        }
    }

    rgb = mix(rgb, enh_rgb, fire_mask);
    alpha = mix(alpha, enh_alpha, fire_mask);
}

void main() {
    const uint b0 = push_data.buf_offsets0.x;
    const uint d0 = (b0 & 0xFFu) >> 2u;
    const uint d1 = ((b0 >> 8u) & 0xFFu) >> 2u;

    const vec4 s_lo = texture(sampler2D(fs_img0, fs_samp0), fs_in_attr2_p.xy);
    const vec4 s_hi = texture(sampler2D(fs_img0, fs_samp0), fs_in_attr3_p.xy);

    const vec3 lo_sq = s_lo.rgb * s_lo.rgb;
    const vec3 hi_sq = s_hi.rgb * s_hi.rgb;
    const float blend_t = fs_in_attr3_p.z;

    const vec3 col = fma(vec3(blend_t), hi_sq - lo_sq, lo_sq);
    const float alpha = fma(blend_t, s_hi.a - s_lo.a, s_lo.a);

    const vec4 tint = vec4(s1f(d0 + 36u), s1f(d0 + 37u), s1f(d0 + 38u), s1f(d0 + 39u)) * fs_in_attr0_p;
    vec3 rgb_out;
    float a_out = alpha * tint.w;
    if (ssbo_1.data[d0 + 95u] == 2u) {
        const vec3 ca = col * alpha;
        const vec3 lo_br = (tint.rgb * ca) * 2.0;
        const vec3 inv_t = vec3(1.0) - tint.rgb;
        const vec3 hi_br = vec3(1.0) - 2.0 * (inv_t - alpha * (col * inv_t));
        rgb_out = vec3((ca.x < 0.5) ? lo_br.x : hi_br.x,
                       (ca.y < 0.5) ? lo_br.y : hi_br.y,
                       (0.5 > ca.z) ? lo_br.z : hi_br.z);
    } else {
        rgb_out = tint.rgb * col;
    }

    ApplyEnhancedFire2026(rgb_out, a_out, fs_in_attr1_p.xyz, fs_in_attr2_p.xy, fs_in_attr3_p.xy);

    if (a_out > 0.0) {
        const float luma = fma(rgb_out.z, 0.114480004,
                           fma(rgb_out.x, 0.298909992, 0.586610019 * rgb_out.y));
        const float luma_m1 = luma - 1.0;

        const float dist_xz = sqrt(fs_in_attr1_p.x * fs_in_attr1_p.x + fs_in_attr1_p.z * fs_in_attr1_p.z);
        const float b8 = s2f(d1 + 8u);
        const float fog_rad = clamp(b8 - b8 * exp2(-(s2f(d1 + 10u) * (dist_xz - s2f(d1 + 9u)))), 0.0, 1.0);

        const vec3 fcol1 = vec3(s2f(d1 + 4u), s2f(d1 + 5u), s2f(d1 + 6u));
        const vec3 mix1 = fma(vec3(fog_rad), fcol1 * fma(s2f(d1 + 7u), luma_m1, 1.0) - rgb_out, rgb_out);

        const float h_fac = clamp(s2f(d1 + 14u) * ((s2f(d1 + 15u) - fs_in_attr1_p.y) - s2f(d1 + 13u)), 0.0, 1.0);
        const float fog_h = clamp(s2f(d1 + 12u) * h_fac, 0.0, 1.0);

        const vec3 fcol2 = vec3(s2f(d1 + 16u), s2f(d1 + 17u), s2f(d1 + 18u));
        const vec3 fogged = fma(vec3(fog_h), fcol2 * fma(s2f(d1 + 19u), luma_m1, 1.0) - mix1, mix1);

        const vec3 k_rgb = vec3(s1f(d0 + 43u), s1f(d0 + 47u), s1f(d0 + 51u));
        rgb_out = k_rgb * fogged;
    }

    frag_color0 = max(vec4(0.0), vec4(rgb_out, a_out));
}
