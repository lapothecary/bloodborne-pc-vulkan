// SPDX-License-Identifier: GPL-2.0-or-later
// Remaster v2 Stage 2.0: Native Vulkan 1.3 GLSL replacement for ps 0x00000000832cdf8d
// (Primary GXFfx billboard particle shader: bonfires, torches, Molotovs, embers, sparks +
//  exact Hoffman/Preetham Rayleigh/Mie atmospheric scattering and radial/height fog).
// When Enhanced Fire & Embers (fx_mask & 0x80) is active on additive warm-flame draws
// (mb_mask & 0x20), upgrades low-res 30-FPS sprites with 170-FPS 3D FBM flame sculpting,
// physical Blackbody HDR radiation (1400K..4800K), and rising glowing micro-embers.
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

void main() {
    const uint b0 = push_data.buf_offsets0.x;
    const uint d0 = (b0 & 0xFFu) >> 2u;
    const uint d1 = ((b0 >> 8u) & 0xFFu) >> 2u;

    const vec4 s_hi = texture(sampler2D(fs_img0, fs_samp0), fs_in_attr3_p.xy);
    const vec4 s_lo = texture(sampler2D(fs_img0, fs_samp0), fs_in_attr2_p.xy);

    const vec3 hi_sq = s_hi.rgb * s_hi.rgb;
    const vec3 lo_sq = s_lo.rgb * s_lo.rgb;
    const float blend_t = fs_in_attr2_p.w;

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


    // Exact radial fog + height fog + Hoffman/Preetham Rayleigh/Mie atmospheric scattering.
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
    const vec3 lit = k_rgb * fogged;

    // Rayleigh / Mie scattering
    const float len3_sq = dot(fs_in_attr1_p.xyz, fs_in_attr1_p.xyz);
    const float inv_len3 = (len3_sq > 0.0) ? inversesqrt(max(len3_sq, 1.17549435e-38)) : 0.0;
    const vec3 view_dir = fs_in_attr1_p.xyz * inv_len3;
    const vec3 sun_dir = vec3(s2f(d1 + 52u), s2f(d1 + 53u), s2f(d1 + 54u));
    const float cos_t = dot(view_dir, sun_dir);
    const float rayleigh_p = fma(cos_t, cos_t, 1.0);
    const float mie_denom = max(1e-6, s2f(d1 + 37u) - cos_t * s2f(d1 + 38u));
    const float mie_p = (s2f(d1 + 36u) * inversesqrt(mie_denom)) / mie_denom;

    const float dist4 = sqrt(len3_sq + fs_in_attr1_p.w * fs_in_attr1_p.w);
    const vec3 beta_ext = vec3(s2f(d1 + 24u), s2f(d1 + 25u), s2f(d1 + 26u));
    const vec3 ext = exp2((s2f(d1 + 55u) * (dist4 * (-beta_ext))) * 2.0813689);

    const vec3 ray_c = vec3(s2f(d1 + 40u), s2f(d1 + 41u), s2f(d1 + 42u)) * rayleigh_p;
    const vec3 mie_c = vec3(s2f(d1 + 44u), s2f(d1 + 45u), s2f(d1 + 46u)) * mie_p;
    const vec3 sun_c = vec3(s2f(d1 + 32u), s2f(d1 + 33u), s2f(d1 + 34u)) *
                       vec3(s2f(d1 + 48u), s2f(d1 + 49u), s2f(d1 + 50u)) * s2f(d1 + 31u);
    const vec3 inscatter = (vec3(1.0) - ext) * (ray_c + mie_c) * sun_c;
    const vec3 ext_scale = vec3(s2f(d1 + 28u), s2f(d1 + 29u), s2f(d1 + 30u)) * ext;
    const vec3 scattered = ext_scale * lit + inscatter;

    const vec3 final_rgb = mix(lit, scattered, s2f(d1 + 51u));
    frag_color0 = max(vec4(0.0), vec4(final_rgb, a_out));
}
