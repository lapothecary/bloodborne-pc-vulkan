#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Auxiliary data block
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

// Soft-Particle Dual-Texture Blend (attr5.z) + Fog 2-MRT — ps_d729c469
layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 2) in vec4 fs_in_attr2_p;
layout(location = 3) in vec4 fs_in_attr3_p;
layout(location = 4) in vec4 fs_in_attr4_p;
layout(location = 5) in vec4 fs_in_attr5_p;

layout(location = 0) out vec4 frag_color0;
layout(location = 1) out vec4 frag_color1;

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { uint data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { uint data[]; } ssbo_2;

layout(set = 0, binding = 2) uniform texture2D fs_img0;
layout(set = 0, binding = 3) uniform texture2D fs_img1;
layout(set = 0, binding = 4) uniform sampler fs_samp0;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }

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
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
               mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

vec3 FireBlackbodyHdr(float t) {
    const vec3 c0 = vec3(1.15, 0.08, 0.005);
    const vec3 c1 = vec3(2.20, 0.55, 0.032);
    const vec3 c2 = vec3(3.45, 2.00, 0.48);
    const vec3 c3 = vec3(5.20, 4.20, 2.35);
    vec3 col = mix(c0, c1, smoothstep(0.06, 0.38, t));
    col = mix(col, c2, smoothstep(0.36, 0.76, t));
    col = mix(col, c3, smoothstep(0.74, 1.20, t));
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
    const float fire_mask = smoothstep(1.35, 2.25, r_over_b) *
                            smoothstep(0.95, 1.20, r_over_g) *
                            smoothstep(1.05, 1.60, max(r_over_b * 0.55, g_over_b));
    if (fire_mask < 1e-3) {
        return;
    }

    const float t_sec = uintBitsToFloat(push_data.buf_offsets2.x);
    const float intensity = 0.40 + float(b_hi & 0xFFu) * (2.00 / 255.0);
    const float turb = float((push_data.buf_offsets1.w >> 24u) & 0xFFu) * (2.00 / 255.0);
    const float embers = float((push_data.buf_offsets1.w >> 16u) & 0xFFu) * (2.00 / 255.0);

    const vec3 p0 = world_pos * 2.8 - vec3(0.0, t_sec * 4.8, 0.0);
    const float wx = FireNoise3(p0 * 0.85 + vec3(3.1, 0.0, 7.4)) - 0.5;
    const float wz = FireNoise3(p0 * 0.85 + vec3(11.7, 2.3, 1.9)) - 0.5;
    const vec3 p_adv = p0 + vec3(wx * 1.10, -0.30 * wx, wz * 1.10);

    const float n1 = FireNoise3(p_adv);
    const float n2 = FireNoise3(p_adv * 2.20 - vec3(0.0, t_sec * 4.0, 0.0));
    const float n3 = FireNoise3(p_adv * 4.60 - vec3(0.0, t_sec * 7.5, 0.0));
    const float fbm = n1 * 0.50 + n2 * 0.32 + n3 * 0.18;
    const float ridge = 1.0 - abs(n2 * 2.0 - 1.0);

    const float raw_e = max(rgb.r * 0.55 + rgb.g * 0.35 + rgb.b * 0.10, max_c * 0.65);
    const float raw_d = clamp(alpha * (0.50 + 0.50 * clamp(raw_e, 0.0, 2.0)), 0.0, 1.0);

    const float turb_mix = clamp(turb * 0.85, 0.0, 1.0);
    const float carve = mix(1.0, clamp(0.32 + 1.25 * fbm + 0.34 * (ridge - 0.45), 0.0, 1.65), turb_mix);
    const float eroded = smoothstep(0.06 * min(turb, 1.5), 0.52, raw_d * carve);

    const float temp = clamp((0.40 * clamp(raw_e, 0.0, 2.2) + 0.60 * eroded) *
                             mix(1.0, 0.56 + 0.78 * fbm + 0.26 * ridge, turb_mix),
                             0.0, 1.35);
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
        if (rnd.x > 0.74) {
            const vec3 center = 0.2 + 0.6 * rnd;
            const float d_sp = length(ef - center);
            const float twinkle = 0.45 + 0.55 * sin(t_sec * 17.0 + rnd.y * 43.0);
            const float sp = smoothstep(0.16, 0.025, d_sp) * twinkle * smoothstep(0.04, 0.30, alpha);
            enh_rgb += vec3(9.0, 3.0, 0.40) * (sp * embers * intensity);
            enh_alpha = max(enh_alpha, sp * 0.85 * min(embers, 1.2));
        }
    }

    rgb = mix(rgb, enh_rgb, fire_mask);
    alpha = mix(alpha, enh_alpha, fire_mask);
}

void main() {
    uint u0 = push_data.buf_offsets0.x;
    uint d0 = bitfieldExtract(u0, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(u0, 8, 8) >> 2u;

    vec4 tex1 = texture(sampler2D(fs_img0, fs_samp0), fs_in_attr5_p.xy);
    vec4 tex0 = texture(sampler2D(fs_img0, fs_samp0), fs_in_attr4_p.xy);
    float blend_w = fs_in_attr5_p.z;

    vec3 sq0 = tex0.xyz * tex0.xyz;
    vec3 sq1 = tex1.xyz * tex1.xyz;
    vec3 tex_rgb = blend_w * (-sq0 + sq1) + sq0;
    float tex_a = blend_w * (tex1.w - tex0.w) + tex0.w;

    bool mode2 = (ssbo_1.data[d0 + 95u] == 2u);
    vec3 c_mod = vec3(s1f(d0 + 36u), s1f(d0 + 37u), s1f(d0 + 38u));
    vec3 v_mod = c_mod * fs_in_attr0_p.xyz;
    vec3 inv_v = -c_mod * fs_in_attr0_p.xyz + 1.0;
    vec3 ta = tex_rgb * tex_a;

    vec3 ov_rgb = vec3(
        (ta.x < 0.5) ? ((v_mod.x * ta.x) * 2.0) : (1.0 - 2.0 * (inv_v.x - tex_a * (tex_rgb.x * inv_v.x))),
        (ta.y < 0.5) ? ((v_mod.y * ta.y) * 2.0) : (1.0 - 2.0 * (inv_v.y - tex_a * (tex_rgb.y * inv_v.y))),
        (0.5 > ta.z) ? ((v_mod.z * ta.z) * 2.0) : (1.0 - 2.0 * (inv_v.z - tex_a * (tex_rgb.z * inv_v.z)))
    );
    vec3 mul_rgb = c_mod * (tex_rgb * fs_in_attr0_p.xyz);
    vec3 base_rgb = mode2 ? ov_rgb : mul_rgb;

    float inv_w = 1.0 / fs_in_attr2_p.w;
    vec2 screen_uv = vec2(fma(0.5, fs_in_attr2_p.x * inv_w, 0.5), fma(-0.5, fs_in_attr2_p.y * inv_w, 0.5));
    vec4 tex_depth = texture(sampler2D(fs_img1, fs_samp0), screen_uv);
    float lin_z = (-s1f(d0 + 16u)) * (1.0 / (s1f(d0 + 17u) * tex_depth.x - s1f(d0 + 18u)));
    float soft_fade = max(0.0, min(fs_in_attr1_p.w, lin_z - fs_in_attr1_p.z));
    float alpha_raw = (s1f(d0 + 39u) * (tex_a * fs_in_attr0_p.w)) * soft_fade;

    ApplyEnhancedFire2026(base_rgb, alpha_raw, fs_in_attr3_p.xyz, fs_in_attr4_p.xy, fs_in_attr5_p.xy);

    vec3 col_scale = vec3(s1f(d0 + 43u), s1f(d0 + 47u), s1f(d0 + 51u));
    vec3 lit_rgb = col_scale * base_rgb;

    float luma_bias = -1.0 + (lit_rgb.z * 0.114480004 + (lit_rgb.x * 0.298909992 + 0.586610019 * lit_rgb.y));
    float rad_mod = s2f(d1 + 7u) * luma_bias + 1.0;
    float hgt_mod = s2f(d1 + 19u) * luma_bias + 1.0;

    float dist_xz = sqrt(fs_in_attr3_p.x * fs_in_attr3_p.x + fs_in_attr3_p.z * fs_in_attr3_p.z);
    float fog_rad = clamp(s2f(d1 + 8u) + s2f(d1 + 8u) * (-exp2(-(s2f(d1 + 10u) * (dist_xz - s2f(d1 + 9u))))), 0.0, 1.0);
    vec3 rad_col = vec3(s2f(d1 + 4u), s2f(d1 + 5u), s2f(d1 + 6u)) * rad_mod;
    vec3 after_rad = fog_rad * (-base_rgb * col_scale + rad_col) + lit_rgb;

    float fog_hgt = clamp(s2f(d1 + 12u) * clamp(s2f(d1 + 14u) * ((s2f(d1 + 15u) - fs_in_attr3_p.y) - s2f(d1 + 13u)), 0.0, 1.0), 0.0, 1.0);
    vec3 hgt_col = vec3(s2f(d1 + 16u), s2f(d1 + 17u), s2f(d1 + 18u)) * hgt_mod;
    vec3 fogged_rgb = fog_hgt * (hgt_col - after_rad) + after_rad;

    vec3 final_rgb = max(vec3(0.0), (0.0 < alpha_raw) ? fogged_rgb : base_rgb);
    float final_a = max(0.0, alpha_raw);

    vec4 out0 = vec4(final_rgb, final_a);
    vec4 mrt1_scale = vec4(s1f(d0 + 88u), s1f(d0 + 89u), s1f(d0 + 90u), s1f(d0 + 91u));
    frag_color1 = mrt1_scale * out0;
    frag_color0 = out0;
}
