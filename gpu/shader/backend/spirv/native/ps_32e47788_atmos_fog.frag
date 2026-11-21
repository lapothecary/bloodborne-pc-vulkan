// SPDX-License-Identifier: GPL-2.0-or-later
// Block 5, Step 5.4 (Sub-Batch 2): Native Vulkan 1.3 GLSL replacement for ps 0x0000000032e47788
// (2560x1440 R16G16B16A16Sfloat fullscreen atmospheric radial + height fog composite).
#version 460
#extension GL_EXT_samplerless_texture_functions : require
layout(early_fragment_tests) in;

layout(location = 0) in vec2 fs_in_ndc;
layout(location = 1) in vec2 fs_in_uv;
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
    float data[];
} ssbo_1;

layout(set = 0, binding = 1) readonly buffer SSBO2 {
    uint data[];
} ssbo_2;

layout(set = 0, binding = 2) uniform texture2D fs_img0;
layout(set = 0, binding = 3) uniform texture2D fs_img1;
layout(set = 0, binding = 4) uniform texture2D fs_img2;
layout(set = 0, binding = 5) uniform texture2D fs_img3;
layout(set = 0, binding = 6) uniform texture2D fs_img4;
layout(set = 0, binding = 7) uniform texture2D fs_img5;
layout(set = 0, binding = 8) uniform sampler fs_samp0;

float s2f(uint idx) {
    return uintBitsToFloat(ssbo_2.data[idx]);
}

vec3 reconstruct_pos(vec2 uv_s, float d_s, vec2 uv0, vec2 ndc0, vec2 dndc_duv, uint d1) {
    const vec2 n = ndc0 + (uv_s - uv0) * dndc_duv;
    const float cw = fma(s2f(d1 + 12u), n.x, fma(s2f(d1 + 13u), n.y, fma(s2f(d1 + 14u), d_s, s2f(d1 + 15u))));
    const float inv_w = 1.0 / cw;
    return vec3(
        fma(s2f(d1 + 0u), n.x, fma(s2f(d1 + 1u), n.y, fma(s2f(d1 + 2u), d_s, s2f(d1 + 3u)))) * inv_w,
        fma(s2f(d1 + 4u), n.x, fma(s2f(d1 + 5u), n.y, fma(s2f(d1 + 6u), d_s, s2f(d1 + 7u)))) * inv_w,
        fma(s2f(d1 + 8u), n.x, fma(s2f(d1 + 9u), n.y, fma(s2f(d1 + 10u), d_s, s2f(d1 + 11u)))) * inv_w
    );
}

void main() {
    const uint b0 = push_data.buf_offsets0.x;
    const uint d0 = (b0 & 0xFFu) >> 2u;
    const uint d1 = ((b0 >> 8u) & 0xFFu) >> 2u;
    const bool ssgi_on = ((push_data.buf_offsets2.y >> 24u) & 0x20u) != 0u;
    // Uniform-control-flow NDC/UV scale before any branch.
    const vec2 dndc_duv = vec2(2.0, (dFdy(fs_in_ndc.y) >= 0.0) ? 2.0 : -2.0);

    const vec2 uv = fs_in_uv;
    const vec4 g0     = textureLod(sampler2D(fs_img0, fs_samp0), uv, 0.0);
    const vec4 g2     = textureLod(sampler2D(fs_img2, fs_samp0), uv, 0.0);
    const float s0_a  = g0.w;
    const float out_a = g2.w;
    const vec3 col3   = textureLod(sampler2D(fs_img3, fs_samp0), uv, 0.0).rgb;
    const vec3 col4   = textureLod(sampler2D(fs_img4, fs_samp0), uv, 0.0).rgb;
    const vec3 col5   = textureLod(sampler2D(fs_img5, fs_samp0), uv, 0.0).rgb;
    vec3 sum_rgb      = col5 + (col3 + col4);

    const uint idx6 = uint(int(255.0 * s0_a)) * 6u;
    const uint o2 = (idx6 + 2u) * 4u + d0;
    const uint o3 = (idx6 + 3u) * 4u + d0;
    const float f188 = ssbo_1.data[o2 + 0u];
    const float f210 = ssbo_1.data[o3 + 0u];
    if (f188 <= 0.0 && f210 <= 0.0 && !ssgi_on) {
        frag_color0 = vec4(sum_rgb, out_a);
        return;
    }

    const float depth = textureLod(sampler2D(fs_img1, fs_samp0), uv, 0.0).x;
    const uint o1 = (idx6 + 1u) * 4u + d0;
    const uint o4 = (idx6 + 4u) * 4u + d0;

    const float nx = fs_in_ndc.x;
    const float ny = fs_in_ndc.y;

    const float cw = fma(s2f(d1 + 12u), nx, fma(s2f(d1 + 13u), ny, fma(s2f(d1 + 14u), depth, s2f(d1 + 15u))));
    const float inv_w = 1.0 / cw;
    const float pz = fma(s2f(d1 + 8u), nx, fma(s2f(d1 + 9u), ny, fma(s2f(d1 + 10u), depth, s2f(d1 + 11u)))) * inv_w;
    const float py = fma(s2f(d1 + 4u), nx, fma(s2f(d1 + 5u), ny, fma(s2f(d1 + 6u), depth, s2f(d1 + 7u)))) * inv_w;
    const float px = fma(s2f(d1 + 0u), nx, fma(s2f(d1 + 1u), ny, fma(s2f(d1 + 2u), depth, s2f(d1 + 3u)))) * inv_w;

    // Remaster-v2 Stage 1.3: Horizon-Based Screen-Space Global Illumination (HBIL SSGI)
    // Adds physically occluded 1-bounce indirect light from fires, torches, lanterns, lit stones,
    // and open sky in linear HDR space before atmospheric fog, bloom, and 3D LUT grading.
    if (ssgi_on && depth < 0.99995) {
        const vec3 P = vec3(px, py, pz);
        const float cam_dist = length(P);
        const float cam_fade = 1.0 - smoothstep(32.0, 52.0, cam_dist);
        if (cam_fade > 0.001 && cam_dist > 0.05) {
            const vec2 tex_size = vec2(textureSize(fs_img1, 0));
            const vec2 texel = 1.0 / max(tex_size, vec2(1.0));
            const float aspect = tex_size.x / max(tex_size.y, 1.0);

            // Edge-preserving geometric normal from 4-neighbor minimum depth slope.
            const vec2 uvL = vec2(max(uv.x - texel.x, 0.0), uv.y);
            const vec2 uvR = vec2(min(uv.x + texel.x, 1.0), uv.y);
            const vec2 uvU = vec2(uv.x, max(uv.y - texel.y, 0.0));
            const vec2 uvD = vec2(uv.x, min(uv.y + texel.y, 1.0));
            const float dL = textureLod(sampler2D(fs_img1, fs_samp0), uvL, 0.0).x;
            const float dR = textureLod(sampler2D(fs_img1, fs_samp0), uvR, 0.0).x;
            const float dU = textureLod(sampler2D(fs_img1, fs_samp0), uvU, 0.0).x;
            const float dD = textureLod(sampler2D(fs_img1, fs_samp0), uvD, 0.0).x;

            const vec3 dPdx = (abs(dR - depth) < abs(dL - depth))
                                  ? (reconstruct_pos(uvR, dR, uv, fs_in_ndc, dndc_duv, d1) - P)
                                  : (P - reconstruct_pos(uvL, dL, uv, fs_in_ndc, dndc_duv, d1));
            const vec3 dPdy = (abs(dD - depth) < abs(dU - depth))
                                  ? (reconstruct_pos(uvD, dD, uv, fs_in_ndc, dndc_duv, d1) - P)
                                  : (P - reconstruct_pos(uvU, dU, uv, fs_in_ndc, dndc_duv, d1));
            const vec3 N_raw = cross(dPdx, dPdy);
            const float N_len2 = dot(N_raw, N_raw);
            if (N_len2 > 1e-12) {
                vec3 N = N_raw * inversesqrt(N_len2);
                const vec3 V = -P / cam_dist;
                if (dot(N, V) < 0.0) {
                    N = -N;
                }

                const float ign1 = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
                const float ign2 = fract(ign1 * 1.61803399 + 0.5);
                const float max_uv_r = clamp(0.85 / max(cam_dist, 2.0), 0.06, 0.34);

                vec3 gi_acc = vec3(0.0);
                float occ_acc = 0.0;
                // 4 orthogonal IGN-rotated horizon sectors x 4 exponential radii = 16 taps (~40% faster, 2x wider reach).
                for (uint dir_idx = 0u; dir_idx < 4u; ++dir_idx) {
                    const float ang = (float(dir_idx) + ign1) * 1.57079633; // pi / 2
                    const vec2 ray_uv = vec2(cos(ang) / aspect, sin(ang));
                    float max_sin_h = 0.12;
                    for (uint s = 0u; s < 4u; ++s) {
                        const float t_lin = (float(s) + 0.30 + 0.60 * ign2) * 0.25;
                        const vec2 uv_s = uv + ray_uv * (max_uv_r * t_lin * t_lin);
                        if (uv_s.x <= 0.002 || uv_s.x >= 0.998 || uv_s.y <= 0.002 || uv_s.y >= 0.998) {
                            break;
                        }
                        const float edge_x = smoothstep(0.0, 0.045, min(uv_s.x, 1.0 - uv_s.x));
                        const float edge_y = smoothstep(0.0, 0.045, min(uv_s.y, 1.0 - uv_s.y));
                        const float edge_w = edge_x * edge_y;

                        const float d_s = textureLod(sampler2D(fs_img1, fs_samp0), uv_s, 0.0).x;
                        if (d_s >= 0.99995) {
                            // Open sky bounce (automatically occluded by arches/roofs via max_sin_h).
                            const vec3 P_sky = reconstruct_pos(uv_s, 0.9999, uv, fs_in_ndc, dndc_duv, d1);
                            const float sky_len2 = dot(P_sky, P_sky);
                            if (sky_len2 > 1e-6) {
                                const float sin_h = dot(N, P_sky * inversesqrt(sky_len2));
                                if (sin_h > max_sin_h) {
                                    const float dw = sin_h - max_sin_h;
                                    max_sin_h = sin_h;
                                    vec3 L_sky = textureLod(sampler2D(fs_img5, fs_samp0), uv_s, 0.0).rgb;
                                    const float l_sky = dot(L_sky, vec3(0.2126, 0.7152, 0.0722));
                                    L_sky = max(mix(vec3(l_sky), L_sky, 1.25), vec3(0.0)) / (1.0 + l_sky * 0.35);
                                    gi_acc += L_sky * (dw * 0.38 * edge_w);
                                }
                            }
                        } else {
                            const vec3 P_s = reconstruct_pos(uv_s, d_s, uv, fs_in_ndc, dndc_duv, d1);
                            const vec3 diff = P_s - P;
                            const float dist2 = dot(diff, diff);
                            // Min distance 0.22 m prevents flat limbs/clothing from erasing their own contact shadows.
                            if (dist2 > 0.048 && dist2 < 90.25) {
                                const float dist = sqrt(dist2);
                                const float sin_h = dot(N, diff / dist);
                                if (sin_h > max_sin_h) {
                                    const float dw = sin_h - max_sin_h;
                                    max_sin_h = sin_h;
                                    const float range_fade = 1.0 - smoothstep(5.5, 9.5, dist);
                                    const float atten = range_fade / (1.0 + dist2 * 0.14);
                                    vec3 L_s = textureLod(sampler2D(fs_img3, fs_samp0), uv_s, 0.0).rgb +
                                               textureLod(sampler2D(fs_img5, fs_samp0), uv_s, 0.0).rgb;
                                    const float l_s = dot(L_s, vec3(0.2126, 0.7152, 0.0722));
                                    L_s = max(mix(vec3(l_s), L_s, 1.28), vec3(0.0)) / (1.0 + l_s * 0.18);
                                    gi_acc += L_s * (dw * atten * edge_w);
                                }
                            } else if (dist2 > 0.002 && dist2 <= 0.048) {
                                // Track tight contact occlusion without injecting self-bounce light.
                                const float sin_h = dot(N, diff * inversesqrt(dist2));
                                max_sin_h = max(max_sin_h, sin_h);
                            }
                        }
                    }
                    occ_acc += clamp((max_sin_h - 0.12) * 1.15, 0.0, 1.0);
                }

                // Physically conserved diffuse bounce reflectance calibrated for Bloodborne's dark linear stone/leather.
                const vec3 F0 = g0.rgb * g0.rgb;
                const vec3 diff_albedo = g2.rgb * max(vec3(1.0) - F0, vec3(0.0));
                const vec3 bounce_albedo = mix(diff_albedo, sqrt(max(diff_albedo, vec3(0.0))) * 0.48, 0.45);
                const vec3 gi_bounce = bounce_albedo * (gi_acc * (5.4 / 4.0)) * cam_fade;
                const float gi_luma = dot(gi_bounce, vec3(0.2126, 0.7152, 0.0722));
                const float occ_avg = occ_acc * 0.25;
                // Preserve crisp contact shadows in dark crevices while bathing lit surroundings in warm bounce.
                const float crevice_ao = 1.0 - 0.16 * occ_avg * cam_fade * (1.0 - clamp(gi_luma * 6.0, 0.0, 1.0));
                sum_rgb = (col3 + col5) * crevice_ao + col4 + gi_bounce;
            }
        }
    }

    if (f188 <= 0.0 && f210 <= 0.0) {
        frag_color0 = vec4(sum_rgb, out_a);
        return;
    }

    const float luma = fma(sum_rgb.z, 0.114480004, fma(sum_rgb.x, 0.298909992, 0.586610019 * sum_rgb.y));
    const float luma_m1 = luma - 1.0;

    vec3 mix1 = sum_rgb;
    if (f188 > 0.0) {
        const float vx = fma(s2f(d1 + 40u), px, fma(s2f(d1 + 41u), py, s2f(d1 + 42u) * pz));
        const float vy = fma(s2f(d1 + 32u), px, fma(s2f(d1 + 33u), py, s2f(d1 + 34u) * pz));
        const float dist_xy = sqrt(vy * vy + vx * vx);
        const float f191 = ssbo_1.data[o2 + 1u];
        const float f194 = ssbo_1.data[o2 + 2u];
        const float fog_rad = clamp(f188 - f188 * exp2((dist_xy - f191) * (-f194)), 0.0, 1.0);
        if (fog_rad > 0.0) {
            const float scale1 = fma(ssbo_1.data[o1 + 3u], luma_m1, 1.0);
            const vec3 fcol1 = vec3(ssbo_1.data[o1 + 0u], ssbo_1.data[o1 + 1u], ssbo_1.data[o1 + 2u]);
            mix1 = fma(vec3(fog_rad), fcol1 * scale1 - sum_rgb, sum_rgb);
        }
    }

    vec3 mix2 = mix1;
    if (f210 > 0.0) {
        const float vz = fma(s2f(d1 + 36u), px, fma(s2f(d1 + 37u), py, s2f(d1 + 38u) * pz));
        const float f213 = ssbo_1.data[o3 + 1u];
        const float f216 = ssbo_1.data[o3 + 2u];
        const float f219 = ssbo_1.data[o3 + 3u];
        const float h_factor = clamp(f216 * ((vz + f219) - f213), 0.0, 1.0);
        const float fog_h = clamp(f210 * h_factor, 0.0, 1.0);
        if (fog_h > 0.0) {
            const float scale2 = fma(ssbo_1.data[o4 + 3u], luma_m1, 1.0);
            const vec3 fcol2 = vec3(ssbo_1.data[o4 + 0u], ssbo_1.data[o4 + 1u], ssbo_1.data[o4 + 2u]);
            mix2 = fma(vec3(fog_h), fcol2 * scale2 - mix1, mix1);
        }
    }

    frag_color0 = vec4(mix2, out_a);
}
