// SPDX-License-Identifier: GPL-2.0-or-later
// Remaster v2: Full-Res Depth-Aware Joint Bilateral SSGI Upsample & In-Place Composite.
// Consumes the 3-pass A-Trous denoised Half-Res SSGI target and blends onto hdr_image:
// dst_rgb = dst_rgb * frag_color0.a + frag_color0.rgb (colorWriteMask = R|G|B).
#version 460
#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 frag_color0;

layout(set = 0, binding = 0) uniform texture2D g0_tex;
layout(set = 0, binding = 1) uniform texture2D depth_tex;
layout(set = 0, binding = 2) uniform texture2D g2_tex;
layout(set = 0, binding = 3) uniform texture2D ssgi_denoised_tex;
layout(set = 0, binding = 4) uniform texture2D rad_mip0_tex;
layout(set = 0, binding = 5) uniform sampler linear_samp;

layout(push_constant) uniform SsgiUpsamplePush {
    float inv_half_w;
    float inv_half_h;
} pc;

void main() {
    const ivec2 depth_sz = max(textureSize(depth_tex, 0), ivec2(1));
    const ivec2 depth_max = depth_sz - ivec2(1);
    const float d_c = texelFetch(depth_tex, clamp(ivec2(uv * vec2(depth_sz)), ivec2(0), depth_max), 0).r;
    if (d_c >= 0.99995) {
        discard;
    }

    // 4-tap exact joint-bilateral upsample on the denoised Half-Res texel grid:
    // uses point texelFetch on both ssgi_denoised_tex and rad_mip0_tex so depth edges never bleed.
    const ivec2 half_sz = max(textureSize(ssgi_denoised_tex, 0), ivec2(1));
    const ivec2 half_max = half_sz - ivec2(1);
    const vec2 hf = uv * vec2(half_sz) - 0.5;
    const ivec2 h0 = ivec2(floor(hf));
    const vec2 f = clamp(hf - vec2(h0), vec2(0.0), vec2(1.0));

    const ivec2 hpts[4] = ivec2[4](
        clamp(h0 + ivec2(0, 0), ivec2(0), half_max),
        clamp(h0 + ivec2(1, 0), ivec2(0), half_max),
        clamp(h0 + ivec2(0, 1), ivec2(0), half_max),
        clamp(h0 + ivec2(1, 1), ivec2(0), half_max)
    );
    const float bw[4] = float[4](
        (1.0 - f.x) * (1.0 - f.y),
        f.x * (1.0 - f.y),
        (1.0 - f.x) * f.y,
        f.x * f.y
    );

    const float inv_d_c = max(1.0 - d_c, 1e-5);
    const float rcp_inv_d_c = 1.0 / inv_d_c;

    vec4 acc = vec4(0.0);
    float w_sum = 0.0;
    float best_dz = 1e9;
    vec4 best_val = vec4(0.0);
    for (int i = 0; i < 4; ++i) {
        const float inv_d_s = texelFetch(rad_mip0_tex, hpts[i], 0).a;
        const vec4 val = texelFetch(ssgi_denoised_tex, hpts[i], 0);
        const float dz_rel = abs(inv_d_s - inv_d_c) * rcp_inv_d_c;
        if (dz_rel < best_dz) {
            best_dz = dz_rel;
            best_val = val;
        }
        const float w = bw[i] * exp(-dz_rel * dz_rel * 48.0);
        acc += val * w;
        w_sum += w;
    }
    const vec4 gi_occ = (w_sum > 1e-4) ? (acc / w_sum) : best_val;

    // Physically conserved diffuse bounce reflectance calibrated for Bloodborne's Gothic stone/cloth.
    const vec3 g0_rgb = textureLod(sampler2D(g0_tex, linear_samp), uv, 0.0).rgb;
    const vec3 g2_rgb = textureLod(sampler2D(g2_tex, linear_samp), uv, 0.0).rgb;
    const vec3 F0 = g0_rgb * g0_rgb;
    const vec3 diff_albedo = g2_rgb * max(vec3(1.0) - F0, vec3(0.0));
    const vec3 bounce_albedo = mix(diff_albedo, sqrt(max(diff_albedo, vec3(0.0))) * 0.40, 0.28);
    const vec3 gi_bounce = bounce_albedo * gi_occ.rgb;
    const float gi_luma = dot(gi_bounce, vec3(0.2126, 0.7152, 0.0722));
    const float crevice_ao = clamp(1.0 - 0.14 * gi_occ.a * (1.0 - clamp(gi_luma * 6.0, 0.0, 1.0)), 0.84, 1.0);

    // Hardware blend: dst_rgb = dst_rgb * crevice_ao + gi_bounce
    frag_color0 = vec4(gi_bounce, crevice_ao);
}
