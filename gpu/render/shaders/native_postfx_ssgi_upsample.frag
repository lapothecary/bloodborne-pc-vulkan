// SPDX-License-Identifier: GPL-2.0-or-later
// Remaster v2 Performance Audit: Full-Res Depth-Aware Bilateral SSGI Upsample & In-Place Composite.
// Rendered onto hdr_image with hardware blend: dst_rgb = dst_rgb * frag_color0.a + frag_color0.rgb
// (colorWriteMask = R|G|B, preserving hdr_image.a untouched and needing zero full-res copies).
#version 460
#extension GL_EXT_samplerless_texture_functions : require

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 frag_color0;

layout(set = 0, binding = 0) uniform texture2D g0_tex;
layout(set = 0, binding = 1) uniform texture2D depth_tex;
layout(set = 0, binding = 2) uniform texture2D g2_tex;
layout(set = 0, binding = 3) uniform texture2D ssgi_raw_tex;
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

    // 9-tap (3x3) depth-aware bilateral upsample & wide denoise from Half-Res SSGI:
    // Both `inv_d_c` and `rad_mip0_tex.a` store `1.0 - depth` (64-256x higher FP16 precision!).
    const float inv_d_c = max(1.0 - d_c, 1e-5);
    const float rcp_inv_d_c = 1.0 / inv_d_c;
    const vec2 step_uv = vec2(pc.inv_half_w, pc.inv_half_h) * 1.35;
    const vec2 offs[9] = vec2[9](
        vec2( 0.0,        0.0),
        vec2(-step_uv.x,  0.0),
        vec2( step_uv.x,  0.0),
        vec2( 0.0,       -step_uv.y),
        vec2( 0.0,        step_uv.y),
        vec2(-step_uv.x, -step_uv.y),
        vec2( step_uv.x, -step_uv.y),
        vec2(-step_uv.x,  step_uv.y),
        vec2( step_uv.x,  step_uv.y)
    );
    const float spatial_w[9] = float[9](1.0, 0.72, 0.72, 0.72, 0.72, 0.48, 0.48, 0.48, 0.48);

    vec4 acc = vec4(0.0);
    float w_sum = 0.0;
    for (int i = 0; i < 9; ++i) {
        const vec2 suv = clamp(uv + offs[i], vec2(0.001), vec2(0.999));
        const float inv_d_s = textureLod(sampler2D(rad_mip0_tex, linear_samp), suv, 0.0).a;
        const float dz_rel = abs(inv_d_s - inv_d_c) * rcp_inv_d_c;
        const float w = spatial_w[i] * exp(-dz_rel * dz_rel * 36.0);
        acc += textureLod(sampler2D(ssgi_raw_tex, linear_samp), suv, 0.0) * w;
        w_sum += w;
    }
    const vec4 gi_occ = (w_sum > 1e-4)
                            ? (acc / w_sum)
                            : textureLod(sampler2D(ssgi_raw_tex, linear_samp), uv, 0.0);

    // Physically conserved diffuse bounce reflectance calibrated for Bloodborne's linear stone/cloth.
    const vec3 g0_rgb = textureLod(sampler2D(g0_tex, linear_samp), uv, 0.0).rgb;
    const vec3 g2_rgb = textureLod(sampler2D(g2_tex, linear_samp), uv, 0.0).rgb;
    const vec3 F0 = g0_rgb * g0_rgb;
    const vec3 diff_albedo = g2_rgb * max(vec3(1.0) - F0, vec3(0.0));
    const vec3 bounce_albedo = mix(diff_albedo, sqrt(max(diff_albedo, vec3(0.0))) * 0.46, 0.38);
    const vec3 gi_bounce = bounce_albedo * gi_occ.rgb;
    const float gi_luma = dot(gi_bounce, vec3(0.2126, 0.7152, 0.0722));
    const float crevice_ao = clamp(1.0 - 0.16 * gi_occ.a * (1.0 - clamp(gi_luma * 6.0, 0.0, 1.0)), 0.80, 1.0);

    // Hardware blend: dst_rgb = dst_rgb * crevice_ao + gi_bounce
    frag_color0 = vec4(gi_bounce, crevice_ao);
}
