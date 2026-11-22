// SPDX-License-Identifier: GPL-2.0-or-later
// Remaster v2 Performance Audit: Full-Res Depth-Aware Bilateral SSGI Upsample & In-Place Composite.
// Rendered onto hdr_image with hardware blend: dst_rgb = dst_rgb * frag_color0.a + frag_color0.rgb
// (colorWriteMask = R|G|B, preserving hdr_image.a untouched and needing zero full-res copies).
#version 460

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
    const float d_c = textureLod(sampler2D(depth_tex, linear_samp), uv, 0.0).r;
    if (d_c >= 0.99995) {
        discard;
    }

    // 5-tap depth-aware bilateral upsample & denoise from Half-Res SSGI:
    // Eliminates 4x4 IGN grain while preserving razor-sharp silhouette depth boundaries!
    const float inv_one_minus_d = 1.0 / max(1.0 - d_c, 1e-4);
    const vec2 step_uv = vec2(pc.inv_half_w, pc.inv_half_h) * 0.85;
    const vec2 offs[5] = vec2[5](
        vec2( 0.0,  0.0),
        vec2(-step_uv.x, -step_uv.y),
        vec2( step_uv.x, -step_uv.y),
        vec2(-step_uv.x,  step_uv.y),
        vec2( step_uv.x,  step_uv.y)
    );
    const float spatial_w[5] = float[5](1.0, 0.65, 0.65, 0.65, 0.65);

    vec4 acc = vec4(0.0);
    float w_sum = 0.0;
    for (int i = 0; i < 5; ++i) {
        const vec2 suv = clamp(uv + offs[i], vec2(0.001), vec2(0.999));
        const float d_s = textureLod(sampler2D(rad_mip0_tex, linear_samp), suv, 0.0).a;
        const float dz_rel = abs(d_s - d_c) * inv_one_minus_d;
        const float w = spatial_w[i] * exp(-dz_rel * dz_rel * 28.0);
        acc += textureLod(sampler2D(ssgi_raw_tex, linear_samp), suv, 0.0) * w;
        w_sum += w;
    }
    const vec4 gi_occ = (w_sum > 1e-4)
                            ? (acc / w_sum)
                            : textureLod(sampler2D(ssgi_raw_tex, linear_samp), uv, 0.0);

    // Exact physically conserved diffuse bounce reflectance (matches original Stage 1.3 calibration).
    const vec3 g0_rgb = textureLod(sampler2D(g0_tex, linear_samp), uv, 0.0).rgb;
    const vec3 g2_rgb = textureLod(sampler2D(g2_tex, linear_samp), uv, 0.0).rgb;
    const vec3 F0 = g0_rgb * g0_rgb;
    const vec3 diff_albedo = g2_rgb * max(vec3(1.0) - F0, vec3(0.0));
    const vec3 bounce_albedo = mix(diff_albedo, sqrt(max(diff_albedo, vec3(0.0))) * 0.48, 0.45);
    const vec3 gi_bounce = bounce_albedo * gi_occ.rgb;
    const float gi_luma = dot(gi_bounce, vec3(0.2126, 0.7152, 0.0722));
    const float crevice_ao = clamp(1.0 - 0.16 * gi_occ.a * (1.0 - clamp(gi_luma * 6.0, 0.0, 1.0)), 0.80, 1.0);

    // Hardware blend: dst_rgb = dst_rgb * crevice_ao + gi_bounce
    frag_color0 = vec4(gi_bounce, crevice_ao);
}
