// SPDX-License-Identifier: GPL-2.0-or-later
// Native Vulkan 1.3 Post-Process — Step 1: Unified Tonemap + Original 3D LUT +
// True Black & Vibrance + Chromatic Aberration + Vignette + Auto-Exposure.
#version 460

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 frag_color0;

layout(set = 0, binding = 0) uniform texture2D hdr_scene;
layout(set = 0, binding = 1) uniform texture3D lut3d_tex;
layout(set = 0, binding = 2) uniform sampler linear_samp;
layout(set = 0, binding = 3, std430) readonly buffer ExposureStateBuffer {
    float adapted_exposure;
    float adapted_luma;
    float target_exposure;
    uint initialized;
} exp_state;

// Packed uniform block pushed via small stream/UBO binding 4 so push constants stay <= 128B.
layout(set = 0, binding = 4, std140) uniform PostFxParams {
    vec4 tone_params;     // x=exposure, y=inv_gamma, z=tone_factor, w=filmic_contrast
    vec4 scene_vig;       // x=inv_scene_lum_scale, y=vig_power, z=vig_tan_half_fov, w=aspect
    vec4 color_corr;      // x=hue_rad, y=saturation, z=value_scale, w=contrast
    vec4 ca_params;       // xy=lat_disp, zw=uni_disp
    vec4 bloom_params;    // x=bloom_intensity, y=reserved, z=reserved, w=reserved
    uvec4 flags_misc;     // x=flags, y=tone_type, z=ca_samples, w=frame_index
} u;

layout(set = 0, binding = 5) uniform texture2D bloom_tex;

// flags bits:
//  bit 0: kLut3dEnabled
//  bit 1: kAutoExposureEnabled
//  bit 2: kCaEnabled
//  bit 3: kTrueBlackEnabled
//  bit 4: kBgraExport
//  bit 5: kBloomEnabled

float IgnNoise(vec2 px, uint frame_idx) {
    const vec2 p = px + vec2(float(frame_idx & 63u) * 5.588238, float(frame_idx & 63u) * 7.131917);
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

vec3 SampleSceneWithCA(vec2 base_uv, bool ca_on, vec2 lat_disp, vec2 uni_disp, uint wave_samples) {
    if (!ca_on || (dot(lat_disp, lat_disp) + dot(uni_disp, uni_disp) < 1e-10)) {
        return max(textureLod(sampler2D(hdr_scene, linear_samp), base_uv, 0.0).rgb, vec3(0.0));
    }
    // Resolution-independent radial + uniform chromatic dispersion matching Yebis2's lens model.
    const vec2 centered = base_uv - vec2(0.5);
    const float r2 = dot(centered, centered);
    // Keep subtle physical lens shift (Yebis2 latDisp is typically 0.005..0.03 in Bloodborne).
    const vec2 shift = centered * (lat_disp * (0.35 + 1.30 * r2)) + uni_disp * 0.01;

    if (wave_samples <= 3u) {
        const float r = textureLod(sampler2D(hdr_scene, linear_samp), clamp(base_uv + shift, 0.0, 1.0), 0.0).r;
        const float g = textureLod(sampler2D(hdr_scene, linear_samp), base_uv, 0.0).g;
        const float b = textureLod(sampler2D(hdr_scene, linear_samp), clamp(base_uv - shift, 0.0, 1.0), 0.0).b;
        return max(vec3(r, g, b), vec3(0.0));
    }

    // 5-tap or 7-tap spectral dispersion (smooth, zero colour banding at high resolutions).
    const int taps = (wave_samples >= 7u) ? 7 : 5;
    const float inv_half = 2.0 / float(taps - 1);
    vec3 accum = vec3(0.0);
    vec3 w_sum = vec3(0.0);
    for (int i = 0; i < taps; ++i) {
        const float t = float(i) * inv_half - 1.0; // -1 (blue) .. 0 (green) .. +1 (red)
        const vec2 suv = clamp(base_uv + shift * t, 0.0, 1.0);
        const vec3 s = max(textureLod(sampler2D(hdr_scene, linear_samp), suv, 0.0).rgb, vec3(0.0));
        // Smooth spectral weights: R peaks at +1, G peaks at 0, B peaks at -1.
        const vec3 w = vec3(
            clamp(0.5 + 0.5 * t, 0.0, 1.0) * clamp(0.5 + 0.5 * t, 0.0, 1.0),
            max(1.0 - abs(t), 0.0),
            clamp(0.5 - 0.5 * t, 0.0, 1.0) * clamp(0.5 - 0.5 * t, 0.0, 1.0)
        );
        accum += s * w;
        w_sum += w;
    }
    return accum / max(w_sum, vec3(1e-4));
}

vec3 ApplyHueSatValContrast(vec3 c, vec4 cc) {
    // Only pay ALU cost when GPARAM actually deviates from identity (hue=0, sat=1, val=1, contrast=1).
    const float hue = cc.x;
    const float sat = cc.y;
    const float val = cc.z;
    const float contrast = cc.w;

    if (abs(hue) > 1e-4) {
        const float cosH = cos(hue);
        const float sinH = sin(hue);
        const vec3 k = vec3(0.57735027); // 1/sqrt(3)
        c = c * cosH + cross(k, c) * sinH + k * dot(k, c) * (1.0 - cosH);
    }
    if (abs(sat - 1.0) > 1e-4) {
        const float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c = mix(vec3(luma), c, sat);
    }
    if (abs(val - 1.0) > 1e-4) {
        c *= val;
    }
    if (abs(contrast - 1.0) > 1e-4) {
        c = (c - 0.5) * contrast + 0.5;
    }
    return clamp(c, 0.0, 1.0);
}

void main() {
    const uint flags = u.flags_misc.x;
    const bool lut3d_on = (flags & 1u) != 0u;
    const bool auto_exp_on = (flags & 2u) != 0u;
    const bool ca_on = (flags & 4u) != 0u;
    const bool true_black_on = (flags & 8u) != 0u;
    const bool bgra_export = (flags & 16u) != 0u;
    const bool bloom_on = (flags & 32u) != 0u;

    // 1. Sample linear HDR scene (with resolution-independent Chromatic Aberration if enabled).
    vec3 scene_rgb = SampleSceneWithCA(uv, ca_on, u.ca_params.xy, u.ca_params.zw, u.flags_misc.z);

    // 2. Optical Vignette (cos^power law from Yebis2, modulated by FOV dependence).
    float vig = 1.0;
    const float vig_power = u.scene_vig.y;
    if (vig_power > 1e-4) {
        const vec2 ndc = uv * 2.0 - 1.0;
        const vec2 d = vec2(ndc.x * u.scene_vig.w, ndc.y) * u.scene_vig.z;
        const float cos_theta = inversesqrt(1.0 + dot(d, d));
        vig = pow(cos_theta, vig_power);
    }

    // 3. Exposure (true-dt Auto-Exposure SSBO or manual GPARAM exposure) * SceneLuminanceScale.
    const float exposure = (auto_exp_on && exp_state.initialized != 0u)
                               ? exp_state.adapted_exposure
                               : u.tone_params.x;
    const vec3 hdr = max(scene_rgb * (exposure * u.scene_vig.x * vig), vec3(0.0));

    // 4. Exact Yebis2 Tone Mapping Curve (matches ps_5e67d866.frag / ps_5c02eaf8.frag).
    const uint tone_type = u.flags_misc.y;
    vec3 tm;
    if (tone_type == 0u) {
        // Linear
        tm = clamp(hdr, 0.0, 1.0);
    } else if (tone_type == 1u) {
        // Reinhard
        tm = hdr / (vec3(1.0) + hdr);
    } else {
        // Yebis2 Filmic / S-Curve Exponential (Bloodborne default):
        //   ex = exp(-tone_factor * max(hdr, 6.1035e-5))
        //   tm = (1 - filmic_contrast * ex)^2 * (1 - ex)
        const float tone_factor = max(u.tone_params.z, 1e-3);
        const float filmic_c = clamp(u.tone_params.w, 0.0, 1.0);
        const vec3 ex = exp2((-1.44269502 * tone_factor) * max(vec3(6.10351562e-05), hdr));
        const vec3 tm_base = fma(vec3(-filmic_c), ex, vec3(1.0));
        tm = clamp((tm_base * tm_base) * (vec3(1.0) - ex), 0.0, 1.0);
    }

    // 4b. FP16 Bloom & Glare Screen Blend (exact Yebis2 formula from ps_5e67d866.frag:
    //     comb = tm + blm - tm * blm, preserving highlights without hard clipping).
    if (bloom_on && u.bloom_params.x > 1e-5) {
        const vec3 bloom_raw = max(textureLod(sampler2D(bloom_tex, linear_samp), uv, 0.0).rgb, vec3(0.0));
        const vec3 blm = clamp(bloom_raw * u.bloom_params.x, 0.0, 1.0);
        tm = clamp(fma(-tm, blm, tm + blm), 0.0, 1.0);
    }

    // 5. Optional GPARAM HSV & Contrast Color Correction (applied before gamma, like Yebis2).
    tm = ApplyHueSatValContrast(tm, u.color_corr);

    // 6. Gamma Encode (pow(tm, 1.0 / gamma) matching ps_5e67d866.frag).
    vec3 c = exp2(u.tone_params.y * log2(max(tm, vec3(2.38418579e-07))));

    // 7. Original FromSoftware 16x16x16 Biome 3D LUT (fused from Part D1 / ps_ed4c7668_lut3d.frag).
    if (lut3d_on) {
        const vec3 lut_uv = fma(clamp(c, 0.0, 1.0), vec3(0.9375), vec3(0.03125));
        c = max(textureLod(sampler3D(lut3d_tex, linear_samp), lut_uv, 0.0).rgb, vec3(0.0));
    }

    // 8. True Black / OLED Shadow De-Haze + Filmic S-Contrast + Skin/Blood-Safe Vibrance
    //    (1:1 with ps_ed4c7668_lut3d.frag).
    if (true_black_on) {
        const float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
        const float shadow_weight = 1.0 - smoothstep(0.0, 0.22, luma);
        const float toe = 0.011 * shadow_weight;
        c = max(c - vec3(toe), vec3(0.0)) / max(1.0 - toe, 1e-4);
        const vec3 s_curve = c * c * (3.0 - 2.0 * c);
        c = mix(c, s_curve, 0.12);
        const float luma2 = dot(c, vec3(0.2126, 0.7152, 0.0722));
        const float max_c = max(c.r, max(c.g, c.b));
        const float min_c = min(c.r, min(c.g, c.b));
        const float sat = max_c - min_c;
        c = clamp(mix(vec3(luma2), c, 1.0 + 0.14 * (1.0 - sat)), 0.0, 1.0);
    }

    // 9. Triangular Interleaved Gradient Noise Dither (+/- 0.5/255) to eliminate 8-bit shadow banding.
    const float n1 = IgnNoise(gl_FragCoord.xy, u.flags_misc.w);
    const float n2 = IgnNoise(gl_FragCoord.xy + vec2(17.0, 31.0), u.flags_misc.w + 1u);
    const float tri_dither = (n1 - n2) * (0.5 / 255.0);
    c = clamp(c + vec3(tri_dither), 0.0, 1.0);

    frag_color0 = vec4(bgra_export ? c.bgr : c, 1.0);
}
