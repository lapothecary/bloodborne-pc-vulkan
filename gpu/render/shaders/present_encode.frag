// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the last pass before the swapchain. SDR: linear color encoded with the sRGB curve,
// its exponent shifted by the user's gamma (1: the standard curve); sRGB-encoded sources read
// through a UNORM view are decoded first. HDR: passed through. sharpen > 0 (no upscaler): RCAS
// (FSR 1's robust contrast-adaptive sharpening) over the source's texel and its 4 neighbours.
#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

layout(binding = 0) uniform sampler2D source;

layout(push_constant) uniform Settings {
    float gamma;
    uint hdr;
    uint srgb_input;
    float sharpen;
    float hdr_peak_nits;
    float hdr_paper_white;
} settings;

// The sRGB curve's constants: linear below the knee, a power above it.
const float Knee = 0.0031308;
const float Scale = 1.055;
const float Offset = 0.055;
const float Slope = 12.92;

// ITU-R BT.2087 exact Rec.709 Linear -> Rec.2020 Linear matrix (GLSL column-major).
const mat3 kRec709ToRec2020 = mat3(
    vec3(0.6274040, 0.0690970, 0.0163916),
    vec3(0.3292820, 0.9195400, 0.0880132),
    vec3(0.0433136, 0.0113612, 0.8955950)
);

vec3 LinearToPQ(vec3 L) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    const vec3 Lm1 = pow(clamp(L, 0.0, 1.0), vec3(m1));
    return pow((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), vec3(m2));
}

vec3 EncodeHdr10PQ(vec3 linear709) {
    vec3 c = max(linear709, vec3(0.0));
    if (abs(settings.gamma - 1.0) > 1e-4) {
        c = pow(c, vec3(2.4 / max(3.4 - settings.gamma, 0.5)));
    }
    const float paper_white = max(settings.hdr_paper_white, 80.0);
    const float peak_ratio = max(settings.hdr_peak_nits / paper_white, 1.0);
    const float max_c = max(c.r, max(c.g, c.b));
    const float min_c = min(c.r, min(c.g, c.b));
    const float sat = (max_c > 1e-4) ? ((max_c - min_c) / max_c) : 0.0;
    // Expand saturated bright sources (torches, bonfires, spells, sunset sky) toward full peak
    // nits while keeping neutral-white HUD text and bright speculars comfortable (~1.85x paper).
    const float target_ratio = mix(min(peak_ratio, 1.85), peak_ratio, smoothstep(0.08, 0.45, sat));
    const float hi = smoothstep(0.28, 1.0, max_c);
    const float boost = 1.0 + (target_ratio - 1.0) * pow(hi, 2.2);
    // Subtle gamut enrichment on bright flames/lights before BT.2020 conversion.
    const float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = max(mix(vec3(luma), c, 1.0 + 0.10 * hi), vec3(0.0)) * boost;
    const vec3 rec2020 = max(kRec709ToRec2020 * c, vec3(0.0));
    return LinearToPQ(rec2020 * (paper_white / 10000.0));
}

vec3 Encode(vec3 linear) {
    const float exponent = 1.0 / (3.4 - settings.gamma);
    const vec3 curve = Scale * pow(linear, vec3(exponent)) - Offset;
    return mix(curve, Slope * linear / settings.gamma, lessThan(linear, vec3(Knee)));
}

// The inverse of Encode at gamma 1.
vec3 Decode(vec3 encoded) {
    const vec3 curve = pow(max(encoded + Offset, 0.0) / Scale, vec3(2.4));
    return mix(curve, encoded / Slope, lessThan(encoded, vec3(Slope * Knee)));
}

vec3 Linear(vec3 rgb) {
    return (settings.srgb_input != 0u) ? Decode(rgb) : rgb;
}

// RCAS: the most negative lobe that keeps the result within the neighbours' range (no ringing),
// at most 3/16, scaled by the strength (1: AMD's strongest).
vec3 Rcas(vec3 e) {
    const vec2 texel_size = 1.0 / vec2(textureSize(source, 0));
    const vec3 b = clamp(Linear(texture(source, uv + vec2(0.0, -texel_size.y)).rgb), 0.0, 1.0);
    const vec3 d = clamp(Linear(texture(source, uv + vec2(-texel_size.x, 0.0)).rgb), 0.0, 1.0);
    const vec3 f = clamp(Linear(texture(source, uv + vec2(texel_size.x, 0.0)).rgb), 0.0, 1.0);
    const vec3 h = clamp(Linear(texture(source, uv + vec2(0.0, texel_size.y)).rgb), 0.0, 1.0);
    const vec3 mn4 = min(min(b, d), min(f, h));
    const vec3 mx4 = max(max(b, d), max(f, h));
    const vec3 hit_min = mn4 / max(4.0 * mx4, 1e-4);
    const vec3 hit_max = (1.0 - mx4) / min(4.0 * mn4 - 4.0, -1e-4);
    const vec3 lobe_rgb = max(-hit_min, hit_max);
    const float lobe = max(-0.1875, min(max(lobe_rgb.r, max(lobe_rgb.g, lobe_rgb.b)), 0.0)) *
                       min(settings.sharpen, 1.0);
    return (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0);
}

void main() {
    vec4 texel = texture(source, uv);
    if (settings.hdr == 1u) {
        color = texel;
        return;
    }
    texel.rgb = Linear(texel.rgb);
    if (settings.sharpen > 0.0) {
        texel.rgb = Rcas(clamp(texel.rgb, 0.0, 1.0));
    }
    if (settings.hdr == 2u) {
        color = vec4(EncodeHdr10PQ(texel.rgb), texel.a);
        return;
    }
    color = vec4(Encode(texel.rgb), texel.a);
}
