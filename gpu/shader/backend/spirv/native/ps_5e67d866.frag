#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Bloodborne Native GLSL 460 — 0x5e67d866 (Post-FX Tonemap + Bloom + Color Grade + Grain BGRA)
layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 3) in vec4 fs_in_attr2_p;
layout(location = 4) in vec4 fs_in_attr3_p;

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

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { uint data[]; } ssbo_1;

layout(set = 0, binding = 1) uniform texture2D fs_img0;
layout(set = 0, binding = 2) uniform texture2D fs_img1;
layout(set = 0, binding = 3) uniform texture2D fs_img2;
layout(set = 0, binding = 4) uniform texture2D fs_img3;
layout(set = 0, binding = 5) uniform sampler fs_samp0;
layout(set = 0, binding = 6) uniform sampler fs_samp1;
layout(set = 0, binding = 7) uniform sampler fs_samp2;
layout(set = 0, binding = 8) uniform sampler fs_samp3;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }

void main() {
    uint d0 = bitfieldExtract(push_data.buf_offsets0.x, 0, 8) >> 2u;

    vec3 c0 = textureLod(sampler2D(fs_img0, fs_samp0), fs_in_attr2_p.xy, 0.0).xyz;
    vec4 c1 = textureLod(sampler2D(fs_img1, fs_samp1), fs_in_attr0_p.xy, 0.0);
    vec3 c2 = textureLod(sampler2D(fs_img2, fs_samp2), fs_in_attr1_p.xy, 0.0).xyz;
    float grain = textureLod(sampler2D(fs_img3, fs_samp3), fs_in_attr3_p.xy, 0.0).x;

    vec3 v0 = fma(c0, vec3(s1f(d0 + 752u)), vec3(s1f(d0 + 753u)));
    float m0 = 0.333333343 * (v0.x + (v0.y + v0.z));
    vec3 g0 = fma(fma(vec3(s1f(d0 + 756u)), v0 - m0, vec3(m0)), vec3(s1f(d0 + 754u)), vec3(s1f(d0 + 755u)));

    vec3 v1 = fma(c0, vec3(s1f(d0 + 744u)), vec3(s1f(d0 + 745u)));
    float m1 = 0.333333343 * (v1.x + (v1.y + v1.z));
    vec3 g1 = fma(clamp(fma(vec3(s1f(d0 + 748u)), v1 - m1, vec3(m1)), 0.0, 1.0), vec3(s1f(d0 + 746u)), vec3(s1f(d0 + 747u)));

    vec3 hdr = g0 * (s1f(d0 + 616u) * c1.xyz);
    vec3 ex = exp2((-1.44269502 * s1f(d0 + 832u)) * max(vec3(6.10351562e-05), hdr));
    vec3 tm_base = fma(vec3(-s1f(d0 + 833u)), ex, vec3(1.0));
    vec3 tm = clamp((tm_base * tm_base) * (vec3(1.0) - ex), 0.0, 1.0);

    vec3 blm = clamp(s1f(d0 + 617u) * fma(c2, g0, g1), 0.0, 1.0);
    vec3 comb = clamp(fma(-tm, blm, tm + blm) + vec3(2.38418579e-07), 0.0, 1.0);
    vec3 gamma_rgb = exp2(s1f(d0 + 808u) * log2(comb));

    vec3 out_rgb = fma(vec3(s1f(d0 + 809u)), vec3(grain), gamma_rgb) + vec3(s1f(d0 + 810u));
    vec4 aw = vec4(s1f(d0 + 620u), s1f(d0 + 621u), s1f(d0 + 622u), s1f(d0 + 623u));
    float out_a = clamp(dot(c1, aw), 0.0, 1.0);

    frag_color0 = vec4(out_rgb.zyx, out_a);
}
