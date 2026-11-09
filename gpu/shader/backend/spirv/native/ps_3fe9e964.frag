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

// Deferred IBL / SSR Ambient Combine 2-MRT — ps_3fe9e964
layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;

layout(location = 0) out vec4 frag_color0;
layout(location = 1) out vec4 frag_color1;

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { uint data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { uint data[]; } ssbo_2;

layout(set = 0, binding = 2) uniform texture2D fs_img0;
layout(set = 0, binding = 3) uniform texture2D fs_img1;
layout(set = 0, binding = 4) uniform texture2D fs_img2;
layout(set = 0, binding = 5) uniform texture2D fs_img3;
layout(set = 0, binding = 6) uniform texture2D fs_img4;
layout(set = 0, binding = 7) uniform texture2D fs_img5;
layout(set = 0, binding = 8) uniform texture2D fs_img6;
layout(set = 0, binding = 9) uniform texture2D fs_img7;
layout(set = 0, binding = 10) uniform sampler fs_samp0;
layout(set = 0, binding = 11) uniform sampler fs_samp1;
layout(set = 0, binding = 12) uniform sampler fs_samp2;
layout(set = 0, binding = 13) uniform sampler fs_samp3;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
vec4 load_s1_v4(uint i) { return vec4(s1f(i), s1f(i + 1u), s1f(i + 2u), s1f(i + 3u)); }

float safe_rsq(float x) {
    return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0;
}

void main() {
    uint u0 = push_data.buf_offsets0.x;
    uint d0 = bitfieldExtract(u0, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(u0, 8, 8) >> 2u;

    vec2 uv = fs_in_attr1_p.xy;
    vec4 tex_gloss = texture(sampler2D(fs_img0, fs_samp0), uv);
    vec4 tex_depth = texture(sampler2D(fs_img1, fs_samp0), uv);
    vec4 tex_nrm   = texture(sampler2D(fs_img2, fs_samp1), uv);
    vec4 tex_spec  = texture(sampler2D(fs_img3, fs_samp0), uv);
    vec4 tex_ao    = texture(sampler2D(fs_img4, fs_samp2), uv);
    vec4 tex_env   = texture(sampler2D(fs_img5, fs_samp0), uv);
    float ssr_mip  = min(5.0, 6.5 - tex_gloss.x * 6.5);
    vec4 tex_ssr   = textureLod(sampler2D(fs_img6, fs_samp3), uv, ssr_mip);
    vec4 tex_diff  = texture(sampler2D(fs_img7, fs_samp0), uv);

    vec4 clip_p = vec4(fs_in_attr0_p.xy, tex_depth.x, 1.0);
    float inv_w = 1.0 / dot(load_s1_v4(d0 + 12u), clip_p);
    vec3 pos_v = vec3(
        dot(load_s1_v4(d0 + 0u), clip_p),
        dot(load_s1_v4(d0 + 4u), clip_p),
        dot(load_s1_v4(d0 + 8u), clip_p)
    ) * inv_w;
    vec3 V = -pos_v * safe_rsq(dot(pos_v, pos_v));

    vec3 N_unorm = fma(vec3(2.0), tex_nrm.xyz, vec3(-1.0));
    vec3 N = N_unorm * safe_rsq(dot(N_unorm, N_unorm));
    vec3 V_world = vec3(
        dot(load_s1_v4(d0 + 32u).xyz, V),
        dot(load_s1_v4(d0 + 36u).xyz, V),
        dot(load_s1_v4(d0 + 40u).xyz, V)
    );
    float NdotV = clamp(dot(V_world, N), 0.0, 1.0);
    float fresnel = exp2(5.0 * log2(1.0 - NdotV));

    vec3 ao_w = vec3(s2f(d1 + 76u), s2f(d1 + 77u), s2f(d1 + 78u));
    vec3 ao_factor = (tex_ao.x * tex_ao.y) * fma(vec3(tex_env.w), -1.0 + (tex_ao.w * ao_w + (1.0 - ao_w)), vec3(1.0));

    vec3 spec_sq = tex_spec.xyz * tex_spec.xyz;
    float gloss_sq = tex_gloss.x * tex_gloss.x;
    vec3 spec_fresnel = spec_sq + ((-spec_sq * fresnel + fresnel) * gloss_sq);
    vec3 spec_ao_factor = ao_factor;
    if (((push_data.buf_offsets2.y >> 24u) & 0x01u) != 0u) {
        // Lagarde / Frostbite Specular Micro-Occlusion: suppresses glowing grazing specular inside crevices
        float raw_ao = clamp(tex_ao.x * tex_ao.y, 0.0, 1.0);
        float spec_ao = clamp(pow(NdotV + raw_ao, exp2(-16.0 * (1.0 - tex_gloss.x) - 1.0)) - 1.0 + raw_ao, 0.0, 1.0);
        spec_ao_factor *= mix(1.0, spec_ao / max(raw_ao, 1e-4), 0.85);
    }
    vec3 spec_out = spec_ao_factor * (tex_ssr.w * (spec_fresnel * tex_ssr.xyz - tex_env.xyz) + tex_env.xyz);

    frag_color1 = vec4(spec_out, 1.0);
    frag_color0 = vec4(tex_diff.xyz * ao_factor, 1.0);
}
