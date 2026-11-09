#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Bloodborne Native GLSL 460 — 0xd92266b9 (Deferred 4-Directional-Light PBR + Translucency + Cloud/Shadow Tint)
layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;

layout(location = 0) out vec4 frag_color0;
layout(location = 1) out vec4 frag_color1;

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

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { float data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { uint data[]; } ssbo_2;
layout(set = 0, binding = 2, std430) readonly buffer ssbo_3_buf { uint data[]; } ssbo_3;

layout(set = 0, binding = 3) uniform texture2D fs_img0;
layout(set = 0, binding = 4) uniform texture2D fs_img1;
layout(set = 0, binding = 5) uniform texture2D fs_img2;
layout(set = 0, binding = 6) uniform texture2D fs_img3;
layout(set = 0, binding = 7) uniform texture2D fs_img4;
layout(set = 0, binding = 8) uniform sampler fs_samp0;
layout(set = 0, binding = 9) uniform sampler fs_samp1;

float safe_rsq(float x) {
    return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0;
}

float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }
float s3f(uint i) { return uintBitsToFloat(ssbo_3.data[i]); }
vec4 load_s1_v4(uint i) { return vec4(ssbo_1.data[i], ssbo_1.data[i + 1u], ssbo_1.data[i + 2u], ssbo_1.data[i + 3u]); }
vec4 load_s2_v4(uint i) { return vec4(s2f(i), s2f(i + 1u), s2f(i + 2u), s2f(i + 3u)); }

void main() {
    uint u0 = push_data.buf_offsets0.x;
    uint d0 = bitfieldExtract(u0, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(u0, 8, 8) >> 2u;
    uint d2 = bitfieldExtract(u0, 16, 8) >> 2u;

    vec2 uv = fs_in_attr1_p.xy;
    vec4 tex_spec  = texture(sampler2D(fs_img0, fs_samp0), uv);
    vec4 tex_nrm   = texture(sampler2D(fs_img1, fs_samp0), uv);
    vec4 tex_mat   = texture(sampler2D(fs_img2, fs_samp0), uv);
    vec4 tex_depth = texture(sampler2D(fs_img3, fs_samp0), uv);
    vec4 tex_sh    = texture(sampler2D(fs_img4, fs_samp1), uv);

    vec3 N_unorm = fma(vec3(2.0), tex_nrm.xyz, vec3(-1.0));
    vec3 N = N_unorm * safe_rsq(dot(N_unorm, N_unorm));

    vec4 clip_p = vec4(fs_in_attr0_p.xy, tex_depth.x, 1.0);
    float inv_w = 1.0 / dot(load_s2_v4(d1 + 12u), clip_p);
    vec3 pos_v = vec3(
        dot(load_s2_v4(d1 + 0u), clip_p),
        dot(load_s2_v4(d1 + 4u), clip_p),
        dot(load_s2_v4(d1 + 8u), clip_p)
    ) * inv_w;

    vec4 r0 = load_s2_v4(d1 + 32u);
    vec4 r1 = load_s2_v4(d1 + 36u);
    vec4 r2 = load_s2_v4(d1 + 40u);
    vec3 view_vec = vec3(
        r0.w - (dot(r0.xyz, pos_v) + r0.w),
        r1.w - (dot(r1.xyz, pos_v) + r1.w),
        r2.w - (dot(r2.xyz, pos_v) + r2.w)
    );
    vec3 V = view_vec * safe_rsq(dot(view_vec, view_vec));

    uint lset_idx = uint(int(255.0 * tex_spec.w));
    uint lbase = lset_idx * 18u;
    uint sb1 = lset_idx * 72u + d0;

    vec4 L0 = load_s1_v4(sb1 + 0u);
    vec4 L1 = load_s1_v4((lbase + 1u) * 4u + d0);
    vec3 D0 = load_s1_v4((lbase + 4u) * 4u + d0).xyz;
    vec3 D1 = load_s1_v4((lbase + 5u) * 4u + d0).xyz;
    vec3 S0 = load_s1_v4((lbase + 8u) * 4u + d0).xyz;
    vec3 S1 = load_s1_v4((lbase + 9u) * 4u + d0).xyz;
    vec4 sh_w = load_s1_v4((lbase + 17u) * 4u + d0);

    float sh0 = fma(sh_w.x, -1.0 + tex_sh.x, 1.0);
    float sh1 = fma(sh_w.y, -1.0 + sh0, 1.0);
    float sh2 = fma(sh_w.z, -1.0 + sh0, 1.0);
    float sh3 = fma(sh_w.w, -1.0 + sh0, 1.0);

    vec3 c_mod = vec3(s3f(d2 + 76u), s3f(d2 + 77u), s3f(d2 + 78u));
    vec3 sh_mod = tex_sh.z * (tex_sh.w * c_mod + (vec3(1.0) - c_mod));

    vec3 spec_sq = tex_spec.xyz * tex_spec.xyz;
    float trans = 10.0 * tex_mat.y;
    bool has_trans = (trans > 0.0);

    float rim = max(0.0, 1.0 + dot(V, N));
    float rim_term = 0.25 * (rim * rim);

    vec3 diff_acc;
    if (has_trans) {
        float inv_wrap = 1.0 / fma(2.0, trans, 1.0);
        float nl0_t = max(dot(L0.xyz, N), -dot(L0.xyz, N) * 0.85);
        float nl1_t = max(dot(L1.xyz, N), -dot(L1.xyz, N) * 0.85);
        float w0 = max(0.0, trans + nl0_t) * inv_wrap;
        float w1 = max(0.0, trans + nl1_t) * inv_wrap;
        vec3 base_d = (sh0 * (D0 * w0)) * sh_mod + sh1 * (D1 * w1);
        float b0 = exp2(32.0 * log2(clamp(dot(L0.xyz, -V), 0.0, 1.0))) * rim_term;
        float b1 = exp2(32.0 * log2(clamp(dot(L1.xyz, -V), 0.0, 1.0))) * rim_term;
        diff_acc = ((trans * (D1 * b1)) * sh1) + (((trans * (D0 * b0)) * sh0) + base_d);
    } else {
        float nl0 = clamp(dot(L0.xyz, N), 0.0, 1.0);
        float nl1 = clamp(dot(L1.xyz, N), 0.0, 1.0);
        diff_acc = (((-spec_sq * nl0) + nl0) * (D0 * sh0)) * sh_mod
                 + (D1 * sh1) * ((-spec_sq * nl1) + nl1);
    }

    vec3 H0 = (V + L0.xyz) * safe_rsq(dot(V + L0.xyz, V + L0.xyz));
    vec3 H1 = (V + L1.xyz) * safe_rsq(dot(V + L1.xyz, V + L1.xyz));
    float exp_g = exp2(13.0 * tex_mat.x);
    float exp_g_s = exp_g * 0.0100021362;

    float sp_e0 = max(1.0, (L0.w * exp_g) * (1.0 / (exp_g_s + L0.w)));
    float sp_e1 = max(1.0, (L1.w * exp_g) * (1.0 / (exp_g_s + L1.w)));
    float bl0 = exp2(sp_e0 * log2(clamp(dot(H0, N), 0.0, 1.0)));
    float bl1 = exp2(sp_e1 * log2(clamp(dot(H1, N), 0.0, 1.0)));
    float fr0 = exp2(5.0 * log2(1.0 - clamp(dot(V, H0), 0.0, 1.0)));
    float fr1 = exp2(5.0 * log2(1.0 - clamp(dot(V, H1), 0.0, 1.0)));
    float h_fade0 = clamp(dot(N, L0.xyz) * 16.0, 0.0, 1.0);
    float h_fade1 = clamp(dot(N, L1.xyz) * 16.0, 0.0, 1.0);

    vec3 spec_acc = (sh_mod * (h_fade0 * (bl0 * (sh0 * (S0 * (sp_e0 * 0.125 + 0.25)))))) * ((-fr0 * spec_sq) + (spec_sq + fr0))
                  + ((-fr1 * spec_sq) + (spec_sq + fr1)) * (h_fade1 * (bl1 * (sh1 * (S1 * (0.125 * sp_e1 + 0.25)))));

    uint mask_base = (lbase + 16u) * 4u + d0;
    if (0u < floatBitsToUint(ssbo_1.data[mask_base])) {
        vec4 L2 = load_s1_v4((lbase + 2u) * 4u + d0);
        vec3 D2 = load_s1_v4((lbase + 6u) * 4u + d0).xyz;
        vec3 S2 = load_s1_v4((lbase + 10u) * 4u + d0).xyz;
        float hemi2 = fma(0.5, dot(L2.xyz, N), 0.5);
        float nl2 = clamp(hemi2 * hemi2, 0.0, 1.0);
        diff_acc += (sh2 * D2) * ((-spec_sq * nl2) + nl2);

        vec3 H2 = (V + L2.xyz) * safe_rsq(dot(V + L2.xyz, V + L2.xyz));
        float sp_e2 = max(1.0, (exp_g * L2.w) * (1.0 / (exp_g_s + L2.w)));
        float bl2 = exp2(sp_e2 * log2(clamp(dot(H2, N), 0.0, 1.0)));
        float fr2 = exp2(5.0 * log2(1.0 - clamp(dot(V, H2), 0.0, 1.0)));
        spec_acc += (bl2 * (sh2 * (S2 * (0.125 * sp_e2 + 0.25)))) * ((-fr2 * spec_sq) + (spec_sq + fr2));
    }

    if (0u < floatBitsToUint(ssbo_1.data[mask_base + 1u])) {
        vec4 L3 = load_s1_v4((lbase + 3u) * 4u + d0);
        vec3 D3 = load_s1_v4((lbase + 7u) * 4u + d0).xyz;
        vec3 S3 = load_s1_v4((lbase + 11u) * 4u + d0).xyz;
        float hemi3 = fma(0.5, dot(L3.xyz, N), 0.5);
        float nl3 = clamp(hemi3 * hemi3, 0.0, 1.0);
        diff_acc += (sh3 * D3) * ((-spec_sq * nl3) + nl3);

        vec3 H3 = (V + L3.xyz) * safe_rsq(dot(V + L3.xyz, V + L3.xyz));
        float sp_e3 = max(1.0, (exp_g * L3.w) * (1.0 / (exp_g_s + L3.w)));
        float bl3 = exp2(sp_e3 * log2(clamp(dot(H3, N), 0.0, 1.0)));
        float fr3 = exp2(5.0 * log2(1.0 - clamp(dot(V, H3), 0.0, 1.0)));
        spec_acc += (bl3 * (sh3 * (S3 * (0.125 * sp_e3 + 0.25)))) * ((-fr3 * spec_sq) + (spec_sq + fr3));
    }

    if (((push_data.buf_offsets2.y >> 24u) & 0x01u) != 0u) {
        // Lagarde / Frostbite Specular Micro-Occlusion for directional sun/moon lighting
        float raw_ao = clamp(tex_sh.z, 0.0, 1.0);
        float ndotv = clamp(dot(V, N), 0.0, 1.0);
        float spec_ao = clamp(pow(ndotv + raw_ao, exp2(-16.0 * (1.0 - tex_mat.x) - 1.0)) - 1.0 + raw_ao, 0.0, 1.0);
        spec_acc *= mix(1.0, spec_ao / max(raw_ao, 1e-4), 0.80);
    }

    frag_color1 = vec4(spec_acc, 1.0);
    frag_color0 = vec4(diff_acc, 1.0);
}
