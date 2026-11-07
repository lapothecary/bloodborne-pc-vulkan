#version 460
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_demote_to_helper_invocation : require

// Bloodborne Native GLSL 460 — 0xbae5454e (1-Spot-Light Deferred PBR + Translucency)
layout(early_fragment_tests) in;

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

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { uint data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { uint data[]; } ssbo_2;

layout(set = 0, binding = 2) uniform texture2D fs_img0;
layout(set = 0, binding = 3) uniform texture2D fs_img1;
layout(set = 0, binding = 4) uniform texture2D fs_img2;
layout(set = 0, binding = 5) uniform texture2D fs_img3;
layout(set = 0, binding = 6) uniform texture2D fs_img4;
layout(set = 0, binding = 7) uniform sampler fs_samp0;
layout(set = 0, binding = 8) uniform sampler fs_samp1;

float safe_rsq(float x) {
    return (x > 0.0) ? inversesqrt(max(x, 1.17549435e-38)) : 0.0;
}

float s1f(uint idx) { return uintBitsToFloat(ssbo_1.data[idx]); }
float s2f(uint idx) { return uintBitsToFloat(ssbo_2.data[idx]); }
vec3 s1_v3(uint idx) { return vec3(s1f(idx), s1f(idx + 1u), s1f(idx + 2u)); }
vec4 s1_v4(uint idx) { return vec4(s1f(idx), s1f(idx + 1u), s1f(idx + 2u), s1f(idx + 3u)); }
vec3 s2_v3(uint idx) { return vec3(s2f(idx), s2f(idx + 1u), s2f(idx + 2u)); }

void main() {
    uint d0 = bitfieldExtract(push_data.buf_offsets0.x, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(push_data.buf_offsets0.x, 8, 8) >> 2u;

    vec2 uv = fs_in_attr1_p.xy;
    float depth = texture(sampler2D(fs_img0, fs_samp0), uv).x;

    vec4 clip_p = vec4(fs_in_attr0_p.xy, depth, 1.0);
    float inv_w = 1.0 / dot(s1_v4(d0 + 12u), clip_p);
    vec3 pos = vec3(
        dot(s1_v4(d0 + 0u), clip_p),
        dot(s1_v4(d0 + 4u), clip_p),
        dot(s1_v4(d0 + 8u), clip_p)
    ) * inv_w;

    float px = dot(s2_v3(d1 + 16u), pos) + s2f(d1 + 19u);
    float py = dot(s2_v3(d1 + 20u), pos) + s2f(d1 + 23u);
    float pz = dot(s2_v3(d1 + 24u), pos) + s2f(d1 + 27u);
    float pw = dot(s2_v3(d1 + 28u), pos) + s2f(d1 + 31u);

    if ((0.0 >= pz) || (px * px + py * py >= pw * pw)) {
        demote;
        frag_color1 = vec4(0.0, 0.0, 0.0, 1.0);
        frag_color0 = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec4 t_norm = texture(sampler2D(fs_img1, fs_samp0), uv);
    vec4 t_mat  = texture(sampler2D(fs_img2, fs_samp0), uv);
    vec4 t_alb  = texture(sampler2D(fs_img3, fs_samp0), uv);
    float ao    = texture(sampler2D(fs_img4, fs_samp1), uv).x;

    vec3 n_raw = fma(vec3(2.0), t_norm.xyz, vec3(-1.0));
    vec3 n_ts = n_raw * safe_rsq(dot(n_raw, n_raw));
    vec3 N = vec3(
        dot(s1_v3(d0 + 16u), n_ts),
        dot(s1_v3(d0 + 20u), n_ts),
        dot(s1_v3(d0 + 24u), n_ts)
    );

    float pos_rsq = safe_rsq(dot(pos, pos));
    vec3 V_dir = pos * pos_rsq;
    vec3 V = -pos * safe_rsq(dot(-pos, -pos));
    vec3 R_v = V_dir - N * (2.0 * dot(N, V_dir));

    float rim = 0.5 * max(0.0, 1.0 + dot(-V_dir, N));
    float rim_factor = rim * rim;

    float exp_13 = exp2(13.0 * t_mat.x);
    float trans = 10.0 * t_mat.y;

    vec3 L_vec = s2_v3(d1 + 0u) - pos;
    float d2 = dot(L_vec, L_vec);
    vec3 L = L_vec * safe_rsq(d2);
    float NdotL = dot(N, L);
    float NdotL_pos = clamp(NdotL, 0.0, 1.0);

    float r = s2f(d1 + 3u);
    float inv_r2 = 1.0 / (r * r);
    float inv_pw = 1.0 / pw;
    float radial = clamp(1.0 - ((py * inv_pw) * (py * inv_pw) + (px * inv_pw) * (px * inv_pw)), 0.0, 1.0);

    float spec_pow = max(1.0, (s2f(d1 + 7u) * exp_13) / (exp_13 * 0.0100021362 + s2f(d1 + 7u)));
    vec3 light_col = s2_v3(d1 + 4u);
    vec3 spec_col  = s2_v3(d1 + 8u);
    float trans_scale = s2f(d1 + 12u);
    float trans_pow   = s2f(d1 + 13u) * 32.0;

    float att_diff;
    vec3 spec_contrib;
    if (s2f(d1 + 15u) > 0.0) {
        float cone_att = radial;
        att_diff = max(0.0, (1.0 / d2) - inv_r2) * cone_att;
        float att_spec = (1.0 - clamp(d2 * inv_r2, 0.0, 1.0)) * cone_att;
        vec3 H = (L + V) * safe_rsq(dot(L + V, L + V));
        float NdotH_pow = exp2(spec_pow * log2(clamp(dot(H, N), 0.0, 1.0)));
        float NdotL_grazing = exp2(0.0999755859 * log2(NdotL_pos));
        spec_contrib = (0.125 * spec_pow + 0.25) * (att_spec * (NdotL_grazing * (spec_col * NdotH_pow)));
    } else {
        float cone_att = radial * radial;
        float dist_att = clamp(1.0 - (pw / r), 0.0, 1.0);
        att_diff = cone_att * (dist_att * dist_att);
        float RdotL_pow = exp2((0.25 * spec_pow) * log2(clamp(dot(L, R_v), 0.0, 1.0))) * (0.125 * spec_pow + 0.25);
        spec_contrib = att_diff * (spec_col * RdotL_pow);
    }

    vec3 diff_contrib;
    if (trans == 0.0) {
        diff_contrib = att_diff * (light_col * NdotL_pos);
    } else {
        float wrap = att_diff * (max(0.0, trans + NdotL) / fma(2.0, trans, 1.0));
        float trans_spec = exp2(trans_pow * log2(clamp(dot(L, V_dir), 0.0, 1.0))) * (trans_scale * rim_factor);
        diff_contrib = light_col * (wrap + trans * (att_diff * trans_spec));
    }

    vec3 albedo2 = t_alb.xyz * t_alb.xyz;
    frag_color1 = vec4(albedo2 * (ao * spec_contrib), 1.0);
    frag_color0 = vec4(ao * (diff_contrib - (albedo2 * diff_contrib) * 0.5), 1.0);
}
