#version 460
#extension GL_EXT_samplerless_texture_functions : require

// Bloodborne Native GLSL 460 — 0xe0305cef (Post-FX Velocity-Guided Bilateral Motion Blur)
layout(location = 0) in vec4 fs_in_attr0_p;

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
layout(set = 0, binding = 3) uniform sampler fs_samp0;
layout(set = 0, binding = 4) uniform sampler fs_samp1;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }

void main() {
    uint d0 = bitfieldExtract(push_data.buf_offsets0.x, 0, 8) >> 2u;

    vec2 uv = fs_in_attr0_p.xy;
    vec4 c0 = textureLod(sampler2D(fs_img0, fs_samp0), uv, 0.0);
    vec2 v0_raw = textureLod(sampler2D(fs_img1, fs_samp1), uv, 0.0).xy;

    vec2 v_scale = vec2(s1f(d0 + 744u), s1f(d0 + 745u));
    vec2 v_bias  = vec2(s1f(d0 + 746u), s1f(d0 + 747u));
    vec2 vel = v_scale * v0_raw + v_bias;

    if (s1f(d0 + 754u) <= (abs(vel.x) + abs(vel.y))) {
        float v_len = sqrt(vel.y * vel.y + vel.x * vel.x);
        vec2 step_uv = vec2(s1f(d0 + 748u), s1f(d0 + 749u)) * v0_raw + vec2(s1f(d0 + 750u), s1f(d0 + 751u));
        float min_w = s1f(d0 + 757u) + c0.w;
        float min_dot = v_len * (s1f(d0 + 756u) * v_len);

        vec2 uv_m = uv - step_uv;
        vec2 uv_p = uv + step_uv;
        vec2 b_scale = vec2(s1f(d0 + 760u), s1f(d0 + 761u));
        vec2 b_bias  = vec2(s1f(d0 + 762u), s1f(d0 + 763u));
        vec2 bm = b_scale * uv_m + b_bias;
        vec2 bp = b_scale * uv_p + b_bias;

        vec4 acc = c0;
        float weight = 1.0;

        if ((1.0 >= abs(bm.y)) && (1.0 >= abs(bm.x))) {
            vec2 vm_raw = textureLod(sampler2D(fs_img1, fs_samp1), uv_m, 0.0).xy;
            vec4 cm = textureLod(sampler2D(fs_img0, fs_samp0), uv_m, 0.0);
            vec2 vel_m = v_scale * vm_raw + v_bias;
            float d_m = vel_m.y * vel.y + vel.x * vel_m.x;
            if ((cm.w >= min_w) || (d_m > min_dot)) {
                acc += cm;
                weight = 2.0;
            }
        }

        if ((1.0 >= abs(bp.y)) && (1.0 >= abs(bp.x))) {
            vec2 vp_raw = textureLod(sampler2D(fs_img1, fs_samp1), uv_p, 0.0).xy;
            vec4 cp = textureLod(sampler2D(fs_img0, fs_samp0), uv_p, 0.0);
            vec2 vel_p = v_scale * vp_raw + v_bias;
            float d_p = vel_p.y * vel.y + vel.x * vel_p.x;
            if ((cp.w >= min_w) || (d_p > min_dot)) {
                acc += cp;
                weight = 1.0 + weight;
            }
        }

        float inv_w = 1.0 / weight;
        vec4 tint = vec4(s1f(d0 + 628u), s1f(d0 + 629u), s1f(d0 + 630u), s1f(d0 + 631u)) * inv_w;
        frag_color0 = acc * tint;
    } else {
        frag_color0 = c0;
    }
}
