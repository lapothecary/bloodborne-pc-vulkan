// SPDX-License-Identifier: GPL-2.0-or-later
// Native Vulkan 1.3 GLSL replacement for ps 0x00000000d3c8bb21
// (Yebis2 Motion Blur camera + object velocity combine pass, half-res R16G16B16A16Sfloat).
// Eliminates software BaryCoordKHR interpolation and vectorizes matrix reprojection.
#version 460

layout(constant_id = 0) const uint kUnnormObjUv = 0u;

layout(location = 0) in vec4 fs_in_attr0; // depth UV
layout(location = 1) in vec4 fs_in_attr1; // NDC xy
layout(location = 2) in vec4 fs_in_attr2; // object velocity UV
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

layout(set = 0, binding = 0) readonly buffer SSBO1 {
    uint data[];
} ssbo_1;

layout(set = 0, binding = 1) uniform texture2D fs_img0;
layout(set = 0, binding = 2) uniform texture2D fs_img1;
layout(set = 0, binding = 3) uniform sampler fs_samp0;
layout(set = 0, binding = 4) uniform sampler fs_samp1;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
vec4 s1v4(uint i) { return vec4(s1f(i), s1f(i + 1u), s1f(i + 2u), s1f(i + 3u)); }

void main() {
    const uint b0 = (push_data.buf_offsets0.x & 0xFFu) >> 2u;

    const float d_raw = textureLod(sampler2D(fs_img0, fs_samp0), fs_in_attr0.xy, 0.0).r;
    vec2 obj_uv = fs_in_attr2.xy;
    if (kUnnormObjUv != 0u) {
        obj_uv /= vec2(textureSize(sampler2D(fs_img1, fs_samp1), 0));
    }
    const vec4 obj = textureLod(sampler2D(fs_img1, fs_samp1), obj_uv, 0.0);

    const float d_scaled = fma(s1f(b0 + 0u), d_raw, s1f(b0 + 1u));
    const float d_ndc = clamp(fma(s1f(b0 + 620u), d_scaled, s1f(b0 + 621u)), 0.0, 1.0);
    const float w_clip = (-s1f(b0 + 62u)) / fma(d_ndc, s1f(b0 + 63u), -s1f(b0 + 61u));

    const vec2 ndc = fs_in_attr1.xy;
    const vec4 p = vec4(ndc.x * w_clip, ndc.y * w_clip, d_ndc * w_clip, w_clip);

    const float prev_x = dot(s1v4(b0 + 896u), p);
    const float prev_y = dot(s1v4(b0 + 900u), p);
    const float prev_z = dot(s1v4(b0 + 904u), p);
    const float prev_w = max(0.000976559997, dot(s1v4(b0 + 908u), p));
    const float inv_pw = 1.0 / prev_w;

    vec2 cam_vel = vec2(prev_x, prev_y) * inv_pw - ndc;
    const float cam_len = length(cam_vel);
    if (cam_len < 0.0004) {
        cam_vel = vec2(0.0);
    } else {
        cam_vel *= (cam_len - 0.0004) / cam_len;
    }

    // Independent Camera (environment: obj.a == 0) and Object (Hunter/weapons/enemies: obj.a == 1)
    // motion blur weights packed in push_data.buf_offsets[38] (bits 16..23 of buf_offsets2.y).
    //
    // Verified in eboot.bin (0x136d9c9..0x136da01):
    //   s1[624..625] = 1.0 / s1[616..617]  and  s1[626..627] = (2.0, -1.0).
    // Therefore:
    //   - Camera pixels (obj.a == 0): v_scaled = cam_vel * s1[616..617], where s1[616..617]
    //     already contains fps_scale * max(cam_m, obj_m) / max_len_m from Yebis2; scaling
    //     cam_vel by cam_m / max_m yields exact independent Camera strength.
    //   - Object pixels (obj.a == 1): s1[624..625] * s1[616..617] === 1.0, so Yebis2's
    //     blurTimeRatio (fps_scale * max_m) cancels out completely on object pixels!
    //     Instead, Pass 0x2e (ps_34bc187c / ps_749e4f9e / ps_b25e4fae) already pre-scales
    //     (cur_ndc - prev_ndc) * vel_scale by obj_fps_scale into full 30-FPS 8-bit UNORM
    //     range, and ps_e0305cef_post multiplies v_scaled by maxBlurLength * max_len_m.
    //     Scaling obj_vel by obj_m / max_len_m (without dividing by obj_fps_scale) yields
    //     exact FPS-invariant 30-FPS PS4 Object Motion Blur (0.0 / 0.45 / 1.00 / 1.75) that
    //     is 100% independent of Camera Motion Blur.
    const uint mb_packed = (push_data.buf_offsets2.y >> 16u) & 0xFFu;
    const uint cam_lvl = mb_packed & 3u;
    const uint obj_lvl = (mb_packed >> 2u) & 3u;
    const float kMbMult[4]  = float[4](0.0, 0.45, 1.00, 1.75);
    const float kLenMult[4] = float[4](0.0, 0.65, 1.00, 1.60);
    const float cam_m = kMbMult[cam_lvl];
    const float obj_m = kMbMult[obj_lvl];
    const float max_m = max(cam_m, obj_m);
    const float max_len_m = max(kLenMult[cam_lvl], kLenMult[obj_lvl]);
    const float cam_rel   = (max_m > 0.0)     ? (cam_m / max_m)     : 0.0;
    const float obj_scale = (max_len_m > 0.0) ? (obj_m / max_len_m) : 0.0;

    cam_vel *= cam_rel;

    vec2 obj_raw = fma(obj.xy, vec2(s1f(b0 + 626u)), vec2(s1f(b0 + 627u)));
    // Suppress 8-bit UNORM 0.5 quantization floor (128/255 - 0.5 = +0.00392/axis -> 0.00555 mag).
    const float obj_raw_len = length(obj_raw);
    if (obj_raw_len < 0.0065) {
        obj_raw = vec2(0.0);
    } else {
        obj_raw *= (obj_raw_len - 0.0065) / obj_raw_len;
    }
    const vec2 obj_vel = (obj_raw * vec2(s1f(b0 + 624u), s1f(b0 + 625u))) * obj_scale;
    const float blend = clamp(obj.a, 0.0, 1.0);
    const vec2 vel = mix(cam_vel, obj_vel, blend);

    vec2 v_scaled = vel * vec2(s1f(b0 + 616u), s1f(b0 + 617u));
    const float len_sq = dot(v_scaled, v_scaled);
    if (len_sq > 1.0) {
        v_scaled *= inversesqrt(len_sq);
    }

    const vec2 enc = fma(v_scaled, vec2(s1f(b0 + 618u)), vec2(s1f(b0 + 619u)));
    frag_color0 = vec4(enc.x, enc.y, 0.0, prev_z * inv_pw);
}
