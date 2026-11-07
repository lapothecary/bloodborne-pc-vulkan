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

// VFX Dual-Texture Blend + Fog 2-MRT — ps_44a82ffb
layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 2) in vec4 fs_in_attr2_p;
layout(location = 3) in vec4 fs_in_attr3_p;

layout(location = 0) out vec4 frag_color0;
layout(location = 1) out vec4 frag_color1;

layout(set = 0, binding = 0, std430) readonly buffer ssbo_1_buf { uint data[]; } ssbo_1;
layout(set = 0, binding = 1, std430) readonly buffer ssbo_2_buf { uint data[]; } ssbo_2;

layout(set = 0, binding = 2) uniform texture2D fs_img0;
layout(set = 0, binding = 3) uniform sampler fs_samp0;

float s1f(uint i) { return uintBitsToFloat(ssbo_1.data[i]); }
float s2f(uint i) { return uintBitsToFloat(ssbo_2.data[i]); }

void main() {
    uint u0 = push_data.buf_offsets0.x;
    uint d0 = bitfieldExtract(u0, 0, 8) >> 2u;
    uint d1 = bitfieldExtract(u0, 8, 8) >> 2u;

    vec4 tex0 = texture(sampler2D(fs_img0, fs_samp0), fs_in_attr2_p.xy);
    vec4 tex1 = texture(sampler2D(fs_img0, fs_samp0), fs_in_attr3_p.xy);
    vec3 sq0 = tex0.xyz * tex0.xyz;
    vec3 sq1 = tex1.xyz * tex1.xyz;
    vec3 tex_rgb = fs_in_attr3_p.z * (-sq0 + sq1) + sq0;
    float tex_a = fs_in_attr3_p.z * (tex1.w - tex0.w) + tex0.w;

    bool mode2 = (ssbo_1.data[d0 + 95u] == 2u);
    vec3 c_mod = vec3(s1f(d0 + 36u), s1f(d0 + 37u), s1f(d0 + 38u));
    vec3 v_mod = c_mod * fs_in_attr0_p.xyz;
    vec3 inv_v = -c_mod * fs_in_attr0_p.xyz + 1.0;
    vec3 ta = tex_rgb * tex_a;

    vec3 ov_rgb = vec3(
        (ta.x < 0.5) ? ((v_mod.x * ta.x) * 2.0) : (1.0 - 2.0 * (inv_v.x - tex_a * (tex_rgb.x * inv_v.x))),
        (ta.y < 0.5) ? ((v_mod.y * ta.y) * 2.0) : (1.0 - 2.0 * (inv_v.y - tex_a * (tex_rgb.y * inv_v.y))),
        (0.5 > ta.z) ? ((v_mod.z * ta.z) * 2.0) : (1.0 - 2.0 * (inv_v.z - tex_a * (tex_rgb.z * inv_v.z)))
    );
    vec3 mul_rgb = c_mod * (tex_rgb * fs_in_attr0_p.xyz);
    vec3 base_rgb = mode2 ? ov_rgb : mul_rgb;
    float alpha_raw = s1f(d0 + 39u) * (tex_a * fs_in_attr0_p.w);

    float luma_bias = -1.0 + (base_rgb.z * 0.114480004 + (base_rgb.x * 0.298909992 + 0.586610019 * base_rgb.y));
    float rad_mod = s2f(d1 + 7u) * luma_bias + 1.0;
    float hgt_mod = s2f(d1 + 19u) * luma_bias + 1.0;

    float dist_xz = sqrt(fs_in_attr1_p.x * fs_in_attr1_p.x + fs_in_attr1_p.z * fs_in_attr1_p.z);
    float fog_rad = clamp(s2f(d1 + 8u) + s2f(d1 + 8u) * (-exp2(-(s2f(d1 + 10u) * (dist_xz - s2f(d1 + 9u))))), 0.0, 1.0);
    vec3 rad_col = vec3(s2f(d1 + 4u), s2f(d1 + 5u), s2f(d1 + 6u)) * rad_mod;
    vec3 after_rad = fog_rad * (rad_col - base_rgb) + base_rgb;

    float fog_hgt = clamp(s2f(d1 + 12u) * clamp(s2f(d1 + 14u) * ((s2f(d1 + 15u) - fs_in_attr1_p.y) - s2f(d1 + 13u)), 0.0, 1.0), 0.0, 1.0);
    vec3 hgt_col = vec3(s2f(d1 + 16u), s2f(d1 + 17u), s2f(d1 + 18u)) * hgt_mod;
    vec3 col_scale = vec3(s1f(d0 + 43u), s1f(d0 + 47u), s1f(d0 + 51u));
    vec3 fogged_rgb = col_scale * (fog_hgt * (hgt_col - after_rad) + after_rad);

    vec3 final_rgb = max(vec3(0.0), (0.0 < alpha_raw) ? fogged_rgb : base_rgb);
    float final_a = max(0.0, alpha_raw);

    vec4 out0 = vec4(final_rgb, final_a);
    vec4 mrt1_scale = vec4(s1f(d0 + 88u), s1f(d0 + 89u), s1f(d0 + 90u), s1f(d0 + 91u));
    frag_color1 = mrt1_scale * out0;
    frag_color0 = out0;
}
