// SPDX-License-Identifier: GPL-2.0-or-later
// Step 2.3: Native Vulkan 1.3 GLSL 460 replacement for ps 0x000000004baeaa00
// (10a. Scaleform HUD Premultiplied Color-Transform BGRA Output Pass).
#version 460

layout(location = 0) in vec4 fs_in_attr0_p;
layout(location = 1) in vec4 fs_in_attr1_p;
layout(location = 2) in vec4 fs_in_attr2_p;
layout(location = 3) in vec4 fs_in_attr3_p;

layout(location = 0) out vec4 frag_color0;

void main() {
    const vec4 c = fma(fs_in_attr0_p, fs_in_attr3_p, fs_in_attr2_p);
    const float a = c.a * fs_in_attr1_p.a;
    frag_color0 = vec4(c.bgr * a, a);
}
