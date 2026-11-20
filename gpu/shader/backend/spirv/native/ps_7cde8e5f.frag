// SPDX-License-Identifier: GPL-2.0-or-later
// Native Vulkan 1.3 GLSL replacement for ps 0x000000007cde8e5f
// (4096x4096 CSM/VSM shadow depth pass: packed RGB8 depth, depth-only, and clip-distance VSM).
// Eliminates software BaryCoordKHR interpolation across all 5 permutations (_0.._4).
#version 460

#if defined(PACK_DEPTH)
layout(location = 0) in vec4 fs_in_attr0;
layout(location = 0) out vec4 frag_color0;

void main() {
    const float d = fs_in_attr0.x / fs_in_attr0.y;
    const float r = fract(d * 65025.0);
    const float g = fract(d * 255.0);
    const float kInv255 = 0.00392156886;
    frag_color0 = vec4(r, fma(r, -kInv255, g), fma(g, -kInv255, d), 0.0);
}
#elif defined(HAS_CLDIST)
#extension GL_EXT_demote_to_helper_invocation : require
layout(location = 0) in vec4 cldist_attr0;

void main() {
    if (any(lessThan(cldist_attr0, vec4(0.0)))) {
        demote;
    }
}
#else
void main() {}
#endif
