/* SPDX-License-Identifier: GPL-2.0-or-later
 * Cross-platform 48 kHz 7.1 Binaural HRTF (Diffuse-Field Equalized MIT KEMAR) & unity stereo
 * downmix engine. Zero external libraries; bit-identical on Linux (PipeWire/ALSA) & Windows.
 */
#include "runtime_audio_hrtf.h"
#include "runtime_audio_hrtf_data.h"

#include <math.h>
#include <string.h>

float bb_audio_soft_limit(float x) {
    const float knee = 0.85f;
    const float a = fabsf(x);
    if (a <= knee) {
        return x;
    }
    const float y = knee + 0.14f * tanhf((a - knee) * (1.0f / 0.14f));
    return x < 0.0f ? -y : y;
}

void bb_audio_fold8(const float in_ps4[8], float *out, int out_channels) {
    const float L = in_ps4[0];
    const float R = in_ps4[1];
    const float C = in_ps4[2];
    const float LFE = in_ps4[3];
    const float SL = in_ps4[4];
    const float SR = in_ps4[5];
    const float BL = in_ps4[6];
    const float BR = in_ps4[7];
    const float k = 0.70710678f;

    switch (out_channels) {
    case 1:
        out[0] = 0.5f * ((L + R) + 2.0f * k * C + LFE + k * (SL + SR + BL + BR));
        break;
    case 2:
        out[0] = L + k * C + 0.5f * LFE + k * SL + k * BL;
        out[1] = R + k * C + 0.5f * LFE + k * SR + k * BR;
        break;
    case 3: /* 2.1 */
        out[0] = L + k * C + k * SL + k * BL;
        out[1] = R + k * C + k * SR + k * BR;
        out[2] = LFE;
        break;
    case 4: /* 4.0 */
        out[0] = L + k * C + 0.5f * LFE + k * SL;
        out[1] = R + k * C + 0.5f * LFE + k * SR;
        out[2] = BL;
        out[3] = BR;
        break;
    case 5: /* 4.1 */
        out[0] = L + k * C + k * SL;
        out[1] = R + k * C + k * SR;
        out[2] = LFE;
        out[3] = BL;
        out[4] = BR;
        break;
    case 6: /* 5.1 (SDL/WAVE order: L R C LFE BL BR) */
        out[0] = L;
        out[1] = R;
        out[2] = C;
        out[3] = LFE;
        out[4] = BL + k * SL;
        out[5] = BR + k * SR;
        break;
    case 7: /* 6.1 */
        out[0] = L;
        out[1] = R;
        out[2] = C;
        out[3] = LFE;
        out[4] = k * (BL + BR);
        out[5] = SL;
        out[6] = SR;
        break;
    default:
        return;
    }
    for (int i = 0; i < out_channels; ++i) {
        out[i] = bb_audio_soft_limit(out[i]);
    }
}

void bb_audio_hrtf_init(BbAudioHrtfState *st, int start_hrtf) {
    memset(st, 0, sizeof(*st));
    for (int p = 0; p < BB_HRTF_PROFILES; ++p) {
        const float d = bb_hrtf_delay[p];
        const int di = (int)floorf(d);
        const float frac = d - (float)di;
        const float omf = 1.0f - frac;
        st->d_int[p] = di;
        st->k_ir[p][0] = omf * bb_hrtf_ir[p][0];
        for (int k = 1; k < BB_HRTF_TAPS; ++k) {
            st->k_ir[p][k] = omf * bb_hrtf_ir[p][k] + frac * bb_hrtf_ir[p][k - 1];
        }
        st->k_ir[p][BB_HRTF_TAPS] = frac * bb_hrtf_ir[p][BB_HRTF_TAPS - 1];
    }
    st->blend = start_hrtf ? 1.0f : 0.0f;
    st->initialized = 1;
}

void bb_audio_hrtf_reset(BbAudioHrtfState *st, int start_hrtf) {
    if (!st->initialized) {
        bb_audio_hrtf_init(st, start_hrtf);
        return;
    }
    memset(st->hist, 0, sizeof(st->hist));
    st->shelf_z1[0] = st->shelf_z1[1] = 0.0;
    st->shelf_z2[0] = st->shelf_z2[1] = 0.0;
    st->blend = start_hrtf ? 1.0f : 0.0f;
}

static inline float convolve65(const float *x_newest, const float *ir65) {
    float acc = 0.0f;
    for (int k = 0; k < 65; ++k) {
        acc += x_newest[-k] * ir65[k];
    }
    return acc;
}

static inline float apply_lowshelf(double *z1, double *z2, float in) {
    const double x = (double)in;
    const double b0 = (double)bb_hrtf_shelf_b0;
    const double b1 = (double)bb_hrtf_shelf_b1;
    const double b2 = (double)bb_hrtf_shelf_b2;
    const double a1 = (double)bb_hrtf_shelf_a1;
    const double a2 = (double)bb_hrtf_shelf_a2;
    const double y = b0 * x + *z1;
    double nz1 = b1 * x - a1 * y + *z2;
    double nz2 = b2 * x - a2 * y;
    /* Prevent x86 subnormal stalls during extended silence. */
    if (fabs(nz1) < 1e-24) nz1 = 0.0;
    if (fabs(nz2) < 1e-24) nz2 = 0.0;
    *z1 = nz1;
    *z2 = nz2;
    return (float)y;
}

void bb_audio_hrtf_process(BbAudioHrtfState *st, const float *in_ps4, float *out_stereo,
                           size_t frames, int use_hrtf) {
    if (!st->initialized) {
        bb_audio_hrtf_init(st, use_hrtf);
    }
    const float target_blend = use_hrtf ? 1.0f : 0.0f;
    const float blend_step = 1.0f / 1024.0f; /* 21.3 ms smooth clickless crossfade @ 48 kHz */
    const float k = 0.70710678f;

    /* Profile index table for PS4 channel order:
     *   0: FL (-30 deg) -> L=1(ipsi), R=2(cntr)
     *   1: FR (+30 deg) -> L=2(cntr), R=1(ipsi)
     *   2: FC (  0 deg) -> L=0,       R=0
     *   3: LFE          -> phase-aligned bypass (BB_HRTF_LFE_DELAY)
     *   4: SL (-90 deg) -> L=3(ipsi), R=4(cntr)
     *   5: SR (+90 deg) -> L=4(cntr), R=3(ipsi)
     *   6: BL (-135)    -> L=5(ipsi), R=6(cntr)
     *   7: BR (+135)    -> L=6(cntr), R=5(ipsi) */
    static const int prof_L[8] = {1, 2, 0, -1, 3, 4, 5, 6};
    static const int prof_R[8] = {2, 1, 0, -1, 4, 3, 6, 5};

    size_t done = 0;
    while (done < frames) {
        size_t n = frames - done;
        if (n > BB_HRTF_CHUNK_MAX) {
            n = BB_HRTF_CHUNK_MAX;
        }
        const float *src = in_ps4 + done * 8u;
        float *dst = out_stereo + done * 2u;

        /* De-interleave incoming chunk right after the BB_HRTF_PAD history prefix. */
        for (size_t f = 0; f < n; ++f) {
            const float *s = src + f * 8u;
            const size_t idx = BB_HRTF_PAD + f;
            st->hist[0][idx] = s[0];
            st->hist[1][idx] = s[1];
            st->hist[2][idx] = s[2];
            st->hist[3][idx] = s[3];
            st->hist[4][idx] = s[4];
            st->hist[5][idx] = s[5];
            st->hist[6][idx] = s[6];
            st->hist[7][idx] = s[7];
        }

        const int need_hrtf = (use_hrtf != 0) || (st->blend > 0.0f);
        const int need_stereo = (use_hrtf == 0) || (st->blend < 1.0f);

        for (size_t f = 0; f < n; ++f) {
            const size_t idx = BB_HRTF_PAD + f;
            float sL = 0.0f, sR = 0.0f;
            float hL = 0.0f, hR = 0.0f;

            if (need_stereo) {
                const float *s = src + f * 8u;
                sL = s[0] + k * s[2] + 0.5f * s[3] + k * s[4] + k * s[6];
                sR = s[1] + k * s[2] + 0.5f * s[3] + k * s[5] + k * s[7];
            }

            if (need_hrtf) {
                /* Pure textbook 100% Diffuse-Field Equalized MIT KEMAR 7.1 HRTF (OpenAL Soft
                 * MinPHR03 + 2nd-order RBJ low-shelf): zero custom coloration or phase-blend
                 * comb filtering, full 12-16 kHz bandwidth, universal across all headphones. */
                float rawL = 0.0f, rawR = 0.0f;
                for (int c = 0; c < 8; ++c) {
                    if (c == 3) {
                        continue; /* LFE bypasses HRIR & low-shelf */
                    }
                    const int pL = prof_L[c];
                    const int pR = prof_R[c];
                    const float *ch = st->hist[c];
                    rawL += convolve65(ch + idx - (size_t)st->d_int[pL], st->k_ir[pL]);
                    rawR += convolve65(ch + idx - (size_t)st->d_int[pR], st->k_ir[pR]);
                }
                const float lfe = 0.5f * st->hist[3][idx - BB_HRTF_LFE_DELAY];
                hL = apply_lowshelf(&st->shelf_z1[0], &st->shelf_z2[0], rawL) + lfe;
                hR = apply_lowshelf(&st->shelf_z1[1], &st->shelf_z2[1], rawR) + lfe;
            } else {
                /* Keep shelf state zeroed while pure Stereo is steady so re-entry has zero transient. */
                st->shelf_z1[0] = st->shelf_z1[1] = 0.0;
                st->shelf_z2[0] = st->shelf_z2[1] = 0.0;
            }

            float b = st->blend;
            if (b < target_blend) {
                b += blend_step;
                if (b > target_blend) b = target_blend;
                st->blend = b;
            } else if (b > target_blend) {
                b -= blend_step;
                if (b < target_blend) b = target_blend;
                st->blend = b;
            }

            const float outL = (1.0f - b) * sL + b * hL;
            const float outR = (1.0f - b) * sR + b * hR;
            dst[f * 2u + 0u] = bb_audio_soft_limit(outL);
            dst[f * 2u + 1u] = bb_audio_soft_limit(outR);
        }

        /* Preserve the last BB_HRTF_PAD samples at [0..BB_HRTF_PAD-1] for the next chunk. */
        for (int c = 0; c < 8; ++c) {
            memmove(st->hist[c], st->hist[c] + n, BB_HRTF_PAD * sizeof(float));
        }

        done += n;
    }
}
