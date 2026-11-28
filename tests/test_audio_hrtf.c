/* SPDX-License-Identifier: GPL-2.0-or-later
 * Acoustic & DSP verification suite for 7.1 MIT KEMAR Binaural HRTF and unity stereo fold8.
 */
#include "runtime_audio_hrtf.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SR 48000.0
#define PI 3.14159265358979323846

static void measure_tone(int ch, double hz, double *out_rms_l, double *out_rms_r,
                         double *out_lag_samples) {
    BbAudioHrtfState st;
    bb_audio_hrtf_init(&st, 1);
    const size_t total = 8192;
    const size_t settle = 2048;
    float in[8192 * 8];
    float out[8192 * 2];
    memset(in, 0, sizeof(in));
    for (size_t f = 0; f < total; ++f) {
        in[f * 8u + (size_t)ch] = 0.25f * (float)sin(2.0 * PI * hz * (double)f / SR);
    }
    /* Feed in Bloodborne's real 256-frame periods to verify chunk boundary continuity. */
    for (size_t off = 0; off < total; off += 256) {
        bb_audio_hrtf_process(&st, in + off * 8u, out + off * 2u, 256, 1);
    }
    double sum_l = 0.0, sum_r = 0.0;
    for (size_t f = settle; f < total; ++f) {
        double l = out[f * 2u], r = out[f * 2u + 1u];
        assert(isfinite(l) && isfinite(r));
        sum_l += l * l;
        sum_r += r * r;
    }
    double n = (double)(total - settle);
    *out_rms_l = sqrt(sum_l / n) / (0.25 / sqrt(2.0));
    *out_rms_r = sqrt(sum_r / n) / (0.25 / sqrt(2.0));
    (void)out_lag_samples;
}

/* Measure broadband Interaural Time Delay (R arrival minus L arrival, samples) via unit impulse. */
static double measure_itd_impulse(int ch) {
    BbAudioHrtfState st;
    bb_audio_hrtf_init(&st, 1);
    float in[512 * 8], out[512 * 2];
    memset(in, 0, sizeof(in));
    in[64 * 8 + ch] = 0.5f;
    bb_audio_hrtf_process(&st, in, out, 256, 1);
    bb_audio_hrtf_process(&st, in + 256 * 8, out + 256 * 2, 256, 1);
    double best_cc = -1e30;
    int best_lag = 0;
    for (int lag = -50; lag <= 50; ++lag) {
        double cc = 0.0;
        for (int f = 60; f < 450; ++f) {
            cc += (double)out[f * 2] * (double)out[(f + lag) * 2 + 1];
        }
        if (cc > best_cc) {
            best_cc = cc;
            best_lag = lag;
        }
    }
    return (double)best_lag;
}

int main(void) {
    /* 1. Soft-knee limiter linearity & ceiling. */
    assert(bb_audio_soft_limit(0.5f) == 0.5f);
    assert(bb_audio_soft_limit(-0.85f) == -0.85f);
    assert(bb_audio_soft_limit(10.0f) > 0.98f && bb_audio_soft_limit(10.0f) < 0.995f);
    assert(bb_audio_soft_limit(-10.0f) < -0.98f && bb_audio_soft_limit(-10.0f) > -0.995f);

    /* 2. Stereo unity fold8 loudness check (no -13 dB SDL attenuation). */
    {
        float in_fl[8] = {0.5f, 0, 0, 0, 0, 0, 0, 0}, out[2] = {0};
        bb_audio_fold8(in_fl, out, 2);
        assert(fabsf(out[0] - 0.5f) < 1e-6f && fabsf(out[1]) < 1e-6f);
        float in_fc[8] = {0, 0, 0.5f, 0, 0, 0, 0, 0};
        bb_audio_fold8(in_fc, out, 2);
        assert(fabsf(out[0] - 0.5f * 0.70710678f) < 1e-5f && fabsf(out[0] - out[1]) < 1e-6f);
    }

    /* 3. Center channel (FC) exact L==R symmetry + 7-channel bass flatness (50 Hz vs 1 kHz). */
    double fc_l_1k, fc_r_1k, fc_l_50, fc_r_50;
    measure_tone(2, 1000.0, &fc_l_1k, &fc_r_1k, NULL);
    measure_tone(2, 50.0, &fc_l_50, &fc_r_50, NULL);
    assert(fabs(fc_l_1k - fc_r_1k) < 1e-5);
    assert(fabs(fc_l_50 - fc_r_50) < 1e-5);
    double p50 = fc_l_50 * fc_l_50 + fc_r_50 * fc_r_50;
    double p1k = fc_l_1k * fc_l_1k + fc_r_1k * fc_r_1k;
    static const int pair_ch[3] = {0, 4, 6};
    for (int i = 0; i < 3; ++i) {
        double l50, r50, l1k, r1k;
        measure_tone(pair_ch[i], 50.0, &l50, &r50, NULL);
        measure_tone(pair_ch[i], 1000.0, &l1k, &r1k, NULL);
        p50 += 2.0 * (l50 * l50 + r50 * r50);
        p1k += 2.0 * (l1k * l1k + r1k * r1k);
    }
    double bass_ratio_db = 10.0 * log10(p50 / p1k);
    assert(fabs(bass_ratio_db) < 1.0); /* 7-ch average 50 Hz matches 1 kHz within +/-1.0 dB */

    /* 4. Exact L/R mirror symmetry & directional ITD + ILD for FL/FR, SL/SR, BL/BR. */
    double fl_l, fl_r, fr_l, fr_r, sl_l, sl_r, sr_l, sr_r, bl_l, bl_r, br_l, br_r;
    double fl_lag = measure_itd_impulse(0);
    double sl_lag = measure_itd_impulse(4);
    double bl_lag = measure_itd_impulse(6);
    measure_tone(0, 1500.0, &fl_l, &fl_r, NULL);
    measure_tone(1, 1500.0, &fr_l, &fr_r, NULL);
    assert(fl_l > fl_r * 1.5 && fl_lag >= 10.0 && fl_lag <= 15.0);
    assert(fabs(fl_l - fr_r) < 1e-5 && fabs(fl_r - fr_l) < 1e-5);

    measure_tone(4, 1500.0, &sl_l, &sl_r, NULL);
    measure_tone(5, 1500.0, &sr_l, &sr_r, NULL);
    assert(sl_l > sl_r * 1.7 && sl_lag >= 34.0 && sl_lag <= 43.0);
    assert(fabs(sl_l - sr_r) < 1e-5 && fabs(sl_r - sr_l) < 1e-5);
    {
        double sl_4k_l, sl_4k_r;
        measure_tone(4, 4000.0, &sl_4k_l, &sl_4k_r, NULL);
        assert(sl_4k_l > sl_4k_r * 2.0); /* stronger high-frequency head shadowing at 4 kHz */
    }

    measure_tone(6, 1500.0, &bl_l, &bl_r, NULL);
    measure_tone(7, 1500.0, &br_l, &br_r, NULL);
    assert(bl_l > bl_r * 1.5 && bl_lag >= 10.0 && bl_lag <= 16.0);
    assert(fabs(bl_l - br_r) < 1e-5 && fabs(bl_r - br_l) < 1e-5);

    /* 5. Front-vs-Back pinna spectral contrast at 6 kHz (FL vs BL ipsilateral ear). */
    double fl_6k_l, fl_6k_r, bl_6k_l, bl_6k_r;
    measure_tone(0, 6000.0, &fl_6k_l, &fl_6k_r, NULL);
    measure_tone(6, 6000.0, &bl_6k_l, &bl_6k_r, NULL);
    assert(fl_6k_l > bl_6k_l * 2.0); /* authentic KEMAR pinna cue distinguishes front vs behind */

    /* 6. LFE bypasses low-shelf (+0.5 exact unity, equal L/R, zero +10.5 dB bloat). */
    double lfe_l, lfe_r;
    measure_tone(3, 60.0, &lfe_l, &lfe_r, NULL);
    assert(fabs(lfe_l - 0.5) < 1e-3 && fabs(lfe_r - 0.5) < 1e-3);

    /* 7. Clickless live crossfade Stereo <-> HRTF on continuous 440 Hz tone. */
    {
        BbAudioHrtfState st;
        bb_audio_hrtf_init(&st, 0);
        float in[256 * 8], out[256 * 2];
        float prev_l = 0.0f, prev_r = 0.0f;
        float max_step = 0.0f;
        for (int blk = 0; blk < 32; ++blk) {
            int want_hrtf = (blk >= 8 && blk < 20) ? 1 : 0;
            for (int f = 0; f < 256; ++f) {
                memset(&in[f * 8], 0, 8 * sizeof(float));
                in[f * 8 + 0] = 0.35f * (float)sin(2.0 * PI * 440.0 * (double)(blk * 256 + f) / SR);
            }
            bb_audio_hrtf_process(&st, in, out, 256, want_hrtf);
            for (int f = 0; f < 256; ++f) {
                if (blk > 0 || f > 0) {
                    float dl = fabsf(out[f * 2] - prev_l);
                    float dr = fabsf(out[f * 2 + 1] - prev_r);
                    if (dl > max_step) max_step = dl;
                    if (dr > max_step) max_step = dr;
                }
                prev_l = out[f * 2];
                prev_r = out[f * 2 + 1];
            }
        }
        assert(max_step < 0.05f); /* zero click across live Stereo <-> HRTF toggles */
        printf("PASS: 7.1 HRTF acoustic test (FC 50Hz/1kHz=%+.2f dB, ITD FL=%.0f SL=%.0f BL=%.0f samp, "
               "6kHz F/B=%.1fx, LFE=%.3f, xfade max_step=%.4f)\n",
               bass_ratio_db, fl_lag, sl_lag, bl_lag, fl_6k_l / bl_6k_l, lfe_l, max_step);
    }
    return 0;
}
