/* SPDX-License-Identifier: GPL-2.0-or-later
 * Cross-platform 7.1 surround downmix & Diffuse-Field Equalized MIT KEMAR Binaural HRTF
 * virtualizer for Bloodborne's 48 kHz 8-channel float audio output (Linux & Windows).
 */
#ifndef BB_RUNTIME_AUDIO_HRTF_H
#define BB_RUNTIME_AUDIO_HRTF_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum BbAudioMode {
    BB_AUDIO_MODE_STEREO = 0,   /* Clean unity-front 7.1 -> 2.0 fold (speakers / default) */
    BB_AUDIO_MODE_HRTF = 1,     /* Built-in 48 kHz MIT KEMAR 7.1 Binaural HRTF (headphones/IEMs) */
    BB_AUDIO_MODE_SURROUND = 2, /* Discrete 5.1/7.1 passthrough + Windows Spatial Sound (Atmos/DTS) */
    BB_AUDIO_MODE_COUNT = 3
};

#define BB_HRTF_CHUNK_MAX 256
#define BB_HRTF_PAD 128

typedef struct BbAudioHrtfState {
    /* Per-channel contiguous history + current chunk (PS4 order: FL FR FC LFE SL SR BL BR). */
    float hist[8][BB_HRTF_PAD + BB_HRTF_CHUNK_MAX];
    /* Pre-interpolated 65-tap fractional-ITD kernels & integer delays (L1-resident). */
    float k_ir[7][65];
    int d_int[7];
    /* 2nd-order RBJ low-shelf biquad state (Direct Form II Transposed, double precision). */
    double shelf_z1[2];
    double shelf_z2[2];
    /* Smooth 21.3 ms (1024-sample) clickless blend between Stereo (0.0) and HRTF (1.0). */
    float blend;
    int initialized;
} BbAudioHrtfState;

void bb_audio_hrtf_init(BbAudioHrtfState *st, int start_hrtf);
void bb_audio_hrtf_reset(BbAudioHrtfState *st, int start_hrtf);

/* Transparent C1 soft-knee peak limiter: linear for |x| <= 0.85, smooth tanh ceiling at 0.99. */
float bb_audio_soft_limit(float x);

/* Unity-front 7.1 (PS4 order: FL FR FC LFE SL SR BL BR) -> N-channel (1..7) fold with limiter. */
void bb_audio_fold8(const float in_ps4[8], float *out, int out_channels);

/* Process `frames` of interleaved PS4 7.1 float PCM (FL FR FC LFE SL SR BL BR) into interleaved
 * stereo float PCM (L R). `use_hrtf`: 0 = Stereo fold, 1 = 7.1 Binaural HRTF (crossfades live). */
void bb_audio_hrtf_process(BbAudioHrtfState *st, const float *in_ps4, float *out_stereo,
                           size_t frames, int use_hrtf);

#ifdef __cplusplus
}
#endif

#endif /* BB_RUNTIME_AUDIO_HRTF_H */
