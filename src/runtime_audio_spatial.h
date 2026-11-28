/* SPDX-License-Identifier: GPL-2.0-or-later
 * Native Windows Spatial Audio (ISpatialAudioClient: Dolby Atmos for Headphones, DTS Headphone:X,
 * Windows Sonic) static 7.1 bed output for Surround (7.1) mode. No-op stub on Linux.
 */
#ifndef BB_RUNTIME_AUDIO_SPATIAL_H
#define BB_RUNTIME_AUDIO_SPATIAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 if Windows Spatial Sound is enabled on the default device and the 7.1 bed opened. */
int runtime_audio_spatial_open(void);

/* Writes `frames` of interleaved SDL/WAVE-order 7.1 float32 PCM (FL FR FC LFE BL BR SL SR).
 * Returns 0 if Windows invalidated the output device so caller can fall back to SDL3. */
int runtime_audio_spatial_write(const float *frames_sdl71, uint32_t frames);

void runtime_audio_spatial_close(void);

#ifdef __cplusplus
}
#endif

#endif /* BB_RUNTIME_AUDIO_SPATIAL_H */
