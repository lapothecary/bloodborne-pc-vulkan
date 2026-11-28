// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* libSceAudioOut on SDL3 audio streams with cross-platform 7.1 Binaural HRTF (Diffuse-Field
 * Equalized MIT KEMAR @ 48 kHz), unity-front stereo downmix (fixing SDL3's -13 dB 7.1->2.0
 * matrix), and discrete 5.1/7.1 + Windows Spatial Sound (ISpatialAudioClient) passthrough.
 *
 * sceAudioOutOutput returns once per buffer period on a steady clock, like PS4 hardware:
 * FMOD's output thread reads 256-frame slots out of its mixer's 512-frame ring, and returning in
 * bursts lets it overtake the mixer and click every 10.7 ms. Without an audio device (headless
 * runs, BB_AUDIO=none) ports only follow the clock. */
#define _GNU_SOURCE
#include "runtime.h"
#include "runtime_audio_hrtf.h"
#include "runtime_audio_spatial.h"
#include "../gpu/bbgpu.h"
#include "../gpu/shim/bb_env.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <SDL3/SDL.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#define PORTS 25
#define ERR_NOT_OPENED ((int32_t)0x80260001)
#define ERR_INVALID_PORT ((int32_t)0x80260003)
#define ERR_INVALID_POINTER ((int32_t)0x80260004)
#define ERR_PORT_FULL ((int32_t)0x80260005)
#define ERR_INVALID_SIZE ((int32_t)0x80260006)
#define ERR_INVALID_FORMAT ((int32_t)0x80260007)
#define ERR_INVALID_FREQ ((int32_t)0x80260008)
#define ERR_INVALID_VOLUME ((int32_t)0x80260009)
#define ERR_INVALID_TYPE ((int32_t)0x8026000A)
#define ERR_ALREADY_INIT ((int32_t)0x8026000E)
#define ERR_NOT_INIT ((int32_t)0x8026000F)
#define VOLUME_0DB 32768

typedef struct {
    int used, type, channels, out_channels, is_float, frames, sample_bytes, std_layout;
    int32_t volume[8];
    SDL_AudioStream *stream;
    int spatial;             /* Windows ISpatialAudioClient 7.1 bed active instead of `stream` */
    int active_mode;         /* Last applied BbAudioMode on this port */
    int primed, drop_warned;
    BbAudioHrtfState *hrtf;  /* Allocated for 8-ch Main ports */
    uint64_t next_deadline_ns;
    int64_t adjust_ns;
    int window_min, window_count;
    uint64_t last_output_us;
    pthread_mutex_t lock;
    FILE *dump;              /* BB_AUDIO_DUMP: final output PCM per port */
    FILE *guest_dump;        /* BB_AUDIO_DUMP_GUEST: raw PS4 guest PCM per port */
    int stats;
    uint64_t stat_start_ns, stat_last_ns, stat_max_gap_ns;
    unsigned stat_starved, stat_buffers;
    int stat_min_queued;
} Port;
typedef struct { uint16_t output; uint8_t channel, reserved; int16_t volume; uint16_t reroute; uint64_t flag, reserved64[2]; } PortState;
_Static_assert(sizeof(PortState)==32,"AudioOut port state layout");

static pthread_mutex_t table_lock=PTHREAD_MUTEX_INITIALIZER;
static Port ports[PORTS];
static int initialized, sdl_ready=-1;
static int device_channels, device_period_frames;
static float master_gain=1.0f;
static size_t buffers_out, ports_opened;

static const char *audio_mode_label(int mode) {
    switch (mode) {
    case BB_AUDIO_MODE_HRTF: return "Headphones (7.1 HRTF)";
    case BB_AUDIO_MODE_SURROUND: return "Surround (7.1)";
    default: return "Stereo";
    }
}

static int current_audio_mode(void) {
    int mode = bbgpu_audio_mode();
    if (mode < 0 || mode >= BB_AUDIO_MODE_COUNT) mode = BB_AUDIO_MODE_STEREO;
    return mode;
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}
static void sleep_until(uint64_t deadline) {
    struct timespec t = {(time_t)(deadline / 1000000000u), (long)(deadline % 1000000000u)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL)) {}
}

#ifdef _WIN32
static void raise_audio_thread(void) {
    static __thread int done;
    if (done) return;
    done = 1;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
}
#endif

static int sdl_audio(void) {
    if (sdl_ready < 0) {
        const char *mode = bb_env("BB_AUDIO");
        sdl_ready = (!mode || strcmp(mode, "none")) && SDL_InitSubSystem(SDL_INIT_AUDIO);
        printf("Runtime: audio backend %s\n", sdl_ready ? SDL_GetCurrentAudioDriver() : "timer (silent)");
    }
    return sdl_ready;
}

static void probe_audio_device(void) {
    if (device_channels) return;
    const char *env = bb_env("BB_AUDIO_GAIN");
    if (env && env[0]) {
        char *end = NULL;
        float g = strtof(env, &end);
        if (end != env && g > 0.0f && g <= 8.0f) master_gain = g;
    }
    SDL_AudioSpec spec;
    int frames = 0;
    if (SDL_GetAudioDeviceFormat(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, &frames) &&
        spec.channels >= 1 && spec.channels <= 8) {
        device_channels = spec.channels;
        if (frames > 0 && frames <= 16384 && spec.freq > 0) {
            device_period_frames = (int)((int64_t)frames * 48000 / spec.freq);
        }
    } else {
        device_channels = -1;
    }
    printf("Runtime: audio device %d ch, period %d frames (%.1f ms), mode %s, gain %.2f\n",
           device_channels > 0 ? device_channels : 0, device_period_frames,
           device_period_frames * 1000.0 / 48000.0, audio_mode_label(current_audio_mode()),
           master_gain);
}

static int queue_low_bytes(const Port *p) {
    int bpf = p->out_channels * p->sample_bytes;
    if (bpf < 1) bpf = 1;
    int frames = device_period_frames > 0 ? device_period_frames * 2 : 0;
#ifdef _WIN32
    /* WASAPI needs a slightly larger cushion on loaded CPUs to avoid underrun crackles. */
    if (device_period_frames > 0) frames = device_period_frames * 4;
    if (frames < 3840) frames = 3840;
#endif
    if (frames < p->frames * 2) frames = p->frames * 2;
    if (frames > 9600) frames = 9600;
    return frames * bpf;
}

static void put_silence(SDL_AudioStream *stream, int bytes) {
    static const unsigned char silence[4096];
    while (bytes > 0) {
        int n = bytes > (int)sizeof(silence) ? (int)sizeof(silence) : bytes;
        if (!SDL_PutAudioStreamData(stream, silence, n)) break;
        bytes -= n;
    }
}

static int target_out_channels(const Port *p, int mode) {
    if (p->channels != 8) return p->channels;
    if (mode == BB_AUDIO_MODE_SURROUND) {
        if (device_channels > 0 && device_channels < 8) return device_channels;
        return 8;
    }
    /* Stereo & Headphones (7.1 HRTF): open a native 2-ch stream (bypasses SDL's -13 dB matrix). */
    return (device_channels == 1) ? 1 : 2;
}

static void open_port_output(Port *p, int mode) {
    if (p->stream) {
        SDL_DestroyAudioStream(p->stream);
        p->stream = NULL;
    }
    if (p->spatial) {
        runtime_audio_spatial_close();
        p->spatial = 0;
    }
    p->primed = 0;
    p->drop_warned = 0;
    p->window_count = 0;
    p->adjust_ns = 0;
    p->active_mode = mode;

    if (!sdl_audio()) {
        p->out_channels = target_out_channels(p, mode);
        return;
    }
    probe_audio_device();

    /* On Windows in Surround (7.1) mode, route the 8-ch Main port to Windows Spatial Sound
     * (Dolby Atmos / DTS:X / Windows Sonic) if enabled on the playback device. */
    if (mode == BB_AUDIO_MODE_SURROUND && p->type == 0 && p->channels == 8 && p->is_float &&
        runtime_audio_spatial_open()) {
        p->spatial = 1;
        p->out_channels = 8;
        return;
    }

    int out = target_out_channels(p, mode);
    SDL_AudioSpec spec = {p->is_float ? SDL_AUDIO_F32 : SDL_AUDIO_S16, out, 48000};
    p->stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, NULL, NULL);
    if (p->stream) {
        p->out_channels = out;
        SDL_ResumeAudioStreamDevice(p->stream);
    } else {
        p->out_channels = out;
        fprintf(stderr, "Runtime: SDL audio stream failed (%s); port %d uses the timer sink\n",
                SDL_GetError(), (int)(p - ports));
    }
}

static float take_sample(const void *data, int is_float, size_t index, float gain) {
    if (is_float) {
        float v;
        memcpy(&v, (const char *)data + index * 4u, 4u);
        return v * gain;
    }
    int16_t s;
    memcpy(&s, (const char *)data + index * 2u, 2u);
    return ((float)s / 32768.0f) * gain;
}

static void put_sample(void *dst, int is_float, size_t index, float v) {
    if (is_float) {
        memcpy((char *)dst + index * 4u, &v, 4u);
        return;
    }
    if (v > 1.0f) v = 1.0f;
    else if (v < -1.0f) v = -1.0f;
    int16_t s = (int16_t)(v * 32767.0f);
    memcpy((char *)dst + index * 2u, &s, 2u);
}

static int port_range(int type, int *first, int *last) {
    switch (type) {
    case 0: *first=0; *last=7; return 1;      /* main */
    case 1: *first=8; *last=8; return 1;      /* bgm */
    case 2: *first=9; *last=12; return 1;     /* voice */
    case 3: *first=13; *last=16; return 1;    /* personal */
    case 4: *first=17; *last=20; return 1;    /* pad speaker */
    case 126: *first=21; *last=22; return 1;  /* audio3d */
    case 127: *first=23; *last=24; return 1;  /* aux */
    default: return 0;
    }
}
static Port *port_of(int32_t handle, int32_t *error) {
    int id=handle & 0xff;
    if ((handle & 0x3f000000)!=0x20000000 || id>=PORTS) { *error=ERR_INVALID_PORT; return NULL; }
    if (!ports[id].used) { *error=ERR_NOT_OPENED; return NULL; }
    return &ports[id];
}

static ABI int32_t audio_init(void) {
    pthread_mutex_lock(&table_lock);
    int32_t r=initialized ? ERR_ALREADY_INIT : 0;
    initialized=1;
    pthread_mutex_unlock(&table_lock);
    return r;
}
static ABI int32_t audio_open(int32_t user, int32_t type, int32_t index, uint32_t length, uint32_t freq, uint32_t param) {
    (void)user; (void)index;
    if (!initialized) return ERR_NOT_INIT;
    if (!length || length>2048 || (length & 0xff)) return ERR_INVALID_SIZE;
    if (freq!=48000) return ERR_INVALID_FREQ;
    uint32_t format=param & 0xff;
    if (format>7) return ERR_INVALID_FORMAT;
    int first, last;
    if (!port_range(type,&first,&last)) return ERR_INVALID_TYPE;
    static const int channels[8]={1,2,8,1,2,8,8,8};
    pthread_mutex_lock(&table_lock);
    int id=-1;
    for (int i=first;i<=last;++i) if (!ports[i].used) { id=i; break; }
    if (id<0) { pthread_mutex_unlock(&table_lock); return ERR_PORT_FULL; }
    Port *p=&ports[id];
    memset(p,0,sizeof(*p));
    pthread_mutex_init(&p->lock,NULL);
    p->used=1; p->type=type; p->channels=channels[format]; p->is_float=format>=3 && format!=6;
    p->sample_bytes=p->is_float ? 4 : 2; p->frames=(int)length; p->std_layout=format>=6;
    for (int c=0;c<8;++c) p->volume[c]=VOLUME_0DB;
    const int mode = current_audio_mode();
    if (p->channels == 8 && p->type == 0) {
        p->hrtf = (BbAudioHrtfState *)calloc(1, sizeof(BbAudioHrtfState));
        if (p->hrtf) bb_audio_hrtf_init(p->hrtf, mode == BB_AUDIO_MODE_HRTF);
    }
    open_port_output(p, mode);
    p->stats=bb_env("BB_AUDIO_STATS")!=NULL;
    const char *dump=bb_env("BB_AUDIO_DUMP");
    if (dump) {
        char path[4096];
        snprintf(path,sizeof(path),"%s.port%d.%dch.%s",dump,id,p->out_channels,p->is_float ? "f32" : "s16");
        p->dump=fopen(path,"wb");
    }
    const char *guest_dump=bb_env("BB_AUDIO_DUMP_GUEST");
    if (guest_dump) {
        char path[4096];
        snprintf(path,sizeof(path),"%s.port%d.%dch.%s.%s",guest_dump,id,p->channels,
                 p->std_layout ? "std" : "ps4",p->is_float ? "f32" : "s16");
        p->guest_dump=fopen(path,"wb");
    }
    ++ports_opened;
    pthread_mutex_unlock(&table_lock);
    printf("Runtime: audio port %d opened (type %d, %d ch -> %d ch, %s, %u frames, %s)\n",
           id, type, p->channels, p->out_channels, p->is_float ? "float" : "s16", length,
           p->spatial ? "Windows Spatial 7.1" : audio_mode_label(mode));
    return (type<<16) | id | 0x20000000;
}
static ABI int32_t audio_close(int32_t handle) {
    pthread_mutex_lock(&table_lock);
    int32_t error=0;
    Port *p=port_of(handle,&error);
    if (p) {
        pthread_mutex_lock(&p->lock);
        if (p->stream) SDL_DestroyAudioStream(p->stream);
        if (p->spatial) runtime_audio_spatial_close();
        p->spatial=0;
        if (p->hrtf) { free(p->hrtf); p->hrtf=NULL; }
        if (p->dump) fclose(p->dump);
        if (p->guest_dump) fclose(p->guest_dump);
        p->dump=NULL;
        p->guest_dump=NULL;
        p->stream=NULL; p->used=0;
        pthread_mutex_unlock(&p->lock);
    }
    pthread_mutex_unlock(&table_lock);
    return p ? 0 : error;
}
/* `pace`: wait for this port's next period. sceAudioOutOutputs waits once for all its ports. */
static int32_t output_port(int32_t handle, const void *data, int pace) {
    int32_t error=0;
    pthread_mutex_lock(&table_lock);
    Port *p=port_of(handle,&error);
    pthread_mutex_unlock(&table_lock);
    if (!p) return error;
#ifdef _WIN32
    raise_audio_thread();
#endif
    pthread_mutex_lock(&p->lock);
    size_t samples=(size_t)p->frames*(size_t)p->channels;
    uint64_t period=(uint64_t)p->frames*1000000000u/48000u;
    if (data) {
        if (p->guest_dump) fwrite(data, (size_t)p->sample_bytes, samples, p->guest_dump);

        /* Live audio mode check: Stereo <-> Headphones (7.1 HRTF) stays on the same 2-ch stream
         * with a 21.3 ms clickless crossfade; entering/leaving Surround (7.1) cleanly reopens. */
        const int mode = current_audio_mode();
        if (p->channels == 8 && mode != p->active_mode) {
            const int was_surround = (p->active_mode == BB_AUDIO_MODE_SURROUND);
            const int now_surround = (mode == BB_AUDIO_MODE_SURROUND);
            if (was_surround != now_surround) {
                open_port_output(p, mode);
                if (p->hrtf && !now_surround) {
                    bb_audio_hrtf_reset(p->hrtf, mode == BB_AUDIO_MODE_HRTF);
                }
            } else {
                p->active_mode = mode;
            }
            if (p->type == 0) {
                printf("Runtime: audio mode -> %s (%d ch -> %d ch%s)\n",
                       audio_mode_label(mode), p->channels, p->out_channels,
                       p->spatial ? ", Windows Spatial" : "");
            }
        }

        size_t out_bytes = (size_t)p->frames * (size_t)p->out_channels * (size_t)p->sample_bytes;
        unsigned char converted[2048 * 8 * 4];
        static const int from_std[8] = {0, 1, 2, 3, 6, 7, 4, 5}; /* SDL/WAVE -> PS4 (L R C LFE SL SR BL BR) */
        static const int to_std[8] = {0, 1, 2, 3, 6, 7, 4, 5};   /* PS4 -> SDL/WAVE (L R C LFE BL BR SL SR) */

        if (p->channels == 8 && p->out_channels == 2 && p->type == 0 && p->hrtf) {
            /* Main 8-ch port -> Stereo / 7.1 Binaural HRTF (with live clickless crossfade). */
            float in_ps4[2048 * 8];
            float out_st[2048 * 2];
            for (size_t f = 0; f < (size_t)p->frames; ++f) {
                for (int s = 0; s < 8; ++s) {
                    int guest = p->std_layout ? from_std[s] : s;
                    float gain = (float)p->volume[guest] / VOLUME_0DB * master_gain;
                    in_ps4[f * 8u + (size_t)s] =
                        take_sample(data, p->is_float, f * 8u + (size_t)guest, gain);
                }
            }
            bb_audio_hrtf_process(p->hrtf, in_ps4, out_st, (size_t)p->frames,
                                  mode == BB_AUDIO_MODE_HRTF);
            if (p->is_float) {
                memcpy(converted, out_st, out_bytes);
            } else {
                for (size_t i = 0; i < (size_t)p->frames * 2u; ++i) {
                    put_sample(converted, 0, i, out_st[i]);
                }
            }
        } else if (p->channels == 8 && p->out_channels < 8) {
            /* BGM 8-ch port or Surround on <8ch device -> unity-front downmix + soft limiter. */
            for (size_t f = 0; f < (size_t)p->frames; ++f) {
                float in_ps4[8] = {0}, out_ch[8] = {0};
                for (int s = 0; s < 8; ++s) {
                    int guest = p->std_layout ? from_std[s] : s;
                    float gain = (float)p->volume[guest] / VOLUME_0DB * master_gain;
                    in_ps4[s] = take_sample(data, p->is_float, f * 8u + (size_t)guest, gain);
                }
                bb_audio_fold8(in_ps4, out_ch, p->out_channels);
                for (int c = 0; c < p->out_channels; ++c) {
                    put_sample(converted, p->is_float, f * (size_t)p->out_channels + (size_t)c,
                               out_ch[c]);
                }
            }
        } else {
            /* Direct 1:1 passthrough (1ch, 2ch, or discrete 8ch Surround -> SDL/WAVE order). */
            for (size_t f = 0; f < (size_t)p->frames; ++f) {
                for (int c = 0; c < p->channels; ++c) {
                    int target = (p->channels == 8 && !p->std_layout) ? to_std[c] : c;
                    size_t from = f * (size_t)p->channels + (size_t)c;
                    size_t to = f * (size_t)p->channels + (size_t)target;
                    float gain = (float)p->volume[c] / VOLUME_0DB * master_gain;
                    float v = take_sample(data, p->is_float, from, gain);
                    put_sample(converted, p->is_float, to, v);
                }
            }
        }

        if (p->dump) fwrite(converted, 1, out_bytes, p->dump);
        if (p->stats) {
            uint64_t now = now_ns();
            if (!p->stat_start_ns) { p->stat_start_ns = now; p->stat_min_queued = 1 << 30; }
            else if (now - p->stat_last_ns > p->stat_max_gap_ns) p->stat_max_gap_ns = now - p->stat_last_ns;
            p->stat_last_ns = now; ++p->stat_buffers;
            if (p->stream) {
                int queued = SDL_GetAudioStreamQueued(p->stream);
                if (queued < p->stat_min_queued) p->stat_min_queued = queued;
                if (queued == 0) ++p->stat_starved;
            }
            if (now - p->stat_start_ns >= 5000000000u) {
                printf("Audio stats port %d: %u buffers/5s (expected %u), max gap %.2f ms, min queued %d bytes, starved %u\n",
                       (int)(p - ports), p->stat_buffers, (unsigned)(5u * 48000u / (unsigned)p->frames),
                       p->stat_max_gap_ns / 1e6, p->stat_min_queued, p->stat_starved);
                p->stat_start_ns = now; p->stat_max_gap_ns = 0; p->stat_starved = 0; p->stat_buffers = 0; p->stat_min_queued = 1 << 30;
            }
        }
        if (pace) {
            uint64_t now = now_ns();
            if (!p->next_deadline_ns || now > p->next_deadline_ns + 8 * period) p->next_deadline_ns = now;
            else sleep_until(p->next_deadline_ns);
            p->next_deadline_ns += period;
        }
        if (p->spatial && !runtime_audio_spatial_write((const float *)converted, (uint32_t)p->frames)) {
            /* Windows invalidated spatial output (e.g. device unplugged): fall back to SDL3 cleanly. */
            runtime_audio_spatial_close();
            p->spatial = 0;
            int out = target_out_channels(p, mode);
            SDL_AudioSpec spec = {p->is_float ? SDL_AUDIO_F32 : SDL_AUDIO_S16, out, 48000};
            p->stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, NULL, NULL);
            p->out_channels = out;
            if (p->stream) SDL_ResumeAudioStreamDevice(p->stream);
        } else if (p->stream) {
            int bpf = p->out_channels * p->sample_bytes;
            int low = queue_low_bytes(p);
            int extra = device_period_frames > p->frames ? device_period_frames : p->frames;
            int high = low + extra * bpf * 2;
            int queued = SDL_GetAudioStreamQueued(p->stream);
            if (queued < 0) queued = 0;
            if (!p->primed) {
                int fill = low - queued;
                if (bpf > 1) fill -= fill % bpf;
                if (fill > 0) put_silence(p->stream, fill);
                p->primed = 1;
                queued = SDL_GetAudioStreamQueued(p->stream);
                if (queued < 0) queued = 0;
            } else if (queued < (int)out_bytes) {
                int quantum = device_period_frames > 0 ? device_period_frames * bpf : (int)out_bytes;
                int fill = quantum - queued;
                if (fill < (int)out_bytes) fill = (int)out_bytes;
                if (fill > low - queued) fill = low - queued;
                if (bpf > 1) fill -= fill % bpf;
                if (fill > 0) put_silence(p->stream, fill);
                queued = SDL_GetAudioStreamQueued(p->stream);
                if (queued < 0) queued = 0;
            }
            if (!p->window_count || queued < p->window_min) p->window_min = queued;
            if (++p->window_count == 32) {
                p->adjust_ns = p->window_min > low + 2 * (int)out_bytes ? (int64_t)(period / 32)
                             : p->window_min < low ? -(int64_t)(period / 32)
                                                   : 0;
                p->window_count = 0;
            }
            p->next_deadline_ns += (uint64_t)p->adjust_ns;
            if (queued > high) {
                if (!p->drop_warned) {
                    fprintf(stderr, "Runtime: audio queue backed up (%d bytes, limit %d), dropping one buffer\n",
                            queued, high);
                    p->drop_warned = 1;
                }
            } else {
                if (queued < low) p->drop_warned = 0;
                SDL_PutAudioStreamData(p->stream, converted, (int)out_bytes);
            }
        }
        ++buffers_out;
    }
    p->last_output_us = now_ns() / 1000;
    pthread_mutex_unlock(&p->lock);
    return data ? (int32_t)samples : 0;
}
static ABI int32_t audio_output(int32_t handle, const void *data) { return output_port(handle,data,1); }
typedef struct { int32_t handle; const void *data; } OutputParam;
static ABI int32_t audio_outputs(const OutputParam *params, uint32_t count) {
    if (!params) return ERR_INVALID_POINTER;
    if (!count || count>PORTS) return ERR_PORT_FULL;
    int32_t result=0;
    /* One period per call for all ports (FMOD feeds main and BGM together): pacing every
       port would halve the rate and starve both. */
    for (uint32_t i=0;i<count;++i) { int32_t r=output_port(params[i].handle,params[i].data,i==0); if (r<0) return r; result=r; }
    return result;
}
static ABI int32_t audio_volume(int32_t handle, int32_t flags, const int32_t *volume) {
    int32_t error=0;
    pthread_mutex_lock(&table_lock);
    Port *p=port_of(handle,&error);
    pthread_mutex_unlock(&table_lock);
    if (!p) return error;
    if (!volume) return ERR_INVALID_POINTER;
    pthread_mutex_lock(&p->lock);
    for (int c=0;c<8;++c) if (flags & (1<<c)) {
        if (volume[c]<0 || volume[c]>VOLUME_0DB) { pthread_mutex_unlock(&p->lock); return ERR_INVALID_VOLUME; }
        p->volume[c]=volume[c];
    }
    pthread_mutex_unlock(&p->lock);
    return 0;
}
static ABI int32_t audio_state(int32_t handle, PortState *state) {
    int32_t error=0;
    pthread_mutex_lock(&table_lock);
    Port *p=port_of(handle,&error);
    pthread_mutex_unlock(&table_lock);
    if (!p) return error;
    if (!state) return ERR_INVALID_POINTER;
    memset(state,0,sizeof(*state));
    switch (p->type) {
    case 2: case 3: state->output=0x40; state->channel=1; break;          /* headphone */
    case 4: state->output=0x04; state->channel=1; state->volume=127; break; /* pad speaker */
    default: state->output=0x01; state->channel=(uint8_t)(p->channels>2 ? 2 : p->channels); break;
    }
    return 0;
}
static ABI int32_t audio_last_time(int32_t handle, uint64_t *time) {
    int32_t error=0;
    pthread_mutex_lock(&table_lock);
    Port *p=port_of(handle,&error);
    pthread_mutex_unlock(&table_lock);
    if (!p) return error;
    if (!time) return ERR_INVALID_POINTER;
    *time=p->last_output_us; return 0;
}

static const RuntimeExport exports[]={
    {"sceAudioOutInit",audio_init}, {"sceAudioOutOpen",audio_open}, {"sceAudioOutClose",audio_close},
    {"sceAudioOutOutput",audio_output}, {"sceAudioOutOutputs",audio_outputs},
    {"sceAudioOutSetVolume",audio_volume}, {"sceAudioOutGetPortState",audio_state},
    {"sceAudioOutGetLastOutputTime",audio_last_time},
};
uintptr_t runtime_audio_resolve(const char *name) { return RUNTIME_LOOKUP(exports,name); }
void runtime_audio_report(void) { printf("Runtime: audio ports opened=%zu, buffers output=%zu\n",ports_opened,buffers_out); }
