/* SPDX-License-Identifier: GPL-2.0-or-later
 * Native Windows Spatial Sound (ISpatialAudioClient) static 7.1 speaker-bed output.
 * Activated only on Windows when Audio Output is set to Surround (7.1) and Windows Spatial Sound
 * (Dolby Atmos for Headphones, DTS Headphone:X, or Windows Sonic) is active on the output device.
 */
#include "runtime_audio_spatial.h"

#ifdef _WIN32
#define COBJMACROS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <objbase.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <spatialaudioclient.h>
#include <propidl.h>
#include <mmreg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BB_SPATIAL_CH 8u
#define BB_SPATIAL_RING_FRAMES 8192u /* power-of-two, ~170 ms @ 48 kHz */
#define BB_SPATIAL_RING_MASK (BB_SPATIAL_RING_FRAMES - 1u)
#define BB_SPATIAL_PREFILL 1536u /* ~32 ms cushion before starting playback */

typedef struct {
    float buf[BB_SPATIAL_RING_FRAMES * BB_SPATIAL_CH];
    atomic_uint written;
    atomic_uint read;
    atomic_uint underruns;
    atomic_uint dropped;
    int playing;
} SpatialRing;

static const GUID clsid_mm_enum = {
    0xbcde0395, 0xe52f, 0x467c, {0x8e, 0x3d, 0xc4, 0x57, 0x92, 0x91, 0x69, 0x2e}};
static const GUID iid_mm_enum = {
    0xa95664d2, 0x9614, 0x4f35, {0xa7, 0x46, 0xde, 0x8d, 0xb6, 0x36, 0x17, 0xe6}};
static const GUID iid_spatial_client = {
    0xbbf8e066, 0xaaaa, 0x49be, {0x9a, 0x4d, 0xfd, 0x2a, 0x85, 0x8e, 0xa2, 0x7f}};
static const GUID iid_spatial_stream = {
    0xbab5f473, 0xb423, 0x477b, {0x85, 0xf5, 0xb5, 0xa3, 0x32, 0xa0, 0x41, 0x53}};

/* SDL / WAVE 7.1 order: FL FR FC LFE BL BR SL SR. */
static const AudioObjectType object_types[BB_SPATIAL_CH] = {
    AudioObjectType_FrontLeft, AudioObjectType_FrontRight, AudioObjectType_FrontCenter,
    AudioObjectType_LowFrequency, AudioObjectType_BackLeft, AudioObjectType_BackRight,
    AudioObjectType_SideLeft, AudioObjectType_SideRight};

static SpatialRing ring;
static HANDLE thread_handle, ready_event;
static atomic_int stop_requested, ended;
static int started_ok;

static void ring_write(SpatialRing *r, const float *interleaved, uint32_t frames) {
    uint32_t w = atomic_load_explicit(&r->written, memory_order_relaxed);
    uint32_t rd = atomic_load_explicit(&r->read, memory_order_acquire);
    uint32_t used = w - rd;
    uint32_t space = used < BB_SPATIAL_RING_FRAMES ? (BB_SPATIAL_RING_FRAMES - used) : 0u;
    if (frames > space) {
        atomic_fetch_add_explicit(&r->dropped, frames - space, memory_order_relaxed);
        frames = space;
    }
    for (uint32_t f = 0; f < frames; ++f) {
        uint32_t slot = (w + f) & BB_SPATIAL_RING_MASK;
        memcpy(&r->buf[slot * BB_SPATIAL_CH], &interleaved[f * BB_SPATIAL_CH],
               BB_SPATIAL_CH * sizeof(float));
    }
    atomic_store_explicit(&r->written, w + frames, memory_order_release);
}

static void ring_read(SpatialRing *r, float *const out[BB_SPATIAL_CH], uint32_t frames) {
    uint32_t w = atomic_load_explicit(&r->written, memory_order_acquire);
    uint32_t rd = atomic_load_explicit(&r->read, memory_order_relaxed);
    uint32_t avail = w - rd;
    if (!r->playing) {
        if (avail < BB_SPATIAL_PREFILL) {
            for (uint32_t c = 0; c < BB_SPATIAL_CH; ++c) {
                if (out[c]) memset(out[c], 0, frames * sizeof(float));
            }
            return;
        }
        r->playing = 1;
    }
    uint32_t take = avail < frames ? avail : frames;
    for (uint32_t f = 0; f < take; ++f) {
        uint32_t slot = (rd + f) & BB_SPATIAL_RING_MASK;
        const float *s = &r->buf[slot * BB_SPATIAL_CH];
        for (uint32_t c = 0; c < BB_SPATIAL_CH; ++c) {
            if (out[c]) out[c][f] = s[c];
        }
    }
    if (take < frames) {
        atomic_fetch_add_explicit(&r->underruns, 1u, memory_order_relaxed);
        r->playing = 0;
        for (uint32_t c = 0; c < BB_SPATIAL_CH; ++c) {
            if (out[c]) memset(out[c] + take, 0, (frames - take) * sizeof(float));
        }
    }
    atomic_store_explicit(&r->read, rd + take, memory_order_release);
}

static ISpatialAudioObjectRenderStream *start_stream(ISpatialAudioClient **client_out,
                                                     HANDLE quantum_event) {
    IMMDeviceEnumerator *enumerator = NULL;
    IMMDevice *device = NULL;
    ISpatialAudioClient *client = NULL;
    ISpatialAudioObjectRenderStream *stream = NULL;
    HRESULT hr = CoCreateInstance(&clsid_mm_enum, NULL, CLSCTX_ALL, &iid_mm_enum,
                                  (void **)&enumerator);
    if (SUCCEEDED(hr)) {
        hr = IMMDeviceEnumerator_GetDefaultAudioEndpoint(enumerator, eRender, eConsole, &device);
    }
    if (SUCCEEDED(hr)) {
        hr = IMMDevice_Activate(device, &iid_spatial_client, CLSCTX_INPROC_SERVER, NULL,
                                (void **)&client);
    }
    if (SUCCEEDED(hr)) {
        hr = ISpatialAudioClient_IsSpatialAudioStreamAvailable(client, &iid_spatial_stream, NULL);
    }
    WAVEFORMATEX format = {WAVE_FORMAT_IEEE_FLOAT, 1, 48000, 48000 * 4, 4, 32, 0};
    if (SUCCEEDED(hr)) {
        hr = ISpatialAudioClient_IsAudioObjectFormatSupported(client, &format);
    }
    AudioObjectType mask = 0;
    for (uint32_t c = 0; c < BB_SPATIAL_CH; ++c) mask = (AudioObjectType)(mask | object_types[c]);
    if (SUCCEEDED(hr)) {
        SpatialAudioObjectRenderStreamActivationParams params = {
            &format, mask, 0, 0, AudioCategory_GameEffects, quantum_event, NULL};
        PROPVARIANT activation;
        PropVariantInit(&activation);
        activation.vt = VT_BLOB;
        activation.blob.cbSize = sizeof(params);
        activation.blob.pBlobData = (BYTE *)&params;
        hr = ISpatialAudioClient_ActivateSpatialAudioStream(client, &activation,
                                                            &iid_spatial_stream, (void **)&stream);
    }
    if (SUCCEEDED(hr)) {
        hr = ISpatialAudioObjectRenderStream_Start(stream);
    }
    if (FAILED(hr)) {
        if (stream) ISpatialAudioObjectRenderStream_Release(stream);
        stream = NULL;
    }
    if (device) IMMDevice_Release(device);
    if (enumerator) IMMDeviceEnumerator_Release(enumerator);
    if (stream) {
        *client_out = client;
    } else if (client) {
        ISpatialAudioClient_Release(client);
    }
    return stream;
}

static DWORD WINAPI spatial_thread(void *unused) {
    (void)unused;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
    HANDLE quantum_event = CreateEventW(NULL, FALSE, FALSE, NULL);
    ISpatialAudioClient *client = NULL;
    ISpatialAudioObjectRenderStream *stream = start_stream(&client, quantum_event);
    started_ok = stream != NULL;
    SetEvent(ready_event);
    if (stream) {
        ISpatialAudioObject *objects[BB_SPATIAL_CH] = {0};
        int late = 0;
        while (!atomic_load(&stop_requested)) {
            if (WaitForSingleObject(quantum_event, 200) != WAIT_OBJECT_0) {
                if (++late >= 5) break;
                continue;
            }
            late = 0;
            UINT32 dynamic = 0, frames = 0;
            if (FAILED(ISpatialAudioObjectRenderStream_BeginUpdatingAudioObjects(
                    stream, &dynamic, &frames))) {
                break;
            }
            float *out[BB_SPATIAL_CH] = {0};
            int ok = 1;
            for (uint32_t c = 0; c < BB_SPATIAL_CH; ++c) {
                if (!objects[c]) {
                    if (FAILED(ISpatialAudioObjectRenderStream_ActivateSpatialAudioObject(
                            stream, object_types[c], &objects[c]))) {
                        objects[c] = NULL;
                        ok = 0;
                        break;
                    }
                }
                BYTE *buffer = NULL;
                UINT32 bytes = 0;
                if (FAILED(ISpatialAudioObject_GetBuffer(objects[c], &buffer, &bytes))) {
                    ok = 0;
                    break;
                }
                if (bytes / sizeof(float) >= frames) out[c] = (float *)buffer;
            }
            if (ok) ring_read(&ring, out, frames);
            ISpatialAudioObjectRenderStream_EndUpdatingAudioObjects(stream);
            if (!ok) break;
        }
        atomic_store(&ended, 1);
        ISpatialAudioObjectRenderStream_Stop(stream);
        for (uint32_t c = 0; c < BB_SPATIAL_CH; ++c) {
            if (objects[c]) ISpatialAudioObject_Release(objects[c]);
        }
        ISpatialAudioObjectRenderStream_Release(stream);
        ISpatialAudioClient_Release(client);
    }
    CloseHandle(quantum_event);
    CoUninitialize();
    return 0;
}

int runtime_audio_spatial_open(void) {
    if (thread_handle) return 1;
    atomic_store(&ring.written, 0u);
    atomic_store(&ring.read, 0u);
    atomic_store(&ring.underruns, 0u);
    atomic_store(&ring.dropped, 0u);
    ring.playing = 0;
    atomic_store(&stop_requested, 0);
    atomic_store(&ended, 0);
    ready_event = CreateEventW(NULL, TRUE, FALSE, NULL);
    thread_handle = CreateThread(NULL, 0, spatial_thread, NULL, 0, NULL);
    if (!thread_handle) {
        CloseHandle(ready_event);
        return 0;
    }
    WaitForSingleObject(ready_event, 5000);
    CloseHandle(ready_event);
    if (!started_ok) {
        runtime_audio_spatial_close();
        return 0;
    }
    puts("Runtime: Windows Spatial Sound active (discrete 7.1 bed)");
    return 1;
}

int runtime_audio_spatial_write(const float *frames_sdl71, uint32_t frames) {
    if (atomic_load(&ended)) return 0;
    ring_write(&ring, frames_sdl71, frames);
    return 1;
}

void runtime_audio_spatial_close(void) {
    if (!thread_handle) return;
    atomic_store(&stop_requested, 1);
    WaitForSingleObject(thread_handle, 2000);
    CloseHandle(thread_handle);
    thread_handle = NULL;
}
#else
int runtime_audio_spatial_open(void) { return 0; }
int runtime_audio_spatial_write(const float *frames_sdl71, uint32_t frames) {
    (void)frames_sdl71;
    (void)frames;
    return 0;
}
void runtime_audio_spatial_close(void) {}
#endif
