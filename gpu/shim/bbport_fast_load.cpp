// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// bb_vulkan: Fast Location & Respawn Loading Subsystem (BB_FAST_LOAD)
// File: gpu/shim/bbport_fast_load.cpp
// ============================================================================
#include "bbport_fast_load.h"

#include "bb_env.h"
#include "bbport_crash_blackbox.h"
#include "bbport_frame_rate.h"
#include "bbport_guest_code_hooks.h"
#include "bbport_settings.h"
#include "common/types.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <new>
#include <sys/mman.h>
#include <unistd.h>

extern "C" {
int runtime_memory_is_mapped(std::uintptr_t address, std::uint64_t size);
void runtime_memory_prepare_cpu_write(std::uintptr_t address, std::uint64_t size);
void runtime_memory_note_cpu_write(std::uintptr_t address, std::uint64_t size);
}

namespace BbFastLoad {
namespace {

constexpr std::uint64_t kHostZlibBridgeMagic = 0x42425A4C42524447ull; // "BBZLBRDG"
constexpr std::uint64_t kBurstActiveWindowNs = 250'000'000ull;        // 250 ms post-inflate hold
constexpr std::uint64_t kMaxLoadingScreenDraws = 160ull;              // 3D gameplay is 450..1500+
constexpr std::uint64_t kMaxNowLoadingDraws = 320ull;                 // sanity guard while MenuMan NowLoading is active

// Zero thread_local variables (strictly preserves PT_TLS <= 73.00 KiB ceiling).
std::atomic<unsigned char*> g_image_base{nullptr};
std::atomic<std::uint64_t> g_image_size{0};
std::atomic<bool> g_tips_timer_site_verified{false};
std::atomic<std::uint64_t> g_last_inflate_ns{0};
std::atomic<std::uint64_t> g_inflated_bytes{0};
std::atomic<std::uint64_t> g_inflate_calls{0};
std::atomic<bool> g_loading_burst_active{false};

void PS4_SYSV_ABI HostZlibBridgeFree(void* opaque, void* address) noexcept {
    (void)opaque;
    if (!address) {
        return;
    }
    auto* bridge = static_cast<HostZlibBridgeState*>(address);
    if (bridge->magic == kHostZlibBridgeMagic) {
        if (bridge->host_strm.total_out >= 65536u) {
            BbCrashBlackBox::RecordEventFmt("DCX", "inflated %lu KiB -> %lu KiB",
                                            static_cast<unsigned long>(bridge->host_strm.total_in >> 10),
                                            static_cast<unsigned long>(bridge->host_strm.total_out >> 10));
        }
        bridge->magic = 0;
        ::inflateEnd(&bridge->host_strm);
        delete bridge;
    }
}

int PS4_SYSV_ABI HostInflateInit2_(GuestZStream* strm, int window_bits, const char* version,
                                   int stream_size) noexcept {
    if (!strm || !version || version[0] != '1' ||
        stream_size != static_cast<int>(sizeof(GuestZStream))) {
        return Z_VERSION_ERROR;
    }
    strm->msg = nullptr;
    auto* bridge = new (std::nothrow) HostZlibBridgeState{};
    if (!bridge) {
        return Z_MEM_ERROR;
    }
    bridge->magic = kHostZlibBridgeMagic;
    bridge->saved_guest_zfree = strm->zfree;
    bridge->window_bits = window_bits;
    bridge->fake_window_must_be_zero = 0;
    bridge->host_strm.zalloc = Z_NULL;
    bridge->host_strm.zfree = Z_NULL;
    bridge->host_strm.opaque = Z_NULL;
    bridge->host_strm.next_in = Z_NULL;
    bridge->host_strm.avail_in = 0;

    const int rc = inflateInit2(&bridge->host_strm, window_bits);
    if (rc != Z_OK) {
        bridge->magic = 0;
        delete bridge;
        strm->state = nullptr;
        return rc;
    }

    strm->state = bridge;
    strm->zfree = reinterpret_cast<void*>(&HostZlibBridgeFree);
    strm->total_in = 0;
    strm->total_out = 0;
    strm->data_type = 0;
    strm->adler = static_cast<std::uint64_t>(bridge->host_strm.adler);
    return Z_OK;
}

int PS4_SYSV_ABI HostInflateInit_(GuestZStream* strm, const char* version,
                                  int stream_size) noexcept {
    return HostInflateInit2_(strm, 15, version, stream_size);
}

int PS4_SYSV_ABI HostInflateReset(GuestZStream* strm) noexcept {
    if (!strm || !strm->state) {
        return Z_STREAM_ERROR;
    }
    auto* bridge = static_cast<HostZlibBridgeState*>(strm->state);
    if (bridge->magic != kHostZlibBridgeMagic) {
        return Z_STREAM_ERROR;
    }
    const int rc = ::inflateReset(&bridge->host_strm);
    if (rc == Z_OK) {
        strm->total_in = 0;
        strm->total_out = 0;
        strm->msg = nullptr;
        strm->data_type = 0;
        strm->adler = static_cast<std::uint64_t>(bridge->host_strm.adler);
    }
    return rc;
}

int PS4_SYSV_ABI HostInflateReset2(GuestZStream* strm, int window_bits) noexcept {
    if (!strm || !strm->state) {
        return Z_STREAM_ERROR;
    }
    auto* bridge = static_cast<HostZlibBridgeState*>(strm->state);
    if (bridge->magic != kHostZlibBridgeMagic) {
        return Z_STREAM_ERROR;
    }
    const int rc = ::inflateReset2(&bridge->host_strm, window_bits);
    if (rc == Z_OK) {
        bridge->window_bits = window_bits;
        strm->total_in = 0;
        strm->total_out = 0;
        strm->msg = nullptr;
        strm->data_type = 0;
        strm->adler = static_cast<std::uint64_t>(bridge->host_strm.adler);
    }
    return rc;
}

int PS4_SYSV_ABI HostInflate(GuestZStream* strm, int flush) noexcept {
    if (!strm || !strm->state) {
        return Z_STREAM_ERROR;
    }
    auto* bridge = static_cast<HostZlibBridgeState*>(strm->state);
    if (bridge->magic != kHostZlibBridgeMagic) {
        return Z_STREAM_ERROR;
    }

    const std::uint8_t* const in_before = strm->next_in;
    std::uint8_t* const out_before = strm->next_out;
    const std::uint32_t avail_in_before = strm->avail_in;
    const std::uint32_t avail_out_before = strm->avail_out;

    if (out_before && avail_out_before > 0) {
        runtime_memory_prepare_cpu_write(reinterpret_cast<std::uintptr_t>(out_before),
                                         avail_out_before);
    }

    bridge->host_strm.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(in_before));
    bridge->host_strm.avail_in = avail_in_before;
    bridge->host_strm.next_out = reinterpret_cast<Bytef*>(out_before);
    bridge->host_strm.avail_out = avail_out_before;

    const int rc = ::inflate(&bridge->host_strm, flush);

    const std::uint64_t consumed = avail_in_before - bridge->host_strm.avail_in;
    const std::uint64_t produced = avail_out_before - bridge->host_strm.avail_out;

    strm->next_in = reinterpret_cast<const std::uint8_t*>(bridge->host_strm.next_in);
    strm->avail_in = bridge->host_strm.avail_in;
    strm->total_in += consumed;
    strm->next_out = reinterpret_cast<std::uint8_t*>(bridge->host_strm.next_out);
    strm->avail_out = bridge->host_strm.avail_out;
    strm->total_out += produced;
    strm->msg = nullptr; // Never store a 64-bit host .rodata pointer into guest z_stream
    strm->data_type = bridge->host_strm.data_type;
    strm->adler = static_cast<std::uint64_t>(bridge->host_strm.adler);

    if (out_before && produced > 0) {
        runtime_memory_note_cpu_write(reinterpret_cast<std::uintptr_t>(out_before), produced);
        g_inflated_bytes.fetch_add(produced, std::memory_order_relaxed);
        const std::uint64_t call_idx = g_inflate_calls.fetch_add(1, std::memory_order_relaxed);
        if ((call_idx & 7ull) == 0ull || rc == Z_STREAM_END) {
            g_last_inflate_ns.store(BbFrameRate::NowMonotonicNs(), std::memory_order_relaxed);
        }
    }
    return rc;
}

int PS4_SYSV_ABI HostInflateEnd(GuestZStream* strm) noexcept {
    if (!strm || !strm->state) {
        return Z_STREAM_ERROR;
    }
    auto* bridge = static_cast<HostZlibBridgeState*>(strm->state);
    if (bridge->magic != kHostZlibBridgeMagic) {
        return Z_STREAM_ERROR;
    }
    strm->zfree = bridge->saved_guest_zfree;
    strm->state = nullptr;
    strm->msg = nullptr;
    bridge->magic = 0;
    const int rc = ::inflateEnd(&bridge->host_strm);
    delete bridge;
    return rc;
}

std::uint8_t* MapStubPageNear(const std::uint8_t* image, std::size_t alloc_bytes) {
    const long gran = std::max<long>(::sysconf(_SC_PAGESIZE), 65536L);
    const auto base = reinterpret_cast<std::int64_t>(image);
    constexpr std::int64_t kReach = 0x7000'0000ll;
    for (std::int64_t step = 1; step * (32ll << 20) < kReach; ++step) {
        for (const std::int64_t sign : {-1ll, 1ll}) {
            const std::int64_t at = (base + sign * step * (32ll << 20)) & ~std::int64_t(gran - 1);
            if (at <= 0) {
                continue;
            }
            void* p = ::mmap(reinterpret_cast<void*>(at), alloc_bytes,
                             PROT_READ | PROT_WRITE | PROT_EXEC,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
            if (p != MAP_FAILED) {
                return static_cast<std::uint8_t*>(p);
            }
        }
    }
    return nullptr;
}

bool WriteVerifiedBytes(std::uint8_t* addr, const std::uint8_t* bytes, std::size_t len,
                        std::size_t page) {
    const auto begin = reinterpret_cast<std::uintptr_t>(addr) & ~(page - 1u);
    const auto end = (reinterpret_cast<std::uintptr_t>(addr) + len + page - 1u) & ~(page - 1u);
    if (::mprotect(reinterpret_cast<void*>(begin), end - begin,
                   PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
        return false;
    }
    if (len > 1) {
        std::memcpy(addr + 1, bytes + 1, len - 1);
    }
    __atomic_store_n(addr, bytes[0], __ATOMIC_SEQ_CST);
    ::mprotect(reinterpret_cast<void*>(begin), end - begin, Core::GuestCodeProtection());
    return true;
}

} // namespace

bool Enabled() noexcept {
    static const bool enabled = [] {
        const char* env = bb_env("BB_FAST_LOAD");
        return !env || !*env || !(env[0] == '0' && env[1] == '\0');
    }();
    return enabled;
}

void PatchImage(unsigned char* image, std::uint64_t size) {
    g_image_size.store(size, std::memory_order_relaxed);
    g_image_base.store(image, std::memory_order_release);
    if (!Enabled()) {
        std::printf("[bb_vulkan] Fast Load: disabled (BB_FAST_LOAD=0)\n");
        return;
    }

    struct DetourSpec {
        std::uint64_t vma;
        std::array<std::uint8_t, 8> prologue;
        void* host_fn;
    };

    const DetourSpec specs[] = {
        // Copy #1: embedded zlib 1.2.7 (Scaleform .gfx, PNG, DLZlibReader)
        {Vma::Z127_InflateReset,
         {0xb8, 0xfe, 0xff, 0xff, 0xff, 0x48, 0x85, 0xff},
         reinterpret_cast<void*>(&HostInflateReset)},
        {Vma::Z127_InflateReset2,
         {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56},
         reinterpret_cast<void*>(&HostInflateReset2)},
        {Vma::Z127_InflateInit2_,
         {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56},
         reinterpret_cast<void*>(&HostInflateInit2_)},
        {Vma::Z127_InflateInit_,
         {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56},
         reinterpret_cast<void*>(&HostInflateInit_)},
        {Vma::Z127_Inflate,
         {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56},
         reinterpret_cast<void*>(&HostInflate)},
        {Vma::Z127_InflateEnd,
         {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89},
         reinterpret_cast<void*>(&HostInflateEnd)},

        // Copy #2: embedded zlib 1.2.3 (.dcx DCP\0DFLT & DCP\0EDGE archives)
        {Vma::Z123_InflateReset,
         {0xb8, 0xfe, 0xff, 0xff, 0xff, 0x48, 0x85, 0xff},
         reinterpret_cast<void*>(&HostInflateReset)},
        {Vma::Z123_InflateInit2_,
         {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x41},
         reinterpret_cast<void*>(&HostInflateInit2_)},
        {Vma::Z123_InflateInit_,
         {0x89, 0xd0, 0x48, 0x89, 0xf1, 0xbe, 0x0f, 0x00},
         reinterpret_cast<void*>(&HostInflateInit_)},
        {Vma::Z123_Inflate,
         {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56},
         reinterpret_cast<void*>(&HostInflate)},
        {Vma::Z123_InflateEnd,
         {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89},
         reinterpret_cast<void*>(&HostInflateEnd)},
    };

    const std::size_t page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const std::size_t stub_alloc = std::max<std::size_t>(page, 4096u);
    std::uint8_t* stubs = MapStubPageNear(image, stub_alloc);
    if (!stubs) {
        std::fprintf(stderr, "[bb_vulkan] Fast Load: MapStubPageNear failed\n");
        return;
    }

    std::size_t installed = 0;
    constexpr std::size_t kSlotSize = 16;
    for (const auto& sp : specs) {
        if (sp.vma + sp.prologue.size() > size) {
            continue;
        }
        std::uint8_t* guest_fn = image + sp.vma;
        if (std::memcmp(guest_fn, sp.prologue.data(), sp.prologue.size()) != 0) {
            continue;
        }

        std::uint8_t* stub = stubs + installed * kSlotSize;
        stub[0] = 0xff;
        stub[1] = 0x25;
        stub[2] = 0x00;
        stub[3] = 0x00;
        stub[4] = 0x00;
        stub[5] = 0x00;
        const std::uint64_t target = reinterpret_cast<std::uint64_t>(sp.host_fn);
        std::memcpy(stub + 6, &target, 8);
        stub[14] = 0xcc;
        stub[15] = 0xcc;

        const std::int64_t rel =
            reinterpret_cast<std::int64_t>(stub) - reinterpret_cast<std::int64_t>(guest_fn + 5);
        if (rel < -(1ll << 31) || rel >= (1ll << 31)) {
            continue;
        }
        std::uint8_t jmp[5];
        jmp[0] = 0xe9;
        const auto rel32 = static_cast<std::int32_t>(rel);
        std::memcpy(jmp + 1, &rel32, 4);
        if (WriteVerifiedBytes(guest_fn, jmp, sizeof(jmp), page)) {
            ++installed;
        }
    }
    ::mprotect(stubs, stub_alloc, PROT_READ | PROT_EXEC);

    // Upgrade DcxDfltDecompressor streaming chunk size from 16 KiB (0x4000) to 64 KiB (0x10000).
    bool chunk_64k = false;
    static constexpr std::array<std::uint8_t, 10> kExpectedDcxChunk = {
        0xb8, 0x00, 0x40, 0x00, 0x00, // mov eax, 0x4000
        0xb9, 0x00, 0x00, 0x01, 0x00, // mov ecx, 0x10000
    };
    if (Vma::DcxDflt_ChunkImm + kExpectedDcxChunk.size() <= size &&
        std::memcmp(image + Vma::DcxDflt_ChunkImm, kExpectedDcxChunk.data(),
                    kExpectedDcxChunk.size()) == 0) {
        static constexpr std::array<std::uint8_t, 5> kPatchedMov64k = {
            0xb8, 0x00, 0x00, 0x01, 0x00, // mov eax, 0x10000
        };
        chunk_64k = WriteVerifiedBytes(image + Vma::DcxDflt_ChunkImm, kPatchedMov64k.data(),
                                       kPatchedMov64k.size(), page);
    }

    // Verify StageFlow's 12.0s artificial item-tips reading timer site (0x01938b62):
    //   74 0a                   je  +10 (skip 12.0f -> 0.0f on Load Game)
    //   c5 fa 10 05 5c f5 fe 02 vmovss xmm0, [rip+0x2fef55c]  # 0x049280c8 = 12.0f
    static constexpr std::array<std::uint8_t, 10> kExpectedTipsTimerJe = {
        0x74, 0x0a, 0xc5, 0xfa, 0x10, 0x05, 0x5c, 0xf5, 0xfe, 0x02,
    };
    if (Vma::StageFlow_TipsTimerJe + kExpectedTipsTimerJe.size() <= size &&
        std::memcmp(image + Vma::StageFlow_TipsTimerJe, kExpectedTipsTimerJe.data(),
                    kExpectedTipsTimerJe.size()) == 0) {
        g_tips_timer_site_verified.store(true, std::memory_order_release);
        SyncLoadingTipsTimer(BbSettings::Get().loading_tips_timer.load(std::memory_order_relaxed));
    }

    std::printf("[bb_vulkan] Fast Load: host SIMD zlib %s (%zu/11 hooks, libz %s), DcxDflt 64 KiB "
                "chunk %s, 500 Hz loading pacer ON, 12s tips timer %s\n",
                installed == 11 ? "ON" : "PARTIAL", installed, ::zlibVersion(),
                chunk_64k ? "ON" : "OFF",
                g_tips_timer_site_verified.load(std::memory_order_relaxed)
                    ? (BbSettings::Get().loading_tips_timer.load(std::memory_order_relaxed)
                           ? "KEPT (menu=On)"
                           : "BYPASSED (menu=Off)")
                    : "UNAVAILABLE");
}

void SyncLoadingTipsTimer(bool keep_tips_timer) noexcept {
    if (!g_tips_timer_site_verified.load(std::memory_order_acquire)) {
        return;
    }
    unsigned char* base = g_image_base.load(std::memory_order_acquire);
    if (!base) {
        return;
    }
    const std::size_t page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    const std::uint8_t opcode = keep_tips_timer ? 0x74u : 0xebu; // je (12s) vs jmp (0s)
    WriteVerifiedBytes(base + Vma::StageFlow_TipsTimerJe, &opcode, 1, page);
}

bool UpdateFrameAndCheckBurst(std::uint64_t now_ns, std::uint64_t frame_draws) noexcept {
    if (!Enabled()) {
        return false;
    }

    // 1. Exact engine ground-truth: read Bloodborne's MenuMan NowLoading state (+0x804) and
    //    NowLoadingOpenFlag (ImageBase + 0x0556286b).
    //    IMPORTANT: MenuManPtr lives in eboot.bin's ELF .bss (ALWAYS mapped in [base, base+size)),
    //    NOT in runtime_memory's direct/flexible vmas[] table — so check offset < g_image_size
    //    instead of runtime_memory_is_mapped() on the .bss pointer itself!
    bool engine_now_loading = false;
    bool have_menu_man = false;
    std::uint32_t observed_state = 0;
    float observed_timer_before = 0.0f;
    float observed_timer_after = 0.0f;
    const unsigned char* base = g_image_base.load(std::memory_order_acquire);
    const std::uint64_t img_size = g_image_size.load(std::memory_order_relaxed);
    if (base && Vma::MenuManPtr + sizeof(std::uint64_t) <= img_size) {
        const std::uint8_t open_flag = *(base + Vma::NowLoadingOpenFlag);
        const std::uint64_t menu_man =
            *reinterpret_cast<const std::uint64_t*>(base + Vma::MenuManPtr);
        if (menu_man >= 0x10000ull &&
            runtime_memory_is_mapped(
                static_cast<std::uintptr_t>(menu_man + Vma::MenuMan_LoadingStateOff),
                sizeof(std::uint32_t))) {
            have_menu_man = true;
            observed_state = *reinterpret_cast<const std::uint32_t*>(
                menu_man + Vma::MenuMan_LoadingStateOff);
            if (observed_state == 1u || observed_state == 2u ||
                (open_flag != 0u && observed_state != 3u)) {
                engine_now_loading = true;
                const auto timer_addr =
                    static_cast<std::uintptr_t>(menu_man + Vma::MenuMan_MinTimerOff);
                if (runtime_memory_is_mapped(timer_addr, sizeof(float))) {
                    auto* min_timer = reinterpret_cast<float*>(timer_addr);
                    observed_timer_before = *min_timer;
                    if (!BbSettings::Get().loading_tips_timer.load(std::memory_order_relaxed) &&
                        *min_timer > 0.0f) {
                        *min_timer = 0.0f;
                    }
                    observed_timer_after = *min_timer;
                }
            }
        }
    }

    // Track exact loading-screen transitions in the log for verification.
    static bool prev_engine_loading = false;
    static std::uint64_t loading_enter_ns = 0;
    static std::uint64_t loading_frames = 0;
    if (engine_now_loading) {
        if (!prev_engine_loading) {
            loading_enter_ns = now_ns;
            loading_frames = 0;
            std::printf("[bb_vulkan] Fast Load: NowLoading ENTER (state=%u, tips_timer=%.2fs -> %.2fs, draws=%llu)\n",
                        observed_state, double(observed_timer_before), double(observed_timer_after),
                        static_cast<unsigned long long>(frame_draws));
        }
        ++loading_frames;
    } else if (prev_engine_loading) {
        const double dur_s =
            (now_ns >= loading_enter_ns) ? double(now_ns - loading_enter_ns) * 1e-9 : 0.0;
        std::printf("[bb_vulkan] Fast Load: NowLoading EXIT (%.2f s, %llu frames, state=%u, draws=%llu)\n",
                    dur_s, static_cast<unsigned long long>(loading_frames), observed_state,
                    static_cast<unsigned long long>(frame_draws));
    }
    prev_engine_loading = engine_now_loading;

    // 2. Fallback heuristic ONLY for early boot before MenuMan exists. Once MenuMan is mapped,
    //    trust engine_now_loading exclusively so background .dcx asset streaming during gameplay
    //    never false-triggers 500 Hz burst / drops frames / bypasses PresentPacing.
    const std::uint64_t last_ns = g_last_inflate_ns.load(std::memory_order_relaxed);
    const bool recent_inflate =
        (last_ns != 0) && (now_ns >= last_ns ? (now_ns - last_ns <= kBurstActiveWindowNs) : true);
    const bool active =
        have_menu_man
            ? (engine_now_loading && frame_draws < kMaxNowLoadingDraws)
            : (recent_inflate && frame_draws < kMaxLoadingScreenDraws);
    g_loading_burst_active.store(active, std::memory_order_relaxed);
    return active;
}

bool IsLoadingBurstActive() noexcept {
    return g_loading_burst_active.load(std::memory_order_relaxed);
}

std::uint64_t DecompressedBytes() noexcept {
    return g_inflated_bytes.load(std::memory_order_relaxed);
}

} // namespace BbFastLoad
