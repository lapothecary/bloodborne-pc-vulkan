#include "bbport_write_log.h"
#include "bbport_frame_rate.h"
#include "bbport_game_menu.h"
#include "bbport_gnm_hooks.h"
#include "bb_golden_engine.h"
// bbport: glue between the C loader and the vendored shadPS4 video core.
#include "bbport_overlay.h"
#include "bbport_guest_hooks.h"
#include "bbport_settings.h"
#include "game_profile.h"
#include "bbport_copy.h"
#include <sys/resource.h>
#include "bbport_free_check.h"
#include "bbport_toggles.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include "common/polyfill_thread.h"
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <chrono>
#include <thread>
#include <vector>
#include <SDL3/SDL.h>
#include "../bbgpu.h"
#include "platform/kernel/event_queue.h"
#include "platform/videoout/videoout.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/rdtsc.h"
#include "platform/errors.h"
#include "core/libraries/libs.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "core/signals.h"
#include "sdl_window.h"
#include "render/rasterizer.h"

extern "C" {
// runtime_memory.c
int runtime_memory_is_mapped(uintptr_t address, uint64_t size);
int runtime_memory_write_backing(uintptr_t address, const void* data, uint64_t size);
void runtime_memory_read_backing(uintptr_t address, void* data, uint64_t size);
uint64_t runtime_memory_clamp(uintptr_t address, uint64_t size);
int runtime_memory_region(uintptr_t address, uintptr_t* start, uintptr_t* end, int* mapped);
const uint64_t* runtime_memory_generation(void);
void runtime_memory_gpu_protect(uintptr_t address, uint64_t size, int read, int write);
typedef void (*RuntimeGpuRange)(uintptr_t address, uint64_t size);
// runtime_kernel.c: one clock for guest and GPU timestamps
uint64_t runtime_process_time_us(void);
uint64_t runtime_process_time_counter(void);
uint64_t runtime_tsc_frequency(void);
int32_t* runtime_errno(void);
void runtime_memory_set_gpu_hooks(RuntimeGpuRange map, RuntimeGpuRange unmap, RuntimeGpuRange invalidate);
void runtime_memory_set_note_write_hook(RuntimeGpuRange note);
void runtime_memory_set_cpu_write_hook(RuntimeGpuRange hook);
}

Frontend::WindowSDL* g_window = nullptr;

namespace BbGnm { void RegisterLib(Core::Loader::SymbolsResolver* sym); }
namespace BbAvPlayer { void RegisterLib(Core::Loader::SymbolsResolver* sym); }

namespace {
struct Symbol {
    std::string nid, library, module;
    u64 address;
};
std::vector<Symbol> g_symbols;
u32 g_sdk_version;
} // namespace

namespace Core::Loader {
void SymbolsResolver::AddSymbol(const char* nid, const char* library, const char* module, SymbolType,
                                u64 address) {
    g_symbols.push_back({nid, library, module, address});
}
} // namespace Core::Loader

namespace Core {
void MemoryManager::SetRasterizer(Vulkan::Rasterizer* rasterizer_) {
    rasterizer = rasterizer_;
    // Existing guest mappings are replayed by the runtime when hooks are installed.
    runtime_memory_set_gpu_hooks(
        [](uintptr_t address, uint64_t size) {
            auto* r = Memory::Instance()->GetRasterizer();
            r->RegisterMemory(address, size);
            r->MapMemory(address, size);
        },
        [](uintptr_t address, uint64_t size) { Memory::Instance()->GetRasterizer()->UnmapMemory(address, size); },
        [](uintptr_t address, uint64_t size) {
            Memory::Instance()->GetRasterizer()->InvalidateMemory(address, size);
        });
    // bbport: data written into GPU memory by a path the GPU side hears of (file reads, the game's
    // resource loaders): invalidated, and an asset (it may keep a VRAM copy).
    runtime_memory_set_note_write_hook([](uintptr_t address, uint64_t size) {
        Memory::Instance()->GetRasterizer()->NoteAssetWrite(address, size);
    });
    // bbport: the game's libc copies (memcpy, memset, memmove) over pages the caches watch.
    runtime_memory_set_cpu_write_hook([](uintptr_t address, uint64_t size) {
        Memory::Instance()->GetRasterizer()->OnCpuWrite(address, size);
    });
}
void MemoryManager::InvalidateMemory(VAddr address, u64 size) {
    if (rasterizer) rasterizer->InvalidateMemory(address, size);
}
namespace {
// bbport: the mapped region of the last lookup, per thread, valid for the mapping table generation
// it was read at. The GPU command thread clamps and copies thousands of constant ranges a frame,
// nearly all within the region of the one before: two calls into the runtime each (~5% of it).
struct RegionCache {
    static constexpr u32 kSets = 256u;
    static constexpr u32 kWays = 4u;
    static constexpr u32 kMaxSpans = 256u;

    struct Span {
        uintptr_t start = 0;
        uintptr_t end = 0;
        u8 mapped = 0;
    };
    struct TlbSlot {
        u64 page2m = ~u64{0};
        u16 span_idx = 0;
        u16 lru = 0;
    };

    u64 generation = 1; // odd: invalid
    u32 last_span = 0;
    u32 span_count = 0;
    u32 span_next = 0;
    u16 lru_clock = 0;
    std::array<Span, kMaxSpans> spans{};
    std::array<TlbSlot, kSets * kWays> tlb{};

    static size_t SetIndex(u64 page2m) noexcept {
        u64 h = page2m * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 32;
        return static_cast<size_t>(h & (kSets - 1u));
    }
};
// Heap-allocated per thread to save ~1 KiB of static PT_TLS (FMOD threads use 64 KiB stacks).
thread_local std::unique_ptr<RegionCache> tls_region_cache;
const uint64_t* g_memory_generation_ptr = nullptr;

// V6.2+: Resolves the maximal contiguous mapped/unmapped span [start, end) containing `address`
// via an O(1) 1,024-entry 4-way 2 MiB TLB + 256-span table, preventing linear scans and FIFO
// rwlock thrashing even after map streaming fragments guest VMAs into >64 non-contiguous spans.
bool LookupMappedEnd(VAddr address, uintptr_t& out_end) {
    const uint64_t* gen_ptr = g_memory_generation_ptr;
    if (!gen_ptr) [[unlikely]] {
        gen_ptr = runtime_memory_generation();
        g_memory_generation_ptr = gen_ptr;
    }
    const u64 generation = __atomic_load_n(gen_ptr, __ATOMIC_ACQUIRE);
    if (generation & 1) {
        return false; // being changed
    }
    RegionCache* ptr = tls_region_cache.get();
    if (!ptr) [[unlikely]] {
        tls_region_cache = std::make_unique<RegionCache>();
        ptr = tls_region_cache.get();
    }
    auto& cache = *ptr;
    const u64 page2m = address >> 21;
    const size_t set_base = RegionCache::SetIndex(page2m) * RegionCache::kWays;

    auto insert_tlb = [&](u16 s_idx) noexcept {
        size_t victim = set_base;
        for (size_t w = 0; w < RegionCache::kWays; ++w) {
            const auto& sl = cache.tlb[set_base + w];
            if (sl.page2m == ~u64{0} || sl.page2m == page2m) {
                victim = set_base + w;
                break;
            }
            if (static_cast<u16>(cache.lru_clock - sl.lru) >
                static_cast<u16>(cache.lru_clock - cache.tlb[victim].lru)) {
                victim = set_base + w;
            }
        }
        cache.tlb[victim] = {page2m, s_idx, ++cache.lru_clock};
    };

    if (cache.generation == generation) {
        if (cache.span_count != 0) {
            const auto& mru = cache.spans[cache.last_span];
            if (address >= mru.start && address < mru.end) {
                if (!mru.mapped) {
                    return false;
                }
                out_end = mru.end;
                return true;
            }
            for (size_t w = 0; w < RegionCache::kWays; ++w) {
                auto& sl = cache.tlb[set_base + w];
                if (sl.page2m == page2m && sl.span_idx < cache.span_count) {
                    const auto& sp = cache.spans[sl.span_idx];
                    if (address >= sp.start && address < sp.end) {
                        sl.lru = ++cache.lru_clock;
                        cache.last_span = sl.span_idx;
                        if (!sp.mapped) {
                            return false;
                        }
                        out_end = sp.end;
                        return true;
                    }
                }
            }
            for (u32 i = 0; i < cache.span_count; ++i) {
                const auto& sp = cache.spans[i];
                if (address >= sp.start && address < sp.end) {
                    cache.last_span = i;
                    insert_tlb(static_cast<u16>(i));
                    if (!sp.mapped) {
                        return false;
                    }
                    out_end = sp.end;
                    return true;
                }
            }
        }
    } else {
        cache.generation = generation;
        cache.last_span = 0;
        cache.span_count = 0;
        cache.span_next = 0;
        cache.lru_clock = 0;
        cache.tlb.fill({});
    }

    uintptr_t start = 0, end = 0;
    int mapped = 0;
    if (!runtime_memory_region(address, &start, &end, &mapped) || end <= start) {
        return false;
    }
    if (!mapped) {
        if (__atomic_load_n(gen_ptr, __ATOMIC_ACQUIRE) == generation) {
            const u32 slot = (cache.span_count < RegionCache::kMaxSpans)
                                 ? cache.span_count++
                                 : (cache.span_next++ & (RegionCache::kMaxSpans - 1u));
            cache.spans[slot] = {start, end, 0u};
            cache.last_span = slot;
            insert_tlb(static_cast<u16>(slot));
        }
        return false;
    }
    uintptr_t contig_start = start;
    while (contig_start > 0) {
        uintptr_t ps = 0, pe = 0;
        int pm = 0;
        if (!runtime_memory_region(contig_start - 1, &ps, &pe, &pm) || !pm ||
            pe != contig_start || ps >= contig_start) {
            break;
        }
        contig_start = ps;
    }
    uintptr_t contig_end = end;
    while (contig_end < ~uintptr_t{0}) {
        uintptr_t ns = 0, ne = 0;
        int nm = 0;
        if (!runtime_memory_region(contig_end, &ns, &ne, &nm) || !nm ||
            ns != contig_end || ne <= contig_end) {
            break;
        }
        contig_end = ne;
    }
    if (__atomic_load_n(gen_ptr, __ATOMIC_ACQUIRE) != generation) {
        return false;
    }
    const u32 slot = (cache.span_count < RegionCache::kMaxSpans)
                         ? cache.span_count++
                         : (cache.span_next++ & (RegionCache::kMaxSpans - 1u));
    cache.spans[slot] = {contig_start, contig_end, 1u};
    cache.last_span = slot;
    insert_tlb(static_cast<u16>(slot));
    out_end = contig_end;
    return true;
}

} // namespace

extern "C" bool BbLookupMappedEnd(u64 address, uintptr_t& out_end) {
    return LookupMappedEnd(address, out_end);
}

namespace {

bool CachedMapped(VAddr address, u64 size) {
    uintptr_t contig_end = 0;
    return LookupMappedEnd(address, contig_end) && size <= (contig_end - address);
}
} // namespace

u64 MemoryManager::ClampRangeSize(VAddr virtual_addr, u64 size) {
    if (size == 0) {
        return 0;
    }
    uintptr_t contig_end = 0;
    if (LookupMappedEnd(virtual_addr, contig_end)) {
        const u64 avail = contig_end - virtual_addr;
        return size <= avail ? size : avail;
    }
    return runtime_memory_clamp(virtual_addr, size);
}
static void CopySparseSerial(VAddr source, u8* dest, u64 size) {
    // BB_READBACKS=2 protects GPU-written pages against reads: the copy threads and recorders
    // read through the backing view, as a read fault on them would wait for the GPU thread,
    // which waits for them.
    static const bool precise =
        EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Precise;
    if (precise) {
        runtime_memory_read_backing(source, dest, size);
        return;
    }
    if (size && CachedMapped(source, size)) {
        std::memcpy(dest, reinterpret_cast<const void*>(source), size);
        return;
    }
    while (size) {
        uintptr_t start = 0, end = 0;
        int mapped = 0;
        if (!runtime_memory_region(source, &start, &end, &mapped)) {
            end = source + size;
        }
        const u64 n = std::min<u64>(size, end - source);
        if (mapped) std::memcpy(dest, reinterpret_cast<const void*>(source), n);
        else std::memset(dest, 0, n);
        source += n; dest += n; size -= n;
    }
}
void MemoryManager::CopySparseMemory(VAddr source, u8* dest, u64 size) {
    // bbport: large uploads (streaming) are split across the copy threads.
    constexpr u64 Chunk = 512 * 1024;
    if (size < 4 * Chunk || !BbCopy::Enabled()) {
        return CopySparseSerial(source, dest, size);
    }
    BbCopy::ParallelFor((size + Chunk - 1) / Chunk, [&](std::size_t i) {
        const u64 offset = i * Chunk;
        CopySparseSerial(source + offset, dest + offset, std::min(Chunk, size - offset));
    });
}
void MemoryManager::ReadBacking(VAddr address, void* data, u64 size) {
    runtime_memory_read_backing(address, data, size);
}
bool MemoryManager::TryWriteBacking(void* address, const void* data, u64 size) {
    BbWriteLog::Note(reinterpret_cast<uintptr_t>(address), data, size, BbWriteLog::Backing);
    return runtime_memory_write_backing(reinterpret_cast<uintptr_t>(address), data, size) != 0;
}
void AddressSpace::Protect(VAddr virtual_addr, u64 size, MemoryPermission perms) {
    BbStats::Timer timer{BbStats::t_protect};
    const u64 pages = (size + 4095) / 4096;
    BbStats::protect_calls.fetch_add(1, std::memory_order_relaxed);
    BbStats::protect_pages.fetch_add(pages, std::memory_order_relaxed);
    if (!True(perms & MemoryPermission::Write)) {
        BbStats::protect_revoke_calls.fetch_add(1, std::memory_order_relaxed);
        BbStats::protect_revoke_pages.fetch_add(pages, std::memory_order_relaxed);
    }
    runtime_memory_gpu_protect(virtual_addr, size, True(perms & MemoryPermission::Read),
                               True(perms & MemoryPermission::Write));
}
boost::icl::interval_set<VAddr> AddressSpace::GetUsableRegions() {
    boost::icl::interval_set<VAddr> set;
    set += boost::icl::interval<VAddr>::right_open(0x10'0000'0000ULL, 0xfc'0000'0000ULL);
    return set;
}
} // namespace Core

// Kernel services the vendored libraries call directly.
namespace Libraries::Kernel {
u64 PS4_SYSV_ABI sceKernelGetTscFrequency() { return runtime_tsc_frequency(); }
u64 PS4_SYSV_ABI sceKernelReadTsc() { return Common::FencedRDTSC(); }
u64 PS4_SYSV_ABI sceKernelGetProcessTime() { return runtime_process_time_us(); }
u64 PS4_SYSV_ABI sceKernelGetProcessTimeCounter() { return runtime_process_time_counter(); }
u64 PS4_SYSV_ABI sceKernelGetProcessTimeCounterFrequency() { return 1'000'000'000; }
s32 PS4_SYSV_ABI sceKernelIsNeoMode() { return 0; } // base PS4
s32 PS4_SYSV_ABI sceKernelGetCompiledSdkVersion(s32* ver) {
    if (!ver) return BbErr::KernelInvalid;
    *ver = s32(g_sdk_version);
    return BbErr::Ok;
}
s32 PS4_SYSV_ABI sceKernelUsleep(u32 microseconds) {
    std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
    return BbErr::Ok;
}
} // namespace Libraries::Kernel

namespace {
// SDL video must be driven from one thread: the window lives on its own host thread.
std::thread g_window_thread;
std::mutex g_window_mutex;
std::condition_variable g_window_cv;
bool g_window_ready;
} // namespace

#ifdef BB_PGO_GENERATE
extern "C" void __gcov_dump(void);
extern "C" void __gcov_reset(void);
// Instrumented build (BB_PGO=generate): the game often ends through _exit (watchdog, guest
// exit), which skips the profile write at exit. Write every 30 s and reset: the files sum the
// intervals.
static void StartProfileWriter() {
    std::thread([] {
        Common::SetCurrentThreadName("bb:pgo");
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            __gcov_dump();
            __gcov_reset();
            std::printf("PGO: profile written\n");
        }
    }).detach();
}
#endif

extern "C" int bbgpu_init(const BbGpuConfig* config) {
    BbSettings::Load();
    // What the translator knows about this game (games/), before anything asks for it.
    Game::Select(config->serial);
#ifdef BB_PGO_GENERATE
    StartProfileWriter();
#endif
    g_sdk_version = config->sdk_version;
    if (config->user_dir) setenv("BB_GPU_USER_DIR", config->user_dir, 0);
    Common::ElfInfo::Instance().Set(config->serial ? config->serial : "UNKNOWN",
                                    config->title ? config->title : "", config->sdk_version,
                                    config->psf_attributes);
    const std::string title = config->title ? config->title : "Bloodborne";
    const s32 width = config->width, height = config->height;
    g_window_thread = std::thread([title, width, height] {
        Common::SetCurrentThreadName("bb:window");
        auto* window = new Frontend::WindowSDL(width, height, title.c_str());
        {
            std::scoped_lock lock{g_window_mutex};
            g_window = window;
            g_window_ready = true;
        }
        g_window_cv.notify_all();
        while (window->PollEvents()) {
            SDL_Delay(2);
        }
        LOG_INFO(Frontend, "Window closed by user");
        std::fflush(stdout);
        std::_Exit(0);
    });
    g_window_thread.detach();
    {
        std::unique_lock lock{g_window_mutex};
        g_window_cv.wait(lock, [] { return g_window_ready; });
    }
    Core::Loader::SymbolsResolver resolver;
    // GnmDriver creates the presenter that the VideoOut present thread uses.
    BbGnm::RegisterLib(&resolver);
    BbVideoOut::RegisterLib(&resolver);
    return 0;
}

extern "C" void bbgpu_register_kernel(void) {
    Core::Loader::SymbolsResolver resolver;
    BbKernel::RegisterEventQueue(&resolver);
    BbAvPlayer::RegisterLib(&resolver);
}

extern "C" uintptr_t bbgpu_resolve(const char* scoped_nid) {
    const char* hash = std::strchr(scoped_nid, '#');
    const size_t length = hash ? size_t(hash - scoped_nid) : std::strlen(scoped_nid);
    for (const auto& symbol : g_symbols) {
        if (symbol.nid.size() == length && !std::memcmp(symbol.nid.data(), scoped_nid, length)) {
            return uintptr_t(symbol.address);
        }
    }
    return 0;
}

extern "C" int bbgpu_handle_fault(void* ucontext, void* address) {
    // BB_LABEL_TRAP (diagnostic): its read-only label pages first. Not passed on to GPU page
    // tracking (a write fault drains the draw pipe: thousands a second would change the timing
    // under test); label pages are not expected to be GPU-tracked.
    if (BbFreeCheck::OnTrapFault(ucontext, reinterpret_cast<std::uint64_t>(address))) {
        return 1;
    }
    if (Core::Signals::Instance()->DispatchAccessViolation(ucontext, address)) {
        return 1;
    }
    return BbFreeCheck::OnStaleTrapFault(reinterpret_cast<std::uint64_t>(address)) ? 1 : 0;
}

extern "C" void bbgpu_patch_image(unsigned char* image, uint64_t size) {
    BbGnmHooks::PatchImage(image, size);
    BbGameMenu::PatchImage(image, size);
    BbFrameRate::PatchImage(image, size);
    BbGoldenEngine::PatchImage(image, size);
}

extern "C" unsigned bbgpu_symbol_count(void) {
    return unsigned(g_symbols.size());
}

namespace Libraries::Kernel {
int* PS4_SYSV_ABI __Error() {
    return runtime_errno();
}
} // namespace Libraries::Kernel

extern "C" int bbgpu_overlay_captures_input(void) {
    return (BbOverlay::CapturesInput() || BbGameMenu::CapturesInput()) ? 1 : 0;
}

extern "C" void bbgpu_mouse_take(double* dx, double* dy, int* wheel_up, int* wheel_down) {
    double x = 0, y = 0;
    int up = 0, down = 0;
    if (g_window) {
        g_window->TakeMouse(x, y, up, down);
    }
    *dx = x;
    *dy = y;
    *wheel_up = up;
    *wheel_down = down;
}

extern "C" int bbgpu_mouse_captured(void) {
    return g_window && g_window->MouseCaptured() ? 1 : 0;
}

extern "C" void bbgpu_mouse_capture(void) {
    if (g_window) {
        g_window->RequestMouseCapture();
    }
}

extern "C" uint64_t bbgpu_camera_state(void) {
    return BbGuestHooks::CameraState();
}

extern "C" int bbgpu_free_camera(void) {
    return BbGuestHooks::FreeCameraActive() ? 1 : 0;
}

extern "C" int bbgpu_camera_look(float yaw, float pitch) {
    if (!BbGuestHooks::CameraLookAvailable()) {
        return 0;
    }
    return BbGuestHooks::CameraLookAdd(pitch, yaw) ? 1 : 0;
}

extern "C" void bbgpu_mouse_look_enable(int enabled) {
    Frontend::WindowSDL::EnableMouseLook(enabled != 0);
}

extern "C" int bbgpu_text_input_begin(const char* initial, const char* prompt) {
    if (!g_window) return 0;
    g_window->BeginTextInput(initial ? initial : "", prompt ? prompt : "Text");
    return 1;
}

extern "C" int bbgpu_text_input_poll(char* out, uint64_t size) {
    if (!g_window) return 2;
    std::string text;
    const int state = g_window->PollTextInput(text);
    if (size) {
        const size_t n = std::min<size_t>(text.size(), size - 1);
        std::memcpy(out, text.data(), n);
        out[n] = 0;
    }
    return state;
}

extern "C" unsigned long long bbgpu_world_us(void) {
    return static_cast<unsigned long long>(runtime_process_time_us());
}

extern "C" void bbgpu_trophy_setup(const char* sce_sys_host_dir, const char* user_dir) {
    (void)sce_sys_host_dir;
    (void)user_dir;
}

extern "C" int bbgpu_trophy_unlock(int id) {
    (void)id;
    return -1;
}

extern "C" int bbgpu_rebuild_shader_caches(void) {
    return 0;
}

extern "C" int bbgpu_rebuild_shader_cache(const char* dir) {
    (void)dir;
    return 0;
}

extern "C" int bbgpu_predict_shaders_test(const char* cache_dir, const char* pack,
                                          const char* game_dir, const char* out_dir, int jobs) {
    (void)cache_dir; (void)pack; (void)game_dir; (void)out_dir; (void)jobs;
    return 0;
}

extern "C" int bbgpu_merge_shader_metas(const char* cache_dir, const char* pack,
                                        const char* game_dir, const char* meta_dir,
                                        const char* out_pack, const char* work_dir, int jobs) {
    (void)cache_dir; (void)pack; (void)game_dir; (void)meta_dir;
    (void)out_pack; (void)work_dir; (void)jobs;
    return 0;
}

extern "C" int bbgpu_meta_reconstruct_test(const char* cache_dir, const char* pack,
                                           const char* game_dir) {
    (void)cache_dir; (void)pack; (void)game_dir;
    return 0;
}

extern "C" int bbgpu_merge_shader_packs(const char* cache_dir, const char* pack_a,
                                        const char* pack_b, const char* game_dir,
                                        const char* out_pack, const char* work_dir, int jobs) {
    (void)cache_dir; (void)pack_a; (void)pack_b; (void)game_dir;
    (void)out_pack; (void)work_dir; (void)jobs;
    return 0;
}

extern "C" int bbgpu_export_shader_recipes(const char* cache_dir, const char* game_dir,
                                           const char* out_path) {
    (void)cache_dir; (void)game_dir; (void)out_path;
    return 0;
}

extern "C" int bbgpu_import_shader_recipes(const char* cache_dir, const char* pack,
                                           const char* game_dir, unsigned long long motion_params,
                                           unsigned long long motion_positions) {
    (void)cache_dir; (void)pack; (void)game_dir; (void)motion_params; (void)motion_positions;
    return 0;
}

extern "C" int bbgpu_gpu_touched(std::uint64_t va) {
    (void)va;
    return 0;
}

extern "C" void bbgpu_straddle_report() {}
