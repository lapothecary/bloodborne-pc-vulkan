// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// bb_vulkan: Universal Linux & Windows Engine Crash Black-Box Reporter
// File: gpu/shim/bbport_crash_blackbox.cpp
// ============================================================================
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "bbport_crash_blackbox.h"

#include "bb_env.h"
#include "bb_golden_engine.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#endif

#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <ucontext.h>
#include <unistd.h>

#ifndef _WIN32
#include <cxxabi.h>
#include <sys/syscall.h>
#endif

extern "C" {
std::uint64_t runtime_process_time_us(void);
const char* runtime_file_user_dir(void);
int runtime_savedata_active_slot(char* dir, std::size_t size);
void runtime_set_flight_recorder(void (*fn)(const char* category, const char* detail));
#ifdef _WIN32
void bb_win32_describe_address(const void* address, char* out, unsigned out_size);
#endif
}

namespace BbCrashBlackBox {
namespace {

using namespace DantelionHeap;

// ============================================================================
// Zero-TLS Static BSS State (0 bytes added to PT_TLS)
// ============================================================================
constexpr std::size_t kFlightRingCap = 128; // Must be power of 2

struct alignas(64) FlightEntry {
    std::atomic<std::uint64_t> seq{0}; // Odd = writing, Even (non-zero) = committed
    std::uint64_t time_us{0};
    std::uint32_t frame{0};
    std::uint32_t tid{0};
    char thread[20]{};
    char category[16]{};
    char detail[168]{};
};

FlightEntry g_flight_ring[kFlightRingCap]{};
alignas(64) std::atomic<std::uint64_t> g_flight_write_idx{0};

std::atomic<const unsigned char*> g_guest_image{nullptr};
std::atomic<std::uint64_t> g_guest_image_size{0};

std::atomic<std::uint32_t> g_frame_counter{0};
std::atomic<bool> g_has_player_pos{false};
std::atomic<float> g_player_x{0.0f};
std::atomic<float> g_player_y{0.0f};
std::atomic<float> g_player_z{0.0f};

std::atomic<bool> g_initialized{false};
std::atomic<bool> g_crash_in_progress{false};
std::atomic<std::uint32_t> g_crash_owner_tid{0};

#ifndef _WIN32
// Static demangle scratch buffer (only touched while g_crash_in_progress == true)
char g_demangle_scratch[1024]{};
#endif

// ============================================================================
// Low-Level Platform Helpers (Non-Faulting Memory Read & Dual File/Stderr Sink)
// ============================================================================
std::uint32_t CurrentThreadIdFast() noexcept {
#ifdef _WIN32
    return static_cast<std::uint32_t>(::GetCurrentThreadId());
#elif defined(__linux__) && defined(SYS_gettid)
    return static_cast<std::uint32_t>(::syscall(SYS_gettid));
#else
    return 0u;
#endif
}

void CurrentThreadNameSafe(char* out, std::size_t out_size) noexcept {
    if (!out || out_size == 0) return;
    out[0] = '\0';
    if (::pthread_getname_np(::pthread_self(), out, out_size) != 0 || out[0] == '\0') {
        std::snprintf(out, out_size, "tid-%u", CurrentThreadIdFast());
    }
    out[out_size - 1] = '\0';
}

bool SafeReadMem(std::uintptr_t remote_addr, void* dst, std::size_t len) noexcept {
    if (len == 0) return true;
    if (remote_addr < 0x1000u || !dst) return false;
    // Reject non-canonical x86-64 user addresses before syscall
    const std::uint64_t top16 = static_cast<std::uint64_t>(remote_addr) >> 48;
    if (top16 != 0ULL) return false;

#ifdef _WIN32
    SIZE_T done = 0;
    if (!::ReadProcessMemory(::GetCurrentProcess(), reinterpret_cast<LPCVOID>(remote_addr),
                             dst, len, &done) ||
        done != len) {
        return false;
    }
    return true;
#else
    struct iovec local_iov{dst, len};
    struct iovec remote_iov{reinterpret_cast<void*>(remote_addr), len};
    const ssize_t n = ::process_vm_readv(::getpid(), &local_iov, 1, &remote_iov, 1, 0);
    return n == static_cast<ssize_t>(len);
#endif
}

template <typename T>
bool SafeReadVal(std::uintptr_t addr, T* out) noexcept {
    return SafeReadMem(addr, out, sizeof(T));
}

// Static extra reason string populated by std::terminate / Win32 UnhandledExceptionFilter
char g_extra_reason[512]{};

#ifdef _WIN32
void Utf8ToWideStatic(const char* utf8, wchar_t* out, int out_cap) noexcept {
    if (!out || out_cap <= 0) return;
    out[0] = L'\0';
    if (!utf8 || utf8[0] == '\0') return;
    const int n = ::MultiByteToWideChar(CP_UTF8, 0, utf8, -1, out, out_cap);
    if (n <= 0) {
        // Fallback to ANSI if invalid UTF-8
        ::MultiByteToWideChar(CP_ACP, 0, utf8, -1, out, out_cap);
    }
    out[out_cap - 1] = L'\0';
}

void DescribeWin32ModuleOffset(const void* addr, char* out, std::size_t out_cap) noexcept {
    if (!out || out_cap == 0) return;
    out[0] = '\0';
    if (!addr) return;
    HMODULE mod = nullptr;
    if (::GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            static_cast<LPCSTR>(addr), &mod) &&
        mod) {
        char mod_path[512]{};
        if (::GetModuleFileNameA(mod, mod_path, static_cast<DWORD>(sizeof(mod_path))) > 0) {
            const char* slash = std::strrchr(mod_path, '\\');
            const char* fslash = std::strrchr(mod_path, '/');
            const char* base = (slash && (!fslash || slash > fslash)) ? (slash + 1)
                                                                      : (fslash ? (fslash + 1) : mod_path);
            const auto off = reinterpret_cast<std::uintptr_t>(addr) -
                             reinterpret_cast<std::uintptr_t>(mod);
            std::snprintf(out, out_cap, "%s+0x%llx", base, static_cast<unsigned long long>(off));
        }
    }
}
#endif

struct CrashSink {
#ifdef _WIN32
    HANDLE file_handle = INVALID_HANDLE_VALUE;
#else
    int file_fd = -1;
#endif
    char file_path[1024]{};

    void OpenCrashLogFile() noexcept {
        static char logs_dir[896]{};
        logs_dir[0] = '\0';
        file_path[0] = '\0';
        const char* data_dir = bb_env("BB_DATA_DIR");
        if (data_dir && data_dir[0] != '\0') {
            std::snprintf(logs_dir, sizeof(logs_dir), "%s/logs", data_dir);
        } else {
            const char* udir = runtime_file_user_dir();
            if (udir && udir[0] != '\0') {
                const std::size_t ulen = std::strlen(udir);
                if (ulen >= 5 &&
                    (std::strcmp(udir + ulen - 5, "/user") == 0 ||
                     std::strcmp(udir + ulen - 5, "\\user") == 0)) {
                    std::snprintf(logs_dir, sizeof(logs_dir), "%.*s/logs",
                                  static_cast<int>(ulen - 5), udir);
                }
            }
            if (logs_dir[0] == '\0') {
                std::snprintf(logs_dir, sizeof(logs_dir), "logs");
            }
        }

        const std::time_t now = std::time(nullptr);
        struct tm tm_buf{};
#ifdef _WIN32
        ::localtime_s(&tm_buf, &now);
#else
        ::localtime_r(&now, &tm_buf);
#endif
        char stamp[32]{};
        std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm_buf);
        std::snprintf(file_path, sizeof(file_path), "%s/crash-%s.log", logs_dir, stamp);

#ifdef _WIN32
        static wchar_t wdir[1024]{};
        static wchar_t wpath[1024]{};
        Utf8ToWideStatic(logs_dir, wdir, 1024);
        ::CreateDirectoryW(wdir, nullptr);
        Utf8ToWideStatic(file_path, wpath, 1024);
        file_handle = ::CreateFileW(wpath, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                    nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle == INVALID_HANDLE_VALUE) {
            // Fallback to current working directory `./logs`
            ::CreateDirectoryW(L"logs", nullptr);
            std::snprintf(file_path, sizeof(file_path), "logs/crash-%s.log", stamp);
            Utf8ToWideStatic(file_path, wpath, 1024);
            file_handle = ::CreateFileW(wpath, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                        nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        }
#else
        ::mkdir(logs_dir, 0755);
        file_fd = ::open(file_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (file_fd < 0) {
            ::mkdir("logs", 0755);
            std::snprintf(file_path, sizeof(file_path), "logs/crash-%s.log", stamp);
            file_fd = ::open(file_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        }
#endif
    }

    void WriteRaw(const char* data, std::size_t len) noexcept {
        if (!data || len == 0) return;
#ifdef _WIN32
        ::_write(2, data, static_cast<unsigned int>(len));
        if (file_handle != INVALID_HANDLE_VALUE) {
            DWORD written = 0;
            ::WriteFile(file_handle, data, static_cast<DWORD>(len), &written, nullptr);
        }
#else
        const ssize_t w1 = ::write(2, data, len);
        (void)w1;
        if (file_fd >= 0) {
            const ssize_t w2 = ::write(file_fd, data, len);
            (void)w2;
        }
#endif
    }

    void Printf(const char* fmt, ...) noexcept
#if defined(__GNUC__) || defined(__clang__)
        __attribute__((format(printf, 2, 3)))
#endif
    {
        static char buf[1536];
        va_list ap;
        va_start(ap, fmt);
        const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        if (n > 0) {
            const std::size_t len = std::min<std::size_t>(static_cast<std::size_t>(n), sizeof(buf) - 1);
            WriteRaw(buf, len);
        }
    }

    void FlushOnly() noexcept {
#ifdef _WIN32
        if (file_handle != INVALID_HANDLE_VALUE) {
            ::FlushFileBuffers(file_handle);
        }
#else
        if (file_fd >= 0) {
            ::fsync(file_fd);
        }
#endif
    }

    void FlushAndClose() noexcept {
#ifdef _WIN32
        if (file_handle != INVALID_HANDLE_VALUE) {
            ::FlushFileBuffers(file_handle);
            ::CloseHandle(file_handle);
            file_handle = INVALID_HANDLE_VALUE;
        }
#else
        if (file_fd >= 0) {
            ::fsync(file_fd);
            ::close(file_fd);
            file_fd = -1;
        }
#endif
    }
};

const char* BaseNameOf(const char* path) noexcept {
    if (!path) return "?";
    const char* slash = std::strrchr(path, '/');
    const char* bslash = std::strrchr(path, '\\');
    const char* p = (bslash && (!slash || bslash > slash)) ? (bslash + 1)
                                                           : (slash ? (slash + 1) : path);
    return *p ? p : path;
}

// ============================================================================
// Windows Dynamic DbgHelp Bridge (Optional .pdb / COFF / Export Symbol Lookup)
// ============================================================================
#ifdef _WIN32
struct Win32DbgHelp {
    struct SymInfoRaw {
        ULONG SizeOfStruct;
        ULONG TypeIndex;
        ULONG64 Reserved[2];
        ULONG Index;
        ULONG Size;
        ULONG64 ModBase;
        ULONG Flags;
        ULONG64 Value;
        ULONG64 Address;
        ULONG Register;
        ULONG Scope;
        ULONG Tag;
        ULONG NameLen;
        ULONG MaxNameLen;
        CHAR Name[512];
    };
    struct Line64Raw {
        DWORD SizeOfStruct;
        PVOID Key;
        DWORD LineNumber;
        PCHAR FileName;
        DWORD64 Address;
    };

    using FnSymSetOptions = DWORD(WINAPI*)(DWORD);
    using FnSymInitialize = BOOL(WINAPI*)(HANDLE, PCSTR, BOOL);
    using FnSymFromAddr = BOOL(WINAPI*)(HANDLE, DWORD64, PDWORD64, SymInfoRaw*);
    using FnSymGetLineFromAddr64 = BOOL(WINAPI*)(HANDLE, DWORD64, PDWORD, Line64Raw*);

    bool ready = false;
    FnSymFromAddr sym_from_addr = nullptr;
    FnSymGetLineFromAddr64 sym_get_line = nullptr;

    void InitOnce() noexcept {
        static bool attempted = false;
        if (attempted) return;
        attempted = true;
        HMODULE mod = ::LoadLibraryA("dbghelp.dll");
        if (!mod) return;
        auto set_opts = reinterpret_cast<FnSymSetOptions>(::GetProcAddress(mod, "SymSetOptions"));
        auto sym_init = reinterpret_cast<FnSymInitialize>(::GetProcAddress(mod, "SymInitialize"));
        sym_from_addr = reinterpret_cast<FnSymFromAddr>(::GetProcAddress(mod, "SymFromAddr"));
        sym_get_line = reinterpret_cast<FnSymGetLineFromAddr64>(::GetProcAddress(mod, "SymGetLineFromAddr64"));
        if (!sym_init || !sym_from_addr) return;
        if (set_opts) {
            // SYMOPT_UNDNAME (0x02) | SYMOPT_DEFERRED_LOADS (0x04) | SYMOPT_LOAD_LINES (0x10)
            set_opts(0x02u | 0x04u | 0x10u);
        }
        ready = (sym_init(::GetCurrentProcess(), nullptr, TRUE) != FALSE);
    }

    bool Resolve(std::uintptr_t addr, char* out, std::size_t out_size) noexcept {
        InitOnce();
        if (!ready || !sym_from_addr) return false;
        SymInfoRaw info{};
        info.SizeOfStruct = sizeof(SymInfoRaw) - sizeof(info.Name) + 1;
        // SYMBOL_INFO on x64 is 88 bytes before flexible Name[1]
        info.SizeOfStruct = 88u;
        info.MaxNameLen = sizeof(info.Name) - 1;
        DWORD64 disp = 0;
        if (!sym_from_addr(::GetCurrentProcess(), static_cast<DWORD64>(addr), &disp, &info) ||
            info.Name[0] == '\0') {
            return false;
        }
        info.Name[sizeof(info.Name) - 1] = '\0';
        Dl_info dli{};
        const char* mod_name = (dladdr(reinterpret_cast<const void*>(addr), &dli) && dli.dli_fname)
                                   ? BaseNameOf(dli.dli_fname)
                                   : "host";
        if (sym_get_line) {
            Line64Raw line{};
            line.SizeOfStruct = sizeof(Line64Raw);
            DWORD line_disp = 0;
            if (sym_get_line(::GetCurrentProcess(), static_cast<DWORD64>(addr), &line_disp, &line) &&
                line.FileName) {
                std::snprintf(out, out_size, "%s!%s+0x%llx (%s:%lu)",
                              mod_name, info.Name, static_cast<unsigned long long>(disp),
                              BaseNameOf(line.FileName), static_cast<unsigned long>(line.LineNumber));
                return true;
            }
        }
        std::snprintf(out, out_size, "%s!%s+0x%llx",
                      mod_name, info.Name, static_cast<unsigned long long>(disp));
        return true;
    }
};
Win32DbgHelp g_win32_dbghelp{};
#endif

// ============================================================================
// Known Engine Guest Subroutines (for instant human-readable eboot.bin symbols)
// ============================================================================
struct KnownGuestRange {
    std::uint64_t begin_vma;
    std::uint64_t end_vma;
    const char* name;
};

constexpr KnownGuestRange kKnownGuestRanges[] = {
    {0x0143ab60ULL, 0x0143ed80ULL, "ChrExFollowCam::Update"},
    {0x01fc58e0ULL, 0x01fc599fULL, "DLDefaultHeapStrategy::Contains"},
    {0x01fc5ac0ULL, 0x01fc5c54ULL, "DLHeapObject::~DLHeapObject"},
    {0x01fc5fc0ULL, 0x01fc6010ULL, "DLDefaultHeapStrategy::Free"},
    {0x0207b720ULL, 0x0207b8bfULL, "DLHeapManager::FindAllocator"},
    {0x0207bbf0ULL, 0x0207bc90ULL, "DLHeapManager::ResolveHeapForPtr"},
    {0x02085350ULL, 0x020856bfULL, "DLRegularHeap::allocate"},
    {0x020856c0ULL, 0x0208589fULL, "DLRegularHeap::getBlockSize"},
    {0x020858a0ULL, 0x02085a4fULL, "DLRegularHeap::free"},
    {0x0263b810ULL, 0x0263b9afULL, "CGFixedBlockPool::Allocate"},
    {0x0263b9b0ULL, 0x0263ba3fULL, "CGFixedBlockPool::Free"},
    {0x0263bc40ULL, 0x0263bcf0ULL, "CGFixedBlockPool::Finalize"},
};

const KnownGuestRange* LookupKnownGuestRange(std::uint64_t vma) noexcept {
    for (const auto& r : kKnownGuestRanges) {
        if (vma >= r.begin_vma && vma < r.end_vma) {
            return &r;
        }
    }
    return nullptr;
}

// ============================================================================
// Unified Host + Guest Code Address Symbolizer (Linux + Windows, 0 .pdb needed)
// ============================================================================
bool ResolveCodeSymbol(std::uintptr_t addr, const unsigned char* guest_img,
                       std::uint64_t guest_size, char* out, std::size_t out_size) noexcept {
    if (!out || out_size == 0 || addr < 0x1000u) return false;
    out[0] = '\0';

    const auto img_base = reinterpret_cast<std::uintptr_t>(guest_img);
    const std::uint64_t eff_size = guest_size ? guest_size : 0x10000000ULL;

    // 1. Inside mapped PS4 guest image (`eboot.bin` + linked `libc.prx` / system modules)
    if (img_base != 0 && addr >= img_base && (addr - img_base) < eff_size) {
        const std::uint64_t vma = static_cast<std::uint64_t>(addr - img_base);
        BbGoldenEngine::HostSymbolMatch gx{};
        if (BbGoldenEngine::ResolveGuestVma(vma, &gx)) {
            std::snprintf(out, out_size, "eboot.bin+0x%llx (%s+0x%llx [%s])",
                          static_cast<unsigned long long>(vma), gx.name,
                          static_cast<unsigned long long>(gx.host_offset), gx.module_name);
            return true;
        }
        if (const auto* kr = LookupKnownGuestRange(vma)) {
            std::snprintf(out, out_size, "eboot.bin+0x%llx (%s+0x%llx)",
                          static_cast<unsigned long long>(vma), kr->name,
                          static_cast<unsigned long long>(vma - kr->begin_vma));
            return true;
        }
        if (vma >= 0x05800000ULL) {
            std::snprintf(out, out_size, "guest_module+0x%llx (libc.prx / linked PRX)",
                          static_cast<unsigned long long>(vma));
        } else {
            std::snprintf(out, out_size, "eboot.bin+0x%llx", static_cast<unsigned long long>(vma));
        }
        return true;
    }

    // 2. Inside decompiled C++20 `dantelion_gx` (`2,941` functions + `BbSafeFixedBlockPool_*`)
    BbGoldenEngine::HostSymbolMatch gx{};
    if (BbGoldenEngine::ResolveHostAddress(addr, &gx)) {
        std::snprintf(out, out_size, "dantelion_gx!%s+0x%llx [%s, PS4 +0x%llx]%s",
                      gx.name, static_cast<unsigned long long>(gx.host_offset),
                      gx.module_name, static_cast<unsigned long long>(gx.ps4_vma),
                      gx.via_near_stub ? " (near-stub)" : "");
        return true;
    }

    // 3. Host shared libraries / executable (`libbbgpu`, `bb-probe`, Vulkan driver, system DLLs)
#ifdef _WIN32
    if (g_win32_dbghelp.Resolve(addr, out, out_size)) {
        return true;
    }
#endif
    Dl_info dli{};
    if (dladdr(reinterpret_cast<const void*>(addr), &dli) && dli.dli_fname) {
        const char* base = BaseNameOf(dli.dli_fname);
        const auto mod_off = static_cast<unsigned long long>(
            addr - reinterpret_cast<std::uintptr_t>(dli.dli_fbase));
        if (dli.dli_sname && dli.dli_sname[0] != '\0') {
            const char* sym_name = dli.dli_sname;
#ifndef _WIN32
            int status = -1;
            std::size_t len = sizeof(g_demangle_scratch);
            if (char* dem = abi::__cxa_demangle(dli.dli_sname, g_demangle_scratch, &len, &status);
                status == 0 && dem == g_demangle_scratch) {
                sym_name = g_demangle_scratch;
            }
#endif
            const auto sym_off = static_cast<unsigned long long>(
                addr - reinterpret_cast<std::uintptr_t>(dli.dli_saddr));
            std::snprintf(out, out_size, "%s!%s+0x%llx (mod+0x%llx)",
                          base, sym_name, sym_off, mod_off);
        } else {
            std::snprintf(out, out_size, "%s+0x%llx", base, mod_off);
        }
        return true;
    }
    return false;
}

// Heuristic check forRSP stack scanning: does `ret_addr` lie right after a valid x86-64 `call`?
bool LooksLikeReturnAddress(std::uintptr_t ret_addr, const unsigned char* guest_img,
                            std::uint64_t guest_size) noexcept {
    if (ret_addr < 0x10000u || (ret_addr >> 48) != 0) return false;

    const auto img_base = reinterpret_cast<std::uintptr_t>(guest_img);
    const std::uint64_t eff_size = guest_size ? guest_size : 0x10000000ULL;
    const bool in_guest = (img_base != 0 && ret_addr >= img_base && (ret_addr - img_base) < eff_size);

    Dl_info dli{};
    BbGoldenEngine::HostSymbolMatch gx{};
    const bool in_gx = !in_guest && BbGoldenEngine::ResolveHostAddress(ret_addr, &gx);
    const bool in_host_mod = !in_guest && !in_gx &&
                             (dladdr(reinterpret_cast<const void*>(ret_addr), &dli) != 0 &&
                              dli.dli_fname != nullptr);
    if (!in_guest && !in_gx && !in_host_mod) {
        return false;
    }

    // Verify preceding bytes at [ret_addr - 8 .. ret_addr) match a common x86-64 CALL encoding:
    //   - E8 rel32              (5 bytes: buf[3] == 0xE8)
    //   - FF /2                 (2..7 bytes: call r/m64, ModRM reg == 2 -> (modrm & 0x38) == 0x10)
    //   - REX (40..4F) + FF /2
    std::uint8_t prev[8]{};
    if (!SafeReadMem(ret_addr - 8u, prev, sizeof(prev))) {
        return false;
    }
    if (prev[3] == 0xE8u) return true;                         // call rel32
    if (prev[6] == 0xFFu && (prev[7] & 0x38u) == 0x10u) return true; // 2-byte call *reg / *(reg)
    if (prev[5] == 0xFFu && (prev[6] & 0x38u) == 0x10u) return true; // 3-byte call disp8/SIB
    if (prev[4] == 0xFFu && (prev[5] & 0x38u) == 0x10u) return true; // 4-byte call
    if (prev[2] == 0xFFu && (prev[3] & 0x38u) == 0x10u) return true; // 6-byte call *disp32(%rip/%reg)
    if (prev[1] == 0xFFu && (prev[2] & 0x38u) == 0x10u) return true; // 7-byte REX + call disp32
    return false;
}

// ============================================================================
// Sealed FromSoftware Heap Snapshot & Chunk Corruption Analyzer
// ============================================================================
struct DiscoveredHeap {
    std::uint64_t allocator = 0;
    std::uint64_t heap_impl = 0;
    std::uint64_t reg_heap_addr = 0;
    std::uint64_t arena_start = 0;
    std::uint64_t arena_end = 0;
    std::uint64_t first_chunk = 0;
    std::uint64_t end_sentinel = 0;
    std::uint64_t used_bytes = 0;
    std::uint64_t alloc_count = 0;
    std::uint32_t heap_id = 0;
    bool is_regular_heap = false;
    char name[48]{};
};

constexpr std::size_t kMaxDiscoveredHeaps = 48;

void ReadHeapNameSafe(std::uintptr_t name_ptr, char* out, std::size_t out_size) noexcept {
    if (!out || out_size == 0) return;
    out[0] = '\0';
    if (name_ptr < 0x10000u) return;

    std::uint8_t raw[96]{};
    // Read in 32-byte steps in case the string sits near a page end
    std::size_t avail = 0;
    for (std::size_t chunk : {std::size_t(96), std::size_t(32), std::size_t(16)}) {
        if (SafeReadMem(name_ptr, raw, chunk)) {
            avail = chunk;
            break;
        }
    }
    if (avail < 4) return;

    // Detect UTF-16LE (printable ASCII char in even byte, 0x00 in odd byte)
    if (raw[0] >= 0x20u && raw[0] < 0x7Fu && raw[1] == 0x00u &&
        (raw[2] == 0x00u || (raw[2] >= 0x20u && raw[2] < 0x7Fu && raw[3] == 0x00u))) {
        std::size_t w = 0;
        for (std::size_t i = 0; i + 1 < avail && w + 1 < out_size; i += 2) {
            const std::uint16_t ch = static_cast<std::uint16_t>(raw[i]) |
                                     (static_cast<std::uint16_t>(raw[i + 1]) << 8);
            if (ch == 0) break;
            out[w++] = (ch >= 0x20u && ch < 0x7Fu) ? static_cast<char>(ch) : '?';
        }
        out[w] = '\0';
        return;
    }

    // Standard UTF-8 / ASCII printable string
    if (raw[0] >= 0x20u && raw[0] < 0x7Fu) {
        std::size_t w = 0;
        for (std::size_t i = 0; i < avail && w + 1 < out_size; ++i) {
            if (raw[i] == 0) break;
            if (raw[i] < 0x20u || raw[i] >= 0x7Fu) {
                w = 0;
                break;
            }
            out[w++] = static_cast<char>(raw[i]);
        }
        out[w] = '\0';
    }
}

std::size_t EnumerateFromSoftwareHeaps(const unsigned char* guest_img,
                                       DiscoveredHeap (&out_heaps)[kMaxDiscoveredHeaps]) noexcept {
    if (!guest_img) return 0;
    const auto img_base = reinterpret_cast<std::uintptr_t>(guest_img);

    std::uint64_t gx_main_alloc = 0;
    SafeReadVal(img_base + kGXMainAllocatorGlobalVma, &gx_main_alloc);

    std::uint64_t mgr_ptr = 0;
    if (!SafeReadVal(img_base + kDLHeapManagerGlobalVma, &mgr_ptr) || mgr_ptr < 0x10000u) {
        return 0;
    }
    std::uint64_t vec_ptr = 0;
    if (!SafeReadVal(static_cast<std::uintptr_t>(mgr_ptr + kDLHeapManagerVecOffset), &vec_ptr) ||
        vec_ptr < 0x10000u) {
        return 0;
    }
    DLHeapArenaVector vec{};
    if (!SafeReadVal(static_cast<std::uintptr_t>(vec_ptr), &vec) ||
        vec.begin_ptr < 0x10000u || vec.end_ptr < vec.begin_ptr) {
        return 0;
    }

    const std::uint64_t byte_span = vec.end_ptr - vec.begin_ptr;
    const std::size_t count = std::min<std::size_t>(
        static_cast<std::size_t>(byte_span / sizeof(DLHeapArenaEntry)), kMaxDiscoveredHeaps);

    std::size_t n_valid = 0;
    for (std::size_t i = 0; i < count; ++i) {
        DLHeapArenaEntry entry{};
        if (!SafeReadVal(static_cast<std::uintptr_t>(vec.begin_ptr + i * sizeof(DLHeapArenaEntry)),
                         &entry)) {
            break;
        }
        if (entry.arena_start == 0 || entry.arena_end <= entry.arena_start) {
            continue;
        }
        DiscoveredHeap& h = out_heaps[n_valid++];
        h = DiscoveredHeap{};
        h.allocator = entry.allocator;
        h.arena_start = entry.arena_start;
        h.arena_end = entry.arena_end;

        // Inspect DLDefaultHeapStrategy (`*(allocator + 0x08) == heap_impl`)
        std::uint64_t impl_ptr = 0;
        if (entry.allocator >= 0x10000u &&
            SafeReadVal(static_cast<std::uintptr_t>(entry.allocator + 0x08u), &impl_ptr) &&
            impl_ptr >= 0x10000u) {
            h.heap_impl = impl_ptr;
            SafeReadVal(static_cast<std::uintptr_t>(impl_ptr + kImplHeapIdOffset), &h.heap_id);

            std::uint64_t name_ptr = 0;
            if (SafeReadVal(static_cast<std::uintptr_t>(impl_ptr + kImplHeapNamePtrOffset), &name_ptr)) {
                ReadHeapNameSafe(static_cast<std::uintptr_t>(name_ptr), h.name, sizeof(h.name));
            }

            // Read embedded DLRegularHeap header (first 0x28 bytes before bins[64])
            struct RegularHeadCompact {
                std::uint64_t bitmap;
                std::uint64_t first_chunk;
                std::uint64_t end_sentinel;
                std::uint64_t used_bytes;
                std::uint64_t alloc_count;
            } reg{};
            if (SafeReadVal(static_cast<std::uintptr_t>(impl_ptr + kImplRegularHeapOffset), &reg)) {
                if (reg.first_chunk >= entry.arena_start &&
                    reg.first_chunk < entry.arena_end &&
                    reg.end_sentinel > reg.first_chunk &&
                    reg.end_sentinel <= entry.arena_end + 0x1000u) {
                    h.is_regular_heap = true;
                    h.reg_heap_addr = impl_ptr + kImplRegularHeapOffset;
                    h.first_chunk = reg.first_chunk;
                    h.end_sentinel = reg.end_sentinel;
                    h.used_bytes = reg.used_bytes;
                    h.alloc_count = reg.alloc_count;
                }
            }
        }
        if (h.name[0] == '\0') {
            if (gx_main_alloc != 0 && entry.allocator == gx_main_alloc) {
                std::snprintf(h.name, sizeof(h.name), "GFX (g_pGXMainAllocator)");
            } else {
                std::snprintf(h.name, sizeof(h.name), "Heap#%u", h.heap_id);
            }
        } else if (gx_main_alloc != 0 && entry.allocator == gx_main_alloc &&
                   std::strstr(h.name, "g_pGXMain") == nullptr) {
            char tagged[48];
            std::snprintf(tagged, sizeof(tagged), "%s (GFX Main)", h.name);
            std::snprintf(h.name, sizeof(h.name), "%s", tagged);
        }
    }
    return n_valid;
}

const DiscoveredHeap* FindHeapByArenaAddr(const DiscoveredHeap* heaps, std::size_t count,
                                          std::uint64_t addr) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        if (addr >= heaps[i].arena_start && addr < heaps[i].arena_end) {
            return &heaps[i];
        }
    }
    return nullptr;
}

const DiscoveredHeap* FindHeapByControlAddr(const DiscoveredHeap* heaps, std::size_t count,
                                            std::uint64_t addr) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        if (addr == heaps[i].allocator ||
            (heaps[i].heap_impl != 0 && addr >= heaps[i].heap_impl - 0x08u &&
             addr < heaps[i].heap_impl + 0x4c0u)) {
            return &heaps[i];
        }
    }
    return nullptr;
}

struct ChunkInspection {
    bool valid_header_read = false;
    bool corrupted = false;
    std::uint64_t user_ptr = 0;
    std::uint64_t hdr_addr = 0;
    std::uint64_t prev_phys_hdr = 0;
    std::uint64_t next_and_flags = 0;
    std::uint64_t next_hdr = 0;
    std::uint64_t usable_bytes = 0;
    bool curr_in_use = false;
    bool prev_in_use = false;
    bool has_next_hdr_val = false;
    std::uint64_t next_hdr_val = 0;
    char summary[256]{};
};

ChunkInspection InspectRegularHeapChunk(const DiscoveredHeap& h, std::uint64_t raw_ptr) noexcept {
    ChunkInspection ci{};
    // DLRegularHeap payloads & chunk headers are 16-byte aligned; inspect aligned candidate
    const std::uint64_t aligned = raw_ptr & ~0xFULL;
    if (aligned < h.first_chunk + 0x10u || aligned >= h.end_sentinel) {
        std::snprintf(ci.summary, sizeof(ci.summary),
                      "inside [%s] arena (+0x%llx from base)",
                      h.name, static_cast<unsigned long long>(raw_ptr - h.arena_start));
        return ci;
    }

    // If raw_ptr == aligned, check `[aligned - 0x10]` as user payload header first
    ci.user_ptr = aligned;
    ci.hdr_addr = aligned - 0x10u;
    DLRegularChunkHeader hdr{};
    if (!SafeReadVal(static_cast<std::uintptr_t>(ci.hdr_addr), &hdr)) {
        std::snprintf(ci.summary, sizeof(ci.summary), "inside [%s], header unreadable", h.name);
        return ci;
    }

    // Also check if `aligned` is itself a chunk header (e.g. `r12 = rbx - 0x10` or `r15 = next_hdr`
    // in `DLRegularHeap::free`): if `[aligned - 0x08]` does NOT look like a valid chunk header inside
    // `[first_chunk, end_sentinel]`, whereas `[aligned + 0x08]` DOES!
    const std::uint64_t cand_next = hdr.next_and_flags & ~kChunkFlagsMask;
    const bool as_user_plausible =
        ((cand_next & 0xFULL) == 0 && cand_next > ci.hdr_addr && cand_next <= h.end_sentinel + 0x20u);
    if (!as_user_plausible && aligned + 0x10u <= h.end_sentinel) {
        DLRegularChunkHeader self_hdr{};
        if (SafeReadVal(static_cast<std::uintptr_t>(aligned), &self_hdr)) {
            const std::uint64_t self_next = self_hdr.next_and_flags & ~kChunkFlagsMask;
            if ((self_next & 0xFULL) == 0 && self_next > aligned &&
                self_next <= h.end_sentinel + 0x20u) {
                hdr = self_hdr;
                ci.hdr_addr = aligned;
                ci.user_ptr = aligned + 0x10u;
            }
        }
    }

    ci.valid_header_read = true;
    ci.prev_phys_hdr = hdr.prev_phys_hdr;
    ci.next_and_flags = hdr.next_and_flags;
    ci.curr_in_use = (hdr.next_and_flags & kChunkFlagCurrInUse) != 0;
    ci.prev_in_use = (hdr.next_and_flags & kChunkFlagPrevInUse) != 0;
    ci.next_hdr = hdr.next_and_flags & ~kChunkFlagsMask;

    // 1. Validate current chunk's `next_hdr`
    if ((ci.next_hdr & 0xFULL) != 0 || ci.next_hdr <= ci.hdr_addr ||
        ci.next_hdr > h.end_sentinel + 0x20u) {
        ci.corrupted = true;
        std::snprintf(ci.summary, sizeof(ci.summary),
                      "Heap \"%s\" chunk user=0x%llx *** CORRUPTED CURRENT HEADER [hdr+0x8]=0x%llx "
                      "(invalid next_hdr=0x%llx) ***",
                      h.name, static_cast<unsigned long long>(ci.user_ptr),
                      static_cast<unsigned long long>(ci.next_and_flags),
                      static_cast<unsigned long long>(ci.next_hdr));
        return ci;
    }

    ci.usable_bytes = (ci.next_hdr - ci.user_ptr) + 8ULL;

    // 2. Validate previous physical chunk link if `prev_in_use == 0`
    if (!ci.prev_in_use && ci.hdr_addr > h.first_chunk) {
        if ((ci.prev_phys_hdr & 0xFULL) != 0 || ci.prev_phys_hdr < h.first_chunk - 0x20u ||
            ci.prev_phys_hdr >= ci.hdr_addr) {
            ci.corrupted = true;
            std::snprintf(ci.summary, sizeof(ci.summary),
                          "Heap \"%s\" chunk user=0x%llx (usable=%lluB) *** CORRUPTED PREV LINK "
                          "[hdr+0x0]=0x%llx while prev_in_use=0 ***",
                          h.name, static_cast<unsigned long long>(ci.user_ptr),
                          static_cast<unsigned long long>(ci.usable_bytes),
                          static_cast<unsigned long long>(ci.prev_phys_hdr));
            return ci;
        }
    }

    // 3. Validate NEXT physical chunk's header at `[next_hdr + 0x08]`
    if (ci.next_hdr < h.end_sentinel &&
        SafeReadVal(static_cast<std::uintptr_t>(ci.next_hdr + 0x08u), &ci.next_hdr_val)) {
        ci.has_next_hdr_val = true;
        const std::uint64_t next_next = ci.next_hdr_val & ~kChunkFlagsMask;
        const bool bad_next = ((next_next & 0xFULL) != 0) ||
                              (next_next != 0 && (next_next <= ci.next_hdr ||
                                                  next_next > h.end_sentinel + 0x40u));
        if (bad_next) {
            ci.corrupted = true;
            std::snprintf(ci.summary, sizeof(ci.summary),
                          "Heap \"%s\" chunk user=0x%llx (usable=%lluB, %s) *** CORRUPTED NEXT CHUNK "
                          "HEADER at [0x%llx+0x8]=0x%llx (buffer overrun / UAF past +0x%llx!) ***",
                          h.name, static_cast<unsigned long long>(ci.user_ptr),
                          static_cast<unsigned long long>(ci.usable_bytes),
                          ci.curr_in_use ? "IN_USE" : "FREE",
                          static_cast<unsigned long long>(ci.next_hdr),
                          static_cast<unsigned long long>(ci.next_hdr_val),
                          static_cast<unsigned long long>(ci.usable_bytes));
            return ci;
        }
    }

    std::snprintf(ci.summary, sizeof(ci.summary),
                  "Heap \"%s\" chunk user=0x%llx%+lld (usable=%lluB, %s, prev=%s)",
                  h.name, static_cast<unsigned long long>(ci.user_ptr),
                  static_cast<long long>(raw_ptr - ci.user_ptr),
                  static_cast<unsigned long long>(ci.usable_bytes),
                  ci.curr_in_use ? "IN_USE" : "FREE",
                  ci.prev_in_use ? "IN_USE" : "FREE");
    return ci;
}

void DumpHexAsciiWindow(CrashSink& sink, const char* title, std::uintptr_t center_addr,
                        std::uintptr_t highlight_addr) noexcept {
    if (center_addr < 0x1040u) return;
    const std::uintptr_t base = (center_addr - 64u) & ~static_cast<std::uintptr_t>(15u);
    sink.Printf("    %s (around 0x%016llx):\n", title,
                static_cast<unsigned long long>(center_addr));
    for (int row = 0; row < 8; ++row) {
        const std::uintptr_t line_addr = base + static_cast<std::uintptr_t>(row * 16);
        std::uint8_t bytes[16]{};
        if (!SafeReadMem(line_addr, bytes, sizeof(bytes))) {
            sink.Printf("      0x%016llx: <unmapped / unreadable>\n",
                        static_cast<unsigned long long>(line_addr));
            continue;
        }
        char hex_part[64]{};
        char asc_part[20]{};
        int hp = 0;
        for (int b = 0; b < 16; ++b) {
            hp += std::snprintf(hex_part + hp, sizeof(hex_part) - hp, "%s%02x",
                                b == 8 ? "  " : " ", bytes[b]);
            asc_part[b] = (bytes[b] >= 0x20u && bytes[b] < 0x7Fu) ? static_cast<char>(bytes[b]) : '.';
        }
        asc_part[16] = '\0';
        const bool mark = (highlight_addr >= line_addr && highlight_addr < line_addr + 16u);
        sink.Printf("    %s 0x%016llx:%s  |%s|\n", mark ? "=>" : "  ",
                    static_cast<unsigned long long>(line_addr), hex_part, asc_part);
    }
}

void AnnotatePointerValue(std::uint64_t val, std::uintptr_t rsp,
                          const unsigned char* guest_img, std::uint64_t guest_size,
                          const DiscoveredHeap* heaps, std::size_t heap_count,
                          char* out, std::size_t out_size) noexcept {
    out[0] = '\0';
    if (val < 0x1000u) return;

    // 1. Check FromSoftware heap control block (`DLAllocator*`, `DLRegularHeap*`)
    if (const auto* hc = FindHeapByControlAddr(heaps, heap_count, val)) {
        if (val == hc->allocator) {
            std::snprintf(out, out_size, "-> [Heap \"%s\" DLAllocator*]", hc->name);
        } else if (hc->is_regular_heap && val >= hc->reg_heap_addr &&
                   val < hc->reg_heap_addr + sizeof(DLRegularHeapState)) {
            std::snprintf(out, out_size, "-> [Heap \"%s\" DLRegularHeap+0x%llx]",
                          hc->name, static_cast<unsigned long long>(val - hc->reg_heap_addr));
        } else {
            std::snprintf(out, out_size, "-> [Heap \"%s\" host struct]", hc->name);
        }
        return;
    }

    // 2. Check FromSoftware heap arena allocation
    if (const auto* ha = FindHeapByArenaAddr(heaps, heap_count, val)) {
        if (ha->is_regular_heap) {
            const ChunkInspection ci = InspectRegularHeapChunk(*ha, val);
            std::snprintf(out, out_size, "-> [%s]", ci.summary);
        } else {
            std::snprintf(out, out_size, "-> [Heap \"%s\" arena +0x%llx]",
                          ha->name, static_cast<unsigned long long>(val - ha->arena_start));
        }
        return;
    }

    // 3. Check current thread stack proximity (`[RSP - 256 .. RSP + 65536]`)
    if (rsp != 0 && val >= rsp - 256u && val <= rsp + 65536u) {
        const long long delta = static_cast<long long>(val) - static_cast<long long>(rsp);
        std::snprintf(out, out_size, "-> [thread stack RSP%+lld]", delta);
        return;
    }

    // 4. Check guest / host code symbol
    char sym[192]{};
    if (ResolveCodeSymbol(static_cast<std::uintptr_t>(val), guest_img, guest_size, sym, sizeof(sym))) {
        std::snprintf(out, out_size, "-> %s", sym);
        return;
    }

    // 5. Check printable ASCII string literal (`>= 4` chars)
    char str_preview[48]{};
    ReadHeapNameSafe(static_cast<std::uintptr_t>(val), str_preview, sizeof(str_preview));
    if (std::strlen(str_preview) >= 4) {
        std::snprintf(out, out_size, "-> \"%s\"", str_preview);
    }
}

const char* SignalName(int sig) noexcept {
    switch (sig) {
    case SIGSEGV: return "SIGSEGV (Segmentation Fault / Access Violation)";
#ifdef SIGBUS
    case SIGBUS:  return "SIGBUS (Bus / Misaligned / Unmapped Memory Error)";
#endif
    case SIGILL:  return "SIGILL (Illegal CPU Instruction)";
    case SIGFPE:  return "SIGFPE (Arithmetic / Divide-by-Zero Exception)";
#ifdef SIGTRAP
    case SIGTRAP: return "SIGTRAP (Breakpoint / Trap Instruction)";
#endif
    case SIGABRT: return "SIGABRT (Abort)";
    default:      return "FATAL SIGNAL";
    }
}

void BbTerminateHandler() {
    if (std::exception_ptr ep = std::current_exception()) {
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            std::snprintf(g_extra_reason, sizeof(g_extra_reason),
                          "Uncaught C++ exception: %s", e.what() ? e.what() : "<null>");
        } catch (...) {
            std::snprintf(g_extra_reason, sizeof(g_extra_reason),
                          "Uncaught non-std C++ exception (std::terminate)");
        }
    } else {
        std::snprintf(g_extra_reason, sizeof(g_extra_reason),
                      "std::terminate() called without active exception (noexcept or joinable std::thread)");
    }
#ifdef _WIN32
    CONTEXT ctx{};
    ::RtlCaptureContext(&ctx);
    ucontext_t uc{};
    BbWin32ToUcontext(&ctx, 0, &uc);
    WriteCrashReport(SIGABRT, reinterpret_cast<const void*>(ctx.Rip), &uc, nullptr, 0);
#else
    WriteCrashReport(SIGABRT, __builtin_return_address(0), nullptr, nullptr, 0);
#endif
    std::_Exit(134);
}

#ifdef _WIN32
extern "C" void BbWin32AbortHandler(int sig) {
    if (g_extra_reason[0] == '\0') {
        std::snprintf(g_extra_reason, sizeof(g_extra_reason),
                      "CRT abort() / SIGABRT (%d) intercepted before Win32 __fastfail", sig);
    }
    CONTEXT ctx{};
    ::RtlCaptureContext(&ctx);
    ucontext_t uc{};
    BbWin32ToUcontext(&ctx, 0, &uc);
    WriteCrashReport(SIGABRT, reinterpret_cast<const void*>(ctx.Rip), &uc, nullptr, 0);
    std::_Exit(134);
}

LONG WINAPI BbWin32UnhandledExceptionFilter(EXCEPTION_POINTERS* pointers) {
    const EXCEPTION_RECORD* rec = pointers ? pointers->ExceptionRecord : nullptr;
    const DWORD code = rec ? rec->ExceptionCode : 0;
    const void* addr = rec ? rec->ExceptionAddress : nullptr;
    char mod_buf[192]{};
    DescribeWin32ModuleOffset(addr, mod_buf, sizeof(mod_buf));
    std::snprintf(g_extra_reason, sizeof(g_extra_reason),
                  "Unhandled Win32 Exception 0x%08lX at %s (addr=0x%llx)",
                  static_cast<unsigned long>(code),
                  mod_buf[0] ? mod_buf : "<unknown module>",
                  static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(addr)));
    ucontext_t uc{};
    if (pointers && pointers->ContextRecord) {
        const unsigned long long err_mark = (code == EXCEPTION_STACK_OVERFLOW) ? 0x10000ULL : 0ULL;
        BbWin32ToUcontext(pointers->ContextRecord, err_mark, &uc);
    }
    WriteCrashReport(SIGSEGV, addr, (pointers && pointers->ContextRecord) ? &uc : nullptr, nullptr, 0);
    std::_Exit(139);
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

} // namespace

// ============================================================================
// Public Lock-Free Flight Recorder & Lifecycle API
// ============================================================================
void Initialize() noexcept {
    if (!g_initialized.exchange(true, std::memory_order_acq_rel)) {
        runtime_set_flight_recorder(&bbgpu_flight_record);
        std::set_terminate(BbTerminateHandler);
#ifdef _WIN32
        ULONG stack_guarantee = 65536;
        ::SetThreadStackGuarantee(&stack_guarantee);
        std::signal(SIGABRT, BbWin32AbortHandler);
        ::SetUnhandledExceptionFilter(BbWin32UnhandledExceptionFilter);
#endif
        RecordEvent("INIT", "Crash Black-Box V2 (terminate + Win32 SEH/SIGABRT/stack-overflow) armed");
    }
}

void SetGuestImage(const unsigned char* image, std::uint64_t size) noexcept {
    Initialize();
    g_guest_image.store(image, std::memory_order_release);
    g_guest_image_size.store(size, std::memory_order_release);
}

void RecordEvent(const char* category, const char* detail) noexcept {
    if (!category || !detail || detail[0] == '\0') return;
    if (g_crash_in_progress.load(std::memory_order_relaxed)) return;

    // Deduplicate consecutive identical event strings so tight polling loops don't evict history
    const std::uint64_t prev_idx = g_flight_write_idx.load(std::memory_order_relaxed);
    if (prev_idx > 0) {
        const FlightEntry& prev = g_flight_ring[(prev_idx - 1u) & (kFlightRingCap - 1u)];
        const std::uint64_t pseq = prev.seq.load(std::memory_order_acquire);
        if ((pseq & 1ULL) == 0 && pseq != 0 &&
            std::strncmp(prev.category, category, sizeof(prev.category) - 1) == 0 &&
            std::strncmp(prev.detail, detail, sizeof(prev.detail) - 1) == 0) {
            return;
        }
    }

    const std::uint64_t ticket = g_flight_write_idx.fetch_add(1, std::memory_order_relaxed);
    FlightEntry& slot = g_flight_ring[ticket & (kFlightRingCap - 1u)];
    const std::uint64_t write_seq = (ticket + 1ULL) * 2ULL;

    slot.seq.store(write_seq - 1ULL, std::memory_order_release); // Odd: write in progress
    slot.time_us = runtime_process_time_us();
    slot.frame = g_frame_counter.load(std::memory_order_relaxed);
    slot.tid = CurrentThreadIdFast();
    CurrentThreadNameSafe(slot.thread, sizeof(slot.thread));
    std::snprintf(slot.category, sizeof(slot.category), "%s", category);
    std::snprintf(slot.detail, sizeof(slot.detail), "%s", detail);
    slot.seq.store(write_seq, std::memory_order_release); // Even: committed
}

void RecordEventFmt(const char* category, const char* fmt, ...) noexcept {
    if (!category || !fmt) return;
    if (g_crash_in_progress.load(std::memory_order_relaxed)) return;
    char buf[168]{};
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    RecordEvent(category, buf);
}

void UpdatePlayerPosition(float x, float y, float z) noexcept {
    if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
        g_player_x.store(x, std::memory_order_relaxed);
        g_player_y.store(y, std::memory_order_relaxed);
        g_player_z.store(z, std::memory_order_relaxed);
        g_has_player_pos.store(true, std::memory_order_relaxed);
    }
    g_frame_counter.fetch_add(1, std::memory_order_relaxed);
}

void NoteFrameTick() noexcept {
    g_frame_counter.fetch_add(1, std::memory_order_relaxed);
}

// ============================================================================
// Master Crash Black-Box Writer (Linux + Windows Unified Output)
// ============================================================================
void WriteCrashReport(int sig, const void* fault_addr, void* ucontext_ptr,
                      const void* guest_image, std::uint64_t guest_size) noexcept {
    const std::uint32_t self_tid = CurrentThreadIdFast();
    std::uint32_t expected_owner = 0;
    if (!g_crash_owner_tid.compare_exchange_strong(expected_owner, self_tid,
                                                   std::memory_order_acq_rel)) {
        // Distinguish a CONCURRENT crash on a sibling thread from a TRUE recursive
        // fault on the primary reporting thread: a sibling thread MUST park forever so it
        // cannot return to fault() and call _exit(139) before the primary thread finishes
        // writing and flushing logs/crash-YYYYMMDD_HHMMSS.log!
        if (expected_owner != self_tid) {
            const char msg[] = "[BB-FAULT-CONCURRENT] sibling thread faulted; parking while primary writes crash log...\n";
#ifdef _WIN32
            ::_write(2, msg, static_cast<unsigned int>(sizeof(msg) - 1));
            for (;;) ::Sleep(INFINITE);
#else
            const ssize_t w = ::write(2, msg, sizeof(msg) - 1);
            (void)w;
            for (;;) ::pause();
#endif
        }
        const char msg[] = "[BB-FAULT-REENTRANT] secondary fault on primary crash thread\n";
#ifdef _WIN32
        ::_write(2, msg, static_cast<unsigned int>(sizeof(msg) - 1));
#else
        const ssize_t w = ::write(2, msg, sizeof(msg) - 1);
        (void)w;
#endif
        return;
    }
    g_crash_in_progress.store(true, std::memory_order_release);

    const auto* guest_img = static_cast<const unsigned char*>(
        guest_image ? guest_image : g_guest_image.load(std::memory_order_acquire));
    const std::uint64_t eff_guest_size =
        guest_size ? guest_size : g_guest_image_size.load(std::memory_order_acquire);

    static CrashSink sink{};
    sink.OpenCrashLogFile();

    const std::uint64_t crash_us = runtime_process_time_us();
    const double uptime_s = static_cast<double>(crash_us) * 1e-6;
    const std::time_t wall_now = std::time(nullptr);
    struct tm tm_buf{};
#ifdef _WIN32
    ::localtime_s(&tm_buf, &wall_now);
    const char* os_name = "Windows x86-64 (Win32 VEH/SEH -> Crash Black-Box V2)";
#else
    ::localtime_r(&wall_now, &tm_buf);
    const char* os_name = "Linux x86-64";
#endif
    char wall_str[64]{};
    std::strftime(wall_str, sizeof(wall_str), "%Y-%m-%d %H:%M:%S", &tm_buf);

    char thread_name[32]{};
    CurrentThreadNameSafe(thread_name, sizeof(thread_name));
    const std::uint32_t tid = CurrentThreadIdFast();

    auto* uc = static_cast<ucontext_t*>(ucontext_ptr);
    const greg_t* gregs = uc ? uc->uc_mcontext.gregs : nullptr;
    const std::uintptr_t rip = gregs ? static_cast<std::uintptr_t>(gregs[REG_RIP]) : 0u;
    const std::uintptr_t rsp = gregs ? static_cast<std::uintptr_t>(gregs[REG_RSP]) : 0u;
    const std::uintptr_t rbp = gregs ? static_cast<std::uintptr_t>(gregs[REG_RBP]) : 0u;
    const std::uint64_t err_bits = gregs ? static_cast<std::uint64_t>(gregs[REG_ERR]) : 0ULL;
    const auto faddr = reinterpret_cast<std::uintptr_t>(fault_addr);

    const char* access_kind = "UNKNOWN";
    if (sig == SIGSEGV) {
        if (err_bits & 0x10000ULL) access_kind = "STACK OVERFLOW (0xC00000FD)";
        else if (err_bits & 16ULL) access_kind = "INSTRUCTION FETCH (jumped to invalid/non-exec RIP)";
        else if (err_bits & 2ULL) access_kind = "MEMORY WRITE";
        else access_kind = "MEMORY READ";
    }

    static char rip_sym[256]{};
    rip_sym[0] = '\0';
    if (!ResolveCodeSymbol(rip, guest_img, eff_guest_size, rip_sym, sizeof(rip_sym))) {
        std::snprintf(rip_sym, sizeof(rip_sym), "<unmapped/unknown code @ 0x%016llx>",
                      static_cast<unsigned long long>(rip));
    }

    // ========================================================================
    // 1. HEADER & AUTOMATIC HUMAN-READABLE DIAGNOSIS (Flushed BEFORE heap walk)
    // ========================================================================
    sink.Printf("\n"
                "================================================================================\n"
                "                 BLOODBORNE PC — ENGINE CRASH BLACK BOX                         \n"
                "================================================================================\n");
    sink.Printf("  Timestamp      : %s (uptime %.3f s, frame ~%u)\n",
                wall_str, uptime_s, g_frame_counter.load(std::memory_order_relaxed));
    sink.Printf("  Platform       : %s\n", os_name);
    if (sink.file_path[0] != '\0') {
        sink.Printf("  Crash Log File : %s\n", sink.file_path);
    }
    if (g_extra_reason[0] != '\0') {
        sink.Printf("  Reason         : %s\n", g_extra_reason);
    }
    sink.Printf("  Signal         : %s [sig=%d, access=%s, err=0x%llx]\n",
                SignalName(sig), sig, access_kind, static_cast<unsigned long long>(err_bits));
    sink.Printf("  Fault Thread   : \"%s\" (tid=%u)\n", thread_name, tid);
    sink.Printf("  Fault RIP      : 0x%016llx -> %s\n",
                static_cast<unsigned long long>(rip), rip_sym);
    sink.Printf("  Fault Address  : 0x%016llx\n", static_cast<unsigned long long>(faddr));

#ifdef _WIN32
    {
        char mod_off[192]{};
        DescribeWin32ModuleOffset(reinterpret_cast<const void*>(rip), mod_off, sizeof(mod_off));
        if (mod_off[0] != '\0') {
            sink.Printf("  Fault Module   : %s\n", mod_off);
        }
        char win_mem[256]{};
        bb_win32_describe_address(fault_addr, win_mem, static_cast<unsigned>(sizeof(win_mem)));
        if (win_mem[0] != '\0') {
            sink.Printf("  OS Mapping     : %s\n", win_mem);
        }
    }
#endif
    sink.FlushOnly();

    // Discover FromSoftware heaps in static storage (zero stack pressure)
    static DiscoveredHeap heaps[kMaxDiscoveredHeaps]{};
    std::memset(heaps, 0, sizeof(heaps));
    const std::size_t heap_count = EnumerateFromSoftwareHeaps(guest_img, heaps);

    if (guest_img) {
        sink.Printf("  Guest Image    : base=0x%016llx, size=0x%llx\n",
                    static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(guest_img)),
                    static_cast<unsigned long long>(eff_guest_size));
    }
    if (g_has_player_pos.load(std::memory_order_relaxed)) {
        sink.Printf("  Player Pos XYZ : (%.2f, %.2f, %.2f)\n",
                    static_cast<double>(g_player_x.load(std::memory_order_relaxed)),
                    static_cast<double>(g_player_y.load(std::memory_order_relaxed)),
                    static_cast<double>(g_player_z.load(std::memory_order_relaxed)));
    }
    {
        char save_dir[256]{};
        const int slot = runtime_savedata_active_slot(save_dir, sizeof(save_dir));
        if (slot >= 0) {
            sink.Printf("  Active Save    : userdata%04d (%s)\n", slot, save_dir);
        }
    }
    {
        const char* gx_mods = bb_env("BB_GX_MODULES");
        const char* fast_ld = bb_env("BB_FAST_LOAD");
        const char* fps_env = bb_env("BB_FPS");
        sink.Printf("  Active Flags   : BB_GX_MODULES=%s, BB_FAST_LOAD=%s, BB_FPS=%s\n",
                    gx_mods ? gx_mods : "all",
                    fast_ld ? fast_ld : "1(default)",
                    fps_env ? fps_env : "uncap");
    }

    // Print RIP instruction bytes (helps verify exact instruction on any build)
    {
        std::uint8_t code_bytes[16]{};
        if (SafeReadMem(rip, code_bytes, sizeof(code_bytes))) {
            char hex_insn[64]{};
            int p = 0;
            for (int i = 0; i < 16; ++i) {
                p += std::snprintf(hex_insn + p, sizeof(hex_insn) - p, "%s%02x",
                                   i ? " " : "", code_bytes[i]);
            }
            sink.Printf("  Bytes at [RIP] : %s\n", hex_insn);
        }
    }

    // Automatic plain-language diagnosis summary
    sink.Printf("\n  [QUICK DIAGNOSIS]:\n");
    const auto img_base = reinterpret_cast<std::uintptr_t>(guest_img);
    const std::uint64_t guest_vma =
        (img_base && rip >= img_base && (rip - img_base) < (eff_guest_size ? eff_guest_size : 0x10000000ULL))
            ? static_cast<std::uint64_t>(rip - img_base)
            : ~0ULL;

    if (guest_vma >= kDLRegularHeapFreeBegin && guest_vma < kDLRegularHeapFreeEnd) {
        sink.Printf("    * Crash occurred inside FromSoftware `DLRegularHeap::free` (+0x%llx)!\n"
                    "    * A freed pointer or an adjacent physical heap chunk header was corrupted\n"
                    "      (Use-After-Free, double-free, or buffer overflow before this free call).\n"
                    "    * Check Section 3 (caller C++/guest function) and Section 4 (corrupted chunk).\n",
                    static_cast<unsigned long long>(guest_vma));
    } else if (guest_vma >= kDLRegularHeapAllocBegin && guest_vma < kDLRegularHeapAllocEnd) {
        sink.Printf("    * Crash occurred inside FromSoftware `DLRegularHeap::allocate` (+0x%llx)!\n"
                    "    * Free-list bin links were corrupted by an earlier Use-After-Free or overflow.\n",
                    static_cast<unsigned long long>(guest_vma));
    } else if (sig == SIGSEGV && (err_bits & 16ULL)) {
        std::uintptr_t caller_on_stack = 0;
        char caller_sym[256]{};
        if (SafeReadVal(rsp, &caller_on_stack) &&
            ResolveCodeSymbol(caller_on_stack, guest_img, eff_guest_size, caller_sym, sizeof(caller_sym))) {
            sink.Printf("    * Indirect CALL / JMP to invalid function pointer RIP=0x%llx!\n"
                        "    * Return address at [RSP] points to caller: %s\n"
                        "    * Likely destroyed vtable or dangling object pointer.\n",
                        static_cast<unsigned long long>(rip), caller_sym);
        } else {
            sink.Printf("    * Indirect CALL / JMP to invalid function pointer RIP=0x%llx.\n",
                        static_cast<unsigned long long>(rip));
        }
    } else if (sig == SIGSEGV && faddr < 0x10000u) {
        sink.Printf("    * Null or near-null struct pointer dereference at offset +0x%llx during %s.\n",
                    static_cast<unsigned long long>(faddr), access_kind);
    } else if (sig == SIGSEGV && (faddr >> 48) != 0 && (faddr >> 48) != 0xFFFFu) {
        sink.Printf("    * Non-canonical / corrupted 64-bit pointer 0x%016llx accessed during %s.\n",
                    static_cast<unsigned long long>(faddr), access_kind);
    } else if (const auto* fh = FindHeapByArenaAddr(heaps, heap_count, faddr)) {
        sink.Printf("    * Fault address 0x%016llx lies inside FromSoftware heap \"%s\".\n",
                    static_cast<unsigned long long>(faddr), fh->name);
    } else {
        sink.Printf("    * Unhandled %s at %s.\n", access_kind, rip_sym);
    }

    // ========================================================================
    // 2. FULL 16-REGISTER DUMP WITH AUTOMATIC POINTER / HEAP ANNOTATIONS
    // ========================================================================
    sink.Printf("\n--------------------------------------------------------------------------------\n"
                "  1. CPU REGISTERS & POINTER ANNOTATIONS\n"
                "--------------------------------------------------------------------------------\n");
    struct RegSpec {
        const char* name;
        int idx;
    };
    static constexpr RegSpec kGprs[] = {
        {"rax", REG_RAX}, {"rbx", REG_RBX}, {"rcx", REG_RCX}, {"rdx", REG_RDX},
        {"rsi", REG_RSI}, {"rdi", REG_RDI}, {"rbp", REG_RBP}, {"rsp", REG_RSP},
        {"r8 ", REG_R8},  {"r9 ", REG_R9},  {"r10", REG_R10}, {"r11", REG_R11},
        {"r12", REG_R12}, {"r13", REG_R13}, {"r14", REG_R14}, {"r15", REG_R15},
    };
    if (gregs) {
        for (const auto& r : kGprs) {
            const auto val = static_cast<std::uint64_t>(gregs[r.idx]);
            char note[280]{};
            AnnotatePointerValue(val, rsp, guest_img, eff_guest_size, heaps, heap_count,
                                 note, sizeof(note));
            sink.Printf("    %s = 0x%016llx  %s\n",
                        r.name, static_cast<unsigned long long>(val), note);
        }
        sink.Printf("    rip = 0x%016llx  -> %s\n",
                    static_cast<unsigned long long>(rip), rip_sym);
        sink.Printf("    efl = 0x%016llx\n",
                    static_cast<unsigned long long>(gregs[REG_EFL]));
    } else {
        sink.Printf("    <no CPU register context available>\n");
    }

    // ========================================================================
    // 3. HYBRID CALL STACK (RBP CHAIN + RAW RSP SCAN ACROSS GUEST & C++20 HOST)
    // ========================================================================
    sink.Printf("\n--------------------------------------------------------------------------------\n"
                "  2. HYBRID CALL STACK (GUEST eboot.bin <-> NATIVE dantelion_gx / libbbgpu)\n"
                "--------------------------------------------------------------------------------\n");
    sink.Printf("  [2A. Frame-Pointer (RBP) Unwind]:\n");
    sink.Printf("    #00 [RIP] 0x%016llx  %s\n",
                static_cast<unsigned long long>(rip), rip_sym);

    std::uintptr_t rbp_return_addrs[32]{};
    std::size_t rbp_depth = 0;

    // If RIP itself was an invalid call target (`call *rax` to 0 / garbage), `[RSP]` holds caller!
    if (sig == SIGSEGV && (err_bits & 16ULL) && rsp != 0) {
        std::uintptr_t top_ret = 0;
        if (SafeReadVal(rsp, &top_ret) && top_ret >= 0x10000u) {
            char sym[256]{};
            if (ResolveCodeSymbol(top_ret, guest_img, eff_guest_size, sym, sizeof(sym))) {
                sink.Printf("    #00b[*RSP] 0x%016llx  %s\n",
                            static_cast<unsigned long long>(top_ret), sym);
                rbp_return_addrs[rbp_depth++] = top_ret;
            }
        }
    }

    {
        std::uintptr_t cur_rbp = rbp;
        std::uintptr_t visited_rbps[32]{};
        std::size_t n_visited = 0;
        for (int depth = 1; depth <= 28 && cur_rbp >= 0x1000u && (cur_rbp & 0x7u) == 0; ++depth) {
            bool cycle = false;
            for (std::size_t v = 0; v < n_visited; ++v) {
                if (visited_rbps[v] == cur_rbp) {
                    cycle = true;
                    break;
                }
            }
            if (cycle) break;
            if (n_visited < 32) visited_rbps[n_visited++] = cur_rbp;

            std::uintptr_t frame[2]{0, 0}; // [0] = saved RBP, [1] = return RIP
            if (!SafeReadMem(cur_rbp, frame, sizeof(frame))) {
                break;
            }
            const std::uintptr_t ret_pc = frame[1];
            if (ret_pc < 0x1000u) {
                break;
            }
            char sym[256]{};
            if (!ResolveCodeSymbol(ret_pc, guest_img, eff_guest_size, sym, sizeof(sym))) {
                std::snprintf(sym, sizeof(sym), "<unknown @ 0x%016llx>",
                              static_cast<unsigned long long>(ret_pc));
            }
            sink.Printf("    #%02d [RBP=0x%012llx] 0x%016llx  %s\n",
                        depth, static_cast<unsigned long long>(cur_rbp),
                        static_cast<unsigned long long>(ret_pc), sym);
            if (rbp_depth < 32) {
                rbp_return_addrs[rbp_depth++] = ret_pc;
            }

            const std::uintptr_t next_rbp = frame[0];
            if (next_rbp == 0) break;
            // Normal stack frames grow downward (next_rbp > cur_rbp), EXCEPT across `enter_on_stack`
            // / `bb_run_on_stack` where the 8 MiB low guest stack bridges back to the OS host stack.
            if (next_rbp <= cur_rbp) {
                const bool crossing_to_host_stack = (cur_rbp < (1ULL << 40)) && (next_rbp >= (1ULL << 40));
                if (!crossing_to_host_stack) {
                    break;
                }
            } else if ((next_rbp - cur_rbp) > (16ULL << 20) && !(cur_rbp < (1ULL << 40) && next_rbp >= (1ULL << 40))) {
                break;
            }
            cur_rbp = next_rbp;
        }
    }

    // 2B. Raw Stack (`RSP`) Return-Address Scan (`[RSP .. RSP + 1536]`)
    // Catches frameless `-O2`/`-O3` C++ callers, Win64 DLL frames (`vulkan-1.dll`, GPU drivers),
    // and leaf functions that haven't pushed `RBP` yet!
    if (rsp >= 0x1000u) {
        sink.Printf("\n  [2B. Raw Stack (RSP) Return-Address Scan (RSP .. RSP+0x600, catches frameless C++/DLLs)]:\n");
        constexpr std::size_t kScanSlots = 192; // 192 * 8 = 1536 bytes
        std::uintptr_t last_printed_ret = 0;
        int shown = 0;
        for (std::size_t s = 0; s < kScanSlots && shown < 24; ++s) {
            const std::uintptr_t slot_addr = rsp + s * sizeof(std::uintptr_t);
            std::uintptr_t val = 0;
            if (!SafeReadVal(slot_addr, &val)) break;
            if (val == last_printed_ret) continue;
            if (!LooksLikeReturnAddress(val, guest_img, eff_guest_size)) continue;

            bool in_rbp_chain = false;
            for (std::size_t k = 0; k < rbp_depth; ++k) {
                if (rbp_return_addrs[k] == val) {
                    in_rbp_chain = true;
                    break;
                }
            }
            char sym[256]{};
            if (!ResolveCodeSymbol(val, guest_img, eff_guest_size, sym, sizeof(sym))) continue;
            sink.Printf("    [RSP+0x%03zx] 0x%016llx  %s%s\n",
                        s * sizeof(std::uintptr_t),
                        static_cast<unsigned long long>(val),
                        sym,
                        in_rbp_chain ? "  (in RBP)" : "  <-- [frameless/extra]");
            last_printed_ret = val;
            ++shown;
        }
        if (shown == 0) {
            sink.Printf("    <no return addresses detected in [RSP..RSP+0x600]>\n");
        }
    }

    // ========================================================================
    // 4. FROMSOFTWARE HEAP MAP & CORRUPT CHUNK HEADER INSPECTOR
    // ========================================================================
    sink.Printf("\n--------------------------------------------------------------------------------\n"
                "  3. FROMSOFTWARE HEAP MAP & CHUNK CORRUPTION INSPECTOR\n"
                "--------------------------------------------------------------------------------\n");
    if (heap_count == 0) {
        sink.Printf("    <DLHeapManager not yet initialized or guest image unavailable>\n");
    } else {
        sink.Printf("  Registered FromSoftware Heaps (%zu):\n", heap_count);
        for (std::size_t i = 0; i < heap_count; ++i) {
            const auto& h = heaps[i];
            const double cap_mb = static_cast<double>(h.arena_end - h.arena_start) / (1024.0 * 1024.0);
            if (h.is_regular_heap) {
                const double used_mb = static_cast<double>(h.used_bytes) / (1024.0 * 1024.0);
                sink.Printf("    [%2zu] %-24s alloc=0x%010llx  arena=[0x%010llx..0x%010llx) "
                            "%6.1f MiB | used=%6.2f MiB (%llu allocs)\n",
                            i, h.name,
                            static_cast<unsigned long long>(h.allocator),
                            static_cast<unsigned long long>(h.arena_start),
                            static_cast<unsigned long long>(h.arena_end),
                            cap_mb, used_mb, static_cast<unsigned long long>(h.alloc_count));
            } else {
                sink.Printf("    [%2zu] %-24s alloc=0x%010llx  arena=[0x%010llx..0x%010llx) %6.1f MiB\n",
                            i, h.name,
                            static_cast<unsigned long long>(h.allocator),
                            static_cast<unsigned long long>(h.arena_start),
                            static_cast<unsigned long long>(h.arena_end),
                            cap_mb);
            }
        }

        // Deep-inspect all unique heap pointers held in `fault_addr` and `rax..r15`
        struct CandidatePtr {
            const char* label;
            std::uint64_t addr;
        };
        CandidatePtr cands[18]{};
        std::size_t n_cands = 0;
        cands[n_cands++] = {"fault_addr", static_cast<std::uint64_t>(faddr)};
        if (gregs) {
            for (const auto& r : kGprs) {
                cands[n_cands++] = {r.name, static_cast<std::uint64_t>(gregs[r.idx])};
            }
        }

        std::uint64_t inspected_hdrs[18]{};
        std::size_t n_inspected = 0;
        for (std::size_t c = 0; c < n_cands; ++c) {
            const std::uint64_t p = cands[c].addr;
            if (p < 0x10000u) continue;
            const auto* h = FindHeapByArenaAddr(heaps, heap_count, p);
            if (!h || !h->is_regular_heap) continue;

            const ChunkInspection ci = InspectRegularHeapChunk(*h, p);
            if (!ci.valid_header_read) continue;

            bool dup = false;
            for (std::size_t k = 0; k < n_inspected; ++k) {
                if (inspected_hdrs[k] == ci.hdr_addr) {
                    dup = true;
                    break;
                }
            }
            if (dup) continue;
            inspected_hdrs[n_inspected++] = ci.hdr_addr;

            sink.Printf("\n  Heap Chunk Inspection (via `%s` = 0x%016llx in Heap \"%s\"):\n",
                        cands[c].label, static_cast<unsigned long long>(p), h->name);
            sink.Printf("    chunk_hdr    = 0x%016llx | user_payload = 0x%016llx | usable_size = %llu bytes\n",
                        static_cast<unsigned long long>(ci.hdr_addr),
                        static_cast<unsigned long long>(ci.user_ptr),
                        static_cast<unsigned long long>(ci.usable_bytes));
            sink.Printf("    [hdr+0x00]   = 0x%016llx (prev_phys_hdr; valid when prev_in_use==0)\n",
                        static_cast<unsigned long long>(ci.prev_phys_hdr));
            sink.Printf("    [hdr+0x08]   = 0x%016llx -> next_hdr=0x%016llx | curr_in_use=%d | prev_in_use=%d\n",
                        static_cast<unsigned long long>(ci.next_and_flags),
                        static_cast<unsigned long long>(ci.next_hdr),
                        ci.curr_in_use ? 1 : 0,
                        ci.prev_in_use ? 1 : 0);
            if (ci.has_next_hdr_val) {
                sink.Printf("    [next+0x08]  = 0x%016llx (at 0x%016llx)\n",
                            static_cast<unsigned long long>(ci.next_hdr_val),
                            static_cast<unsigned long long>(ci.next_hdr + 0x08u));
            }
            if (ci.corrupted) {
                sink.Printf("    >>> *** CORRUPTED HEAP HEADER DETECTED ***: %s\n", ci.summary);
            }
            // Dump [-64..+64] bytes around the chunk header (shows tail of previous object + start of this payload)
            DumpHexAsciiWindow(sink, "Memory around current chunk header [hdr-0x40..hdr+0x40]",
                               static_cast<std::uintptr_t>(ci.hdr_addr),
                               static_cast<std::uintptr_t>(ci.hdr_addr + 0x08u));
            if (ci.corrupted && ci.next_hdr >= 0x10000u &&
                (ci.next_hdr > ci.hdr_addr + 64u || ci.next_hdr < ci.hdr_addr)) {
                DumpHexAsciiWindow(sink, "Memory around CORRUPTED next_hdr [next-0x40..next+0x40]",
                                   static_cast<std::uintptr_t>(ci.next_hdr),
                                   static_cast<std::uintptr_t>(ci.next_hdr + 0x08u));
            }
        }
    }

    // ========================================================================
    // 5. FLIGHT RECORDER DUMP (CHRONOLOGICAL LAST EVENTS BEFORE CRASH)
    // ========================================================================
    sink.Printf("\n--------------------------------------------------------------------------------\n"
                "  4. FLIGHT RECORDER (LAST ENGINE / ASSET / SAVE EVENTS BEFORE CRASH)\n"
                "--------------------------------------------------------------------------------\n");
    const std::uint64_t total_written = g_flight_write_idx.load(std::memory_order_acquire);
    if (total_written == 0) {
        sink.Printf("    <no events recorded>\n");
    } else {
        const std::uint64_t start_ticket =
            (total_written > kFlightRingCap) ? (total_written - kFlightRingCap) : 0ULL;
        int printed = 0;
        for (std::uint64_t t = start_ticket; t < total_written; ++t) {
            const FlightEntry& e = g_flight_ring[t & (kFlightRingCap - 1u)];
            const std::uint64_t s1 = e.seq.load(std::memory_order_acquire);
            if ((s1 & 1ULL) != 0 || s1 == 0) {
                sink.Printf("    [T-  ?.???s | f=%6u | %-14s] %-10s : <in-flight concurrent event>\n",
                            e.frame, e.thread, e.category);
                continue;
            }
            const double rel_s =
                (crash_us >= e.time_us) ? (static_cast<double>(crash_us - e.time_us) * 1e-6) : 0.0;
            sink.Printf("    [T-%7.3fs | f=%6u | %-14s] %-10s : %s\n",
                        rel_s, e.frame, e.thread, e.category, e.detail);
            ++printed;
        }
        if (printed == 0) {
            sink.Printf("    <no committed events>\n");
        }
    }

    sink.Printf("================================================================================\n");
    if (sink.file_path[0] != '\0') {
        sink.Printf("  Saved standalone crash report -> %s\n", sink.file_path);
        sink.Printf("================================================================================\n");
    }
    sink.FlushAndClose();
}

} // namespace BbCrashBlackBox

// ============================================================================
// C ABI Exports for `src/probe.c` & `src/runtime_*.c`
// ============================================================================
extern "C" {

void bbgpu_flight_record(const char* category, const char* detail) {
    BbCrashBlackBox::RecordEvent(category, detail);
}

int bbgpu_resolve_code_address(std::uintptr_t addr, char* out_buf, std::size_t out_size) {
    const auto* img = BbCrashBlackBox::g_guest_image.load(std::memory_order_acquire);
    const std::uint64_t sz = BbCrashBlackBox::g_guest_image_size.load(std::memory_order_acquire);
    return BbCrashBlackBox::ResolveCodeSymbol(addr, img, sz, out_buf, out_size) ? 1 : 0;
}

void bbgpu_write_crash_blackbox(int sig, const void* fault_addr, void* ucontext_ptr,
                                const void* guest_image, std::uint64_t guest_size) {
    BbCrashBlackBox::WriteCrashReport(sig, fault_addr, ucontext_ptr, guest_image, guest_size);
}

} // extern "C"
