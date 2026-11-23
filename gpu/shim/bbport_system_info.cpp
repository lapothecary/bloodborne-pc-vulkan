// SPDX-License-Identifier: GPL-2.0-or-later
#include "bbport_system_info.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "bbport_frame_rate.h"
#include "bbport_settings.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

#ifndef _WIN32
#include <dirent.h>
#include <fcntl.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace BbSysInfo {

namespace {

constexpr double kInvGiB = 1.0 / (1024.0 * 1024.0 * 1024.0);

void CpuidLeaf(std::uint32_t leaf, std::uint32_t out[4]) noexcept {
    out[0] = out[1] = out[2] = out[3] = 0;
#if defined(_MSC_VER)
    int regs[4]{};
    __cpuid(regs, static_cast<int>(leaf));
    for (int i = 0; i < 4; ++i) {
        out[i] = static_cast<std::uint32_t>(regs[i]);
    }
#elif defined(__x86_64__) || defined(__i386__)
    __get_cpuid(leaf, &out[0], &out[1], &out[2], &out[3]);
#endif
}

std::string ReadCpuBrandString() {
    std::uint32_t regs[4]{};
    CpuidLeaf(0x80000000u, regs);
    if (regs[0] < 0x80000004u) {
        return "Unknown x86_64 CPU";
    }
    char raw[49]{};
    for (std::uint32_t i = 0; i < 3u; ++i) {
        CpuidLeaf(0x80000002u + i, regs);
        std::memcpy(raw + i * 16u, regs, 16u);
    }
    raw[48] = '\0';
    // Collapse leading/trailing/repeated spaces for a clean compact log line.
    std::string clean;
    clean.reserve(48);
    bool prev_space = true;
    for (char c : std::string_view(raw)) {
        if (c == '\0') {
            break;
        }
        const bool is_sp = (c == ' ' || c == '\t' || c == '\r' || c == '\n');
        if (is_sp) {
            if (!prev_space) {
                clean.push_back(' ');
            }
            prev_space = true;
        } else {
            clean.push_back(c);
            prev_space = false;
        }
    }
    while (!clean.empty() && clean.back() == ' ') {
        clean.pop_back();
    }
    return clean.empty() ? "Unknown x86_64 CPU" : clean;
}

#ifndef _WIN32
bool ReadFirstLine(const char* path, char* buf, std::size_t cap) noexcept {
    if (!buf || cap == 0) {
        return false;
    }
    buf[0] = '\0';
    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        return false;
    }
    const ssize_t n = ::read(fd, buf, cap - 1);
    ::close(fd);
    if (n <= 0) {
        return false;
    }
    buf[n] = '\0';
    for (ssize_t i = 0; i < n; ++i) {
        if (buf[i] == '\n' || buf[i] == '\r') {
            buf[i] = '\0';
            break;
        }
    }
    return buf[0] != '\0';
}

bool ReadLinuxMemInfo(std::uint64_t& total_kb, std::uint64_t& avail_kb,
                      std::uint64_t& swap_total_kb, std::uint64_t& swap_free_kb) noexcept {
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) {
        return false;
    }
    total_kb = avail_kb = swap_total_kb = swap_free_kb = 0;
    bool got_total = false, got_avail = false;
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
        unsigned long long v = 0;
        if (!got_total && std::sscanf(line, "MemTotal: %llu kB", &v) == 1) {
            total_kb = v;
            got_total = true;
        } else if (!got_avail && std::sscanf(line, "MemAvailable: %llu kB", &v) == 1) {
            avail_kb = v;
            got_avail = true;
        } else if (std::sscanf(line, "SwapTotal: %llu kB", &v) == 1) {
            swap_total_kb = v;
        } else if (std::sscanf(line, "SwapFree: %llu kB", &v) == 1) {
            swap_free_kb = v;
        }
    }
    std::fclose(f);
    return got_total;
}

int CountLinuxPhysicalCores() noexcept {
    std::set<std::pair<int, int>> unique_cores;
    DIR* dir = ::opendir("/sys/devices/system/cpu");
    if (!dir) {
        return 0;
    }
    char path[160];
    char buf[32];
    while (struct dirent* ent = ::readdir(dir)) {
        if (std::strncmp(ent->d_name, "cpu", 3) != 0 ||
            ent->d_name[3] < '0' || ent->d_name[3] > '9') {
            continue;
        }
        int pkg = 0;
        int core = -1;
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/%s/topology/physical_package_id",
                      ent->d_name);
        if (ReadFirstLine(path, buf, sizeof(buf))) {
            pkg = std::atoi(buf);
        }
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/%s/topology/core_id",
                      ent->d_name);
        if (ReadFirstLine(path, buf, sizeof(buf))) {
            core = std::atoi(buf);
        }
        if (core >= 0) {
            unique_cores.emplace(pkg, core);
        }
    }
    ::closedir(dir);
    return static_cast<int>(unique_cores.size());
}

std::string FindLinuxCpuTempPath() {
    // Prefer dedicated CPU package/Tctl sensors (AMD k10temp/zenpower, Intel coretemp) over acpitz.
    constexpr const char* kPreferred[] = {"k10temp", "zenpower", "coretemp", "cpu_thermal", "acpitz"};
    std::string fallback;
    int best_rank = 999;

    DIR* dir = ::opendir("/sys/class/hwmon");
    if (!dir) {
        return {};
    }
    char name_path[192];
    char name_buf[64];
    char label_path[192];
    char label_buf[64];
    char temp_path[192];

    while (struct dirent* ent = ::readdir(dir)) {
        if (std::strncmp(ent->d_name, "hwmon", 5) != 0) {
            continue;
        }
        std::snprintf(name_path, sizeof(name_path), "/sys/class/hwmon/%s/name", ent->d_name);
        if (!ReadFirstLine(name_path, name_buf, sizeof(name_buf))) {
            continue;
        }
        int rank = -1;
        for (int i = 0; i < int(sizeof(kPreferred) / sizeof(kPreferred[0])); ++i) {
            if (std::strcmp(name_buf, kPreferred[i]) == 0) {
                rank = i;
                break;
            }
        }
        if (rank < 0 || rank >= best_rank) {
            continue;
        }
        // Within this hwmon, prefer Tctl / Package id 0 / Tdie, else temp1_input.
        std::string chosen;
        for (int t = 1; t <= 8; ++t) {
            std::snprintf(temp_path, sizeof(temp_path), "/sys/class/hwmon/%s/temp%d_input",
                          ent->d_name, t);
            if (::access(temp_path, R_OK) != 0) {
                continue;
            }
            if (chosen.empty()) {
                chosen = temp_path;
            }
            std::snprintf(label_path, sizeof(label_path), "/sys/class/hwmon/%s/temp%d_label",
                          ent->d_name, t);
            if (ReadFirstLine(label_path, label_buf, sizeof(label_buf))) {
                if (std::strcmp(label_buf, "Tctl") == 0 ||
                    std::strcmp(label_buf, "Tdie") == 0 ||
                    std::strncmp(label_buf, "Package id", 10) == 0) {
                    chosen = temp_path;
                    break;
                }
            }
        }
        if (!chosen.empty()) {
            best_rank = rank;
            fallback = std::move(chosen);
        }
    }
    ::closedir(dir);
    return fallback;
}

int SampleLinuxCpuClockMhz(int logical_cpus) noexcept {
    // Fast average of cpufreq scaling_cur_freq across online logical cores (matches Mission Center).
    const int limit = std::min(std::max(logical_cpus, 1), 64);
    std::uint64_t sum_khz = 0;
    int count = 0;
    char path[96];
    char buf[32];
    for (int i = 0; i < limit; ++i) {
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq",
                      i);
        if (ReadFirstLine(path, buf, sizeof(buf))) {
            const unsigned long long khz = std::strtoull(buf, nullptr, 10);
            if (khz > 100000ULL && khz < 10000000ULL) {
                sum_khz += khz;
                ++count;
            }
        }
    }
    if (count > 0) {
        return static_cast<int>((sum_khz / static_cast<std::uint64_t>(count)) / 1000ULL);
    }
    return 0;
}
#endif

} // namespace

std::string RedactHomePath(std::string_view path) {
    if (path.empty()) {
        return {};
    }
#ifdef _WIN32
    const char* home = std::getenv("USERPROFILE");
    if (home && home[0] != '\0') {
        const std::size_t hlen = std::strlen(home);
        if (path.size() >= hlen) {
            bool match = true;
            for (std::size_t i = 0; i < hlen; ++i) {
                const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(path[i])));
                const char b = static_cast<char>(std::tolower(static_cast<unsigned char>(home[i])));
                if (a != b && !((a == '/' || a == '\\') && (b == '/' || b == '\\'))) {
                    match = false;
                    break;
                }
            }
            if (match && (path.size() == hlen || path[hlen] == '\\' || path[hlen] == '/')) {
                std::string out = "%USERPROFILE%";
                out.append(path.substr(hlen));
                return out;
            }
        }
    }
#else
    const char* home = std::getenv("HOME");
    if (home && home[0] == '/' && home[1] != '\0') {
        std::string_view hv{home};
        while (hv.size() > 1 && hv.back() == '/') {
            hv.remove_suffix(1);
        }
        if (path.size() >= hv.size() && path.substr(0, hv.size()) == hv &&
            (path.size() == hv.size() || path[hv.size()] == '/')) {
            std::string out = "~";
            out.append(path.substr(hv.size()));
            return out;
        }
    }
#endif
    return std::string{path};
}

std::string FormatActiveUpscalerLabel() {
    const auto& s = BbSettings::Get();
    const int eff_up = BbSettings::EffectiveUpscaler();
    const int aa_mode = s.aa_mode.load(std::memory_order_relaxed);
    const int cur_p = s.dlss_preset.load(std::memory_order_relaxed);
    const bool dlaa_cnn =
        (s.upscaler.load(std::memory_order_relaxed) == BbSettings::UpscalerOff &&
         aa_mode == BbSettings::AaDlaaCnn) ||
        cur_p == 6 || cur_p == 5;
    const char* base_aa_str =
        eff_up == BbSettings::UpscalerFsr3     ? "FSR 3.1"
        : eff_up == BbSettings::UpscalerFsr4   ? "FSR 4"
        : eff_up == BbSettings::UpscalerFsr411 ? "FSR 4.1.1"
        : eff_up == BbSettings::UpscalerTaa
            ? (aa_mode == BbSettings::AaSmaaT2x ? "SMAA T2x" : "TAA")
        : eff_up == BbSettings::UpscalerDlss
            ? (BbSettings::RenderPreset() == BbSettings::NativeAA
                   ? (dlaa_cnn ? "DLAA 3 CNN" : "DLAA 4")
                   : (cur_p == 13                  ? "DLSS 4.5 M"
                      : cur_p == 12                ? "DLSS 4.5 L"
                      : (cur_p == 6 || cur_p == 5) ? "DLSS 3 CNN"
                      : cur_p != 0                 ? "DLSS 4 K"
                                                   : "DLSS Auto"))
            : (aa_mode == BbSettings::AaSmaaT2x               ? "SMAA T2x"
               : aa_mode == BbSettings::AaSmaa                ? "SMAA 1x"
               : s.effects[4].load(std::memory_order_relaxed) ? "FXAA"
                                                              : "Off");
    const bool rcas_on = s.sharpen.load(std::memory_order_relaxed) &&
                         s.sharpness.load(std::memory_order_relaxed) > 0.01f;
    if (rcas_on) {
        return std::string(base_aa_str) + " + RCAS";
    }
    return std::string(base_aa_str);
}

namespace {

std::mutex g_settings_log_mutex;
std::string g_last_logged_settings;

std::string BuildActiveSettingsLine() {
    const auto& s = BbSettings::Get();
    const auto out = BbSettings::CurrentOutput();
    const int rw = s.active_render_width.load(std::memory_order_relaxed);
    const int rh = s.active_render_height.load(std::memory_order_relaxed);
    const int win_mode = s.window_mode.load(std::memory_order_relaxed);
    const char* win_str = win_mode == BbSettings::WindowModeFullscreen   ? "Fullscreen"
                          : win_mode == BbSettings::WindowModeBorderless ? "Borderless"
                                                                         : "Windowed";
    const char* vsync_str =
        BbSettings::PresentModeKey(s.vsync_mode.load(std::memory_order_relaxed));
    const int cap = BbFrameRate::Limit();
    const std::string up_label = FormatActiveUpscalerLabel();
    const char* preset_str = BbSettings::PresetName(BbSettings::RenderPreset());
    const int dlss_p = s.dlss_preset.load(std::memory_order_relaxed);
    const int af = s.anisotropic.load(std::memory_order_relaxed);
    const bool nat_pfx = s.native_postfx.load(std::memory_order_relaxed);
    const int dof_on = s.effects[1].load(std::memory_order_relaxed) ? 1 : 0;
    const int mb_cam = s.effects[2].load(std::memory_order_relaxed)
                           ? s.motion_blur_camera.load(std::memory_order_relaxed)
                           : 0;
    const int mb_obj = s.effects[2].load(std::memory_order_relaxed)
                           ? s.motion_blur_object.load(std::memory_order_relaxed)
                           : 0;
    const int bloom_on = s.effects[7].load(std::memory_order_relaxed) ? 1 : 0;
    const int haze_on = s.fire_heat_haze.load(std::memory_order_relaxed) ? 1 : 0;
    const int ca_on = s.effects[0].load(std::memory_order_relaxed) ? 1 : 0;
    const int ssao_on = s.effects[3].load(std::memory_order_relaxed) ? 1 : 0;
    const int ssr_on = s.effects[6].load(std::memory_order_relaxed) ? 1 : 0;
    const int ssgi_on = s.ssgi_enabled.load(std::memory_order_relaxed) ? 1 : 0;
    const int pcss_on = s.pcss_shadows.load(std::memory_order_relaxed) ? 1 : 0;
    const int hdr_on = s.hdr_output.load(std::memory_order_relaxed) ? 1 : 0;
    const int fg_on = s.frame_generation.load(std::memory_order_relaxed) ? 1 : 0;
    const int jitter_on = s.jitter.load(std::memory_order_relaxed) ? 1 : 0;

    char buf[512];
    std::snprintf(
        buf, sizeof(buf),
        "Active Settings: %dx%d -> %dx%d (%s, %s, Cap %d) | Up: %s (%s, P=%d, Jit=%d, FG=%d) | "
        "PostFX: %s (DoF=%d MB=%d/%d Bloom=%d Haze=%d CA=%d) | "
        "SSAO=%d SSR=%d SSGI=%d PCSS=%d HDR=%d AF=%dx",
        rw, rh, out.width, out.height, win_str, vsync_str, cap, up_label.c_str(), preset_str,
        dlss_p, jitter_on, fg_on, nat_pfx ? "Native" : "Yebis2", dof_on, mb_cam, mb_obj, bloom_on,
        haze_on, ca_on, ssao_on, ssr_on, ssgi_on, pcss_on, hdr_on, af);
    return std::string(buf);
}

} // namespace

void LogSettingsIfChanged() {
    std::string cur = BuildActiveSettingsLine();
    std::scoped_lock lk{g_settings_log_mutex};
    if (g_last_logged_settings.empty()) {
        g_last_logged_settings = std::move(cur);
        return;
    }
    if (cur == g_last_logged_settings) {
        return;
    }
    g_last_logged_settings = cur;
    std::printf("[Settings] %s\n", cur.c_str());
    std::fflush(stdout);
}

void LogSystemPassport() {
    static std::once_flag once;
    std::call_once(once, [] {
        const std::string cpu_brand = ReadCpuBrandString();
        const int logical_cpus = std::max<int>(1, static_cast<int>(std::thread::hardware_concurrency()));

#ifdef _WIN32
        // 1. OS version via RtlGetVersion (accurate Windows 10/11 build without manifest quirks)
        struct RtlOsVerInfoW {
            unsigned long dwOSVersionInfoSize;
            unsigned long dwMajorVersion;
            unsigned long dwMinorVersion;
            unsigned long dwBuildNumber;
            unsigned long dwPlatformId;
            wchar_t szCSDVersion[128];
        };
        using FnRtlGetVersion = long(__stdcall*)(RtlOsVerInfoW*);
        std::string os_str = "Windows";
        if (HMODULE ntdll = GetModuleHandleA("ntdll.dll")) {
            if (auto fn = reinterpret_cast<FnRtlGetVersion>(GetProcAddress(ntdll, "RtlGetVersion"))) {
                RtlOsVerInfoW vi{};
                vi.dwOSVersionInfoSize = sizeof(vi);
                if (fn(&vi) == 0) {
                    const char* win_name =
                        (vi.dwMajorVersion == 10 && vi.dwBuildNumber >= 22000) ? "Windows 11"
                        : (vi.dwMajorVersion == 10)                            ? "Windows 10"
                                                                               : "Windows";
                    char tmp[96];
                    std::snprintf(tmp, sizeof(tmp), "%s (NT %lu.%lu Build %lu)", win_name,
                                  vi.dwMajorVersion, vi.dwMinorVersion, vi.dwBuildNumber);
                    os_str = tmp;
                }
            }
        }
        std::printf("System OS: %s | Session: Win32 | Arch: x86_64\n", os_str.c_str());

        // 2. CPU cores (physical via GetLogicalProcessorInformationEx)
        int phys_cores = 0;
        DWORD buf_len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buf_len);
        if (buf_len > 0) {
            std::string buf(buf_len, '\0');
            if (GetLogicalProcessorInformationEx(
                    RelationProcessorCore,
                    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
                    &buf_len)) {
                DWORD offset = 0;
                while (offset < buf_len) {
                    const auto* info =
                        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
                            buf.data() + offset);
                    if (info->Relationship == RelationProcessorCore) {
                        ++phys_cores;
                    }
                    if (info->Size == 0) {
                        break;
                    }
                    offset += info->Size;
                }
            }
        }
        if (phys_cores > 0) {
            std::printf("System CPU: %s | %d logical (%d phys)\n", cpu_brand.c_str(), logical_cpus,
                        phys_cores);
        } else {
            std::printf("System CPU: %s | %d logical\n", cpu_brand.c_str(), logical_cpus);
        }

        // 3. System RAM + Pagefile
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) {
            const double tot_gb = double(ms.ullTotalPhys) * kInvGiB;
            const double avail_gb = double(ms.ullAvailPhys) * kInvGiB;
            const std::uint64_t pf_tot_b =
                ms.ullTotalPageFile > ms.ullTotalPhys ? (ms.ullTotalPageFile - ms.ullTotalPhys) : 0;
            const std::uint64_t pf_avail_b =
                ms.ullAvailPageFile > ms.ullAvailPhys ? (ms.ullAvailPageFile - ms.ullAvailPhys) : 0;
            const double pf_tot_gb = double(pf_tot_b) * kInvGiB;
            const double pf_used_gb = pf_tot_b > pf_avail_b ? double(pf_tot_b - pf_avail_b) * kInvGiB : 0.0;
            std::printf("System RAM: %.1f GiB (Avail %.1f GiB) | Pagefile: %.1f/%.1f GiB\n",
                        tot_gb, avail_gb, pf_used_gb, pf_tot_gb);
        }
#else
        // 1. Linux OS distro + kernel + display session (strictly no hostname / username)
        std::string distro = "Linux";
        if (FILE* f = std::fopen("/etc/os-release", "r")) {
            char line[256];
            while (std::fgets(line, sizeof(line), f)) {
                if (std::strncmp(line, "PRETTY_NAME=", 12) == 0) {
                    std::string val{line + 12};
                    while (!val.empty() && (val.back() == '\n' || val.back() == '\r' || val.back() == '"')) {
                        val.pop_back();
                    }
                    if (!val.empty() && val.front() == '"') {
                        val.erase(val.begin());
                    }
                    if (!val.empty()) {
                        distro = std::move(val);
                    }
                    break;
                }
            }
            std::fclose(f);
        }
        struct utsname un{};
        const char* kernel = (::uname(&un) == 0) ? un.release : "unknown";
        const char* arch = (un.machine[0] != '\0') ? un.machine : "x86_64";
        const char* session = "unknown";
        if (const char* gs = std::getenv("GAMESCOPE_WAYLAND_DISPLAY"); gs && gs[0]) {
            session = "Gamescope";
        } else if (const char* st = std::getenv("XDG_SESSION_TYPE"); st && st[0]) {
            session = st;
        } else if (const char* wl = std::getenv("WAYLAND_DISPLAY"); wl && wl[0]) {
            session = "wayland";
        } else if (const char* x11 = std::getenv("DISPLAY"); x11 && x11[0]) {
            session = "x11";
        }
        std::printf("System OS: %s | Kernel %s | Session: %s | Arch: %s\n",
                    distro.c_str(), kernel, session, arch);

        // 2. CPU brand + cores + scaling governor + boost
        const int phys_cores = CountLinuxPhysicalCores();
        char gov[48] = "n/a";
        ReadFirstLine("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", gov, sizeof(gov));
        char boost_buf[16]{};
        const char* boost_str = "n/a";
        if (ReadFirstLine("/sys/devices/system/cpu/cpufreq/boost", boost_buf, sizeof(boost_buf))) {
            boost_str = (boost_buf[0] == '1') ? "on" : "off";
        } else if (ReadFirstLine("/sys/devices/system/cpu/intel_pstate/no_turbo", boost_buf,
                                 sizeof(boost_buf))) {
            boost_str = (boost_buf[0] == '0') ? "on" : "off";
        } else if (ReadFirstLine("/sys/devices/system/cpu/cpu0/cpufreq/boost", boost_buf,
                                 sizeof(boost_buf))) {
            boost_str = (boost_buf[0] == '1') ? "on" : "off";
        }
        if (phys_cores > 0) {
            std::printf("System CPU: %s | %d logical (%d phys) | Gov: %s | Boost: %s\n",
                        cpu_brand.c_str(), logical_cpus, phys_cores, gov, boost_str);
        } else {
            std::printf("System CPU: %s | %d logical | Gov: %s | Boost: %s\n",
                        cpu_brand.c_str(), logical_cpus, gov, boost_str);
        }

        // 3. System RAM + Swap
        std::uint64_t mem_tot_kb = 0, mem_avail_kb = 0, sw_tot_kb = 0, sw_free_kb = 0;
        if (ReadLinuxMemInfo(mem_tot_kb, mem_avail_kb, sw_tot_kb, sw_free_kb)) {
            constexpr double kKbToGiB = 1.0 / (1024.0 * 1024.0);
            const double tot_gb = double(mem_tot_kb) * kKbToGiB;
            const double avail_gb = double(mem_avail_kb) * kKbToGiB;
            const double sw_tot_gb = double(sw_tot_kb) * kKbToGiB;
            const double sw_used_gb =
                sw_tot_kb > sw_free_kb ? double(sw_tot_kb - sw_free_kb) * kKbToGiB : 0.0;
            std::printf("System RAM: %.1f GiB (Avail %.1f GiB) | Swap: %.1f/%.1f GiB\n",
                        tot_gb, avail_gb, sw_used_gb, sw_tot_gb);
        }
#endif

        // 4. Compact Active Graphics & Engine Settings Summary (1 line)
        const std::string settings_line = BuildActiveSettingsLine();
        {
            std::scoped_lock lk{g_settings_log_mutex};
            g_last_logged_settings = settings_line;
        }
        std::printf("%s\n", settings_line.c_str());
        std::fflush(stdout);
    });
}

void PollLiveSystemSensors(LiveSysSensors& out) {
    out = LiveSysSensors{};

#ifdef _WIN32
    // 1. System & Process RAM (uses K32GetProcessMemoryInfo dynamically so no -lpsapi link needed)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms) && ms.ullTotalPhys > 0) {
        out.ram_sys_total_gb = float(double(ms.ullTotalPhys) * kInvGiB);
        const std::uint64_t used_b =
            ms.ullTotalPhys > ms.ullAvailPhys ? (ms.ullTotalPhys - ms.ullAvailPhys) : 0;
        out.ram_sys_used_gb = float(double(used_b) * kInvGiB);
        out.has_ram = true;
    }
    struct ProcMemCounters {
        DWORD cb;
        DWORD PageFaultCount;
        SIZE_T PeakWorkingSetSize;
        SIZE_T WorkingSetSize;
        SIZE_T QuotaPeakPagedPoolUsage;
        SIZE_T QuotaPagedPoolUsage;
        SIZE_T QuotaPeakNonPagedPoolUsage;
        SIZE_T QuotaNonPagedPoolUsage;
        SIZE_T PagefileUsage;
        SIZE_T PeakPagefileUsage;
    };
    using FnGetProcMem = BOOL(__stdcall*)(HANDLE, ProcMemCounters*, DWORD);
    static FnGetProcMem fn_get_mem = []() -> FnGetProcMem {
        if (HMODULE k32 = GetModuleHandleA("kernel32.dll")) {
            if (auto f = reinterpret_cast<FnGetProcMem>(GetProcAddress(k32, "K32GetProcessMemoryInfo"))) {
                return f;
            }
        }
        if (HMODULE ps = LoadLibraryA("psapi.dll")) {
            return reinterpret_cast<FnGetProcMem>(GetProcAddress(ps, "GetProcessMemoryInfo"));
        }
        return nullptr;
    }();
    if (fn_get_mem) {
        ProcMemCounters pmc{};
        pmc.cb = sizeof(pmc);
        if (fn_get_mem(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            out.ram_proc_gb = float(double(pmc.WorkingSetSize) * kInvGiB);
            out.has_ram = true;
        }
    }

    // 2. System & Process CPU % via GetSystemTimes + GetProcessTimes deltas
    auto ft_to_u64 = [](const FILETIME& ft) -> std::uint64_t {
        return (static_cast<std::uint64_t>(ft.dwHighDateTime) << 32) |
               static_cast<std::uint64_t>(ft.dwLowDateTime);
    };
    static std::uint64_t prev_sys_idle = 0, prev_sys_total = 0;
    static std::uint64_t prev_proc_time = 0;
    static bool cpu_seeded = false;

    FILETIME ft_idle{}, ft_kern{}, ft_user{};
    FILETIME ft_create{}, ft_exit{}, ft_pkern{}, ft_puser{};
    if (GetSystemTimes(&ft_idle, &ft_kern, &ft_user)) {
        const std::uint64_t cur_idle = ft_to_u64(ft_idle);
        const std::uint64_t cur_total = ft_to_u64(ft_kern) + ft_to_u64(ft_user);
        std::uint64_t cur_proc = 0;
        const bool have_proc =
            GetProcessTimes(GetCurrentProcess(), &ft_create, &ft_exit, &ft_pkern, &ft_puser) != 0;
        if (have_proc) {
            cur_proc = ft_to_u64(ft_pkern) + ft_to_u64(ft_puser);
        }
        if (cpu_seeded && cur_total > prev_sys_total) {
            const std::uint64_t d_tot = cur_total - prev_sys_total;
            const std::uint64_t d_idle = cur_idle >= prev_sys_idle ? (cur_idle - prev_sys_idle) : 0;
            const std::uint64_t d_busy = d_tot > d_idle ? (d_tot - d_idle) : 0;
            out.sys_cpu_pct = std::clamp(int((double(d_busy) * 100.0 / double(d_tot)) + 0.5), 0, 100);
            if (have_proc && cur_proc >= prev_proc_time) {
                const std::uint64_t d_proc = cur_proc - prev_proc_time;
                out.proc_cpu_pct =
                    std::clamp(int((double(d_proc) * 100.0 / double(d_tot)) + 0.5), 0, 100);
            }
        }
        prev_sys_idle = cur_idle;
        prev_sys_total = cur_total;
        if (have_proc) {
            prev_proc_time = cur_proc;
        }
        cpu_seeded = true;
    }

    // 3. Live Average CPU Clock (MHz) via CallNtPowerInformation(ProcessorInformation) — no admin/driver needed
    {
        struct WinProcPowerInfo {
            ULONG Number;
            ULONG MaxMhz;
            ULONG CurrentMhz;
            ULONG MhzLimit;
            ULONG MaxIdleState;
            ULONG CurrentIdleState;
        };
        using FnCallNtPowerInfo = LONG(__stdcall*)(int, PVOID, ULONG, PVOID, ULONG);
        static FnCallNtPowerInfo fn_power_info = []() -> FnCallNtPowerInfo {
            if (HMODULE lib = LoadLibraryA("powrprof.dll")) {
                return reinterpret_cast<FnCallNtPowerInfo>(
                    GetProcAddress(lib, "CallNtPowerInformation"));
            }
            return nullptr;
        }();
        static const DWORD win_logical_cpus = []() -> DWORD {
            SYSTEM_INFO si{};
            GetNativeSystemInfo(&si);
            return si.dwNumberOfProcessors > 0 ? si.dwNumberOfProcessors : 1u;
        }();
        if (fn_power_info && win_logical_cpus > 0 && win_logical_cpus <= 256u) {
            std::vector<WinProcPowerInfo> info(win_logical_cpus);
            const ULONG bytes = static_cast<ULONG>(info.size() * sizeof(WinProcPowerInfo));
            // InformationLevel 11 == ProcessorInformation
            if (fn_power_info(11, nullptr, 0, info.data(), bytes) == 0) {
                std::uint64_t sum_mhz = 0;
                int count = 0;
                for (const auto& p : info) {
                    if (p.CurrentMhz > 100 && p.CurrentMhz < 10000) {
                        sum_mhz += p.CurrentMhz;
                        ++count;
                    }
                }
                if (count > 0) {
                    out.cpu_clock_mhz = static_cast<int>((sum_mhz + count / 2) / count);
                }
            }
        }
    }
#else
    static const long page_size = std::max<long>(4096, ::sysconf(_SC_PAGESIZE));
    static const int logical_cpus = std::max<int>(1, static_cast<int>(::sysconf(_SC_NPROCESSORS_ONLN)));

    // 1. Process RSS RAM (/proc/self/statm: size resident shared ...)
    {
        char buf[128];
        if (ReadFirstLine("/proc/self/statm", buf, sizeof(buf))) {
            unsigned long long size_p = 0, res_p = 0;
            if (std::sscanf(buf, "%llu %llu", &size_p, &res_p) == 2 && res_p > 0) {
                out.ram_proc_gb = float(double(res_p) * double(page_size) * kInvGiB);
                out.has_ram = true;
            }
        }
    }

    // 2. System RAM used/total (/proc/meminfo: MemTotal - MemAvailable, matches Mission Center)
    {
        std::uint64_t tot_kb = 0, avail_kb = 0, sw_tot = 0, sw_free = 0;
        if (ReadLinuxMemInfo(tot_kb, avail_kb, sw_tot, sw_free) && tot_kb > 0) {
            constexpr double kKbToGiB = 1.0 / (1024.0 * 1024.0);
            out.ram_sys_total_gb = float(double(tot_kb) * kKbToGiB);
            const std::uint64_t used_kb = tot_kb > avail_kb ? (tot_kb - avail_kb) : 0;
            out.ram_sys_used_gb = float(double(used_kb) * kKbToGiB);
            out.has_ram = true;
        }
    }

    // 3. System CPU % (/proc/stat) + Process CPU % (/proc/self/stat)
    {
        static std::uint64_t prev_sys_tot = 0, prev_sys_idle = 0, prev_proc_ticks = 0;
        static bool seeded = false;

        std::uint64_t cur_sys_tot = 0, cur_sys_idle = 0;
        bool have_sys = false;
        char stat_line[256];
        if (ReadFirstLine("/proc/stat", stat_line, sizeof(stat_line)) &&
            std::strncmp(stat_line, "cpu ", 4) == 0) {
            unsigned long long u = 0, n = 0, s = 0, id = 0, iow = 0, irq = 0, sirq = 0, st = 0;
            const int matched =
                std::sscanf(stat_line + 4, "%llu %llu %llu %llu %llu %llu %llu %llu", &u, &n, &s,
                            &id, &iow, &irq, &sirq, &st);
            if (matched >= 4) {
                cur_sys_idle = id + iow;
                cur_sys_tot = u + n + s + id + iow + irq + sirq + st;
                have_sys = true;
            }
        }

        std::uint64_t cur_proc_ticks = 0;
        bool have_proc = false;
        char self_buf[512];
        if (ReadFirstLine("/proc/self/stat", self_buf, sizeof(self_buf))) {
            if (const char* rparen = std::strrchr(self_buf, ')'); rparen && rparen[1] != '\0') {
                const char* p = rparen + 2;
                for (int f = 0; f < 11 && *p; ++f) {
                    while (*p == ' ') ++p;
                    while (*p && *p != ' ') ++p;
                }
                char* endp = nullptr;
                const unsigned long long ut = std::strtoull(p, &endp, 10);
                if (endp && endp != p) {
                    const unsigned long long st = std::strtoull(endp, &endp, 10);
                    cur_proc_ticks = ut + st;
                    have_proc = true;
                }
            }
        }

        if (have_sys && seeded && cur_sys_tot > prev_sys_tot) {
            const std::uint64_t d_tot = cur_sys_tot - prev_sys_tot;
            const std::uint64_t d_idle =
                cur_sys_idle >= prev_sys_idle ? (cur_sys_idle - prev_sys_idle) : 0;
            const std::uint64_t d_busy = d_tot > d_idle ? (d_tot - d_idle) : 0;
            out.sys_cpu_pct = std::clamp(int((double(d_busy) * 100.0 / double(d_tot)) + 0.5), 0, 100);
            if (have_proc && cur_proc_ticks >= prev_proc_ticks) {
                const std::uint64_t d_proc = cur_proc_ticks - prev_proc_ticks;
                out.proc_cpu_pct =
                    std::clamp(int((double(d_proc) * 100.0 / double(d_tot)) + 0.5), 0, 100);
            }
        }
        if (have_sys) {
            prev_sys_tot = cur_sys_tot;
            prev_sys_idle = cur_sys_idle;
            if (have_proc) {
                prev_proc_ticks = cur_proc_ticks;
            }
            seeded = true;
        }
    }

    // 4. Average CPU Clock (MHz) and Package/Tctl Temperature (C)
    out.cpu_clock_mhz = SampleLinuxCpuClockMhz(logical_cpus);
    static const std::string cpu_temp_path = FindLinuxCpuTempPath();
    if (!cpu_temp_path.empty()) {
        char tbuf[32];
        if (ReadFirstLine(cpu_temp_path.c_str(), tbuf, sizeof(tbuf))) {
            const long long millic = std::strtoll(tbuf, nullptr, 10);
            if (millic > 0 && millic < 200000) {
                out.cpu_temp_c = static_cast<int>((millic + 500) / 1000);
            }
        }
    }
#endif
}

} // namespace BbSysInfo
