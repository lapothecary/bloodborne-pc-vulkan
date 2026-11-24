// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Universal Hardware-Adaptive CPU Topology & Thread Affinity Scheduler.
// Automatically detects Multi-CCD / Multi-L3-Domain CPUs (AMD Ryzen 9 / Threadripper, including
// asymmetric 3D V-Cache X3D), Hybrid P-Core + E-Core CPUs (Intel 12th-15th Gen), Single-CCD
// Homogeneous CPUs (Ryzen 5/7, X3D, Intel 6c/8c), and Compact Handhelds (Steam Deck 4c/8t).

#pragma once

#include "bb_env.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#ifndef _WIN32
#include <dirent.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace BbThreads {

#ifndef _WIN32
/// Captures the OS/taskset/container process-wide CPU affinity mask once at DSO load time, BEFORE
/// ApplyProcessThreadTopology() pins MOMainThread to a single dedicated core. Without this,
/// calling sched_getaffinity(0, ...) from MOMainThread (or child threads spawned from it) after
/// topology pinning erroneously reported 1 hardware thread instead of 32.
inline const cpu_set_t& InitialProcessMask() {
    static const cpu_set_t mask = [] {
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_COUNT(&set) > 0) {
            return set;
        }
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        CPU_ZERO(&set);
        for (unsigned c = 0; c < hw && c < CPU_SETSIZE; ++c) {
            CPU_SET(static_cast<int>(c), &set);
        }
        return set;
    }();
    return mask;
}
#endif

/// Hardware threads the OS lets this process run on (respects taskset / containers).
inline unsigned Available() {
#ifdef _WIN32
    return bb_win32_available_cpus();
#else
    static const unsigned count =
        static_cast<unsigned>(std::max(1, CPU_COUNT(&InitialProcessMask())));
    return count;
#endif
}

#ifndef _WIN32
inline const unsigned kPrimeInitialProcessMask = Available();
#endif

/// The calling thread only runs on otherwise idle cores (falls back to the lowest nice level).
inline void MakeBackground() {
#ifdef _WIN32
    bb_win32_background_thread();
#else
    sched_param param{};
    if (sched_setscheduler(0, SCHED_IDLE, &param) != 0) {
        setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 19);
    }
#endif
}

/// Low-priority SCHED_OTHER helper on >=16-thread CPUs (never starved like SCHED_IDLE, while
/// yielding gracefully to nice-0 game and GPU threads); falls back to MakeBackground() on smaller CPUs.
inline void MakeLowPriority() {
#ifdef _WIN32
    bb_win32_background_thread();
#else
    if (Available() >= 16) {
        setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 10);
    } else {
        MakeBackground();
    }
#endif
}

/// Helper threads of the GPU side that the GPU command thread's progress depends on (Vulkan
/// recording threads, guest memory copy threads): a fault there must never wait for that thread
/// (a read trap's copy back goes through it for the game's threads). Set once per thread.
inline thread_local bool gpu_side_helper = false;

#ifndef _WIN32
inline std::atomic<int> g_gpu_comm_tid{0};
inline std::atomic<int> g_draw_rec_tid{0};
inline std::atomic<int> g_vk_recorder_tid{0};
inline std::atomic<int> g_present_tid{0};

/// Lowest (highest-priority) nice value permitted by the current process's RLIMIT_NICE ceiling.
/// Linux encodes RLIMIT_NICE as [1..40] mapping to nice [19..-20] via (20 - rlim_cur).
inline int MinAllowedNice() {
    static const int min_nice = [] {
        struct rlimit rl{};
        if (getrlimit(RLIMIT_NICE, &rl) == 0 && rl.rlim_cur > 20) {
            const long clamped = static_cast<long>(std::min<rlim_t>(rl.rlim_cur, 40));
            return static_cast<int>(std::clamp(20L - clamped, -20L, 0L));
        }
        return 0;
    }();
    return min_nice;
}

/// Raises a thread's scheduling priority within the OS RLIMIT_NICE ceiling so critical game and
/// RHI pacing threads outrank background desktop apps boosted by ananicy-cpp (nice -8/-4) while
/// staying strictly below realtime audio/compositor threads (pipewire -12, kwin_wayland SCHED_RR).
inline void BoostThreadPriority(int tid, int desired_nice) {
    if (tid <= 0 || desired_nice >= 0) {
        return;
    }
    const int target = std::max(desired_nice, MinAllowedNice());
    if (target < 0 && setpriority(PRIO_PROCESS, static_cast<id_t>(tid), target) == 0) {
        return;
    }
    setpriority(PRIO_PROCESS, static_cast<id_t>(tid), std::max(desired_nice, -2));
}

struct TopologyMasks {
    cpu_set_t game_main;    // MOMainThread (dedicated 2-core failover pool in CCD0, avoiding CPU 0)
    cpu_set_t gpu_comm;     // bb:GpuComm / Stage A (dedicated RHI critical pool in CCD1, sharing L3)
    cpu_set_t draw_rec;     // bb:DrawRec / Stage B (dedicated RHI critical pool in CCD1, sharing L3)
    cpu_set_t vk_recorder;  // bb:VkRecorder (dedicated RHI critical pool in CCD1, sharing L3)
    cpu_set_t present;      // bb:Present (dedicated RHI critical pool in CCD1 for sub-ms pacing)
    cpu_set_t game_workers; // HavokWorker*, CSChrThread*, CSClothThread*, SpCloth*, GXRender*
    cpu_set_t gx_workers;   // GXWorker* (game L3 domain `game_workers` by default; secondary GPU CCD with BB_GX_WORKERS_GPU_CCD=1)
    cpu_set_t gpu;          // bb:Resolve*, bb:TexBind, bb:VkRecorder1+, driver helper threads
    cpu_set_t helper;       // bb:Copy*, bb:DrawScan, EzWorkPool*, Core.Res.*, FMOD*, NGX/CUDA/SDL helpers
    bool enabled = false;
};

inline void ParseCpuList(const char* text, const cpu_set_t& allowed, cpu_set_t* out) {
    CPU_ZERO(out);
    for (const char* at = text; *at && *at != '\n';) {
        char* end = nullptr;
        const long first = std::strtol(at, &end, 10);
        if (end == at) {
            break;
        }
        long last = first;
        if (*end == '-') {
            last = std::strtol(end + 1, &end, 10);
        }
        for (long c = first; c <= last && c < CPU_SETSIZE; ++c) {
            if (c >= 0 && CPU_ISSET(c, &allowed)) {
                CPU_SET(c, out);
            }
        }
        at = (*end == ',') ? (end + 1) : end;
    }
}

inline void FormatCpuList(const cpu_set_t& set, char* out, std::size_t out_size) {
    if (out_size == 0) {
        return;
    }
    out[0] = '\0';
    std::size_t pos = 0;
    int first = -1;
    int prev = -1;
    const auto flush = [&](int a, int b) {
        if (a < 0) {
            return;
        }
        int n = 0;
        if (a == b) {
            n = std::snprintf(out + pos, out_size - pos, "%s%d", pos ? "," : "", a);
        } else {
            n = std::snprintf(out + pos, out_size - pos, "%s%d-%d", pos ? "," : "", a, b);
        }
        if (n > 0 && pos + std::size_t(n) < out_size) {
            pos += std::size_t(n);
        }
    };
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &set)) {
            if (first < 0) {
                first = prev = cpu;
            } else if (cpu == prev + 1) {
                prev = cpu;
            } else {
                flush(first, prev);
                first = prev = cpu;
            }
        }
    }
    flush(first, prev);
}

inline const TopologyMasks& GetTopology() {
    static const TopologyMasks topo = [] {
        TopologyMasks none{};
        CPU_ZERO(&none.game_main);
        CPU_ZERO(&none.gpu_comm);
        CPU_ZERO(&none.draw_rec);
        CPU_ZERO(&none.vk_recorder);
        CPU_ZERO(&none.present);
        CPU_ZERO(&none.game_workers);
        CPU_ZERO(&none.gx_workers);
        CPU_ZERO(&none.gpu);
        CPU_ZERO(&none.helper);

        const char* env_ccd = std::getenv("BB_GPU_CCD");
        const char* env_topo = std::getenv("BB_THREAD_TOPOLOGY");
        if ((env_ccd && (env_ccd[0] == 'o' || env_ccd[0] == '-')) ||
            (env_topo && (env_topo[0] == '0' || env_topo[0] == 'o' || env_topo[0] == '-'))) {
            return none;
        }

        const cpu_set_t allowed = InitialProcessMask();
        if (CPU_COUNT(&allowed) <= 1) {
            return none;
        }

        const auto read_text = [](const char* path, char* out, std::size_t size) {
            FILE* file = std::fopen(path, "r");
            if (!file) {
                return false;
            }
            const bool ok = std::fgets(out, int(size), file) != nullptr;
            std::fclose(file);
            return ok;
        };

        struct CpuInfo {
            bool valid = false;
            int primary_sibling = -1;
            cpu_set_t siblings{};
            cpu_set_t l3_set{};
            long l3_size_kb = 0;
            long pref_rank = 0;
            long capacity = 0;
            long max_freq = 0;
            long score = 0;
        };

        CpuInfo cpus[256]{};
        struct Domain {
            cpu_set_t cpus{};
            long l3_size_kb = 0;
            long best_score = 0;
            int phys_cores = 0;
        };
        Domain domains[16]{};
        int domain_count = 0;
        int total_phys_cores = 0;
        long max_score = 0;
        long min_score = 0x7fffffffL;

        for (int cpu = 0; cpu < 256 && cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &allowed)) {
                continue;
            }
            CpuInfo& info = cpus[cpu];
            info.valid = true;
            CPU_ZERO(&info.siblings);
            CPU_SET(cpu, &info.siblings);
            info.primary_sibling = cpu;

            char path[160], text[256];
            std::snprintf(path, sizeof(path),
                          "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
            if (read_text(path, text, sizeof(text))) {
                ParseCpuList(text, allowed, &info.siblings);
                for (int s = 0; s < 256; ++s) {
                    if (CPU_ISSET(s, &info.siblings)) {
                        info.primary_sibling = s;
                        break;
                    }
                }
            }
            if (info.primary_sibling == cpu) {
                ++total_phys_cores;
            }

            CPU_ZERO(&info.l3_set);
            std::snprintf(path, sizeof(path),
                          "/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", cpu);
            if (read_text(path, text, sizeof(text))) {
                ParseCpuList(text, allowed, &info.l3_set);
            } else {
                info.l3_set = allowed;
            }

            std::snprintf(path, sizeof(path),
                          "/sys/devices/system/cpu/cpu%d/cache/index3/size", cpu);
            if (read_text(path, text, sizeof(text))) {
                info.l3_size_kb = std::atol(text);
            }

            std::snprintf(path, sizeof(path),
                          "/sys/devices/system/cpu/cpu%d/cpufreq/amd_pstate_prefcore_ranking", cpu);
            if (read_text(path, text, sizeof(text))) {
                info.pref_rank = std::atol(text);
            }

            std::snprintf(path, sizeof(path),
                          "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu);
            if (read_text(path, text, sizeof(text))) {
                info.capacity = std::atol(text);
            }

            std::snprintf(path, sizeof(path),
                          "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
            if (read_text(path, text, sizeof(text))) {
                info.max_freq = std::atol(text) / 1000; // MHz
            }

            // Unified performance score for both AMD prefcore and Intel hybrid P/E capacity/freq.
            info.score = info.pref_rank * 10000L + info.capacity * 100L + info.max_freq;
            if (info.score == 0) {
                info.score = 1000L;
            }
            max_score = std::max(max_score, info.score);
            min_score = std::min(min_score, info.score);

            int found = -1;
            for (int d = 0; d < domain_count; ++d) {
                if (CPU_EQUAL(&domains[d].cpus, &info.l3_set)) {
                    found = d;
                    break;
                }
            }
            if (found < 0 && domain_count < 16) {
                found = domain_count++;
                domains[found].cpus = info.l3_set;
                domains[found].l3_size_kb = info.l3_size_kb;
                domains[found].best_score = info.score;
                domains[found].phys_cores = (info.primary_sibling == cpu) ? 1 : 0;
            } else if (found >= 0) {
                domains[found].l3_size_kb = std::max(domains[found].l3_size_kb, info.l3_size_kb);
                domains[found].best_score = std::max(domains[found].best_score, info.score);
                if (info.primary_sibling == cpu) {
                    ++domains[found].phys_cores;
                }
            }
        }

        if (domain_count == 0) {
            return none;
        }

        TopologyMasks result{};
        result.enabled = true;

        // Pick the best primary physical core from `pool`. When `avoid_cpu0` is true, skip CPU 0
        // (Linux's default boot/IRQ/kworker/nv_queue housekeeping core) whenever any non-0 core
        // in the same pool is within 95% of the top performance score.
        const auto pick_best_core = [&](const cpu_set_t& pool, bool avoid_cpu0) {
            long max_sc = -1;
            for (int c = 0; c < 256; ++c) {
                if (cpus[c].valid && CPU_ISSET(c, &pool) && cpus[c].primary_sibling == c) {
                    max_sc = std::max(max_sc, cpus[c].score);
                }
            }
            int best = -1;
            long best_sc = -1;
            if (avoid_cpu0 && max_sc > 0) {
                const long non0_floor = (max_sc * 95L) / 100L;
                for (int c = 1; c < 256; ++c) {
                    if (cpus[c].valid && CPU_ISSET(c, &pool) && cpus[c].primary_sibling == c &&
                        cpus[c].score >= non0_floor && cpus[c].score > best_sc) {
                        best_sc = cpus[c].score;
                        best = c;
                    }
                }
            }
            if (best < 0) {
                for (int c = 0; c < 256; ++c) {
                    if (cpus[c].valid && CPU_ISSET(c, &pool) && cpus[c].primary_sibling == c) {
                        if (cpus[c].score > best_sc) {
                            best_sc = cpus[c].score;
                            best = c;
                        }
                    }
                }
            }
            if (best < 0) {
                for (int c = 0; c < 256; ++c) {
                    if (cpus[c].valid && CPU_ISSET(c, &pool)) {
                        best = c;
                        break;
                    }
                }
            }
            return best;
        };

        const auto clear_siblings = [&](int primary, cpu_set_t* dst) {
            if (primary < 0 || !dst) {
                return;
            }
            for (int s = 0; s < 256; ++s) {
                if (CPU_ISSET(s, &cpus[primary].siblings)) {
                    CPU_CLR(s, dst);
                }
            }
        };

        // Carve `isolated_count` dedicated primary physical cores (excluding both their primary and
        // SMT siblings from `sub_a`/`sub_b`), plus `shared_failover_count` additional primary cores
        // added to `out` without removing them from worker pools so critical threads ALWAYS have
        // >= 2 physical cores in the same L3 domain and can escape external OS/compositor
        // preemptions (kwin_wayland SCHED_RR, Discord, kworker) in ~2 us instead of stalling 5-12 ms.
        const auto carve_critical_pool = [&](const cpu_set_t& src_pool, int isolated_count,
                                             int shared_failover_count, bool avoid_cpu0,
                                             cpu_set_t* out, cpu_set_t* sub_a, cpu_set_t* sub_b) {
            CPU_ZERO(out);
            cpu_set_t avail = src_pool;
            for (int i = 0; i < isolated_count; ++i) {
                const int c = pick_best_core(avail, avoid_cpu0);
                if (c < 0) {
                    break;
                }
                CPU_SET(c, out);
                clear_siblings(c, &avail);
                clear_siblings(c, sub_a);
                clear_siblings(c, sub_b);
            }
            for (int i = 0; i < shared_failover_count; ++i) {
                const int c = pick_best_core(avail, avoid_cpu0);
                if (c < 0) {
                    break;
                }
                CPU_SET(c, out);
                clear_siblings(c, &avail);
            }
            if (CPU_COUNT(out) == 0) {
                *out = src_pool;
            }
        };

        if (domain_count >= 2) {
            // =================================================================================
            // CASE 1: Multi-CCD / Multi-L3-Domain CPU (e.g. Ryzen 9 5900X/5950X/7900X/7950X/
            // 7950X3D/9900X/9950X/9950X3D, Threadripper).
            // - Primary L3 domain (largest L3 V-Cache first, then highest preferred-core score)
            //   is dedicated to the Guest Game Engine:
            //   * MOMainThread gets a dedicated 2-core failover pool avoiding CPU 0 (SMT siblings
            //     excluded from workers when >= 8 phys cores; 1 isolated + 1 failover on 6-core CCD)
            //     so it runs L1/L2-hot with zero SMT contention and instant ~2 us failover if a
            //     realtime compositor/IRQ preempts its primary core.
            //   * HavokWorker*, CSChrThread*, CSClothThread*, SpClothVertex*, GXRenderThread, and
            //     GXWorker* share the remaining cores of the EXACT SAME L3 domain (~4 ns L3 hit,
            //     0 Infinity Fabric crossings across per-frame fork-join barriers).
            // - Secondary L3 domain (CCD1) is dedicated to the Host Vulkan RHI Pipeline:
            //   * bb:GpuComm (Stage A), bb:DrawRec (Stage B), bb:VkRecorder, and bb:Present share
            //     a dedicated multi-core primary-only RHI Critical Pool (SMT siblings idle, ~10 ns
            //     same-L3 handoff, zero single-core preemption starvation from background desktop
            //     loads). Remaining CCD1 cores host bb:Resolve*, driver helpers, Core.Res.*, FMOD.
            // =================================================================================
            int game_dom = 0;
            for (int d = 1; d < domain_count; ++d) {
                const bool larger_l3 = domains[d].l3_size_kb > domains[game_dom].l3_size_kb;
                const bool same_l3_faster =
                    (domains[d].l3_size_kb == domains[game_dom].l3_size_kb) &&
                    (domains[d].best_score > domains[game_dom].best_score);
                if (larger_l3 || same_l3_faster) {
                    game_dom = d;
                }
            }
            int gpu_dom = (game_dom + 1) % domain_count;
            if (env_ccd && env_ccd[0] >= '0' && env_ccd[0] <= '9') {
                gpu_dom = std::clamp(std::atoi(env_ccd), 0, domain_count - 1);
                if (gpu_dom == game_dom && domain_count >= 2) {
                    game_dom = (gpu_dom + 1) % domain_count;
                }
            }

            const cpu_set_t& game_pool = domains[game_dom].cpus;
            const cpu_set_t& gpu_pool = domains[gpu_dom].cpus;

            result.game_workers = game_pool;
            if (domains[game_dom].phys_cores >= 8) {
                carve_critical_pool(game_pool, 2, 0, true, &result.game_main, &result.game_workers,
                                    nullptr);
            } else if (domains[game_dom].phys_cores >= 6) {
                carve_critical_pool(game_pool, 1, 1, true, &result.game_main, &result.game_workers,
                                    nullptr);
            } else {
                result.game_main = game_pool;
            }

            result.gpu = gpu_pool;
            result.helper = gpu_pool;

            cpu_set_t rhi_crit{};
            if (domains[gpu_dom].phys_cores >= 8) {
                carve_critical_pool(gpu_pool, 4, 0, true, &rhi_crit, &result.gpu, &result.helper);
            } else if (domains[gpu_dom].phys_cores >= 6) {
                carve_critical_pool(gpu_pool, 3, 0, true, &rhi_crit, &result.gpu, &result.helper);
            } else if (domains[gpu_dom].phys_cores >= 4) {
                carve_critical_pool(gpu_pool, 2, 0, true, &rhi_crit, &result.gpu, &result.helper);
            } else {
                rhi_crit = result.gpu;
            }
            result.gpu_comm = rhi_crit;
            result.draw_rec = rhi_crit;
            result.vk_recorder = rhi_crit;
            result.present = rhi_crit;

            const char* env_gx_gpu = std::getenv("BB_GX_WORKERS_GPU_CCD");
            const bool gx_on_gpu_ccd = env_gx_gpu && env_gx_gpu[0] == '1';
            if (gx_on_gpu_ccd && CPU_COUNT(&result.gpu) >= 4) {
                result.gx_workers = result.gpu;
            } else {
                result.gx_workers = result.game_workers;
            }
        } else {
            // Single L3 domain: check if Hybrid P-Core + E-Core (Intel 12th-15th Gen) or
            // Homogeneous (Ryzen 5/7, 5800X3D/7800X3D/9800X3D, Intel 6c/8c, Steam Deck).
            bool is_hybrid = false;
            cpu_set_t p_cores{}, e_cores{};
            CPU_ZERO(&p_cores);
            CPU_ZERO(&e_cores);
            int p_phys_cores = 0;

            if (min_score > 0 && max_score > (min_score * 115L) / 100L) {
                const long threshold = (max_score * 85L) / 100L;
                for (int c = 0; c < 256; ++c) {
                    if (!cpus[c].valid) {
                        continue;
                    }
                    if (cpus[c].score >= threshold) {
                        CPU_SET(c, &p_cores);
                        if (cpus[c].primary_sibling == c) {
                            ++p_phys_cores;
                        }
                    } else {
                        CPU_SET(c, &e_cores);
                    }
                }
                if (CPU_COUNT(&p_cores) >= 4 && CPU_COUNT(&e_cores) >= 2) {
                    is_hybrid = true;
                }
            }

            if (is_hybrid) {
                // =============================================================================
                // CASE 2: Single-L3 Hybrid P-Core + E-Core CPU (Intel 12th-15th Gen).
                // - MOMainThread gets a 2-P-core failover pool (avoiding CPU 0).
                // - Critical RHI threads stay strictly on P-cores (dedicated 2-3 P-core pool when >= 8P).
                // - Bursty game/GX workers allow E-core overflow on <= 7P CPUs so 14+ threads do not
                //   starve on 5 P-cores and stall main in ThreadSync; async/IO/FMOD stay on E-cores.
                // =============================================================================
                cpu_set_t p_remaining = p_cores;
                if (p_phys_cores >= 8) {
                    carve_critical_pool(p_cores, 2, 0, true, &result.game_main,
                                        &p_remaining, nullptr);
                    cpu_set_t rhi_crit{};
                    if (p_phys_cores >= 10) {
                        carve_critical_pool(p_remaining, 3, 0, true, &rhi_crit,
                                            &p_remaining, nullptr);
                    } else {
                        carve_critical_pool(p_remaining, 2, 1, true, &rhi_crit,
                                            &p_remaining, nullptr);
                    }
                    result.gpu_comm = rhi_crit;
                    result.draw_rec = rhi_crit;
                    result.vk_recorder = rhi_crit;
                    result.present = rhi_crit;
                } else if (p_phys_cores >= 6) {
                    carve_critical_pool(p_cores, 1, 1, true, &result.game_main,
                                        &p_remaining, nullptr);
                    result.gpu_comm = p_remaining;
                    result.draw_rec = p_remaining;
                    result.vk_recorder = p_remaining;
                    result.present = p_remaining;
                } else {
                    result.game_main = p_cores;
                    result.gpu_comm = p_cores;
                    result.draw_rec = p_cores;
                    result.vk_recorder = p_cores;
                    result.present = p_cores;
                }
                cpu_set_t p_plus_e{};
                CPU_OR(&p_plus_e, &p_remaining, &e_cores);
                result.gpu = p_remaining;
                result.game_workers = p_plus_e;
                result.gx_workers = (p_phys_cores >= 8) ? p_remaining : p_plus_e;
                result.helper = e_cores;
            } else if (total_phys_cores >= 6) {
                // =============================================================================
                // CASE 3: Single-L3 Homogeneous CPU with >= 6 Physical Cores
                // (e.g. Ryzen 5 5600X/7600X, Ryzen 7 5800X3D/7800X3D/9800X3D, Intel 6c/8c).
                // - MOMainThread always has a 2-core failover pool (avoiding CPU 0) so external
                //   OS/compositor preemptions never stall it on a single core.
                // - When >= 8 physical cores, critical RHI threads share a dedicated 2-3 core pool
                //   while leaving >= 5 physical cores (10 threads) for workers.
                // =============================================================================
                result.game_workers = allowed;
                if (total_phys_cores >= 10) {
                    carve_critical_pool(allowed, 2, 0, true, &result.game_main,
                                        &result.game_workers, nullptr);
                    cpu_set_t rhi_crit{};
                    carve_critical_pool(result.game_workers, 3, 0, true, &rhi_crit,
                                        &result.game_workers, nullptr);
                    result.gpu_comm = rhi_crit;
                    result.draw_rec = rhi_crit;
                    result.vk_recorder = rhi_crit;
                    result.present = rhi_crit;
                } else if (total_phys_cores >= 8) {
                    carve_critical_pool(allowed, 1, 1, true, &result.game_main,
                                        &result.game_workers, nullptr);
                    cpu_set_t rhi_crit{};
                    carve_critical_pool(result.game_workers, 2, 1, true, &rhi_crit,
                                        &result.game_workers, nullptr);
                    result.gpu_comm = rhi_crit;
                    result.draw_rec = rhi_crit;
                    result.vk_recorder = rhi_crit;
                    result.present = rhi_crit;
                } else {
                    carve_critical_pool(allowed, 1, 1, true, &result.game_main,
                                        &result.game_workers, nullptr);
                    result.gpu_comm = result.game_workers;
                    result.draw_rec = result.game_workers;
                    result.vk_recorder = result.game_workers;
                    result.present = result.game_workers;
                }
                result.gx_workers = result.game_workers;
                result.gpu = result.game_workers;
                result.helper = result.game_workers;
            } else {
                // =============================================================================
                // CASE 4: Compact CPU (< 6 physical cores, e.g. Steam Deck 4c/8t).
                // Keep all cores shared and rely on thread priorities (nice / IDLE).
                // =============================================================================
                result.game_main = allowed;
                result.gpu_comm = allowed;
                result.draw_rec = allowed;
                result.vk_recorder = allowed;
                result.present = allowed;
                result.game_workers = allowed;
                result.gx_workers = allowed;
                result.gpu = allowed;
                result.helper = allowed;
            }
        }

        char main_str[64], rhi_str[64], work_str[64], gxw_str[64], gpu_str[64], help_str[64];
        FormatCpuList(result.game_main, main_str, sizeof(main_str));
        FormatCpuList(result.gpu_comm, rhi_str, sizeof(rhi_str));
        FormatCpuList(result.game_workers, work_str, sizeof(work_str));
        FormatCpuList(result.gx_workers, gxw_str, sizeof(gxw_str));
        FormatCpuList(result.gpu, gpu_str, sizeof(gpu_str));
        FormatCpuList(result.helper, help_str, sizeof(help_str));
        const int min_nice = MinAllowedNice();
        const int crit_nice = std::max(-10, min_nice);
        const int work_nice = std::max(-6, min_nice);
        std::printf("CPU Topology (%d L3 domain%s, %d phys cores, nice %d/%d [rlimit %d]): "
                    "MOMainThread=[%s], RHI-Crit(StageA/B/VkRec/Present)=[%s], "
                    "Game/Havok=[%s], GXWorker=[%s], Vulkan RHI=[%s], Helpers/IO=[%s]\n",
                    domain_count, domain_count == 1 ? "" : "s", total_phys_cores,
                    crit_nice, work_nice, min_nice,
                    main_str, rhi_str, work_str, gxw_str, gpu_str, help_str);
        std::fflush(stdout);
        return result;
    }();
    return topo;
}

inline const TopologyMasks& GetL3Domains() {
    return GetTopology();
}

/// Pins the single Stage A DirectVk command processor thread (bb:GpuComm) to the dedicated
/// CCD1 RHI critical pool (sharing L3 with bb:DrawRec, bb:VkRecorder, bb:Present) and records
/// its TID so any child threads spawned by the Vulkan ICD get migrated onto `topo.gpu`.
inline void PinToGpuCommCore() {
    const int tid = int(::gettid());
    g_gpu_comm_tid.store(tid, std::memory_order_relaxed);
    const auto& domain = GetTopology().gpu_comm;
    if (CPU_COUNT(&domain) != 0) {
        sched_setaffinity(0, sizeof(domain), &domain);
    }
    BoostThreadPriority(tid, -10);
}

/// Pins the Stage B draw recording thread (bb:DrawRec) to the dedicated CCD1 RHI critical pool
/// and records its TID so child driver threads migrate onto `topo.gpu`.
inline void PinToDrawRecCore() {
    const int tid = int(::gettid());
    g_draw_rec_tid.store(tid, std::memory_order_relaxed);
    const auto& domain = GetTopology().draw_rec;
    if (CPU_COUNT(&domain) != 0) {
        sched_setaffinity(0, sizeof(domain), &domain);
    }
    BoostThreadPriority(tid, -10);
}

/// Pins the primary Vulkan command buffer recording + submission thread (bb:VkRecorder) to the
/// dedicated CCD1 RHI critical pool and records its TID so lazy ICD threads migrate onto `topo.gpu`.
inline void PinToVkRecorderCore() {
    const int tid = int(::gettid());
    g_vk_recorder_tid.store(tid, std::memory_order_relaxed);
    const auto& domain = GetTopology().vk_recorder;
    if (CPU_COUNT(&domain) != 0) {
        sched_setaffinity(0, sizeof(domain), &domain);
    }
    BoostThreadPriority(tid, -10);
}

/// Pins the frame-pacing presentation thread (bb:Present) to the dedicated CCD1 RHI critical pool
/// with high scheduling priority so PresentPacing deadlines never suffer multi-ms OS wakeup lag.
inline void PinToPresentCore() {
    const int tid = int(::gettid());
    g_present_tid.store(tid, std::memory_order_relaxed);
    const auto& domain = GetTopology().present;
    if (CPU_COUNT(&domain) != 0) {
        sched_setaffinity(0, sizeof(domain), &domain);
    }
    BoostThreadPriority(tid, -10);
}

/// Pins secondary host Vulkan pipeline threads (bb:Resolve*, bb:TexBind, bb:VkRecorder1+)
/// to the GPU domain (CCD1 on Multi-CCD Ryzen; P-cores on Intel Hybrid; non-MOMainThread cores on Single-CCD).
inline void PinToGpuDomain() {
    const auto& domain = GetTopology().gpu;
    if (CPU_COUNT(&domain) != 0) {
        sched_setaffinity(0, sizeof(domain), &domain);
    }
}

/// Pins auxiliary streaming/speculative helpers (bb:Copy*, bb:DrawScan, bb:SchedOps)
/// to the helper domain (CCD1 on Multi-CCD Ryzen; E-cores on Intel Hybrid; non-MOMainThread cores on Single-CCD).
inline void PinToHelperDomain() {
    const auto& domain = GetTopology().helper;
    if (CPU_COUNT(&domain) != 0) {
        sched_setaffinity(0, sizeof(domain), &domain);
    }
}

/// Scans `/proc/self/task` and applies the hardware-adaptive affinity masks and priorities to all
/// guest and host threads (MOMainThread, HavokWorkerThre, CSChrThread*, CSClothThread*,
/// SpClothVertexUp*, GXRenderThread, GXWorker*, EzWorkPool*, FMOD*, and bb:* host threads).
/// Safe and fast to call periodically (e.g. from StallWatch every 500 ms): caches `(tid, comm)`
/// so already-pinned threads incur zero syscalls once named.
inline void ApplyProcessThreadTopology() {
    const auto& topo = GetTopology();
    if (!topo.enabled) {
        return;
    }

    static std::mutex mtx;
    std::scoped_lock lock{mtx};

    struct SeenThread {
        int tid = 0;
        char comm[20]{};
    };
    static SeenThread seen[256]{};
    static int seen_count = 0;

    DIR* dir = ::opendir("/proc/self/task");
    if (!dir) {
        return;
    }

    const auto starts_with = [](const char* s, const char* prefix) {
        return std::strncmp(s, prefix, std::strlen(prefix)) == 0;
    };

    const int main_pid = int(::getpid());

    while (struct dirent* ent = ::readdir(dir)) {
        if (ent->d_name[0] < '0' || ent->d_name[0] > '9') {
            continue;
        }
        const int tid = std::atoi(ent->d_name);
        if (tid <= 0) {
            continue;
        }

        char path[64], comm[32]{};
        std::snprintf(path, sizeof(path), "/proc/self/task/%d/comm", tid);
        FILE* f = std::fopen(path, "r");
        if (!f) {
            continue;
        }
        const bool ok = std::fgets(comm, sizeof(comm), f) != nullptr;
        std::fclose(f);
        if (!ok) {
            continue;
        }
        for (char* p = comm; *p; ++p) {
            if (*p == '\n' || *p == '\r') {
                *p = '\0';
                break;
            }
        }
        const bool is_main_tid = (tid == main_pid);
        // Wait until non-main guest threads set their real pthread name (not the initial process name),
        // while immediately evacuating any unnamed host/driver/NGX child threads that inherited a
        // dedicated critical-pool mask from MOMainThread or the RHI critical threads onto `topo.helper`.
        if (!is_main_tid && (comm[0] == '\0' || std::strcmp(comm, "bb-probe") == 0 ||
                             std::strcmp(comm, "probe") == 0)) {
            if (CPU_COUNT(&topo.helper) > 0) {
                cpu_set_t cur_mask;
                CPU_ZERO(&cur_mask);
                if (sched_getaffinity(tid, sizeof(cur_mask), &cur_mask) == 0 &&
                    !CPU_EQUAL(&cur_mask, &topo.helper) &&
                    ((CPU_COUNT(&topo.game_main) > 0 && CPU_EQUAL(&cur_mask, &topo.game_main)) ||
                     (CPU_COUNT(&topo.gpu_comm) > 0 && CPU_EQUAL(&cur_mask, &topo.gpu_comm)) ||
                     (CPU_COUNT(&topo.draw_rec) > 0 && CPU_EQUAL(&cur_mask, &topo.draw_rec)) ||
                     (CPU_COUNT(&topo.vk_recorder) > 0 && CPU_EQUAL(&cur_mask, &topo.vk_recorder)) ||
                     (CPU_COUNT(&topo.present) > 0 && CPU_EQUAL(&cur_mask, &topo.present)))) {
                    sched_setaffinity(tid, sizeof(cpu_set_t), &topo.helper);
                }
            }
            continue;
        }

        int slot = -1;
        for (int i = 0; i < seen_count; ++i) {
            if (seen[i].tid == tid) {
                slot = i;
                break;
            }
        }
        if (slot >= 0 && std::strncmp(seen[slot].comm, comm, sizeof(seen[slot].comm) - 1) == 0) {
            continue; // Already pinned with this exact name
        }

        const cpu_set_t* target_mask = nullptr;
        int target_nice = 0;
        const int gpu_comm_tid = g_gpu_comm_tid.load(std::memory_order_relaxed);
        const int draw_rec_tid = g_draw_rec_tid.load(std::memory_order_relaxed);
        const int vk_rec_tid = g_vk_recorder_tid.load(std::memory_order_relaxed);
        const int present_tid = g_present_tid.load(std::memory_order_relaxed);

        static const bool fast_load_res = [] {
            const char* env = bb_env("BB_FAST_LOAD");
            return !env || !*env || !(env[0] == '0' && env[1] == '\0');
        }();

        if (is_main_tid) {
            target_mask = &topo.game_main;
            target_nice = -10;
        } else if (gpu_comm_tid > 0 && tid == gpu_comm_tid) {
            target_mask = &topo.gpu_comm;
            target_nice = -10;
        } else if (draw_rec_tid > 0 && tid == draw_rec_tid) {
            target_mask = &topo.draw_rec;
            target_nice = -10;
        } else if (vk_rec_tid > 0 && tid == vk_rec_tid) {
            target_mask = &topo.vk_recorder;
            target_nice = -10;
        } else if ((present_tid > 0 && tid == present_tid) ||
                   std::strcmp(comm, "bb:Present") == 0) {
            target_mask = &topo.present;
            target_nice = -10;
        } else if (starts_with(comm, "GXRenderThread")) {
            target_mask = &topo.game_workers;
            target_nice = -10;
        } else if (starts_with(comm, "GXWorker")) {
            target_mask = &topo.gx_workers;
            target_nice = -6;
        } else if (starts_with(comm, "HavokWorker") ||
                   starts_with(comm, "CSChrThread") ||
                   starts_with(comm, "CSClothThread") ||
                   starts_with(comm, "SpClothVertex") ||
                   starts_with(comm, "FD4JobWorker") ||
                   starts_with(comm, "Battle Worker") ||
                   starts_with(comm, "CS Filtering") ||
                   (fast_load_res && starts_with(comm, "Core.Res."))) {
            target_mask = &topo.game_workers;
            if (!starts_with(comm, "Core.Res.")) {
                target_nice = -6;
            }
        } else if (starts_with(comm, "bb:Resolve") || std::strcmp(comm, "bb:TexBind") == 0) {
            target_mask = &topo.gpu;
            target_nice = -6;
        } else if (std::strcmp(comm, "bb:GpuComm") == 0 ||
                   std::strcmp(comm, "bb:DrawRec") == 0 ||
                   starts_with(comm, "bb:VkRecorder") ||
                   starts_with(comm, "[vkps]") ||
                   starts_with(comm, "vk_") ||
                   starts_with(comm, "nvidia") ||
                   starts_with(comm, "radv") ||
                   starts_with(comm, "amdgpu")) {
            // Note: any child driver thread spawned from bb:GpuComm/bb:DrawRec inherits comm
            // with tid != owner_tid, and is immediately migrated onto topo.gpu!
            target_mask = &topo.gpu;
        } else if (starts_with(comm, "bb:") ||
                   starts_with(comm, "bb-probe:") ||
                   std::strcmp(comm, "MOMainThread") == 0 ||
                   starts_with(comm, "MO Update") ||
                   starts_with(comm, "Mv_") ||
                   starts_with(comm, "CSChrBackInit") ||
                   starts_with(comm, "NexusRevolution") ||
                   starts_with(comm, "FrpgSaveLoad") ||
                   starts_with(comm, "DLUserPreferenc") ||
                   starts_with(comm, "Core.Logging") ||
                   starts_with(comm, "EzWorkPool") ||
                   starts_with(comm, "Core.Res.") ||
                   starts_with(comm, "FMOD") ||
                   starts_with(comm, "FSBank") ||
                   starts_with(comm, "SLSession") ||
                   starts_with(comm, "SL ") ||
                   starts_with(comm, "CSTrophy") ||
                   starts_with(comm, "CSMenu") ||
                   starts_with(comm, "SaveData") ||
                   starts_with(comm, "FrpgNet") ||
                   starts_with(comm, "SystemService") ||
                   starts_with(comm, "ImeEvent") ||
                   starts_with(comm, "FaceGen") ||
                   starts_with(comm, "KatanaMain")) {
            target_mask = &topo.helper;
        } else {
            // Keep any third-party host threads (SDL audio, PipeWire/PulseAudio, NGX/CUDA workers)
            // off the dedicated critical pools (MOMainThread, RHI-Crit).
            target_mask = &topo.helper;
        }

        if (!target_mask || CPU_COUNT(target_mask) == 0) {
            continue;
        }

        if (sched_setaffinity(tid, sizeof(cpu_set_t), target_mask) == 0) {
            if (target_nice < 0) {
                BoostThreadPriority(tid, target_nice);
            }
            if (slot < 0 && seen_count < 256) {
                slot = seen_count++;
                seen[slot].tid = tid;
            }
            if (slot >= 0) {
                std::strncpy(seen[slot].comm, comm, sizeof(seen[slot].comm) - 1);
                seen[slot].comm[sizeof(seen[slot].comm) - 1] = '\0';
            }
        }
    }

    ::closedir(dir);
}
#else
inline void PinToGpuCommCore() { bb_win32_pin_gpu_comm_core(); }
inline void PinToDrawRecCore() { bb_win32_pin_draw_rec_core(); }
inline void PinToVkRecorderCore() { bb_win32_pin_vk_recorder_core(); }
inline void PinToPresentCore() { bb_win32_pin_present_core(); }
inline void PinToGpuDomain() { bb_win32_pin_gpu_domain(); }
inline void PinToHelperDomain() { bb_win32_pin_helper_domain(); }
inline void ApplyProcessThreadTopology() { bb_win32_apply_thread_topology(); }
#endif

} // namespace BbThreads
