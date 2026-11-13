// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Universal Hardware-Adaptive CPU Topology & Thread Affinity Scheduler.
// Automatically detects Multi-CCD / Multi-L3-Domain CPUs (AMD Ryzen 9 / Threadripper, including
// asymmetric 3D V-Cache X3D), Hybrid P-Core + E-Core CPUs (Intel 12th-15th Gen), Single-CCD
// Homogeneous CPUs (Ryzen 5/7, X3D, Intel 6c/8c), and Compact Handhelds (Steam Deck 4c/8t).

#pragma once

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

struct TopologyMasks {
    cpu_set_t game_main;    // MOMainThread (dedicated #1 physical core in CCD0 when >= 6 phys cores)
    cpu_set_t gpu_comm;     // bb:GpuComm / Stage A (dedicated #1 physical core in CCD1, sharing L3 with DrawRec & VkRecorder)
    cpu_set_t draw_rec;     // bb:DrawRec / Stage B (dedicated #2 physical core in CCD1, sharing L3 with GpuComm & VkRecorder)
    cpu_set_t vk_recorder;  // bb:VkRecorder (dedicated #3 physical core in CCD1, sharing L3 with GpuComm & DrawRec)
    cpu_set_t game_workers; // HavokWorker*, CSChrThread*, CSClothThread*, SpCloth*, GXRender*, GXWorker*
    cpu_set_t gpu;          // bb:Resolve*, bb:TexBind, bb:VkRecorder1+, bb:Present, driver helper threads
    cpu_set_t helper;       // bb:Copy*, bb:DrawScan, EzWorkPool*, Core.Res.*, FMOD*, NGX/CUDA/SDL helpers
    int best_main_cpu = -1;
    int best_main_sibling = -1;
    int best_gpu_comm_cpu = -1;
    int best_gpu_comm_sibling = -1;
    int best_draw_rec_cpu = -1;
    int best_draw_rec_sibling = -1;
    int best_vk_rec_cpu = -1;
    int best_vk_rec_sibling = -1;
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
        CPU_ZERO(&none.game_workers);
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

        // Helper to pick the #1 physical core inside a given domain mask and optionally
        // carve out both its primary thread and SMT sibling from worker masks.
        const auto pick_best_core = [&](const cpu_set_t& pool, int* out_primary, int* out_sibling) {
            int best = -1;
            long best_sc = -1;
            for (int c = 0; c < 256; ++c) {
                if (cpus[c].valid && CPU_ISSET(c, &pool) && cpus[c].primary_sibling == c) {
                    if (cpus[c].score > best_sc) {
                        best_sc = cpus[c].score;
                        best = c;
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
            *out_primary = best;
            *out_sibling = -1;
            if (best >= 0) {
                for (int s = 0; s < 256; ++s) {
                    if (s != best && CPU_ISSET(s, &cpus[best].siblings)) {
                        *out_sibling = s;
                        break;
                    }
                }
            }
        };

        if (domain_count >= 2) {
            // =================================================================================
            // CASE 1: Multi-CCD / Multi-L3-Domain CPU (e.g. Ryzen 9 5900X/5950X/7900X/7950X/
            // 7950X3D/9900X/9950X/9950X3D, Threadripper).
            // - Primary L3 domain (largest L3 V-Cache first, then highest preferred-core score)
            //   is dedicated to the Guest Game Engine:
            //   * MOMainThread gets its own #1 physical core (with its SMT sibling left empty
            //     so MOMainThread has 100% of ALU/FPU/L1/L2 with zero SMT contention).
            //   * HavokWorkerThre (x5), CSChrThread0..5 (x6), CSClothThread0..5 (x6),
            //     SpClothVertexUp (x6), GXRenderThread, and GXWorker:1..5 (x5) share the
            //     remaining cores of the EXACT SAME L3 domain (~4 ns L3 hit, 0 Infinity Fabric
            //     crossings across all ~10,000 per-frame Havok/character fork-join barriers).
            // - Secondary L3 domain (CCD1) is dedicated to the Host Vulkan RHI Pipeline
            //   (bb:GpuComm, bb:DrawRec, bb:VkRecorder*, bb:Resolve*, bb:DrawScan, bb:Copy*)
            //   plus background guest I/O & audio threads (EzWorkPool*, Core.Res.*, FMOD*).
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

            pick_best_core(game_pool, &result.best_main_cpu, &result.best_main_sibling);
            result.game_workers = game_pool;
            if (domains[game_dom].phys_cores >= 6 && result.best_main_cpu >= 0) {
                CPU_ZERO(&result.game_main);
                CPU_SET(result.best_main_cpu, &result.game_main);
                // Isolate MOMainThread's physical core (and its SMT sibling) from workers.
                for (int s = 0; s < 256; ++s) {
                    if (CPU_ISSET(s, &cpus[result.best_main_cpu].siblings)) {
                        CPU_CLR(s, &result.game_workers);
                    }
                }
            } else {
                result.game_main = game_pool;
            }

            result.gpu = gpu_pool;
            result.helper = gpu_pool;

            if (domains[gpu_dom].phys_cores >= 4) {
                // Multi-CCD Ryzen / Threadripper (e.g. 5900X/5950X/7900X/7950X/7950X3D/9900X/9950X/9950X3D):
                // Co-locate `bb:GpuComm` (Stage A), `bb:DrawRec` (Stage B), and `bb:VkRecorder`
                // inside the EXACT SAME CCD1 32 MB L3 domain (~10 ns L3 cache-to-cache handoff,
                // zero ~85 ns CCD0<->CCD1 Infinity Fabric SPSC ping-pong across ~1,800 draws/frame),
                // while carving THREE separate physical cores (SMT siblings idle) so Stage A,
                // Stage B, and VkRecorder never contend on ALU/L1/L2!
                pick_best_core(result.gpu, &result.best_gpu_comm_cpu, &result.best_gpu_comm_sibling);
                if (result.best_gpu_comm_cpu >= 0) {
                    CPU_ZERO(&result.gpu_comm);
                    CPU_SET(result.best_gpu_comm_cpu, &result.gpu_comm);
                    for (int s = 0; s < 256; ++s) {
                        if (CPU_ISSET(s, &cpus[result.best_gpu_comm_cpu].siblings)) {
                            CPU_CLR(s, &result.gpu);
                            CPU_CLR(s, &result.helper);
                        }
                    }
                } else {
                    result.gpu_comm = result.gpu;
                }

                if (domains[gpu_dom].phys_cores >= 6) {
                    pick_best_core(result.gpu, &result.best_draw_rec_cpu, &result.best_draw_rec_sibling);
                    if (result.best_draw_rec_cpu >= 0) {
                        CPU_ZERO(&result.draw_rec);
                        CPU_SET(result.best_draw_rec_cpu, &result.draw_rec);
                        for (int s = 0; s < 256; ++s) {
                            if (CPU_ISSET(s, &cpus[result.best_draw_rec_cpu].siblings)) {
                                CPU_CLR(s, &result.gpu);
                                CPU_CLR(s, &result.helper);
                            }
                        }
                    } else {
                        result.draw_rec = result.gpu;
                    }

                    pick_best_core(result.gpu, &result.best_vk_rec_cpu, &result.best_vk_rec_sibling);
                    if (result.best_vk_rec_cpu >= 0) {
                        CPU_ZERO(&result.vk_recorder);
                        CPU_SET(result.best_vk_rec_cpu, &result.vk_recorder);
                        for (int s = 0; s < 256; ++s) {
                            if (CPU_ISSET(s, &cpus[result.best_vk_rec_cpu].siblings)) {
                                CPU_CLR(s, &result.gpu);
                                CPU_CLR(s, &result.helper);
                            }
                        }
                    } else {
                        result.vk_recorder = result.gpu;
                    }
                } else {
                    result.draw_rec = result.gpu;
                    result.vk_recorder = result.gpu;
                }
            } else {
                result.gpu_comm = result.gpu;
                result.draw_rec = result.gpu;
                result.vk_recorder = result.gpu;
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
                // - MOMainThread gets the #1 P-core (isolated from its SMT sibling when >=6P).
                // - When >=8 P-cores (13700K/13900K/14700K/14900K), bb:GpuComm gets the #2 P-core
                //   and bb:VkRecorder gets the #3 P-core (isolated from SMT siblings, same L3).
                // - Havok/CSChr/GX workers + Vulkan helpers stay strictly on remaining P-cores.
                // - Async helpers, file decompression, and FMOD audio run on E-cores.
                // =============================================================================
                pick_best_core(p_cores, &result.best_main_cpu, &result.best_main_sibling);
                result.game_workers = p_cores;
                if (p_phys_cores >= 6 && result.best_main_cpu >= 0) {
                    CPU_ZERO(&result.game_main);
                    CPU_SET(result.best_main_cpu, &result.game_main);
                    for (int s = 0; s < 256; ++s) {
                        if (CPU_ISSET(s, &cpus[result.best_main_cpu].siblings)) {
                            CPU_CLR(s, &result.game_workers);
                        }
                    }
                } else {
                    result.game_main = p_cores;
                }
                if (p_phys_cores >= 6) {
                    pick_best_core(result.game_workers, &result.best_gpu_comm_cpu,
                                   &result.best_gpu_comm_sibling);
                    if (result.best_gpu_comm_cpu >= 0) {
                        CPU_ZERO(&result.gpu_comm);
                        CPU_SET(result.best_gpu_comm_cpu, &result.gpu_comm);
                        for (int s = 0; s < 256; ++s) {
                            if (CPU_ISSET(s, &cpus[result.best_gpu_comm_cpu].siblings)) {
                                CPU_CLR(s, &result.game_workers);
                            }
                        }
                    } else {
                        result.gpu_comm = result.game_workers;
                    }
                    if (p_phys_cores >= 8) {
                        pick_best_core(result.game_workers, &result.best_draw_rec_cpu,
                                       &result.best_draw_rec_sibling);
                        if (result.best_draw_rec_cpu >= 0) {
                            CPU_ZERO(&result.draw_rec);
                            CPU_SET(result.best_draw_rec_cpu, &result.draw_rec);
                            for (int s = 0; s < 256; ++s) {
                                if (CPU_ISSET(s, &cpus[result.best_draw_rec_cpu].siblings)) {
                                    CPU_CLR(s, &result.game_workers);
                                }
                            }
                        } else {
                            result.draw_rec = result.game_workers;
                        }
                    } else {
                        result.draw_rec = result.game_workers;
                    }
                    if (p_phys_cores >= 10) {
                        pick_best_core(result.game_workers, &result.best_vk_rec_cpu,
                                       &result.best_vk_rec_sibling);
                        if (result.best_vk_rec_cpu >= 0) {
                            CPU_ZERO(&result.vk_recorder);
                            CPU_SET(result.best_vk_rec_cpu, &result.vk_recorder);
                            for (int s = 0; s < 256; ++s) {
                                if (CPU_ISSET(s, &cpus[result.best_vk_rec_cpu].siblings)) {
                                    CPU_CLR(s, &result.game_workers);
                                }
                            }
                        } else {
                            result.vk_recorder = result.game_workers;
                        }
                    } else {
                        result.vk_recorder = result.game_workers;
                    }
                } else {
                    result.gpu_comm = result.game_workers;
                    result.draw_rec = result.game_workers;
                    result.vk_recorder = result.game_workers;
                }
                result.gpu = result.game_workers;
                result.helper = e_cores;
            } else if (total_phys_cores >= 6) {
                // =============================================================================
                // CASE 3: Single-L3 Homogeneous CPU with >= 6 Physical Cores
                // (e.g. Ryzen 5 5600X/7600X, Ryzen 7 5800X3D/7800X3D/9800X3D, Intel 6c/8c).
                // - MOMainThread gets #1 physical core (SMT sibling idle).
                // - bb:GpuComm gets #2 physical core (SMT sibling idle, >= 6 physical cores).
                // - When >= 8 physical cores (5800X3D/7800X3D/9800X3D), bb:DrawRec gets #3
                //   physical core (SMT sibling idle, 5 physical cores = 10 threads left for workers).
                // - When >= 10 physical cores, bb:VkRecorder also gets #4 physical core.
                // =============================================================================
                pick_best_core(allowed, &result.best_main_cpu, &result.best_main_sibling);
                result.game_workers = allowed;
                if (result.best_main_cpu >= 0) {
                    CPU_ZERO(&result.game_main);
                    CPU_SET(result.best_main_cpu, &result.game_main);
                    for (int s = 0; s < 256; ++s) {
                        if (CPU_ISSET(s, &cpus[result.best_main_cpu].siblings)) {
                            CPU_CLR(s, &result.game_workers);
                        }
                    }
                } else {
                    result.game_main = allowed;
                }
                pick_best_core(result.game_workers, &result.best_gpu_comm_cpu,
                               &result.best_gpu_comm_sibling);
                if (result.best_gpu_comm_cpu >= 0) {
                    CPU_ZERO(&result.gpu_comm);
                    CPU_SET(result.best_gpu_comm_cpu, &result.gpu_comm);
                    for (int s = 0; s < 256; ++s) {
                        if (CPU_ISSET(s, &cpus[result.best_gpu_comm_cpu].siblings)) {
                            CPU_CLR(s, &result.game_workers);
                        }
                    }
                } else {
                    result.gpu_comm = result.game_workers;
                }
                if (total_phys_cores >= 8) {
                    pick_best_core(result.game_workers, &result.best_draw_rec_cpu,
                                   &result.best_draw_rec_sibling);
                    if (result.best_draw_rec_cpu >= 0) {
                        CPU_ZERO(&result.draw_rec);
                        CPU_SET(result.best_draw_rec_cpu, &result.draw_rec);
                        for (int s = 0; s < 256; ++s) {
                            if (CPU_ISSET(s, &cpus[result.best_draw_rec_cpu].siblings)) {
                                CPU_CLR(s, &result.game_workers);
                            }
                        }
                    } else {
                        result.draw_rec = result.game_workers;
                    }
                } else {
                    result.draw_rec = result.game_workers;
                }
                if (total_phys_cores >= 10) {
                    pick_best_core(result.game_workers, &result.best_vk_rec_cpu,
                                   &result.best_vk_rec_sibling);
                    if (result.best_vk_rec_cpu >= 0) {
                        CPU_ZERO(&result.vk_recorder);
                        CPU_SET(result.best_vk_rec_cpu, &result.vk_recorder);
                        for (int s = 0; s < 256; ++s) {
                            if (CPU_ISSET(s, &cpus[result.best_vk_rec_cpu].siblings)) {
                                CPU_CLR(s, &result.game_workers);
                            }
                        }
                    } else {
                        result.vk_recorder = result.game_workers;
                    }
                } else {
                    result.vk_recorder = result.game_workers;
                }
                result.gpu = result.game_workers;
                result.helper = result.game_workers;
            } else {
                // =============================================================================
                // CASE 4: Compact CPU (< 6 physical cores, e.g. Steam Deck 4c/8t).
                // Keep all cores shared and rely on thread priorities (MakeLowPriority/IDLE).
                // =============================================================================
                pick_best_core(allowed, &result.best_main_cpu, &result.best_main_sibling);
                result.game_main = allowed;
                result.gpu_comm = allowed;
                result.draw_rec = allowed;
                result.vk_recorder = allowed;
                result.game_workers = allowed;
                result.gpu = allowed;
                result.helper = allowed;
            }
        }

        char main_str[64], gcomm_str[64], drec_str[64], vkrec_str[64], work_str[64], gpu_str[64], help_str[64];
        FormatCpuList(result.game_main, main_str, sizeof(main_str));
        FormatCpuList(result.gpu_comm, gcomm_str, sizeof(gcomm_str));
        FormatCpuList(result.draw_rec, drec_str, sizeof(drec_str));
        FormatCpuList(result.vk_recorder, vkrec_str, sizeof(vkrec_str));
        FormatCpuList(result.game_workers, work_str, sizeof(work_str));
        FormatCpuList(result.gpu, gpu_str, sizeof(gpu_str));
        FormatCpuList(result.helper, help_str, sizeof(help_str));
        std::printf("CPU Topology (%d L3 domain%s, %d phys cores): "
                    "MOMainThread=[%s] (SMT %d idle), "
                    "GpuComm/StageA=[%s] (SMT %d idle), "
                    "DrawRec/StageB=[%s] (SMT %d idle), "
                    "VkRecorder=[%s] (SMT %d idle), "
                    "Game/Havok/GX=[%s], Vulkan RHI=[%s], Helpers/IO=[%s]\n",
                    domain_count, domain_count == 1 ? "" : "s", total_phys_cores,
                    main_str, result.best_main_sibling,
                    gcomm_str, result.best_gpu_comm_sibling,
                    drec_str, result.best_draw_rec_sibling,
                    vkrec_str, result.best_vk_rec_sibling,
                    work_str, gpu_str, help_str);
        std::fflush(stdout);
        return result;
    }();
    return topo;
}

inline const TopologyMasks& GetL3Domains() {
    return GetTopology();
}

/// Pins the single Stage A DirectVk command processor thread (bb:GpuComm) to its dedicated
/// isolated physical core in CCD1 (sharing L3 with bb:VkRecorder) and records its exact Linux TID
/// so any child threads spawned by the Vulkan ICD get migrated off its core onto `topo.gpu`.
inline void PinToGpuCommCore() {
    g_gpu_comm_tid.store(int(::gettid()), std::memory_order_relaxed);
    const auto& domain = GetTopology().gpu_comm;
    if (CPU_COUNT(&domain) != 0) {
        sched_setaffinity(0, sizeof(domain), &domain);
        setpriority(PRIO_PROCESS, static_cast<id_t>(::gettid()), -2);
    }
}

/// Pins the Stage B draw recording thread (bb:DrawRec) to its own dedicated isolated physical core
/// in the exact same CCD1 L3 domain as bb:GpuComm and bb:VkRecorder (SMT sibling idle), and records
/// its TID so any child driver threads spawned during Stage B calls migrate onto `topo.gpu`.
inline void PinToDrawRecCore() {
    g_draw_rec_tid.store(int(::gettid()), std::memory_order_relaxed);
    const auto& domain = GetTopology().draw_rec;
    if (CPU_COUNT(&domain) != 0) {
        sched_setaffinity(0, sizeof(domain), &domain);
        setpriority(PRIO_PROCESS, static_cast<id_t>(::gettid()), -2);
    }
}

/// Pins the primary Vulkan command buffer recording + submission thread (bb:VkRecorder) to its own
/// dedicated physical core in the same CCD1 L3 domain as bb:GpuComm (SMT sibling idle), and records
/// its TID so lazy Vulkan ICD driver threads spawned during vkQueueSubmit migrate onto `topo.gpu`.
inline void PinToVkRecorderCore() {
    g_vk_recorder_tid.store(int(::gettid()), std::memory_order_relaxed);
    const auto& domain = GetTopology().vk_recorder;
    if (CPU_COUNT(&domain) != 0) {
        sched_setaffinity(0, sizeof(domain), &domain);
        setpriority(PRIO_PROCESS, static_cast<id_t>(::gettid()), -2);
    }
}

/// Pins secondary host Vulkan pipeline threads (bb:DrawRec, bb:VkRecorder*, bb:Present)
/// to the GPU domain (CCD1 on Multi-CCD Ryzen; P-cores on Intel Hybrid; non-MOMainThread cores on Single-CCD).
inline void PinToGpuDomain() {
    const auto& domain = GetTopology().gpu;
    if (CPU_COUNT(&domain) != 0) {
        sched_setaffinity(0, sizeof(domain), &domain);
    }
}

/// Pins auxiliary streaming/speculative helpers (bb:Resolve*, bb:Copy*, bb:DrawScan, bb:SchedOps)
/// to the helper domain (CCD1 on Multi-CCD Ryzen; E-cores on Intel Hybrid; non-MOMainThread cores on Single-CCD).
inline void PinToHelperDomain() {
    const auto& domain = GetTopology().helper;
    if (CPU_COUNT(&domain) != 0) {
        sched_setaffinity(0, sizeof(domain), &domain);
    }
}

/// Scans `/proc/self/task` and applies the hardware-adaptive affinity masks to all guest and host
/// threads (MOMainThread, HavokWorkerThre, CSChrThread*, CSClothThread*, SpClothVertexUp*,
/// GXRenderThread, GXWorker*, EzWorkPool*, FMOD*, and bb:* host threads).
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
        // 1-core isolated mask from MOMainThread, bb:GpuComm, or bb:VkRecorder onto `topo.helper`.
        if (!is_main_tid && (comm[0] == '\0' || std::strcmp(comm, "bb-probe") == 0 ||
                             std::strcmp(comm, "probe") == 0)) {
            if (CPU_COUNT(&topo.helper) > 0) {
                cpu_set_t cur_mask;
                CPU_ZERO(&cur_mask);
                if (sched_getaffinity(tid, sizeof(cur_mask), &cur_mask) == 0 &&
                    CPU_COUNT(&cur_mask) <= 1 &&
                    ((CPU_COUNT(&topo.game_main) > 0 && CPU_EQUAL(&cur_mask, &topo.game_main)) ||
                     (CPU_COUNT(&topo.gpu_comm) > 0 && CPU_EQUAL(&cur_mask, &topo.gpu_comm)) ||
                     (CPU_COUNT(&topo.draw_rec) > 0 && CPU_EQUAL(&cur_mask, &topo.draw_rec)) ||
                     (CPU_COUNT(&topo.vk_recorder) > 0 && CPU_EQUAL(&cur_mask, &topo.vk_recorder)))) {
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
        bool boost_prio = false;
        const int gpu_comm_tid = g_gpu_comm_tid.load(std::memory_order_relaxed);
        const int draw_rec_tid = g_draw_rec_tid.load(std::memory_order_relaxed);
        const int vk_rec_tid = g_vk_recorder_tid.load(std::memory_order_relaxed);

        if (is_main_tid) {
            target_mask = &topo.game_main;
            boost_prio = true;
        } else if (gpu_comm_tid > 0 && tid == gpu_comm_tid) {
            target_mask = &topo.gpu_comm;
            boost_prio = true;
        } else if (draw_rec_tid > 0 && tid == draw_rec_tid) {
            target_mask = &topo.draw_rec;
            boost_prio = true;
        } else if (vk_rec_tid > 0 && tid == vk_rec_tid) {
            target_mask = &topo.vk_recorder;
            boost_prio = true;
        } else if (starts_with(comm, "HavokWorker") ||
                   starts_with(comm, "CSChrThread") ||
                   starts_with(comm, "CSClothThread") ||
                   starts_with(comm, "SpClothVertex") ||
                   starts_with(comm, "FD4JobWorker") ||
                   starts_with(comm, "GXRenderThread") ||
                   starts_with(comm, "GXWorker") ||
                   starts_with(comm, "Battle Worker") ||
                   starts_with(comm, "CS Filtering")) {
            target_mask = &topo.game_workers;
            if (starts_with(comm, "HavokWorker") || starts_with(comm, "GXRenderThread")) {
                boost_prio = true;
            }
        } else if (std::strcmp(comm, "bb:GpuComm") == 0 ||
                   std::strcmp(comm, "bb:DrawRec") == 0 ||
                   starts_with(comm, "bb:Resolve") ||
                   std::strcmp(comm, "bb:TexBind") == 0 ||
                   starts_with(comm, "bb:VkRecorder") ||
                   std::strcmp(comm, "bb:Present") == 0 ||
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
            // off the isolated single-core lanes (MOMainThread, GpuComm, VkRecorder).
            target_mask = &topo.helper;
        }

        if (!target_mask || CPU_COUNT(target_mask) == 0) {
            continue;
        }

        if (sched_setaffinity(tid, sizeof(cpu_set_t), target_mask) == 0) {
            if (boost_prio) {
                // Give MOMainThread a mild priority boost if permitted by OS limits.
                setpriority(PRIO_PROCESS, static_cast<id_t>(tid), -2);
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
inline void PinToGpuDomain() { bb_win32_pin_gpu_domain(); }
inline void PinToHelperDomain() { bb_win32_pin_helper_domain(); }
inline void ApplyProcessThreadTopology() { bb_win32_apply_thread_topology(); }
#endif

} // namespace BbThreads
