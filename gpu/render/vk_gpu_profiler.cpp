// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <unistd.h>

#include "bbport_toggles.h"
#include "render/vk_gpu_profiler.h"
#include "shader/backend/spirv/native_shaders.h"
#include "vulkan/instance.h"
#include "render/scheduler.h"
#include "../../dantelion_gx/include/gx_direct_vulkan.h"

namespace Vulkan {

namespace {
std::atomic<bool> g_manual_req{false};
std::atomic<bool> g_manual_req_per_draw{true};
std::atomic<bool> g_manual_active{false};
std::atomic<u32> g_manual_last_labels{0};
std::atomic<u32> g_manual_last_draws{0};
} // namespace

void GpuProfiler::RequestManualCapture(bool per_draw) noexcept {
    g_manual_req_per_draw.store(per_draw, std::memory_order_relaxed);
    g_manual_req.store(true, std::memory_order_release);
}

bool GpuProfiler::IsManualCaptureActive() noexcept {
    return g_manual_active.load(std::memory_order_relaxed) ||
           g_manual_req.load(std::memory_order_relaxed);
}

u32 GpuProfiler::GetLastManualLabelCount() noexcept {
    return g_manual_last_labels.load(std::memory_order_relaxed);
}

u32 GpuProfiler::GetLastManualDrawCount() noexcept {
    return g_manual_last_draws.load(std::memory_order_relaxed);
}

void GpuProfiler::Init(const Instance& instance, Scheduler& scheduler) {
    if (stored_ptr) {
        return;
    }
    stored_ptr = new GpuProfiler(instance, scheduler);
    const char* env = std::getenv("BB_GPU_PROFILE");
    if (env && env[0] == '2') {
        stored_ptr->permanent = true;
        const char* draws_env = std::getenv("BB_GPU_PROFILE_DRAWS");
        stored_ptr->per_draw_mode = (draws_env && draws_env[0] == '1');
        instance_ptr = stored_ptr;
    }
}

void GpuProfiler::StartCapture(bool per_draw) {
    for (u32 i = 0; i < NumSlices; ++i) {
        if (used[i]) {
            device.resetQueryPool(*pool, i * SliceQueries, used[i]);
        }
        used[i] = 0;
        keys[i].clear();
        pending[i] = false;
    }
    slice = 0;
    frames = 0;
    draw_frames = 0;
    current = 0;
    per_draw_mode = per_draw;
    totals.clear();
    draw_stats.clear();
    gaps.clear();
    window = std::chrono::steady_clock::now();
    instance_ptr = this;
}

void GpuProfiler::StopCapture() {
    if (used[slice] > 0 && used[slice] < SliceQueries) {
        scheduler.EndRendering();
        const u32 query = slice * SliceQueries + used[slice]++;
        scheduler.Record([pool = *pool, query](vk::CommandBuffer cmdbuf) {
            cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool, query);
        });
    }
    per_draw_mode = false;
    manual_dump_mode = false;
    g_manual_active.store(false, std::memory_order_release);
    instance_ptr = nullptr;
}

void GpuProfiler::OnFrame() {
    if (!stored_ptr) {
        return;
    }
    static u64 last_draws = 0;
    static u32 scene_frames = 0;
    static bool captured_scene = false;
    const u64 cur_draws = BbStats::draws.load(std::memory_order_relaxed);
    const u64 frame_draws = cur_draws >= last_draws ? cur_draws - last_draws : 0;
    last_draws = cur_draws;
    if (frame_draws >= 800) {
        ++scene_frames;
    } else {
        scene_frames = 0;
    }

    const u64 f = ++stored_ptr->seen_frames;
    if (instance_ptr) {
        instance_ptr->BeginFrame();
        if (!stored_ptr->permanent && stored_ptr->frames >= 120) {
            stored_ptr->Print(true);
            stored_ptr->StopCapture();
            captured_scene = true;
        }
        return;
    }
    if (!stored_ptr->permanent) {
        if (g_manual_req.exchange(false, std::memory_order_acq_rel)) {
            stored_ptr->manual_dump_mode = true;
            g_manual_active.store(true, std::memory_order_release);
            stored_ptr->StartCapture(g_manual_req_per_draw.load(std::memory_order_relaxed));
            return;
        }
        static const bool auto_profile_env = [] {
            const char* e = std::getenv("BB_GPU_PROFILE");
            return e && e[0] != '\0' && e[0] != '0';
        }();
        const bool trigger_file = (f & 31) == 0 && ::access("/tmp/bb_gpu_profile_trigger", F_OK) == 0;
        const bool auto_scene =
            auto_profile_env && ((scene_frames == 120 && !captured_scene) ||
                                 (scene_frames >= 120 && (scene_frames % 1200) == 120));
        if (trigger_file || auto_scene) {
            if (trigger_file) {
                ::unlink("/tmp/bb_gpu_profile_trigger");
            }
            const char* draws_env = std::getenv("BB_GPU_PROFILE_DRAWS");
            stored_ptr->manual_dump_mode = false;
            stored_ptr->StartCapture(draws_env && draws_env[0] == '1');
        }
    }
}

GpuProfiler::GpuProfiler(const Instance& instance, Scheduler& scheduler_)
    : device{instance.GetDevice()}, scheduler{scheduler_} {
    const vk::QueryPoolCreateInfo info = {
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = NumSlices * SliceQueries,
    };
    pool = Check<"create timestamp query pool">(device.createQueryPoolUnique(info));
    device.resetQueryPool(*pool, 0, NumSlices * SliceQueries);
    period_ns = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
    for (auto& k : keys) {
        k.reserve(SliceQueries);
    }
    hud_enabled = gx_direct_vk::EnabledV7Sub(gx_direct_vk::kV7Sub_HudFullBreakdown);
    if (hud_enabled) {
        const vk::QueryPoolCreateInfo hud_info = {
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = HudSlices * HudSliceQueries,
        };
        hud_pool = Check<"create HUD timestamp query pool">(device.createQueryPoolUnique(hud_info));
        device.resetQueryPool(*hud_pool, 0, HudSlices * HudSliceQueries);
    }
    std::printf("GPU profile: on (timestamp period %.2f ns)\n", period_ns);
}

bool GpuProfiler::TryCollectHudSlice(u32 s) {
    if (!hud_pending[s]) {
        return true;
    }
    if (!scheduler.IsFree(hud_last_tick[s])) {
        return false;
    }
    const u32 count = hud_used[s];
    if (count >= 2) {
        std::array<u64, HudSliceQueries> stamps{};
        const auto res = device.getQueryPoolResults(
            *hud_pool, s * HudSliceQueries, count, count * sizeof(u64), stamps.data(), sizeof(u64),
            vk::QueryResultFlagBits::e64);
        if (res == vk::Result::eSuccess) {
            double busy_ns = 0.0;
            double gap_ns = 0.0;
            double inter_gap_ns = 0.0;
            std::array<double, BbStats::GpuCat_Count> cat_ns{};
            std::array<double, BbStats::GpuSub_Count> subcat_ns{};
            u64 cursor = stamps[0];
            if (hud_has_prev_end) {
                if (stamps[0] > hud_prev_end_stamp) {
                    const double cross_ns = double(stamps[0] - hud_prev_end_stamp) * period_ns;
                    if (cross_ns < 2e8) {
                        inter_gap_ns += cross_ns;
                    }
                } else {
                    cursor = hud_prev_end_stamp;
                }
            }
            for (u32 i = 0; i + 1 < count; ++i) {
                const u64 s1 = stamps[i + 1];
                const u8 subcat = hud_cats[s][i];
                if (s1 > cursor) {
                    const double dt_ns = double(s1 - cursor) * period_ns;
                    cursor = s1;
                    if (subcat == BbStats::GpuSub_Gap) {
                        gap_ns += dt_ns;
                    } else if (subcat < BbStats::GpuSub_Count) {
                        busy_ns += dt_ns;
                        subcat_ns[subcat] += dt_ns;
                        const u8 cat = BbStats::ParentGpuCat(subcat);
                        if (cat < BbStats::GpuCat_Count) {
                            cat_ns[cat] += dt_ns;
                        }
                    }
                }
            }
            hud_prev_end_stamp = cursor;
            hud_has_prev_end = true;
            BbStats::hw_gpu_busy_ns.fetch_add(static_cast<u64>(busy_ns), std::memory_order_relaxed);
            BbStats::hw_gpu_gap_ns.fetch_add(static_cast<u64>(gap_ns), std::memory_order_relaxed);
            BbStats::hw_gpu_inter_gap_ns.fetch_add(static_cast<u64>(inter_gap_ns),
                                                   std::memory_order_relaxed);
            for (u32 c = 0; c < BbStats::GpuCat_Count; ++c) {
                if (cat_ns[c] > 0.0) {
                    BbStats::hw_gpu_cat_ns[c].fetch_add(static_cast<u64>(cat_ns[c]),
                                                        std::memory_order_relaxed);
                }
            }
            for (u32 sc = 0; sc < BbStats::GpuSub_Count; ++sc) {
                if (subcat_ns[sc] > 0.0) {
                    BbStats::hw_gpu_subcat_ns[sc].fetch_add(static_cast<u64>(subcat_ns[sc]),
                                                            std::memory_order_relaxed);
                }
            }
            BbStats::hw_gpu_frames.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (count > 0) {
        device.resetQueryPool(*hud_pool, s * HudSliceQueries, count);
    }
    hud_used[s] = 0;
    hud_pending[s] = false;
    return true;
}

void GpuProfiler::HudBeginSubmission(const Scheduler* sched) {
    auto* self = stored_ptr;
    if (!self || !self->hud_enabled || !self->Records(sched)) {
        return;
    }
    if (self->hud_slice_blocked) {
        if (!self->TryCollectHudSlice(self->hud_slice)) {
            return;
        }
        self->hud_slice_blocked = false;
    }
    if (!self->hud_subm_open && self->hud_used[self->hud_slice] + 2 <= HudSliceQueries) {
        const u32 idx = self->hud_used[self->hud_slice]++;
        self->hud_cats[self->hud_slice][idx] = self->hud_cur_cat;
        self->hud_last_marked_cat = self->hud_cur_cat;
        const u32 query = self->hud_slice * HudSliceQueries + idx;
        self->hud_subm_open = true;
        self->scheduler.Record([pool_h = *self->hud_pool, query](vk::CommandBuffer cmdbuf) {
            cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, pool_h, query);
        });
    }
}

void GpuProfiler::HudSetNextPassCat(const Scheduler* sched, u8 subcat) {
    auto* self = stored_ptr;
    if (!self || !self->hud_enabled || !self->Records(sched)) {
        return;
    }
    self->hud_cur_cat = subcat;
}

void GpuProfiler::HudOnBeginRendering(const Scheduler* sched) {
    auto* self = stored_ptr;
    if (!self || !self->hud_enabled || !self->Records(sched)) {
        return;
    }
    if (!self->hud_subm_open || self->hud_slice_blocked) {
        return;
    }
    const u8 cat = self->hud_cur_cat;
    if (cat == self->hud_last_marked_cat) {
        return;
    }
    if (self->hud_used[self->hud_slice] + 2 <= HudSliceQueries) {
        const u32 idx = self->hud_used[self->hud_slice]++;
        self->hud_cats[self->hud_slice][idx] = cat;
        self->hud_last_marked_cat = cat;
        const u32 query = self->hud_slice * HudSliceQueries + idx;
        self->scheduler.Record([pool_h = *self->hud_pool, query](vk::CommandBuffer cmdbuf) {
            cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool_h, query);
        });
    }
}

void GpuProfiler::HudOnSubPassChange(const Scheduler* sched, u8 subcat) {
    auto* self = stored_ptr;
    if (!self || !self->hud_enabled || !self->Records(sched)) {
        return;
    }
    self->hud_cur_cat = subcat;
    if (!self->hud_subm_open || self->hud_slice_blocked) {
        return;
    }
    if (subcat == self->hud_last_marked_cat) {
        return;
    }
    // Reserve 64 slots for RenderPass/submission boundaries; record inline without breaking the pass.
    if (self->hud_used[self->hud_slice] + 64 <= HudSliceQueries) {
        const u32 idx = self->hud_used[self->hud_slice]++;
        self->hud_cats[self->hud_slice][idx] = subcat;
        self->hud_last_marked_cat = subcat;
        const u32 query = self->hud_slice * HudSliceQueries + idx;
        self->scheduler.Record([pool_h = *self->hud_pool, query](vk::CommandBuffer cmdbuf) {
            cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllGraphics, pool_h, query);
        });
    }
}

void GpuProfiler::HudMarkPass(const Scheduler* sched, u8 subcat) {
    auto* self = stored_ptr;
    if (!self || !self->hud_enabled || !self->Records(sched)) {
        return;
    }
    self->hud_cur_cat = subcat;
    if (!self->hud_subm_open || self->hud_slice_blocked) {
        return;
    }
    if (subcat == self->hud_last_marked_cat) {
        return;
    }
    if (self->hud_used[self->hud_slice] + 2 <= HudSliceQueries) {
        self->scheduler.EndRendering();
        const u32 idx = self->hud_used[self->hud_slice]++;
        self->hud_cats[self->hud_slice][idx] = subcat;
        self->hud_last_marked_cat = subcat;
        const u32 query = self->hud_slice * HudSliceQueries + idx;
        self->scheduler.Record([pool_h = *self->hud_pool, query](vk::CommandBuffer cmdbuf) {
            cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool_h, query);
        });
    }
}

void GpuProfiler::HudEndSubmission(const Scheduler* sched, u64 tick, bool is_frame_end) {
    auto* self = stored_ptr;
    if (!self || !self->hud_enabled || !self->Records(sched)) {
        return;
    }
    if (self->hud_subm_open) {
        const u32 idx = self->hud_used[self->hud_slice]++;
        self->hud_cats[self->hud_slice][idx] = BbStats::GpuSub_Gap;
        self->hud_last_marked_cat = BbStats::GpuSub_Gap;
        const u32 query = self->hud_slice * HudSliceQueries + idx;
        self->hud_subm_open = false;
        self->hud_last_tick[self->hud_slice] = tick;
        self->scheduler.Record([pool_h = *self->hud_pool, query](vk::CommandBuffer cmdbuf) {
            cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool_h, query);
        });
    }
    if (!is_frame_end) {
        return;
    }
    self->hud_cur_cat = BbStats::GpuSub_Comp_CopiesAndHle;
    if (self->hud_used[self->hud_slice] >= 2) {
        self->hud_pending[self->hud_slice] = true;
        self->hud_slice = (self->hud_slice + 1) % HudSlices;
        if (self->hud_pending[self->hud_slice] && !self->TryCollectHudSlice(self->hud_slice)) {
            self->hud_slice_blocked = true;
        }
    } else if (self->hud_slice_blocked && self->hud_pending[self->hud_slice]) {
        if (self->TryCollectHudSlice(self->hud_slice)) {
            self->hud_slice_blocked = false;
        }
    }
}

void GpuProfiler::NoteDraw(u64 vs_hash, u64 ps_hash, u32 width, u32 height, u32 num_mrts,
                           u64 vtx_instances) {
    u64 key = vs_hash ^ (ps_hash * 0x9E3779B97F4A7C15ull);
    key ^= (u64(width) << 32) ^ (u64(height) << 16) ^ u64(num_mrts);
    auto& st = draw_stats[key];
    if (st.draws == 0) {
        st.vs_hash = vs_hash;
        st.ps_hash = ps_hash;
        st.width = width;
        st.height = height;
        st.num_mrts = num_mrts;
    }
    ++st.draws;
    st.vtx_instances += vtx_instances;
}

void GpuProfiler::WriteTimestamp(u64 key) {
    if (used[slice] + 1 >= SliceQueries) {
        return; // the frame's slice is full: the rest of the frame goes to the last label
    }
    // Outside render passes: radv_CmdWriteTimestamp2 crashed inside some. Marks sit where a
    // pass, dispatch or submission ends anyway.
    scheduler.EndRendering();
    const u32 query = slice * SliceQueries + used[slice]++;
    keys[slice].push_back(key);
    current = key;
    scheduler.Record([pool = *pool, query](vk::CommandBuffer cmdbuf) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool, query);
    });
}

void GpuProfiler::WriteTimestampInline(u64 key) {
    if (used[slice] + 64 >= SliceQueries) {
        return; // keep headroom for pass/frame end boundaries
    }
    const u32 query = slice * SliceQueries + used[slice]++;
    keys[slice].push_back(key);
    current = key;
    scheduler.Record([pool = *pool, query](vk::CommandBuffer cmdbuf) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllGraphics, pool, query);
    });
}

void GpuProfiler::BeginFrame() {
    // Close the frame: one more timestamp without a label.
    if (used[slice] > 0 && used[slice] < SliceQueries) {
        ++draw_frames;
        scheduler.EndRendering();
        const u32 query = slice * SliceQueries + used[slice]++;
        scheduler.Record([pool = *pool, query](vk::CommandBuffer cmdbuf) {
            cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool, query);
        });
        pending[slice] = true;
    }
    slice = (slice + 1) % NumSlices;
    // The oldest slice: its frame was submitted four frames ago.
    if (pending[slice]) {
        Collect(slice);
    }
    if (used[slice]) {
        device.resetQueryPool(*pool, slice * SliceQueries, used[slice]);
    }
    used[slice] = 0;
    keys[slice].clear();
    pending[slice] = false;
    Print();
}

void GpuProfiler::Collect(u32 which) {
    const u32 count = used[which];
    std::vector<u64> stamps(count);
    // Four frames on this is complete unless the GPU lags that far: then wait for it.
    const auto result = device.getQueryPoolResults(
        *pool, which * SliceQueries, count, count * sizeof(u64), stamps.data(), sizeof(u64),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (result != vk::Result::eSuccess) {
        return;
    }
    for (u32 i = 0; i + 1 < count; ++i) {
        if (stamps[i] == 0u || stamps[i + 1] <= stamps[i]) {
            continue;
        }
        const double delta_ns = double(stamps[i + 1] - stamps[i]) * period_ns;
        if (!(delta_ns > 0.0 && delta_ns < 2e8)) {
            continue;
        }
        const double ms = delta_ns / 1e6;
        auto& total = totals[keys[which][i]];
        total.ms += ms;
        ++total.segments;
        if (keys[which][i] >= SubmissionGapKey && keys[which][i] < SubmissionGapKey + 0x20) {
            gaps.push_back(ms);
        }
    }
    ++frames;
}

void GpuProfiler::Print(bool to_file) {
    const auto now = std::chrono::steady_clock::now();
    if (!to_file && (now - window < std::chrono::seconds(5) || frames == 0)) {
        return;
    }
    if (frames == 0) {
        return;
    }
    window = now;
    std::vector<std::pair<u64, Total>> sorted(totals.begin(), totals.end());
    std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    auto is_idle_key = [](u64 k) noexcept {
        return k >= SubmissionGapKey && k < SubmissionGapKey + 0x20;
    };
    double sum = 0.0;
    double idle_sum = 0.0;
    for (const auto& [key, total] : sorted) {
        sum += total.ms;
        if (is_idle_key(key)) {
            idle_sum += total.ms;
        }
    }
    const double active_sum = std::max(0.0, sum - idle_sum);
    static const bool verbose = [] {
        const char* e = std::getenv("BB_VERBOSE_STATS");
        return e && e[0] == '1';
    }();
#ifdef _WIN32
    constexpr const char* kDefaultProfilePath = "logs/bb_gpu_profile.txt";
    constexpr const char* kSessionProfilePath = "logs/bb_gpu_profile_session.txt";
#else
    constexpr const char* kDefaultProfilePath = "/tmp/bb_gpu_profile.txt";
    constexpr const char* kSessionProfilePath = "/tmp/bb_gpu_profile_session.txt";
#endif
    const char* profile_path = manual_dump_mode ? "logs/gpu_profile_dump.txt" : kDefaultProfilePath;
    if (manual_dump_mode) {
        std::error_code ec;
        std::filesystem::create_directories("logs", ec);
    }
    const double total_ms_f = sum / double(frames);
    const double idle_ms_f = idle_sum / double(frames);
    const double active_ms_f = active_sum / double(frames);
    const double gpu_w = double(BbStats::gpu_power_mw.load(std::memory_order_relaxed)) * 0.001;
    const double active_w = (gpu_w > 32.0) ? (gpu_w - 30.0) : 0.0;
    const size_t max_labels = manual_dump_mode ? sorted.size() : std::min<size_t>(sorted.size(), 40);
    FILE* f = std::fopen(profile_path, "w");
    std::printf("GPU profile: %.2f ms active (+ %.2f ms idle) over %llu frames, %zu labels (details in %s)\n",
                active_ms_f, idle_ms_f, static_cast<unsigned long long>(frames), sorted.size(), profile_path);
    if (f) {
        std::fprintf(f, "=== Bloodborne PC Vulkan — GPU Pass & Draw Bottleneck Profile ===\n");
        std::fprintf(f, "Frames: %llu | Per-Draw Timestamps: %s | Unique GPU segments: %zu | Unique shader pairs: %zu\n",
                     static_cast<unsigned long long>(frames), per_draw_mode ? "ON" : "OFF",
                     sorted.size(), draw_stats.size());
        std::fprintf(f, "Active GPU Work: %.3f ms/frame | Idle/Cap Wait: %.3f ms/frame | Total Window: %.3f ms/frame\n",
                     active_ms_f, idle_ms_f, total_ms_f);
        std::fprintf(f, "Live Board Power: %.1f W (Est. Active Power: %.1f W above 30W idle)\n\n",
                     gpu_w, active_w);

        // [0] Smart Bottleneck Filter: automatically separates actionable [PS4-SPIRV] targets
        // and heavy native passes (>= 3% active GPU) from known-cheap micro-draws (< 0.015 ms).
        if (manual_dump_mode) {
            std::fprintf(f, "--- [0A] ACTIONABLE [PS4-SPIRV] HOTSPOTS (>= 0.010 ms/frame, NOT YET NATIVE GLSL) ---\n");
            std::fprintf(f, "  ms/frame | %%Act  |  ~W   | seg/f | description\n");
            u32 spirv_hits = 0;
            for (const auto& [key, total] : sorted) {
                if (is_idle_key(key)) continue;
                const double ms_f = total.ms / double(frames);
                if (ms_f < 0.010) continue;
                const auto& desc = described[key];
                if (desc.find("[PS4-SPIRV]") == std::string::npos) continue;
                const double pct = (active_ms_f > 1e-6) ? (ms_f * 100.0 / active_ms_f) : 0.0;
                const double est_w = (active_ms_f > 1e-6) ? (active_w * (ms_f / active_ms_f)) : 0.0;
                std::fprintf(f, "  %8.4f | %5.1f%% | %5.1f | %5.1f | %s\n",
                             ms_f, pct, est_w, double(total.segments) / double(frames), desc.c_str());
                ++spirv_hits;
            }
            if (spirv_hits == 0) {
                std::fprintf(f, "  (none — all shaders >= 0.010 ms in this scene are already Native GLSL / Vulkan!)\n");
            }
            std::fprintf(f, "\n--- [0B] HEAVY NATIVE / VULKAN PASSES (>= 3.0%% OF ACTIVE GPU) ---\n");
            std::fprintf(f, "  ms/frame | %%Act  |  ~W   | seg/f | description\n");
            u32 heavy_native = 0;
            for (const auto& [key, total] : sorted) {
                if (is_idle_key(key)) continue;
                const double ms_f = total.ms / double(frames);
                const double pct = (active_ms_f > 1e-6) ? (ms_f * 100.0 / active_ms_f) : 0.0;
                if (pct < 3.0) continue;
                const auto& desc = described[key];
                if (desc.find("[PS4-SPIRV]") != std::string::npos) continue;
                const double est_w = (active_ms_f > 1e-6) ? (active_w * (ms_f / active_ms_f)) : 0.0;
                std::fprintf(f, "  %8.4f | %5.1f%% | %5.1f | %5.1f | %s\n",
                             ms_f, pct, est_w, double(total.segments) / double(frames), desc.c_str());
                ++heavy_native;
            }
            if (heavy_native == 0) {
                std::fprintf(f, "  (none)\n");
            }
            std::fprintf(f, "\n");
        }

        std::fprintf(f, "--- [1] ALL GPU PASSES & SHADER DRAWS SORTED BY GPU TIME (HEAVIEST FIRST) ---\n");
        std::fprintf(f, "  ms/frame | %%Act  |  ~W   | seg/f | description\n");
    }
    for (size_t i = 0; i < max_labels; ++i) {
        const auto& [key, total] = sorted[i];
        const double ms_f = total.ms / double(frames);
        const bool is_idle = is_idle_key(key);
        const double pct = (!is_idle && active_ms_f > 1e-6) ? (ms_f * 100.0 / active_ms_f) : 0.0;
        const double est_w = (!is_idle && active_ms_f > 1e-6) ? (active_w * (ms_f / active_ms_f)) : 0.0;
        if (verbose) {
            std::printf("  %6.3f ms/frame %5.1f/frame  %s\n", ms_f,
                        double(total.segments) / double(frames), described[key].c_str());
        }
        if (f) {
            if (is_idle) {
                std::fprintf(f, "  %8.4f |  IDLE |   0.0 | %5.1f | %s\n",
                             ms_f, double(total.segments) / double(frames),
                             described[key].c_str());
            } else {
                std::fprintf(f, "  %8.4f | %5.1f%% | %5.1f | %5.1f | %s\n",
                             ms_f, pct, est_w, double(total.segments) / double(frames),
                             described[key].c_str());
            }
        }
    }
    if (manual_dump_mode) {
        g_manual_last_labels.store(static_cast<u32>(sorted.size()), std::memory_order_relaxed);
        g_manual_last_draws.store(static_cast<u32>(draw_stats.size()), std::memory_order_relaxed);
    }
    if (!gaps.empty()) {
        // The gaps one by one: the presenter's own submissions (outside this scheduler) fall in
        // them too, so a steady gap per frame is its work, scattered long ones are waits.
        std::ranges::sort(gaps);
        const auto at = [&](double q) { return gaps[size_t(q * double(gaps.size() - 1))]; };
        double over = 0;
        for (const double gap : gaps) {
            over += gap >= 0.25 ? gap : 0;
        }
        if (verbose) {
            std::printf("  submission gaps: %zu, median %.3f ms, p90 %.3f, max %.3f; %.3f ms/frame "
                        "in gaps of 0.25 ms or more\n",
                        gaps.size(), at(0.5), at(0.9), gaps.back(), over / frames);
        }
        if (f) {
            std::fprintf(f,
                         "  submission gaps: %zu, median %.3f ms, p90 %.3f, max %.3f; %.3f ms/frame "
                         "in gaps of 0.25 ms or more\n",
                         gaps.size(), at(0.5), at(0.9), gaps.back(), over / frames);
        }
        gaps.clear();
    }
    if (!draw_stats.empty() && draw_frames > 0) {
        std::vector<DrawShaderStat> sorted_draws;
        sorted_draws.reserve(draw_stats.size());
        for (const auto& [_, st] : draw_stats) {
            sorted_draws.push_back(st);
        }
        std::ranges::sort(sorted_draws, [](const DrawShaderStat& a, const DrawShaderStat& b) {
            return (a.draws != b.draws) ? (a.draws > b.draws)
                                        : (a.vtx_instances > b.vtx_instances);
        });
        if (verbose) {
            std::printf("  top shader pairs by draws/frame (%zu unique across %llu frames):\n",
                        sorted_draws.size(), static_cast<unsigned long long>(draw_frames));
        }
        if (f) {
            std::fprintf(f, "\n--- [2] ALL UNIQUE SHADER PAIRS BY DRAWS/FRAME (%zu unique across %llu frames) ---\n",
                         sorted_draws.size(), static_cast<unsigned long long>(draw_frames));
        }
        const double inv_df = 1.0 / double(draw_frames);
        const size_t max_draws = manual_dump_mode ? sorted_draws.size() : std::min<size_t>(sorted_draws.size(), 25);
        for (size_t i = 0; i < max_draws; ++i) {
            const auto& st = sorted_draws[i];
            const char* nat_ps = Shader::Backend::SPIRV::NativeShaderName(st.ps_hash);
            const char* nat_vs = Shader::Backend::SPIRV::NativeShaderName(st.vs_hash);
            const char* tag = nat_ps ? nat_ps : (nat_vs ? nat_vs : "PS4-SPIRV");
            if (verbose) {
                std::printf("    %6.1f draws/f %9.1f idx/f  %4ux%-4u [%u MRT] vs %016llx ps %016llx [%s]\n",
                            double(st.draws) * inv_df, double(st.vtx_instances) * inv_df, st.width,
                            st.height, st.num_mrts, static_cast<unsigned long long>(st.vs_hash),
                            static_cast<unsigned long long>(st.ps_hash), tag);
            }
            if (f) {
                std::fprintf(
                    f, "    %6.1f draws/f %9.1f idx/f  %4ux%-4u [%u MRT] vs %016llx ps %016llx [%s]\n",
                    double(st.draws) * inv_df, double(st.vtx_instances) * inv_df, st.width,
                    st.height, st.num_mrts, static_cast<unsigned long long>(st.vs_hash),
                    static_cast<unsigned long long>(st.ps_hash), tag);
            }
        }
    }
    if (f) {
        std::fclose(f);
    }

    // Accumulate into session-wide profile (/tmp/bb_gpu_profile_session.txt) across all visited
    // locations (Clinic, Central Yharnam, Hunter's Dream, etc.).
    static std::unordered_map<u64, Total> s_session_totals;
    static std::unordered_map<u64, std::string> s_session_described;
    static std::unordered_map<u64, DrawShaderStat> s_session_draws;
    static u64 s_session_frames = 0;
    static u64 s_session_draw_frames = 0;
    s_session_frames += frames;
    s_session_draw_frames += draw_frames;
    for (const auto& [k, tot] : totals) {
        auto& dst = s_session_totals[k];
        dst.ms += tot.ms;
        dst.segments += tot.segments;
        if (s_session_described.find(k) == s_session_described.end()) {
            s_session_described[k] = described[k];
        }
    }
    for (const auto& [k, st] : draw_stats) {
        auto& dst = s_session_draws[k];
        if (dst.draws == 0) {
            dst = st;
        } else {
            dst.draws += st.draws;
            dst.vtx_instances += st.vtx_instances;
        }
    }
    if (FILE* sf = std::fopen(kSessionProfilePath, "w")) {
        std::vector<std::pair<u64, Total>> sess_sorted(s_session_totals.begin(), s_session_totals.end());
        std::ranges::sort(sess_sorted, [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
        double sess_sum = 0.0;
        for (const auto& [_, tot] : sess_sorted) {
            sess_sum += tot.ms;
        }
        std::fprintf(sf, "Session GPU profile: %.2f ms/frame over %llu frames, %zu unique labels\n",
                     sess_sum / double(s_session_frames),
                     static_cast<unsigned long long>(s_session_frames), sess_sorted.size());
        for (size_t i = 0; i < std::min<size_t>(sess_sorted.size(), 80); ++i) {
            const auto& [key, total] = sess_sorted[i];
            std::fprintf(sf, "  %6.3f ms/frame %5.1f/frame  %s\n",
                         total.ms / double(s_session_frames),
                         double(total.segments) / double(s_session_frames),
                         s_session_described[key].c_str());
        }
        if (!s_session_draws.empty() && s_session_draw_frames > 0) {
            std::vector<DrawShaderStat> sess_draws;
            sess_draws.reserve(s_session_draws.size());
            for (const auto& [_, st] : s_session_draws) {
                sess_draws.push_back(st);
            }
            std::ranges::sort(sess_draws, [](const DrawShaderStat& a, const DrawShaderStat& b) {
                return (a.draws != b.draws) ? (a.draws > b.draws)
                                            : (a.vtx_instances > b.vtx_instances);
            });
            std::fprintf(sf, "  top session shader pairs by draws/frame (%zu unique across %llu frames):\n",
                         sess_draws.size(), static_cast<unsigned long long>(s_session_draw_frames));
            const double inv_sdf = 1.0 / double(s_session_draw_frames);
            for (size_t i = 0; i < std::min<size_t>(sess_draws.size(), 60); ++i) {
                const auto& st = sess_draws[i];
                std::fprintf(sf, "    %6.1f draws/f %9.1f idx/f  %4ux%-4u [%u MRT] vs %016llx ps %016llx\n",
                             double(st.draws) * inv_sdf, double(st.vtx_instances) * inv_sdf,
                             st.width, st.height, st.num_mrts,
                             static_cast<unsigned long long>(st.vs_hash),
                             static_cast<unsigned long long>(st.ps_hash));
            }
        }
        std::fclose(sf);
    }

    totals.clear();
    draw_stats.clear();
    frames = 0;
    draw_frames = 0;
}

} // namespace Vulkan
