// SPDX-License-Identifier: GPL-2.0-or-later
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sys/uio.h>
#include <unistd.h>

#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>

#include "bbport_write_log.h"
#include "bbport_gnm_hooks.h"
#include "core/emulator_settings.h"
#include "buffers/direct_buffers.h"
#include "cp/cp_diagnostics.h"
#include "cp/cp_packets.h"
#include "cp/cp_pm4.h"
#include "render/direct_states.h"
#include "render/direct_targets.h"

template <>
struct magic_enum::customize::enum_range<BbCp::Op> {
    static constexpr int min = 0;
    static constexpr int max = 255;
};

extern "C" int runtime_memory_vma_info(uintptr_t address, int* prot, int* type, uintptr_t* end);
extern "C" void runtime_memory_read_backing(uintptr_t address, void* data, u64 size);

namespace BbCp::Diagnostics {

std::string OpName(u32 op) {
    const auto name = magic_enum::enum_name(Op(op));
    return name.empty() ? fmt::format("{:#x}", op) : std::string(name);
}

namespace {

struct LabelRecord {
    std::atomic<u64> when_ns{0}; ///< steady clock; 0: not written yet
    u64 address;
    u64 value;
    u32 bytes;
    LabelStep step;
};
constexpr size_t LabelRingSize = 256;
std::array<LabelRecord, LabelRingSize> label_ring;
std::atomic<u64> label_next{0};

u64 NowNs() {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count());
}

} // namespace

void NoteLabel(LabelStep step, u64 address, u64 value, u32 bytes) {
    LabelRecord& record = label_ring[label_next.fetch_add(1, std::memory_order_relaxed) %
                                     LabelRingSize];
    record.when_ns.store(0, std::memory_order_relaxed);
    record.address = address;
    record.value = value;
    record.bytes = bytes;
    record.step = step;
    record.when_ns.store(NowNs(), std::memory_order_release);
}

bool ReadGuest(const void* address, void* out, u32 bytes) {
    int prot, type;
    uintptr_t end;
    const auto at = reinterpret_cast<uintptr_t>(address);
    if (bytes == 0 || !runtime_memory_vma_info(at, &prot, &type, &end) || end < at + bytes) {
        return false;
    }
    runtime_memory_read_backing(at, out, bytes);
    return true;
}

std::string RecentLabels(size_t count) {
    static constexpr const char* steps[] = {"compute decoded", "compute GPU write",
                                            "compute written", "graphics GPU write",
                                            "graphics written"};
    const u64 next = label_next.load(std::memory_order_acquire);
    const u64 now = NowNs();
    count = std::min<u64>({count, next, LabelRingSize});
    std::string out;
    for (u64 i = next - count; i < next; ++i) {
        const LabelRecord& record = label_ring[i % LabelRingSize];
        const u64 when = record.when_ns.load(std::memory_order_acquire);
        if (when == 0) {
            continue;
        }
        // What the guest reads there now (unreadable: a trapped or unmapped page).
        u64 current = 0;
        const u32 bytes = std::min<u32>(record.bytes, 8);
        const bool readable =
            ReadGuest(reinterpret_cast<const void*>(record.address), &current, bytes);
        out += fmt::format("  {:>8.1f} ms ago  {:<18} {:#x} = {:#x}{}\n",
                           double(now - when) / 1e6, steps[u32(record.step) % 5], record.address,
                           record.value,
                           !readable ? " (memory unreadable now)"
                           : current != (bytes == 8 ? record.value : record.value & 0xFFFFFFFFull)
                               ? fmt::format(" (memory holds {:#x} now)", current)
                               : std::string{});
    }
    return out;
}

namespace {

struct DcbStats {
    u64 buffers[2]{}, with_draws[2]{}, draws[2]{}, dispatches[2]{}, max_draws[2]{};
    u64 clear_first[2]{}, ctx_before[2]{}, sh_before[2]{}, ibs[2]{}, dwords[2]{};
    u64 hist[2][6]{}; // draws per buffer: 0, 1-9, 10-99, 100-499, 500-1999, 2000+
    // Packets by opcode (any depth), EVENT_WRITE by event type, EOP by data and interrupt
    // select, EOS by command, WAIT_REG_MEM by memory space and engine.
    u64 opcodes[256]{}, events[64]{}, eop_data[8]{}, eop_int[4]{}, eos_cmd[8]{}, wait_mem[2][2]{};
    u64 wait_outcome[4]{};
    // Dwords by opcode, and the port's Direct Vulkan NOP payloads by DirectVk::Op (count, dwords):
    // what share of the stream is still raw PS4 register/PM4 traffic.
    u64 op_dwords[256]{}, direct_ops[64]{}, direct_dwords[64]{};
    // Stage 0 Stage-A elimination audit: DirectVk vs Raw PM4 draw source, UID coverage & PSO churn.
    u64 audit_direct_draws = 0, audit_raw_draws = 0;
    u64 audit_direct_by_op[64]{}, audit_raw_by_op[256]{};
    u64 audit_cb_uid_hits = 0, audit_db_uid_hits = 0, audit_any_uid_hits = 0;
    u64 audit_shader_changes = 0, audit_pso_changes = 0;
    u64 audit_last_vs = ~0ULL, audit_last_ps = ~0ULL, audit_last_pso = ~0ULL;
    std::unordered_set<u64> audit_unique_shaders;
    std::unordered_set<u64> audit_unique_psos;
    struct RawDrawKey {
        u64 vs_addr;
        u64 ps_addr;
        u32 op_code;
        u32 mrt_mask;
        u16 rt0_w;
        u16 rt0_h;
        u16 cb0_uid;
        u16 db_uid;
        bool operator==(const RawDrawKey&) const = default;
    };
    struct RawDrawKeyHash {
        size_t operator()(const RawDrawKey& k) const noexcept {
            u64 h = k.vs_addr ^ (k.ps_addr * 0x9e3779b97f4a7c15ULL);
            h ^= (u64(k.op_code) << 32) ^ (u64(k.mrt_mask) << 16) ^ u64(k.rt0_w) ^
                 (u64(k.rt0_h) << 48);
            h ^= (u64(k.cb0_uid) << 8) ^ (u64(k.db_uid) << 24);
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            return size_t(h);
        }
    };
    std::unordered_map<RawDrawKey, u64, RawDrawKeyHash> audit_raw_sigs;
    double wait_ms = 0;
    std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
};
DcbStats stats;

// Atomic counters written from Stage B (bb:DrawRec) / Resolve0 and drained by Stage A every 5s when BB_DCB_STATS=1.
// kind: 0=ring_hit, 1=ring_not_copied, 2=cue_obtain, 3=direct_hit, 4=direct_miss_obtain,
//       5=other_obtain_ro, 6=other_obtain_rw, 7=hle_flush, 8=ring_alloc_fail
std::array<std::atomic<u64>, 9> g_sb_buf_counts{};
std::atomic<u64> g_sb_cue_obtain_bytes{0};
std::atomic<u64> g_sb_other_obtain_bytes{0};
// Stage 2A / 3C-1 FastPSO breakdown (Stage A only — plain non-atomic counters, zero LOCK XADD).
// kind: 0=sigEpochHit, 1=sigPairHit, 2=sigFullBuild, 3=flatRun, 4=flatSkipEpoch, 5=flatSkipMem,
//       6=matchUdSkip, 7=matchFastOk, 8=matchFastMiss, 9=matchNoCompile, 10=vsharpEpochSkip,
//       11=vsharpBatched, 12=matchSpecSkip, 13=matchSpecSkipErr
std::array<u64, 14> g_fast_pso_counts{};

const char* DirectVkOpShortName(u32 op) {
    switch (DirectVk::Op(op)) {
    case DirectVk::Op::CopyBufferBatch: return "CopyBufBatch";
    case DirectVk::Op::FillBuffer: return "FillBuf";
    case DirectVk::Op::ClearColorImage: return "ClearColor";
    case DirectVk::Op::CopyTextureRegion: return "CopyTex";
    case DirectVk::Op::DecompressDepth: return "DecompDepth";
    case DirectVk::Op::EliminateFastClear: return "ElimFastClr";
    case DirectVk::Op::ResolveColor: return "ResolveCol";
    case DirectVk::Op::UploadTextureSubresource: return "UploadTex";
    case DirectVk::Op::ClearDepthStencil: return "ClearDS";
    case DirectVk::Op::SetIndexState: return "SetIdxState";
    case DirectVk::Op::DrawIndexed: return "DrawIdx";
    case DirectVk::Op::DrawIndexAddr: return "DrawIdxAddr";
    case DirectVk::Op::DrawAuto: return "DrawAuto";
    case DirectVk::Op::DrawIndirect: return "DrawIndir";
    case DirectVk::Op::DispatchDirect: return "DispDirect";
    case DirectVk::Op::DispatchIndirect: return "DispIndir";
    case DirectVk::Op::SetStageUserData: return "SetUserData";
    case DirectVk::Op::SetContextRegsPacked: return "SetCtxPacked";
    case DirectVk::Op::SetRenderTarget: return "SetRT";
    case DirectVk::Op::SetDepthRenderTarget: return "SetDB";
    case DirectVk::Op::ReleaseLabelsBatch: return "RelLabels";
    case DirectVk::Op::OcclusionQuery: return "OccQuery";
    case DirectVk::Op::ReleaseU32LabelsBatch: return "RelU32Labels";
    case DirectVk::Op::TimestampQuery: return "TsQuery";
    case DirectVk::Op::BindVsShader: return "BindVS";
    case DirectVk::Op::BindPsShader: return "BindPS";
    case DirectVk::Op::BindCsShader: return "BindCS";
    case DirectVk::Op::BindEsShader: return "BindES";
    case DirectVk::Op::BindGsShader: return "BindGS";
    case DirectVk::Op::BindHsShader: return "BindHS";
    case DirectVk::Op::BindLsShader: return "BindLS";
    case DirectVk::Op::InitDefaultHardwareState: return "InitHwState";
    case DirectVk::Op::SetVgtControl: return "SetVgt";
    case DirectVk::Op::DrawBundle: return "DrawBundle";
    case DirectVk::Op::ExecuteSecondaryCmds: return "ExecSec";
    case DirectVk::Op::NativePostProcess: return "NativePostFX";
    case DirectVk::Op::GenerateMipsSpd: return "GenMipsSpd";
    case DirectVk::Op::SetDirectStateIds: return "SetDirState";
    default: return "Op?";
    }
}

bool IsDraw(Op op) {
    switch (op) {
    case Op::DrawIndex2:
    case Op::DrawIndexOffset2:
    case Op::DrawIndexAuto:
    case Op::DrawIndirect:
    case Op::DrawIndirectMulti:
    case Op::DrawIndexIndirect:
    case Op::DrawIndexIndirectMulti:
    case Op::DrawIndexIndirectCountMulti:
        return true;
    default:
        return false;
    }
}

} // namespace

bool DcbStatsEnabled() {
    static const bool enabled = EmulatorSettingsImpl::Flag("BB_DCB_STATS", false);
    return enabled;
}

void NoteWait(int outcome, double ms) {
    ++stats.wait_outcome[outcome];
    stats.wait_ms += ms;
}

void NoteStageBBufAudit(u32 kind, u64 bytes) noexcept {
    if (kind < g_sb_buf_counts.size()) {
        g_sb_buf_counts[kind].fetch_add(1u, std::memory_order_relaxed);
    }
    if (kind == 2u) {
        g_sb_cue_obtain_bytes.fetch_add(bytes, std::memory_order_relaxed);
    } else if (kind == 4u || kind == 5u || kind == 6u) {
        g_sb_other_obtain_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
}

void NoteFastPsoAudit(u32 kind) noexcept {
    if (kind < g_fast_pso_counts.size()) {
        ++g_fast_pso_counts[kind];
    }
}

void NoteDrawAudit(bool is_direct_vk, u32 op_code, u64 vs_addr, u64 ps_addr, u32 mrt_mask,
                   u32 rt0_w, u32 rt0_h, u16 cb0_uid, u16 db_uid, u64 pso_state_sig) {
    auto& st = stats;
    if (is_direct_vk) {
        ++st.audit_direct_draws;
        ++st.audit_direct_by_op[op_code & 63u];
    } else {
        ++st.audit_raw_draws;
        ++st.audit_raw_by_op[op_code & 255u];
        if (st.audit_raw_sigs.size() < 512u) {
            const DcbStats::RawDrawKey key{
                vs_addr, ps_addr, op_code, mrt_mask,
                static_cast<u16>(std::min<u32>(rt0_w, 65535u)),
                static_cast<u16>(std::min<u32>(rt0_h, 65535u)), cb0_uid, db_uid};
            ++st.audit_raw_sigs[key];
        }
    }
    if (cb0_uid != 0u) {
        ++st.audit_cb_uid_hits;
    }
    if (db_uid != 0u) {
        ++st.audit_db_uid_hits;
    }
    if (cb0_uid != 0u || db_uid != 0u) {
        ++st.audit_any_uid_hits;
    }
    const u64 sh_key = (vs_addr * 0x9e3779b97f4a7c15ULL) ^ (ps_addr + 0xbf58476d1ce4e5b9ULL);
    const u64 pso_key = sh_key ^ (pso_state_sig * 0x94d049bb133111ebULL);
    if (vs_addr != st.audit_last_vs || ps_addr != st.audit_last_ps) {
        ++st.audit_shader_changes;
        st.audit_last_vs = vs_addr;
        st.audit_last_ps = ps_addr;
    }
    if (pso_key != st.audit_last_pso) {
        ++st.audit_pso_changes;
        st.audit_last_pso = pso_key;
    }
    if (st.audit_unique_shaders.size() < 4096u) {
        st.audit_unique_shaders.insert(sh_key);
    }
    if (st.audit_unique_psos.size() < 8192u) {
        st.audit_unique_psos.insert(pso_key);
    }
}

void ScanDcb(std::span<const u32> dcb, int depth) {
    auto& st = stats;
    const int d = depth > 0 ? 1 : 0;
    u64 draws = 0, dispatches = 0, ctx = 0, sh = 0, ibs = 0;
    bool clear_first = false;
    st.dwords[d] += dcb.size();
    for (size_t at = 0; at < dcb.size();) {
        const Header header{dcb[at]};
        if (header.Type() == 2) {
            ++at;
            continue;
        }
        if (header.Type() != 3) {
            break;
        }
        const Op op = header.Opcode();
        const u32* body = dcb.data() + at + 1;
        ++st.opcodes[u8(op)];
        st.op_dwords[u8(op)] += header.BodyWords() + 1;
        if (op == Op::Nop && header.BodyWords() >= 2 && body[0] == NopPayload::DirectVulkan) {
            const u32 vk_op = body[1] & 0xFFFFu;
            st.direct_ops[vk_op & 63u] += 1;
            st.direct_dwords[vk_op & 63u] += header.BodyWords() + 1;
        }
        if (op == Op::EventWrite) {
            ++st.events[body[0] & 63];
        } else if (op == Op::EventWriteEop) {
            ++st.eop_data[(body[2] >> 29) & 7];
            ++st.eop_int[(body[2] >> 24) & 3];
        } else if (op == Op::EventWriteEos) {
            ++st.eos_cmd[(body[2] >> 29) & 7];
        } else if (op == Op::WaitRegMem) {
            ++st.wait_mem[(body[0] >> 4) & 1][(body[0] >> 8) & 1];
        }
        if (IsDraw(op)) {
            ++draws;
        } else if (op == Op::DispatchDirect || op == Op::DispatchIndirect) {
            ++dispatches;
        } else if (op == Op::IndirectBuffer) {
            ++ibs;
        } else if (draws == 0) {
            if (op == Op::ClearState) {
                clear_first = true;
            } else if (op == Op::SetContextReg) {
                ctx += header.BodyWords() - 1;
            } else if (op == Op::SetShReg) {
                sh += header.BodyWords() - 1;
            }
        }
        at += header.BodyWords() + 1;
    }
    ++st.buffers[d];
    st.draws[d] += draws;
    st.dispatches[d] += dispatches;
    st.ibs[d] += ibs;
    st.max_draws[d] = std::max(st.max_draws[d], draws);
    if (draws) {
        ++st.with_draws[d];
        st.clear_first[d] += clear_first;
        st.ctx_before[d] += ctx;
        st.sh_before[d] += sh;
    }
    const int bucket = draws == 0 ? 0 : draws < 10 ? 1 : draws < 100 ? 2 : draws < 500 ? 3
                     : draws < 2000 ? 4 : 5;
    ++st.hist[d][bucket];

    const auto now = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(now - st.window).count();
    if (depth != 0 || secs < 5.0) {
        return;
    }
    for (int k = 0; k < 2; ++k) {
        const double with = double(std::max<u64>(st.with_draws[k], 1));
        std::printf("DCB stats %s: %.0f buffers/s (%.0f with draws), %.0f draws/s, %.0f "
                    "dispatches/s, max %llu draws in one buffer, %.0f IBs/s, %.0f kdw/s; "
                    "draw buffers: %.0f%% start with ClearState, %.0f ctx + %.0f sh reg "
                    "writes before first draw; draws/buffer hist 0:%llu 1-9:%llu 10-99:%llu "
                    "100-499:%llu 500-1999:%llu 2000+:%llu\n",
                    k ? "indirect" : "submitted", st.buffers[k] / secs, st.with_draws[k] / secs,
                    st.draws[k] / secs, st.dispatches[k] / secs,
                    (unsigned long long)st.max_draws[k], st.ibs[k] / secs,
                    st.dwords[k] / secs / 1000.0, 100.0 * st.clear_first[k] / with,
                    st.ctx_before[k] / with, st.sh_before[k] / with,
                    (unsigned long long)st.hist[k][0], (unsigned long long)st.hist[k][1],
                    (unsigned long long)st.hist[k][2], (unsigned long long)st.hist[k][3],
                    (unsigned long long)st.hist[k][4], (unsigned long long)st.hist[k][5]);
    }
    std::string line;
    for (u32 op = 0; op < 256; ++op) {
        if (st.opcodes[op]) {
            line += fmt::format(" {}:{:.0f}", OpName(op), st.opcodes[op] / secs);
        }
    }
    std::printf("PM4 packets/s:%s\n", line.c_str());
    line.clear();
    for (u32 op = 0; op < 256; ++op) {
        if (st.op_dwords[op]) {
            line += fmt::format(" {}:{:.0f}", OpName(op), st.op_dwords[op] / secs);
        }
    }
    std::printf("PM4 dwords/s:%s\n", line.c_str());
    line.clear();
    for (u32 op = 0; op < 64; ++op) {
        if (st.direct_ops[op]) {
            line += fmt::format(" {}:{:.0f}/{:.0f}dw", op, st.direct_ops[op] / secs,
                                st.direct_dwords[op] / secs);
        }
    }
    std::printf("DirectVk ops/s (op:count/dwords):%s\n", line.c_str());
    line.clear();
    for (u32 event = 0; event < 64; ++event) {
        if (st.events[event]) {
            line += fmt::format(" {}:{:.0f}", event, st.events[event] / secs);
        }
    }
    std::printf("PM4 events/s:%s; EOP data_sel %.0f/%.0f/%.0f/%.0f/%.0f (none/32/64/clock/perf)"
                " int_sel %.0f/%.0f/%.0f/%.0f; EOS cmd %.0f/%.0f/%.0f/%.0f/%.0f; WAIT_REG_MEM "
                "reg me/pfp %.0f/%.0f mem me/pfp %.0f/%.0f\n",
                line.c_str(), st.eop_data[0] / secs, st.eop_data[1] / secs, st.eop_data[2] / secs,
                st.eop_data[3] / secs, st.eop_data[4] / secs, st.eop_int[0] / secs,
                st.eop_int[1] / secs, st.eop_int[2] / secs, st.eop_int[3] / secs,
                st.eos_cmd[0] / secs, st.eos_cmd[1] / secs, st.eos_cmd[2] / secs,
                st.eos_cmd[3] / secs, st.eos_cmd[4] / secs, st.wait_mem[0][0] / secs,
                st.wait_mem[0][1] / secs, st.wait_mem[1][0] / secs, st.wait_mem[1][1] / secs);
    std::printf("WAIT_REG_MEM outcomes/s: VO label %.0f, pending fence %.0f, met %.0f, waited "
                "%.0f (%.2f ms/s)\n",
                st.wait_outcome[0] / secs, st.wait_outcome[1] / secs, st.wait_outcome[2] / secs,
                st.wait_outcome[3] / secs, st.wait_ms / secs);

    // Stage 0 Stage-A elimination audit summary
    {
        u64 total_pkts = 0, total_dw = 0, dvk_pkts = 0, dvk_dw = 0;
        for (u32 op = 0; op < 256; ++op) {
            total_pkts += st.opcodes[op];
            total_dw += st.op_dwords[op];
        }
        for (u32 op = 0; op < 64; ++op) {
            dvk_pkts += st.direct_ops[op];
            dvk_dw += st.direct_dwords[op];
        }
        const u64 raw_pkts = total_pkts >= dvk_pkts ? (total_pkts - dvk_pkts) : 0;
        const u64 raw_dw = total_dw >= dvk_dw ? (total_dw - dvk_dw) : 0;
        const u64 tot_draws = st.audit_direct_draws + st.audit_raw_draws;
        const double inv_draws = tot_draws > 0 ? (100.0 / double(tot_draws)) : 0.0;
        std::printf(
            "[StageA-Audit] Stream: DirectVk %.1f%% pkts (%.0f/s) / %.1f%% dw (%.0f kdw/s) | "
            "RawPM4 %.1f%% pkts (%.0f/s) / %.1f%% dw (%.0f kdw/s)\n",
            total_pkts ? 100.0 * dvk_pkts / total_pkts : 0.0, dvk_pkts / secs,
            total_dw ? 100.0 * dvk_dw / total_dw : 0.0, dvk_dw / secs / 1000.0,
            total_pkts ? 100.0 * raw_pkts / total_pkts : 0.0, raw_pkts / secs,
            total_dw ? 100.0 * raw_dw / total_dw : 0.0, raw_dw / secs / 1000.0);
        const auto dt_stats = Vulkan::DirectTargets::GetStats();
        const auto ds_stats = Vulkan::DirectStates::GetStats();
        std::printf(
            "[StageA-Audit] Draws: DirectVk %.0f/s (%.2f%%) vs RawPM4 %.0f/s (%.2f%%) | "
            "RT-UID coverage: any=%.1f%% cb0=%.1f%% db=%.1f%% (slots=%u stackSkips=%llu wraps=%llu) | "
            "DirectStates(rs=%u bs=%u ds=%u psU=%u) | "
            "PSO locality: uniqSh=%zu uniqPSO=%zu, shChg=%.1f%% psoChg=%.1f%%\n",
            st.audit_direct_draws / secs, st.audit_direct_draws * inv_draws,
            st.audit_raw_draws / secs, st.audit_raw_draws * inv_draws,
            st.audit_any_uid_hits * inv_draws, st.audit_cb_uid_hits * inv_draws,
            st.audit_db_uid_hits * inv_draws, dt_stats.allocated_slots,
            (unsigned long long)dt_stats.stack_skips, (unsigned long long)dt_stats.wrap_count,
            ds_stats.raster_unique, ds_stats.blend_unique, ds_stats.ds_unique,
            ds_stats.ps_usage_unique,
            st.audit_unique_shaders.size(),
            st.audit_unique_psos.size(), st.audit_shader_changes * inv_draws,
            st.audit_pso_changes * inv_draws);
        std::string dline;
        for (u32 op = 0; op < 64; ++op) {
            if (st.audit_direct_by_op[op]) {
                dline += fmt::format(" {}:{:.0f}/s", DirectVkOpShortName(op),
                                     st.audit_direct_by_op[op] / secs);
            }
        }
        if (!dline.empty()) {
            std::printf("[StageA-Audit] DirectVk Draws:%s\n", dline.c_str());
        }
        if (st.audit_raw_draws > 0) {
            std::string rline;
            for (u32 op = 0; op < 256; ++op) {
                if (st.audit_raw_by_op[op]) {
                    rline += fmt::format(" {}:{:.0f}/s", OpName(op),
                                         st.audit_raw_by_op[op] / secs);
                }
            }
            std::printf("[StageA-Audit] RawPM4 Draws:%s\n", rline.c_str());
            std::vector<std::pair<DcbStats::RawDrawKey, u64>> top(st.audit_raw_sigs.begin(),
                                                                  st.audit_raw_sigs.end());
            std::sort(top.begin(), top.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            const size_t show = std::min<size_t>(top.size(), 12u);
            for (size_t i = 0; i < show; ++i) {
                const auto& [k, cnt] = top[i];
                std::printf(
                    "  [RawDraw #%zu] %.1f/s (%.2f%%) op=%s vs=%#llx ps=%#llx mrt=%#x "
                    "rt=%ux%u cbUid=%u dbUid=%u\n",
                    i + 1, cnt / secs, cnt * inv_draws, OpName(k.op_code).c_str(),
                    (unsigned long long)k.vs_addr, (unsigned long long)k.ps_addr, k.mrt_mask,
                    k.rt0_w, k.rt0_h, k.cb0_uid, k.db_uid);
            }
        } else {
            std::printf("[StageA-Audit] RawPM4 Draws: ZERO (100.00%% DirectVk!)\n");
        }
        {
            const u64 ring_hit = g_sb_buf_counts[0].exchange(0, std::memory_order_relaxed);
            const u64 ring_nc = g_sb_buf_counts[1].exchange(0, std::memory_order_relaxed);
            const u64 cue_obt = g_sb_buf_counts[2].exchange(0, std::memory_order_relaxed);
            const u64 dir_hit = g_sb_buf_counts[3].exchange(0, std::memory_order_relaxed);
            const u64 dir_miss = g_sb_buf_counts[4].exchange(0, std::memory_order_relaxed);
            const u64 oth_ro = g_sb_buf_counts[5].exchange(0, std::memory_order_relaxed);
            const u64 oth_rw = g_sb_buf_counts[6].exchange(0, std::memory_order_relaxed);
            const u64 hle_fl = g_sb_buf_counts[7].exchange(0, std::memory_order_relaxed);
            const u64 r_fail = g_sb_buf_counts[8].exchange(0, std::memory_order_relaxed);
            const u64 cue_b = g_sb_cue_obtain_bytes.exchange(0, std::memory_order_relaxed);
            const u64 oth_b = g_sb_other_obtain_bytes.exchange(0, std::memory_order_relaxed);
            std::printf(
                "[StageB-BufAudit] ringHit=%.0f/s ringNotCopied=%.0f/s ringAllocFail=%.0f/s | "
                "cueObtain=%.0f/s (%.1f MiB/s) directHit=%.0f/s directMissObt=%.0f/s "
                "otherRO=%.0f/s otherRW=%.0f/s (%.1f MiB/s) hleFlush=%.0f/s | cueGpuMod=%d\n",
                ring_hit / secs, ring_nc / secs, r_fail / secs, cue_obt / secs,
                cue_b / secs / (1024.0 * 1024.0), dir_hit / secs, dir_miss / secs, oth_ro / secs,
                oth_rw / secs, oth_b / secs / (1024.0 * 1024.0), hle_fl / secs,
                int(VideoCore::DirectBuffers::IsCueHeapEverGpuModified()));
            const u64 p0 = g_fast_pso_counts[0];
            const u64 p1 = g_fast_pso_counts[1];
            const u64 p2 = g_fast_pso_counts[2];
            const u64 p3 = g_fast_pso_counts[3];
            const u64 p4 = g_fast_pso_counts[4];
            const u64 p5 = g_fast_pso_counts[5];
            const u64 p6 = g_fast_pso_counts[6];
            const u64 p7 = g_fast_pso_counts[7];
            const u64 p8 = g_fast_pso_counts[8];
            const u64 p9 = g_fast_pso_counts[9];
            const u64 p10 = g_fast_pso_counts[10];
            const u64 p11 = g_fast_pso_counts[11];
            const u64 p12 = g_fast_pso_counts[12];
            const u64 p13 = g_fast_pso_counts[13];
            g_fast_pso_counts.fill(0u);
            if ((p0 + p1 + p2) > 0u) {
                std::printf(
                    "[FastPSO-Audit] Sig(epoch=%.0f/s pair=%.0f/s full=%.0f/s) | "
                    "Flat(skipEp=%.0f/s skipMem=%.0f/s run=%.0f/s) | "
                    "Match(udSkip=%.0f/s specSkip=%.0f/s fastOk=%.0f/s fastMiss=%.0f/s "
                    "noComp=%.0f/s specErr=%llu) | "
                    "V#(skip=%.0f/s batch=%.0f/s)\n",
                    p0 / secs, p1 / secs, p2 / secs, p4 / secs, p5 / secs, p3 / secs, p6 / secs,
                    p12 / secs, p7 / secs, p8 / secs, p9 / secs,
                    static_cast<unsigned long long>(p13), p10 / secs, p11 / secs);
            }
        }
        std::fflush(stdout);
    }
    st = DcbStats{};
}

namespace {
struct SubmittedCopyPool {
    static constexpr std::size_t MaxEntries = 128;
    static constexpr std::size_t MaxReusableCapacityDwords = 512 * 1024; // 2 MiB
    std::mutex mutex;
    std::vector<SubmittedCopy*> free_list;

    ~SubmittedCopyPool() {
        for (SubmittedCopy* c : free_list) {
            delete c;
        }
    }
};

SubmittedCopyPool& GetSubmittedCopyPool() {
    // Intentionally leaked at process exit so surviving background/shared_ptr deleters never race
    // static destruction order.
    static auto* pool = new SubmittedCopyPool();
    return *pool;
}
} // namespace

std::shared_ptr<SubmittedCopy> AcquireSubmittedCopy(std::size_t dwords) {
    SubmittedCopy* raw = nullptr;
    {
        auto& pool = GetSubmittedCopyPool();
        std::scoped_lock lk{pool.mutex};
        if (!pool.free_list.empty()) {
            raw = pool.free_list.back();
            pool.free_list.pop_back();
        }
    }
    if (!raw) {
        raw = new SubmittedCopy();
    }
    raw->data.resize_uninit(dwords);
    raw->guest_dcb = nullptr;
    raw->dcb_dwords = 0;
    raw->guest_ccb = nullptr;
    raw->ccb_dwords = 0;
    raw->bad_dcb_at_submit = ~std::size_t(0);
    raw->inspected = false;
    return std::shared_ptr<SubmittedCopy>(raw, [](SubmittedCopy* c) {
        if (c->data.capacity() <= SubmittedCopyPool::MaxReusableCapacityDwords) {
            auto& pool = GetSubmittedCopyPool();
            std::scoped_lock lk{pool.mutex};
            if (pool.free_list.size() < SubmittedCopyPool::MaxEntries) {
                pool.free_list.push_back(c);
                return;
            }
        }
        delete c;
    });
}

bool PreflightInspectEnabled() noexcept {
    static const bool enabled = [] {
        const char* trace = std::getenv("BB_GNM_WRITES_TRACE");
        const char* preflight = std::getenv("BB_PM4_PREFLIGHT");
        return (trace && trace[0] == '1') || (preflight && preflight[0] == '1');
    }();
    return enabled;
}

void InspectSubmittedCopy(SubmittedCopy& copy) {
    if (copy.inspected) {
        return;
    }
    copy.inspected = true;
    copy.bad_dcb_at_submit = ~std::size_t(0);
    for (size_t at = 0; at < copy.dcb_dwords;) {
        const u32 header = copy.data[at];
        if ((header >> 30) == 2) {
            ++at;
            continue;
        }
        const size_t length = ((header >> 16) & 0x3fff) + 2;
        if ((header >> 30) != 3 || length > copy.dcb_dwords - at) {
            copy.bad_dcb_at_submit = at;
            return;
        }
        at += length;
    }
}

void RememberSubmittedCopy(std::shared_ptr<const SubmittedCopy> copy) {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_GNM_WRITES_TRACE");
        return env && env[0] == '1';
    }();
    if (!enabled) return;
    InspectSubmittedCopy(const_cast<SubmittedCopy&>(*copy));
    struct Kept { u64 number; std::shared_ptr<const SubmittedCopy> copy; };
    static std::mutex mutex;
    static std::deque<Kept> history;
    static u64 next = 0, kept_bytes = 0;
    std::scoped_lock lock{mutex};
    const u64 number = next++;
    const auto begin = reinterpret_cast<uintptr_t>(copy->guest_dcb);
    if (copy->bad_dcb_at_submit != ~size_t(0)) {
        const auto bad = begin + copy->bad_dcb_at_submit * 4;
        std::fprintf(stderr, "PM4 preflight history #%llu: source %#llx +%#llx bytes, first bad +%#llx\n",
                     (unsigned long long)number, (unsigned long long)begin,
                     (unsigned long long)(copy->dcb_dwords * 4),
                     (unsigned long long)(copy->bad_dcb_at_submit * 4));
        unsigned shown = 0;
        for (auto it = history.rbegin(); it != history.rend() && shown < 3; ++it) {
            const auto& old = *it->copy;
            const auto old_begin = reinterpret_cast<uintptr_t>(old.guest_dcb);
            if (bad < old_begin || bad >= old_begin + old.dcb_dwords * 4) continue;
            const size_t word = (bad - old_begin) / 4;
            size_t packet = 0;
            while (packet < old.dcb_dwords) {
                const auto h = old.data[packet];
                const size_t n = (h >> 30) == 2 ? 1 : ((h >> 16) & 0x3fff) + 2;
                if (((h >> 30) != 3 && (h >> 30) != 2) ||
                    n > old.dcb_dwords - packet || packet + n > word) break;
                packet += n;
            }
            std::fprintf(stderr, "  prior #%llu source %#llx +%#llx bytes; bad address was +%#llx, "
                                 "packet at +%#llx header %#x\n",
                         (unsigned long long)it->number, (unsigned long long)old_begin,
                         (unsigned long long)(old.dcb_dwords * 4), (unsigned long long)(word * 4),
                         (unsigned long long)(packet * 4), packet < old.dcb_dwords ? old.data[packet] : 0);
            const size_t from = word > 16 ? word - 16 : 0;
            std::fprintf(stderr, "    prior words from +%#llx:", (unsigned long long)(from * 4));
            for (size_t p = from; p < std::min(old.dcb_dwords, word + 16); ++p)
                std::fprintf(stderr, p == word ? " [%08x]" : " %08x", old.data[p]);
            std::fprintf(stderr, "\n");
            const char* dir = std::getenv("BB_DATA_DIR");
            const auto path = fmt::format("{}/out/pm4-history-{}-prior-{}.bin", dir ? dir : ".",
                                          number, it->number);
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(old.data.data(), sizeof(u32), old.dcb_dwords, f);
                std::fclose(f);
                std::fprintf(stderr, "    prior snapshot: %s\n", path.c_str());
            }
            ++shown;
        }
        if (!shown) std::fprintf(stderr, "  no retained submission covers the bad address\n");
        const auto from = copy->bad_dcb_at_submit > 16 ? copy->bad_dcb_at_submit - 16 : 0;
        BbGnmHooks::ReportDriverWrites(begin + from * 4,
                                       std::min<size_t>(32, copy->dcb_dwords - from) * 4);
    }
    kept_bytes += copy->data.size() * sizeof(u32);
    history.push_back({number, std::move(copy)});
    while (history.size() > 512 || kept_bytes > (64u << 20)) {
        kept_bytes -= history.front().copy->data.size() * sizeof(u32);
        history.pop_front();
    }
}

void ReportBadPacket(uintptr_t base, std::size_t dwords, const u32* at, u64 seq, int depth,
                     const SubmittedCopy* copy) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(at);
    int prot = 0, type = -1;
    uintptr_t end = 0;
    runtime_memory_vma_info(address, &prot, &type, &end);
    std::fprintf(stderr,
                 "Bad PM4 packet at %#llx: dword %#x, %#llx bytes into a %#llx-byte buffer at %#llx "
                 "(IB depth %d, submission %llu; memory type %d, prot %#x)\n",
                 (unsigned long long)address, *at, (unsigned long long)(address - base),
                 (unsigned long long)(dwords * 4), (unsigned long long)base, depth,
                 (unsigned long long)seq, type, prot);
    const u32* first = reinterpret_cast<const u32*>(std::max(base, address - 64));
    const u32* last = std::min(reinterpret_cast<const u32*>(base) + dwords, at + 16);
    for (const u32* p = first; p < last; p += 8) {
        std::fprintf(stderr, "  %#llx:", (unsigned long long)reinterpret_cast<uintptr_t>(p));
        for (const u32* q = p; q < std::min(p + 8, last); ++q) {
            std::fprintf(stderr, q == at ? " [%08x]" : " %08x", *q);
        }
        std::fprintf(stderr, "\n");
    }
    if (copy && depth == 0 &&
        base == reinterpret_cast<uintptr_t>(copy->data.data())) {
        InspectSubmittedCopy(const_cast<SubmittedCopy&>(*copy));
        const size_t bad = size_t(at - copy->data.data());
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - copy->submitted).count();
        std::fprintf(stderr, "  submission source %p, snapshot age %.3f ms; ",
                     static_cast<const void*>(copy->guest_dcb), ms);
        if (copy->bad_dcb_at_submit == ~size_t(0)) {
            std::fprintf(stderr, "snapshot had valid packet boundaries at submit\n");
        } else {
            std::fprintf(stderr, "snapshot ALREADY invalid at submit +%#llx bytes\n",
                         (unsigned long long)(copy->bad_dcb_at_submit * 4));
        }
        const size_t from = bad > 16 ? bad - 16 : 0;
        const size_t count = std::min<size_t>(32, copy->dcb_dwords - from);
        std::array<u32, 32> source{};
        iovec local{source.data(), count * 4};
        iovec remote{const_cast<u32*>(copy->guest_dcb + from), count * 4};
        const bool read = process_vm_readv(getpid(), &local, 1, &remote, 1, 0) ==
                              ssize_t(count * 4) ||
                          ReadGuest(copy->guest_dcb + from, source.data(), u32(count * 4));
        if (read) {
            std::fprintf(stderr, "  source now at bad header: %#x (snapshot %#x)\n",
                         source[bad - from], *at);
            for (size_t p = 0; p < count; p += 8) {
                std::fprintf(stderr, "  source +%04llx:", (unsigned long long)((from + p) * 4));
                for (size_t q = p; q < std::min(p + 8, count); ++q) {
                    std::fprintf(stderr, from + q == bad ? " [%08x]" : " %08x", source[q]);
                }
                std::fprintf(stderr, "\n");
            }
        } else {
            std::fprintf(stderr, "  source memory is no longer readable\n");
        }
        // Track writes to guest memory, not the malloc'ed snapshot (which has none by design).
        BbWriteLog::DumpRange(reinterpret_cast<uintptr_t>(copy->guest_dcb), dwords * 4);
        BbGnmHooks::ReportDriverWrites(reinterpret_cast<uintptr_t>(copy->guest_dcb + from), count * 4);
        const char* data_dir = std::getenv("BB_DATA_DIR");
        const std::string name = std::string(data_dir ? data_dir : ".") +
                                 "/out/pm4-bad-" + std::to_string(seq) + ".snapshot.bin";
        if (FILE* file = std::fopen(name.c_str(), "wb")) {
            const size_t written = std::fwrite(copy->data.data(), sizeof(u32), copy->dcb_dwords, file);
            std::fclose(file);
            if (written == copy->dcb_dwords) std::fprintf(stderr, "  full snapshot: %s\n", name.c_str());
        }
    } else {
        BbWriteLog::DumpRange(base, dwords * 4);
    }
    std::fflush(stderr);
}

void CheckSubmittedCopy(const SubmittedCopy& copy, u64 seq) {
    // The guest memory may be unmapped by now: read it without faulting.
    static std::atomic<u64> changed{0}, checked{0};
    const auto first_change = [](const u32* guest, const u32* kept, std::size_t dwords) -> s64 {
        std::array<u32, 1024> buf;
        for (std::size_t at = 0; at < dwords; at += buf.size()) {
            const std::size_t n = std::min(buf.size(), dwords - at);
            iovec local{buf.data(), n * 4}, remote{const_cast<u32*>(guest + at), n * 4};
            if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) != ssize_t(n * 4)) {
                // bbport BB_GUEST_IN_PLACE: dma-buf guest memory is not readable that way; still
                // mapped, it is read directly.
                int prot = 0, type = -1;
                uintptr_t end = 0;
                const auto from = reinterpret_cast<uintptr_t>(guest + at);
                if (!runtime_memory_vma_info(from, &prot, &type, &end) || end < from + n * 4) {
                    return -2 - s64(at); // unmapped
                }
                std::memcpy(buf.data(), guest + at, n * 4);
            }
            for (std::size_t i = 0; i < n; ++i) {
                if (buf[i] != kept[at + i]) {
                    return s64(at + i);
                }
            }
        }
        return -1;
    };
    checked.fetch_add(1, std::memory_order_relaxed);
    const s64 dcb_change = first_change(copy.guest_dcb, copy.data.data(), copy.dcb_dwords);
    const s64 ccb_change =
        first_change(copy.guest_ccb, copy.data.data() + copy.dcb_dwords, copy.ccb_dwords);
    if (dcb_change == -1 && ccb_change == -1) {
        return;
    }
    const u64 n = changed.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n > 16 && n % 100 != 0) {
        return;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                copy.submitted)
                          .count();
    const s64 at = dcb_change != -1 ? dcb_change : ccb_change;
    std::printf("Command buffers: the guest changed submission %llu's %s at %p +%#llx (%s) "
                "%.3f ms after submitting it, before it was decoded; %llu of %llu so far "
                "(decoded from the copy)\n",
                (unsigned long long)seq, dcb_change != -1 ? "dcb" : "ccb",
                static_cast<const void*>(dcb_change != -1 ? copy.guest_dcb : copy.guest_ccb),
                (unsigned long long)((at < -1 ? -2 - at : at) * 4),
                at < -1 ? "unmapped" : "rewritten", ms, (unsigned long long)n,
                (unsigned long long)checked.load());
    if (n > 4 || at < 0) {
        return;
    }
    // What changed, and the packets before it (a WaitRegMem the guest releases after patching,
    // or a write of ours into its own command buffer).
    const u32* kept = dcb_change != -1 ? copy.data.data() : copy.data.data() + copy.dcb_dwords;
    const u32* guest = dcb_change != -1 ? copy.guest_dcb : copy.guest_ccb;
    const std::size_t dwords = dcb_change != -1 ? copy.dcb_dwords : copy.ccb_dwords;
    const std::size_t from = at >= 8 ? at - 8 : 0, to = std::min<std::size_t>(dwords, at + 8);
    std::array<u32, 16> now{};
    iovec local{now.data(), (to - from) * 4}, remote{const_cast<u32*>(guest + from), (to - from) * 4};
    const bool readable = process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == ssize_t((to - from) * 4);
    std::printf("  submitted:");
    for (std::size_t i = from; i < to; ++i) {
        std::printf(i == std::size_t(at) ? " [%08x]" : " %08x", kept[i]);
    }
    std::printf("\n  now:      ");
    for (std::size_t i = from; readable && i < to; ++i) {
        std::printf(i == std::size_t(at) ? " [%08x]" : " %08x", now[i - from]);
    }
    std::printf("\n  packets before it (dword offset:opcode):");
    std::array<std::pair<u32, u32>, 12> last{};
    std::size_t count = 0;
    for (std::size_t p = 0; p < std::size_t(at) && p < dwords;) {
        const u32 header = kept[p];
        if ((header >> 30) == 3) {
            last[count++ % last.size()] = {u32(p), (header >> 8) & 0xff};
            p += ((header >> 16) & 0x3fff) + 2;
        } else if ((header >> 30) == 2) {
            ++p;
        } else {
            break;
        }
    }
    for (std::size_t i = count > last.size() ? count - last.size() : 0; i < count; ++i) {
        std::printf(" %#x:%#x", last[i % last.size()].first, last[i % last.size()].second);
    }
    std::printf("\n");
}

} // namespace BbCp::Diagnostics
