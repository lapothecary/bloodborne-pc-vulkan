// SPDX-License-Identifier: GPL-2.0-or-later
// bbport cp: the command processor. Games submit command buffers (graphics: draw and constant
// engine buffers; compute: ring buffers per queue); one thread (bb:GpuComm) decodes them in order,
// keeps the register state, hands draws and dispatches to the renderer and does what the other
// packets ask (fences, memory writes, waits, predication, the constant RAM).
//
// Each submission is a walk over its buffer and the indirect buffers it calls, kept as a stack
// of positions: a packet that has to wait (WAIT_REG_MEM, MEM_SEMAPHORE, REWIND, the constant
// engine) leaves the walk where it is, and the thread goes on with the other queues' work, as the
// hardware's queues run side by side. Commands from other threads (SendCommand) run between
// packets.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <span>
#include <thread>
#include <vector>

#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "cp/cp_packets.h"
#include "platform/gnm/gnm.h"
#include "gcn/regs_targets.h"
#include "gcn/regs.h"

namespace Vulkan {
class Rasterizer;
class DrawPreparation;
class NativeStreamShadow;
} // namespace Vulkan

namespace BbVideoOut {
struct Port;
}

namespace BbCp::Diagnostics {
struct SubmittedCopy;
}

namespace BbCp {

using Gcn::CbDbExtent;
using Gcn::ComputeProgram;
using Gcn::RegDirty;
using Gcn::Regs;

class CommandProcessor {
public:
    static constexpr u32 GfxQueueId = 0;
    static constexpr u32 NumGfxRings = 1;     ///< the high-priority ring is the system's
    static constexpr u32 NumComputePipes = 7; ///< the eighth is the system's
    static constexpr u32 NumQueuesPerPipe = 8;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;
    /// Top-level graphics submissions are numbered for the draw preparation workers.
    static constexpr u64 NoSeq = ~0ull;

    CommandProcessor();
    ~CommandProcessor();

    /// force_copy owns transient internal commands even when guest buffer copying is disabled.
    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb, bool force_copy = false);
    void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);
    /// sceGnmSubmitDone. `frame`: the guest's frame number, reported finished
    /// (BbGnm::NoteFramesRetired) once the GPU has done everything submitted so far.
    void SubmitDone(u64 frame = 0) noexcept;
    void WaitGpuIdle() noexcept;
    /// Nothing to decode, and (BB_WORK_RETIRED, default) the draw recording thread and the fences
    /// it deferred are done too. sceGnmSubmitDone does not block the guest when this is true.
    bool IsGpuIdle() const;

    void SetVoPort(BbVideoOut::Port* port) {
        vo_port = port;
    }
    void BindRasterizer(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
    }

    /// Runs `func` on the command processor's thread, between packets.
    template <bool wait_done = false>
    void SendCommand(auto&& func) {
        if constexpr (wait_done) {
            std::binary_semaphore done{0};
            {
                std::scoped_lock lock{submit_mutex};
                commands.emplace_back([&done, &func] {
                    func();
                    done.release();
                });
                ++num_commands;
                submit_cv.notify_one();
            }
            done.acquire();
        } else {
            std::scoped_lock lock{submit_mutex};
            commands.emplace_back(std::move(func));
            ++num_commands;
            submit_cv.notify_one();
        }
    }

    /// The compute program of the queue being decoded.
    ComputeProgram& GetCsRegs() {
        return queues[current_queue < 0 ? 0 : current_queue].cs_state;
    }

    std::thread::id GetGpuCommandProcessorThread() const {
        return thread_id;
    }
    u32 GetGpuCommandProcessorThreadId() const {
        return thread_tid;
    }

    /// A compute ring the game mapped (sceGnmMapComputeQueue).
    struct AscQueueInfo {
        static constexpr size_t Pm4BufferSize = 1024;
        VAddr map_addr;
        u32* read_addr;
        u32 ring_size_dw;
        u32 pipe_id;
        std::array<u32, Pm4BufferSize> tmp_packet; ///< a packet split by the ring's end
        u32 tmp_dwords;
    };
    Common::SlotVector<AscQueueInfo> asc_queues{};

    /// Stage 3B / 3C-1: Direct dense 10-bit Raster/Blend/DS state IDs, exact RT format signature,
    /// and per-HW-stage spec-affecting user-data epoch (`stage_spec_epoch[0..5]`) tracked
    /// alongside `Gcn::Regs` for lossless state-keyed PSO lookup, CB-only `SpecWordCheck` bypass,
    /// and `NativeCmdStream`.
    struct DirectStateTrack {
        u16 rs_id = 0;
        u16 bs_id = 0;
        u16 ds_id = 0;
        u16 ps_usage_id = 0;
        u16 rt_fmt_dirty = 1;
        u64 rt_fmt_sig = 0;
        u64 pso_misc_state_epoch = 1;
        /// Monotonic per-HW-stage (0=PS..5=LS) epoch incremented ONLY when user_data or shader
        /// bindings may affect `SpecWordCheck` (not bumped by `kUdFlagSpecInvariant` CB-only updates).
        std::array<u64, 6> stage_spec_epoch{1, 1, 1, 1, 1, 1};

        void ResetAll() noexcept {
            rs_id = 0;
            bs_id = 0;
            ds_id = 0;
            ps_usage_id = 0;
            rt_fmt_dirty = 1;
            ++pso_misc_state_epoch;
            for (u64& ep : stage_spec_epoch) {
                ++ep;
            }
        }
    };

    /// The register-writing packets' effect on `regs` (shared with the draw preparation workers,
    /// which replay the stream): the blocks written go to `dirty`, the packet into `checksum`
    /// (equal checksums at a draw mean equal register files). Returns true when the packet is a
    /// pure register write that needs no further handling in CommandProcessor::Graphics.
    static bool ApplyGraphicsRegisterPacket(Regs& regs, const u32* packet, u64& checksum,
                                            RegDirty* dirty = nullptr,
                                            bool compute_checksum = true,
                                            u64* pso_sig_epoch = nullptr,
                                            u64* dyn_reg_epoch = nullptr,
                                            u64* pso_non_shader_epoch = nullptr,
                                            u64* stage_ud_epoch = nullptr,
                                            u64* flat_invalidation_epoch = nullptr,
                                            DirectStateTrack* direct_state = nullptr);
    static bool InitGnmDefaultRegs(Regs& regs, u32 gen = 3u, RegDirty* dirty = nullptr,
                                   bool track_all = false, u64* pso_sig_epoch = nullptr,
                                   u64* dyn_reg_epoch = nullptr) noexcept;
    static bool ApplyDrawBundleRegs(Regs& regs, RegDirty* dirty, bool track_all, u32 op_word,
                                    const u32* body, u32 body_words,
                                    u64* pso_sig_epoch = nullptr,
                                    u64* dyn_reg_epoch = nullptr,
                                    u64* pso_non_shader_epoch = nullptr,
                                    u64* stage_ud_epoch = nullptr,
                                    DirectStateTrack* direct_state = nullptr) noexcept;
    static u64 HashRegisterPacket(u64 checksum, const u32* words, u32 count);

    Regs regs{};
    /// Render target extents from the driver's NOP hints after CB/DB setup.
    std::array<CbDbExtent, Gcn::NUM_COLOR_BUFFERS> last_cb_extent{};
    CbDbExtent last_db_extent{};
    /// V3.1: Direct pre-registered RenderTarget / DepthTarget 12-bit UIDs (0 = unregistered).
    std::array<u16, Gcn::NUM_COLOR_BUFFERS> last_cb_uid{};
    u16 last_db_uid{};
    /// Stage 3B: Direct state IDs & RT format signature tracked in lockstep with `regs`.
    DirectStateTrack direct_state{};
    /// Register blocks written since the last draw handed to the draw recording thread.
    RegDirty pipe_dirty;
    /// Running checksum of graphics-register packets (ApplyGraphicsRegisterPacket).
    u64 gfx_reg_checksum{};
    /// Monotonic epoch incremented when any register affecting BuildFastSelectSig changes.
    u64 pso_sig_epoch = 1;
    /// Monotonic epoch incremented when any context/uconfig register affecting DynamicState changes.
    u64 dyn_reg_epoch = 1;
    /// Stage 2A: Monotonic epoch incremented on PSO-affecting state changes OTHER than full
    /// DirectVk BindVsShader / BindPsShader / GenerateAndEmitPsShaderUsage.
    u64 pso_non_shader_epoch = 1;
    /// Stage 2A: Per-HW-stage (0=PS..5=LS) monotonic epoch incremented when user_data SGPRs change.
    std::array<u64, 6> stage_ud_epoch{1, 1, 1, 1, 1, 1};
    /// Stage 2A: Invalidated on submission start, ClearState, InitHwState, DMA/Copy/WriteData,
    /// ExecuteSecondaryCmds, and Raw PM4 register writes / draws.
    u64 flat_invalidation_epoch = 1;
    /// Stage 2A: True when the active draw originated from DirectVk (`dantelion_gx`).
    bool current_draw_is_direct_vk = false;
    /// Stage 3B-1: Lazily allocated NativeCmdStream shadow producer & Gcn::Regs parity verifier
    /// (`nullptr` when `BB_NATIVE_STREAM=0`, zero overhead on default path).
    std::unique_ptr<Vulkan::NativeStreamShadow> native_stream_shadow;

    void OnStageADraw(bool is_direct_vk, u32 op_code, u32 start_index = 0u,
                      const u32* raw_pkt = nullptr, u32 raw_dwords = 0u) noexcept;

private:
    struct Level {
        const u32* begin;
        const u32* at;
        const u32* end;
        int depth;
        bool prepared; ///< the submission's own buffer: its direct draws were prepared ahead
        bool entered;  ///< its start seen (statistics scan, constant engine counters reset)
        const u32* prev_at = nullptr; ///< the packet walked before this one, and the header it was walked with
        u32 prev_header = 0;
        int recoveries = 0;
    };
    /// A command buffer and the indirect buffers it called, as a stack.
    struct Walk {
        std::array<Level, 10> levels;
        int top = -1;
        void Push(std::span<const u32> buffer, int depth, bool prepared = false) {
            ASSERT_MSG(top + 1 < int(levels.size()), "indirect buffers nested too deep");
            const u32* begin = buffer.data();
            levels[++top] = {begin, begin, begin + buffer.size(), depth, prepared, false, nullptr, 0, 0};
        }
        bool Done() const {
            return top < 0;
        }
    };
    /// The constant engine's counters (shared by the CCB and DCB of a submission).
    struct ConstantEngine {
        u32 ce_count = 0;
        u32 de_count = 0;
        void Reset() {
            ce_count = de_count = 0;
        }
    };
    using SubmittedCopy = Diagnostics::SubmittedCopy;
    enum class WaitKind : u8 {
        None = 0,
        RewindBit31, ///< `(*word32 >> 31) != 0`
        RegMem32,    ///< `WaitRegMem::Compare(fn, *word32, ref, mask)`
        Semaphore64, ///< `__atomic_load_n(word64, __ATOMIC_ACQUIRE) > 0`
    };
    struct WaitHint {
        WaitKind kind = WaitKind::None;
        u8 function = 0;
        VAddr address = 0;
        u32 reference = 0;
        u32 mask = 0;

        [[nodiscard]] bool IsReady() const noexcept {
            switch (kind) {
            case WaitKind::RewindBit31:
                return (*reinterpret_cast<const volatile u32*>(address) >> 31) != 0u;
            case WaitKind::RegMem32:
                return WaitRegMem::Compare(function,
                                           *reinterpret_cast<const volatile u32*>(address),
                                           reference, mask);
            case WaitKind::Semaphore64:
                return __atomic_load_n(reinterpret_cast<const u64*>(address), __ATOMIC_ACQUIRE) >
                       0u;
            default:
                return true;
            }
        }
    };
    /// One submission of any queue.
    struct Work {
        std::span<const u32> dcb; ///< graphics: the draw buffer; compute: the ring part
        std::span<const u32> ccb; ///< the constant engine's buffer
        Walk walk;
        Walk ce;
        u32 queue = 0;
        u64 seq = NoSeq;
        bool started = false;
        bool ce_started = false;
        std::shared_ptr<SubmittedCopy> copy;
        /// The packet that waited, tried again next time (its registers are applied already).
        const u32* waiting_at = nullptr;
        WaitHint wait_hint{};
        std::chrono::steady_clock::time_point wait_start{};
        bool wait_met_at_once = false;
        /// Compute: a packet completed from the ring's end and start, and the dwords of the
        /// level it takes.
        std::vector<u32> assembled;
        u32 assembled_used = 0;
        /// Compute: DMA_DATA that patch a later DISPATCH_DIRECT's dimensions (an indirect
        /// dispatch).
        struct IndirectPatch {
            const u32* packet;
            VAddr args;
        };
        std::vector<IndirectPatch> patches;
    };
    enum class Status { Done, Wait };
    enum class Step { Next, Moved, Wait };

    struct alignas(64) Queue {
        std::mutex mutex;
        std::deque<std::unique_ptr<Work>> works;
        std::atomic<u32> pending_works{0};
        ComputeProgram cs_state{};
    };

    // Thread.
    void Run(std::stop_token stop);
    /// BB_STALL_REPORT (default on, =0 off): when the queues make no progress, or the guest
    /// submits nothing, for 5 s, prints once what each queue is at, the fences waiting for the
    /// GPU and the last labels written (a guest waiting for a label that never comes).
    void StallWatch(std::stop_token stop);
    void ReportStall(const char* why);
    void WaitForWork(std::stop_token stop);
    void RunCommands();
    void AfterBatch();
    void SignalDecodedFrames();
    void SignalGpuIdle(u64 generation);
    size_t QueueSize(u32 queue);

    // Graphics.
    Status RunGraphics(Work& work);
    void BeginGraphics(Work& work);
    void EndGraphics(Work& work);
    Step Graphics(Work& work, Level& level, const u32* packet, u32 body_words);
    Status RunConstantEngine(Work& work);
    void SetContextRegHints(const u32* body, u32 body_words);
    void Draw(Level& level, Op op, const u32* packet, u32 body_words);
    bool WaitRegMemGraphics(Work& work, const u32* body, bool retry);
    void CopyData(const u32* body);
    bool PredicationSkips(u32 header);
    void SyncForCpuRead(VAddr address, u64 size);

    // Compute.
    Status RunCompute(Work& work);
    Step Compute(Work& work, Level& level, const u32* packet, u32 body_words);
    void ReleaseMemory(const ReleaseMem& release, u32 pipe_id);

    // Fences handed to the draw recording thread: met in stream order for WAIT_REG_MEM.
    void NotePending(VAddr address, u64 value, u32 bytes, bool eop);
    void NotePendingEop(const EventWriteEop& eop);
    void NotePendingEos(const EventWriteEos& eos);
    void NotePendingWrite(VAddr address, const u32* data, u32 bytes);
    bool PendingFenceValue(VAddr address, u32& value);
    void DropReachedFences();

    Vulkan::Rasterizer* rasterizer{};
    BbVideoOut::Port* vo_port{};
    const bool guest_markers_enabled;

    std::array<Queue, NumTotalQueues> queues{};
    std::atomic<u32> num_mapped_queues{1}; ///< graphics, and compute up to the highest queue used
    int current_queue = -1;

    VAddr indirect_args_addr{};
    u32 num_counter_pairs = 8; ///< occlusion counter pairs (depth blocks) of a base PS4
    u64 pixel_counter{};
    ConstantEngine cblock;
    static std::array<u8, 48 * 1024> constant_ram;

    /// SET_PREDICATION: evaluated at the first predicated draw or dispatch after it, once the GPU
    /// has done the work before (the results are GPU-written).
    struct Predication {
        struct Test {
            u32 op;
            VAddr address;
        };
        std::vector<Test> tests; ///< ORed (continued predicates)
        bool active = false;
        bool evaluated = false;
        bool skip = false;
        bool draw_if_true = true;
        bool hint_no_wait = false;
        u64 skipped = 0;
    } predication;

    struct PendingFence {
        VAddr address;
        u64 data;
        u32 num_bytes;
        u64 position;
    };
    std::deque<PendingFence> pending_fences;

    std::jthread thread;
    std::jthread stall_watch;
    std::thread::id thread_id;
    u32 thread_tid = 0;
    std::mutex submit_mutex;
    std::condition_variable_any submit_cv;
    std::deque<Common::UniqueFunction<void>> commands;
    std::atomic<u32> num_commands{0};
    std::atomic<u32> num_submits{0};
    std::atomic<bool> submit_done{false};
    std::atomic<bool> has_pending_frame_end{false};
    std::atomic<bool> work_retired{true};
    // Under submit_mutex.
    u64 submissions_total = 0;
    u64 decoded_total = 0;
    std::deque<std::pair<u64, u64>> frame_ends; ///< (submissions, frame) per sceGnmSubmitDone
    u64 gfx_submit_seq = 0;
    Work* last_decoded = nullptr; ///< BB_TIMELINE
};

} // namespace BbCp
