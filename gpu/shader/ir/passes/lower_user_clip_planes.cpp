// SPDX-License-Identifier: GPL-2.0-or-later
// The fixed-function user clip planes (PA_CL_UCP_*, enabled by PA_CL_CLIP_CNTL's UCP_ENA bits)
// as clip distances of the vertex shader: the plane equations change per draw and come from a
// per-draw uniform buffer (its binding is reserved for every vertex shader).
#include <algorithm>
#include <array>

#include "common/logging/log.h"
#include "shader/ir/ir_emitter.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/info.h"

namespace Shader::Optimization {

namespace {
constexpr u32 NumUserClipPlanes = 6;
}

void LowerUserClipPlanes(IR::Program& program, const RuntimeInfo& runtime_info) {
    Info& info = program.info;
    if (info.hw_stage != HwStage::Vertex) {
        return;
    }
    const u32 binding = static_cast<u32>(info.buffers.size());
    info.buffers.push_back(BufferResource{
        .used_types = IR::Type::F32,
        .buffer_type = BufferType::ClipPlanes,
    });
    const auto& vs = runtime_info.hw.vs;
    const u32 enabled = vs.user_clip_plane_mask;

    // Scan existing ClipDistance exports (e.g. Blood Paint `_Pnt` VS exporting only ClipDist0..1)
    // and the last Position0 export block.
    u32 exported_clip_mask = 0u;
    std::array<IR::Value, 4> position{};
    IR::Block* target_block{};
    IR::Block::iterator target_it{};
    for (IR::Block* const block : program.blocks) {
        for (auto inst = block->begin(); inst != block->end(); ++inst) {
            if (inst->GetOpcode() == IR::Opcode::SetAttribute) {
                if (inst->Arg(0).Attribute() == IR::Attribute::ClipDistance) {
                    const u32 elem = inst->Arg(2).U32();
                    if (elem < 8u) {
                        exported_clip_mask |= (1u << elem);
                    }
                } else if (inst->Arg(0).Attribute() == IR::Attribute::Position0) {
                    position[inst->Arg(2).U32()] = inst->Arg(1);
                    target_block = block;
                    target_it = std::next(inst);
                }
            }
        }
    }

    // The hardware clips by user planes and exported distances independently; here both share
    // the distance slots and the exports own theirs. Crucially, `InjectClipDistanceAttributes`
    // in the fragment shader tests ALL `MaxEmulatedClipDistances` (4) components `< 0.0f -> Discard`,
    // so any unwritten `cldist_attr0` component MUST be initialized to `1.0f` (`>= 0.0f`) rather
    // than leaving uninitialized GPU attribute-RAM garbage that discards Blood Paint splatters!
    if (exported_clip_mask != 0u) {
        if (enabled != 0u) {
            LOG_WARNING(Render_Vulkan,
                        "Skipping user clip plane lowering: shader {:#x} exports its own clip "
                        "distances",
                        info.pgm_hash);
        }
        if ((exported_clip_mask & 0xFu) != 0xFu && !program.blocks.empty()) {
            IR::Block* const entry = program.blocks.front();
            auto it = std::ranges::find(*entry, IR::Opcode::Prologue, &IR::Inst::GetOpcode);
            if (it != entry->end()) {
                ++it;
            }
            IR::IREmitter ir{*entry, it};
            for (u32 plane = 0; plane < MaxEmulatedClipDistances; ++plane) {
                if (((exported_clip_mask >> plane) & 1u) == 0u) {
                    ir.SetAttribute(IR::Attribute::ClipDistance, ir.Imm32(1.0f), plane);
                }
            }
        }
        return;
    }

    if (enabled == 0) {
        return;
    }
    if (std::ranges::any_of(position, &IR::Value::IsEmpty)) {
        return;
    }
    IR::IREmitter ir{*target_block, target_it};
    const IR::U32 handle = ir.Imm32(binding);
    for (u32 plane = 0; plane < NumUserClipPlanes; ++plane) {
        if (((enabled >> plane) & 1) == 0) {
            if (plane < MaxEmulatedClipDistances) {
                ir.SetAttribute(IR::Attribute::ClipDistance, ir.Imm32(1.0f), plane);
            }
            continue;
        }
        // A plane is four dwords in the buffer: distance = dot(position, plane).
        const IR::Value equation = ir.LoadBufferF32(4, handle, ir.Imm32(plane * 4), {});
        IR::F32 distance{ir.FPMul(IR::F32{position[0]}, IR::F32{ir.CompositeExtract(equation, 0)})};
        for (u32 comp = 1; comp < 4; ++comp) {
            distance = IR::F32{ir.FPFma(IR::F32{position[comp]},
                                        IR::F32{ir.CompositeExtract(equation, comp)}, distance)};
        }
        ir.SetAttribute(IR::Attribute::ClipDistance, distance, plane);
    }
}

} // namespace Shader::Optimization
