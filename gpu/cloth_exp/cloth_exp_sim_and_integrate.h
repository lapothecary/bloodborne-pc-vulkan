// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// File: gpu/cloth_exp/cloth_exp_sim_and_integrate.h
//
// Batch 3 (6 functions, 12,555 B PS4 code — 100% 1-to-1 with eboot.bin):
//   - hclSimulateOperator::preStep                 (0x076fb90, 1717 B)
//   - hclSimulateOperator::simulate                (0x076e370, 3276 B)
//   - hclSimulateOperator::transferMotion          (0x0771290, 2054 B)
//   - hclSimulateOperator::integrate               (0x0773df0, 1130 B)
//   - hclSimulateOperator::pinchAndRelease         (0x076dcc0, 1496 B)
//   - hclSimulateOperator::updateParticleNormals   (0x076f040, 2882 B)
// ============================================================================
#pragma once

#include "common/types.h"

namespace BbClothExp {

extern "C" {

void PS4_SYSV_ABI ClothExp_hclSimulateOperator_preStep(u8* op, u8* step_info) noexcept;

int PS4_SYSV_ABI ClothExp_hclSimulateOperator_simulate(u8* op, u8* ctx, float dt) noexcept;

void PS4_SYSV_ABI ClothExp_hclSimulateOperator_transferMotion(const u8* op, u8* sim,
                                                              const void* old_xform,
                                                              const void* new_xform,
                                                              float dt) noexcept;

void PS4_SYSV_ABI ClothExp_hclSimulateOperator_integrate(const u8* op, u8* ctx, u8* sim,
                                                         float substep_dt) noexcept;

void PS4_SYSV_ABI ClothExp_hclSimulateOperator_pinchAndRelease(const u8* op, u8* sim,
                                                               u8* planes_buf) noexcept;

void PS4_SYSV_ABI ClothExp_hclSimulateOperator_updateParticleNormals(const u8* op,
                                                                     u8* sim) noexcept;

} // extern "C"

} // namespace BbClothExp
