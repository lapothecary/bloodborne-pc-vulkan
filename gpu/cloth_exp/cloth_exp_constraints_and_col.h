// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// File: gpu/cloth_exp/cloth_exp_constraints_and_col.h
//
// Batch 2 (16 functions, 8,668 B PS4 code — 100% 1-to-1 with eboot.bin):
//   - hclCollidable::setTransform                     (0x0767cf0,  541 B)
//   - hclSimulateOperator::solveConstraints           (0x0771f60,  537 B)
//   - hclSimulateOperator::solveConstraintsExec       (0x0772180,  620 B)
//   - hclSimulateOperator::collideParticles           (0x0774260, 1261 B)
//   - hclSimClothInstance::collideParticlesInternal   (0x0774750,  968 B)
//   - hclLocalRangeConstraintSet::apply               (0x06fb220, 1547 B)
//   - hclStretchLinkConstraintSet::apply              (0x071ac50,  270 B)
//   - hclStretchLinkConstraintSetMx::apply            (0x0720dc0,  224 B)
//   - hclStandardLinkConstraintSet::apply             (0x07422b0,  329 B)
//   - hclStandardLinkConstraintSetMx::apply           (0x07420c0,  224 B)
//   - hclBendStiffnessConstraintSet::apply            (0x07386e0,  471 B)
//   - hclBendStiffnessConstraintSetMx::apply          (0x0741f20,  284 B)
//   - hclBendLinkConstraintSet::apply                 (0x07464d0,  415 B)
//   - hclBendLinkConstraintSetMx::apply               (0x072b2d0,  224 B)
//   - hclCompressibleLinkConstraintSet::apply         (0x07308c0,  364 B)
//   - hclVolumeConstraint::apply                      (0x076c7e0,  389 B)
// ============================================================================
#pragma once

#include "common/types.h"

namespace BbClothExp {

extern "C" {

void PS4_SYSV_ABI ClothExp_hclCollidable_setTransform(u8* col, const float* new_mat,
                                                      const float* old_mat, int normalize_quats,
                                                      float dt) noexcept;

void PS4_SYSV_ABI ClothExp_hclSimulateOperator_solveConstraints(const u8* op, u8* ctx, u8* sim,
                                                                int substep_idx,
                                                                float substep_dt) noexcept;

void PS4_SYSV_ABI ClothExp_hclSimulateOperator_solveConstraintsExec(const u8* op, u8* ctx, u8* sim,
                                                                    int substep_idx,
                                                                    float substep_dt) noexcept;

void PS4_SYSV_ABI ClothExp_hclSimulateOperator_collideParticles(const u8* op, u8* ctx, u8* sim,
                                                                float substep_dt) noexcept;

void PS4_SYSV_ABI ClothExp_hclSimClothInstance_collideParticlesInternal(u8* sim, int do_pinch,
                                                                        int do_non_pinch,
                                                                        int record_contacts,
                                                                        float substep_dt) noexcept;

void PS4_SYSV_ABI ClothExp_hclLocalRangeConstraintSet_apply(const u8* cset, u8* sim, u8* ctx,
                                                            int cset_idx, int use_factor,
                                                            float factor) noexcept;

void PS4_SYSV_ABI ClothExp_hclStretchLinkConstraintSet_apply(const u8* cset, u8* sim, u8* ctx,
                                                             int cset_idx, int use_factor,
                                                             float factor) noexcept;

void PS4_SYSV_ABI ClothExp_hclStretchLinkConstraintSetMx_apply(const u8* cset, u8* sim, u8* ctx,
                                                               int cset_idx, int use_factor,
                                                               float factor) noexcept;

void PS4_SYSV_ABI ClothExp_hclStandardLinkConstraintSet_apply(const u8* cset, u8* sim, u8* ctx,
                                                              int cset_idx, int use_factor,
                                                              float factor) noexcept;

void PS4_SYSV_ABI ClothExp_hclStandardLinkConstraintSetMx_apply(const u8* cset, u8* sim, u8* ctx,
                                                                int cset_idx, int use_factor,
                                                                float factor) noexcept;

void PS4_SYSV_ABI ClothExp_hclBendStiffnessConstraintSet_apply(const u8* cset, u8* sim, u8* ctx,
                                                               int cset_idx, int use_factor,
                                                               float factor) noexcept;

void PS4_SYSV_ABI ClothExp_hclBendStiffnessConstraintSetMx_apply(const u8* cset, u8* sim, u8* ctx,
                                                                 int cset_idx, int use_factor,
                                                                 float factor) noexcept;

void PS4_SYSV_ABI ClothExp_hclBendLinkConstraintSet_apply(const u8* cset, u8* sim, u8* ctx,
                                                          int cset_idx, int use_factor,
                                                          float factor) noexcept;

void PS4_SYSV_ABI ClothExp_hclBendLinkConstraintSetMx_apply(const u8* cset, u8* sim, u8* ctx,
                                                            int cset_idx, int use_factor,
                                                            float factor) noexcept;

void PS4_SYSV_ABI ClothExp_hclCompressibleLinkConstraintSet_apply(const u8* cset, u8* sim, u8* ctx,
                                                                  int cset_idx, int use_factor,
                                                                  float factor) noexcept;

void PS4_SYSV_ABI ClothExp_hclVolumeConstraint_apply(const u8* cset, u8* sim, u8* ctx, int cset_idx,
                                                     int use_factor, float factor) noexcept;

} // extern "C"

} // namespace BbClothExp
