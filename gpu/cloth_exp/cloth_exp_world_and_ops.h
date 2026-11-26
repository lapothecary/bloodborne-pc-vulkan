// SPDX-License-Identifier: GPL-2.0-or-later
// ============================================================================
// Bloodborne Experimental Native Cloth & Wind Engine (C++20)
// File: gpu/cloth_exp/cloth_exp_world_and_ops.h
//
// Batch 1 (14 functions, 5,729 B PS4 code — 100% 1-to-1 with eboot.bin):
//   - hclSimClothInstance::getActiveConfig       (0x0748c50,   21 B)
//   - hclSimClothInstance::updateParticlesAabb   (0x0748a00,  161 B)
//   - hclSimClothInstance::updateAabbs           (0x0748ab0,  410 B)
//   - hclConstraintStiffnessFactor               (0x1167a80,  166 B)
//   - hclMoveParticlesOperator::process          (0x06fffb0,   16 B)
//   - hclMoveParticlesOperator::moveSomeSIMD     (0x06fffc0,  439 B)
//   - hclMoveParticlesOperator::moveSomeScalar   (0x0700180,  439 B)
//   - hclWorld::step                             (0x0769450,  120 B)
//   - hclWorld::addClothJobs                     (0x07694d0,  659 B)
//   - hclWorld::stepSingleThread                 (0x0769190,  701 B)
//   - hclOperator::execute                       (0x076a960, 1436 B)
//   - FD4HavokCloth::step                        (0x0fd5440,  346 B)
//   - FD4HavokCloth::queryWind                   (0x0fd5ab0,  255 B)
//   - FD4Wind::step                              (0x0fd8120,  560 B)
// ============================================================================
#pragma once

#include "common/types.h"

namespace BbClothExp {

extern "C" {

u8* PS4_SYSV_ABI ClothExp_hclSimClothInstance_getActiveConfig(u8* sim) noexcept;
void PS4_SYSV_ABI ClothExp_hclSimClothInstance_updateParticlesAabb(u8* sim) noexcept;
void PS4_SYSV_ABI ClothExp_hclSimClothInstance_updateAabbs(u8* sim) noexcept;

float PS4_SYSV_ABI ClothExp_hclConstraintStiffnessFactor(const u8* cset, int mode, int iter,
                                                         int substeps, float factor0,
                                                         float factor1) noexcept;

void PS4_SYSV_ABI ClothExp_hclMoveParticlesOperator_process(const u8* op, u8* sim,
                                                            const u8* ref_buf) noexcept;
void PS4_SYSV_ABI ClothExp_hclMoveParticlesOperator_moveSomeSIMD(const u8* op, u8* sim,
                                                                 const u8* ref_buf) noexcept;
void PS4_SYSV_ABI ClothExp_hclMoveParticlesOperator_moveSomeScalar(const u8* op, u8* sim,
                                                                   const u8* ref_buf) noexcept;

void PS4_SYSV_ABI ClothExp_hclWorld_step(u8* world, u8* job_queue, int num_jobs, u8* scratch,
                                         float dt) noexcept;
void PS4_SYSV_ABI ClothExp_hclWorld_addClothJobs(u8* world, u8* job_queue, int num_jobs,
                                                 u8* scratch, float dt) noexcept;
void PS4_SYSV_ABI ClothExp_hclWorld_stepSingleThread(u8* world, u8* scratch, float dt) noexcept;
void PS4_SYSV_ABI ClothExp_hclOperator_execute(u8* op, u8* step_info) noexcept;

void PS4_SYSV_ABI ClothExp_FD4HavokCloth_step(u8* self, float dt) noexcept;
std::int64_t PS4_SYSV_ABI ClothExp_FD4HavokCloth_queryWind(u8* self, const void* target) noexcept;
void PS4_SYSV_ABI ClothExp_FD4Wind_step(u8* self, const u8* step_ctx) noexcept;

} // extern "C"

} // namespace BbClothExp
