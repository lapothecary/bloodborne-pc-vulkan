#pragma once

#include <cstddef>
#include <cstdint>

namespace gx_csm {

// ============================================================================
// Sealed C++20 Layouts for FromSoftware Dantelion GX 4-Cascade Directional CSM
// Extracted from PS4 Bloodborne eboot.bin (CUSA03173 v1.09):
//   - sub_0eac210 (VMA 0x0eac210): CsmManager::ConfigureFromShadowParams
//   - sub_0eac540 (VMA 0x0eac540): CsmManager::SliceCameraForCascade
//   - sub_0eac680 (VMA 0x0eac680): CsmManager::RecordCascadeRenderPass
//   - sub_0eaca30 (VMA 0x0eaca30): CsmManager::BindForScreenSpaceShadowEval
//   - sub_0eacb30 (VMA 0x0eacb30): CsmManager::ExecuteQueueCommand (ops 0..5)
//   - sub_0eb0d90 (VMA 0x0eb0d90): CsmCascadeProjector::PrepareMatricesAndClear
//   - sub_0eb2840 (VMA 0x0eb2840): CsmCascadeProjector::BuildBiasedShadowMatrix
//   - sub_0eb2d70 (VMA 0x0eb2d70): CsmCascadeProjector::ExportCascadeEvalData
//   - sub_22f0600 (VMA 0x22f0600): SceneConstantBuffer::ComposeInvViewProjWithCsm
// ============================================================================

/// Camera parameter block passed to `sub_229cb70`, `sub_eac540`, and `sub_2291c30` (0x84 bytes).
struct alignas(16) GXCsmCameraParams {
    float    camToWorld[4][4]; // +0x00..0x3f: Row0=Right, Row1=Up, Row2=Forward, Row3=EyeWorldPos
    float    upOrTarget[4];    // +0x40..0x4f
    float    fovY;             // +0x50
    float    nearZ;            // +0x54 (overwritten per cascade by sub_eac540)
    float    farZ;             // +0x58 (overwritten per cascade by sub_eac540)
    uint32_t viewportWidth;    // +0x5c
    uint32_t viewportHeight;   // +0x60
    float    aspectOverride;   // +0x64
    uint32_t pad68;            // +0x68
    float    orthoScale;       // +0x6c
    uint8_t  extra70[0x10];    // +0x70..0x7f
    uint32_t flags80;          // +0x80
    uint8_t  pad84[0x0c];      // +0x84..0x8f (stack alignment pad)
};
static_assert(offsetof(GXCsmCameraParams, camToWorld) == 0x00);
static_assert(offsetof(GXCsmCameraParams, fovY) == 0x50);
static_assert(offsetof(GXCsmCameraParams, nearZ) == 0x54);
static_assert(offsetof(GXCsmCameraParams, farZ) == 0x58);
static_assert(offsetof(GXCsmCameraParams, viewportWidth) == 0x5c);
static_assert(offsetof(GXCsmCameraParams, flags80) == 0x80);

/// Per-cascade projector / shadow-map state (`[pCsmMgr + 0x130 + cascadeIdx * 8]`, VMA 0xeb0d90).
struct GXCsmCascadeProjector {
    void*    vtbl;              // +0x00
    uint32_t projectionMode;    // +0x08 (0 = fixed box sub_eb1030, non-zero = frustum-fit sub_eb1ef0)
    uint32_t flags0c;           // +0x0c
    uint8_t  cfgBytes10[0x0c];  // +0x10..0x1b
    float    normalOffsetScale; // +0x1c (multiplies -lightDir in sub_eb2840)
    float    depthBiasOffset;   // +0x20 (added to clip Z in sub_eb2840)
    uint32_t pad24;             // +0x24
    float    orthoHalfExtent;   // +0x28
    float    frustumFitParam;   // +0x2c
    float    shadowStrength;    // +0x30
    uint8_t  cfgBytes34[0x05];  // +0x34..0x38
    uint8_t  stabilizeSnap;     // +0x39 (texel stabilization flag passed as r9d to sub_eb1ef0)
    uint8_t  pad3a[0x02];       // +0x3a..0x3b
    float    filterParam0;      // +0x3c
    float    filterParam1;      // +0x40
    float    filterParam2;      // +0x44
    float    pcfFilterRadius;   // +0x48
    uint8_t  cfgBytes4c[0x08];  // +0x4c..0x53
    uint32_t shadowColorRgba8;  // +0x54
    uint8_t  pad58[0x08];       // +0x58..0x5f
    float    lightView[4][4];   // +0x60..0x9f (World -> Light View, written in sub_eb0d90)
    float    lightProj[4][4];   // +0xa0..0xdf (Light View -> Light Clip, written in sub_eb0d90)
    float    lightDir[4];       // +0xe0..0xef
    uint32_t atlasWidth;        // +0xf0 (4096)
    uint32_t atlasHeight;       // +0xf4 (4096)
    float    scaleF8;           // +0xf8
    uint32_t padFc;             // +0xfc
    void*    pDepthTexture;     // +0x100 (GXDepthStencilResource*, 0x219f700 -> DSV)
    uint8_t  unk108[0x10];      // +0x108..0x117
    uint8_t  clearFullAtlas;    // +0x118 (set to `cascadeIdx == 0` at 0xeacbe5)
    uint8_t  resolutionShift;   // +0x119 (1 for 2x2 quadrant -> 4096 >> 1 = 2048)
    uint8_t  pad11a[0x06];      // +0x11a..0x11f
};
static_assert(offsetof(GXCsmCascadeProjector, projectionMode) == 0x08);
static_assert(offsetof(GXCsmCascadeProjector, normalOffsetScale) == 0x1c);
static_assert(offsetof(GXCsmCascadeProjector, shadowColorRgba8) == 0x54);
static_assert(offsetof(GXCsmCascadeProjector, lightView) == 0x60);
static_assert(offsetof(GXCsmCascadeProjector, lightProj) == 0xa0);
static_assert(offsetof(GXCsmCascadeProjector, lightDir) == 0xe0);
static_assert(offsetof(GXCsmCascadeProjector, atlasWidth) == 0xf0);
static_assert(offsetof(GXCsmCascadeProjector, pDepthTexture) == 0x100);
static_assert(offsetof(GXCsmCascadeProjector, clearFullAtlas) == 0x118);
static_assert(offsetof(GXCsmCascadeProjector, resolutionShift) == 0x119);

/// Per-cascade configuration entry inside `GXCsmManagerLayout` (`0x50` bytes each at `+0x550`).
struct GXCsmCascadeConfig {
    uint8_t  bytes[0x24];       // +0x00..0x23
    float    cullRadius;        // +0x24 (+0x574 in CsmManager)
    uint32_t pad28;             // +0x28
    float    slopeScaleDepthBias; // +0x2c (+0x57c in CsmManager)
    uint8_t  casterMaskFlag;    // +0x30 (+0x580 in CsmManager)
    uint8_t  rest[0x1b];        // +0x31..0x4b
    uint32_t shadowColorRgba8;  // +0x4c (+0x59c in CsmManager)
};
static_assert(sizeof(GXCsmCascadeConfig) == 0x50);
static_assert(offsetof(GXCsmCascadeConfig, cullRadius) == 0x24);
static_assert(offsetof(GXCsmCascadeConfig, slopeScaleDepthBias) == 0x2c);
static_assert(offsetof(GXCsmCascadeConfig, casterMaskFlag) == 0x30);
static_assert(offsetof(GXCsmCascadeConfig, shadowColorRgba8) == 0x4c);

/// Embedded 4-Cascade CSM Manager at `viewBytes + 0x1320` (`0x868` bytes, VMA 0xeac210..0xeacb30).
/// Note: `worldToAtlasMatrices[0..3]` at `+0x750..+0x84f` are pure WORLD-SPACE -> 2x2 Atlas UV+Z
/// matrices; `sub_22f0600` (VMA 0x22f1450) multiplies them by the current frame's `InvCameraViewProj`
/// every frame, so retaining `worldToAtlasMatrices[2..3]` across frames produces zero camera-rotation
/// shadow swimming.
struct GXCsmManagerLayout {
    uint8_t                baseHeader[0x130];          // +0x000..0x12f
    GXCsmCascadeProjector* projectors[4];              // +0x130..0x14f
    alignas(16) uint8_t    cullVolumes[4][0x100];      // +0x150..0x54f
    GXCsmCascadeConfig     cascadeConfigs[4];          // +0x550..0x68f
    void*                  pCsmParamCBuffer;           // +0x690
    void*                  cascadeRasterizerStates[4]; // +0x698..0x6b7
    uint8_t                unk6b8[0x08];               // +0x6b8..0x6bf
    float                  cachedLightDir[4];          // +0x6c0..0x6cf
    float                  splitNear;                  // +0x6d0
    float                  splitFarOffset;             // +0x6d4
    float                  fadeDistance;               // +0x6d8
    float                  split0;                     // +0x6dc (Cascade 0 -> 1 split Z)
    float                  split1;                     // +0x6e0 (Cascade 1 -> 2 split Z)
    float                  split2;                     // +0x6e4 (Cascade 2 -> 3 split Z)
    uint8_t                pad6e8[0x08];               // +0x6e8..0x6ef
    float                  fadeAndTexelParams[4];      // +0x6f0..0x6ff
    float                  packedSplits[4];            // +0x700..0x70f (split0, split1, split2, 0)
    float                  shadowColorUnpacked[4];     // +0x710..0x71f
    float                  filterParamsA[4];           // +0x720..0x72f
    float                  invAtlasAndRadius[4];       // +0x730..0x73f
    float                  cascadeTransitionRatios[4]; // +0x740..0x74f
    float                  worldToAtlasMatrices[4][16];// +0x750..0x84f (World -> Quadrant UV+Z)
    void*                  pBoundDepthSrv;             // +0x850
    uint8_t                unk858[0x08];               // +0x858..0x85f
    uint32_t               atlasWidth;                 // +0x860 (default 4096)
    uint32_t               atlasHeight;                // +0x864 (default 4096)
};
static_assert(offsetof(GXCsmManagerLayout, projectors) == 0x130);
static_assert(offsetof(GXCsmManagerLayout, cullVolumes) == 0x150);
static_assert(offsetof(GXCsmManagerLayout, cascadeConfigs) == 0x550);
static_assert(offsetof(GXCsmManagerLayout, pCsmParamCBuffer) == 0x690);
static_assert(offsetof(GXCsmManagerLayout, cascadeRasterizerStates) == 0x698);
static_assert(offsetof(GXCsmManagerLayout, cachedLightDir) == 0x6c0);
static_assert(offsetof(GXCsmManagerLayout, splitNear) == 0x6d0);
static_assert(offsetof(GXCsmManagerLayout, split0) == 0x6dc);
static_assert(offsetof(GXCsmManagerLayout, split1) == 0x6e0);
static_assert(offsetof(GXCsmManagerLayout, split2) == 0x6e4);
static_assert(offsetof(GXCsmManagerLayout, fadeAndTexelParams) == 0x6f0);
static_assert(offsetof(GXCsmManagerLayout, packedSplits) == 0x700);
static_assert(offsetof(GXCsmManagerLayout, worldToAtlasMatrices) == 0x750);
static_assert(offsetof(GXCsmManagerLayout, pBoundDepthSrv) == 0x850);
static_assert(offsetof(GXCsmManagerLayout, atlasWidth) == 0x860);
static_assert(offsetof(GXCsmManagerLayout, atlasHeight) == 0x864);
static_assert(sizeof(GXCsmManagerLayout) == 0x870);

} // namespace gx_csm

extern "C" {
// Cross-TU hooks (Linux ELF + Windows COFF64 compatible) pairing a CSM DepthRenderTarget
// (`dsvBase + 0x10`) with the 4-bit quadrant clear mask (`bit c = Cascade c`) for the next clear.
void bb_gx_set_csm_quadrant_clear(const void* depth_target, uint32_t quadrant_mask) noexcept;
uint32_t bb_gx_consume_csm_quadrant_clear(const void* depth_target) noexcept;
}
