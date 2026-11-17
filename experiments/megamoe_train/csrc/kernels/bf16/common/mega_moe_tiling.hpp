// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026

/*!
 * \file mega_moe_tiling.hpp
 * \brief
 */

#ifndef ASCENDC_MEGA_MOE_TILING
#define ASCENDC_MEGA_MOE_TILING

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "routing/moe_init_routing_v3/arch35/moe_init_routing_v3_arch35_tiling_def.h"

using namespace AscendC;

// P1 GMM2-grad writes one FP32 partial per [packed row, 256-wide F tile]
// instead of materializing the full FP32 dHp*H matrix; P6 performs the small
// final reduction.  Defined here (host-safe) so the kernel workspace layout
// (layout/mega_moe_workspace_view.hpp) and the host-side workspace layout
// (csrc/bf16/training.cpp) agree on the same per-tile stride.
constexpr uint32_t BACKWARD_DP_PARTIAL_TILE_N = 256U;

// SwiGLU activation clamp limit (dsv4 swiglu_limit): gate = min(gate, L),
// up = clamp(up, -L, L). FLT_MAX disables the clamp.
static constexpr float kMegaMoeSwigluLimitDisabled = 3.402823466e+38F;

// Activation family for the BF16 training pipeline. SwiGLU keeps the dsv4
// swiglu_limit clamp; SiTU-GLU (Kimi K3) applies a smooth softcap
// softcap(x, b) = b * tanh(x / b) to both branches instead and is
// incompatible with the hard clamp (the host rejects the combination).
constexpr uint8_t kMegaMoeActivationSwiglu = 0;
constexpr uint8_t kMegaMoeActivationSituGlu = 1;
// Bit 0 selects the activation family. Higher bits are epilogue variant
// knobs: the host ORs kMegaMoeActivationTanhPackagedBit in by default to
// swap the SiTU-GLU tanh from the register exp identity to the packaged
// high-level Tanh, and kMegaMoeActivationSituFusedBit (opt-in via
// MEGAMOE_SITU_FUSED) to run the SiTU-GLU backward full-tile epilogue as
// the single-pass fused variant instead of the stepwise aux -> store ->
// reload -> grad structure. Family compares must mask with
// kMegaMoeActivationKindMask; the epilogues strip the extra bits when
// comparing the family.
constexpr uint8_t kMegaMoeActivationKindMask = 0x01;
constexpr uint8_t kMegaMoeActivationTanhPackagedBit = 0x02;
constexpr uint8_t kMegaMoeActivationSituFusedBit = 0x04;
// Kimi K3 defaults: beta_gate = 4, beta_up = 25.
constexpr float kMegaMoeSituBetaGateDefault = 4.0F;
constexpr float kMegaMoeSituBetaUpDefault = 25.0F;

struct MegaMoeTilingData {
    MoeInitRoutingV3Arch35TilingData moeInitRoutingTilingData;
    uint32_t expertPerRank;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t epWorldSize;
    uint32_t blockNumPerEP;  // if blockNumPerEP < dispatchRows, then dispatchRows is invalid
    uint32_t maxOutputSize;
    uint32_t rankId;
    uint32_t topK;
    int32_t dispatchRows;
    uint8_t groupListType;
    bool transX;
    bool transW;
    bool transW2;
    // dsv4 swiglu_limit clamp; kMegaMoeSwigluLimitDisabled disables it.
    float swigluLimit;
    // Activation family (kMegaMoeActivation*): 0 = SwiGLU, 1 = SiTU-GLU.
    // Betas only apply to SiTU-GLU; paths that never assign them keep the
    // SwiGLU default through the member initializers below.
    uint8_t activationKind{kMegaMoeActivationSwiglu};
    float actBetaGate{kMegaMoeSituBetaGateDefault};
    float actBetaUp{kMegaMoeSituBetaUpDefault};
    // BF16 only: 0=auto, 1=none, 2=BF16, 3=MXFP8. Host resolves auto.
    uint32_t savedGateUpMode{2};
    // Ordinary BF16 only: Host resolves auto/diagnostic overrides and writes
    // the effective per-launch token-count exchange protocol here.
    // 0=dense, 1=compact. Experimental variants always keep this at zero.
    uint32_t compactTokenCountExchange{0};
    // TopK weights and their gradients: 0=BF16, 1=FP32. GMM operands stay BF16.
    uint32_t topkWeightType{0};
};
#endif
