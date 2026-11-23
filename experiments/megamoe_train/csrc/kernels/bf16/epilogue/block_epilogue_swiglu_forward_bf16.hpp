// Copyright (c) 2026, Lu Lu
// Modified by xixing 2026

/*
 * BF16 SwiGLU epilogue for the DeepEP MegaMoe pipeline. At entry, BF16
 * Gate/Up occupy two UB tiles: the UB handoff path waits for GMM1 Fixpipe,
 * while the GM staging path first fills them through LoadRawTile().
 * With R denoting BF16 rounding and Gate/Up clipped as configured,
 * H = SiLU_FP32(gate) * up; the weighted variant produces
 * Hp = R(H * p) before GMM2. Internal arithmetic stays FP32,
 * and dispatched BF16/FP32 route weights are prefetched from GM.
 */
#ifndef DEEPEP_BLOCK_EPILOGUE_SWIGLU_FORWARD_BF16_HPP
#define DEEPEP_BLOCK_EPILOGUE_SWIGLU_FORWARD_BF16_HPP

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "catlass/arch/arch.hpp"
#include "tla/layout.hpp"
#include "layout/mega_moe_context.hpp"
#include "epilogue/block_saved_gate_up_mxfp8.hpp"
#include <type_traits>

namespace MegaMoeBF16Impl {

static constexpr AscendC::Reg::DivSpecificMode kDivMode = {AscendC::Reg::MaskMergeMode::ZEROING, true};
static constexpr AscendC::Reg::CastTrait kCastBF16ToFP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::Reg::CastTrait kCastFP32ToBF16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <typename L0TileShape_, bool ApplyTopkWeight_, typename TopkWeightT = bfloat16_t>
class BlockEpilogueSwigluForwardBF16 {
public:
    static_assert(std::is_same_v<TopkWeightT, bfloat16_t> || std::is_same_v<TopkWeightT, float>,
                  "TopK weights must be BF16 or FP32");
    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR groupFlagListGmAddr{nullptr};
        GM_ADDR routedTopkWeightGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint32_t baseM{0};
        uint32_t baseN{0};
        // dsv4 swiglu_limit clamp; kMegaMoeSwigluLimitDisabled disables it.
        float swigluLimit{kMegaMoeSwigluLimitDisabled};
        GM_ADDR savedGateGmAddr{nullptr};
        GM_ADDR savedUpGmAddr{nullptr};
        GM_ADDR savedGateScaleGmAddr{nullptr};
        GM_ADDR savedUpScaleGmAddr{nullptr};
        GM_ADDR savedGateHighGmAddr{nullptr};
        GM_ADDR savedUpLowGmAddr{nullptr};
        GM_ADDR savedUpHighGmAddr{nullptr};
        // Activation family (kMegaMoeActivation*); betas apply to SiTU-GLU.
        uint8_t activationKind{kMegaMoeActivationSwiglu};
        float actBetaGate{kMegaMoeSituBetaGateDefault};
        float actBetaUp{kMegaMoeSituBetaUpDefault};
    };

    using Params = Arguments;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

    __aicore__ inline void Init(Params const& params)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            params_ = params;
        }
    }

    __aicore__ inline void UpdateGlobalAddr(const BlockCoord& baseOffset)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            savedBaseOffset_ = Get<0>(baseOffset);
            outputGlobal_.SetGlobalBuffer((__gm__ bfloat16_t*)params_.yGmAddr + Get<0>(baseOffset));
            // Gate/up use the same packed-row x F coordinates as GMM2 input.
            if (params_.savedGateGmAddr != nullptr) {
                savedGateGlobal_.SetGlobalBuffer((__gm__ bfloat16_t*)params_.savedGateGmAddr + Get<0>(baseOffset));
                savedUpGlobal_.SetGlobalBuffer((__gm__ bfloat16_t*)params_.savedUpGmAddr + Get<0>(baseOffset));
                // Persisted activations are consumed by a later backward.
                savedGateGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                savedUpGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            groupFlagListGmAddr_ = (__gm__ int32_t*)params_.groupFlagListGmAddr + Get<2>(baseOffset) * INT_CACHELINE;
            if constexpr (ApplyTopkWeight_) {
                routedTopkWeightGlobal_.SetGlobalBuffer((__gm__ TopkWeightT*)params_.routedTopkWeightGmAddr +
                                                        Get<3>(baseOffset));
                routedTopkWeightGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
        }
    }

    __aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape)
    {
        outputN_ = Get<1>(problemShape);
    }

    template <uint32_t SaveMode>
    __aicore__ inline void Run(const BlockShape& blockShape, const BlockCoord& blockCoord)
    {
        if constexpr (g_coreType != AscendC::AIV) {
            return;
        }

        const uint32_t singleM = Get<0>(blockShape);
        const uint32_t singleN = Get<1>(blockShape);
        if (singleM == 0 || singleN == 0) {
            return;
        }

        const uint32_t tileRow = static_cast<uint32_t>(Get<0>(blockCoord) / outputN_);
        if constexpr (SaveMode != MegaMoeBF16Training::kSaveNone) {
            SaveGateUp<SaveMode>(singleM, singleN, Get<0>(blockCoord));
        }
        if ((params_.activationKind & kMegaMoeActivationKindMask) == kMegaMoeActivationSituGlu) {
            // SiTU-GLU applies its own smooth softcap; the dsv4 hard clamp
            // never runs. The packaged high-level Tanh path (host default)
            // reuses the UB beyond the work tiles, so the MXFP8 save
            // epilogue keeps the validated register formula instead.
            if constexpr (SaveMode != MegaMoeBF16Training::kSaveMxFp8) {
                if ((params_.activationKind & kMegaMoeActivationTanhPackagedBit) != 0) {
                    ComputeSituGluTanhOpDirect(singleM, singleN, tileRow);
                } else {
                    ComputeSituGluDirect(singleM, singleN, tileRow);
                }
            } else {
                ComputeSituGluDirect(singleM, singleN, tileRow);
            }
        } else {
            ComputeSwigluDirect(singleM, singleN, tileRow);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(singleM),
                                              static_cast<uint32_t>(singleN * sizeof(bfloat16_t)), 0,
                                              static_cast<uint32_t>((outputN_ - singleN) * sizeof(bfloat16_t)), 0};
        AscendC::DataCopyPad(outputGlobal_[Get<0>(blockCoord)], l0cOutUbFirst_, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::AtomicAdd(groupFlagListGmAddr_, 1);
    }

    __aicore__ inline void LoadRawTile(GM_ADDR gate, GM_ADDR up, uint64_t expertOffset, uint16_t mSize, uint16_t nSize,
                                       int64_t offset)
    {
        AscendC::GlobalTensor<bfloat16_t> rawGate, rawUp;
        rawGate.SetGlobalBuffer((__gm__ bfloat16_t*)gate + expertOffset);
        rawUp.SetGlobalBuffer((__gm__ bfloat16_t*)up + expertOffset);
        rawGate.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        rawUp.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        AscendC::DataCopyExtParams copyParams{mSize, static_cast<uint32_t>(nSize * sizeof(bfloat16_t)),
                                              static_cast<uint32_t>((outputN_ - nSize) * sizeof(bfloat16_t)), 0, 0};
        AscendC::DataCopyPad(l0cOutUbFirst_, rawGate[offset], copyParams, {});
        AscendC::DataCopyPad(l0cOutUbSecond_, rawUp[offset], copyParams, {});
        AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();
    }

    // Debug forensics (MEGAMOE_BF16_TILE_DEBUG; see ForwardTileDebugEntries
    // in mega_moe_layout.hpp): fp32-accumulate one slot into 64 bf16 partial
    // sums staged through the SiTU Tanh scratch, then DataCopyPad them to the
    // caller's GM record (128 bytes per slot). phase 0 reads the raw Gate
    // (l0cOutUbFirst_) and Up (l0cOutUbSecond_) tiles — call between
    // WaitForCube and Run; phase 1 reads the SwiGLU result from
    // l0cOutUbFirst_ — call after Run. No extra UB: only pre-existing
    // buffers are touched.
    __aicore__ inline void EmitTileSums(__gm__ bfloat16_t* gmEntry, uint16_t mSize, uint16_t nSize, uint8_t phase)
    {
        constexpr uint16_t lanes = 256 / sizeof(float);
        const uint16_t stride = static_cast<uint16_t>(Ops::Base::CeilAlign(uint32_t(nSize), uint32_t(16)));
        auto* gate = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        auto* up = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        auto* stage = (__ubuf__ bfloat16_t*)situTanhTmp_.GetPhyAddr();
        const uint8_t slotCount = phase == 0U ? 2U : 1U;
        AscendC::GlobalTensor<bfloat16_t> gmT;
        gmT.SetGlobalBuffer(gmEntry);
        AscendC::LocalTensor<bfloat16_t> stageView{AscendC::TPosition::VECCALC, kSituTanhTmpOffset, 192};
        for (uint8_t slot = 0; slot < slotCount; ++slot) {
            auto* src = (slot == 0U || phase != 0U) ? gate : up;
            const uint16_t stageOffset = static_cast<uint16_t>(phase == 0U ? slot * 64U : 128U);
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<float> acc;
                AscendC::Reg::Duplicate(acc, 0.F);
                for (uint16_t row = 0; row < mSize; ++row) {
                    uint32_t remaining = nSize;
                    for (uint16_t col = 0; col < nSize; col += lanes) {
                        auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                        AscendC::Reg::RegTensor<bfloat16_t> v;
                        AscendC::Reg::RegTensor<float> vf;
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                            v, src + row * stride + col);
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(vf, v, mask);
                        AscendC::Reg::Add(acc, acc, vf, mask);
                    }
                }
                AscendC::Reg::RegTensor<bfloat16_t> out;
                auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(out, acc, all);
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    stage + stageOffset, out, all);
            }
            AscendC::SyncFunc<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyExtParams dump{1, 128U, 0, 0, 0};
            AscendC::DataCopyPad(gmT[stageOffset], stageView[stageOffset], dump);
            AscendC::SyncFunc<AscendC::HardEvent::MTE3_V>();
        }
    }

private:
    static constexpr uint32_t kTileRows = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t kTileColumns = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t kTileElements = kTileRows * kTileColumns;
    static constexpr uint32_t kTileBytes = kTileElements * sizeof(bfloat16_t);
    static_assert(2U * kTileElements * sizeof(bfloat16_t) <= Catlass::Arch::Ascend950::UB_SIZE,
                  "The BF16 SwiGLU input tiles must fit in A5 UB");

    template <uint32_t SaveMode>
    __aicore__ inline void SaveGateUp(uint16_t mSize, uint16_t nSize, int64_t offset)
    {
        if constexpr (SaveMode == MegaMoeBF16Training::kSaveMxFp8) {
            quantSave_.Save(l0cOutUbFirst_, l0cOutUbSecond_, params_.savedGateGmAddr, params_.savedUpGmAddr,
                            params_.savedGateScaleGmAddr, params_.savedUpScaleGmAddr, params_.savedGateHighGmAddr,
                            params_.savedUpLowGmAddr, params_.savedUpHighGmAddr, params_.swigluLimit, mSize, nSize,
                            outputN_, savedBaseOffset_ + offset);
            return;
        }
        // This BF16 save runs on the UB handoff path, with or without clipping.
        // The caller's WaitForCube has completed both Fixpipe input tiles.
        // Save raw BF16 gate/up before clipping, activation, or route weighting.
        // DataCopyPad uses the same aligned UB row stride as ComputeSwigluDirect.
        AscendC::SyncFunc<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyExtParams copyParams{mSize, static_cast<uint32_t>(nSize * sizeof(bfloat16_t)), 0,
                                              static_cast<uint32_t>((outputN_ - nSize) * sizeof(bfloat16_t)), 0};
        AscendC::DataCopyPad(savedGateGlobal_[offset], l0cOutUbFirst_, copyParams);
        AscendC::DataCopyPad(savedUpGlobal_[offset], l0cOutUbSecond_, copyParams);
        // MTE3 must finish reading both tiles before vector work overwrites UB.
        AscendC::SyncFunc<AscendC::HardEvent::MTE3_V>();
    }

    __aicore__ inline void ComputeSwigluDirect(uint16_t mSize, uint16_t nSize, uint32_t tileRow)
    {
        constexpr uint16_t lanes = 256 / sizeof(float);
        const uint32_t stride = Ops::Base::CeilAlign(uint32_t(nSize), uint32_t(16));
        const bool clip = params_.swigluLimit != kMegaMoeSwigluLimitDisabled;
        const float limit = params_.swigluLimit;
        auto* gate = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        auto* up = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        // Place weights beyond every MXFP8 data/scale/clamp-mask buffer: Save
        // may still have MTE3 readers. No alias and no early pipeline drain.
        constexpr uint32_t weightBytes = (kTileRows * sizeof(TopkWeightT) + 31) / 32 * 32;
        constexpr uint32_t weightOffset = Catlass::Arch::Ascend950::UB_SIZE - weightBytes;
        static_assert(weightOffset >= SavedGateUpMxFp8<kTileRows>::Layout::totalBytesWithClipMasks);
        AscendC::LocalTensor<TopkWeightT> weightUb{AscendC::TPosition::VECIN, weightOffset, kTileRows};
        if constexpr (ApplyTopkWeight_) {
            AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyExtParams p{1, uint32_t(mSize * sizeof(TopkWeightT)), 0, 0, 0};
            AscendC::DataCopyPad(weightUb, routedTopkWeightGlobal_[tileRow], p, {});
            AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();
        }
        auto* weights = (__ubuf__ TopkWeightT*)weightUb.GetPhyAddr();
        // Direct B16 unpack follows the reference implementation. Arithmetic,
        // accurate division and clipping remain in FP32 until the output cast.
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> lim, negLim;
            if (clip) {
                AscendC::Reg::Duplicate(lim, limit);
                AscendC::Reg::Duplicate(negLim, -limit);
            }
            for (uint16_t row = 0; row < mSize; ++row) {
                AscendC::Reg::RegTensor<float> weightFp;
                if constexpr (ApplyTopkWeight_) {
                    if constexpr (std::is_same_v<TopkWeightT, float>) {
                        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(weightFp, weights + row);
                    } else {
                        AscendC::Reg::RegTensor<bfloat16_t> weightBf;
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_BRC_B16>(weightBf,
                                                                                                 weights + row);
                        auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(weightFp, weightBf, all);
                    }
                }
                uint32_t remaining = nSize;
                for (uint16_t col = 0; col < nSize; col += lanes) {
                    auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                    AscendC::Reg::RegTensor<bfloat16_t> g, u, output;
                    AscendC::Reg::RegTensor<float> gf, uf, neg, exp, denom, silu, value;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        g, gate + row * stride + col);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        u, up + row * stride + col);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(gf, g, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(uf, u, mask);
                    if (clip) {
                        AscendC::Reg::Min(gf, gf, lim, mask);
                        AscendC::Reg::Max(uf, uf, negLim, mask);
                        AscendC::Reg::Min(uf, uf, lim, mask);
                    }
                    AscendC::Reg::Muls(neg, gf, -1.0F, mask);
                    AscendC::Reg::Exp(exp, neg, mask);
                    AscendC::Reg::Adds(denom, exp, 1.0F, mask);
                    AscendC::Reg::Div<float, &kDivMode>(silu, gf, denom, mask);
                    AscendC::Reg::Mul(value, silu, uf, mask);
                    if constexpr (ApplyTopkWeight_) {
                        AscendC::Reg::Mul(value, value, weightFp, mask);
                    }
                    // Keep H in FP32 through route weighting; only the GEMM input Hp is BF16.
                    AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(output, value, mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                        gate + row * stride + col, output, mask);
                }
            }
        }
    }

    // SiTU-GLU (Kimi K3): y = (bG*tanh(g/bG))*sigmoid(g) * (bU*tanh(u/bU)).
    // Both branches are smoothly softcapped, so there is no clamp path and
    // no clip-mask save.  Numerical convention matches this repo's SwiGLU
    // contract (design doc §1.3): the activation stays FP32 through route
    // weighting and rounds once at the GEMM input; tanh uses the exp
    // identity with the same Exp/Adds/Div primitives as ComputeSwigluDirect.
    __aicore__ inline void ComputeSituGluDirect(uint16_t mSize, uint16_t nSize, uint32_t tileRow)
    {
        constexpr uint16_t lanes = 256U / sizeof(float);
        const uint32_t stride = Ops::Base::CeilAlign(uint32_t(nSize), uint32_t(16));
        constexpr float kOne = 1.0F;
        constexpr float kTwo = 2.0F;
        constexpr float kNegOne = -1.0F;
        const float betaGate = params_.actBetaGate;
        const float betaUp = params_.actBetaUp;
        // beta divides as reciprocal multiplies (x/beta == x*(1/beta), beta
        // is a host scalar; ops-nn convention); FP32-internal only.
        const float recipBetaGate = 1.0F / betaGate;
        const float recipBetaUp = 1.0F / betaUp;
        // Reference implementation's unspecified linear_beta: beta_up == 0
        // keeps the up branch linear (B(u) = u), capping only the gate.
        const bool linearUp = betaUp == 0.0F;
        auto* gate = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        auto* up = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        constexpr uint32_t weightBytes = (kTileRows * sizeof(TopkWeightT) + 31) / 32 * 32;
        constexpr uint32_t weightOffset = Catlass::Arch::Ascend950::UB_SIZE - weightBytes;
        static_assert(weightOffset >= SavedGateUpMxFp8<kTileRows>::Layout::totalBytesWithClipMasks);
        AscendC::LocalTensor<TopkWeightT> weightUb{AscendC::TPosition::VECIN, weightOffset, kTileRows};
        if constexpr (ApplyTopkWeight_) {
            AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyExtParams p{1, uint32_t(mSize * sizeof(TopkWeightT)), 0, 0, 0};
            AscendC::DataCopyPad(weightUb, routedTopkWeightGlobal_[tileRow], p, {});
            AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();
        }
        auto* weights = (__ubuf__ TopkWeightT*)weightUb.GetPhyAddr();
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < mSize; ++row) {
                AscendC::Reg::RegTensor<float> weightFp;
                if constexpr (ApplyTopkWeight_) {
                    if constexpr (std::is_same_v<TopkWeightT, float>) {
                        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(weightFp, weights + row);
                    } else {
                        AscendC::Reg::RegTensor<bfloat16_t> weightBf;
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_BRC_B16>(weightBf,
                                                                                                 weights + row);
                        auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(weightFp, weightBf, all);
                    }
                }
                AscendC::Reg::RegTensor<float> oneReg, twoReg;
                AscendC::Reg::Duplicate(oneReg, kOne);
                AscendC::Reg::Duplicate(twoReg, kTwo);
                uint32_t remaining = nSize;
                for (uint16_t col = 0; col < nSize; col += lanes) {
                    auto mask = AscendC::Reg::UpdateMask<float>(remaining);
                    const uint32_t offset = row * stride + col;
                    AscendC::Reg::RegTensor<bfloat16_t> g, u, output;
                    AscendC::Reg::RegTensor<float> gf, uf, d, arg, ex, den, inv, t, a, sg, value;
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(g, gate + offset);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(u, up + offset);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(gf, g, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(uf, u, mask);
                    // ---- gate branch: A = betaGate * tanh(g/betaGate) * sigmoid(g), FP32
                    AscendC::Reg::Muls(d, gf, recipBetaGate, mask);
                    AscendC::Reg::Muls(arg, d, kTwo, mask);            // tanh identity
                    AscendC::Reg::Exp(ex, arg, mask);
                    AscendC::Reg::Adds(den, ex, kOne, mask);
                    AscendC::Reg::Div<float, &kDivMode>(inv, twoReg, den, mask);
                    AscendC::Reg::Sub(t, oneReg, inv, mask);
                    AscendC::Reg::Muls(a, t, betaGate, mask);
                    AscendC::Reg::Muls(arg, gf, kNegOne, mask);        // sigmoid(g)
                    AscendC::Reg::Exp(ex, arg, mask);
                    AscendC::Reg::Adds(den, ex, kOne, mask);
                    AscendC::Reg::Div<float, &kDivMode>(sg, oneReg, den, mask);
                    AscendC::Reg::Mul(value, a, sg, mask);
                    // ---- up branch: B = betaUp * tanh(u/betaUp); linearUp keeps u
                    if (linearUp) {
                        AscendC::Reg::Mul(value, value, uf, mask);
                    } else {
                        AscendC::Reg::Muls(d, uf, recipBetaUp, mask);
                        AscendC::Reg::Muls(arg, d, kTwo, mask);
                        AscendC::Reg::Exp(ex, arg, mask);
                        AscendC::Reg::Adds(den, ex, kOne, mask);
                        AscendC::Reg::Div<float, &kDivMode>(inv, twoReg, den, mask);
                        AscendC::Reg::Sub(t, oneReg, inv, mask);
                        AscendC::Reg::Muls(a, t, betaUp, mask);
                        AscendC::Reg::Mul(value, value, a, mask);
                    }
                    if constexpr (ApplyTopkWeight_) {
                        // Keep H in FP32 through route weighting; only the GEMM input Hp is BF16.
                        AscendC::Reg::Mul(value, value, weightFp, mask);
                    }
                    AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(output, value, mask);
                    AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                        gate + offset, output, mask);
                }
            }
        }
    }

    // SiTU-GLU with the packaged high-level Tanh (AscendC::Tanh, callable
    // inside __VEC_SCOPE__; on 950PR/DT the default TanhAlgo::INTRINSIC maps
    // to the hardware tanh vector instruction, which has no Reg-level API).
    // Same FP32-through convention as ComputeSituGluDirect; only the tanh
    // evaluation differs (<= 1 ulp FP32-internal). The Tanh operands live in
    // UB, so each repeat stages its rounded-div inputs through a small FP32
    // buffer with LocalMemBar handoff; the Tanh call stays at repeat level
    // (the bisheng backend rejects one nested inside the part loop).
    __aicore__ inline void ComputeSituGluTanhOpDirect(uint16_t mSize, uint16_t nSize, uint32_t tileRow)
    {
        constexpr uint16_t kElementsPerRepeat = 256U / sizeof(bfloat16_t);
        const uint16_t repeatsPerRow = static_cast<uint16_t>(
            Ops::Base::CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(kElementsPerRepeat)));
        constexpr float kOne = 1.0F;
        constexpr float kNegOne = -1.0F;
        const float betaGate = params_.actBetaGate;
        const float betaUp = params_.actBetaUp;
        // beta divides as reciprocal multiplies (x/beta == x*(1/beta), beta
        // is a host scalar; ops-nn convention); FP32-internal only.
        const float recipBetaGate = 1.0F / betaGate;
        const float recipBetaUp = 1.0F / betaUp;
        const bool linearUp = betaUp == 0.0F;
        const uint32_t situStride = Ops::Base::CeilAlign(uint32_t(nSize), uint32_t(16));
        __ubuf__ bfloat16_t* first = (__ubuf__ bfloat16_t*)l0cOutUbFirst_.GetPhyAddr();
        __ubuf__ bfloat16_t* second = (__ubuf__ bfloat16_t*)l0cOutUbSecond_.GetPhyAddr();
        __ubuf__ float* tanhIn = (__ubuf__ float*)situTanhIn_.GetPhyAddr();
        __ubuf__ float* tanhGateOut = (__ubuf__ float*)situTanhGateOut_.GetPhyAddr();
        __ubuf__ float* tanhUpOut = (__ubuf__ float*)situTanhUpOut_.GetPhyAddr();
        constexpr uint32_t weightBytes = (kTileRows * sizeof(TopkWeightT) + 31) / 32 * 32;
        constexpr uint32_t weightOffset = Catlass::Arch::Ascend950::UB_SIZE - weightBytes;
        AscendC::LocalTensor<TopkWeightT> weightUb{AscendC::TPosition::VECIN, weightOffset, kTileRows};
        if constexpr (ApplyTopkWeight_) {
            AscendC::SyncFunc<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopyExtParams p{1, uint32_t(mSize * sizeof(TopkWeightT)), 0, 0, 0};
            AscendC::DataCopyPad(weightUb, routedTopkWeightGlobal_[tileRow], p, {});
            AscendC::SyncFunc<AscendC::HardEvent::MTE2_V>();
        }
        auto* weights = (__ubuf__ TopkWeightT*)weightUb.GetPhyAddr();
        __VEC_SCOPE__
        {
            for (uint16_t row = 0; row < mSize; ++row) {
                AscendC::Reg::RegTensor<float> weightFp;
                if constexpr (ApplyTopkWeight_) {
                    if constexpr (std::is_same_v<TopkWeightT, float>) {
                        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(weightFp, weights + row);
                    } else {
                        AscendC::Reg::RegTensor<bfloat16_t> weightBf;
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_BRC_B16>(weightBf,
                                                                                                 weights + row);
                        auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(weightFp, weightBf, all);
                    }
                }
                uint32_t remaining = nSize;
                for (uint16_t repeat = 0; repeat < repeatsPerRow; ++repeat) {
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<bfloat16_t>(remaining);
                    AscendC::Reg::RegTensor<float> oneReg;
                    AscendC::Reg::Duplicate(oneReg, kOne, mask);
                    AscendC::Reg::RegTensor<bfloat16_t> gate0, gate1, up0, up1;
                    AscendC::Reg::RegTensor<float> gateFp0, gateFp1, upFp0, upFp1;
                    const uint32_t offset = row * situStride + repeat * kElementsPerRepeat;
                    AscendC::Reg::DataCopy(gate0, first + offset);
                    AscendC::Reg::DataCopy(up0, second + offset);
                    AscendC::Reg::Interleave(gate0, gate1, gate0, gate1);
                    AscendC::Reg::Interleave(up0, up1, up0, up1);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(gateFp0, gate0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(gateFp1, gate1, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(upFp0, up0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(upFp1, up1, mask);
                    constexpr uint16_t kParts = sizeof(float) / sizeof(bfloat16_t);
                    constexpr uint16_t kPartSize = kElementsPerRepeat / kParts;
                    // ---- stage 1: d1 = g / betaGate -> buffer
                    for (uint16_t part = 0; part < kParts; ++part) {
                        auto g = part == 0 ? gateFp0 : gateFp1;
                        AscendC::Reg::RegTensor<float> d;
                        AscendC::Reg::Muls(d, g, recipBetaGate, mask);
                        AscendC::Reg::DataCopy(tanhIn + part * kPartSize, d, mask);
                    }
                    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                    AscendC::Tanh(situTanhGateOut_, situTanhIn_, situTanhTmp_, kElementsPerRepeat);
                    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                    // ---- stage 2 (skipped for linearUp): e1 = u / betaUp. The
                    // gate result is already safe in its own destination buffer.
                    if (!linearUp) {
                        for (uint16_t part = 0; part < kParts; ++part) {
                            auto u = part == 0 ? upFp0 : upFp1;
                            AscendC::Reg::RegTensor<float> d;
                            AscendC::Reg::Muls(d, u, recipBetaUp, mask);
                            AscendC::Reg::DataCopy(tanhIn + part * kPartSize, d, mask);
                        }
                        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                        AscendC::Tanh(situTanhUpOut_, situTanhIn_, situTanhTmp_, kElementsPerRepeat);
                        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                    }
                    // ---- stage 3: reload t1/t2 and finish in FP32
                    for (uint16_t part = 0; part < kParts; ++part) {
                        auto g = part == 0 ? gateFp0 : gateFp1;
                        auto u = part == 0 ? upFp0 : upFp1;
                        AscendC::Reg::RegTensor<float> t, a, sg, acc, arg, ex, den;
                        AscendC::Reg::RegTensor<bfloat16_t> rb;
                        AscendC::Reg::DataCopy(t, tanhGateOut + part * kPartSize);
                        AscendC::Reg::Muls(a, t, betaGate, mask);
                        AscendC::Reg::Muls(arg, g, kNegOne, mask);             // sigmoid(g)
                        AscendC::Reg::Exp(ex, arg, mask);
                        AscendC::Reg::Adds(den, ex, kOne, mask);
                        AscendC::Reg::Div<float, &kDivMode>(sg, oneReg, den, mask);
                        AscendC::Reg::Mul(acc, a, sg, mask);                   // A = a * sg
                        if (linearUp) {
                            AscendC::Reg::Mul(acc, acc, u, mask);              // y = A * u
                        } else {
                            AscendC::Reg::DataCopy(t, tanhUpOut + part * kPartSize);
                            AscendC::Reg::Muls(a, t, betaUp, mask);            // B = bU * t2
                            AscendC::Reg::Mul(acc, acc, a, mask);              // y = A * B
                        }
                        if constexpr (ApplyTopkWeight_) {
                            AscendC::Reg::Mul(acc, acc, weightFp, mask);
                        }
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(rb, acc, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            first + offset + part * kPartSize, rb, mask);
                    }
                }
            }
        }
    }

    AscendC::LocalTensor<bfloat16_t> l0cOutUbFirst_{AscendC::TPosition::VECIN, 0, kTileElements};
    AscendC::LocalTensor<bfloat16_t> l0cOutUbSecond_{AscendC::TPosition::VECIN, kTileElements * sizeof(bfloat16_t),
                                                     kTileElements};
    // SiTU-GLU packaged-Tanh scratch (ComputeSituGluTanhOpDirect): one source
    // buffer plus separate gate/up destinations (Tanh forbids src/dst
    // overlap), each holding one repeat of FP32 elements. All offsets are
    // 32B aligned (general AscendC address-alignment constraint) and placed
    // after the two work tiles. The MXFP8 save epilogue owns the UB beyond
    // the work tiles while it runs; the packaged path never engages with
    // SaveMode == kSaveMxFp8.
    static constexpr uint32_t kSituTanhElems = 256U / sizeof(bfloat16_t);  // one repeat of FP32 elements (512B)
    static constexpr uint32_t kSituTanhInOffset = 2U * kTileBytes;
    static constexpr uint32_t kSituTanhGateOutOffset = kSituTanhInOffset + kSituTanhElems * sizeof(float);
    static constexpr uint32_t kSituTanhUpOutOffset = kSituTanhGateOutOffset + kSituTanhElems * sizeof(float);
    static constexpr uint32_t kSituTanhTmpBytes = 8U * 1024U;
    static constexpr uint32_t kSituTanhTmpOffset = kSituTanhUpOutOffset + kSituTanhElems * sizeof(float);
    static constexpr uint32_t kSituTanhEnd = kSituTanhTmpOffset + kSituTanhTmpBytes;
    static_assert(kSituTanhEnd <= Catlass::Arch::Ascend950::UB_SIZE - (kTileRows * sizeof(TopkWeightT) + 31) / 32 * 32,
                  "The SiTU-GLU packaged-Tanh scratch must stay clear of the prefetched route weights");
    AscendC::LocalTensor<float> situTanhIn_{AscendC::TPosition::VECCALC, kSituTanhInOffset, kSituTanhElems};
    AscendC::LocalTensor<float> situTanhGateOut_{AscendC::TPosition::VECCALC, kSituTanhGateOutOffset,
                                                 kSituTanhElems};
    AscendC::LocalTensor<float> situTanhUpOut_{AscendC::TPosition::VECCALC, kSituTanhUpOutOffset, kSituTanhElems};
    AscendC::LocalTensor<uint8_t> situTanhTmp_{AscendC::TPosition::VECCALC, kSituTanhTmpOffset, kSituTanhTmpBytes};
    AscendC::GlobalTensor<bfloat16_t> outputGlobal_;
    AscendC::GlobalTensor<bfloat16_t> savedGateGlobal_;
    AscendC::GlobalTensor<bfloat16_t> savedUpGlobal_;
    AscendC::GlobalTensor<TopkWeightT> routedTopkWeightGlobal_;
    __gm__ int32_t* groupFlagListGmAddr_{nullptr};
    Params params_{};
    int64_t outputN_{0};
    int64_t savedBaseOffset_{0};
    SavedGateUpMxFp8<kTileRows> quantSave_;
    static_assert(SavedGateUpMxFp8<kTileRows>::Layout::totalBytesWithClipMasks <= Catlass::Arch::Ascend950::UB_SIZE,
                  "Saved gate/up MXFP8 scratch must fit AIV0 UB");
    static_assert((kTileRows == 128 || kTileRows == 192) && kTileColumns == 256,
                  "Update saved quant UB for a new tile shape");
};

}  // namespace MegaMoeBF16Impl

#endif  // DEEPEP_BLOCK_EPILOGUE_SWIGLU_FORWARD_BF16_HPP
