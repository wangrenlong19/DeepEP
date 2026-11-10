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
    // no clip-mask save. tanh(x) = 1 - 2/(1 + exp(2x)) keeps the same
    // Exp/Adds/Div primitives as the SwiGLU path.
    // Numerical convention: the eager reference graph materializes every
    // elementwise op as BF16, so each R() point below rounds the FP32
    // register to BF16 and back. Skipping them passes the loose forward
    // gate but fails the backward precision gate. The route weight is
    // applied exactly like the epilogue's BF16 save order: y = R(A*B) is
    // materialized first, then Hp = R(y * p) is the final store cast.
    __aicore__ inline void ComputeSituGluDirect(uint16_t mSize, uint16_t nSize, uint32_t tileRow)
    {
        constexpr uint16_t kElementsPerRepeat = 256U / sizeof(bfloat16_t);
        const uint16_t repeatsPerRow = static_cast<uint16_t>(
            Ops::Base::CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(kElementsPerRepeat)));
        constexpr float kOne = 1.0F;
        constexpr float kTwo = 2.0F;
        constexpr float kNegOne = -1.0F;
        const float betaGate = params_.actBetaGate;
        const float betaUp = params_.actBetaUp;
        // Reference implementation's unspecified linear_beta: beta_up == 0
        // keeps the up branch linear (B(u) = u), capping only the gate.
        const bool linearUp = betaUp == 0.0F;
        const uint32_t situStride = Ops::Base::CeilAlign(uint32_t(nSize), uint32_t(16));
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
                uint32_t remaining = nSize;
                for (uint16_t repeat = 0; repeat < repeatsPerRow; ++repeat) {
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<bfloat16_t>(remaining);
                    AscendC::Reg::RegTensor<float> oneReg, twoReg;
                    AscendC::Reg::RegTensor<float> betaGateReg, betaUpReg;
                    AscendC::Reg::Duplicate(oneReg, kOne, mask);
                    AscendC::Reg::Duplicate(twoReg, kTwo, mask);
                    AscendC::Reg::Muls(betaGateReg, oneReg, betaGate, mask);
                    AscendC::Reg::Muls(betaUpReg, oneReg, betaUp, mask);
                    AscendC::Reg::RegTensor<bfloat16_t> gate0, gate1, up0, up1;
                    AscendC::Reg::RegTensor<float> gateFp0, gateFp1, upFp0, upFp1;
                    const uint32_t offset = row * situStride + repeat * kElementsPerRepeat;
                    AscendC::Reg::DataCopy(gate0, gate + offset);
                    AscendC::Reg::DataCopy(up0, up + offset);
                    AscendC::Reg::Interleave(gate0, gate1, gate0, gate1);
                    AscendC::Reg::Interleave(up0, up1, up0, up1);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(gateFp0, gate0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(gateFp1, gate1, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(upFp0, up0, mask);
                    AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(upFp1, up1, mask);
                    constexpr uint16_t kParts = sizeof(float) / sizeof(bfloat16_t);
                    constexpr uint16_t kPartSize = kElementsPerRepeat / kParts;
                    for (uint16_t part = 0; part < kParts; ++part) {
                        auto g = part == 0 ? gateFp0 : gateFp1;
                        auto u = part == 0 ? upFp0 : upFp1;
                        AscendC::Reg::RegTensor<float> d, t, a, sg, acc, arg, ex, den, inv;
                        AscendC::Reg::RegTensor<float> neg, b;
                        AscendC::Reg::RegTensor<bfloat16_t> r;
                        // ---- gate branch: a = R(R(bG*R(tanh(R(g/bG)))) * R(sigmoid(g)))
                        AscendC::Reg::Div<float, &kDivMode>(d, g, betaGateReg, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(r, d, mask);
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(d, r, mask);
                        AscendC::Reg::Muls(arg, d, kTwo, mask);            // tanh identity
                        AscendC::Reg::Exp(ex, arg, mask);
                        AscendC::Reg::Adds(den, ex, kOne, mask);
                        AscendC::Reg::Div<float, &kDivMode>(inv, twoReg, den, mask);
                        AscendC::Reg::Sub(t, oneReg, inv, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(r, t, mask);
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(t, r, mask);
                        AscendC::Reg::Muls(a, t, betaGate, mask);          // eager: beta * tanh
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(r, a, mask);
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(a, r, mask);
                        AscendC::Reg::Muls(neg, g, kNegOne, mask);         // sigmoid(g)
                        AscendC::Reg::Exp(ex, neg, mask);
                        AscendC::Reg::Adds(den, ex, kOne, mask);
                        AscendC::Reg::Div<float, &kDivMode>(sg, oneReg, den, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(r, sg, mask);
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(sg, r, mask);
                        AscendC::Reg::Mul(acc, a, sg, mask);               // (beta*tanh) * sigmoid
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(r, acc, mask);
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(acc, r, mask);
                        // ---- up branch: b = R(bU*R(tanh(R(u/bU))));
                        //      linearUp skips the cap entirely: y = R(A*u)
                        if (linearUp) {
                            AscendC::Reg::Mul(acc, acc, u, mask);
                        } else {
                            AscendC::Reg::Div<float, &kDivMode>(d, u, betaUpReg, mask);
                            AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(r, d, mask);
                            AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(d, r, mask);
                            AscendC::Reg::Muls(arg, d, kTwo, mask);
                            AscendC::Reg::Exp(ex, arg, mask);
                            AscendC::Reg::Adds(den, ex, kOne, mask);
                            AscendC::Reg::Div<float, &kDivMode>(inv, twoReg, den, mask);
                            AscendC::Reg::Sub(t, oneReg, inv, mask);
                            AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(r, t, mask);
                            AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(t, r, mask);
                            AscendC::Reg::Muls(b, t, betaUp, mask);
                            AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(r, b, mask);
                            AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(b, r, mask);
                            // ---- y = R(A * B)
                            AscendC::Reg::Mul(acc, acc, b, mask);
                        }
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(r, acc, mask);
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(acc, r, mask);
                        if constexpr (ApplyTopkWeight_) {
                            // Same order as the BF16 save epilogue: weight the
                            // materialized activation, not its FP32 product.
                            AscendC::Reg::Mul(acc, acc, weightFp, mask);
                        }
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(r, acc, mask);
                        AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                            gate + offset + part * kPartSize, r, mask);
                    }
                }
            }
        }
    }

    // SiTU-GLU with the packaged high-level Tanh (AscendC::Tanh, callable
    // inside __VEC_SCOPE__; on 950PR/DT the default TanhAlgo::INTRINSIC maps
    // to the hardware tanh vector instruction, which has no Reg-level API,
    // so the register chain above expands tanh as 1 - 2/(1 + exp(2x))).
    // Identical body to ComputeSituGluDirect except each tanh identity block
    // is replaced by: R-round the input in registers, store it to a small
    // FP32 UB buffer, call the packaged Tanh (src/dst/tmp must not overlap
    // and must satisfy the general 32B alignment), reload and R-round the
    // result. The per-op BF16 rounding chain is preserved point-for-point;
    // the only numerical difference is the FP32-internal tanh (<= 1 ulp).
    // Per repeat the gate input goes through Tanh first, then the up input
    // reuses the same source buffer with its own destination so t1/t2 both
    // stay live for the final loop.
    __aicore__ inline void ComputeSituGluTanhOpDirect(uint16_t mSize, uint16_t nSize, uint32_t tileRow)
    {
        constexpr uint16_t kElementsPerRepeat = 256U / sizeof(bfloat16_t);
        const uint16_t repeatsPerRow = static_cast<uint16_t>(
            Ops::Base::CeilDiv(static_cast<uint64_t>(nSize), static_cast<uint64_t>(kElementsPerRepeat)));
        constexpr float kOne = 1.0F;
        constexpr float kNegOne = -1.0F;
        const float betaGate = params_.actBetaGate;
        const float betaUp = params_.actBetaUp;
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
                    AscendC::Reg::RegTensor<float> oneReg, betaGateReg, betaUpReg;
                    AscendC::Reg::Duplicate(oneReg, kOne, mask);
                    AscendC::Reg::Muls(betaGateReg, oneReg, betaGate, mask);
                    AscendC::Reg::Muls(betaUpReg, oneReg, betaUp, mask);
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
                    // ---- gate tanh input: d1 = R(g / beta_gate) -> buffer
                    for (uint16_t part = 0; part < kParts; ++part) {
                        auto g = part == 0 ? gateFp0 : gateFp1;
                        AscendC::Reg::RegTensor<float> d;
                        AscendC::Reg::RegTensor<bfloat16_t> rb;
                        AscendC::Reg::Div<float, &kDivMode>(d, g, betaGateReg, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(rb, d, mask);
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(d, rb, mask);
                        AscendC::Reg::DataCopy(tanhIn + part * kPartSize, d, mask);
                    }
                    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                    AscendC::Tanh(situTanhGateOut_, situTanhIn_, situTanhTmp_, kElementsPerRepeat);
                    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                    // ---- up tanh input (skipped for linearUp): the gate
                    // result is already safe in its own destination buffer.
                    if (!linearUp) {
                        for (uint16_t part = 0; part < kParts; ++part) {
                            auto u = part == 0 ? upFp0 : upFp1;
                            AscendC::Reg::RegTensor<float> d;
                            AscendC::Reg::RegTensor<bfloat16_t> rb;
                            AscendC::Reg::Div<float, &kDivMode>(d, u, betaUpReg, mask);
                            AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(rb, d, mask);
                            AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(d, rb, mask);
                            AscendC::Reg::DataCopy(tanhIn + part * kPartSize, d, mask);
                        }
                        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                        AscendC::Tanh(situTanhUpOut_, situTanhIn_, situTanhTmp_, kElementsPerRepeat);
                        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
                    }
                    // ---- rest of the chain, identical to ComputeSituGluDirect:
                    // t1 = R(tanh(d1)); a1 = R(bG*t1); sg = R(sigmoid(g));
                    // A = R(a1*sg); B = R(bU*t2) or linear u; y = R(A*B).
                    for (uint16_t part = 0; part < kParts; ++part) {
                        auto g = part == 0 ? gateFp0 : gateFp1;
                        auto u = part == 0 ? upFp0 : upFp1;
                        AscendC::Reg::RegTensor<float> t, t2, a, sg, acc, b;
                        AscendC::Reg::RegTensor<float> neg, ex, den;
                        AscendC::Reg::RegTensor<bfloat16_t> rb;
                        AscendC::Reg::DataCopy(t, tanhGateOut + part * kPartSize);
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(rb, t, mask);   // t1 = R(tanh(d1))
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(t, rb, mask);
                        AscendC::Reg::Muls(a, t, betaGate, mask);                              // a1 = bG*t1
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(rb, a, mask);
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(a, rb, mask);
                        AscendC::Reg::Muls(neg, g, kNegOne, mask);                             // sigmoid(g)
                        AscendC::Reg::Exp(ex, neg, mask);
                        AscendC::Reg::Adds(den, ex, kOne, mask);
                        AscendC::Reg::Div<float, &kDivMode>(sg, oneReg, den, mask);
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(rb, sg, mask);
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(sg, rb, mask);
                        AscendC::Reg::Mul(acc, a, sg, mask);                                   // A = a1 * sg
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(rb, acc, mask);
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(acc, rb, mask);
                        if (linearUp) {
                            AscendC::Reg::Mul(acc, acc, u, mask);                              // y = A * u
                        } else {
                            AscendC::Reg::DataCopy(t2, tanhUpOut + part * kPartSize);
                            AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(rb, t2, mask);  // t2 = R(tanh(e1))
                            AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(t2, rb, mask);
                            AscendC::Reg::Muls(b, t2, betaUp, mask);                           // B = bU*t2
                            AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(rb, b, mask);
                            AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(b, rb, mask);
                            AscendC::Reg::Mul(acc, acc, b, mask);                              // y = A * B
                        }
                        AscendC::Reg::Cast<bfloat16_t, float, kCastFP32ToBF16>(rb, acc, mask);
                        AscendC::Reg::Cast<float, bfloat16_t, kCastBF16ToFP32>(acc, rb, mask);
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
    static constexpr uint32_t kSituTanhEnd = kSituTanhUpOutOffset + kSituTanhElems * sizeof(float) + kSituTanhTmpBytes;
    static_assert(kSituTanhEnd <= Catlass::Arch::Ascend950::UB_SIZE - (kTileRows * sizeof(TopkWeightT) + 31) / 32 * 32,
                  "The SiTU-GLU packaged-Tanh scratch must stay clear of the prefetched route weights");
    AscendC::LocalTensor<float> situTanhIn_{AscendC::TPosition::VECCALC, kSituTanhInOffset, kSituTanhElems};
    AscendC::LocalTensor<float> situTanhGateOut_{AscendC::TPosition::VECCALC, kSituTanhGateOutOffset,
                                                 kSituTanhElems};
    AscendC::LocalTensor<float> situTanhUpOut_{AscendC::TPosition::VECCALC, kSituTanhUpOutOffset, kSituTanhElems};
    AscendC::LocalTensor<uint8_t> situTanhTmp_{AscendC::TPosition::VECCALC,
                                               kSituTanhUpOutOffset + kSituTanhElems * sizeof(float),
                                               kSituTanhTmpBytes};
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
