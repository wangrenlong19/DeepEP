// Copyright (c) 2026, Lu Lu
// Modified by wangrenlong 2026

#pragma once
#include "kernel_operator.h"

/*
 * SiTU-GLU (Kimi K3) math primitives shared by the BF16 forward and backward
 * epilogues. The upstream twopath formula in situglu_activation.hpp is
 * intentionally NOT reused: the register exp/Div decomposition measured
 * slower than the packaged high-level Tanh, which on 3510 maps to the
 * hardware tanh vector instruction (TanhAlgo::INTRINSIC) and has no
 * Reg-level API. The tanh legs therefore stage their divided inputs through
 * small FP32 UB buffers with LocalMemBar handoffs; the packaged call must
 * stay at repeat level because the bisheng backend rejects one nested
 * inside the part loop.
 *
 * Numerical convention (matching the situglu golden references):
 *   softcap(x, b) = b * tanh(x / b)
 *   A(g) = softcap(g, betaGate) * sigmoid(g)
 *   B(u) = softcap(u, betaUp), or the linear u when betaUp == 0
 *   A'(g) = (1 - t1^2) * sig + betaGate * t1 * sig * (1 - sig)
 * All internal arithmetic stays FP32; beta divides as reciprocal multiplies
 * (ops-nn convention) and sigmoid keeps the accurate division.
 *
 * Signature convention: every RegTensor/MaskReg parameter is a plain
 * non-const reference, exactly like swiglu_math.hpp -- the Reg-level
 * AscendC APIs bind non-const references only, so a const-ref helper
 * parameter fails overload resolution inside (see the Muls match error).
 */

namespace MegaMoeImpl::Activation {

// Accurate division for every SiTU-GLU sigmoid, matching the SwiGLU forward
// and backward division flavors so mixed-activation plans keep one taste.
static constexpr AscendC::Reg::DivSpecificMode kSituDivMode = {AscendC::Reg::MaskMergeMode::ZEROING, true};

// One repeat of packaged-tanh staging: 256B / BF16 = 128 elements, split into
// two FP32 register parts by the callers' Interleave+Cast pattern.
constexpr uint16_t kSituTanhRepeatElements = 256U / sizeof(bfloat16_t);
constexpr uint16_t kSituTanhParts = 2U;
constexpr uint16_t kSituTanhPartSize = kSituTanhRepeatElements / kSituTanhParts;
// 3x512B operand buffers + 8KB packaged-Tanh scratch (old-branch sizing).
constexpr uint32_t kSituTanhOperandBytes = kSituTanhRepeatElements * sizeof(float);
constexpr uint32_t kSituTanhTmpBytes = 8U * 1024U;

struct SituParams {
    float betaGate;
    float betaUp;
    float recipBetaGate;
    float recipBetaUp;
    // betaUp == 0 sentinel: B(u) = u with no softcap.
    bool linearUp;
};

__aicore__ inline SituParams MakeSituParams(float betaGate, float betaUp)
{
    SituParams p;
    p.betaGate = betaGate;
    p.betaUp = betaUp;
    p.recipBetaGate = 1.0F / betaGate;
    p.recipBetaUp = betaUp != 0.0F ? 1.0F / betaUp : 0.0F;
    p.linearUp = betaUp == 0.0F;
    return p;
}

// sigmoid(g) = 1 / (1 + e^-g) with the accurate division. The saturating
// vector Exp mirrors the swiglu epilogues (large negative g saturates to 0).
__simd_callee__ inline void SituSigmoidFp32(AscendC::Reg::RegTensor<float>& sigOut,
                                            AscendC::Reg::RegTensor<float>& gFp,
                                            AscendC::Reg::RegTensor<float>& oneReg,
                                            AscendC::Reg::RegTensor<float>& neg, AscendC::Reg::RegTensor<float>& ex,
                                            AscendC::Reg::RegTensor<float>& den, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::Muls(neg, gFp, -1.0F, mask);
    AscendC::Reg::Exp(ex, neg, mask);
    AscendC::Reg::Adds(den, ex, 1.0F, mask);
    AscendC::Reg::Div<float, &kSituDivMode>(sigOut, oneReg, den, mask);
}

// A(g) = betaGate * t1 * sigmoid(g); t1 comes from the hardware tanh leg.
__simd_callee__ inline void SituGateActFp32(
    AscendC::Reg::RegTensor<float>& aOut, AscendC::Reg::RegTensor<float>& t1, AscendC::Reg::RegTensor<float>& gFp,
    AscendC::Reg::RegTensor<float>& oneReg, AscendC::Reg::RegTensor<float>& neg, AscendC::Reg::RegTensor<float>& ex,
    AscendC::Reg::RegTensor<float>& den, AscendC::Reg::RegTensor<float>& sig, AscendC::Reg::MaskReg& mask,
    float betaGate)
{
    SituSigmoidFp32(sig, gFp, oneReg, neg, ex, den, mask);
    AscendC::Reg::Muls(aOut, t1, betaGate, mask);
    AscendC::Reg::Mul(aOut, aOut, sig, mask);
}

// B(u) = betaUp * t2 (linear up branches fold u at their call sites).
__simd_callee__ inline void SituUpActFp32(AscendC::Reg::RegTensor<float>& bOut, AscendC::Reg::RegTensor<float>& t2,
                                          AscendC::Reg::MaskReg& mask, float betaUp)
{
    AscendC::Reg::Muls(bOut, t2, betaUp, mask);
}

// Gate derivative: dGate = da * ((1 - t1^2) * sig + betaGate * t1 * sig * (1 - sig)).
// da carries dH * B(u) (or dH * u on the linear branch); the sigmoid of g is
// computed here. dGateOut doubles as scratch until the final multiply.
__simd_callee__ inline void SituGateGradFp32(
    AscendC::Reg::RegTensor<float>& dGateOut, AscendC::Reg::RegTensor<float>& da, AscendC::Reg::RegTensor<float>& t1,
    AscendC::Reg::RegTensor<float>& gFp, AscendC::Reg::RegTensor<float>& oneReg, AscendC::Reg::RegTensor<float>& neg,
    AscendC::Reg::RegTensor<float>& ex, AscendC::Reg::RegTensor<float>& den, AscendC::Reg::RegTensor<float>& sig,
    AscendC::Reg::RegTensor<float>& inner, AscendC::Reg::RegTensor<float>& term, AscendC::Reg::MaskReg& mask,
    float betaGate)
{
    SituSigmoidFp32(sig, gFp, oneReg, neg, ex, den, mask);
    AscendC::Reg::Mul(inner, t1, t1, mask);
    AscendC::Reg::Sub(inner, oneReg, inner, mask);
    AscendC::Reg::Mul(inner, inner, sig, mask);  // (1 - t1^2) * sg
    AscendC::Reg::Sub(term, oneReg, sig, mask);  // 1 - sg
    AscendC::Reg::Muls(dGateOut, t1, betaGate, mask);
    AscendC::Reg::Mul(dGateOut, dGateOut, sig, mask);
    AscendC::Reg::Mul(dGateOut, dGateOut, term, mask);  // betaGate * t1 * sg * (1 - sg)
    AscendC::Reg::Add(inner, inner, dGateOut, mask);
    AscendC::Reg::Mul(dGateOut, da, inner, mask);
}

// Up-branch derivative: dUp = dA * B * (1 - t2^2), where dA = dH * B's gate
// counterpart. aVal is A(g) (or the folded per-element value on the linear
// branch callers already multiplied in); tmp doubles as scratch.
__simd_callee__ inline void SituUpGradFp32(AscendC::Reg::RegTensor<float>& dUpOut,
                                           AscendC::Reg::RegTensor<float>& dA, AscendC::Reg::RegTensor<float>& t2,
                                           AscendC::Reg::RegTensor<float>& aVal, AscendC::Reg::RegTensor<float>& tmp,
                                           AscendC::Reg::RegTensor<float>& oneReg, AscendC::Reg::MaskReg& mask,
                                           float betaUp)
{
    AscendC::Reg::Muls(tmp, t2, betaUp, mask);  // B = betaUp * t2
    AscendC::Reg::Mul(dUpOut, dA, tmp, mask);
    AscendC::Reg::Mul(dUpOut, dUpOut, aVal, mask);
    AscendC::Reg::Mul(tmp, t2, t2, mask);
    AscendC::Reg::Sub(tmp, oneReg, tmp, mask);  // 1 - t2^2
    AscendC::Reg::Mul(dUpOut, dUpOut, tmp, mask);
}

// One tanh leg: stage d = v * recipBeta for both FP32 parts into the shared
// UB source buffer, run the packaged hardware tanh once at repeat level, and
// hand results back through the destination buffer. Reloads read outPtr with
// kSituTanhPartSize strides at the caller's stage-3 loop.
__aicore__ inline void SituStageTanhLeg(AscendC::LocalTensor<float>& inUb, AscendC::LocalTensor<float>& outUb,
                                        AscendC::LocalTensor<uint8_t>& tmpUb, __ubuf__ float* inPtr,
                                        __ubuf__ float* outPtr, AscendC::Reg::RegTensor<float>& part0,
                                        AscendC::Reg::RegTensor<float>& part1, float recipBeta,
                                        AscendC::Reg::MaskReg& mask)
{
    for (uint16_t part = 0; part < kSituTanhParts; ++part) {
        auto v = part == 0 ? part0 : part1;
        AscendC::Reg::RegTensor<float> d;
        AscendC::Reg::Muls(d, v, recipBeta, mask);
        AscendC::Reg::DataCopy(inPtr + part * kSituTanhPartSize, d, mask);
    }
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    AscendC::Tanh(outUb, inUb, tmpUb, kSituTanhRepeatElements);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    (void)outPtr;
}

}  // namespace MegaMoeImpl::Activation
