// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#include "tiling.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>

#include <tiling/platform/platform_ascendc.h>

#include "csrc/kernels/bf16/common/mega_moe_tiling.hpp"
#include "tiling_impl.hpp"
#include "csrc/kernels/bf16/layout/mega_moe_layout.hpp"

namespace {

bool ResolveCompactTokenCountExchange(bool bf16DirectDispatch, int64_t worldSize, int64_t expertPerRank)
{
    if (!bf16DirectDispatch) {
        return false;
    }
    const char* rawMode = std::getenv("MEGAMOE_BF16_TOKEN_COUNT_EXCHANGE");
    const std::string mode = rawMode == nullptr || rawMode[0] == '\0' ? "auto" : rawMode;
    if (mode == "auto") {
        return MegaMoeBF16Training::UseCompactTokenCountExchangeByDefault(static_cast<uint64_t>(worldSize),
                                                                          static_cast<uint64_t>(expertPerRank));
    }
    if (mode == "dense") {
        return false;
    }
    TORCH_CHECK(mode == "compact", "MEGAMOE_BF16_TOKEN_COUNT_EXCHANGE must be auto, dense or compact; got '", mode,
                "'");
    TORCH_CHECK(MegaMoeBF16Training::CanUseCompactTokenCountExchange(static_cast<uint64_t>(worldSize),
                                                                     static_cast<uint64_t>(expertPerRank)),
                "forced compact BF16 token-count exchange requires 2..1024 EP ranks, "
                "1..8 local experts");
    return true;
}

}  // namespace

namespace deepep::megamoe_train::ops::detail {

MegaMoeHostTiling MakeMegaMoeHostTiling(MegaMoeTilingData& tiling, int64_t m, int64_t k, int64_t n, int64_t topk,
                                        int64_t world_size, int64_t expert_per_rank, int64_t max_recv_tokens, int rank,
                                        bool bf16, double swiglu_limit, bool bf16_direct_dispatch,
                                        const std::string& save_gmm1_output, uint32_t topk_weight_type,
                                        const std::string& activation, double act_beta_gate, double act_beta_up)
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t aic_num = platform->GetCoreNumAic();
    const uint32_t aiv_num = platform->GetCoreNumAiv();
    uint64_t ub_size = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    TORCH_CHECK(aic_num > 0 && aiv_num > 0 && ub_size > 0, "failed to query AscendC platform resources");
    // Ordinary BF16 forward and backward read saved/direct source offsets.
    // Direct-dispatch mode skips the legacy bookkeeping guard on world_size.
    TORCH_CHECK(!bf16_direct_dispatch || bf16, "direct Dispatch offsets require a BF16 kernel");
    TORCH_CHECK(bf16_direct_dispatch || world_size <= static_cast<int64_t>(aic_num) * 10,
                "world_size exceeds the kernel dispatch bookkeeping capacity for ", aic_num, " AIC cores");

    if (bf16_direct_dispatch) {
        // Backward consumes state from the matching ordinary forward, so
        // keep its forward-capacity checks too. Backward also checks its own
        // phase-by-phase UB footprint in MegaMoeBF16Backward().
        const MegaMoeBF16Training::ForwardDispatchUbLayout dispatchUb(world_size, expert_per_rank, k, 2);
        TORCH_CHECK(dispatchUb.totalBytes <= ub_size, "MegaMoe BF16 Dispatch requires ", dispatchUb.totalBytes,
                    " UB bytes, but the device has ", ub_size);
        TORCH_CHECK(MegaMoeBF16Training::ForwardDispatchUbLayout::kCountResetChunkBytes <= ub_size,
                    "MegaMoe BF16 count reset chunk exceeds device UB capacity");
    }

    const float normalizedSwigluLimit =
        swiglu_limit > 0.0 ? static_cast<float>(swiglu_limit) : kMegaMoeSwigluLimitDisabled;
    // Resolve the activation family once here so every caller downstream
    // (kernel epilogues, saved-state layouts) sees a validated kind.
    uint8_t activationKind = kMegaMoeActivationSwiglu;
    if (activation == "situglu") {
        activationKind = kMegaMoeActivationSituGlu;
    } else {
        TORCH_CHECK(activation == "swiglu", "activation must be swiglu or situglu, got '", activation, "'");
    }
    if (activationKind == kMegaMoeActivationSituGlu) {
        // SiTU-GLU bounds both branches with its own smooth softcap; the
        // dsv4 hard clamp is a different (mutually exclusive) stabilizer.
        TORCH_CHECK(swiglu_limit <= 0.0,
                    "swiglu_limit is incompatible with activation='situglu'; pass swiglu_limit=0");
        // beta_up == 0 is the reference implementation's unspecified
        // linear_beta: the up branch stays linear (B(u) = u).
        TORCH_CHECK(act_beta_gate > 0.0, "act_betas[0] (beta_gate) must be positive for activation='situglu'");
        TORCH_CHECK(act_beta_up >= 0.0,
                    "act_betas[1] (beta_up) must be 0 (linear up branch) or positive for activation='situglu'");
    }
    // SiTU-GLU tanh selection (forward AND backward epilogues): the default
    // is the register exp-identity formula; the packaged high-level Tanh
    // (hardware INTRINSIC algorithm on 950PR/DT, <= 1 ulp FP32-internal
    // difference) is OPT-IN via MEGAMOE_ACT_TANH=1/true/on/yes. On the source
    // repo the packaged path cut the K3-class fwd+bwd step by ~18% after a
    // green gate; enable it here only after this repo's own device A/B.
    // The saved-MXFP8 combination always runs the formula path (kernel-side
    // guard: the quant epilogue owns the UB beyond the work tiles).
    if (activationKind == kMegaMoeActivationSituGlu) {
        const char* tanhMode = std::getenv("MEGAMOE_ACT_TANH");
        if (tanhMode != nullptr && tanhMode[0] != '\0') {
            const std::string mode(tanhMode);
            if (mode == "1" || mode == "true" || mode == "on" || mode == "yes") {
                activationKind |= kMegaMoeActivationTanhPackagedBit;
            } else {
                TORCH_CHECK(mode == "0" || mode == "false" || mode == "off",
                            "MEGAMOE_ACT_TANH must be 1/true/on/yes or 0/false/off, got '", mode, "'");
            }
        }
    }
    // SiTU-GLU backward full-tile fused epilogue (docs/mega_moe_situ_fulltile_design.md):
    // a single-pass register loop over the resident Gate/Up/dHp work tiles
    // replacing the stepwise aux -> store Hp/dp -> reload Gate/Up -> grad
    // structure on every full 128x256 tile. OPT-IN via MEGAMOE_SITU_FUSED;
    // the default keeps the validated stepwise path. The packaged-Tanh
    // guard is kernel-side (the fused packaged variant never engages when
    // savedGateUpMode is kSaveMxFp8, mirroring the stepwise dispatchers).
    if (activationKind == kMegaMoeActivationSituGlu) {
        const char* fusedMode = std::getenv("MEGAMOE_SITU_FUSED");
        if (fusedMode != nullptr && fusedMode[0] != '\0') {
            const std::string mode(fusedMode);
            if (mode == "1" || mode == "true" || mode == "on" || mode == "yes") {
                activationKind |= kMegaMoeActivationSituFusedBit;
            } else {
                TORCH_CHECK(mode == "0" || mode == "false" || mode == "off",
                            "MEGAMOE_SITU_FUSED must be 1/true/on/yes or 0/false/off, got '", mode, "'");
            }
        }
    }
    const float actBetaGate = static_cast<float>(act_beta_gate);
    const float actBetaUp = static_cast<float>(act_beta_up);
    using namespace MegaMoeBF16Training;
    uint32_t saveMode = kSaveAuto;
    if (save_gmm1_output == "none")
        saveMode = kSaveNone;
    else if (save_gmm1_output == "bf16")
        saveMode = kSaveBf16;
    else if (save_gmm1_output == "mxfp8")
        saveMode = kSaveMxFp8;
    else
        TORCH_CHECK(save_gmm1_output == "auto", "save_gmm1_output must be auto, none, bf16 or mxfp8");
    TORCH_CHECK(bf16 || saveMode == kSaveAuto, "save_gmm1_output is a BF16-only option");
    saveMode = ResolveSavedGateUpMode(saveMode, normalizedSwigluLimit);
    if (bf16 && saveMode == kSaveMxFp8) {
        const uint64_t quantUbBytes = SaveGateUpClipMasks(normalizedSwigluLimit, saveMode)
                                          ? SavedGateUpQuantUbLayout::totalBytesWithClipMasks
                                          : SavedGateUpQuantUbLayout::totalBytes;
        const uint64_t ubBudget = std::min<uint64_t>(ub_size, 248ULL * 1024ULL);
        TORCH_CHECK(quantUbBytes <= ubBudget, "MegaMoe BF16 saved MXFP8 epilogue requires ", quantUbBytes,
                    " UB bytes, have ", ubBudget);
        TORCH_CHECK((n / 2) % 32 == 0, "saved MXFP8 requires F aligned to 32");
    }

    const uint32_t blocks_per_ep = std::max<uint32_t>(1, aic_num / world_size);
    bool ok = false;
    ok = optiling::SetInitTilingData<uint16_t, DISPATCH_QUANT_MODE_NO_QUANT>(
        tiling, static_cast<uint32_t>(m), static_cast<uint32_t>(k), static_cast<uint32_t>(n),
        static_cast<uint32_t>(topk), static_cast<uint32_t>(world_size), static_cast<uint32_t>(expert_per_rank),
        blocks_per_ep, static_cast<uint32_t>(max_recv_tokens), static_cast<uint32_t>(rank), 2, 1, aiv_num, ub_size);
    TORCH_CHECK(ok, "failed to generate MegaMoe BF16 tiling data");
    tiling.swigluLimit = normalizedSwigluLimit;
    tiling.savedGateUpMode = saveMode;
    // SetInitTilingData resets the struct defaults, so the activation fields
    // must be written after it (the local activationKind carries the
    // packaged-Tanh bit decision).
    tiling.activationKind = activationKind;
    tiling.actBetaGate = actBetaGate;
    tiling.actBetaUp = actBetaUp;
    TORCH_CHECK(topk_weight_type <= kTopkWeightFp32, "unsupported TopK weight type");
    tiling.topkWeightType = topk_weight_type;
    tiling.compactTokenCountExchange =
        ResolveCompactTokenCountExchange(bf16_direct_dispatch, world_size, expert_per_rank) ? 1U : 0U;
    const auto workspace_info = optiling::CalcWorkspaceSize(tiling);
    const auto peer_info = optiling::CalcPeermemSize(tiling);
    uint64_t peerBytes = static_cast<uint64_t>(peer_info.totalSize);
    if (bf16_direct_dispatch) {
        // CalcPeermemSize retains the 60 KiB legacy header used by the
        // non-BF16 and experimental variants. Ordinary BF16 uses its 72 KiB
        // synchronization header and, at large EP, an aligned compact-count
        // inbox before the unchanged dense token-count table.
        const MegaMoeBF16Training::CompactTokenCountExchangeLayout countLayout(static_cast<uint64_t>(world_size),
                                                                               static_cast<uint64_t>(expert_per_rank),
                                                                               tiling.compactTokenCountExchange != 0);
        peerBytes += countLayout.tokenCountsOffset - MegaMoeBF16Training::kPeerBaseDataOffset;
    }
    return {static_cast<uint64_t>(workspace_info.totalSize), peerBytes};
}

MegaMoeHostTiling GetBf16EarlyTopkWeightBufferBytes(const MegaMoeTilingData& tiling)
{
    return {static_cast<uint64_t>(optiling::CalcBf16EarlyTopkWeightWorkspaceSize(tiling)),
            static_cast<uint64_t>(optiling::CalcBf16EarlyTopkWeightPeermemSize(tiling))};
}

}  // namespace deepep::megamoe_train::ops::detail
