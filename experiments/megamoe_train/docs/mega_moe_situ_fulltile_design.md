# SiTU-GLU 反向 Epilogue 单遍融合方案

## 背景与测量

Kimi K3 类几何（k=3584, n=6144, m=8192, topk=16, EP=8, recompute 开启）下，MegaMoe BF16 训练算子在 SiTU-GLU 激活时明显慢于分立 fused 算子路径，且对 n 的增长呈超敏感；SwiGLU 激活则不受影响。8 卡单机实测（fsdp_turbo `bench_ep_dispatch.py`，每数据点 10 步平均，bf16）：

| 几何 | 激活 | fused 基线 | mega | ratio |
|---|---|---|---|---|
| k=5120, n=4608 | situglu (Tanh) | 130.2ms | 119.4ms | 0.916 |
| k=5120, n=4608 | situglu (exp) | 130.2ms | 127.2ms | 0.977 |
| k=5120, n=4608 | swiglu | 127.3ms | **107.6ms** | **0.846** |
| k=3584, n=6144 | situglu (Tanh) | 115.6ms | 136.8ms | **1.184** |
| k=3584, n=6144 | situglu (exp) | 115.7ms | 165.3ms | **1.429** |
| k=3584, n=6144 | swiglu | 111.6ms | **97.8ms** | **0.876** |

两次几何的通信字节 ∝ k（−30%）、GMM FLOPs ∝ k·n（−6.7%）、激活面 ∝ 2n（+33%）。SwiGLU 下 mega 绝对耗时随几何变化正常下降（107.6→97.8ms），说明通信与 GMM 部分行为正常；SiTU 下的全部超额开销来自激活 epilogue。以 mega-swiglu 为基底分离，SiTU 分步 epilogue 的净增量为 11.8ms（2n=9216）→ 39.0ms（2n=12288），激活面仅 +33% 而增量 3.3 倍，呈超线性——与"每个 tile 都按尾块分步方式工作"的结构性放大一致。`MEGAMOE_ACT_TANH=1`（硬件 Tanh intrinsic）只能收回其中一部分（步时 6.1% → 17.2% 随 n 增长），在 2n=12288 时 SiTU epilogue 仍占 mega 步时约 40%，说明瓶颈是分步路径的访存与流水结构，而非超越函数本身。

## 现状数据流

前向 epilogue（`block_epilogue_swiglu_forward_bf16.hpp`）的 SwiGLU 与 SiTU 均为 Direct 单遍：tile 经 L0C Fixpipe 已在 UB，寄存器按 chunk 单遍算完激活并写 Hp，无 GM 重载，两种激活结构对等，不需要改动。

问题集中在反向 epilogue（`block_epilogue_swiglu_backward_bf16.hpp`）。满 tile 的 SwiGLU 走 `ComputeFusedFullTile`（:1199）：dHp/Gate/Up 三个 128×256 工作 tile 常驻 UB（各 64KiB），全部数学在向量寄存器内按 64 元素 chunk 单遍完成——h 重构、dp 部分和、Hp、dUp、dGate 一次算完，Gate/Up 只从 GM 读一次，流水相位切换最少。SiTU 则在 tile 派发处被排除在该路径之外（:176-183 的 `situActivation` 守卫），落入如下分步结构：

1. `ComputeForwardAux(Situ|SituTanhOp)`（:457/:546/:637）：重算 h、dp 部分和、Hp，写 Hp 与 dp 到 GM；
2. V→MTE3→MTE2 全流水排空后，`LoadGateUpTiles` **第二次从 GM 读取同一份 Gate/Up**（:197-200）；
3. `ComputeSituGluGrad(TanhOp)`（:774-776）：重读 dHp，**重新计算一遍 tanh/sigmoid**，得 dGate/dUp，写 dZ 到 GM。

即每个满 tile 相比融合路径多出：一次 Gate/Up 的 GM 读（128KiB）、一次超越函数全量的重复计算、四次流水相位切换（aux 计算后 Hp/dp 的 MTE3 排空、MTE3→MTE2 交接、grad 后两次 retire）。dims 全为 128/256 整数倍时（本 bench 的两种几何均是）SwiGLU 永不落入分步路径，SiTU 则每个 tile 都落入——这就是两种激活对 n 增长"不同步"的全部来源。

代码注释（:180-182）给出的分步保留理由是"per-op BF16 chain 与 packaged-Tanh 变体在该形态下完成验证，fused path 是 SwiGLU-only 优化"——属于验证覆盖问题，不是结构性约束。下述方案补齐 SiTU 的融合变体并完成同等验证。

## 方案：`ComputeFusedFullTileSitu`

新增与 `ComputeFusedFullTile` 同构的寄存器单遍函数，供反向满 tile SiTU 使用。核心思路：dHp/Gate/Up 三个工作 tile 的 UB 驻留与 SwiGLU 融合路径完全相同；区别仅在寄存器 chunk 循环内把 SwiGLU 公式替换为 SiTU softcap 组合，并在同一遍内同时产出 aux 输出（dp、Hp）与梯度输出（dUp、dGate）。

### 单遍寄存器数据流

按行（128 行）×64 元素 chunk 循环，对每个 chunk：

1. 从 UB 工作 tile 读 gate/up/dHp 三个 BF16 chunk，Cast 到 FP32（`kBackwardCastBF16ToFP32`）；
2. **h 重构**：逐操作复刻 `ComputeForwardAuxSitu` 的 FP32 算子顺序——gate 分支 `A = βg·tanh(g/βg)·σ(g)`（tanh 走 exp 恒等式：`tanh(x)=1−2/(1+exp(2x))`，Div 用 `kBackwardDivMode`），up 分支 `B = u`（`linearUp`）或 `B = βu·tanh(u/βu)`；
3. **dp 与 Hp**：`dp += ReduceSum(dHp·h)`（FP32 行部分和，累加方式与现两函数一致）；`Hp = R(h·p)` 单次舍入到 BF16；
4. **梯度**（复用本 chunk 已算出的中间量，不重做超越函数）：`dH = dHp·p`（FP32）；`dUp = dH·A·B'(u)`，其中 `linearUp` 时 `B'=1`，否则 `B'(u)=1−tanh²(u/βu)`（复用 up 分支已得的 tanh 值）；`dGate = dH·B·A'(g)`，其中 `A'(g) = tanh'(g/βg)·σ(g) + βg·tanh(g/βg)·σ(g)·(1−σ(g))`，tanh、σ 均复用 gate 分支已得值，σ' 由 `σ·(1−σ)` 现算。**逐操作顺序必须复刻 `ComputeSituGluGrad` 的现有 FP32 边界**（dS 保持 FP32、不落地 BF16 dS 等），仅把两次遍历合并；
5. dUp/dGate/Hp 三个 BF16 chunk 写回 UB 工作 tile 原位（与 `ComputeFusedFullTile` 的写回模式一致）。

相比现分步结构，每 tile 的收益：Gate/Up 的 GM 读减半；超越函数（Exp/Div）全量只算一遍（现 aux+grad 各算一遍）；七相位流水收敛为加载→单遍计算→存储三段。

### Tanh 双模式

与现有代码同构，用模板参数选择：

- **exp 恒等式模式**（默认，三种 saved 模式均可用）：纯寄存器实现，算子顺序与 `ComputeForwardAuxSitu`/fwd `ComputeSituGluDirect` 的公式路径一致，无额外 UB。数值上这就是现有 validated 公式路径本身。
- **packaged-Tanh 模式**（`MEGAMOE_ACT_TANH=1` 时且 `savedGateUpMode != mxfp8`，守卫与现状一致）：硬件 Tanh 指令没有 Reg 级 API，操作数必须经 UB。现有 `ComputeForwardAuxSituTanhOp` 已给出可复制的模式——**repeat 级**小暂存（`kSituTanhElems=128` 个 FP32，三缓冲 + tmp 合计约 4KiB，声明见 :1656-1658）配 `LocalMemBar` 交接，每 repeat 把圆整后的 tanh 输入写入小缓冲、调 `AscendC::Tanh`、读回。融合循环内按相同方式每 64/128 元素 chunk 交接即可，不需要整 tile 的 FP32 平面。

### UB 预算

三个 128×256 BF16 工作 tile（192KiB）+ repeat 级 Tanh 暂存（约 4KiB）+ packedWeight 与 fusedDp 部分和（沿用现有 `kWeightOffset`/`fusedDpUb_` 布局）——与 SwiGLU 融合路径占用相同，现有 `static_assert`（:1326 等）直接沿用。SiTU 无 `swiglu_limit`（tiling 侧已互斥校验），`UseSavedClipMask` 恒为 false，不需要 clip-mask 相关 UB。

### 调度接线与 saved 模式

- tile 派发守卫（:177）改为激活无关：满 tile 时按 `activationKind` 分派 `ComputeFusedFullTile`（swiglu）或 `ComputeFusedFullTileSitu`（situ）；分步路径保留，仅服务两种激活的真实尾块（M/N 非满或行距非满对齐），现状不变。
- 门控：host 侧 tiling 新增一位（如 `activationKind` 的独立 bit 或独立字段 `situFusedEpilogue`），默认关闭，灰度打开；回退即关开关。
- `LoadGateUpTiles` 的三种来源（save=none 读 recompute GM、saved 读前向持久 GM、mxfp8 解码进 UB）均输出相同布局的 UB 工作 tile，融合路径原样可用；mxfp8 模式下 packaged-Tanh 守卫降级为 exp 模式，与 fwd 的 `SaveMode != kSaveMxFp8` 互斥逻辑对齐。
- 前向 epilogue 不改动。

## 验证计划

按 `mega_moe_bf16_design.md` 的验证约定执行：

1. **数值对拍（逐位）**：同一输入与 seed 下，fused SiTU 与分步 SiTU 的四类输出（dp 部分和、Hp、dGate/dUp）要求逐位一致——两者复刻同一 FP32 算子顺序，理论上应逐位相同；若 Div/Exp 的指令选择导致差异，则按现有惯例给出 FP32 内部 ulp 上界并单独说明。
2. **梯度正确性**：对照参考实现覆盖四类梯度（dX/dW1/dW2/路由权重 p）。
3. **形态矩阵**：三种 saved 模式（none/bf16/mxfp8）、Tanh 开关两态、空专家、单/多 M tile、偏斜路由、连续多轮调用；mxfp8+Tanh 组合确认降级路径。
4. **性能 A/B**：fsdp_turbo bench 复跑本文档背景节的六组数据点（两种激活 × 两种几何 × Tanh 两态），外加 mega 自身 Tanh on/off 对比确认收益保持。
5. **回退演练**：开关关闭时与现状逐位一致（默认关闭合入，绿测后灰度）。

## 预期收益

以 run2（K3 类几何，2n=12288）为基准：分步 epilogue 净增量（相对 mega-swiglu 基底）为 Tanh 模式 39.0ms、exp 模式 67.5ms。融合单遍消除重载 GM 读、双份超越函数与四段流水排空，增量预期收敛到"纯激活计算"量级（参照 SwiGLU 融合路径的实测，epilogue 近乎免费），即 mega+situ 步时预期从 136.8ms 降到 ~100-110ms（对 fused 基线 115.6ms 的 ratio 从 1.18 升至 ~0.9）；run1（2n=9216）同理由 119.4ms 降到 ~109ms。最终以设备 A/B 为准（设计文档既有约定：性能收益不能由设计推断，须单独实测报告）。


## 实现状态

已按本方案实现，与设计的差异点：

- 门控走 `activationKind` 的新 bit `kMegaMoeActivationSituFusedBit`（0x04，`mega_moe_tiling.hpp`），由环境变量 `MEGAMOE_SITU_FUSED` opt-in（解析与错误处理镜像 `MEGAMOE_ACT_TANH`，默认关闭），不用独立 tiling 字段；位随 `tilingData->activationKind` 既有通路到达反向 epilogue，前向与家族比较均掩码处理、对新位透明。
- `ComputeFusedFullTileSitu<UseTanhPackaged>` 的两种 Tanh 模式分别镜像其分步基线的循环形态，以保证逐位一致：exp 模式用 64-lane 顺序 chunk（对齐 `ComputeForwardAuxSitu` + `ComputeSituGluGrad` 的列序归约）；packaged 模式用 128-BF16 repeat + Interleave 双 part 结构（对齐 `ComputeForwardAuxSituTanhOp` + `ComputeSituGluGradTanhOp` 的 stage 划分与 part 求和顺序），Tanh 高层调用与 LocalMemBar 交接原样保留。
- 派发位于 `SwiGLUGrad` 的满 tile 判定处：满 tile 且 situ 且开启 fused bit 时按 TanhPackaged 位与 savedGateUpMode 选择模板实参（packaged 在 savedGateUpMode == kSaveMxFp8 时降级 exp，与分步守卫一致）；分步路径仅服务两种激活的真实尾块，未改动。
- 存储尾（dp 部分和、Hp、dZ 两半的 GM 写回与 MTE2/MTE3 排空）复用 `RunFusedFullTile` 的结构，新增 `RunFusedFullTileSitu<UseTanhPackaged>`。
- UB 预算与既有静态断言不变：三个 128×256 BF16 工作 tile + repeat 级 Tanh 暂存（约 4KiB，位于 MXFP8 解码区底部）+ packedWeight/fusedDp；SiTU 无 limit，无 clip-mask UB。

验收仍按上文验证计划执行；`MEGAMOE_SITU_FUSED=1` 与关闭两种状态下先做 fused vs 分步的逐位对拍，再跑性能 A/B。
