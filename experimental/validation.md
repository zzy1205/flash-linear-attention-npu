# experimental 验证记录

当前基线为 main `d51ab687`，当前真实实验算子只有 RWKV8（来源 PR #446）。本文件按验证基线记录结论，不累计不同版本的通过数。历史完整命令和日志索引保留在 Git 历史；运行时的 JSON/日志放在 CI artifact。

## 2026-11-10 主干同步（main `d51ab687`）

同步主干新增的真实 ELF 导出校验；两个实验测试夹具复用主干的 `write_elf_exports`，不修改生产校验规则或 RWKV8 实现。仍保留“PR #446 源码基线”和“experimental 完整适配”两个提交。

| 检查 | 本次结果 |
| --- | --- |
| Linux CI 契约测试 | 424 项：423 通过、1 项备选 ATK 检查默认跳过 |
| 正式 ABI、overlay、Stable 门禁/runtime 与 wheel 环境回归 | 149 项：148 通过、1 项跳过；正式 overlay 回归不表示 experimental 支持 overlay |
| 实验 Python loader / CPU golden | 21/21、3/3 |
| 正式阶段调度集成 | 通过 |
| 新版 ABI helper 审计已有真实 RWKV8 OPP | 通过；ELF 导出、公共头、ABI 清单及文件哈希一致 |
| Windows 实验契约测试 | 139 项：99 通过、40 项因平台或环境条件跳过；以 Linux 结果补充覆盖 |

Linux 回归在 A2 原生 Python 3.11 环境执行，日志保存在
`/home/z00879345/experimental-main-sync-20261110-fevnxU/` 的 `contracts.log`、`stable.log`、
`stages.log`、`loader.log`、`golden.log` 和 `real_elf.log`。
真实 ELF 审计复用下节已构建的 OPP；本次没有重建 run/wheel，也未重跑 NPU 精度或 A5 构建。
下节的构建和数值结果属于原基线，不计为本次主干同步后的全量验收。

## 2026-11-10 精简与独立门禁（main `959d6b19`）

变更范围：移除正式 GDR 内的实验串联、正式 wheel/共存/overlay 环境准备和多 profile 配置；合并实验 CI 入口与装载用例入口；构建来源指纹排除开发文档/测试；合并 CPU reference；独立实验 job/check。运行时逐次检查未修改。

本轮在 A2 上从精简后的工作树重新构建并运行完整入口，不复用旧 wheel。
环境：aarch64 / 910B，现有 `fla-npu-ci:9.1.0-910b` 镜像，CANN 9.1.0、Python 3.12.13、torch 2.7.1+cpu / torch_npu 2.7.1.post5，无 ATK/Kombu。

| 检查 | 本轮结果 |
| --- | --- |
| 依赖预检 | 通过；缺 ATK 只 warning，后续构建正常执行 |
| 实验 `.run`、独立安装、包/ABI 审计、wheel 和 host 装载 | 全部通过；算子集合只有 RWKV8 |
| 真实 `.run` 拒绝正式包根、其 `opp` 和新建子目录 | 3/3，受保护目录不变 |
| 装载隔离 | 22/22，无 FAIL/BLOCKED，受保护目录不变 |
| NPU / CPU reference，默认队列 | 209/209；执行证据确认 `queue` |
| NPU / CPU reference，强制 inline | 209/209；执行证据确认 `inline` |
| 共用 reference 后的 PTA 原有边界/负例 | 13/13，使用同一新 wheel |
| Linux CI 契约测试 | 424 项：423 通过、1 项备选 ATK 比较器测试按默认配置跳过 |
| 实验 Python loader / CPU golden | 21/21、3/3 |
| 正式阶段调度集成测试 / 独立状态发布器 | 通过；真实 A2 summary 通过新发布器的报告校验 |

Linux 契约测试使用原生 Python 3.11 环境；NPU 完整入口及 PTA 使用上述无 ATK 的 CI 镜像。必要依赖失败注入、缺失/短报告、错误 head/run/attempt、旧 PR head、上传失败和正式/实验状态独立性均有契约测试。

构建清单记录 `source_commit=f2dd8b93b59c2428f7c01a30bf14d49ecc70ba8b`、`source_dirty=true`，表示在主干同步提交上验证待提交的精简工作树；16 个算子/公共头输入文件哈希与提交输入一致。算子、launcher 和 Python runtime 本轮未改。证据保存在 A2 工作目录 `experimental-minimal-20261110-SrrP3X` 的 `result/` 及 `contracts.log`、`loader.log`、`stages.log`、`golden.log`、`pta.log`。

| 证据/制品 | SHA256 |
| --- | --- |
| `result/summary.json` | `5e46e6854e4e55754d038a2a918a535f6c30c5a9e2e50d13232f6326cbee15a8` |
| `fla_npu_experimental_linux-aarch64.run` | `75abdb4e7dc099bb7391e401aa2d47f056b51edc258109124580fdd256b1bf04` |
| `flash_linear_attention_npu_experimental_a2-0.1.0.dev0+799b4432c42a-py3-none-linux_aarch64.whl` | `baa9ae98a36f72ecaeb2ed77d59f81439c0f83ea88846d1ded94440d31c2b78c` |

本轮没有重跑 A5，也没有运行合入后的 GitHub 新 job；不得把本地入口通过等同于新 GitHub 门禁已经生效。

## 2026-11-10 直接 aclnn 输出布局补测

针对当前实现 `ef34d887`，复用上节同实现的实验 OPP，在 A2 / CANN 9.1.0 镜像内用 C++
直接调用 `aclnnFusedRecurrentRwkv8GetWorkspaceSize` 和执行接口，不经过 Python adapter。
这是一组入口布局补测，不计入原 209 条 Python 精度用例，也没有重跑全量门禁。

固定输入：`B=H=1`、`T=2`、`K=V=8`，六个连续 FP32 输入全部为零；无初态，`scale=1`、
`reverse=false`、`outputChunkState=false`、`outputSa=false`、`chunkLen=16`。
两组输出逻辑 shape 都是 `(1,1,2,8)`，存储预填 7，数学期望为 16 个零。

| 输出布局 | 存储与 strides（元素为单位） | 结果 |
| --- | --- | --- |
| 连续 | 分配 16 个 FP32；strides=`(16,16,8,1)` | workspace 查询返回 0，大小 1536 B；执行并同步后 16/16 输出为零 |
| 非连续 view | 分配 32 个 FP32；strides=`(32,32,16,2)`，逻辑元素占用存储下标 `0,2,...,30`，无重叠、无越界 | workspace 查询返回 `561103`；消息 `Output tensor should not from workspace`；未调用执行接口 |

这证明当前直接 aclnn 的合法非连续输出无法执行，不证明已发生静默结果错误。
`sOut/saOut` 的同类处理仅做了代码检查，尚未分别复现。Python adapter 自行分配连续输出，
没有调用方提供的 `out=`，因此已有 Python 精度结果与本限制并不矛盾。
该写回逻辑在来源 PR #446 的 `049eb0d` 中已经存在，尚未修复；原因与修复方向见
[设计 §6.2](../docs/architecture/experimental%20构建适配设计.md#62-rwkv8-直接-aclnn-的非连续输出限制)。

## 已有参考证据（不计为本轮重跑）

| 版本/日期 | 环境 | 已证实范围 |
| --- | --- | --- |
| `ee344941` / 2026-11-09 | A2，CANN 9.1.0 CI 镜像，torch 2.7.1 / torch_npu 2.7.1.post5，无 ATK/Kombu | 复用旧 wheel，直接 NPU / CPU reference 默认和 inline 各 209/209；依赖预检缺 ATK 只 warning |
| 2026-11-09 | A2 原生，CANN 9.0.0，torch/torch_npu 2.12.0 | 旧 28 项矩阵及 quick 3×2，通过；Linux 契约测试 314 项，含备选 ATK 比较器对齐 |
| 2026-11-06 | A2，910B2，CANN 9.0.0，ATK 26.9.8 | RWKV8-only `.run` / wheel，`--ops=all` 和单选构建；旧矩阵 28/28；原 ATK 默认/inline 各 3/3 |
| 2026-11-03 | A5，CANN 9.2.0 | 当时版本的编译、包内容与 host 装载；未验证 A5 数值 |

2026-11-06 卡 2 曾出现基础 `isfinite` 异常，在不加载 FLA 的全 1 张量上也可复现；当轮最终 ATK 数值结论来自卡 1。该异常不作为算子精度通过证据。

旧 GitHub [NPU CI run](https://github.com/flashserve/flash-linear-attention-npu/actions/runs/42574483532) 中 A5 容器步骤成功，但 artifact 上传失败，A2 被取消；该次整体门禁未通过。

## 通过标准与限制

- 正式/实验集合分别构建，实验制品只含 RWKV8。
- 实验 `.run` 拒绝正式包目录，wheel 使用独立身份；装载负例必须因指定原因被拒绝。
- A2 默认与 inline 对拍，逐输出数值和实际库来源符合契约；A5 只计构建/host。
- 新独立 GitHub job 需要工作流合入受保护目标分支后生效；分支保护同步添加实验 context。
- 未完成 reverse、并发、多 stream、确定性、MSS、性能或 A3/A5 数值全量验收；不声明正式发布验收完成。
