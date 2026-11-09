# experimental CI 接入验证（2026-11-09）

本次只修改 CI 调度、测试准备、结果检查与文档，未修改 RWKV8 输入、CPU 标杆、精度阈值或
算子实现。基于 PR #1097 的 `866b858f` 工作树验证；原 framework / RWKV8 两次提交保留。

## 调度和报告

`run_checks.sh` 的 `gdr-example-st` 阶段执行：正式 GDR → 实验总入口 → 正式 GDR。
实验总入口执行原构建验证脚本，再在 A2 执行真实 wheel 装载矩阵和 ATK；A5 的 runtime 明确
记录 skipped。实验失败及正式后置回归失败均传递到既有 required status。

与 main `fb865165` 的 `run_checks.sh`、`run_ci_container.sh` 三方合并无冲突；合并后的调度脚本
通过 main 的 `test_run_checks_stage_pipeline.sh`。没有修改 workflow 或原有状态名；实验分项
结果写入当前 main workflow 已上传的 stage JSON。

## A2 实测

服务器：`z00879345@179.178.1.2`，910B2，CANN 环境
`/home/z00879345/Ascend/9.0.0/ascend-toolkit/set_env.sh`，Python 3.11、torch / torch_npu 2.12.0、
ATK 26.9.8、Kombu 5.6.2。正式控制包复用 2026-11-06 构建的 A2 制品，实验包由本次脚本重新
构建。主验证使用空闲物理卡 5。

| 项目 | 结果 |
| --- | --- |
| 实验全量 `.run`、空目录安装、ABI / 来源审计、wheel 和 host 装载 | 通过 |
| 自动准备隔离 Python 环境 | 通过；不继承父环境的 FLA 包或 `.pth` |
| 真实 wheel 装载矩阵 | 28/28；正式控制包哈希不变 |
| 原 RWKV8 ATK，默认启动模式，quick 前 3 条 | 3/3；NPU DUT / CPU golden |
| 原 RWKV8 ATK，inline，quick 前 3 条 | 3/3；NPU DUT / CPU golden |
| 独立 ATK wheel 安装与正式库路径清理（物理卡 6） | inline 3/3；使用固定 Kombu 依赖 |
| Linux CI 契约单测 | 306/306，无跳过；另通过原调度集成测试 |
| 失败传播与报告契约 | 覆盖构建、隔离、精度、正式前/后置失败，以及缺失/短跑报告拒绝 |

主验证证据位于服务器：

```text
/home/z00879345/experimental-ci-20261109-YAB6tv/
  run-final.log
  contracts-publish.log
  atk-wheel-option-pinned/accuracy-inline.json
  results/experimental-ci-wgs7ttgd/
    summary.json
    build-host.log
    profiles.json
    isolation/summary.json
    isolation/L01…L28/observation.json
    accuracy-queue.json
    accuracy-inline.json
    atk-queue/…/report/*.xlsx
    atk-inline/…/report/*.xlsx
```

首次验证发现专用 ATK 启动脚本只解析 entry point、没有调用入口，已修正并重跑上述完整链路。
ATK wheel 自动安装路径发现 Kombu 5.7 拒绝 ATK 使用的 `:` / `@` exchange 名称，固定
`ci/experimental-atk-constraints.txt` 为 5.6.2，并增加运行前检查，防止任务无执行却持续等待。
正式 GDR 导出的 vendor 库路径在实验子进程中清理，保留 CANN 与驱动路径。

## 验证边界

- 本轮执行 quick 精度；full 已接入原 209 条 × 两种启动模式，但本轮没有重跑 full。
- 28 项矩阵是装载隔离验证，不执行 NPU 算子；数值结论来自上述 ATK 报告。
- 本地未执行正式全量构建和两轮完整 GDR；其调用顺序、失败传播通过调度测试验证，真实执行
  仍由 PR 的 A2+A5 NPU CI 门禁确认。A5 本地未复验。
- reverse、更多 stream/并发、确定性、MSS 和性能不在本次 CI quick 精度验收范围。
