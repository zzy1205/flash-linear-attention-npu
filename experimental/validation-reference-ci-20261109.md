# experimental 独立 CPU reference CI 验证（2026-11-09）

本轮在 `4e0c6a2e` 上修改 CI 依赖预检、精度执行入口、契约测试与文档，未修改算子、launcher、
ABI、输入范围或精度阈值。默认路径直接比较 NPU 与 CPU reference；ATK 工程保留为手工备选。
原 ATK 接入实测属于 [前一轮记录](validation-ci-20261109.md)，不计作本轮独立脚本的执行结果。

## 改动与验收口径

- `environment-contracts` 在正式构建前检查必要 Python 依赖，正式 Example ST 的原依赖检查
  也提前执行。缺少 ATK/Kombu 只 warning，默认精度检查继续运行。独立实验入口也在编译前预检。
- 原精度 JSON：quick 前 3 条，full 全部 209 条；默认启动模式与 inline 各执行一次。
- 复用原量化输入和 CPU FP64 reference，golden 转 FP32；按输出原 dtype 使用 ATK 26.9.8
  `mixed_tolerance_bm` v1 的阈值。每个输出检查 shape/dtype/device/None/有限值。
- AST 检查防止独立 reference 与自包含 ATK executor 漂移。额外与已安装 ATK 比较器对拍，
  覆盖三种 dtype、随机误差、99% 比例边界和绝对误差异常点，判定一致。
  此项通过 `FLA_EXPERIMENTAL_TEST_ATK_PARITY=1` 显式启用；默认契约测试也不导入 ATK。
- 数值失败、空/短报告、重复或错误用例、执行异常、超时均阻断门禁；保留逐输出指标与日志。

## A2 实测环境与制品

无 ATK 的精度验证使用现有 CI 镜像 `fla-npu-ci:9.1.0-910b`，镜像 ID：
`sha256:9d57fac03ca967f1620084dc0dfe52d4a191e4226173a8df0424c4c35a3994a9`。
环境为 910B2、CANN 9.1.0、Python 3.12.13、torch 2.7.1+cpu、torch_npu 2.7.1.post5。
只映射空闲物理卡 5，容器内逻辑设备为 0。预检报告的 ATK 与 Kombu 均为 null，状态为 success。

装载矩阵及 quick 调度回归另使用同一 A2 服务器的独立 Python 环境、空闲卡 6：CANN 9.0.0、
Python 3.11、torch / torch_npu 2.12.0；该环境另有 ATK 26.9.8 / Kombu 5.6.2，可执行比较器一致性测试。
默认精度入口仍只调用独立脚本。

两轮均复用前一轮构建并通过审计的实验 wheel，版本为 `0.1.0.dev0+799b4432c42a`，SHA256：
`f5c554fd125e367a530043829e0cd547218e90b40ed7d9909ffff6a663153197`。
正式控制包同样复用已验证制品；本轮本地没有重新编译 C++。

| 项目 | 本轮结果 |
| --- | --- |
| 无 ATK 的 CI 镜像依赖预检 | 通过，缺 ATK 只 warning |
| 默认启动模式，NPU / CPU reference，full | 209/209 |
| inline，NPU / CPU reference，full | 209/209 |
| 自动创建私有 Python 环境、装载隔离 | 28/28；正式包内容哈希不变 |
| quick 调度，默认 / inline | 各 3/3；包含同步后的实验库来源证据 |
| Linux CI 契约单测（含 ATK 比较器对齐） | 314/314，无跳过 |
| 最终代码的调度测试 | 8/8；原阶段集成测试通过 |
| 当前 main 的 CI 脚本兼容 | 三方合并无冲突，合并后的阶段集成测试通过 |
| 必要依赖缺失的失败注入 | environment-contracts 失败，构建及后续阶段均不执行 |

所有输出的逐用例误差、阈值、状态保存在 JSON，默认 CI 的分项结果嵌入已有 stage artifact。
本地证据保留于 A2 的专用验证目录：

```text
/home/z00879345/experimental-reference-20261109-s1vU5t/
  preflight.json
  contracts-final.log
  container.log
  precision-only/
    accuracy-queue.json
    accuracy-inline.json
    accuracy-queue-trace/experimental-*.json
    accuracy-inline-trace/experimental-*.json
  runtime-native.log
  runtime-reused/
    summary.json
    isolation/summary.json
    accuracy-queue.json
    accuracy-inline.json
```

首次手工容器试跑因继承非实验 OPP 路径被正确拒绝；按 CI 的 `clean_environment` 启动后通过。
另一次试跑把宿主物理卡号误作单卡容器的可见设备编号，CANN 拒绝初始化，报告明确失败。
修正验证启动命令后复跑；未修改 runtime 的隔离规则。这两次失败不计为精度通过。

## 验证边界

- 上述为复用制品的运行回归；新的实验 `.run` / wheel 构建、正式全量构建、前后两轮完整 GDR
  及 A5 验证由更新后 PR 的双平台 CI 再确认。
- 28 项装载矩阵不执行算子，数值结果来自独立对拍报告，不能相互替代。
- 本轮未重跑 ATK 确定性、MSS、性能或更多 stream/并发场景，不声明完成全部正式验收。
