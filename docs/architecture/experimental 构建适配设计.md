# experimental 目录收敛与转正式设计

作者：zzy1205。更新：2026-11-12。代码基线：main `c39b0eb3`，包含已合入的 PR #1177（合入提交 `101e92d0`）。

**状态：待确认的重构方案。** 本轮仅同步主干、解决冲突及修订设计；目录迁移、功能删减和转正式工具尚未实施。
当前代码的执行命令见 [README](../../experimental/README.md)，验证结果见 [验证记录](../../experimental/validation.md)。
下文“目标”“迁移”“删除”均指确认后实施，不代表当前代码已满足新布局。

## 1. 三条约束

1. **实验算子不跑 CI。** 不增加 workflow、触发参数、job、必需状态、自动用例发现或正式 CI 中的实验调用。验证只手动执行。
2. **实验资产归属于 `experimental/`。** 不仅是 kernel 和 torch adapter，Python wrapper、安装/打包脚本、reference、测试、ATK、示例和设计文档也必须放在该一级目录内。CMake 构建接入允许在 `cmake/`；根 `build.sh` 只保留已有 `--experimental` 模式及必要分发。
3. **转正式是可检视的源码迁移。** 不重写 kernel、host 或数学 reference，不靠改 vendor 名称把实验二进制变成正式包。移动完整算子资产，补正式注册和公共入口，再按正式路径重新构建、验证。

实验仍使用独立 vendor、run/wheel、Python 包和 dispatcher；正式构建与实验构建互不包含。
独立安装目录和新 Python 进程仍是使用前提。experimental 不支持 overlay、增量组件拼装、热更新或正式/实验同进程混用。

## 2. 目标目录与归属

```text
experimental/
├── DESIGN.md                          # 本设计的目标位置
├── README.md / validation.md           # 使用方法及手动验证记录
├── _framework/                        # 实验专用发现、构建、打包、安装及来源清单
├── _python/fla_npu_experimental/       # 通用实验 loader/launcher 桥接；不写算子专用逻辑
├── _tests/                            # 框架手动测试、安装与装载隔离测试
├── tools/promote.py                    # 拟新增：转正式的计划与机械迁移工具
└── fused_recurrent_rwkv8/
    ├── CMakeLists.txt
    ├── op_host/ / op_kernel/           # 与正式算子一致的源码布局
    ├── adapter/
    │   ├── stable_fused_recurrent_rwkv8.cpp
    │   └── python.py                  # 真签名 wrapper、参数检查、公开入口声明
    ├── tests/
    │   ├── reference.py               # 本算子唯一的数学 reference
    │   ├── pta/ / cases.py
    │   └── atk/                       # JSON、YAML、generator、executor、来源记录
    ├── examples/
    └── docs/ / README.md
cmake/                                 # 允许保留模式选择、依赖闭包、制品生成接入
build.sh                               # 仅模式参数及分发；不承载实验运行逻辑
```

`_` 开头的目录为框架，不参与实验算子发现；`tools/` 也显式排除。普通算子目录必须含 `CMakeLists.txt`。
正式模式的发现根保持 `fla/ops/ascendc/`，不递归扫描实验树；实验模式只发现实验算子，不引用正式算子 adapter。

### 2.1 当前文件到目标位置

| 当前内容 | 重构动作 |
| --- | --- |
| `scripts/experimental_manifest.py` | 移至 `experimental/_framework/manifest.py`，更新 CMake 和实验脚本引用 |
| `cmake/scripts/experimental/` 的安装脚本和帮助文本 | 移至 `experimental/_framework/package/`；CMake 仅引用这些文件，例外不扩展到脚本资产 |
| `scripts/check_build_ops.py` 的实验发现分支 | 移入实验框架；正式脚本恢复主干，模式分发由 build.sh/CMake 负责 |
| `tests/atk/fused_recurrent_rwkv8/` | 整体移至算子的 `tests/atk/`，手动入口显式接受该目录，不注册到正式 ATK 发现集合 |
| `tests/atk/common/experimental_isolation_trace.py` | 移入 `experimental/_tests/`；不修改正式 ATK common |
| `experimental/_tests/rwkv8_reference.py` | 移入算子的 `tests/reference.py`；PTA、独立对拍和 dump 引用同一实现 |
| `experimental/_python/fla_npu_experimental/_rwkv8.py` 及其专用参数检查 | 移入算子的 `adapter/python.py`；包构建时仅收集选中算子的 wrapper |
| 当前 `docs/architecture/experimental 构建适配设计.md` | 移至 `experimental/DESIGN.md`；使用说明及历史链接一并更新，不保留两份正文 |
| `torch_custom/fla_npu/csrc/build_stable.py` 中本 PR 抽取的编译 helper | 实验专用编排收回 `_framework/`；正式 builder 恢复主干，复用现有公共查询函数 |
| `torch_custom/fla_npu/csrc/src/runtime.cpp` 中 `FLA_STABLE_EXPERIMENTAL` 特例 | 改由实验桥接显式绑定路径与校验自有符号，正式 runtime 恢复主干 |
| `scripts/build_wheel.py`、`scripts/stable_build_policy.py` 中的实验专用拦截 | 收回实验构建/安装边界；正式 wheel 路径必须拒绝实验产物，通过负例后才能删除原拦截 |
| `tests/test_stable_gates.py` 中针对本 PR helper 的测试 | 随 helper 收回实验测试目录，正式测试恢复主干 |
| `build.sh` 的实验发现、ABI 检查和 vendor 逻辑 | 只保留模式分发；具体实验处理归入 `_framework/` 或 CMake，删除无效的实验 PR 定向构建分支 |

实验代码可以调用仓库已有的公共构建、ABI、tensor/stream 工具；这些公共工具不等于实验资产。
不允许为了实验新增的特例继续散落在正式模块中。确需修改公共能力时，单独按公共改动检视，不以实验功能名义扩大本 PR。

## 3. 构建、安装、运行的最小方案

### 3.1 build.sh 参数兼容边界

| 参数/能力 | 当前支持 |
| --- | --- |
| `--pkg --experimental --soc=<soc> --ops=<op>` | 支持 |
| `--ops=op_a,op_b`；不传或 `all/ALL` 全选实验集合 | 支持 |
| `--experimental --list-ops` | 支持 |
| `--list-ops --experimental` | 不支持 |
| `-j`、`-O`、`--build-type`、`--verbose`、`--bisheng_flags`、`--oom` | 支持 |
| `--kernel_template_input`，按主干单算子限制 | 支持 |
| 固定实验 vendor、单 SoC | 支持 |
| 自定义 vendor、多 SoC 同包 | 不支持 |
| `--static`、`--jit`、分目标构建、UT/示例、PR 定向构建 | 不支持 |
| experimental overlay | 不支持 |

本次重构不顺带补齐未支持的构建组合。参数语义保持，修改范围限于目录归属、薄接入和转正式。

### 3.2 制品与安装

- `--experimental` 选择实验源码、依赖闭包及 adapter，生成固定实验 vendor；正式模式完全保持主干路径。
- `.run` 仅安装到显式独立目录，拒绝正式 wheel/OPP 及其子目录；wheel 使用独立 distribution/package。
- wheel 内嵌与本次 launcher 匹配的实验 OPP；不提供骨架实验 wheel、run 原位更新或外部 OPP 拼装。
- 首次装载检查包身份、算子集合、实际 ABI 和库来源；ABI 检查继续复用公共工具，不另建一套实验 ABI 协议。
- 来源清单仅保留构建追踪与制品匹配必需信息。重复的源树、Python 全量哈希和多套来源证明在确认后收敛。

### 3.3 与 #1177 的接入边界

同步 #1177 是为了保持公共编译、动态库绑定和正式增量安装兼容，不为 experimental 增加 overlay。
正式包继续使用主干的 layer manifest、适配层/OPP 组合及安装管理器；实验包不注册到这套管理器，不维护 active/pending 等组件状态。

目标实现使用实验目录内的薄桥接，显式调用共享 `Runtime::init(custom_lib_path)` 绑定唯一实验 host 库。
首次装载时确认全部自有 ACLNN 符号来自该库，缺失则报错；通用 SDK 符号继续使用主干运行时。
实验 namespace、注册单元及桥接代码均从实验目录生成，不复制整份正式 runtime，也不在正式 runtime 保留实验宏分支。
桥接在开放算子调用前完成显式初始化及自有符号来源检查，并缓存结果；检查失败不进入计算路径。
编译继续保留 #1177 的 `-fno-gnu-unique` 等隔离设置；转正式后使用主干正式注册机制。

当前主干合入的是 #1177 的 `f87e0b38` 版本；PR 正文中标为“本地待推送”的后续方案不是本次实现依据。

## 4. 删除与保留清单

| 内容 | 决定 |
| --- | --- |
| 实验 CI、suite 参数、job、必需状态、自动用例发现 | 已删除；不恢复 |
| 同环境共存与安装顺序兼容矩阵、实验 overlay、增量组件状态机 | 不支持，不建设 |
| 每次调用重新读取清单、遍历文件元数据、扫描进程映射 | 拟删除；来源/完整性检查收敛到首次装载 |
| 任意文件加载后篡改、恶意替换等监控及专属用例 | 拟删除；不承诺热替换，升级后重启进程 |
| 多套数学 reference、重复包装和重复使用文档 | 合并；数学内容不变 |
| 专门的失败状态发布、报告上传、CI 身份字段 | 已删除；手动报告只保留诊断需要的信息 |
| 正式/实验源码选择、安装拒绝、包/namespace 区分 | 保留 |
| 首次装载的 ABI、算子集合和库来源检查 | 保留 |
| 官方 Python 入口的同进程混用拒绝、轻量环境路径检查 | 保留；不支持绕过公开入口手动加载另一套原生库 |
| 手动构建、精度对拍、安装/首次装载隔离用例 | 保留；删除不再承诺的场景，不再固定要求旧 22 项矩阵 |

## 5. 快速转正式

### 5.1 工具边界

拟新增一个小工具 `experimental/tools/promote.py`，只负责确定性的文件移动、入口补丁和清单输出，不改算法、不做任意源码字符串替换、不发布制品。
算子自身的特殊参数检查保留在 `adapter/python.py`；实验/正式差异集中在调用后端和公共入口注册。
算子包含其他实验依赖时，必须同时迁移依赖闭包或先将依赖转正，不允许正式源码继续引用实验路径。

```bash
# 计划：打印源/目标文件、依赖和所需注册补丁，不写文件。（拟新增命令，当前不可执行）
python experimental/tools/promote.py fused_recurrent_rwkv8 --group gdn --dry-run
# 应用：目标不存在且源文件与计划一致时，执行移动及支持的确定性注册补丁。（拟新增）
python experimental/tools/promote.py fused_recurrent_rwkv8 --group gdn --apply
```

工具默认只预览；目标冲突、未支持的 wrapper 结构、缺失依赖或注册重名时在任何修改前退出。
有版本控制的工作区必须干净；应用失败撤销本次已执行的移动，不影响其他算子。只处理明确列出的文件，不做全仓重排。

### 5.2 转正式文件映射

| 实验资产 | 正式落点/动作 |
| --- | --- |
| `<op>/CMakeLists.txt`、`op_host/`、`op_kernel/` | `fla/ops/ascendc/<group>/<op>/`，保留实现和私有头 |
| `<op>/adapter/stable_<op>.cpp` | `torch_custom/fla_npu/csrc/src/stable_<op>.cpp` |
| adapter schema / implementation | 在 `torch_custom/fla_npu/csrc/src/stable_ops.cpp` 增加 include、`m.def` 和 `m.impl` |
| `<op>/adapter/python.py` | 接入 `torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py`，登记 `_ASCENDC_OPS`；有 mutation 时补对应元数据 |
| `<op>/tests/atk/` | `tests/atk/<op>/`，更新 DUT 为正式入口，保留用例和精度标准 |
| `<op>/tests/reference.py`、PTA、示例和算子文档 | 随算子或测试工程迁移；公共测试不得反向依赖 experimental |

当前正式适配层仍有中心注册点，所以**不是只做一次 `git mv` 就完成转正**。
工具产出的注册补丁仍需检视，尤其是真签名 wrapper、别名、mutation 和依赖；不引入自动源码改写框架来隐藏这些必要差异。

### 5.3 转正完成条件

1. 源码及依赖已移出实验集合；正式构建包含该算子，实验全编不再包含它。
2. 正式 wrapper/schema/adapter 覆盖校验、实际调用 ABI 检查通过，#1177 的正式 layer manifest 能包含新入口。
3. 用相同输入、reference 和精度标准在正式 wheel 上重新验证；实验旧 wheel 的结果不能替代正式构建结果。
4. 按正式算子的准入和 CI 规则检视。实验阶段不跑 CI，不代表转正式可以跳过正式门禁。

## 6. 本轮落地状态与验证

### 6.1 当前状态

| 项目 | 本轮状态 |
| --- | --- |
| 同步 main 与已合入 #1177 | 已同步至 `c39b0eb3` |
| 编译 helper 冲突 | 已解决；保留 `-fno-gnu-unique`、主干 layer 协议标记和实验显式编译单元 |
| runtime 路径合并 | 保留 #1177 的显式多路径处理以及当前实验隔离；重构前不提前删除实验宏 |
| 目标目录、来源检查简化、转正式工具 | 待方案确认后实施 |
| 本轮测试 | 记录于 validation.md；不启动 NPU CI，不声明重新完成 NPU 编译/精度验收 |

确认后按“资产归位 → 精简实验桥接和装载检查 → 实现迁移工具 → 手动验证”的顺序推进。
正式目录最终只保留必要的 build.sh/CMake 接入；对照主干逐文件审核，不以重新添加实验 CI 完成验收。

### 6.2 RWKV8 直接 aclnn 的非连续输出限制

此项属于从 PR #446 迁入的 RWKV8 host 输出处理，独立于 experimental 的构建、安装和装载隔离。
需要区分两个入口：

| 调用入口 | 输出缓冲区由谁提供 | 当前影响 |
| --- | --- | --- |
| `fla_npu_experimental.fused_recurrent_rwkv8(...)` | Stable adapter 通过 `allocate_sizes` 新建连续输出；Python 签名没有 `out=` 参数 | 不触发此次非连续输出问题；已有 A2 手动精度验证覆盖此路径 |
| `aclnnFusedRecurrentRwkv8GetWorkspaceSize(..., out, sOut, saOut, ...)` | 调用方提供 aclTensor，可携带非连续 strides | `out` 非连续时已复现 workspace 阶段失败；`sOut/saOut` 有同类代码，但尚未分别实测 |

当前 host 将输出先转连续，再作为写回目的地：

```cpp
auto out_ = l0op::Contiguous(out, executorPtr);
auto viewCopyResult = l0op::ViewCopy(outRet, out_, executorPtr);
```

对非连续输入，生成连续临时张量供 kernel 读取是正常做法；对调用方的输出，写回目的地必须
仍是原始缓冲区。这里 `Contiguous` 为非连续 `out` 生成 executor 管理的连续临时张量，
随后 `ViewCopy` 的目标变为临时张量，而没有向原 `out` 写回。

A2 / CANN 9.1.0 补测中，相同合法 shape 和 FP32 零输入，连续输出执行正确；改用最后一维
stride 为 2 的非连续输出 view，`GetWorkspaceSize` 返回 `561103`，错误为 `Output tensor should not from workspace`。
该次调用在准备阶段失败，未启动计算，不是已证实的静默数值错误。具体复现条件和对照结果见
[验证记录](../../experimental/validation.md#2026-11-10-直接-aclnn-输出布局补测)。

接口文档目前没有声明输出必须连续，故将其记录为直接 aclnn 的实现限制，不能把现有 Python
精度通过扩大解释为任意 aclTensor 输出布局都可用。最小修复方向是直接
`ViewCopy(outRet, out, executorPtr)`，`sOut/saOut` 同理，并补三个输出的非连续 view 对拍、
未选中存储区域保持不变及原连续/可选/空输出回归。此处只记录问题与修复方向，代码尚未修改；
此问题独立于目录迁移；转正式前仍需按算子准入要求处理，不在本轮同步中修改。
