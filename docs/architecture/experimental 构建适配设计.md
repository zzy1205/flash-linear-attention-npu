# experimental 目录收敛与转正式设计

作者：zzy1205。更新：2026-11-12。代码基线：main `c39b0eb3`，包含已合入的 PR #1177（合入提交 `101e92d0`）。

**状态：待确认的重构方案。** 已同步主干并解决冲突；本次继续更新设计要求，尚未实施目录迁移、功能精简或增量安装接入。
当前代码的执行命令见 [README](../../experimental/README.md)，验证结果见 [验证记录](../../experimental/validation.md)。
下文目标布局及增量安装边界均为待实施方案，不表示当前代码已经支持。

## 1. 设计要求

1. **正式与实验构建严格双向隔离。** 正式编包不得包含实验算子，实验编包不得包含正式算子。源码发现、依赖闭包、torch adapter、Python 入口、注册单元以及最终 run/wheel 内容都受此限制；不能只隔离 kernel。
2. **实验资产放在一级 `experimental/` 下。** 实验专属文件保留原有子目录结构，整体下移一级；已有实验目录内的文件不为统一布局反复搬迁。公共 CMake 可保留必要接入，根 `build.sh` 只保留模式选择及分发。
3. **实验算子不跑 CI。** 不增加 workflow、suite 参数、job、必需状态或正式 CI 中的实验调用；实验测试仅手动执行。
4. **增量安装不为实验算子做特殊处理。** 复用通用增量安装机制，不增加实验专属安装模式、base 类型检查、状态机或恢复分支。开发者须在实验算子增量安装前自行保证 base wheel 是实验包，而非正式包；正式增量也不得选择实验 base。
5. **快速转正式以源码迁移为主。** 给出文件映射和注册步骤，不新增自动迁移工具。保留 kernel、host、reference 和用例，补正式入口后重新构建验证。

实验保留独立 vendor、run/wheel、Python 包和 dispatcher。正常构建、打包及独立安装不覆盖正式制品。
构建集合隔离由代码强制；增量安装的 base 归属由开发者保证，不承诺安装器识别或阻断跨集合误装。
这不放宽通用增量安装原有的 ABI、清单和兼容性检查，也不支持进程内热更新或正式/实验同进程混用。

## 2. 目录调整：保留结构，下移一级

```text
experimental/
├── docs/architecture/experimental 构建适配设计.md  # 本文目标位置：原路径增加前缀
├── README.md / validation.md                     # 保持现有位置
├── scripts/                                     # 原根 scripts/ 中的实验专属逻辑
│   └── experimental_manifest.py
├── tests/                                       # 原根 tests/ 中的实验测试
│   └── atk/
│       ├── common/experimental_isolation_trace.py
│       └── fused_recurrent_rwkv8/                # 原有内部结构不变
├── cmake/scripts/experimental/                  # 实验安装脚本、帮助文本
├── torch_custom/                                # 从公共文件提取的实验编译/桥接逻辑
├── _framework/ / _python/ / _tests/              # 已在实验目录内，保持现有布局
└── fused_recurrent_rwkv8/                        # 已在实验目录内，保持现有布局
    ├── CMakeLists.txt
    ├── op_host/ / op_kernel/ / adapter/
    └── tests/ / examples/ / docs/ / README.md
cmake/                                           # 保留必要的公共构建接入
build.sh                                         # 模式选择及分发
```

`scripts/`、`tests/`、`cmake/`、`torch_custom/`、`docs/` 以及 `_` 开头的目录明确排除在实验算子发现之外。
其余实验算子目录必须含 `CMakeLists.txt`。正式发现根保持 `fla/ops/ascendc/`，不扫描实验树。
实验构建可以复用既有公共头、运行时和 ABI 工具，但不能引入正式算子的实现、adapter 或注册集合。

### 2.1 文件处理清单

| 当前内容 | 重构动作 |
| --- | --- |
| `scripts/experimental_manifest.py` | 移至 `experimental/scripts/experimental_manifest.py`，调整引用 |
| `cmake/scripts/experimental/` | 移至 `experimental/cmake/scripts/experimental/`；增量安装复用通用实现，不保留实验专属 overlay 禁止分支 |
| `scripts/check_build_ops.py` 的实验发现分支 | 提取至 `experimental/scripts/`，正式脚本恢复主干 |
| `tests/atk/fused_recurrent_rwkv8/` | 移至 `experimental/tests/atk/fused_recurrent_rwkv8/`，保留内部结构，手动入口显式选择该目录 |
| `tests/atk/common/experimental_isolation_trace.py` | 移至 `experimental/tests/atk/common/experimental_isolation_trace.py`；通用 ATK 工具继续复用 |
| 本设计文档 | 原路径增加 `experimental/` 前缀，更新链接，不保留两份正文 |
| `torch_custom/fla_npu/csrc/build_stable.py` 中本 PR 抽取的编译 helper | 实验编排提取至 `experimental/torch_custom/`，正式 builder 恢复主干；不复制整套正式构建器 |
| `torch_custom/fla_npu/csrc/src/runtime.cpp` 中实验宏分支 | 改由实验目录内的桥接传递通用运行时所需路径，正式 runtime 恢复主干；不复制整份 runtime |
| `scripts/build_wheel.py`、`scripts/stable_build_policy.py` 中实验专属构建拦截 | 收回实验构建边界；正式 wheel 不能纳入实验内容的负例通过后，才能删除原拦截 |
| `tests/test_stable_gates.py` 中本 PR helper 的测试 | 随实验逻辑移入 `experimental/tests/`，正式测试恢复主干 |
| `build.sh` 的实验分支 | 只保留必要分发；具体处理进入实验目录或 CMake，删除无效的实验 PR 定向构建分支 |
| 已在 `experimental/` 内的 wrapper、reference、adapter 和 PTA | 保持现有位置；不再为了按算子聚合而重排 |

公共文件只提取实验逻辑，不把完整的 `scripts/`、`tests/` 或 `torch_custom/` 复制成第二套实现。
通用增量接口如需补充目标路径、包名等参数，应是通用配置，不得变成识别 experimental 类型的专用分支。

## 3. 构建、安装和运行边界

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
| 实验包增量安装 | 不支持 |

上表记录当前实现。实验包增量安装的目标要求见 §3.3；其他未支持的构建参数不在本次扩展。

### 3.2 双向构建与制品隔离

| 检查点 | 正式模式 | 实验模式 |
| --- | --- | --- |
| 算子发现与依赖 | 只选正式集合 | 只选实验集合及其实验依赖 |
| `--ops` 单选、多选、全选 | 只对正式集合生效；实验名报错 | 只对实验集合生效；正式名报错 |
| adapter、Python 入口和注册单元 | 不编译、注册或打包实验入口 | 不编译、注册或打包正式入口 |
| run/wheel 内容 | 只有选中正式算子及其依赖 | 只有选中实验算子及其依赖 |
| 构建目录与缓存复用 | 不继承实验编译列表和产物 | 不继承正式编译列表和产物 |

两种模式切换时，配置和打包校验必须排除另一集合的残留；不能依赖开发者手工清理来保证制品不混装。
复用通用构建工具不等于编入另一集合。默认正式 wheel、实验 wheel 和两类 run 的实际内容都需要手动检查。
独立包身份、普通安装目录和装载 namespace 保持隔离；实验私有打包器不得把产物放入正式包的路径。

### 3.3 增量安装：通用机制，开发者选择 base

增量安装按主干的 [OPP 增量设计](../OPP叠加层增量设计.md) 和 [使用指南](../Wheel增量安装指南.md) 执行。
experimental 不另建一套 overlay 协议、管理器、优先级、状态文件或卸载恢复流程，也不单独扩展主干尚未支持的组合。

| 内容 | 责任与处理 |
| --- | --- |
| 实验增量的 base wheel | 开发者保证已安装并选中实验 base，不能是正式 base |
| 正式增量的 base wheel | 开发者保证使用正式 base，不能是实验 base |
| 安装目标及 Python 环境 | 开发者明确选择，不由安装器根据“实验/正式”自动纠正或切换 |
| ABI、协议、清单及通用兼容性 | 沿用通用增量机制，不绕过、不重复实现 |
| 实验身份专属防混装判断 | 不新增；现有实验专属 overlay 拦截在接入时移除 |
| 安装、移除、启用、回退与重新启动 | 使用通用机制及相同的启动时序 |
| 跨集合误选 base | 违反使用前提，不承诺专门识别、拒绝或自动恢复 |

因此，不能再将“实验 run/wheel 在任意增量安装目标上都必然被拒绝混装”作为验收条件。
严格隔离的构建结果仍必须保证；增量安装正确性在“开发者选对同一集合的 base”这一前提下验证。

当前实验 loader 仍限制唯一内嵌 OPP，安装器仍拒绝 overlay，尚未完成上述接入。
后续需让实验制品提供通用机制要求的元数据，装载通用机制选出的适配层与 OPP 路径；不能仅删除安装拦截就宣称支持。
只接入所需公共能力，不在通用安装器或 runtime 中增加 experimental 类型判断。
通用 ABI 与选中路径的来源检查保留，但不重新包装成一套实验 base 身份门禁。

### 3.4 与 #1177 的同步边界

本轮同步实际已合入的 #1177 `f87e0b38`，保留 `-fno-gnu-unique`、主干 layer 协议标记及 runtime 显式路径处理。
PR 正文中标为“本地待推送”的内容不是当前实现依据。目标方案复用通用增量能力，仍需按实际版本接入和手动验证。
正式主干行为保持；实验桥接放在实验目录内，复用公共运行时和通用层选择结果，不复制实现。

## 4. 删除与保留清单

| 内容 | 决定 |
| --- | --- |
| 实验 CI、suite 参数、job、必需状态、自动用例发现 | 已删除；不恢复 |
| 实验专属 overlay 管理器、状态机、base 类型防混装和跨集合恢复矩阵 | 不建设；通用机制和开发者前提按 §3.3 执行 |
| 原先全面禁止 experimental overlay 的设计与拦截 | 设计取消；代码待接入通用流程时移除 |
| 每次调用读取清单、遍历文件元数据、扫描进程映射及加载后篡改监控 | 拟删除；必要检查收敛到首次装载，更新后重启进程 |
| 新建按算子重排的目录层次和自动 `promote.py` 工具 | 取消；保留现有内部结构和人工迁移清单 |
| 正式/实验源码、依赖、adapter、注册和制品集合隔离 | 保留，必须双向强制执行 |
| 包身份、独立安装路径和 namespace | 保留；增量 base 选择不增加实验专属判断 |
| 首次装载的 ABI 和选中库来源检查 | 保留并复用公共能力，不能阻断合法的同集合增量路径 |
| 手动编译、精度与隔离用例 | 保留；调整过时用例，不把历史 22 项矩阵固定成新验收要求 |

## 5. 快速转正式

不开发自动迁移工具。转正式按文件映射执行移动，修改入口引用和必要注册，然后重新构建验证。
镜像目录中的算子专属文件通常去掉 `experimental/` 前缀即可确定正式落点；实验框架本身不随算子迁移。
已有算子目录按正式分类落位，不能让正式源码反向依赖实验树。

| 实验资产 | 正式落点/动作 |
| --- | --- |
| `experimental/<op>/CMakeLists.txt`、`op_host/`、`op_kernel/` | `fla/ops/ascendc/<group>/<op>/`，保留实现和私有头 |
| `experimental/<op>/adapter/stable_<op>.cpp` | `torch_custom/fla_npu/csrc/src/stable_<op>.cpp` |
| adapter schema / implementation | 在正式 `torch_custom/fla_npu/csrc/src/stable_ops.cpp` 增加 include、`m.def` 和 `m.impl` |
| 实验包中的算子 wrapper | 接入 `torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py`，登记 `_ASCENDC_OPS`；保留签名、参数检查及 mutation 语义 |
| `experimental/tests/atk/<op>/` | `tests/atk/<op>/`，更新 DUT 为正式入口，保留用例和精度标准 |
| reference、PTA、示例和算子文档 | 随算子/测试迁移，调整引用，不改数学定义；共同依赖不能留在实验目录 |

转正式完成条件：

1. 移出实验集合，进入正式集合；正式全编包含该算子，实验全编不再包含。
2. 正式 wrapper/schema/adapter、ABI 与所需 layer 元数据检查通过。
3. 在正式包上用相同输入、reference 和精度标准重新验证；不能沿用实验包结果。
4. 遵守正式算子的准入和 CI；实验阶段不跑 CI 不代表转正式跳过门禁。

## 6. 当前状态与待实施内容

### 6.1 当前状态

| 项目 | 状态 |
| --- | --- |
| 同步 main 与已合入 #1177、解决 helper 冲突 | 已完成，基线 `c39b0eb3` |
| 按原结构下移实验专属文件，提取公共文件中的实验分支 | 待方案确认后实施 |
| 双向构建与制品隔离 | 已有实现；重构后按 §3.2 手动复核 |
| 复用通用增量安装，开发者保证实验 base | 新设计要求，当前代码未接入 |
| 自动转正式工具 | 不实施；采用 §5 的文件映射与注册步骤 |
| 当前验证 | 见 validation.md；不代表新目录或增量安装已验收 |

后续顺序为文件下移及引用调整、提取实验专属逻辑、接入通用增量能力、手动验证。
手动验收覆盖两种构建模式的单选/多选/全选和残留隔离，以及实验 base 加实验增量的正常安装、装载和恢复。
不新增实验 CI，也不要求对跨集合误装提供专属拦截保证。本次仅修订文档，未开始上述代码修改。

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
