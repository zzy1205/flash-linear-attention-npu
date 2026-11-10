# experimental 实验算子

用于仓内共同开发、尚未进入正式算子集合的实现。当前只包含从 [PR #446](https://github.com/flashserve/flash-linear-attention-npu/pull/446) 迁入的 `fused_recurrent_rwkv8`。

- [构建、安装与装载设计](../docs/architecture/experimental%20构建适配设计.md)
- [当前验证记录](validation.md)
- [RWKV8 接口和实现](fused_recurrent_rwkv8/README.md)

## 方案状态

2026-11-12 已同步包含 #1177 的 main `c39b0eb3`。本 README 仍描述当前代码；
[目录收敛与转正式方案](../docs/architecture/experimental%20构建适配设计.md) 是待确认设计，
目标为保留现有子目录结构、整体下移实验专属文件，不新增 `promote.py`。
新增要求是正式/实验编包双向严格隔离，仅显式 `--experimental` 启用实验模式；增量安装沿用通用机制，不额外禁止或做实验专属处理，
由开发者保证实验增量安装前的 base wheel 是实验包。目录迁移和增量接入尚未实施。
**以下命令和限制描述当前代码；其中禁止 overlay、只装内嵌 OPP 的限制是待调整项。**

## 当前实现边界

1. 正式构建只包含正式算子，`--experimental` 只包含实验算子及其实验依赖。
2. 实验 `.run`、wheel、vendor、Python 包、launcher 和 dispatcher 使用独立身份；不得覆盖正式 wheel 或 OPP。
3. experimental 不跑 CI；构建、精度和隔离用例仅供手动验证。

`build.sh` 未传 `--experimental` 时，省略 `--ops` 或使用 `--ops=all` 都只编正式算子。
显式传入 `--experimental` 后，全选也只编实验算子。当前没有独立的 `--all` 参数。

experimental 不支持 overlay、运行中替换 OPP 或同进程混用正式/实验算子。使用独立 Python 环境和新进程；不维护同环境共存、pip 安装顺序或正式 overlay 的兼容矩阵。不同包名和目录的防覆盖措施保留。

## 构建与安装

先初始化 CANN 和匹配的 Python 环境：

```bash
# 关闭正式包的启动注入。
export FLA_NPU_DISABLE_PTH=1
# 编译指定实验算子；多选用 --ops=op1,op2，省略或 all 编全部实验算子。
bash build.sh --pkg --experimental --soc=ascend910b --ops=fused_recurrent_rwkv8 -j4
# 将实验 run 安装到独立空目录。
bash build_out/fla_npu_experimental_linux-$(uname -m).run \
    --install-path=/absolute/empty-experimental-opp --quiet
# 将匹配的实验 launcher 和已安装 OPP 打成独立 wheel。
python experimental/_framework/build.py wheel \
    --manifest build/experimental-build-manifest.json \
    --install-root /absolute/empty-experimental-opp --output /absolute/development-wheels
# 把开发 wheel 安装到当前实验 Python 环境。
python -m pip install --no-deps /absolute/development-wheels/*.whl
```

`.run` 仅接受显式绝对空目录或父目录存在的新目录；拒绝正式 wheel/OPP、默认覆盖安装和 overlay。wheel 包名为 `flash-linear-attention-npu-experimental-a2/a3/a5`，Python 包为 `fla_npu_experimental`，不写正式 `fla_npu`，不安装自动启动 `.pth`。开发 wheel 不进入正式发布流程。

算子放在 `experimental/<op>/`，必须有 `CMakeLists.txt`、`op_host/`、`op_kernel/` 和 wheel 所需 adapter。`_` 开头的一级目录是框架目录。实验依赖必须闭合在实验树内；多算子发现和依赖规则有临时夹具测试。当前真实算子只有 RWKV8，新增算子需同步补充手动精度用例。

构建参数的支持范围见[设计 §3.1](../docs/architecture/experimental%20构建适配设计.md#31-buildsh-参数兼容边界)。不把已有构建选项全部视为已实测。

## 运行

在安装实验 wheel 的环境中，先初始化 CANN，再启用 wheel 内的实验 vendor：

```bash
export FLA_NPU_DISABLE_PTH=1
source /absolute/site-packages/fla_npu_experimental/opp/vendors/fla_npu_experimental_transformer/bin/set_env.bash
python - <<'PY'
from fla_npu_experimental import fused_recurrent_rwkv8  # 先于 torch_npu/acl
import torch
import torch_npu
# 构造目标设备上的输入，再调用 wrapper。
PY
```

实际装入 `fla_npu_experimental/libfla_npu_experimental_stable.so` 和内嵌 vendor 的 `op_api/lib/libcust_opapi.so`，注册 `torch.ops.fla_npu_experimental_stable`。`FLA_NPU_EXPERIMENTAL_OPP_PATH` 必须指向本 wheel 的 `opp`；不能借外部 `.run` 更新内嵌 OPP。更新时重建匹配 wheel，重新启动进程。

运行时保留包身份、ABI、算子集合、Python/launcher 指纹、路径和进程混用校验，以及加载后的环境/文件变更检查。RWKV8 仍返回 `(o, s, sa)`，`chunk_len` 范围为 `[1,4294967295]`。

## 手动验证

experimental 不接入 CI，不提供评论触发、独立 job 或必需检查状态。以下用例仅在开发者显式执行时运行。

在仓库根目录初始化 CANN 和 Python 环境后执行；构建依赖通过正常构建流程准备，离线环境需预先准备 `third_party/`：

```bash
# 关闭正式包启动注入，避免污染实验进程。
export FLA_NPU_DISABLE_PTH=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
# A2：编译并独立安装 run/wheel，检查装载隔离，再用默认/inline 各 3 条对拍 CPU reference。
python3 experimental/_tests/validate.py --soc ascend910b --mode quick --device 0 -j4 --output /absolute/new-results
# A5：只检查构建、独立安装、ABI 和 host 装载，不执行 NPU 精度。
python3 experimental/_tests/validate.py --soc ascend950 --build-only -j4 --output /absolute/new-build-results
# 运行构建/安装规则和验证工具的本地单元测试，不启动 NPU 构建或精度任务。
python -m unittest discover -s experimental/_tests -p 'test_*.py' -b
# 运行 Python 装载隔离的本地单元测试。
python -m unittest discover -s experimental/_python/tests -b
```

输出目录必须尚不存在。`--mode full` 使用全部 209 条精度用例，默认队列与 inline 各运行一次。
统一入口在构建前检查依赖；缺必要依赖立即失败，缺 ATK 只 warning。A2 包含 3 项安装拒绝和
22 项装载隔离。结果写入指定目录的 `summary.json`，同时保留阶段日志、数值明细和库来源记录。
该入口不读取 CI 环境参数，不调用 CI 容器或缓存脚本，也不上传报告或发布 GitHub 状态。

已安装匹配实验 wheel 时，可按 [ATK 工程说明](../tests/atk/fused_recurrent_rwkv8/README.md#手动独立对拍)
直接运行 `experimental/_tests/accuracy.py`。ATK 保留为手工复验备选；独立对拍不依赖 ATK。
PTA/dump 与独立对拍共用 CPU reference；独立对拍沿用 FP64→FP32 和原混合容差。
reverse、并发、确定性、MSS、性能以及 A5 数值不在当前已验证范围内。
