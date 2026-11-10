# experimental 实验算子

用于仓内共同开发、尚未进入正式算子集合的实现。当前只包含从 [PR #446](https://github.com/flashserve/flash-linear-attention-npu/pull/446) 迁入的 `fused_recurrent_rwkv8`。

- [构建、安装与装载设计](../docs/architecture/experimental%20构建适配设计.md)
- [当前验证记录](validation.md)
- [RWKV8 接口和实现](fused_recurrent_rwkv8/README.md)

## 边界

1. 正式构建只包含正式算子，`--experimental` 只包含实验算子及其实验依赖。
2. 实验 `.run`、wheel、vendor、Python 包、launcher 和 dispatcher 使用独立身份；不得覆盖正式 wheel 或 OPP。
3. 实验 CI 独立构建、验证和发布状态，不依赖正式构建先完成。

experimental 不支持 overlay、运行中替换 OPP 或同进程混用正式/实验算子。使用独立 Python 环境和新进程；不维护同环境共存、pip 安装顺序或正式 overlay 的兼容矩阵。不同包名和目录的防覆盖措施保留。

## 构建与安装

先初始化 CANN 和匹配的 Python 环境：

```bash
export FLA_NPU_DISABLE_PTH=1
# 一个算子；多个算子可用 --ops=op1,op2；省略 --ops 或 --ops=all 编全部实验算子。
bash build.sh --pkg --experimental --soc=ascend910b --ops=fused_recurrent_rwkv8 -j4
bash build_out/fla_npu_experimental_linux-$(uname -m).run \
    --install-path=/absolute/empty-experimental-opp --quiet
python experimental/_framework/build.py wheel \
    --manifest build/experimental-build-manifest.json \
    --install-root /absolute/empty-experimental-opp --output /absolute/development-wheels
python -m pip install --no-deps /absolute/development-wheels/*.whl
```

`.run` 仅接受显式绝对空目录或父目录存在的新目录；拒绝正式 wheel/OPP、默认覆盖安装和 overlay。wheel 包名为 `flash-linear-attention-npu-experimental-a2/a3/a5`，Python 包为 `fla_npu_experimental`，不写正式 `fla_npu`，不安装自动启动 `.pth`。开发 wheel 不进入正式发布流程。

算子放在 `experimental/<op>/`，必须有 `CMakeLists.txt`、`op_host/`、`op_kernel/` 和 wheel 所需 adapter。`_` 开头的一级目录是框架目录。实验依赖必须闭合在实验树内；多算子发现和依赖规则有临时夹具测试。当前真实算子只有 RWKV8，新增算子需同步扩充 CI 精度覆盖。

构建参数的兼容目标、当前限制与证据见[设计 §3.1](../docs/architecture/experimental%20构建适配设计.md#31-buildsh-参数兼容边界)。不把已有构建选项全部视为已实测。

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

## CI 与复现

```bash
# A2 独立全链路；输出目录必须尚不存在。
FLA_NPU_DISABLE_PTH=1 CI_SOC=ascend910b CI_MODE=quick CI_CONTAINER_DEVICE=0 \
    python3 ci/run_experimental_ci.py --output /absolute/new-results
# 只做构建、安装、ABI 和 host 装载，不声明 NPU 精度通过。
FLA_NPU_DISABLE_PTH=1 CI_SOC=ascend950 \
    python3 ci/run_experimental_ci.py --build-only --output /absolute/new-build-results
# 使用现有 CI 镜像、挂载和设备锁。
CI_SUITE=experimental CI_SOC=ascend910b CI_MODE=full bash ci/run_ci_container.sh
```

统一入口先检查依赖，缺必要依赖立即失败，缺 ATK 只 warning。随后只构建实验 `.run` 和 wheel，在一个隔离环境中安装并 host 装载。A2 再验证 3 个受保护安装路径和 22 项装载隔离，并用 NPU 输出直接对拍 CPU reference；quick 默认/inline 各 3 条，full 各 209 条。A5 只运行构建和 host 检查。

完整结果保存在输出目录的 `summary.json`；分阶段日志、数值明细和调用后的库来源证据供定位。GitHub 以独立 `NPU CI / Experimental / A2 runtime + A5 build` 状态发布结果。正式 GDR 不再串接实验验证，两个任务没有构建先后依赖。

[ATK 工程](../tests/atk/fused_recurrent_rwkv8/README.md) 保留为备选，默认 CI 不安装、不调用 ATK。PTA/dump 复用同一 CPU reference，保留原 FP32/relative-RMSE 诊断口径和独有边界/负例；正式实验 CI 使用 FP64→FP32 与原混合容差。reverse、并发、确定性、MSS、性能以及 A5 数值不属于本门禁已覆盖范围。
