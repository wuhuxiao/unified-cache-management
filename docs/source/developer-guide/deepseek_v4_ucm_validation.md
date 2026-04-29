# DeepSeek V4 UCM Connector Validation

本文档整理 DeepSeek V4 UCM connector 的测试运行流程。目标是先用 capture 数据离线验证 packed KV 语义，再用 local stub 隔离 UCM 后端，最后只运行少量真实 UCM 端到端验证。

## 适用场景

该流程用于验证 DeepSeek V4 hybrid KV cache manager 下的 UCM packed connector，重点覆盖：

- 256-token prefix key 与 packed tensor list 的映射。
- scheduler metadata 与 worker tensor payload 的一致性。
- HBM block 到 packed row 的 group/block 映射。
- TP rank payload 是否一致，是否可以 rank0-only dump。
- UCM CacheStore dump/load 后 bytes 是否完整一致。
- 真实 vLLM second request 是否 external hit 且输出对齐。

当前 connector 使用两个逻辑 store：

- `dsv4_packed`：每个 256-token key 保存完整 packed row，用于恢复 prefix 边界 tail state。
- `dsv4_group0`：每个 256-token key 只保存 group0 KV，用于 multi-block hit 时读取前 `N-1` 个 block。

这样 multi-block external hit 不会为每个命中 block 重复读取 SWA/C4A/C128 tail。最后一个命中 block 仍读取完整 packed row，用来恢复边界处 state。

当前验证通过的 packed tail 配置为：

```python
GROUP_TAIL_BLOCKS = (None, 2, 2, 2, 16)
```

对应每个 256-token packed row 的 group block 数：

```text
[1, 2, 2, 2, 16]
```

## 前置条件

默认工作目录：

```bash
cd /vllm-workspace/unified-cache-management
```

需要已有 DeepSeek V4 测试脚本和 UCM 配置：

```text
/vllm-workspace/offline_inference.py
/vllm-workspace/deepseek_v4_ucm_cache_config.yaml
```

需要 4 张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 基础静态检查

每次改动 connector 或 validator 后先运行：

```bash
python3 -m py_compile \
  ucm/integration/vllm/ucm_connector.py \
  scripts/verify_deepseek_v4_ucm_capture.py

git diff --check -- \
  ucm/integration/vllm/ucm_connector.py \
  scripts/verify_deepseek_v4_ucm_capture.py
```

预期：两个命令均无输出或 exit code 为 0。

## 离线 Capture 校验

已有 capture 数据时，先跑离线校验，不启动完整 vLLM：

```bash
python3 scripts/verify_deepseek_v4_ucm_capture.py \
  /vllm-workspace/deepseek_v4_ucm_capture_conservative3
```

预期输出包含：

```text
Actual captured UCM load_after matches rank0 dump_before.
Offline DeepSeek V4 packed layout checks passed.
TP rank payloads are identical; single-shard rank0 dump is valid.
```

该步骤会检查：

- `lookup / after_alloc / dispatch_dump / dispatch_load` scheduler metadata。
- `dump_before / load_before / load_after` worker payload。
- `dispatch_dump` 与 `dispatch_load` 的 packed key 是否一致。
- worker tensor 展开顺序是否等于 store `tensor_size_list` 顺序。
- `sum(tensor_size_list)` 是否等于 `shard_size`。
- `load_after` 是否逐 tensor 匹配 rank0 `dump_before`。
- 同 key 下 TP rank0/1/2/3 payload 是否一致。

如果 TP rank payload 一致，connector 可以使用单 shard 语义：

```text
shard_indexs = [0] * len(keys)
dump only rank0
load all ranks from the same packed value
```

如果 TP rank payload 不一致，需要改为 rank-aware shard 存储。

## CacheStore Replay 校验

离线 metadata 通过后，用 capture payload 构造最小 UCM CacheStore replay，避免完整 vLLM 干扰。

单进程 replay：

```bash
python3 scripts/verify_deepseek_v4_ucm_capture.py \
  /vllm-workspace/deepseek_v4_ucm_capture_conservative3 \
  --run-ucm-store
```

4-rank replay：

```bash
python3 scripts/verify_deepseek_v4_ucm_capture.py \
  /vllm-workspace/deepseek_v4_ucm_capture_conservative3 \
  --run-ucm-store-mp \
  --world-size 4
```

预期输出分别包含：

```text
Minimal UCM CacheStore dump/load check passed.
Multiprocess UCM CacheStore dump/load check passed.
```

注意：当前 packed shard 约 26 MB，CacheStore 要求至少 1024 个 shard buffer，因此 replay 配置使用 `cache_buffer_capacity_gb = 32`。

## Local Stub 端到端隔离

CacheStore replay 通过后，先用 local stub 运行一次 E2E。该模式使用本地文件保存 packed tensor list，用来隔离 UCM 后端问题。

```bash
rm -rf /vllm-workspace/deepseek_v4_ucm_stub_conservative3

env \
  PYTHONPATH=/vllm-workspace/unified-cache-management \
  ENABLE_UCM=1 \
  UCM_CONFIG_FILE=/vllm-workspace/deepseek_v4_ucm_cache_config.yaml \
  UCM_DEEPSEEK_V4_LOCAL_STUB=1 \
  UCM_DEEPSEEK_V4_LOCAL_STUB_DIR=/vllm-workspace/deepseek_v4_ucm_stub_conservative3 \
  ENABLE_PREFIX_CACHING=0 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python3 /vllm-workspace/offline_inference.py
```

预期：

```text
first request done.
hit external: 1
second request done.
```

并且 first request 与 second request 的 generated text 完全一致。

如果 local stub 失败，问题优先归因于 packed layout 或 HBM block selection，而不是 UCM 后端。

## HBM Prefix Cache Baseline

当需要确认模型和 prompt 本身稳定时，运行 HBM prefix cache baseline：

```bash
env \
  PYTHONPATH=/vllm-workspace/unified-cache-management \
  ENABLE_UCM=0 \
  ENABLE_PREFIX_CACHING=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python3 /vllm-workspace/offline_inference.py
```

预期：两次请求输出一致。该 baseline 只用于确认 vLLM HBM prefix cache 路径正常。

## 真实 UCM 端到端验证

local stub 通过后，再运行真实 UCM，并开启 capture：

```bash
rm -rf /vllm-workspace/deepseek_v4_ucm_capture_conservative3
mkdir -p /vllm-workspace/deepseek_v4_ucm_capture_conservative3

env \
  PYTHONPATH=/vllm-workspace/unified-cache-management \
  ENABLE_UCM=1 \
  UCM_CONFIG_FILE=/vllm-workspace/deepseek_v4_ucm_cache_config.yaml \
  ENABLE_PREFIX_CACHING=0 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  UCM_DEEPSEEK_V4_CAPTURE=1 \
  UCM_DEEPSEEK_V4_CAPTURE_TENSORS=1 \
  UCM_DEEPSEEK_V4_CAPTURE_DIR=/vllm-workspace/deepseek_v4_ucm_capture_conservative3 \
  python3 /vllm-workspace/offline_inference.py
```

预期：

```text
DeepSeek V4 request_id: ..., hit hbm: 0, hit external: 0
first request done.
DeepSeek V4 request_id: ..., hit hbm: 0, hit external: 1
second request done.
```

并且 second request 输出与 first request 完全一致。

真实 UCM 运行结束后，立即对新 capture 跑离线 validator：

```bash
python3 scripts/verify_deepseek_v4_ucm_capture.py \
  /vllm-workspace/deepseek_v4_ucm_capture_conservative3
```

再跑 CacheStore replay：

```bash
python3 scripts/verify_deepseek_v4_ucm_capture.py \
  /vllm-workspace/deepseek_v4_ucm_capture_conservative3 \
  --run-ucm-store

python3 scripts/verify_deepseek_v4_ucm_capture.py \
  /vllm-workspace/deepseek_v4_ucm_capture_conservative3 \
  --run-ucm-store-mp \
  --world-size 4
```

## 常用环境变量

| Variable | Purpose |
| --- | --- |
| `ENABLE_UCM=1` | 启用 UCM connector。 |
| `ENABLE_PREFIX_CACHING=0` | 关闭 vLLM HBM prefix cache，避免命中路径混淆。 |
| `UCM_CONFIG_FILE` | 指向 UCM cache config yaml。 |
| `UCM_DEEPSEEK_V4_LOCAL_STUB=1` | 使用本地 stub 存取 packed tensor list。 |
| `UCM_DEEPSEEK_V4_LOCAL_STUB_DIR` | 本地 stub 文件目录。 |
| `UCM_DEEPSEEK_V4_CAPTURE=1` | 开启 scheduler/worker capture。 |
| `UCM_DEEPSEEK_V4_CAPTURE_TENSORS=1` | capture worker tensor payload。 |
| `UCM_DEEPSEEK_V4_CAPTURE_DIR` | capture 输出目录。 |
| `UCM_DEEPSEEK_V4_TIMING=1` | 打印 DeepSeek V4 dump/load submit 和 wait 分段耗时。 |

## 1k 纯 UCM 命中 Smoke Test

用于快速确认关闭 HBM prefix cache 后，纯 UCM second request 命中且时延低于完全重算。

关键配置：

```text
ENABLE_UCM=1
ENABLE_PREFIX_CACHING=0
UCM_DEEPSEEK_V4_LOCAL_STUB=0
UCM_DEEPSEEK_V4_CAPTURE=0
UCM_DEEPSEEK_V4_TIMING=1
CUDA_VISIBLE_DEVICES=0,1,2,3
```

当前 1007-token prompt 验证结果：

```text
first request:  hit external: 0, 0.78s
second request: hit external: 3, 0.53s
group0 load wait max:  ~0.0005s
packed load wait max:  ~0.0028s
```

同 prompt 在 `ENABLE_UCM=0` 且 `ENABLE_PREFIX_CACHING=0` 下完全重算约 `0.77s`。

## 失败判断

如果出现 null HBM block，例如：

```text
DeepSeek V4 packed group X block index Y maps to a null HBM block
```

说明 packed tail 选择超出了 external-load path 实际分配的 HBM block 范围。不要简单跳过 null block，因为 CacheStore row 的 `tensor_size_list` 是固定的，应重新确认 group tail。

如果 `load_after` 与 `dump_before` 不一致，但 local stub 通过，问题优先归因于 UCM raw pointer write 或 CacheStore tensor list/shard 语义。

如果 `load_after` 与 `dump_before` 一致，但 E2E 输出不一致，问题优先归因于 packed tensor 集合不完整，常见原因是缺少 C4A/C128 state tail。

如果 TP rank payload 不一致，当前 rank0-only dump 方案不成立，需要改为按 TP rank shard 存储。

## 当前通过结论

在 `GROUP_TAIL_BLOCKS = (None, 2, 2, 2, 16)` 下，当前验证结果为：

- local stub E2E 通过。
- real UCM E2E 通过。
- offline inference second request external hit 为 1。
- 1k smoke test second request external hit 为 3。
- first/second generated text 完全一致。
- `load_after` byte-level 匹配 rank0 `dump_before`。
- TP rank payloads identical，single-shard rank0 dump 有效。
- single-process CacheStore replay 通过。
- 4-rank multiprocess CacheStore replay 通过。
