# hma_connector.py 实现总结

`ucm/integration/vllm/hma_connector.py` 主要实现 vLLM HMA 场景下的 UCM KV cache connector。当前重点是 `UCMFAWAConnector`：它把可复用 prefix KV cache 拆成 full-attention store 和 window-attention store 两套语义，同时支持普通 GPU KV layout 和 Ascend DeepSeek V4 风格的复杂 tensor block 映射。

## 核心目标

- 对 full-attention group，按每个可复用 canonical prefix block 存取 KV。
- 对 window-attention/state group，按每个 prefix boundary 存 tail row，但 external hit load 时只加载最后一个命中 boundary。
- 对 scheduler 侧只生成逻辑 block/segment plan；worker 侧用真实 tensor pointer 执行 store load/dump。
- 保持 FA/WA 两个 store 语义隔离：同一个 canonical key 可以同时存在于两个 store，但 row 内容和 byte size 可以不同。
- 同时覆盖 GPU tensor layout、Ascend tensor layout、chunk prefill partial allocation、TP4/MLA rank0-only dump 等场景。

## 关键数据结构

`KVCacheSegment` 表示一个 KV cache segment：

- `block_id`: vLLM/HMA tensor block id。
- `offset`: 在 tensor block 内的逻辑 token offset。
- `length`: segment 覆盖的逻辑 token 数。

`KVCacheGroupLayout` 负责把一个 vLLM KV cache group 展平成 store 可用的指针布局。它会记录：

- `base_ptrs`
- `block_strides`
- `token_strides`
- `tensor_size_lists`
- `tensor_size_per_token_lists`
- `view_tensor_block_sizes`
- `view_tensors`

它支持这些 tensor 形态：

- `[2, num_blocks, block_size, num_head, head_dim]`
- `[num_blocks, 2, block_size, ...]`
- `[num_blocks, block_size, num_head, head_dim]`
- `[num_blocks, block_size, head_dim]`

后续 load/dump 都通过 `extract_segment_addrs()`、`segment_tensor_size_list()`、`extract_segment_tensor_views()` 等方法，从 logical segment 推导真实 HBM/NPU/GPU tensor pointer。

`FAWARequestMeta` 是 scheduler 侧 per-request 状态，核心字段包括：

- `ucm_block_ids`: canonical hash blocks。
- `hbm_hit_block_num`: HBM prefix cache hit block 数。
- `total_hit_block_num`: HBM + external UCM 总命中 block 数。
- `num_token_ids`: request token 总数。
- `token_processed`: scheduler token 进度。
- `store_block_cursor`: 已持久化到外存的 canonical block cursor。
- `group_block_ids`: canonical block idx 到 per-group `KVCacheSegment` row 的映射。
- `allocated_group_block_ids`: vLLM 已分配的 per-group HMA block ids。

## Block 命名边界

当前实现把 block size 收敛成三类：

- `hash_block_size`: connector 生成 canonical hash、external lookup、FA/WA store key 的粒度。
- `group_token_block_sizes`: 每个 KV cache group 的逻辑 token block 粒度，用来决定一个 canonical boundary 需要哪些 group block。
- `group_tensor_block_sizes`: 每个 KV cache group 在 HBM tensor/page 上的 block 跨度，用来把 group logical block index 映射到 tensor block id 和 block 内 offset。

`KVCacheGroupLayout.view_tensor_block_sizes` 是更底层的 tensor view shape 信息。它只用于把 `KVCacheSegment(offset, length)` 换算成某个具体 tensor view 的 pointer offset 和 copy byte size，不参与 hash key 语义。

## Store 语义

FAWA connector 使用两个 store：

- `fa_store`: full-attention groups，保存每个可复用 prefix block。
- `wa_store`: window-attention/state groups，保存每个 prefix boundary 所需 tail row。

两者使用同一套 canonical key 命名规则：

```python
_block_key(canonical_hash) = request_hasher((b"fawa", canonical_hash))
```

但两者 row 内容不同。FA row 是 full-attention groups 的 segment；WA row 是 window/state groups 的 tail segment。因此同一个 key 可以同时存在于 FA 和 WA store，且 byte size 不必相同。

## Scheduler 侧流程

### 1. 查 external hit

`get_num_new_matched_tokens()` 做这些事：

1. 校验 `num_computed_tokens` 必须按 `hash_block_size` 对齐。
2. 使用 `generate_hash(self.hash_block_size, request.all_token_ids, self._seed)` 生成 canonical hashes。
3. 从 HBM hit 之后开始构造 external keys。
4. 分别查询 `fa_store.lookup_on_prefix()` 和 `wa_store.lookup_on_prefix()`。
5. 取 FA/WA 连续命中的最小值作为 external hit blocks。
6. 记录 `FAWARequestMeta`。

如果命中覆盖整个请求，会保留既有 full-hit 修正逻辑：

```python
if num_total_hit_tokens == request.num_tokens:
    external_hit_tokens -= 1
```

### 2. 记录 HMA allocation

`update_state_after_alloc()` 在新请求 allocation 后调用。它把 vLLM `KVCacheBlocks` 转成 `allocated_group_block_ids`，然后调用 `_record_ready_group_block_ids()` 尝试生成可用 canonical rows。

chunk prefill 场景下，后续 allocation 不一定再走 `update_state_after_alloc()`，而是通过 `build_connector_meta()` 中的 `scheduled_cached_reqs.new_block_ids` 进入 `_record_allocated_group_block_ids()`。

### 3. 生成 canonical group row

`_record_ready_group_block_ids()` 从第一个未记录的 canonical block 开始，逐个调用 `_try_select_group_block_ids()`。

`_try_select_group_block_ids()` 对每个 KV group：

1. 根据 computed end token 计算该 canonical boundary 需要哪些 group block。
2. 将 group logical block index 映射到 tensor allocation index。
3. 将 tensor block id 转成 `KVCacheSegment(block_id, offset, length)`。
4. 对早期 external-hit WA tail，允许 null block 转 scratch segment。
5. 如果某个 group allocation 不完整，则返回 `None`，停止记录后续 canonical blocks。

这保证 chunk prefill partial allocation 不会提前 dump 不完整 row。

### 4. 构造 dispatch metadata

`build_connector_meta()` 输出 `UCMFAWAConnectorMetadata`。

对新请求：

- 如果有 external hit，需要生成 load plan。
- dump 从 `store_block_cursor` 开始，只 dump 已完成且 row 已记录的 contiguous canonical blocks。

对 cached/chunk prefill 请求：

- 先根据 `new_block_ids` append 或 replace allocation。
- `resumed_req_ids` 中的请求使用 replace；否则 append。
- 再生成 dump plan。

关键点：dump 进度由 `store_block_cursor` 控制，而不能只看 `token_processed`。partial allocation 可能已经让 `token_processed` 到达 request length，但 row 后续才补齐；补齐后仍必须允许 dump 剩余完整 rows。

## Worker 侧流程

worker 先通过 `register_kv_caches()` 注册真实 KV tensors。该函数会：

1. 按 vLLM KV cache group 拆分 registered cache。
2. 为每个 group 创建 `KVCacheGroupLayout`。
3. 根据 group layout 创建 `fa_store` 和 `wa_store`。
4. worker store config 中写入 `tensor_size_list`、`shard_size`、`block_size` 等真实 row byte 信息。

### Load

`start_load_kv()` 从 scheduler metadata 读取 load plan：

- FA load：对所有 external-hit canonical blocks 加载 FA rows。
- WA load：只加载最后一个 external-hit boundary。
- 如果 WA 早期 tail 缺失，可以用 scratch target，避免要求每个 external-hit block 都有完整 WA load target。

实际提交给 store 的是 pointer matrix：

```python
store.load_data(keys, shard_indexs, ptrs)
```

### Save

`wait_for_save()` 收集所有 request 的 dump plan：

- 没有 dump keys 时直接返回。
- `tp_rank != 0` 时直接返回，当前 FAWA/MLA TP4 路径只允许 rank0 dump。
- rank0 对同一批 canonical keys 分别向 FA 和 WA store dump。
- FA 和 WA 使用不同的 selected rows 和 pointer matrix。

实际提交给 store 的是：

```python
store.dump_data(keys, shard_indexs, ptrs, event_handle)
```

## GPU Layout 适配

GPU 路径使用基础 `UCMFAWAConnector`，`block_span_layout` 为 `None`。

关键适配在 `KVCacheGroupLayout`：

- 对 `[num_blocks, 2, block_size, ...]`，先拆 K/V 轴，再读取 token block size。
- 对 `[2, num_blocks, block_size, ...]`，按 legacy 方式拆成 K/V 两个 view。
- 对 mixed 3D tensor view block sizes，不假设一个 group 内所有 tensor view 的 block 维度相同，而是通过 `view_tensor_block_sizes` 和 `token_strides` 做 segment 地址换算。

GPU 中 logical block 到 tensor block 的映射通常是：

```python
tensor_idx = group_block_idx * group_token_block_size // group_tensor_block_size
offset = (group_block_idx % token_blocks_per_tensor_block) * group_token_block_size
```

## Ascend Layout 适配

Ascend 使用 `UCMAscendFAWAConnector` 和 `FAWABlockSpanLayout`。

Ascend 检测逻辑要求 KV config 中包含 Ascend group，并包含关键 spec：

- `Compress4AttentionSpec`
- `C4IndexerSpec`
- `Compress128AttentionSpec`

Ascend canonical hash block size 固定为 512 tokens。

部分 compressed FA group 的 logical canonical segment 会落在更大的 tensor block 中：

- `Compress4AttentionSpec`: tensor block 512 tokens。
- `C4IndexerSpec`: tensor block 4096 tokens，8 个 512-token canonical segments 可映射到同一个 tensor block 的不同 offset。
- `Compress128AttentionSpec`: tensor block 16384 tokens，32 个 512-token canonical segments 可映射到同一个 tensor block 的不同 offset。

Ascend KV tensor 注册形态是：

```text
[num_blocks, block_size, num_head, head_dim]
```

同时 vllm-ascend 可能把一层的多个 cache tensor 打包成 tuple。`FAWABlockSpanLayout` 会按 group/layer/spec 顺序建立 tensor index 映射，其中 `C4IndexerSpec` 每层占两个 tensor。

Ascend window/state tail 还会做额外 trim：

- SWA group 保持普通 sliding-window block。
- C4 state group 使用 `window_tokens - compress_ratio` 后的 tail。
- C128 state group 在 `window_tokens <= compress_ratio` 时 tail blocks 可以为 0。
- `_trim_window_segment()` 会把 segment offset/length 修正到真实 tail 范围。

## Chunk Prefill 行为

chunk prefill 的难点是 allocation 可能分批到达。例如第一步只给 FA group block，WA/state group 还没分配齐。

当前实现的保护规则：

- `new_block_ids is None` 时不更新 allocation。
- cached request 非 resumed 时 append allocation。
- resumed request 时 replace allocation。
- 只有 canonical row 的所有 group segment 都可推导时，才记录到 `group_block_ids`。
- dump 只从 `store_block_cursor` 开始，取 contiguous completed rows。
- allocation 补齐后，即使 `token_processed` 已经到 request length，也仍然可以根据 `store_block_cursor` dump 剩余 rows。

这避免了提前 dump 不完整 KV row，也避免了补齐 allocation 后漏 dump。

## 错误处理和边界

- external-hit load plan 必须完整。如果 `total_hit_block_num` 范围内缺少 `group_block_ids`，`_make_dispatch_meta()` 会抛出 load plan missing 错误。
- `KVCacheGroupLayout` 不支持未知 tensor dim，会直接报错。
- 如果 store 未初始化，load/dump 会报明确错误。
- `_extract_group_addrs()` 对空 pointer row 报错。
- null HBM block 只在允许 scratch 的 WA tail 场景下可接受。
- request 完成后，`request_finished_all_groups()` 清理 `requests_meta`，并返回 `(False, None)`，表示 connector 不接管 async block release。

## 当前验证重点

核心验证分三类：

- `test/test_hma_connector_chunk_prefill.py`: 单元级模拟，覆盖 chunk prefill、Ascend mapping、tensor layout、WA final-boundary load 等。
- `test/test_hma_connector_ascend_tp4_e2e.py`: Ascend TP4 端到端模拟，使用当前 accelerator 上的 HBM tensor，实际走 worker `start_load_kv()` / `wait_for_save()`。
- `test/test_hma_connector_gpu_tp4_e2e.py`: GPU TP4 端到端模拟，覆盖 GPU 4D K/V axis、mixed 3D tensor block size、FA/WA store byte size 不同等。

推荐验证命令：

```bash
python3 -m py_compile \
  ucm/integration/vllm/hma_connector.py \
  ucm/integration/vllm/ucm_connector.py \
  test/test_hma_connector_chunk_prefill.py \
  test/test_hma_connector_ascend_tp4_e2e.py \
  test/test_hma_connector_gpu_tp4_e2e.py

python3 -m pytest \
  test/test_hma_connector_chunk_prefill.py \
  test/test_hma_connector_ascend_tp4_e2e.py \
  test/test_hma_connector_gpu_tp4_e2e.py
```
