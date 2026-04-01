# Runtime API

`run(ctx)` 里最常用的是 5 个对象：`ctx.data_client`、`ctx.output`、`ctx.params`、`ctx.logger`、`ctx.run_context`。

## `ctx.data_client`

运行时输入读取入口。它读取的是当前 run 的输入目录 `input/`，不是原始 Bundle 根目录。

常用方法：

| 方法 | 用途 |
|------|------|
| `get_dataset_keys()` | 列出当前 run 可用的 dataset key |
| `get_item_count(dataset_key)` | 获取 dataset item 数量 |
| `get_layout(dataset_key)` | 获取 dataset 布局 |
| `get_frame_columns(dataset_key, index)` | 获取某个 frame item 的逻辑列集合 |
| `read_object(dataset_key, index)` | 从 `object_ndjson_lines` dataset 读取一个对象 |
| `iterate_objects(dataset_key)` | 迭代 `object_ndjson_lines` dataset |
| `read_frame(dataset_key, index, columns=None, chunk_rows=None)` | 全量读取某个 frame item 的逻辑视图 |
| `iter_frame_batches(dataset_key, index, columns=None, batch_rows=None)` | 以 Arrow record batch 流式读取某个 parquet item |
| `iter_frame_chunks(dataset_key, index, columns=None, chunk_rows=None)` | 以 DataFrame chunk 流式读取某个 parquet item |
| `iter_dataset_frame_chunks(dataset_key, columns=None, chunk_rows=None)` | 跨整个 dataset 按 item 顺序流式读取 parquet chunks |

布局相关的边界要特别注意：

- `read_object()` / `iterate_objects()` 只适用于 `object_ndjson_lines`
- `get_frame_columns()` / `read_frame()` / `iter_frame_batches()` / `iter_frame_chunks()` / `iter_dataset_frame_chunks()` 只适用于 `frame_parquet_item_dirs`
- 原始 parquet 路径访问已经取消，算法必须通过 `DataClient` 读取 frame 数据
- 不要在没看过 `get_layout()` 的情况下假设输入数据一定是 frame/parquet

关于全量读取和流式读取：

- `read_frame()` 适合小数据或必须一次性拿完整逻辑视图的算法
- `iter_frame_batches()` 适合能直接处理 Arrow batch、希望减少 pandas 转换开销的算法
- 新的 `iter_frame_chunks()` / `iter_dataset_frame_chunks()` 适合高频大数据和可增量计算的算法
- 这些 frame 读取方法依赖 `pyarrow`；如果算法运行环境没有它，SDK 会在运行时直接报错
- `read_frame()` 内部仍然是按 batch / chunk 处理后再拼接，不需要先把 raw 输入一次性全部读进内存
- `chunk_rows` / `batch_rows` 都是输出窗口，不是 raw 输入窗口
- `chunk_rows` 是可选提示，不传时 SDK 使用默认输出 chunk 大小
- 读取是按需拉取，不需要算法声明“读取速度”
- `iter_frame_chunks()` 内部会做一个很小的透明预读，尽量把 IO / 解码和算法计算重叠起来
- 如果 `input/drs.json` 某个 dataset 带有 `sampling.deliveryHz`，`DataClient` 会在运行时先输出降频后的逻辑视图；当前实现会优先从 `period_us`、其次从 `ts_us` 间隔自动识别 raw core 频率，只支持降频，不支持高于原始频率的重采样

关于算法侧如何设置流式输出窗口：

- 窗口由算法代码显式传给 `chunk_rows` 或 `batch_rows`
- 推荐把窗口配置放到 `ctx.params`，再由算法在运行时读取
- `chunk_rows` / `batch_rows` 的单位都是“输出行数”
- 如果输入启用了 `deliveryHz` 降频，这里的“输出行数”指的是降频后的行数
- `read_frame(..., chunk_rows=...)` 里的 `chunk_rows` 只影响内部读取批次，不影响最终返回值仍然是整张 `DataFrame`

推荐参数名：

- `chunkRows`: DataFrame chunk 窗口
- `batchRows`: Arrow batch 窗口

推荐写法：

```python
chunk_rows = int(ctx.params.get("chunkRows", 50_000))

for chunk in ctx.data_client.iter_frame_chunks(
    "samples_core_stage_5826",
    0,
    columns=["ts_us", "value"],
    chunk_rows=chunk_rows,
):
    process(chunk)
```

如果是跨整个 dataset 逐 item 流式处理：

```python
chunk_rows = int(ctx.params.get("chunkRows", 50_000))

for item_index, chunk in ctx.data_client.iter_dataset_frame_chunks(
    "samples_core_stage_5826",
    columns=["ts_us", "value"],
    chunk_rows=chunk_rows,
):
    process(item_index, chunk)
```

如果算法直接消费 Arrow batch：

```python
batch_rows = int(ctx.params.get("batchRows", 50_000))

for batch in ctx.data_client.iter_frame_batches(
    "samples_core_stage_5826",
    0,
    columns=["ts_us", "value"],
    batch_rows=batch_rows,
):
    process(batch)
```

如果算法最终还是要拿整张表，但希望控制内部读取批次：

```python
chunk_rows = int(ctx.params.get("chunkRows", 50_000))

frame = ctx.data_client.read_frame(
    "samples_core_stage_5826",
    0,
    columns=["ts_us", "deviceId", "signal", "value", "period_us"],
    chunk_rows=chunk_rows,
)
```

示例：

```python
dataset_key = "pressure"

for item_index, chunk in ctx.data_client.iter_dataset_frame_chunks(
    dataset_key,
    columns=["ts_us", "treatingPressure"],
    chunk_rows=50_000,
):
    # 在这里做增量计算，避免把所有 parquet 一次性拼进内存
    process(item_index, chunk)
```

全量读取示例：

```python
dataset_key = "samples_core_stage_5826"
frame = ctx.data_client.read_frame(
    dataset_key,
    0,
    columns=["ts_us", "deviceId", "signal", "value", "period_us"],
)
```

`get_frame_columns()` 返回的是 `DataClient` 暴露给算法的逻辑列集合，不只是 parquet 物理列；例如 item 的 `resolutionParams` 也会作为可读取列出现。

推荐先 `get_frame_columns()` 再决定 `columns=...`，尤其是算法需要兼容不同 bundle 的时间列或信号列命名时。

运行前提：

- `ctx.data_client` 依赖的是已经规范化好的 `input/ds.json`
- 在 SDK 自己的执行链里，这个文件会在 materialize 阶段补齐当前 run 需要的 `layout`
- 如果你在 SDK 外部自行构造运行环境，也需要先把传给运行时的 `input/ds.json` 规范化成可直接消费的形式

## `ctx.output`

运行时输出写入入口。真实上下文属性名就是 `output`。

常用属性和方法：

| 项目 | 说明 |
|------|------|
| `dir` | 固定输出目录 `run/output/artifacts` |
| `path(relative_path)` | 返回并校验 `artifacts/` 下路径 |
| `write_scalar()` | 写标量结果，不落文件 |
| `write_text()` | 写文本文件并登记 |
| `write_json()` | 写 JSON 文件并登记 |
| `write_blob()` / `write_bytes()` | 写二进制文件并登记 |
| `write_file()` | 复制或登记已有文件 |
| `register_file()` | 给已写入 `ctx.output.dir` 的文件补登记 |
| `write_dataframe_csv()` | 写 DataFrame 为 CSV |
| `write_figure_png()` | 写 matplotlib figure 为 PNG |

最小示例：

```python
def run(ctx) -> None:
    out = ctx.output
    out.write_scalar("status", "ok", dataset_key="status")
    out.write_json("summary", {"ok": True}, dataset_key="summary")
```

规则：

- 不要自己实例化 `OutputClient`
- 不要手工写 `output/manifest.json`
- 路径必须位于 `ctx.output.dir` 下
- 如果传 `dataset_key`、`owner`、`dims`、`item_key`，要和 `OutputContract` 保持一致

## `ctx.params`

- 类型：`dict[str, Any]`
- 键名来自 `params.json`
- 算法运行时按普通 dict 访问

示例：

```python
threshold = float(ctx.params.get("threshold", 0.5))
dataset_key = ctx.params.get("datasetKey", "wells")
```

不要把 `ctx.params` 当成 Pydantic 模型对象来访问属性。

## `ctx.logger`

标准 `logging.Logger`，直接用于记录运行日志。

```python
ctx.logger.info("start")
ctx.logger.warning("something unusual")
```

## `ctx.run_context`

运行上下文元数据字典。具体内容取决于当前运行创建方式，一般用于：

- owner 平台信息
- run 级别元数据
- 调试信息
