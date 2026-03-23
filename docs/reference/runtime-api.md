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
| `read_object(dataset_key, index)` | 从 `object_ndjson_lines` dataset 读取一个对象 |
| `iterate_objects(dataset_key)` | 迭代 `object_ndjson_lines` dataset |
| `get_parquet_dir(dataset_key, index)` | 获取 `frame_parquet_item_dirs` item 目录 |
| `get_parquet_files(dataset_key, index)` | 获取某个 parquet item 下的 parquet 文件 |

布局相关的边界要特别注意：

- `read_object()` / `iterate_objects()` 只适用于 `object_ndjson_lines`
- `get_parquet_dir()` / `get_parquet_files()` 只适用于 `frame_parquet_item_dirs`
- 不要在没看过 `get_layout()` 的情况下假设输入数据一定是 parquet

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
