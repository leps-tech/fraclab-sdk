# Runtime API

`run(ctx)` 里最常用的是这几个对象。

## `ctx.data_client`

读取运行输入数据。

常用方法：

| 方法 | 用途 |
|------|------|
| `get_dataset_keys()` | 列出可用 dataset |
| `get_item_count(dataset_key)` | 获取项目数量 |
| `get_layout(dataset_key)` | 获取布局类型 |
| `read_object(dataset_key, index)` | 读取 NDJSON 对象 |
| `iterate_objects(dataset_key)` | 迭代对象 |
| `get_parquet_dir(dataset_key, index)` | 获取 parquet item 目录 |
| `get_parquet_files(dataset_key, index)` | 获取 parquet 文件列表 |

## `ctx.output`

统一写输出结果。

常用属性和方法：

| 项目 | 说明 |
|------|------|
| `dir` | 固定输出目录 `run/output/artifacts` |
| `write_scalar()` | 写标量 |
| `write_json()` | 写 JSON |
| `write_blob()` / `write_bytes()` | 写二进制 |
| `write_file()` | 复制或登记已有文件 |
| `register_file()` | 给已有文件补登记 |
| `write_dataframe_csv()` | 写 CSV |
| `write_figure_png()` | 写 PNG |

## `ctx.params`

- 类型：`dict[str, Any]`
- 键名来自 `params.json`
- 算法运行时按普通 dict 访问

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
