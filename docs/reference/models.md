# 数据模型

这一页只回答一件事：关键 JSON 和模型里有哪些字段，哪些字段是固定取值。

## DataSpec

描述运行输入数据集和实际数据布局。来源可能是：

- Bundle / Snapshot 自带的原始 `ds.json`
- SDK 为当前 run 生成的 `input/ds.json`

常见字段：

| 字段 | 说明 |
|------|------|
| `datasets[*].key` | dataset 标识 |
| `datasets[*].resource` | owner 资源类型 |
| `datasets[*].layout` | 数据布局 |
| `datasets[*].items` | dataset 下的 item 列表 |
| `datasets[*].items[*].owner` | item owner 元数据 |

`layout` 在当前源码和运行时里主要对应两种值：

- `object_ndjson_lines`
- `frame_parquet_item_dirs`

需要特别区分两类 DataSpec：

- 原始 Bundle / Snapshot `ds.json`
  不一定显式带 `layout`
- 运行时 `input/ds.json`
  SDK 会在物化 run 输入时补成可直接消费的布局信息

注意：算法运行时的 `ctx.data_client` 面向的是当前 run 的 `input/ds.json`，不是 Bundle 原目录里的原始 `ds.json`。

注意：`ds.json` 跟随平台原始格式，字段名不是 camelCase 版本的 `datasetKey` / `resourceType` 之类自定义写法。

## DRS

描述算法要求什么输入数据。来源通常是 Bundle 或算法 `dist/drs.json`。

常见字段：

| 字段 | 说明 |
|------|------|
| `datasets[*].key` | 所需 dataset key |
| `datasets[*].resource` | 所需 owner 资源类型 |
| `datasets[*].cardinality` | 需要几个 item |
| `datasets[*].description` | 补充描述 |

`cardinality` 当前允许值：

- `one`
- `many`
- `zeroOrMany`

## OutputContract

算法输出契约，来源是 `schema/output_contract.py`，最终编译成 `dist/output_contract.json`。

一个 `OutputDatasetContract` 至少要定义：

| 字段 | 说明 | 当前允许值 |
|------|------|------------|
| `key` | 输出 dataset key | 任意非空字符串 |
| `kind` | 输出类型 | `frame` / `object` / `blob` / `scalar` |
| `owner` | owner 级别 | `stage` / `well` / `platform` |
| `cardinality` | item 数量约束 | `one` / `many` |
| `schema` | 输出 schema | 跟 `kind` 对齐 |

可选但常用字段：

| 字段 | 说明 | 当前允许值 |
|------|------|------------|
| `role` | 输出角色 | `primary` / `supporting` / `debug` |
| `groupPath` | UI / 分类分组 | `list[str]` |
| `dimensions` | 维度键列表 | `list[str]` |
| `description` | 描述文本 | `str` |

### Output schema 对应关系

| `kind` | 推荐 schema |
|--------|-------------|
| `scalar` | `ScalarOutputSchema(type="scalar", ...)` |
| `frame` | `FrameOutputSchema(type="frame", ...)` |
| `object` | `ObjectOutputSchema(type="object", ...)` 或兼容的 `dict` |
| `blob` | `BlobOutputSchema(type="blob", ...)` |

`FrameOutputSchema.index` 当前允许值：

- `time`
- `depth`
- `none`

## Algorithm Manifest

算法包根目录 `manifest.json` 对应 `FracLabAlgorithmManifestV1`。

几个容易踩错的点：

- `manifestVersion` 当前是固定值 `1`
- `contractVersion` 必须是 semver-like 字符串，例如 `1.2.3`
- `authors` 至少要有一个作者
- `files` 里的路径应该指向 `dist/` 下已生成的文件

## RunOutputManifest

每次运行完成后，SDK 会自动生成 `output/manifest.json`。

它描述：

- 产出了哪些 dataset
- 每个 dataset 下有哪些 item
- 每个 item 对应什么 artifact
- artifact 是标量还是文件

这个 manifest 是系统生成物，不是算法作者手写物。算法代码应该通过 `ctx.output.write_*()` 或 `ctx.output.register_file()` 生成它的内容。
