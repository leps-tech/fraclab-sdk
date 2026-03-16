# Bundle 规范

## 最小目录结构

```text
my-bundle/
  manifest.json
  ds.json
  drs.json
  data/
```

## `manifest.json`

至少要包含：

- `specFiles.dsPath`
- `specFiles.drsPath`
- `specFiles.dsSha256`
- `specFiles.drsSha256`
- `dataRoot`

## `ds.json`

当前平台格式示例：

```json
{
  "schemaVersion": "1.0.0",
  "datasets": [
    {
      "key": "wells",
      "resource": "well",
      "layout": "object_ndjson_lines",
      "items": [
        { "owner": { "wellId": "W001" } }
      ]
    }
  ]
}
```

## `drs.json`

当前平台格式示例：

```json
{
  "schemaVersion": "1.0.0",
  "datasets": [
    {
      "key": "wells",
      "resource": "well",
      "cardinality": "many",
      "description": "井数据集"
    }
  ]
}
```

## 数据布局

### `object_ndjson_lines`

每个 dataset 是一份 NDJSON 行文件，按索引读取对象。

### `frame_parquet_item_dirs`

每个 dataset 下有多个 item 目录，每个 item 内是一组 parquet 文件。

时间窗场景下要注意：

- item 目录通常是 `item-00000` 这种编号
- 选择器给出的 `itemKey` 可能是稀疏集合
- 算法不能假设 item 连续

## 哈希

Bundle 导入和校验依赖 `ds.json` / `drs.json` 的 SHA256。

示例：

```python
from fraclab_sdk.materialize.hash import compute_file_sha256

ds_hash = compute_file_sha256("ds.json")
drs_hash = compute_file_sha256("drs.json")
```
