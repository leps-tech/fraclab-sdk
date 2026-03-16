# 数据模型

## DataSpec

描述输入数据集和实际数据布局。来源是 Bundle / Snapshot 自带的 `ds.json`。

当前平台格式使用：

- `datasets[*].key`
- `datasets[*].resource`
- `datasets[*].layout`
- `datasets[*].items`

## DRS

描述算法要求什么输入数据。来源是 Bundle / 算法 `dist/drs.json`。

常见字段：

- `datasets[*].key`
- `datasets[*].resource`
- `datasets[*].cardinality`
- `datasets[*].description`

## OutputContract

算法输出契约，来源是 `schema/output_contract.py`。

一个 `OutputDatasetContract` 至少要定义：

- `key`
- `kind`
- `owner`
- `cardinality`
- `schema`

## RunOutputManifest

每次运行完成后，SDK 会自动生成 `output/manifest.json`。

它描述：

- 产出了哪些 dataset
- 每个 dataset 下有哪些 item
- 每个 item 对应什么 artifact
- artifact 是标量还是文件

这个 manifest 是系统生成物，不是算法作者手写物。
