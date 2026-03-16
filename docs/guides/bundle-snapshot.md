# Bundle 与 Snapshot

## 三个对象的关系

- `Bundle`: 平台给你的原始输入目录
- `Snapshot`: SDK 导入后的本地快照副本
- `RunInput`: 某次运行真正使用的输入目录

数据流是：

`Bundle -> Snapshot -> SelectionModel -> RunInput -> Runner`

## Bundle 是什么

Bundle 至少包含这些文件：

- `manifest.json`
- `ds.json`
- `drs.json`
- `data/`

`ds.json` 和 `drs.json` 是平台原始规格，不是算法自己写的 JSON Schema。

## Snapshot 是什么

导入 Bundle 后，SDK 会把它落到本地 snapshot 库里，并做：

- 路径安全检查
- `ds.json` / `drs.json` 哈希校验
- 数据目录结构检查

导入后你不应该再手工改 Bundle 或 Snapshot 内容。

## 为什么不能手改 Bundle

Bundle 导入和运行链路依赖：

- `manifest.json` 中声明的相对路径
- `ds.json` / `drs.json` 的哈希
- `data/` 的目录布局

手改后最常见的问题是：

- 哈希不一致
- 目录结构和 `layout` 对不上
- 运行期读取失败

## `SelectionModel` 与 `run_ds`

运行时真正给算法用的是 `RunInput`，不是原始 Bundle。

`SelectionModel` 会把选中的项目重新索引，生成新的运行输入规格：

- `input/ds.json`
- `input/drs.json`
- `input/data/`

所以算法里按 `ctx.data_client` 读取到的索引，是当前 run 的索引，不一定等于原始 snapshot 索引。

对时间窗尤其要注意：

- `timeWindows_*` 里的 `itemKey` 是选择器输出
- 它可能是稀疏的，不保证是 `0..N-1`
- 不能简单假设 `range(count)` 就能和时间窗一一对应

需要查具体 Bundle 字段格式时，看[Bundle 规范](../reference/bundle-spec.md)。
