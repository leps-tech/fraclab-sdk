# 执行模型

## 运行链路

Fraclab SDK 当前的执行链路是：

1. 导入 Bundle，生成 Snapshot
2. 用 `SelectionModel` 选择输入项目
3. 物化当前 run 的输入目录
4. 通过 runner 子进程加载 `main.py`
5. 算法通过 `ctx.data_client` 读输入，通过 `ctx.output` 写输出
6. runner 结束后自动生成 `output/manifest.json`

## 输入侧

算法不直接读 Bundle 原目录，而是读当前 run 的输入目录：

- `input/ds.json`
- `input/drs.json`
- `input/data/`

这样每次运行的输入都是冻结的、可复现的。

这里有一个容易忽略的边界：

- 原始 Bundle / Snapshot `ds.json` 可以保持平台原始格式
- SDK 在构建当前 run 输入时，会把运行时真正需要的布局信息补进 `input/ds.json`

当前布局推断来源按这个顺序处理：

1. `ds.json` 自带的 `layout`
2. Bundle / Snapshot `manifest.json` 中的数据集布局
3. 实际数据目录结构，例如 `data/<dataset>/parquet/` 或 `data/<dataset>/object.ndjson`

所以运行时 `ctx.data_client` 消费的是“当前 run 的规范化输入”，不是未经处理的原始 Bundle `ds.json`。

## 输出侧

算法不自己维护 manifest。

Runner 会：

- 收集显式输出记录
- 自动发现 `ctx.output.dir` 下未登记文件
- 最终写出统一的 `output/manifest.json`

## 当前执行边界

当前 runner 负责：

- 加载算法模块
- 提供标准 `RunContext`
- 管理输入输出目录
- 生成输出 manifest

当前还没有把“第三方 import 白名单限制”做成强制执行的受限 runner。  
如果后面要收紧算法依赖边界，这一层会是最合适的落点。
