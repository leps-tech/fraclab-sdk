# Fraclab SDK

Fraclab SDK 用来开发、打包、导入和执行本地算法。这套文档按“先完成开发闭环，再查细节约束”的方式组织，减少第一次上手时在 README、源码和参考页之间来回跳。

仓库根目录的 `README.md` 现在只保留入口索引；这里的 `docs/` 才是完整人类文档。`docs/`、`AI_GUIDE.md` 和 `llms.txt` 都会随源码分发，也会打进 PyPI 包。

## 先看哪几页

如果你是第一次接触这个 SDK，按这个顺序读：

1. [安装与环境](getting-started/installation.md)
2. [第一个算法](getting-started/first-algorithm.md)
3. [InputSpec](guides/input-spec.md)
4. [Output](guides/output.md)
5. [编译与导入导出](guides/compile-export-import.md)

## 按任务找文档

### 我要把一个算法跑通

从 [第一个算法](getting-started/first-algorithm.md) 开始。它只覆盖最短开发闭环：拿到 Bundle、写 `main.py`、定义参数和输出、编译导出、导入执行。

### 我要定义参数

看 [InputSpec](guides/input-spec.md)。这里说明 `schema/inputspec.py` 怎么生成 `params.schema.json`，以及 camelCase、时间窗、`bindDatasetKey` 的约束。

### 我要写输出

看 [Output](guides/output.md) 和 [Runtime API](reference/runtime-api.md)。运行时代码只应该通过 `ctx.output` 写结果；`output/manifest.json` 由 SDK 自动生成。

### 我要确认字段取值和数据结构

看 [数据模型](reference/models.md)、[Bundle 规范](reference/bundle-spec.md) 和 [Runtime API](reference/runtime-api.md)。这些页面专门回答“字段名是什么”“允许哪些值”“运行时提供什么对象”。

### 我要理解编译、导出、导入流程

看 [编译与导入导出](guides/compile-export-import.md) 和 [执行模型](architecture/execution-model.md)。

### 我要直接对照仓库里的完整样例

先看 `algorithms/` 目录里这 4 个当前样例：

- `algorithms/bh-prop-conc/0.1.0/`: stage 级数据整表读取 + 井筒体积平移估算
- `algorithms/frac-derived-curves/0.1.0/`: stage 级派生曲线与离线分析
- `algorithms/hf-fracture-curves/0.1.0/`: 高频压力数据整表读取后分析
- `algorithms/hf-fracture-curves-streaming/0.1.0/`: 基于 `iter_dataset_frame_chunks()` 的流式处理样例

如果示例代码和文档有冲突，以 `src/fraclab_sdk/` 里的运行时与模型实现为准，再回头修样例。

## 文档边界

- 算法运行时入口是 `main.py` 中的 `run(ctx)`
- 输入读取走 `ctx.data_client`
- 输出写入走 `ctx.output`
- Bundle / Snapshot 自带的 `ds.json` / `drs.json` 保持平台原始字段名，例如 `key` / `resource`
- 算法自己定义的 `params.schema.json`、`output_contract.json`、运行结果 `output/manifest.json` 使用 camelCase

## 给 AI 的入口

如果你正在让 AI 生成或修改算法代码，先读仓库根目录的 `AI_GUIDE.md`。那份文档把：

- 真实运行时上下文
- 允许的枚举值和 `Literal` 约束
- 最容易写错的字段名和输出规则
- 生成代码前必须检查的事项

集中写在了一页里。

## 本地预览文档站

```bash
poetry install --with docs
poetry run mkdocs serve
```

静态构建：

```bash
poetry run mkdocs build
```
