# Fraclab SDK

Fraclab SDK 用来开发、打包、导入和执行本地算法。当前文档站按“上手路径”和“参考手册”拆开，避免再把所有内容堆进一个 README。

## 推荐阅读路径

1. 先看[安装与环境](getting-started/installation.md)
2. 再走一遍[第一个算法](getting-started/first-algorithm.md)
3. 开始写参数和输出时，分别看：
   - [InputSpec](guides/input-spec.md)
   - [Output](guides/output.md)
4. 需要打包、导入、执行时，看[编译与导入导出](guides/compile-export-import.md)
5. 查字段和结构时，直接进[参考](reference/runtime-api.md)

## 文档结构

- `快速开始`: 给第一次接触 Fraclab SDK 的用户
- `算法指南`: 写算法时最常用的规则和示例
- `CLI`: 命令行入口和常见流程
- `Workbench`: 图形界面入口和页面说明
- `参考`: 结构定义、字段表、Bundle 规范、错误处理
- `架构`: 执行链路和内部模型，主要给维护者

## 本地预览文档站

```bash
poetry install --with docs
poetry run mkdocs serve
```

这里是直接基于当前仓库源码本地起站，不依赖 GitHub Pages 或 PyPI。

构建静态站点：

```bash
poetry run mkdocs build
```

## 当前边界

- 算法允许依赖的第三方包是固定白名单，见[安装与环境](getting-started/installation.md)
- 算法输出统一走 `ctx.output`，结果 manifest 由 SDK 自动生成，见[Output](guides/output.md)
- Bundle 自带的 `ds.json` / `drs.json` 使用平台原始格式 `key` / `resource`，不是 camelCase，见[Bundle 规范](reference/bundle-spec.md)
