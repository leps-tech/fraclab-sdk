# Fraclab SDK

> Python `>=3.11`

Fraclab SDK 用来开发、打包、导入和执行本地算法。

`README` 现在只作为入口索引使用，不再重复维护整套说明。规范和参考以 [`docs/`](docs/) 为准。

## 文档分发

- 完整人类文档 `docs/` 会随源码分发，也会打进 PyPI 包
- 面向机器的文档 [`AI_GUIDE.md`](AI_GUIDE.md) 和 [`llms.txt`](llms.txt) 也会打进 PyPI 包
- 也就是说，`pip install fraclab-sdk` 后，本地环境里就同时有 `docs/`、`AI_GUIDE.md` 和 `llms.txt`

如果你在安装后的环境里定位这些文件，可以用：

```python
import inspect
from pathlib import Path

import fraclab_sdk

site_root = Path(inspect.getfile(fraclab_sdk)).resolve().parent.parent
print(site_root / "docs")
print(site_root / "AI_GUIDE.md")
print(site_root / "llms.txt")
```

## 安装

安装核心 SDK：

```bash
pip install fraclab-sdk
```

安装并启用 Workbench：

```bash
pip install "fraclab-sdk[workbench]"
fraclab-workbench
```

如果你当前就在仓库里开发：

```bash
poetry install
poetry run fraclab-sdk --help
```

## 先看哪几页

第一次接触这个项目，建议按这个顺序读：

1. [`docs/index.md`](docs/index.md)
2. [`docs/getting-started/installation.md`](docs/getting-started/installation.md)
3. [`docs/getting-started/first-algorithm.md`](docs/getting-started/first-algorithm.md)
4. [`docs/guides/input-spec.md`](docs/guides/input-spec.md)
5. [`docs/guides/output.md`](docs/guides/output.md)
6. [`docs/guides/compile-export-import.md`](docs/guides/compile-export-import.md)

如果你正在让 AI 生成或修改算法代码，先读：

- [`AI_GUIDE.md`](AI_GUIDE.md)
- [`llms.txt`](llms.txt)

## 常用入口

- 安装与环境: [`docs/getting-started/installation.md`](docs/getting-started/installation.md)
- 第一个算法: [`docs/getting-started/first-algorithm.md`](docs/getting-started/first-algorithm.md)
- InputSpec: [`docs/guides/input-spec.md`](docs/guides/input-spec.md)
- Output: [`docs/guides/output.md`](docs/guides/output.md)
- 编译与导入导出: [`docs/guides/compile-export-import.md`](docs/guides/compile-export-import.md)
- Runtime API: [`docs/reference/runtime-api.md`](docs/reference/runtime-api.md)
- 数据模型: [`docs/reference/models.md`](docs/reference/models.md)
- Bundle 规范: [`docs/reference/bundle-spec.md`](docs/reference/bundle-spec.md)
- 执行模型: [`docs/architecture/execution-model.md`](docs/architecture/execution-model.md)
- CLI: [`docs/cli/overview.md`](docs/cli/overview.md)

## 仓库内示例算法

当前仓库里可直接对照的完整样例主要在 `algorithms/`：

- `bh-prop-conc`: stage 级 frame 读取 + 井底支撑剂浓度估算
- `frac-derived-curves`: stage 级派生曲线分析
- `hf-fracture-curves`: 高频压力整表分析
- `hf-fracture-curves-streaming`: 高频压力流式 chunk 分析

其中流式读取写法优先参考 `hf-fracture-curves-streaming`；旧样例里直接拼 run 输入路径的读法不要再当成公开契约。

## 本地预览文档站

```bash
poetry install --with docs
poetry run mkdocs serve
```
