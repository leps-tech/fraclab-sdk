# 安装与环境

## PyPI 发布版安装

```bash
pip install fraclab-sdk
```

安装 Workbench：

```bash
pip install "fraclab-sdk[workbench]"
fraclab-workbench
```

也可以直接调模块：

```bash
python -m fraclab_sdk.workbench
```

## 仓库源码本地安装

如果你已经在这个仓库里开发或本地试用，不需要先传 GitHub 或 PyPI。

安装核心环境：

```bash
poetry install
poetry run fraclab-sdk --help
```

安装并启用 Workbench + 文档：

```bash
poetry install --extras workbench --with docs
poetry run fraclab-workbench
poetry run mkdocs serve
```

这里的文档站是从仓库里的 `docs/` 和 `mkdocs.yml` 本地构建出来的，不依赖 GitHub Pages 或 PyPI。

## CLI 入口

安装后提供三个入口：

| 命令 | 说明 |
|------|------|
| `fraclab-sdk` | 主 CLI，负责算法、快照、运行、校验 |
| `fraclab-runner` | 算法子进程 runner，通常由系统内部调用 |
| `fraclab-workbench` | 启动图形界面 |

## 算法依赖白名单

算法代码允许使用的第三方包只有这些：

- `fraclab-sdk`
- `numpy>=2.1.1`
- `pandas>=2.2.3`
- `scipy>=1.13.1`
- `matplotlib>=3.9.2`
- `fastapi>=0.115.0`

另外，Python 标准库可以直接使用。

不要把下面这些包当成算法依赖契约：

- `pydantic`
- `plotly`
- `streamlit`
- `streamlit-plotly-events`
- `pyarrow`
- `typer`
- `rich`

这些包可能会随着 SDK 或 Workbench 被安装，但不等于算法可以依赖它们。

## 算法运行时中文字体

Runner 会在加载算法前自动给 matplotlib 注入默认中文字体：

- `WenQuanYi Micro Hei`
- `Noto Sans CJK JP`
- `Noto Serif CJK JP`

对应系统包：

- `fonts-wqy-microhei`
- `fonts-noto-cjk`

当前不支持 `SimHei`。

如果你脱离 runner 单独调试脚本，需要手动调用：

```python
from fraclab_sdk.runtime import configure_matplotlib_runtime_fonts

configure_matplotlib_runtime_fonts()
```
