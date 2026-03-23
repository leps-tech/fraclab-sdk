# 第一个算法

这一页只覆盖最短开发闭环：拿到 Bundle，写一个最小算法，编译导出，导入运行。先把流程跑通，再去看每个细节页。

## 你最终会产出什么

一个最小算法工作区通常长这样：

```text
my-algorithm/
  main.py
  manifest.json
  schema/
    base.py
    inputspec.py
    output_contract.py
```

编译后还会生成：

```text
my-algorithm/
  dist/
    params.schema.json
    output_contract.json
    ds.json
    drs.json
```

## 开发闭环

1. 拿到平台提供的 Bundle：`manifest.json`、`ds.json`、`drs.json`、`data/`
2. 编写算法工作区：`main.py`、`schema/inputspec.py`、`schema/output_contract.py`、`manifest.json`
3. 用 Bundle 编译，生成 `dist/*.json`
4. 导出算法 zip
5. 导入 Snapshot 和算法 zip
6. 创建 run 并执行

最推荐的一条命令：

```bash
fraclab-sdk algo export ./my-algorithm ./my-algorithm.zip --auto-compile --bundle /path/to/bundle
```

这条命令的意义是：缺什么就先补什么。缺 `dist/*.json` 时会先编译，编译缺 DS/DRS 时再从 `--bundle` 复制。

## 第一步：写最小 `main.py`

```python
def run(ctx) -> None:
    dc = ctx.data_client
    out = ctx.output

    summary = {}
    for dataset_key in dc.get_dataset_keys():
        summary[dataset_key] = {
            "layout": dc.get_layout(dataset_key),
            "count": dc.get_item_count(dataset_key),
        }

    out.write_json("summary", summary, dataset_key="summary")
    out.write_scalar("status", "ok", dataset_key="status")
```

这里先记住 4 条规则：

- 入口文件必须是 `main.py`
- 入口函数必须是 `run(ctx)`
- 运行时读取输入走 `ctx.data_client`
- 运行时写输出走 `ctx.output`

不要在示例里发明 `ctx.artifacts` 之类的别名。真实运行时暴露的是 `output`。

## 第二步：写最小 `schema/inputspec.py`

```python
from .base import CamelModel, Field


class InputParams(CamelModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


INPUT_SPEC = InputParams
```

这里的重点不是字段多，而是约束清楚：

- 必须导出 `INPUT_SPEC`
- 推荐继承 `CamelModel`
- 参数名要按 camelCase 设计，这样生成的 `params.schema.json` 和运行时 `ctx.params` 才一致

## 第三步：写最小 `schema/output_contract.py`

```python
from fraclab_sdk.models.output_contract import (
    OutputContract,
    OutputDatasetContract,
    ScalarOutputSchema,
)


OUTPUT_CONTRACT = OutputContract(
    datasets=[
        OutputDatasetContract(
            key="summary",
            kind="object",
            owner="platform",
            cardinality="one",
            schema={"type": "object"},
        ),
        OutputDatasetContract(
            key="status",
            kind="scalar",
            owner="platform",
            cardinality="one",
            schema=ScalarOutputSchema(type="scalar", dtype="string"),
        ),
    ]
)
```

这里最容易写错的是受限取值。当前源码里这些值不是自由字符串，而是固定 `Literal`：

- `kind`: `frame` / `object` / `blob` / `scalar`
- `owner`: `stage` / `well` / `platform`
- `cardinality`: `one` / `many`

如果要查完整约束，直接看[数据模型](../reference/models.md)或仓库根目录 `AI_GUIDE.md`。

## 第四步：编译并导出

```bash
fraclab-sdk algo export ./my-algorithm ./my-algorithm.zip --auto-compile --bundle /path/to/bundle
```

如果你想把步骤拆开：

```bash
fraclab-sdk algo compile ./my-algorithm --bundle /path/to/bundle
fraclab-sdk algo export ./my-algorithm ./my-algorithm.zip
```

## 第五步：导入并运行

```bash
# 1. 校验 bundle
fraclab-sdk validate bundle /path/to/bundle

# 2. 导入 snapshot
fraclab-sdk snapshot import /path/to/bundle

# 3. 导入算法
fraclab-sdk algo import ./my-algorithm.zip

# 4. 创建并执行 run
fraclab-sdk run create <snapshot_id> <algorithm_name> --params params.json
fraclab-sdk run execute <run_id>
```

## 跑通后再看这些页

- 想把参数写严谨：[InputSpec](../guides/input-spec.md)
- 想把输出写完整：[Output](../guides/output.md)
- 想理解 Bundle / Snapshot：[Bundle 与 Snapshot](../guides/bundle-snapshot.md)
- 想查编译、导入、导出的边界：[编译与导入导出](../guides/compile-export-import.md)
- 想查运行时对象和模型字段：[参考](../reference/runtime-api.md)

## 最常见的 5 个误区

1. 把 Bundle 里的 `ds.json` / `drs.json` 当成 camelCase JSON 去写。
2. 在运行时代码里使用不存在的 `ctx.artifacts`。
3. `OutputContract` 里把 `kind`、`owner`、`cardinality` 写成源码里不存在的字符串。
4. `INPUT_SPEC` 定义的是 schema，但运行时代码却假设 `ctx.params` 是 Pydantic 模型对象。
5. 手工维护 `output/manifest.json`，而不是通过 `ctx.output.write_*()` 产出结果。
