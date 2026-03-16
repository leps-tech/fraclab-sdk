# 第一个算法

这一页只讲最短闭环：拿到 Bundle，写算法，编译导出，导入运行。

## 开发闭环

1. 拿到平台提供的 Bundle：`manifest.json`、`ds.json`、`drs.json`、`data/`
2. 编写算法工作区：`main.py`、`schema/inputspec.py`、`schema/output_contract.py`、`manifest.json`
3. 用 Bundle 编译，生成 `dist/*.json`
4. 导出算法 zip
5. 导入 Snapshot 和算法包
6. 创建 run 并执行

最推荐的一条命令：

```bash
fraclab-sdk algo export ./my-algorithm ./my-algorithm.zip --auto-compile --bundle /path/to/bundle
```

## 最小 `main.py`

```python
def run(ctx):
    logger = ctx.logger
    dc = ctx.data_client
    out = ctx.output

    logger.info("start")

    for dataset_key in dc.get_dataset_keys():
        count = dc.get_item_count(dataset_key)
        out.write_scalar(f"{dataset_key}_count", count)

    out.write_json("summary", {"status": "completed"})
```

规则只有这些：

- 入口文件必须是 `main.py`
- 入口函数必须是 `run(ctx)`
- 参数只能有一个 `ctx`
- 返回值会被忽略

## 最小 `schema/inputspec.py`

```python
from .base import CamelModel, Field


class InputParams(CamelModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


INPUT_SPEC = InputParams
```

## 最小 `schema/output_contract.py`

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
            kind="scalar",
            owner="platform",
            cardinality="one",
            schema=ScalarOutputSchema(type="scalar", dtype="string"),
        )
    ]
)
```

## 最短运行路径

```bash
# 1. 校验 bundle
fraclab-sdk validate bundle /path/to/bundle

# 2. 导出算法包
fraclab-sdk algo export ./my-algorithm ./my-algorithm.zip --auto-compile --bundle /path/to/bundle

# 3. 导入 snapshot
fraclab-sdk snapshot import /path/to/bundle

# 4. 导入算法
fraclab-sdk algo import ./my-algorithm.zip

# 5. 创建并执行 run
fraclab-sdk run create <snapshot_id> <algorithm_name> --params params.json
fraclab-sdk run execute <run_id>
```

后续细节分别看：

- 参数定义：[InputSpec](../guides/input-spec.md)
- 输出写法：[Output](../guides/output.md)
- Bundle 结构：[Bundle 与 Snapshot](../guides/bundle-snapshot.md)
- 导出和导入规则：[编译与导入导出](../guides/compile-export-import.md)
