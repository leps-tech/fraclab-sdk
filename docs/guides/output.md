# Output

算法输出统一走 `ctx.output`。

## 核心规则

- 只使用运行时注入的 `ctx.output`
- 不要自己实例化 `OutputClient`
- 不要手写或修改 `output/manifest.json`
- 服务器侧只保证 `ctx.output.dir` 这个固定目录可写
- 不要在算法里创建子目录

## 最常用写法

```python
def run(ctx):
    out = ctx.output

    out.write_scalar("score", 0.95)
    out.write_json("summary", {"status": "ok"})

    plot_path = out.dir / "result_plot.png"
    fig.savefig(plot_path, dpi=200)
```

上面这个 `result_plot.png` 即使没有显式登记，也会在 runner 结束时自动被收进 `output/manifest.json`。

## 显式写入

适合需要自定义 `dataset_key`、`owner`、`dims`、`meta` 的结果。

```python
out.write_scalar(
    "accuracy",
    0.87,
    dataset_key="metrics",
    owner={"wellId": "W001"},
    dims={"stage": 1},
    meta={"unit": "percent"},
)

out.write_json(
    "summary",
    {"status": "ok"},
    dataset_key="summary",
    owner={"platformId": "platform"},
)
```

## 已经写好的文件如何登记

如果某个库只能接受文件路径，就先写到 `ctx.output.dir`，再登记：

```python
report_path = ctx.output.dir / "report.csv"
df.to_csv(report_path, index=False)

ctx.output.register_file(
    "report",
    report_path,
    mime_type="text/csv",
    dataset_key="reports",
)
```

## 常用方法

| 方法 | 用途 |
|------|------|
| `write_scalar()` | 写标量结果 |
| `write_json()` | 写 JSON 结果 |
| `write_blob()` / `write_bytes()` | 写二进制文件 |
| `write_file()` | 复制或登记已有文件 |
| `register_file()` | 给已落到 `ctx.output.dir` 的文件补元数据登记 |
| `write_dataframe_csv()` | 直接写 DataFrame 为 CSV |
| `write_figure_png()` | 直接写 matplotlib figure 为 PNG |

## manifest 是怎么来的

`output/manifest.json` 完全由 SDK 自动生成：

1. 先收集显式 `write_*()` / `register_file()` 的记录
2. 再扫描 `ctx.output.dir` 下未显式登记的文件
3. 合并成最终的 `output/manifest.json`

算法作者只需要产出结果，不需要维护 manifest。

## 与 OutputContract 的关系

`OutputContract` 负责声明“应该产出什么”，`ctx.output` 负责“把结果写出来并登记”。

如果你声明了：

- `owner="well"`
- `dimensions=["stage"]`
- `cardinality="one"`

那实际写结果时就要按这个约束提供元数据。最终一致性由运行后校验负责。
